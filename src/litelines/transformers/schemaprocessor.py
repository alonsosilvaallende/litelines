import re
from typing import Any, List, Type, Union

try:
    import torch  # type: ignore
    from transformers import LogitsProcessor, PreTrainedTokenizer  # type: ignore
except ImportError:
    msg = (
        "For this processor, transformers and pytorch should be installed. "
        "You can install them with pip install transformers[torch]"
    )
    raise ImportError(msg) from None

from ..build_dfa import build_dfa
from ..draw_dfa import draw_dfa


class SchemaProcessor(LogitsProcessor):
    """Build the Logits Processor that enforces the response format

    Examples:

    Args:
        response_format: A Pydantic model, a dictionary, or a regular expression (as a string) that defines the expected response format
        tokenizer: The model's tokenizer or the model name (as a string)
        include_tool_call (optional): Is the Pydantic model expecting a tool call or not.
        tool_call_start (optional): The expected tool call start.
        tool_call_end (optional): The expected tool call end.
        whitespace_pattern (optional): Pattern to use for JSON syntactic whitespace.
        max_same_state (optional): The maximum number of consecutive whitespace tokens allowed in the same DFA state.
        allow_preamble (optional): Whether to allow preamble text before the enforced format.
        max_preamble_tokens (optional): The maximum number of tokens allowed in the preamble.
        trigger_token_ids (optional): Token IDs that trigger the start of the enforced format.

    Returns:
        The logits processor that enforces the response format
    """

    def __init__(
        self,
        response_format: Union[str, dict[int, dict[int, int]], Type[Any]],
        tokenizer: PreTrainedTokenizer,
        include_tool_call: bool = False,
        tool_call_start: str = "<tool_call>",
        tool_call_end: str = "</tool_call>",
        whitespace_pattern: str = r"[\n\t\r ]*",
        max_same_state: int = 5,
        allow_preamble: bool = False,
        max_preamble_tokens: int = 50_000,
        trigger_token_ids: Union[int, List[int]] = None,
    ) -> None:
        self.response_format = response_format
        self.tokenizer = tokenizer
        self.include_tool_call = include_tool_call
        self.tool_call_start = tool_call_start
        self.tool_call_end = tool_call_end
        self.whitespace_pattern = whitespace_pattern
        self.max_same_state = max_same_state
        self.allow_preamble = allow_preamble
        self.max_preamble_tokens = max_preamble_tokens
        if isinstance(trigger_token_ids, int):
            self.trigger_token_ids = [trigger_token_ids]
        else:
            self.trigger_token_ids = trigger_token_ids or []
        self.trajectory = []
        self.previous_input_ids = None
        self.current_state = None

    def __build_dfa(self):
        self.dfa = build_dfa(
            self.response_format,
            self.tokenizer,
            include_tool_call=self.include_tool_call,
            tool_call_start=self.tool_call_start,
            tool_call_end=self.tool_call_end,
            whitespace_pattern=self.whitespace_pattern,
        )

    def __create_dfa(self):
        if isinstance(self.response_format, dict) and all(
            isinstance(k, int)
            and isinstance(v, dict)
            and all(isinstance(k2, int) and isinstance(v2, int) for k2, v2 in v.items())
            for k, v in (self.response_format).items()
        ):
            self.dfa = self.response_format
        elif isinstance(self.response_format, str):
            self.__build_dfa()
        elif hasattr(self.response_format, "model_json_schema"):
            self.__build_dfa()
        else:
            raise ValueError(
                f"Cannot parse schema {self.response_format}. The schema must be either "
                + "a Pydantic model, a dict[int, dict[int, int]] or a string that contains the JSON "
                + "schema specification"
            )

    def __precompute_dfa_tensors(self):
        """Convert DFA dict to tensors for efficient lookup"""
        max_state = max(self.dfa.keys()) + 1
        self.transition_table = torch.full(
            (max_state + 1, self.vocab_size), -1, dtype=torch.long, device=self.device
        )
        self.allowed_mask = torch.zeros(
            (max_state + 1, self.vocab_size), dtype=torch.bool, device=self.device
        )
        self.allowed_mask_if_max_same_state = torch.zeros(
            (max_state + 1, self.vocab_size), dtype=torch.bool, device=self.device
        )
        for state, transitions in self.dfa.items():
            for token, next_state in transitions.items():
                self.transition_table[state, token] = next_state
                self.allowed_mask[state, token] = True
                if state != next_state:
                    self.allowed_mask_if_max_same_state[state, token] = True

    def __precompute_eos_only_mask(self):
        self.eos_only_mask = torch.zeros(
            (self.batch_size, self.vocab_size), dtype=torch.bool, device=self.device
        )
        self.eos_only_mask[:, self.tokenizer.eos_token_id] = True

    def show_graph(self, batch_number=0, trajectory=[]):
        if not trajectory:
            if not self.trajectory:  # first time
                self.__create_dfa()
                trajectory_data = self.trajectory
            else:
                trajectory_data = self.trajectory[batch_number]
        else:
            trajectory_data = trajectory

        return draw_dfa(
            self.response_format,
            self.tokenizer,
            trajectory_data,
            self.include_tool_call,
            self.tool_call_start,
            self.tool_call_end,
            self.whitespace_pattern,
        )

    def __reset_state(self):
        """Reset the processor to its initial state"""
        self.current_state = None

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        scores_processed = scores.clone()
        would_be_chosen_token_id = torch.argmax(scores_processed, dim=1)
        if self.previous_input_ids is not None:
            # Check if we're continuing from the previous sequence
            if not torch.equal(input_ids[:, :-1], self.previous_input_ids):
                # If the history doesn't match, reset the state
                self.__reset_state()
        self.previous_input_ids = input_ids.clone()
        first_time = self.current_state is None
        if first_time:
            self.device = scores.device
            self.neg_inf = torch.tensor(-torch.inf, device=self.device)
            self.vocab_size = scores.shape[-1]
            self.batch_size = input_ids.shape[0]
            # if not self.trajectory:  # first time
            self.trajectory = [
                [] for _ in range(self.batch_size)
            ]  # initialize trajectory
            self.same_state_counter = torch.zeros(
                self.batch_size, dtype=torch.int32, device=self.device
            )
            self.preamble_tokens_counter = torch.zeros(
                self.batch_size, dtype=torch.long, device=self.device
            )
            self.__create_dfa()
            self.inactive_state = torch.tensor(
                [-1], dtype=torch.long, device=self.device
            )
            states = range(len(self.dfa) + 1)
            self.final_states = torch.tensor(
                [state for state in states if state not in list((self.dfa).keys())],
                dtype=torch.long,
                device=self.device,
            )

            if not self.allow_preamble:
                self.previous_state = -torch.ones(
                    self.batch_size, dtype=torch.long, device=self.device
                )
                self.current_state = torch.zeros(
                    self.batch_size, dtype=torch.long, device=self.device
                )
            else:
                self.previous_state = -2 * torch.ones(
                    self.batch_size, dtype=torch.long, device=self.device
                )
                self.current_state = -torch.ones(
                    self.batch_size, dtype=torch.long, device=self.device
                )

                if (
                    not self.trigger_token_ids
                ):  # if no trigger token list has been given
                    if self.include_tool_call:  # it should be a tool call
                        # not the best solution since it excludes '<' in the preamble
                        tokens_containing_open_tool_call = [
                            token_id
                            for token_id in range(self.tokenizer.vocab_size)
                            if "<" in self.tokenizer.decode(token_id)
                        ]
                        self.trigger_token_ids += tokens_containing_open_tool_call
                    else:  # it should be json
                        tokens_containing_open_curly_bracket = [
                            token_id
                            for token_id in range(self.tokenizer.vocab_size)
                            if "{" in self.tokenizer.decode(token_id)
                        ]
                        self.trigger_token_ids += tokens_containing_open_curly_bracket
                    # add eos to triggers
                    self.trigger_token_ids += [
                        self.tokenizer.eos_token_id,
                        self.tokenizer.pad_token_id,
                    ]
            if isinstance(self.trigger_token_ids, list):
                self.trigger_token_ids = torch.tensor(
                    self.trigger_token_ids, dtype=torch.long, device=self.device
                )
            self.__precompute_dfa_tensors()
            self.__precompute_eos_only_mask()
        else:
            self.previous_state = self.current_state
            selected_tokens = input_ids[:, -1]
            # Handle trajectory
            active_state_mask = ~torch.isin(self.current_state, self.inactive_state)
            non_final_state_mask = ~torch.isin(self.current_state, self.final_states)
            for i in range(self.batch_size):
                if active_state_mask[i] and non_final_state_mask[i]:
                    self.trajectory[i] += [selected_tokens[i].item()]
            valid_transitions = self.transition_table[
                self.current_state, selected_tokens
            ]
            non_final_state_mask = ~torch.isin(self.current_state, self.final_states)
            self.current_state = torch.where(
                non_final_state_mask, valid_transitions, self.current_state
            )

            # Handle same state counter
            same_state_mask = self.previous_state == self.current_state
            if same_state_mask.any():
                # Check whitespace pattern for tokens that stayed in same state
                same_state_indices = torch.where(same_state_mask)[0]
                for idx in same_state_indices:
                    token_text = self.tokenizer.decode([selected_tokens[idx]])
                    if re.fullmatch(self.whitespace_pattern, token_text) is not None:
                        self.same_state_counter[idx] += 1
                    else:
                        self.same_state_counter[idx] = 0

            # Reset count for states that changed
            state_changed_mask = self.previous_state != self.current_state
            self.same_state_counter[state_changed_mask] = 0
            # Handle preamble counter
            inactive_state_mask = torch.isin(self.current_state, self.inactive_state)
            self.preamble_tokens_counter[inactive_state_mask] += 1

            triggered = torch.isin(selected_tokens, self.trigger_token_ids)
            self.current_state = torch.where(
                inactive_state_mask & triggered,
                torch.zeros_like(self.current_state),
                self.current_state,
            )
            final_token_idx = torch.tensor(
                [self.tokenizer.eos_token_id, self.tokenizer.pad_token_id],
                device=self.device,
            )
            final_token_mask = torch.isin(would_be_chosen_token_id, final_token_idx)
            self.current_state = torch.where(
                inactive_state_mask & final_token_mask,
                torch.zeros_like(self.current_state),
                self.current_state,
            )
            max_preamble_tokens_mask = (
                self.preamble_tokens_counter > self.max_preamble_tokens
            )
            self.current_state = torch.where(
                inactive_state_mask & max_preamble_tokens_mask,
                torch.zeros_like(self.current_state),
                self.current_state,
            )

        batch_allowed_mask = torch.where(
            (self.current_state >= 0).unsqueeze(-1),
            self.allowed_mask[self.current_state],  # if DFA is active
            torch.ones_like(self.allowed_mask[0]),  # if DFA is inactive
        )

        max_same_state_mask = self.same_state_counter > self.max_same_state
        if max_same_state_mask.any():
            batch_allowed_mask_if_max_same_state = self.allowed_mask_if_max_same_state[
                self.current_state
            ]
            batch_allowed_mask = torch.where(
                max_same_state_mask.unsqueeze(1),
                batch_allowed_mask_if_max_same_state,
                batch_allowed_mask,
            )

        final_state_mask = torch.isin(self.current_state, self.final_states)
        if final_state_mask.any():
            batch_allowed_mask = torch.where(
                final_state_mask.unsqueeze(1), self.eos_only_mask, batch_allowed_mask
            )

        not_final_state_mask = ~torch.isin(self.current_state, self.final_states)
        batch_allowed_mask[not_final_state_mask, self.tokenizer.eos_token_id] = False
        batch_allowed_mask[not_final_state_mask, self.tokenizer.pad_token_id] = False

        scores_processed = torch.where(batch_allowed_mask, scores, self.neg_inf)

        return scores_processed