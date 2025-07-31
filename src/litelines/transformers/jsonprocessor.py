import re
from typing import Optional, Type, Union

import torch
from pydantic import BaseModel
from transformers import LogitsProcessor, PreTrainedTokenizer

from ..build_dfa import build_dfa
from ..draw_dfa import draw_dfa

class JSONProcessor(LogitsProcessor):
    def __init__(
        self,
        response_format: Union[str, dict[int, dict[int, int]], Type[BaseModel]],
        tokenizer: PreTrainedTokenizer,
        include_tool_call: bool = False,
        tool_call_start: str = "<tool_call>",
        tool_call_end: str = "</tool_call>",
        allow_preamble: bool = False,
        whitespace_pattern: str = r"[\n\t\r ]*",
        verbose: bool = False,
        max_same_state_visit_count: int = 5,
    ) -> None:
        self.response_format = response_format
        self.tokenizer = tokenizer
        self.include_tool_call = include_tool_call
        self.tool_call_start = tool_call_start
        self.tool_call_end = tool_call_end
        self.allow_preamble = allow_preamble
        self.whitespace_pattern = whitespace_pattern
        self.dfa = None
        self.verbose = verbose
        self.max_same_state_visit_count = max_same_state_visit_count
        self.same_state_visit_count = 0
        self.current_state = None
        self.previous_state = None
        self.final_states = None
        self.selected_token = None
        self.trajectory = []
        self.previous_input_ids = None
        self.trigger_token_ids = []
        self.triggered = None

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
        elif issubclass(self.response_format, BaseModel):
            self.__build_dfa()
        else:
            raise ValueError(
                f"Cannot parse schema {self.response_format}. The schema must be either "
                + "a Pydantic model, a dict[int, dict[int, int]] or a string that contains the JSON "
                + "schema specification"
            )

    def show_graph(self):
        if self.trajectory == []:  # first time
            self.__create_dfa()
        return draw_dfa(
            self.response_format,
            self.tokenizer,
            self.trajectory,
            self.include_tool_call,
            self.tool_call_start,
            self.tool_call_end,
            self.whitespace_pattern,
        )

    def reset_state(self):
        """Reset the processor to its initial state"""
        self.current_state = 0
        self.final_states = None
        self.selected_token = None

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        scores_processed = scores.clone()
        token_chosen_id = torch.argmax(scores_processed).item()

        if self.previous_input_ids is not None:
            # Check if we're continuing from the previous sequence
            if not torch.equal(input_ids[:, :-1], self.previous_input_ids):
                # If the history doesn't match, reset the state
                self.reset_state()

        if self.final_states is None:  # first time
            if self.dfa is None:
                self.__create_dfa()
            states = range(len(self.dfa) + 1)
            self.final_states = {
                state for state in states if state not in list((self.dfa).keys())
            }
            if self.verbose:
                print(f"states: {states}")
                print(f"final states: {self.final_states}")
            self.previous_input_ids = input_ids.clone()
            if not self.allow_preamble:
                self.current_state = 0  # dfa active
            else:
                self.current_state = -1  # inactive
                # add eos to triggers
                self.trigger_token_ids = [
                    self.tokenizer.eos_token_id,
                    self.tokenizer.pad_token_id,
                ]
                if self.include_tool_call:  # it should be a tool call
                    if (
                        len(
                            self.tokenizer.encode(
                                self.tool_call_start, add_special_tokens=False
                            )
                        )
                        > 1
                    ):
                        raise ValueError(
                            f"{self.tool_call_start} is not a valid token."
                        )
                    self.trigger_token_ids.append(
                        self.tokenizer.encode(
                            self.tool_call_start, add_special_tokens=False
                        )[0]
                    )
                    # not the best solution since it excludes '<' in the preamble
                    tokens_containing_open_tool_call = [
                        token_id
                        for token_id in range(tokenizer.vocab_size)
                        if "<" in tokenizer.decode(token_id)
                    ]
                    self.trigger_token_ids += tokens_containing_open_tool_call
                else:  # it should be json
                    tokens_containing_open_curly_bracket = [
                        token_id
                        for token_id in range(self.tokenizer.vocab_size)
                        if "{" in self.tokenizer.decode(token_id)
                    ]
                    self.trigger_token_ids += tokens_containing_open_curly_bracket

        else:  # not the first time
            self.selected_token = input_ids[:, -1].item()
            if self.current_state != -1:
                self.trajectory.append(self.selected_token)
            if self.verbose:
                print(
                    f"\x1b[32mselected token: {self.selected_token}: {repr(self.tokenizer.decode([self.selected_token]))}\x1b[0m"
                )
            if self.verbose and self.current_state != -1:
                print(f"mapping: {self.dfa[self.current_state]}")

        # activate it if it is triggered
        if (
            self.current_state == -1 and token_chosen_id in self.trigger_token_ids
        ):  # if dfa is inactive
            if self.verbose:
                print(
                    f"\x1b[31mtrigger token: {token_chosen_id}: {self.tokenizer.decode([token_chosen_id])}\x1b[0m"
                )
            self.triggered = True
            self.current_state = 0

        if self.current_state != -1:  # if dfa is active
            if self.triggered:
                self.current_state = 0
                self.triggered = False
            else:
                self.previous_state = self.current_state
                self.current_state = self.dfa[self.current_state][
                    self.selected_token
                ]
                if (
                    self.previous_state == self.current_state
                    and re.fullmatch(
                        self.whitespace_pattern,
                        self.tokenizer.decode([self.selected_token]),
                    )
                    is not None
                ):
                    self.same_state_visit_count += 1
                else:
                    self.same_state_visit_count = 0

        self.previous_input_ids = input_ids.clone()

        vocab_tensor = torch.arange(scores.shape[-1], device=scores.device)

        if self.verbose:
            print(f"\x1b[34mcurrent state: {self.current_state}\x1b[0m")
            print(
                f"\x1b[33msame state visit count: {self.same_state_visit_count}\x1b[0m"
            )

        if self.current_state == -1:
            # print(self.trigger_token_ids)
            forbidden_tokens = torch.tensor(
                self.trigger_token_ids, device=scores.device
            )
            forbidden_tokens_mask = torch.isin(vocab_tensor, forbidden_tokens)
        else:  # if dfa is active
            if self.current_state in self.final_states:
                allowed_tokens = [self.tokenizer.eos_token_id]
            else:
                if self.same_state_visit_count < self.max_same_state_visit_count:
                    allowed_tokens = list(self.dfa[self.current_state].keys())
                else:
                    # Remove tokens that send you to the same current state
                    if self.verbose:
                        print(
                            f"\x1b[31mmaximum same state visit count reached for state {self.current_state}\x1b[0m"
                        )
                    mapping = self.dfa[self.current_state]
                    allowed_tokens = [
                        key
                        for key, value in mapping.items()
                        if value != self.current_state
                    ]
            allowed_tokens = torch.tensor(allowed_tokens, device=scores.device)
            forbidden_tokens_mask = ~torch.isin(vocab_tensor, allowed_tokens)

        scores_processed = torch.where(forbidden_tokens_mask, -torch.inf, scores)
        if self.verbose:
            print(f"\x1b[35mwill be chosen: {torch.argmax(scores_processed).item()}\x1b[0m")

        return scores_processed

