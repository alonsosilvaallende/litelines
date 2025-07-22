from typing import Optional, Type, Union

import torch
from pydantic import BaseModel
from transformers import LogitsProcessor, PreTrainedTokenizer
from ..build_dfa import build_dfa


class JSONProcessor(LogitsProcessor):
    def __init__(
        self,
        response_format=Union[str, dict[int, dict[int, int]], Type[BaseModel]],
        tokenizer=PreTrainedTokenizer,
        whitespace_pattern: Optional[str] = r"[\n\t\r ]*",
        verbose=False,
        max_same_state_visit_count=5,
    ):
        self.response_format = response_format
        self.tokenizer = tokenizer
        self.whitespace_pattern = whitespace_pattern
        self.dfa = None
        self.verbose = verbose
        self.max_same_state_visit_count = max_same_state_visit_count
        self.same_state_visit_count = 0
        self.current_state = 0
        self.previous_state = None
        self.final_states = None
        self.token_taken = None
        self.previous_input_ids = None

    def reset_state(self):
        """Reset the processor to its initial state"""
        self.current_state = 0
        self.final_states = None
        self.token_taken = None

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        current_length = input_ids.shape[-1]

        if self.previous_input_ids is not None:
            # Check if we're continuing from the previous sequence
            if not torch.equal(input_ids[:, :-1], self.previous_input_ids):
                # If the history doesn't match, reset the state
                self.reset_state()

        if self.final_states is None:  # first time
            if isinstance(self.response_format, dict) and all(
                isinstance(k, int)
                and isinstance(v, dict)
                and all(
                    isinstance(k2, int) and isinstance(v2, int) for k2, v2 in v.items()
                )
                for k, v in (self.response_format).items()
            ):
                self.dfa = self.response_format
            elif isinstance(self.response_format, str):
                self.dfa = build_dfa(self.response_format)
            elif issubclass(self.response_format, BaseModel):
                self.dfa = build_dfa(
                    self.response_format,
                    self.tokenizer,
                    whitespace_pattern=self.whitespace_pattern,
                )
            else:
                raise ValueError(
                    f"Cannot parse schema {self.response_format}. The schema must be either "
                    + "a Pydantic class, a dict[int, dict[int, int]] or a string that contains the JSON "
                    + "schema specification"
                )

            self.current_state = 0
            self.previous_state = None
            states = range(len(self.dfa) + 1)
            if self.verbose:
                print(f"states: {states}")
            self.final_states = {
                state for state in states if state not in list((self.dfa).keys())
            }
            if self.verbose:
                print(f"final states: {self.final_states}")
            self.previous_input_ids = input_ids.clone()
        else:
            self.token_taken = input_ids[:, -1].item()
            if self.verbose:
                print(
                    f"\x1b[32m\ntoken taken: {self.token_taken}: {repr(self.tokenizer.decode([self.token_taken]))}\x1b[0m"
                )
            if self.verbose:
                print(f"mapping: {self.dfa[self.current_state]}")
            self.previous_state = self.current_state
            self.current_state = self.dfa[self.current_state][self.token_taken]
            if self.previous_state == self.current_state:
                self.same_state_visit_count += 1
            else:
                self.same_state_visit_count = 0

        if self.verbose:
            print(f"\x1b[34mcurrent state: {self.current_state}\x1b[0m")
        if self.verbose:
            print(
                f"\x1b[33msame state visit count: {self.same_state_visit_count}\x1b[0m"
            )
        self.previous_input_ids = input_ids.clone()
        scores_processed = scores.clone()

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
                    key for key, value in mapping.items() if value != self.current_state
                ]

        if self.verbose:
            print(f"allowed tokens: {allowed_tokens}")

        vocab_tensor = torch.arange(scores.shape[-1], device=scores.device)
        allowed_tokens = torch.tensor(allowed_tokens, device=scores.device)
        forbidden_tokens_mask = ~torch.isin(vocab_tensor, allowed_tokens)
        scores_processed = torch.where(forbidden_tokens_mask, -torch.inf, scores)

        return scores_processed
