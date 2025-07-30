import json
from typing import Optional, Type, Union

from outlines_core import Index, Vocabulary
from pydantic import BaseModel

from .build_regex import build_regex
from .utils import PreTrainedTokenizer, PreTrainedTokenizerFast

def my_recursive(
    state: int, index: Index, mapping: dict[int, int], visited: set[int], final_states: set[int]
) -> None:
    if state in final_states:
        return
    visited.add(state)
    for symbol, new_state in index.get_transitions().get(state, {}).items():
        if new_state in final_states:
            continue  # Skip final states entirely
        if new_state not in mapping:
            mapping[new_state] = len(mapping)
        if new_state not in visited:
            my_recursive(new_state, index, mapping, visited, final_states)

def get_state_mapping(index: Index) -> dict[int, int]:
    initial_state = index.get_initial_state()
    final_states = index.get_final_states()
    num_states = len(index.get_transitions().keys())
    mapping = {}
    # Start from initial state (mapped to 0)
    mapping[initial_state] = 0
    visited = set()
    my_recursive(initial_state, index, mapping, visited, final_states)
    # End with final states (mapped at the end)
    for i, final_state in enumerate(final_states):
        mapping[final_state] = num_states - (i + 1)
    return mapping

def get_dfa(index: Index) -> dict[int,dict[int,int]]:
    mapping = get_state_mapping(index)
    dfa = {}
    for state, transitions in index.get_transitions().items():
        new_transitions = {}
        for token, new_state in transitions.items():
            new_transitions[token] = mapping[new_state]
        if state not in index.get_final_states():
            dfa[mapping[state]] = new_transitions
    return dfa

def build_dfa(
    response_format: Union[dict, str, Type[BaseModel]],
    tokenizer: Union[str, PreTrainedTokenizer, PreTrainedTokenizerFast],
    include_tool_call: bool = False,
    tool_call_start: str = "<tool_call>",
    tool_call_end: str = "</tool_call>",
    whitespace_pattern: str = r"[\n\t\r ]*",
) -> dict[int, dict[int, int]]:

    if isinstance(response_format, str):
        regex_str = response_format
    elif isinstance(response_format, dict) or issubclass(response_format, BaseModel):
        regex_str = build_regex(
            response_format,
            include_tool_call=include_tool_call,
            tool_call_start=tool_call_start,
            tool_call_end=tool_call_end,
            whitespace_pattern=whitespace_pattern,
        )
    else:
        raise ValueError(
            f"Cannot parse {response_format}. The schema must be either "
            + "a Pydantic class, a dictionary or a string that corresponds to "
            + "the regular expression."
        )

    if isinstance(tokenizer, str):
        model_name = tokenizer
    elif isinstance(tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)):
        model_name = tokenizer.name_or_path
    else:
        raise ValueError(
            "The tokenizer must be either "
            + "a PreTrainedTokenizer, a PreTrainedTokenizerFast "
            + "or a string that corresponds to the model name."
        )

    vocabulary = Vocabulary.from_pretrained(model_name)
    index = Index(regex_str, vocabulary)
    dfa = get_dfa(index)
    return dfa
