import json
from typing import Optional, Type, Union

from outlines_core import Index, Vocabulary
from pydantic import BaseModel

from .build_regex import build_regex
from .utils import PreTrainedTokenizer, PreTrainedTokenizerFast

def is_json(string):
    try:
        json.loads(string)
        return True
    except json.JSONDecodeError:
        return False

def add_tool_call_to_index(
    dfa: dict[int, dict[int, int]],
    tokenizer: PreTrainedTokenizer,
    tool_call_start: Optional[str] = "<tool_call>",
    tool_call_end: Optional[str] = "</tool_call>",
    k: Optional[int] = 1,
) -> dict[int, dict[int, int]]:
    original_states = range(len(dfa) + 1)
    final_states = {state for state in original_states if state not in list(dfa.keys())}
    new_dfa = {}
    for state, transitions in dfa.items():
        new_transitions = {
            key: next_state + k for key, next_state in transitions.items()
        }
        new_dfa[state + k] = new_transitions
    if len(tokenizer.encode(tool_call_start, add_special_tokens=False))>1:
        raise ValueError(
            f"{tool_call_start} is not a valid token"
        )
    elif len(tokenizer.encode(tool_call_end, add_special_tokens=False))>1:
        raise ValueError(
            f"{tool_call_end} is not a valid token"
        )
    else:
        new_dfa[0] = {int(tokenizer.encode(tool_call_start, add_special_tokens=False)[0]): 1}
        for final_state in final_states:
            new_dfa[final_state + k] = {
                int(tokenizer.encode(tool_call_end, add_special_tokens=False)[0]): len(dfa) + 2
            }
    return new_dfa


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
    regex_str: Union[str, Type[BaseModel]],
    tokenizer: Union[str, PreTrainedTokenizer, PreTrainedTokenizerFast],
    include_tool_call: Optional[bool] = False,
    tool_call_start: Optional[str] = "<tool_call>",
    tool_call_end: Optional[str] = "</tool_call>",
    whitespace_pattern: Optional[str] = r"[\n\t\r ]*",
) -> dict[int, dict[int, int]]:
    if isinstance(regex_str, str) and is_json(regex_str):
        regex_str = build_regex(
            regex_str, include_tool_call=include_tool_call, whitespace_pattern=whitespace_pattern
        )
    elif issubclass(regex_str, BaseModel):
        regex_str = build_regex(
            regex_str, include_tool_call=include_tool_call, whitespace_pattern=whitespace_pattern
        )
    else:
        raise ValueError(
            f"Cannot parse schema {regex_str}. The schema must be either "
            + "a Pydantic class, a dictionary or a string that contains the JSON "
            + "schema specification"
        )
    if isinstance(tokenizer, str):
        model_name = tokenizer
    elif isinstance(tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)):
        model_name = tokenizer.name_or_path
    else:
        raise ValueError(
            f"Cannot parse schema {regex_str}. The schema must be either "
            + "a Pydantic class, a dictionary or a string that contains the JSON "
            + "schema specification"
        )
    vocabulary = Vocabulary.from_pretrained(model_name)
    index = Index(regex_str, vocabulary)
    if include_tool_call:
        index = add_tool_call_to_index(
            index, tokenizer, tool_call_start=tool_call_start, tool_call_end=tool_call_end
        )
    dfa = get_dfa(index)
    return dfa
