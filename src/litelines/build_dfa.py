import json
from typing import Optional, Type, Union

from outlines_core import Index, Vocabulary
from pydantic import BaseModel

from litelines import build_regex
from litelines.utils import PreTrainedTokenizer, PreTrainedTokenizerFast

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
    """Build a deterministic finite automaton that fullfils the response format requirement

    Examples:
        >>> from typing import Literal
        >>> from pydantic import BaseModel, Field
        >>> from transformers import AutoTokenizer
        >>> from litelines import build_dfa
        >>> 
        >>> model_id = "Qwen/Qwen3-0.6B"
        >>> tokenizer = AutoTokenizer.from_pretrained(model_id)
        >>> build_dfa("A|B", tokenizer)
        {0: {33: 1, 32: 1}}
        >>> build_dfa("A0|B0", tokenizer)
        {1: {15: 3}, 2: {15: 3}, 0: {33: 1, 32: 2}}
        >>>
        >>> class Sentiment(BaseModel):
        ...     "Correctly inferred `Sentiment` with all the required parameters with correct types."
        ...
        ...     label: Literal["positive", "negative"] = Field(
        ...         ..., description="Sentiment of the text"
        ...     )
        >>> build_dfa(Sentiment, tokenizer, whitespace_pattern="")
        {18: {72: 15, 344: 17, 533: 16}, 9: {92: 28}, 20: {72: 21, 12303: 7, 275: 6, 3404: 8}, 23: {2974: 5, 25: 24}, 1: {14380: 2, 75: 25, 4260: 26, 1502: 4}, 14: {10251: 15, 83: 18}, 8: {9207: 28, 1: 9}, 22: {82: 20, 6321: 21, 46865: 6}, 4: {3252: 5, 1: 23, 788: 24}, 0: {4913: 1, 90: 27}, 13: {64: 14, 19488: 17, 266: 18, 1388: 16, 9307: 15}, 10: {68: 8}, 19: {436: 20, 78: 22, 34054: 6, 30724: 21}, 3: {75: 4}, 16: {9207: 28, 1: 9}, 12: {70: 13, 6743: 14}, 7: {586: 8, 85: 10}, 11: {68: 12, 15060: 16, 11188: 14, 791: 13}, 2: {68: 3, 301: 4}, 17: {68: 16}, 27: {92667: 4, 1: 1}, 6: {72: 7, 344: 10, 533: 8}, 5: {2724: 6, 77: 11, 28775: 13, 42224: 16, 79: 19, 5368: 22, 30487: 8, 966: 20, 811: 12}, 26: {1371: 3, 65: 2, 9779: 4}, 15: {586: 16, 85: 17}, 21: {10251: 7, 83: 6}, 24: {1: 5}, 25: {370: 2, 64: 26, 780: 4, 8229: 3}}

    Args:
        response_format: A Pydantic model, a dictionary, or a regular expression (as a string) that defines the expected response format
        tokenizer: The model's tokenizer or the model name (as a string)
        include_tool_call (optional): Is the Pydantic model expecting a tool call or not.
        tool_call_start (optional): The expected tool call start.
        tool_call_end (optional): The expected tool call end.
        whitespace_pattern (optional): Pattern to use for JSON syntactic whitespace.

    Returns:
        The deterministic finite automaton as a dictionary.

    Raises:
        ValueError: An error occurs if the response format is not a Pydantic model, a dictionary, or a string that corresponds to the regular expression.
    """
    if isinstance(response_format, str):
        regex_str = response_format
    elif isinstance(response_format, dict) or isinstance(response_format, type) and issubclass(response_format, BaseModel):
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
            + "a Pydantic model, a dictionary or a string that corresponds to "
            + "the regular expression."
        )

    if isinstance(tokenizer, str):
        model_name = tokenizer
    elif isinstance(tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)):
        model_name = getattr(tokenizer, "name_or_path", None)
        if model_name is None:
            raise ValueError(
                    "Could not determine model name from tokenizer. "
                    + "You can pass it directly to the build_dfa function."
            )
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
