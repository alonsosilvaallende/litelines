import re
import json
from collections import defaultdict
from typing import Optional, Tuple, Type, Union

from pydantic import BaseModel

from .build_dfa import build_dfa
from .build_regex import build_regex
from .utils import PreTrainedTokenizer, PreTrainedTokenizerFast, display_dot_graph


def contains_control_chars(s: str) -> bool:
    """
    Returns True if the string s contains any character in the specified ranges.
    """
    # Compile a regex pattern to detect any character in the range \x00-\x1F or \x7F-\x9F.
    pattern = re.compile(r"[\x00-\x1F\x7F-\x9F]")
    return pattern.search(s) is not None

def build_escaped_label(label: str) -> str:
    """
    Escapes special characters in a string to their corresponding HTML entities
    to ensure proper HTML rendering.

    Args:
        label (str): The input string to be escaped.

    Returns:
        str: The escaped string with special characters converted to HTML entities.

    Special character conversions:
        & -> &amp;
        < -> &lt;
        > -> &gt;
        ' -> &apos;
        [ -> &#91;
        ] -> &#93;
        \\ -> &#92;
    """
    html_entities = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        "'": '&apos;',
        '[': '&#91;',
        ']': '&#93;',
        '\\': '&#92;',
    }
    escaped = label
    for char, entity in html_entities.items():
        escaped = escaped.replace(char, entity)
    return escaped

def build_escaped_title(title: str) -> str:
    """
    Escapes special characters in a string title for safe handling.

    This function replaces backslashes, newlines, tabs, and double quotes
    with their escaped counterparts to prevent formatting issues and ensure
    proper string handling.

    Args:
        title (str): The input string to be escaped.

    Returns:
        str: The escaped string with special characters properly encoded.
    """
    escaped = title.replace('\\', '\\\\')
    escape_chars = {
        "\n": "\\n",
        "\t": "\\t",
        '"': '\\"',
    }
    for char, entity in escape_chars.items():
        escaped = escaped.replace(char, entity)
    return escaped

def create_row(
    token_id: int,
    tokenizer: PreTrainedTokenizer,
    remove_outer_whitespace: Optional[bool] = True,
) -> str:
    """
    Creates an HTML table row for a Graphviz DOT diagram containing token ID and its decoded value.

    Args:
        token_id (int): The ID of the token to be decoded
        tokenizer (PreTrainedTokenizer): The tokenizer used to decode the token ID
        remove_outer_whitespace (Optional[bool]): Whether to strip whitespace from the decoded token. 
            Defaults to True.

    Returns:
        str: An HTML-formatted table row string containing the token ID (in blue) and its decoded value
    """
    token = (
        (tokenizer.decode([token_id]).strip())
        if remove_outer_whitespace
        else tokenizer.decode([token_id])
    )
    escaped_token = build_escaped_label(token)
    if contains_control_chars(escaped_token):
        row = f"""<tr><td align="right"><font color="#00b4d8">{token_id}</font></td><td></td></tr>"""
    else:
        row = f"""<tr><td align="right"><font color="#00b4d8">{token_id}</font></td><td>{escaped_token}</td></tr>"""
    return row

def create_table(
    edges_between_state_and_next_state: list[int],
    tokenizer: PreTrainedTokenizer,
    max_labels_per_edge: Optional[int] = 3,
    remove_outer_whitespace: Optional[bool] = True,
) -> str:
    """
    Creates an HTML-formatted table string for use in a Graphviz DOT graph.
    
    The table displays token IDs and their corresponding decoded tokens, with a limit
    on the number of rows shown. If the number of edges exceeds max_labels_per_edge,
    only the first few entries are shown followed by an ellipsis.

    Args:
        edges_between_state_and_next_state (list(int)): List of token IDs to display in the table
        tokenizer (PreTrainedTokenizer): Tokenizer to decode the token IDs into text
        max_labels_per_edge (int): Maximum number of labels to show before truncating with ellipsis
        remove_outer_whitespace (bool): Whether to strip whitespace from decoded tokens

    Returns:
        A string containing HTML-like table markup for Graphviz DOT
    """
    table_str = '<table border="0" cellborder="1" cellspacing="0">'
    table_str += (
        '<tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr>'
    )
    if max_labels_per_edge is None:
        for token_id in edges_between_state_and_next_state:
            table_str += create_row(token_id, tokenizer, remove_outer_whitespace)
        table_str += """<tr><td align="right"><font color="#00b4d8">...</font></td><td>...</td></tr>"""
    elif len(edges_between_state_and_next_state) > max_labels_per_edge:
        for token_id in edges_between_state_and_next_state[:max_labels_per_edge]:
            table_str += create_row(token_id, tokenizer, remove_outer_whitespace)
        table_str += """<tr><td align="right"><font color="#00b4d8">...</font></td><td>...</td></tr>"""
    else:
        for token_id in edges_between_state_and_next_state:
            table_str += create_row(token_id, tokenizer, remove_outer_whitespace)
    table_str += "</table>"
    return table_str

def from_token_trajectory_to_state_trajectory(
    token_trajectory: list, dfa: dict[int, dict[int, int]]
) -> dict[int, list[int]]:
    """
    Converts a sequence of token IDs into a mapping of state transitions in a DFA.

    Args:
        token_trajectory (list): A list of token IDs representing the path through the DFA.
        dfa (dict[int, dict[int, int]]): A dictionary representing the DFA's transition function,
            where the outer key is the current state, inner key is the token ID,
            and the value is the next state.

    Returns:
        dict: A dictionary mapping each state to a list of its next states in the trajectory,
            where each state appears only once in the list.

    Example:
        token_trajectory = [1, 2, 3]
        dfa = {
            0: {1: 1},
            1: {2: 2},
            2: {3: 3}
        }
        result = {0: [1], 1: [2], 2: [3]}
    """
    state_trajectory = {}
    current_state = 0
    for i, token_id in enumerate(token_trajectory):
        if current_state not in state_trajectory.keys():
            state_trajectory[current_state] = [dfa[current_state][token_id]]
        else:
            if dfa[current_state][token_id] not in state_trajectory[current_state]:
                state_trajectory[current_state].append(dfa[current_state][token_id])
        current_state = dfa[current_state][token_id]
    return state_trajectory

def is_valid_json(s: str) -> bool:
    try:
        json.loads(s)
        return True
    except json.JSONDecodeError:
        return False

def is_valid_regex(pattern: str) -> bool:
    try:
        re.compile(pattern)
        return True
    except re.error:
        return False

def invalid_schema_error(dfa: object) -> None:
    raise ValueError(
            f"Cannot parse schema of type {type(dfa).__name__}: {dfa}. The schema must be either "
            + "a Pydantic class, a dict[int, dict[int, int]], a string that contains the JSON "
            + "schema specification or a string that contains the regular expression specification."
        )

def draw_dfa(
    dfa: Union[dict[int, dict[int, int]], str, Type[BaseModel]],
    tokenizer: PreTrainedTokenizer,
    trajectory: list = [],
    include_tool_call: bool = False,
    tool_call_start: str = "<tool_call>",
    tool_call_end: str = "</tool_call>",
    whitespace_pattern: str = r"[\n\t ]*",
    max_labels_per_edge: int = 3,
    remove_outer_whitespace: bool = True,
    ratio: Optional[Union[float, str]] = None,
    size: Optional[Union[Tuple[float, float], str]] = None,
    render: bool = True,
) -> str | None:
    """Create a graphical representation of a Deterministic Finite Automaton (DFA) using Graphviz DOT language.

    The function visualizes the DFA with:
    
    - states as circles (double circles for final states)
    - directed edges showing transitions between states
    - edge labels containing tables of token IDs and their corresponding text
    - optional red highlighting for edges in the provided trajectory 

    Examples:
        >>> from typing import Literal
        >>> from pydantic import BaseModel, Field
        >>> from transformers import AutoTokenizer
        >>> from litelines import build_dfa
        >>>
        >>> model_id = "Qwen/Qwen3-0.6B"
        >>> tokenizer = AutoTokenizer.from_pretrained(model_id)
        >>> draw_dfa("A|B", tokenizer, render=False)
        #
        >>> draw_dfa("A0|B0", tokenizer, render=False)
        #
        >>>
        >>> class Sentiment(BaseModel):
        ...     "Correctly inferred `Sentiment` with all the required parameters with correct types."
        ...
        ...     label: Literal["positive", "negative"] = Field(
        ...         ..., description="Sentiment of the text"
        ...     )
        >>> draw_dfa(Sentiment, tokenizer, whitespace_pattern="")
        #

    Args:
        dfa: The DFA representation, which can be either:
            A dictionary mapping states to their transitions
            A JSON schema string
            A Pydantic BaseModel class
        tokenizer: The tokenizer used to decode token IDs into readable text
        trajectory: Optional list of tokens representing a path through the DFA
        include_tool_call (optional): Is the Pydantic model expecting a tool call or not.
        tool_call_start (optional): The expected tool call start.
        tool_call_end (optional): The expected tool call end.
        whitespace_pattern (optional): Pattern to use for JSON syntactic whitespace.
        max_labels_per_edge (optional): Maximum number of labels to show per edge
        remove_outer_whitespace (optional): Whether to strip whitespace from token labels in the table.
        render (optional): Whether to return a rendered Graphviz Source object or raw DOT string

    Returns:
        A Graphviz Source object if render=True, otherwise the DOT language string
    """

    if isinstance(dfa, dict) and all(
        isinstance(k, int)
        and isinstance(v, dict)
        and all(isinstance(k2, int) and isinstance(v2, int) for k2, v2 in v.items())
        for k, v in dfa.items()
    ):
        regex = ""
        dfa = dfa
    elif isinstance(dfa, str):
        if is_valid_json(dfa):
            regex = build_regex(dfa, include_tool_call=include_tool_call, whitespace_pattern=whitespace_pattern)
            dfa = build_dfa(dfa, tokenizer=tokenizer, include_tool_call=include_tool_call, whitespace_pattern=whitespace_pattern)
        elif is_valid_regex(dfa):
            regex = dfa
            dfa = build_dfa(dfa, tokenizer=tokenizer, include_tool_call=include_tool_call, whitespace_pattern=whitespace_pattern)
        else:
            invalid_schema_error(dfa)
    elif isinstance(dfa, type) and issubclass(dfa, BaseModel):
        regex = build_regex(dfa,include_tool_call=include_tool_call, whitespace_pattern=whitespace_pattern)
        dfa = build_dfa(dfa, tokenizer=tokenizer, include_tool_call=include_tool_call, whitespace_pattern=whitespace_pattern)
    else:
        invalid_schema_error(dfa)

    if trajectory != []:
        state_trajectory = from_token_trajectory_to_state_trajectory(trajectory,dfa)

    states = range(len(dfa) + 1)
    final_states = {state for state in states if state not in list(dfa.keys())}
    graph_str = '// Allowed Transitions Graph\ndigraph {'
    if regex != "":
        graph_str += f'\n\tgraph [label="Allowed Paths\nRegular expression: {build_escaped_title(regex)}",labelloc="t",labeljust="l"]'
    else:
        graph_str += f'\n\tgraph [label="Allowed Paths",labelloc="t",labeljust="l"]'
    graph_str += f'\n\trankdir=LR;size="{size}";ratio={ratio};'
    # Add states to the graph
    for state in states:
        if state in final_states:
            # Shape the final states with double circle
            graph_str += f'\n\t{state} [label="{state}" shape=doublecircle]'
        else:
            # Shape the other states with a circle
            graph_str += f'\n\t{state} [label="{state}" shape=circle]'
    # Add empty fake node for initial arrow
    graph_str += '\n\tnode [shape=none]\n\t"" [label=""]\n\t"" -> 0'
    # Put together all edges from state to next_state to the graph
    all_edges = defaultdict(list)
    for state, transitions in dfa.items():
        for key, next_state in transitions.items():
            all_edges[(state, next_state)].append(key)
    # Add edges to the graph
    for state in states:
        for next_state in states:
            if all_edges[(state, next_state)] != []:
                table_str = create_table(
                    all_edges[(state, next_state)],
                    tokenizer,
                    max_labels_per_edge=3,
                    remove_outer_whitespace=True,
                )
                if trajectory != [] and state_trajectory != {} and state in state_trajectory.keys() and next_state in state_trajectory[state]:
                    graph_str += f"\n\t{state} -> {next_state} [label=<{table_str}> color=red penwidth=3.0]"
                else:
                    graph_str += f"\n\t{state} -> {next_state} [label=<{table_str}>]"
    graph_str += "\n}\n"
    return display_dot_graph(dot=graph_str, render=render)
