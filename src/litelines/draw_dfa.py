from collections import defaultdict
from typing import Optional

import graphviz
from graphviz import Digraph
from transformers import PreTrainedTokenizer

def create_row(
    token_id: int,
    tokenizer: PreTrainedTokenizer,
    remove_outer_whitespace: Optional[bool] = True,
) -> str:
    token = (
        tokenizer.decode([token_id]).strip()
        if remove_outer_whitespace
        else tokenizer.decode([token_id])
    )
    row = f"""<tr><td align="right"><font color="#00b4d8">{token_id}</font></td><td>{token}</td></tr>"""
    return row

def create_table(
    edges_between_state_and_next_state: list[int],
    tokenizer: PreTrainedTokenizer,
    max_labels_per_edge: Optional[int] = 3,
    remove_outer_whitespace: Optional[bool] = True,
) -> str:
    table_str = '<<table border="0" cellborder="1" cellspacing="0">'
    table_str += (
        '<tr><td bgcolor="#ffebcd">id</td><td bgcolor="#ffebcd">token</td></tr>'
    )
    if len(edges_between_state_and_next_state) > max_labels_per_edge:
        for token_id in edges_between_state_and_next_state[:max_labels_per_edge]:
            table_str += create_row(token_id, tokenizer, remove_outer_whitespace)
        table_str += """<tr><td align="right"><font color="#00b4d8">...</font></td><td>...</td></tr>"""
    else:
        for token_id in edges_between_state_and_next_state:
            table_str += create_row(token_id, tokenizer, remove_outer_whitespace)
    table_str += "</table>>"
    return table_str

def draw_dfa(
    dfa: dict[int, dict[int, int]],
    tokenizer: PreTrainedTokenizer,
    max_labels_per_edge: Optional[int] = 3,
    remove_outer_whitespace: Optional[bool] = True,
    render: Optional[bool] = True,
) -> graphviz.sources.Source:
    states = range(len(dfa) + 1)
    initial_state = 0
    final_states = {state for state in states if state not in list(dfa.keys())}
    graph_str = """// Allowed Transitions Graph\ndigraph {\n\trankdir=LR;ratio=0.1;"""
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
                graph_str += f"\n\t{state} -> {next_state} [label={table_str}]"
    graph_str += "\n}\n"
    return Source(graph_str) if render else graph_str
