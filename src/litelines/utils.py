import re
import json
import subprocess
from typing import List, Optional, Protocol, Tuple, Union, runtime_checkable

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
        + "a Pydantic schema, a dict[int, dict[int, int]], a string that contains the JSON "
        + "schema specification or a string that contains the regular expression specification."
    )

def _in_notebook() -> bool:
    try:
        from IPython import get_ipython  # type: ignore

        if "IPKernelApp" not in get_ipython().config:
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True

def display_dot_graph(
    dot: str,
    render: bool = True,
) -> str | None:
    if not render:
        # we do not show a graph, nor save a graph to disk
        return dot
    try:
        graph = subprocess.check_output(
            ["dot", "-T" + "svg"], input=f"{dot}".encode()
        )
    except (ImportError, FileNotFoundError):
        msg = (
            "the graphviz `dot` binary should be on your PATH."
            "(If not installed you can download here: https://graphviz.org/download/)"
        )
        raise ImportError(msg) from None
        
    if _in_notebook():
        from IPython.display import SVG, display  # type: ignore

        return display(SVG(graph))
    else:
        from pathlib import Path

        current_dir = Path(".")
        file_path = current_dir / "graph.svg"
        file_path.write_bytes(graph)
        return None

# Trick not to use transformers as a dependency
@runtime_checkable
class PreTrainedTokenizer(Protocol):
    def __call__(
        self,
        text: Union[str, List[str]],
        text_pair: Optional[Union[str, List[str]]] = None,
        add_special_tokens: bool = True,
        padding: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
    ) -> dict:
        ...
    
    def encode(
        self,
        text: str,
        text_pair: Optional[str] = None,
        add_special_tokens: bool = True,
    ) -> List[int]:
        ...

    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = False,
    ) -> str:
        ...

@runtime_checkable
class PreTrainedTokenizerFast(PreTrainedTokenizer, Protocol):
    def encode_plus(
        self,
        text: Union[str, List[str]],
        text_pair: Optional[Union[str, List[str]]] = None,
        add_special_tokens: bool = True,
        padding: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
    ) -> dict:
        ...

    def batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[
            List[str],
            List[Tuple[str, str]],
            List[List[str]],
            List[Tuple[List[str], List[str]]],
        ],
        add_special_tokens: bool = True,
        padding: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
    ) -> dict:
        ...

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        ...

    @property
    def vocab_size(self) -> int:
        ...

    def get_vocab(self) -> dict:
        ...
