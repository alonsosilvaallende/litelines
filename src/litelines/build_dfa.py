from pickle import dumps
from typing import (
    Any,
    Dict,
    Hashable,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    Type,
    Union,
)

import json
import interegular
import numpy as np
import xxhash
from numpy.typing import NDArray
from outlines_core.fsm.regex import create_fsm_index_tokenizer, make_deterministic_fsm
from pydantic import BaseModel
from .build_regex import build_regex
from .utils import PreTrainedTokenizer

class Hasher:
    """Hasher that accepts python objects as inputs."""

    dispatch: dict = {}

    def __init__(self):
        self.m = xxhash.xxh64()

    @classmethod
    def hash_bytes(cls, value: Union[bytes, list[bytes]]) -> str:
        value = [value] if isinstance(value, bytes) else value
        m = xxhash.xxh64()
        for x in value:
            m.update(x)
        return m.hexdigest()

    @classmethod
    def hash(cls, value: Any) -> str:
        return cls.hash_bytes(dumps(value))

    def update(self, value: Any) -> None:
        header_for_update = f"=={type(value)}=="
        value_for_update = self.hash(value)
        self.m.update(header_for_update.encode("utf8"))
        self.m.update(value_for_update.encode("utf-8"))

    def hexdigest(self) -> str:
        return self.m.hexdigest()


class Tokenizer(Hashable, Protocol):
    eos_token: str
    eos_token_id: int
    pad_token_id: int
    vocabulary: Dict[str, int]
    special_tokens: Set[int]

    def encode(
        self, prompt: Union[str, List[str]]
    ) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Translate the input prompts into arrays of token ids and attention mask."""
        ...

    def decode(self, token_ids: NDArray[np.int64]) -> List[str]:
        """Translate an array of token ids to a string or list of strings."""
        ...

    def convert_token_to_string(self, token: str) -> str:
        """Convert a token to its equivalent string.

        This is for instance useful for BPE tokenizers where whitespaces are
        represented by the special characted `Ġ`. This prevents matching a raw
        token that includes `Ġ` with a string.
        """
        ...


class TransformerTokenizer(Tokenizer):
    """Represents a tokenizer for models in the `transformers` library."""

    def __init__(self, tokenizer: PreTrainedTokenizer, **kwargs):
        self.tokenizer = tokenizer
        self.eos_token_id = self.tokenizer.eos_token_id
        self.eos_token = self.tokenizer.eos_token

        if not self.tokenizer.pad_token_id:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.pad_token_id = self.eos_token_id
        else:
            self.pad_token_id = self.tokenizer.pad_token_id
            self.pad_token = self.tokenizer.pad_token

        self.special_tokens = set(self.tokenizer.all_special_tokens)

        self.vocabulary = self.tokenizer.get_vocab()
        self.is_llama = False

    def encode(
        self, prompt: Union[str, List[str]], **kwargs
    ) -> Tuple["torch.LongTensor", "torch.LongTensor"]:
        kwargs["padding"] = True
        kwargs["return_tensors"] = "pt"
        output = self.tokenizer(prompt, **kwargs)
        return output["input_ids"], output["attention_mask"]

    def decode(self, token_ids: "torch.LongTensor") -> List[str]:
        text = self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)
        return text

    def convert_token_to_string(self, token: str) -> str:
        string = self.tokenizer.convert_tokens_to_string([token])
        return string

    def __eq__(self, other):
        if isinstance(other, type(self)):
            if hasattr(self, "model_name") and hasattr(self, "kwargs"):
                return (
                    other.model_name == self.model_name and other.kwargs == self.kwargs
                )
            else:
                return other.tokenizer == self.tokenizer
        return NotImplemented

    def __hash__(self):

        return hash(Hasher.hash(self.tokenizer))

    def __getstate__(self):
        state = {"tokenizer": self.tokenizer}
        return state

    def __setstate__(self, state):
        self.__init__(state["tokenizer"])



def is_json(string):
    try:
        json.loads(string)
        return True
    except json.JSONDecodeError:
        return False

def build_dfa(
    regex_str: Union[str, Type[BaseModel]],
    tokenizer: PreTrainedTokenizer,
    include_tool_call: Optional[bool] = False,
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
    list_of_strings_fsm = interegular.parse_pattern(regex_str).to_fsm()
    new_fsm, _ = make_deterministic_fsm(list_of_strings_fsm)
    new_tokenizer = TransformerTokenizer(tokenizer)
    index, _ = create_fsm_index_tokenizer(new_fsm, new_tokenizer)
    return index
