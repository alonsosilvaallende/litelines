# Trick not to use transformers as a dependency
from typing import Protocol, List, Optional, Union, Tuple, runtime_checkable

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
