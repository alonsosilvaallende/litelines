# Trick not to use transformers as a dependency
from typing import Protocol, List, Optional, Union

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
