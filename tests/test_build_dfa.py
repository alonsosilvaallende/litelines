from typing import Literal

import pytest
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

from litelines import build_dfa


model_id = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_id)

@pytest.fixture
def sample_pydantic_model():
    class Sentiment(BaseModel):
        """Correctly inferred `Sentiment` with all the required parameters with correct types."""
    
        label: Literal["positive", "negative"] = Field(
            ..., description="Sentiment of the text"
        )
    return Sentiment

def test_build_dfa(sample_pydantic_model):
    assert build_dfa("A|B", tokenizer) == {0: {33: 1, 32: 1}}
    assert build_dfa("A0|B0", tokenizer) == {1: {15: 3}, 2: {15: 3}, 0: {33: 1, 32: 2}}
    assert build_dfa(sample_pydantic_model, tokenizer, whitespace_pattern="") == {18: {72: 15, 344: 17, 533: 16}, 9: {92: 28}, 20: {72: 21, 12303: 7, 275: 6, 3404: 8}, 23: {2974: 5, 25: 24}, 1: {14380: 2, 75: 25, 4260: 26, 1502: 4}, 14: {10251: 15, 83: 18}, 8: {9207: 28, 1: 9}, 22: {82: 20, 6321: 21, 46865: 6}, 4: {3252: 5, 1: 23, 788: 24}, 0: {4913: 1, 90: 27}, 13: {64: 14, 19488: 17, 266: 18, 1388: 16, 9307: 15}, 10: {68: 8}, 19: {436: 20, 78: 22, 34054: 6, 30724: 21}, 3: {75: 4}, 16: {9207: 28, 1: 9}, 12: {70: 13, 6743: 14}, 7: {586: 8, 85: 10}, 11: {68: 12, 15060: 16, 11188: 14, 791: 13}, 2: {68: 3, 301: 4}, 17: {68: 16}, 27: {92667: 4, 1: 1}, 6: {72: 7, 344: 10, 533: 8}, 5: {2724: 6, 77: 11, 28775: 13, 42224: 16, 79: 19, 5368: 22, 30487: 8, 966: 20, 811: 12}, 26: {1371: 3, 65: 2, 9779: 4}, 15: {586: 16, 85: 17}, 21: {10251: 7, 83: 6}, 24: {1: 5}, 25: {370: 2, 64: 26, 780: 4, 8229: 3}}

