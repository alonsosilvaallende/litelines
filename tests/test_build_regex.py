from typing import Literal

import pytest
from pydantic import BaseModel, Field

from litelines import build_regex

@pytest.fixture
def sample_pydantic_model():
    class Sentiment(BaseModel):
        """Correctly inferred `Sentiment` with all the required parameters with correct types."""
    
        label: Literal["positive", "negative"] = Field(
            ..., description="Sentiment of the text"
        )
    return Sentiment

TEST_CASES = [
    # Basic tests without tool call
    (
        {"whitespace_pattern": ""},
        '\\{"label":("positive"|"negative")\\}',
        "basic_no_whitespace"
    ),
    (
        {"whitespace_pattern": "[ ]?"},
        '[ ]?\\{[ ]?"label"[ ]?:[ ]?("positive"|"negative")[ ]?\\}',
        "basic_with_optional_spaces"
    ),
    (
        {"whitespace_pattern": None},
        '[\\n\\t ]*\\{[\\n\\t ]*"label"[\\n\\t ]*:[\\n\\t ]*("positive"|"negative")[\\n\\t ]*\\}',
        "basic_default_whitespace"
    ),
    
    # Tests with tool call
    (
        {"include_tool_call": True, "whitespace_pattern": ""},
        '<tool_call>\\{"name":"Sentiment","arguments":\\{"label":("positive"|"negative")\\}\\}</tool_call>',
        "with_tool_call_no_whitespace"
    ),
    (
        {"include_tool_call": True, "whitespace_pattern": "[ ]?"},
'[ ]?<tool_call>[ ]?\\{[ ]?"name"[ ]?:[ ]?"Sentiment"[ ]?,[ ]?"arguments"[ ]?:[ ]?\\{[ ]?"label"[ ]?:[ ]?("positive"|"negative")[ ]?\\}[ ]?\\}[ ]?</tool_call>',
        "with_tool_call_and_optional_spaces"
    ),
    (
        {"include_tool_call": True, "whitespace_pattern": None},
        '[\\n\\t ]*<tool_call>[\\n\\t ]*\\{[\\n\\t ]*"name"[\\n\\t ]*:[\\n\\t ]*"Sentiment"[\\n\\t ]*,[\\n\\t ]*"arguments"[\\n\\t ]*:[\\n\\t ]*\\{[\\n\\t ]*"label"[\\n\\t ]*:[\\n\\t ]*("positive"|"negative")[\\n\\t ]*\\}[\\n\\t ]*\\}[\\n\\t ]*</tool_call>',
        "with_tool_call_and default_whitespace"
    ),
    
    # Tests with custom tool call tags
    (
        {"tool_call_start": "<tool_call_start>", "whitespace_pattern":""},
        '\\{"label":("positive"|"negative")\\}',
        "with_custom_start_tag_no_tool_call"
    ),
    (
        {"tool_call_end": "<tool_call_end>", "whitespace_pattern":""},
        '\\{"label":("positive"|"negative")\\}',
        "with_custom_end_tag_no_tool_call"
    ),
    (
        {
            "include_tool_call": True,
            "tool_call_start": "<tool_call_start>",
            "whitespace_pattern": ""
        },
        '<tool_call_start>\\{"name":"Sentiment","arguments":\\{"label":("positive"|"negative")\\}\\}</tool_call>',
        "with_custom_start_tag"
    ),
    (
        {
            "include_tool_call": True,
            "tool_call_end": "<tool_call_end>",
            "whitespace_pattern": ""
        },
        '<tool_call>\\{"name":"Sentiment","arguments":\\{"label":("positive"|"negative")\\}\\}<tool_call_end>',
        "with_custom_end_tag"
    ),
    (
        {
            "include_tool_call": True,
            "tool_call_start": "<tool_call_start>",
            "tool_call_end": "<tool_call_end>",
            "whitespace_pattern": ""
        },
        '<tool_call_start>\\{"name":"Sentiment","arguments":\\{"label":("positive"|"negative")\\}\\}<tool_call_end>',
        "with_custom_tags"
    ),
]

TEST_IDS = [case[2] for case in TEST_CASES]

@pytest.mark.parametrize("test_params,expected,test_id", TEST_CASES, ids=TEST_IDS)
def test_build_regex(sample_pydantic_model, test_params, expected, test_id):
    params = {}
    for key, value in test_params.items():
        if key == "whitespace_pattern" and value is None:
            pass
        else:
            params[key] = value
    assert build_regex(sample_pydantic_model, **params) == expected


#
#def test_build_regex(sample_pydantic_model):
#    assert build_regex(sample_pydantic_model, whitespace_pattern="") == '\\{"label":("positive"|"negative")\\}'
#    assert build_regex(sample_pydantic_model, whitespace_pattern="[ ]?") == '[ ]?\\{[ ]?"label"[ ]?:[ ]?("positive"|"negative")[ ]?\\}'
#    assert build_regex(sample_pydantic_model) == '[\\n\\t ]*\\{[\\n\\t ]*"label"[\\n\\t ]*:[\\n\\t ]*("positive"|"negative")[\\n\\t ]*\\}'
#    assert build_regex(sample_pydantic_model, include_tool_call=True, whitespace_pattern="") == '<tool_call>\\{"name":"Sentiment","arguments":\\{"label":("positive"|"negative")\\}\\}</tool_call>'
#    assert build_regex(sample_pydantic_model, include_tool_call=True, whitespace_pattern="[ ]?") == '[ ]?<tool_call>[ ]?\\{[ ]?"name"[ ]?:[ ]?"Sentiment"[ ]?,[ ]?"arguments"[ ]?:[ ]?\\{[ ]?"label"[ ]?:[ ]?("positive"|"negative")[ ]?\\}[ ]?\\}[ ]?</tool_call>'
#    assert build_regex(sample_pydantic_model, include_tool_call=True) == '[\\n\\t ]*<tool_call>[\\n\\t ]*\\{[\\n\\t ]*"name"[\\n\\t ]*:[\\n\\t ]*"Sentiment"[\\n\\t ]*,[\\n\\t ]*"arguments"[\\n\\t ]*:[\\n\\t ]*\\{[\\n\\t ]*"label"[\\n\\t ]*:[\\n\\t ]*("positive"|"negative")[\\n\\t ]*\\}[\\n\\t ]*\\}[\\n\\t ]*</tool_call>'
#    assert build_regex(sample_pydantic_model, include_tool_call=True, tool_call_start="<tool_call_start>", whitespace_pattern="") == '<tool_call_start>\\{"name":"Sentiment","arguments":\\{"label":("positive"|"negative")\\}\\}</tool_call>'
#    assert build_regex(sample_pydantic_model, include_tool_call=True, tool_call_start="<tool_call_start>", tool_call_end="<tool_call_end>",  whitespace_pattern="") == '<tool_call_start>\\{"name":"Sentiment","arguments":\\{"label":("positive"|"negative")\\}\\}<tool_call_end>'
#    assert build_regex(sample_pydantic_model, include_tool_call=False, tool_call_start="<tool_call_start>", tool_call_end="<tool_call_end>",  whitespace_pattern="")  == '\\{"label":("positive"|"negative")\\}'
#
