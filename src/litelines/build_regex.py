import json
from typing import Any, Type, Union

from outlines_core.json_schema import build_regex_from_schema


def build_regex(
    schema: Union[dict, str, Type[Any]],
    include_tool_call: bool = False,
    tool_call_start: str = "<tool_call>",
    tool_call_end: str = "</tool_call>",
    whitespace_pattern: str = r"[\n\t ]*",
) -> str:
    """Convert a Pydantic model or JSON schema to a regex.

    Examples:
        >>> from typing import Literal
        >>> from pydantic import BaseModel, Field
        >>> from litelines import build_regex
        >>>
        >>> class Sentiment(BaseModel):
        ...     "Correctly inferred `Sentiment` with all the required parameters with correct types."
        ...
        ...     label: Literal["positive", "negative"] = Field(
        ...         ..., description="Sentiment of the text"
        ...     )
        >>> build_regex(Sentiment, whitespace_pattern="")
        '\\\\{"label":("positive"|"negative")\\\\}'
        >>> build_regex(Sentiment, whitespace_pattern="[ ]?")
        '[ ]?\\\\{[ ]?"label"[ ]?:[ ]?("positive"|"negative")[ ]?\\\\}'
        >>> build_regex(Sentiment)
        '[\\\\n\\\\t ]*\\\\{[\\\\n\\\\t ]*"label"[\\\\n\\\\t ]*:[\\\\n\\\\t ]*("positive"|"negative")[\\\\n\\\\t ]*\\\\}'
        >>> build_regex(Sentiment, include_tool_call=True, whitespace_pattern="")
        '<tool_call>\\\\{"name":"Sentiment","arguments":\\\\{"label":("positive"|"negative")\\\\}\\\\}</tool_call>'

    Args:
        schema: The Pydantic model or JSON schema.
        include_tool_call (optional): Is the Pydantic model expecting a tool call or not.
        tool_call_start (optional): The expected tool call start.
        tool_call_end (optional): The expected tool call end.
        whitespace_pattern (optional): Pattern to use for JSON syntactic whitespace.

    Returns:
        The JSON schema converted to a regex.

    Raises:
        ValueError: An error occurs if the schema is not a Pydantic model, a dictionary, or a string.
    """
    if isinstance(schema, dict):
        schema_str = json.dumps(schema)
        name_str = schema["title"]
    elif isinstance(schema, str):
        schema_str = schema
        name_str = json.loads(schema)["title"]
    elif hasattr(schema, "model_json_schema"):
        schema_str = json.dumps(schema.model_json_schema())
        name_str = schema.__name__
    else:
        raise ValueError(
            f"Cannot parse schema {schema}. The schema must be either "
            + "a Pydantic class, a dictionary or a string that contains the JSON "
            + "schema specification"
        )
    _regex_str = build_regex_from_schema(
        schema_str, whitespace_pattern=whitespace_pattern
    )
    if include_tool_call:
        regex_str = (
            whitespace_pattern
            + tool_call_start
            + whitespace_pattern
            + "\\{"
            + whitespace_pattern
            + '"name"'
            + whitespace_pattern
            + ":"
            + whitespace_pattern
            + '"'
            + name_str
            + '"'
            + whitespace_pattern
            + ","
            + whitespace_pattern
            + '"arguments"'
            + whitespace_pattern
            + ":"
            + whitespace_pattern
            + _regex_str
            + whitespace_pattern
            + "\\}"
            + whitespace_pattern
            + tool_call_end
        )
    else:
        regex_str = whitespace_pattern + _regex_str
    return regex_str
