import json
from typing import Optional, Type, Union

from outlines_core.json_schema import build_regex_from_schema
from pydantic import BaseModel


def build_regex(
    json_schema: Union[dict, str, Type[BaseModel]],
    include_tool_call: bool = False,
    tool_call_start: str = "<tool_call>",
    tool_call_end: str = "</tool_call>",
    whitespace_pattern: str = r"[\n\t ]*",
) -> str:
    """Convert a JSON schema to a regex.

    Parameters
    ----------
    json_schema
        The JSON schema.

    Returns
    -------
    str
        The JSON schema converted to a regex.

    Raises
    ------
    ValueError
        If the schema is not a dictionary, a string or a Pydantic class.
    """
    if isinstance(json_schema, dict):
        schema_str = json.dumps(json_schema)
        name_str = json_schema["title"]
    elif isinstance(json_schema, str):
        schema_str = json_schema
        name_str = json.loads(json_schema)["title"]
    elif issubclass(json_schema, BaseModel):
        schema_str = json.dumps(json_schema.model_json_schema())
        name_str = json_schema.__name__
    else:
        raise ValueError(
            f"Cannot parse schema {json_schema}. The schema must be either "
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
