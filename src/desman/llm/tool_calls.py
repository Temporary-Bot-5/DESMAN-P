"""Tool-call contracts and normalization, without tool runtime dependencies."""

from __future__ import annotations

import json
from contextlib import suppress
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


class ToolCall(BaseModel):
    """One call asked for by an executor."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tool: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)]
    args: dict[str, Any] = Field(default_factory=dict)


def normalize_tool_calls(data: Any) -> Any:
    """Normalize native and aliased tool calls before strict contract validation."""
    if not isinstance(data, dict) or not isinstance(data.get("tool_calls"), list):
        return data
    items = []
    for item in data["tool_calls"]:
        if isinstance(item, dict):
            item = dict(item)
            if (
                item.get("type") == "function"
                and isinstance(item.get("function"), dict)
                and not (set(item["function"]) & (set(item) - {"function"}))
            ):
                function = item.pop("function")
                item.pop("id", None)
                if item.get("type") == "function":
                    item.pop("type")
                item = {**item, **function}
            if "name" in item and "tool" not in item:
                item["tool"] = item.pop("name")
            if "arguments" in item and "args" not in item:
                item["args"] = item.pop("arguments")
            if isinstance(item.get("args"), str):
                with suppress(ValueError):
                    item["args"] = json.loads(item["args"])
        items.append(item)
    return {**data, "tool_calls": items}
