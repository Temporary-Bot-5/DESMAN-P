"""Pure prose selection shared by executor final answers and human rendering."""

from typing import Any


def prose_key(result: dict[str, Any]) -> str | None:
    """Select the renderer's prose field, preserving its precedence and fallbacks."""
    for key in (
        "content",
        "text",
        "answer",
        "result",
        "output",
        "response_text",
        "response",
        "message",
        "summary",
        "final_answer",
        "answer_text",
        "reply",
    ):
        if isinstance(result.get(key), str):
            return key
    prose = [
        key
        for key, value in result.items()
        if isinstance(value, str) and ("\n" in value or len(value) >= 80)
    ]
    if len(prose) == 1 and all(
        isinstance(value, (str, int, float, bool, type(None))) for value in result.values()
    ):
        return prose[0]
    if len(result) == 1 and isinstance(next(iter(result.values())), str):
        return next(iter(result))
    return None


def unwrap(result: Any) -> Any:
    """Remove result envelopes consistently for detection and rendering."""
    while isinstance(result, dict) and result.get("type") == "result":
        if result.get("result") is not None:
            result = result["result"]
        else:
            return {key: value for key, value in result.items() if key != "type"}
    return result


def has_prose(result: Any) -> bool:
    """Whether the renderer finds prose, including a wrapped result."""
    result = unwrap(result)
    return isinstance(result, str) or (isinstance(result, dict) and prose_key(result) is not None)
