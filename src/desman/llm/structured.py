"""Structured output: a model's reply as a validated Pydantic model, with one repair request.

Free models often answer almost right: the object in a ```json fence, a sentence around it, a
misspelt field, a verdict outside the allowed values. `complete_json()` finds the JSON object in
the reply and validates it against a schema. If that fails, it asks once more, showing the model
its own reply and what was wrong with it; a second failure raises `StructuredOutputError`.

Only one repair: every call spends free quota, and a model that fails twice in a row is better
served by the retry policy (another attempt, later another strategy, ROADMAP 2.9).

The repair message starts with the original heading ("Execute subtask (repair):"), so prompt
routes of a fake (`FakeLLM().on("Execute", ...)`) serve it from the same queue.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, NoReturn, TypeVar

from pydantic import BaseModel, ValidationError

from desman.llm.base import LLM, Completion, Message, Request

T = TypeVar("T", bound=BaseModel)

MAX_PROBLEMS = 5
"""Validation errors listed in a repair request; the rest are counted."""
REPLY_EXCERPT = 120
"""Characters of the bad reply kept in an error reason (the full reply is in the call log)."""


class StructuredOutputError(ValueError):
    """The reply still did not fit the schema after the repair request."""

    def __init__(self, problem: str, reply: Completion) -> None:
        super().__init__(f"{problem} (after a repair request): {reply.text[:REPLY_EXCERPT]!r}")
        self.problem = problem
        self.reply = reply


def complete_json(llm: LLM, request: Request, schema: type[T], *, heading: str) -> T:
    """Ask `llm` and return its reply validated as `schema`, repairing it once if needed.

    Provider errors propagate untouched: they are not the reply's fault.
    """
    reply = llm.complete(request)
    value, problem = _validate(reply, schema)
    if value is not None:
        return value
    repair = request.model_copy(
        update={
            "messages": [
                *request.messages,
                Message(role="assistant", content=reply.text),
                Message(
                    role="user",
                    content=(
                        f"{heading} (repair):\n{problem}\n"
                        "Reply again with only the corrected JSON object."
                    ),
                ),
            ]
        }
    )
    second = llm.complete(repair)
    value, problem = _validate(second, schema)
    if value is not None:
        return value
    raise StructuredOutputError(problem, second)


def _validate(reply: Completion, schema: type[T]) -> tuple[T, str] | tuple[None, str]:
    cut = " The reply was cut off by the token limit; make it shorter." * (
        reply.finish_reason == "length"
    )
    data = json_object(reply.text)
    if data is None:
        return None, f"The reply is not a JSON object.{cut}"
    try:
        return schema.model_validate(data), ""
    except ValidationError as e:
        return None, f"The JSON object does not match the schema:\n{_problems(e)}{cut}"


def _problems(error: ValidationError) -> str:
    # No input values: the model has its reply, and they could be long.
    errors = error.errors(include_url=False, include_input=False, include_context=False)
    lines = [
        f"- {'.'.join(str(part) for part in e['loc']) or '(object)'}: {e['msg']}"
        for e in errors[:MAX_PROBLEMS]
    ]
    if len(errors) > MAX_PROBLEMS:
        lines.append(f"- and {len(errors) - MAX_PROBLEMS} more")
    return "\n".join(lines)


_FENCE = re.compile(r"```[ \t]*(?:json|JSON)?[ \t]*\n(.*?)\n?[ \t]*```", re.S)


def json_object(text: str) -> dict[str, Any] | None:
    """The reply as a JSON object, or None.

    Models often wrap the object in a ```json fence or add a sentence around it, so after the
    whole reply these are tried: each fenced block, then the span from the first `{` to the
    last `}`. Strict JSON only: Python's parser also accepts NaN, Infinity and overflowing
    numbers such as 1e999, which no JSON store accepts.
    """
    candidates = [text, *_FENCE.findall(text)]
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            value = json.loads(
                candidate, parse_constant=_reject_constant, parse_float=_finite_float
            )
        except ValueError:  # JSONDecodeError is one
            continue
        if isinstance(value, dict):
            return value
    return None


def _reject_constant(name: str) -> NoReturn:
    raise ValueError(f"{name} is not JSON")


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"{text} does not fit a finite number")
    return value
