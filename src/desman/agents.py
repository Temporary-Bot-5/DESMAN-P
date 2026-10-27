"""Minimal LLM-backed executor and reviewer, so `desman run` can drive a task end to end.

Deliberately simple: one prompt each, and the reply must hold a JSON object (bare, fenced in
```json, or with a sentence around it). Phase 2 makes them robust (2.3 repair request, 2.8
reviewer, 2.9 varied retries); the contract with the coordinator stays the same.

Prompts start with "Execute subtask" and "Review subtask", so a fake script can route on them:
`FakeLLM().on("Execute", ...).on("Review", ...)`, or `"routes"` in a JSON script.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, NoReturn

from desman.core.coordinator import Executor, Failure, Review, Reviewer, Success, Work
from desman.core.models import Subtask
from desman.core.retry import FailureKind
from desman.llm.base import LLM, Message, Request

EXECUTOR_SYSTEM = (
    "You complete one subtask of a larger task. Reply with a single JSON object holding the "
    "result, and nothing else."
)
REVIEWER_SYSTEM = (
    "You check whether a subtask's result meets its acceptance criteria. Reply with a single JSON "
    'object: {"verdict": "accept" | "retry" | "escalate", "feedback": "what to fix"}. '
    'Use "escalate" only when a human must decide.'
)


def llm_executor(llm: LLM) -> Executor:
    def execute(work: Work) -> Success | Failure:
        context = work.context
        prompt = {
            "subtask": context.description,
            "acceptance_criteria": context.acceptance_criteria,
            "inputs": context.inputs,
            "results_of_dependencies": [
                {"subtask": d.description, "result": d.content, "truncated": d.truncated}
                for d in context.dependencies
            ],
            "checkpoint": context.checkpoint,
            "attempt": context.attempt,
            "feedback_on_previous_attempt": context.feedback.text if context.feedback else None,
        }
        reply = llm.complete(_request(EXECUTOR_SYSTEM, "Execute subtask", prompt))
        content = _json_object(reply.text)
        if content is None:
            return Failure(
                kind=FailureKind.QUALITY,
                reason=f"reply is not a JSON object: {reply.text[:120]!r}",
            )
        return Success(content=content)

    return execute


def llm_reviewer(llm: LLM) -> Reviewer:
    def review(subtask: Subtask, artifact: dict[str, Any] | None) -> Review:
        prompt = {
            "subtask": subtask.description,
            "acceptance_criteria": subtask.acceptance_criteria,
            "result": artifact,
        }
        reply = llm.complete(_request(REVIEWER_SYSTEM, "Review subtask", prompt))
        verdict = _json_object(reply.text)
        if verdict is None:
            # Not the work's fault: the coordinator counts it as an interrupted review.
            raise ValueError(f"reviewer reply is not a JSON object: {reply.text[:120]!r}")
        return Review.model_validate(verdict)

    return review


def _request(system: str, heading: str, payload: dict[str, Any]) -> Request:
    body = json.dumps(payload, ensure_ascii=False, indent=2)
    return Request(
        messages=[
            Message(role="system", content=system),
            Message(role="user", content=f"{heading}:\n{body}"),
        ]
    )


_FENCE = re.compile(r"```[ \t]*(?:json|JSON)?[ \t]*\n(.*?)\n?[ \t]*```", re.S)


def _json_object(text: str) -> dict[str, Any] | None:
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
