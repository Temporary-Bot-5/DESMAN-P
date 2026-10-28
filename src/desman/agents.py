"""Minimal LLM-backed executor and reviewer, so `desman run` can drive a task end to end.

Deliberately simple: one prompt each, and the reply must hold a JSON object (bare, fenced in
```json, or with a sentence around it) that fits the role's schema; a reply that does not gets
one repair request (`llm.structured`). Phase 2 makes them smarter (2.8 reviewer, 2.9 varied
retries); the contract with the coordinator stays the same.

Prompts start with "Execute subtask" and "Review subtask", so a fake script can route on them:
`FakeLLM().on("Execute", ...).on("Review", ...)`, or `"routes"` in a JSON script.
"""

from __future__ import annotations

import json
from typing import Any, TypeVar

from pydantic import BaseModel, RootModel

from desman.core.coordinator import Executor, Failure, Review, Reviewer, Success, Work
from desman.core.models import Subtask
from desman.core.retry import FailureKind
from desman.llm.base import LLM, Message, Request
from desman.llm.structured import StructuredOutputError, complete_json

T = TypeVar("T", bound=BaseModel)

EXECUTE = "Execute subtask"
REVIEW = "Review subtask"
"""Prompt headings: fake scripts route on them."""

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
        try:
            result = _ask(llm, EXECUTOR_SYSTEM, EXECUTE, prompt, _Result)
        except StructuredOutputError as e:
            return Failure(kind=FailureKind.QUALITY, reason=str(e))
        return Success(content=result.root)

    return execute


def llm_reviewer(llm: LLM) -> Reviewer:
    def review(subtask: Subtask, artifact: dict[str, Any] | None) -> Review:
        prompt = {
            "subtask": subtask.description,
            "acceptance_criteria": subtask.acceptance_criteria,
            "result": artifact,
        }
        # A StructuredOutputError is not the work's fault: the coordinator counts it as an
        # interrupted review, so the work is reviewed again, not redone.
        return _ask(llm, REVIEWER_SYSTEM, REVIEW, prompt, Review)

    return review


class _Result(RootModel[dict[str, Any]]):
    """An executor's result: any JSON object (subtasks have no result schema yet)."""


def _ask(llm: LLM, system: str, heading: str, payload: dict[str, Any], schema: type[T]) -> T:
    body = json.dumps(payload, ensure_ascii=False, indent=2)
    request = Request(
        messages=[
            Message(role="system", content=system),
            Message(role="user", content=f"{heading}:\n{body}"),
        ]
    )
    return complete_json(llm, request, schema, heading=heading)
