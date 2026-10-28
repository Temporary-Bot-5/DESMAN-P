"""Minimal LLM-backed executor and reviewer, so `desman run` can drive a task end to end.

Deliberately simple: one prompt each, and the reply must hold a JSON object (bare, fenced in
```json, or with a sentence around it) that fits the role's schema; a reply that does not gets
one repair request (`llm.structured`).

The executor may answer `{"type": "error", "reason": ...}` when it cannot do the work: the
coordinator counts that as a failed attempt, never as a result. The reviewer must give concrete
feedback with `retry` and `escalate` (a verdict without it gets the repair request); the
feedback is shown to the next attempt.

Each attempt of the executor follows the strategy its number picks (`core.retry.Strategy`): the
first try and the one with the reviewer's feedback use the instructions below as they are; a
rephrased attempt gets them worded differently at a higher temperature; "other model" asks the
`other` LLM (another provider), when there is one.

Prompts start with "Execute subtask" and "Review subtask", so a fake script can route on them:
`FakeLLM().on("Execute", ...).on("Review", ...)`, or `"routes"` in a JSON script.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Annotated, Any, Self, TypeVar

from pydantic import BaseModel, RootModel, StringConstraints, model_validator

from desman.core.coordinator import (
    Executor,
    Failure,
    Review,
    Reviewer,
    Success,
    Verdict,
    Work,
    is_error_report,
)
from desman.core.models import Subtask
from desman.core.retry import FailureKind, Strategy
from desman.llm.base import LLM, Message, Request
from desman.llm.structured import StructuredOutputError, complete_json

T = TypeVar("T", bound=BaseModel)

EXECUTE = "Execute subtask"
REVIEW = "Review subtask"
"""Prompt headings: fake scripts route on them."""

EXECUTOR_SYSTEM = (
    "You complete one subtask of a larger task. Reply with a single JSON object holding the "
    "result, and nothing else. If feedback on a previous attempt is given, fix every point of "
    'it. If you cannot do the subtask, reply {"type": "error", "reason": "why"} instead of '
    "pretending: such a reply is never taken as a result."
)
EXECUTOR_SYSTEM_REPHRASED = (
    "Earlier attempts at this subtask were rejected. Start again from scratch rather than "
    "patching them: read the subtask and its acceptance criteria, then the feedback, and check "
    "each criterion before answering. Your whole reply must be one JSON object with the result. "
    "When the subtask cannot be done, the JSON object is "
    '{"type": "error", "reason": "why"}; never present a guess as a result.'
)
REPHRASE_TEMPERATURE = 0.7
"""A rephrased attempt also samples differently: at 0 the model tends to repeat itself."""
REVIEWER_SYSTEM = (
    "You check whether a subtask's result meets its acceptance criteria. Reply with a single JSON "
    'object: {"verdict": "accept" | "retry" | "escalate", "feedback": "..."}. '
    '"accept": every criterion is met. "retry": the result can be fixed by another attempt; '
    "the feedback must say concretely what is wrong and what to change, point by point, since "
    'it is all the next attempt sees of your review. "escalate": only a human can decide '
    "(the subtask is unclear, impossible or unsafe); the feedback says why. Never accept a "
    'result of type "error".'
)


def llm_executor(llm: LLM, other: LLM | None = None) -> Executor:
    """`other` serves the attempts whose strategy is another model (if None, `llm` does)."""

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
        model, system, temperature = llm, EXECUTOR_SYSTEM, 0.0
        if context.strategy is Strategy.OTHER_MODEL and other is not None:
            model = other
        elif context.strategy in (Strategy.REPHRASE, Strategy.OTHER_MODEL):
            # Without another model to ask, the best one can do differently is rephrase.
            system, temperature = EXECUTOR_SYSTEM_REPHRASED, REPHRASE_TEMPERATURE
        try:
            # An error report is not cached: the next task asking the same may succeed.
            result = _ask(
                model,
                system,
                EXECUTE,
                prompt,
                _Result,
                keep=lambda r: not is_error_report(r.root),
                temperature=temperature,
            )
        except StructuredOutputError as e:
            # The problem only: the reply may echo anything (it is in the call log).
            return Failure(kind=FailureKind.QUALITY, reason=f"unusable reply: {e.problem}")
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
        verdict = _ask(llm, REVIEWER_SYSTEM, REVIEW, prompt, _Verdict)
        return Review(verdict=verdict.verdict, feedback=verdict.feedback)

    return review


class _Verdict(BaseModel):
    """The reviewer model's reply: feedback is required unless the result is accepted."""

    verdict: Verdict
    feedback: Annotated[str, StringConstraints(strip_whitespace=True)] = ""

    @model_validator(mode="after")
    def _says_what_is_wrong(self) -> Self:
        if self.verdict is not Verdict.ACCEPT and not self.feedback:
            raise ValueError(f'a "{self.verdict.value}" verdict needs feedback: what is wrong')
        return self


class _Result(RootModel[dict[str, Any]]):
    """An executor's result: any JSON object (subtasks have no result schema yet)."""


def _ask(
    llm: LLM,
    system: str,
    heading: str,
    payload: dict[str, Any],
    schema: type[T],
    keep: Callable[[T], bool] = lambda value: True,
    temperature: float = 0.0,
) -> T:
    body = json.dumps(payload, ensure_ascii=False, indent=2)
    request = Request(
        messages=[
            Message(role="system", content=system),
            Message(role="user", content=f"{heading}:\n{body}"),
        ],
        temperature=temperature,
    )
    return complete_json(llm, request, schema, heading=heading, keep=keep)
