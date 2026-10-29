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
`other` LLM (another provider), when there is one; "combine" builds the result of a split
subtask from its parts (the split itself is `planner.decomposer`).

The executor acts only through tools (`core.tools`): instead of a result it may reply
`{"tool_calls": [{"tool": ..., "args": {...}}, ...]}`; the calls run through `Work.tools`, their
results go back to the model, and it continues, for at most `MAX_TOOL_ROUNDS` rounds. The key
`tool_calls` is reserved: an object holding it is always a request for tools, never a result.

Prompts start with "Execute subtask" and "Review subtask", so a fake script can route on them:
`FakeLLM().on("Execute", ...).on("Review", ...)`, or `"routes"` in a JSON script. Tool results
go back as "Execute subtask (tool results)", served by the same route.
"""

from __future__ import annotations

import json
from typing import Annotated, Any, Self, TypeVar

from pydantic import BaseModel, ConfigDict, Field, RootModel, StringConstraints, model_validator

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
from desman.core.tools import ToolCall
from desman.llm.base import LLM, Message, Request
from desman.llm.structured import StructuredOutputError, complete_json

T = TypeVar("T", bound=BaseModel)

EXECUTE = "Execute subtask"
REVIEW = "Review subtask"
TOOL_RESULTS = f"{EXECUTE} (tool results)"
"""Prompt headings: fake scripts route on them."""
MAX_TOOL_ROUNDS = 8
"""Rounds of tool calls in one attempt; a model still calling tools after them has failed it."""
MAX_CALLS_PER_ROUND = 10

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
EXECUTOR_SYSTEM_COMBINE = (
    EXECUTOR_SYSTEM + " This subtask was split into smaller parts, which are done: their "
    "results are among the results of dependencies. Build this subtask's result from them."
)
TOOLS_INSTRUCTIONS = (
    ' You cannot touch files or anything else yourself: use the tools listed under "tools". '
    "To call them, reply instead of the result with "
    '{"tool_calls": [{"tool": "name", "args": {...}}, ...]}; you will get their results and '
    "continue. Files you write stay in the task's workspace, where later subtasks can read them; "
    'say in the result which files you wrote. The key "tool_calls" is only for calling tools.'
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
        tools = work.tools.specs()
        if tools:
            prompt["tools"] = tools
        model, system, temperature = llm, EXECUTOR_SYSTEM, 0.0
        if context.strategy is Strategy.OTHER_MODEL and other is not None:
            model = other
        elif context.strategy in (Strategy.REPHRASE, Strategy.OTHER_MODEL):
            # Without another model to ask, the best one can do differently is rephrase.
            system, temperature = EXECUTOR_SYSTEM_REPHRASED, REPHRASE_TEMPERATURE
        elif context.strategy is Strategy.COMBINE:
            system = EXECUTOR_SYSTEM_COMBINE
        if tools:
            system += TOOLS_INSTRUCTIONS
        request = _request(system, EXECUTE, prompt, temperature)
        for _round in range(MAX_TOOL_ROUNDS + 1):
            try:
                # An error report is not cached: the next task asking the same may succeed.
                reply = complete_json(
                    model,
                    request,
                    _Result,
                    heading=EXECUTE,
                    keep=lambda r: not is_error_report(r.root),
                )
            except StructuredOutputError as e:
                # The problem only: the reply may echo anything (it is in the call log).
                return Failure(kind=FailureKind.QUALITY, reason=f"unusable reply: {e.problem}")
            calls = reply.calls()
            if calls is None:
                return Success(content=reply.root)
            if _round == MAX_TOOL_ROUNDS:
                break
            results = [
                {"tool": call.tool, **work.tools.call(call).model_dump(exclude_none=True)}
                for call in calls
            ]
            body = json.dumps(results, ensure_ascii=False, indent=2)
            request = request.model_copy(
                update={
                    "messages": [
                        *request.messages,
                        Message(role="assistant", content=json.dumps(reply.root)),
                        Message(role="user", content=f"{TOOL_RESULTS}:\n{body}"),
                    ]
                }
            )
        return Failure(
            kind=FailureKind.QUALITY,
            reason=f"still calling tools after {MAX_TOOL_ROUNDS} rounds, no result",
        )

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


class _ToolCalls(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_calls: list[ToolCall] = Field(min_length=1, max_length=MAX_CALLS_PER_ROUND)


class _Result(RootModel[dict[str, Any]]):
    """An executor's reply: tool calls, or the result (any other JSON object: subtasks have no
    result schema yet)."""

    @model_validator(mode="after")
    def _calls_are_well_formed(self) -> Self:
        if "tool_calls" in self.root:
            _ToolCalls.model_validate(self.root)  # a bad one gets the repair request
        return self

    def calls(self) -> list[ToolCall] | None:
        if "tool_calls" not in self.root:
            return None
        return _ToolCalls.model_validate(self.root).tool_calls


def _ask(llm: LLM, system: str, heading: str, payload: dict[str, Any], schema: type[T]) -> T:
    return complete_json(llm, _request(system, heading, payload, 0.0), schema, heading=heading)


def _request(system: str, heading: str, payload: dict[str, Any], temperature: float) -> Request:
    body = json.dumps(payload, ensure_ascii=False, indent=2)
    return Request(
        messages=[
            Message(role="system", content=system),
            Message(role="user", content=f"{heading}:\n{body}"),
        ],
        temperature=temperature,
    )
