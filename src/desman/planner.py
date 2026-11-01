"""The planner: a goal becomes a graph of subtasks (D1).

The model is asked for a `Plan`: at most `MAX_SUBTASKS` subtasks, each with acceptance criteria
and the indices of the subtasks it needs. The plan is validated like any task spec (indices in
range, no cycles), so a broken graph gets the one repair request of `llm.structured`, with the
reason, like any other reply that does not fit.

Planning never stops a task from starting. When the model fails (provider error, a reply that
is still wrong after the repair), the template plan is used: one subtask, the goal itself. The
reason is shown to the operator and kept in the audit (`task_created`, `plan`).

Each subtask is tagged with its kind (`Kind`: code, text, analysis): its executor's model is
picked by it (ROADMAP 4.5). A subtask the model leaves untagged is text; a part of a split
subtask left untagged is of the subtask's kind. A subtask that must run code is marked
`runs_code` (`RUN_RULES`): the coordinator, not a model, then checks that it really ran.

`decomposer()` splits one subtask the same way when its retries keep failing (ROADMAP 2.9b).

The prompts start with "Plan task" and "Split subtask", so a fake script can route on them.
"""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, Field, StringConstraints, ValidationError, model_validator

from desman.core.coordinator import Decomposer, Work
from desman.core.models import Kind, Priority, SubtaskSpec, TaskSpec
from desman.llm.base import LLM, Message, ProviderError, Request
from desman.llm.structured import StructuredOutputError, complete_json

PLAN = "Plan task"
"""Prompt heading: fake scripts route on it."""
MAX_SUBTASKS = 8
"""A small model loses the thread of a long plan, and every subtask costs at least two calls."""
TEMPLATE_CRITERIA = "The result fully achieves the stated goal."

CRITERIA_RULES = (
    " Acceptance criteria are checked against the result and what the tools returned (web "
    "search, pages, files), so write only criteria such a check can confirm. Never require "
    "what a search may well not give: an exact time or seat on a date, a booking page, the "
    "current lowest price anywhere. For live data (prices, schedules, availability) ask for "
    "the best that was found, each fact with its source, and a plain statement of what was "
    "not found."
)

KIND_RULES = (
    ' "kind" says what sort of work the subtask is, so a model good at it does it: "code" '
    '(writing, changing or explaining programs), "analysis" (reasoning over given material: '
    'comparing, judging, working something out) or "text" (writing, answering, finding facts).'
)

RUN_RULES = (
    " A subtask that must run code or a command and report what it printed (testing a "
    'program, measuring, checking that something works) also gets "runs_code": true. Only '
    "such a subtask is given a way to run anything, and it is done only once it really ran: "
    "never plan a subtask that says what running something would print without it."
)
"""Not a target of the Improvement Analyst: the analyst may not loosen what code checks."""

PLANNER_RULES = (
    "You split a goal into subtasks for small language models. Each subtask is done by one "
    "model call that sees only that subtask's description, its acceptance criteria and the "
    "results of the subtasks it depends on, never the goal itself, so every description must "
    "make sense on its own. Use as few subtasks as the goal needs (one is fine for a simple "
    "goal). Reply with a single JSON object: "
    '{"subtasks": [{"description": "what to do", "acceptance_criteria": "how to tell it is '
    'done", "kind": "text" | "code" | "analysis", "depends_on": [indices of earlier subtasks, '
    "counting from 0]}]}. The last subtask must produce the final result of the goal."
)
"""The planner's own instructions. Like `KIND_RULES`, `CRITERIA_RULES` and the limits, a plain
value the Improvement Analyst may propose to change (`improve.TARGETS`)."""


def planner_system(limit: int = MAX_SUBTASKS) -> str:
    """The planner's instructions with its limit (`Tuning.max_subtasks`, or the default)."""
    return (
        PLANNER_RULES
        + f" Never more than {limit} subtasks."
        + KIND_RULES
        + RUN_RULES
        + CRITERIA_RULES
    )


PLANNER_SYSTEM = planner_system()

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]


class PlannedSubtask(BaseModel):
    description: Text
    acceptance_criteria: Text
    depends_on: list[int] = Field(default_factory=list)
    kind: Kind | None = None
    """None: the model did not say (text for a planned subtask, the subtask's kind for a part)."""
    runs_code: bool = False


class Plan(BaseModel):
    """What the planner model must reply."""

    subtasks: list[PlannedSubtask] = Field(min_length=1, max_length=MAX_SUBTASKS)

    @model_validator(mode="after")
    def _is_a_valid_graph(self) -> Self:
        last = len(self.subtasks) - 1
        users = [i for i, sub in enumerate(self.subtasks) if last in sub.depends_on]
        if users:
            raise ValueError(
                f"the last subtask must produce the final result, but subtasks {users} depend on it"
            )
        try:
            self.spec("goal")
        except ValidationError as e:
            # The reason, without the input (a repair request lists messages, not values).
            reasons = (str(error.get("ctx", {}).get("error", error["msg"])) for error in e.errors())
            raise ValueError("; ".join(reasons)) from None
        return self

    def spec(self, goal: str, priority: int = Priority.NORMAL) -> TaskSpec:
        """The task graph. A subtask sees only the results of its own dependencies, so the last
        one is made to depend on every result nothing else uses: no work is left out of the
        final result."""
        last = len(self.subtasks) - 1
        used = {dep for sub in self.subtasks for dep in sub.depends_on}
        unused = [i for i in range(last) if i not in used]
        return TaskSpec(
            goal=goal,
            priority=priority,
            subtasks=[
                SubtaskSpec(
                    description=sub.description,
                    acceptance_criteria=sub.acceptance_criteria,
                    depends_on=sub.depends_on + unused if i == last else sub.depends_on,
                    kind=sub.kind or Kind.TEXT,
                    runs_code=sub.runs_code,
                )
                for i, sub in enumerate(self.subtasks)
            ],
        )


@functools.cache
def plan_schema(limit: int) -> type[Plan]:
    """`Plan` holding at most `limit` subtasks."""
    if limit == MAX_SUBTASKS:
        return Plan

    class Limited(Plan):
        subtasks: list[PlannedSubtask] = Field(min_length=1, max_length=limit)

    return Limited


@dataclass(frozen=True)
class Planned:
    spec: TaskSpec
    source: Literal["llm", "template"]
    error: str | None = None
    """Why the model's plan was not used (None when it was, or nobody asked the model)."""

    def details(self) -> dict[str, Any]:
        """For the audit entry of the new task."""
        details: dict[str, Any] = {"plan": self.source}
        if self.error is not None:
            details["plan_error"] = self.error
        return details


def template_plan(goal: str, priority: int = Priority.NORMAL) -> TaskSpec:
    """One subtask: the goal itself."""
    return TaskSpec(
        goal=goal,
        priority=priority,
        subtasks=[SubtaskSpec(description=goal, acceptance_criteria=TEMPLATE_CRITERIA)],
    )


def must_run(spec: TaskSpec) -> TaskSpec:
    """`spec` for work that must run code (the person said so, or the interlocutor did,
    independently of the planner): when no subtask is marked `runs_code`, the last one is,
    since the final result has to rest on a run. A template plan (the goal its only subtask)
    and a plan that overlooked it are held to it all the same."""
    if any(sub.runs_code for sub in spec.subtasks):
        return spec
    *rest, last = spec.subtasks
    return spec.model_copy(
        update={"subtasks": [*rest, last.model_copy(update={"runs_code": True})]}
    )


def plan(
    llm: LLM, goal: str, priority: int = Priority.NORMAL, *, limit: int | None = None
) -> Planned:
    """The model's plan for `goal`, or the template plan if the model fails. `limit`: the most
    subtasks (default `MAX_SUBTASKS`)."""
    limit = limit or MAX_SUBTASKS
    request = Request(
        messages=[
            Message(role="system", content=planner_system(limit)),
            Message(role="user", content=f"{PLAN}:\n{goal}"),
        ]
    )
    try:
        reply = complete_json(llm, request, plan_schema(limit), heading=PLAN)
    except StructuredOutputError as e:
        # The problem only: the reply itself may echo anything (it is in the call log).
        return Planned(
            template_plan(goal, priority), "template", f"{type(e).__name__}: {e.problem}"
        )
    except ProviderError as e:
        return Planned(template_plan(goal, priority), "template", f"{type(e).__name__}: {e}")
    return Planned(reply.spec(goal, priority), "llm")


# ── Splitting a subtask (ROADMAP 2.9b) ───────────────────────────────────────

SPLIT = "Split subtask"
"""Prompt heading: fake scripts route on it."""
MAX_PARTS = 4

SPLITTER_RULES = (
    "A subtask failed review several times. Split it into smaller parts that are easier to get "
    "right. Each part is done by one model call that sees only that part's description, its "
    "acceptance criteria, the subtask's inputs and the results of the parts it depends on, so "
    "every description must make sense on its own. Afterwards the subtask's result is built "
    "from the parts' results; the feedback says what went wrong so far. Reply with a single "
    'JSON object: {"subtasks": [{"description": "what to do", "acceptance_criteria": "how to '
    'tell it is done", "kind": "text" | "code" | "analysis", "depends_on": [indices of earlier '
    "parts, counting from 0]}]}."
)
"""The splitter's own instructions (a target of the Improvement Analyst, like the planner's)."""


def splitter_system(limit: int = MAX_PARTS) -> str:
    """The splitter's instructions with its limit (`Tuning.max_parts`, or the default)."""
    return (
        SPLITTER_RULES
        + f" Make at least 2 parts, at most {limit}."
        + KIND_RULES
        + RUN_RULES
        + CRITERIA_RULES
    )


SPLITTER_SYSTEM = splitter_system()


class Split(BaseModel):
    """What the splitting model must reply."""

    subtasks: list[PlannedSubtask] = Field(min_length=2, max_length=MAX_PARTS)

    @model_validator(mode="after")
    def _is_a_valid_graph(self) -> Self:
        try:
            self.parts()
        except ValidationError as e:
            reasons = (str(error.get("ctx", {}).get("error", error["msg"])) for error in e.errors())
            raise ValueError("; ".join(reasons)) from None
        return self

    def parts(self, kind: Kind = Kind.TEXT) -> list[SubtaskSpec]:
        """The parts; one the model did not tag is of `kind` (the split subtask's)."""
        spec = TaskSpec(
            goal="split",
            subtasks=[
                SubtaskSpec(
                    description=sub.description,
                    acceptance_criteria=sub.acceptance_criteria,
                    depends_on=sub.depends_on,
                    kind=sub.kind or kind,
                    runs_code=sub.runs_code,
                )
                for sub in self.subtasks
            ],
        )
        return spec.subtasks


@functools.cache
def split_schema(limit: int) -> type[Split]:
    """`Split` into at most `limit` parts."""
    if limit == MAX_PARTS:
        return Split

    class Limited(Split):
        subtasks: list[PlannedSubtask] = Field(min_length=2, max_length=limit)

    return Limited


def decomposer(llm: LLM, *, limit: int | None = None) -> Decomposer:
    """Splits a subtask the model's way; None (the coordinator retries normally) when the reply
    stays unusable. Provider errors propagate: the coordinator pauses or retries as usual.
    `limit`: the most parts (default `MAX_PARTS`)."""
    parts = limit or MAX_PARTS

    def split(work: Work) -> list[SubtaskSpec] | None:
        context = work.context
        payload = {
            "subtask": context.description,
            "acceptance_criteria": context.acceptance_criteria,
            "kind": work.subtask.kind.value,
            "inputs": context.inputs,
            "feedback_on_previous_attempt": context.feedback.text if context.feedback else None,
        }
        body = json.dumps(payload, ensure_ascii=False, indent=2)
        request = Request(
            messages=[
                Message(role="system", content=splitter_system(parts)),
                Message(role="user", content=f"{SPLIT}:\n{body}"),
            ]
        )
        try:
            schema = split_schema(parts)
            return complete_json(llm, request, schema, heading=SPLIT).parts(work.subtask.kind)
        except StructuredOutputError:
            return None

    return split
