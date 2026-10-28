"""The planner: a goal becomes a graph of subtasks (D1).

The model is asked for a `Plan`: at most `MAX_SUBTASKS` subtasks, each with acceptance criteria
and the indices of the subtasks it needs. The plan is validated like any task spec (indices in
range, no cycles), so a broken graph gets the one repair request of `llm.structured`, with the
reason, like any other reply that does not fit.

Planning never stops a task from starting. When the model fails (provider error, a reply that
is still wrong after the repair), the template plan is used: one subtask, the goal itself. The
reason is shown to the operator and kept in the audit (`task_created`, `plan`).

The prompt starts with "Plan task", so a fake script can route on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, Field, StringConstraints, ValidationError, model_validator

from desman.core.models import Priority, SubtaskSpec, TaskSpec
from desman.llm.base import LLM, Message, ProviderError, Request
from desman.llm.structured import StructuredOutputError, complete_json

PLAN = "Plan task"
"""Prompt heading: fake scripts route on it."""
MAX_SUBTASKS = 8
"""A small model loses the thread of a long plan, and every subtask costs at least two calls."""
TEMPLATE_CRITERIA = "The result fully achieves the stated goal."

PLANNER_SYSTEM = (
    "You split a goal into subtasks for small language models. Each subtask is done by one "
    "model call that sees only that subtask's description, its acceptance criteria and the "
    "results of the subtasks it depends on, never the goal itself, so every description must "
    "make sense on its own. Use as few subtasks as the goal needs (one is fine for a simple "
    f"goal), at most {MAX_SUBTASKS}. Reply with a single JSON object: "
    '{"subtasks": [{"description": "what to do", "acceptance_criteria": "how to tell it is '
    'done", "depends_on": [indices of earlier subtasks, counting from 0]}]}. '
    "The last subtask must produce the final result of the goal."
)

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]


class PlannedSubtask(BaseModel):
    description: Text
    acceptance_criteria: Text
    depends_on: list[int] = Field(default_factory=list)


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
                )
                for i, sub in enumerate(self.subtasks)
            ],
        )


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


def plan(llm: LLM, goal: str, priority: int = Priority.NORMAL) -> Planned:
    """The model's plan for `goal`, or the template plan if the model fails."""
    request = Request(
        messages=[
            Message(role="system", content=PLANNER_SYSTEM),
            Message(role="user", content=f"{PLAN}:\n{goal}"),
        ]
    )
    try:
        reply = complete_json(llm, request, Plan, heading=PLAN)
    except StructuredOutputError as e:
        # The problem only: the reply itself may echo anything (it is in the call log).
        return Planned(
            template_plan(goal, priority), "template", f"{type(e).__name__}: {e.problem}"
        )
    except ProviderError as e:
        return Planned(template_plan(goal, priority), "template", f"{type(e).__name__}: {e}")
    return Planned(reply.spec(goal, priority), "llm")
