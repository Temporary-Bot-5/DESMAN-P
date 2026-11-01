"""Contracts for tasks and subtasks: what callers submit and what the store returns."""

from __future__ import annotations

import json
import sqlite3
from enum import IntEnum, StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from desman.core.retry import Attempts, RetryPolicy
from desman.core.states import SubtaskStatus


class Priority(IntEnum):
    """Named priority levels; any integer works, higher runs first."""

    LOW = 0
    NORMAL = 50
    HIGH = 100
    CRITICAL = 200


class Tools(StrEnum):
    """What a task's executor may use."""

    GRANTED = "granted"
    """The run's access: the tool policy within the human's permissions."""
    NONE = "none"
    """No tools at all, whatever the permissions: work that reads text which may say anything
    (a chat summary), so nothing written in it can make the work act."""


class Kind(StrEnum):
    """What sort of work a subtask is (ROADMAP 4.5): its executor's model is picked by it."""

    TEXT = "text"
    """Writing, answering, gathering facts: the default."""
    CODE = "code"
    """Writing, changing or explaining programs."""
    ANALYSIS = "analysis"
    """Reasoning over given material: comparing, judging, working something out."""


class SubtaskSpec(BaseModel):
    description: str = Field(min_length=1)
    acceptance_criteria: str = Field(min_length=1)
    inputs: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[int] = Field(default_factory=list)
    """Indices of other subtasks in the same TaskSpec."""
    priority: int | None = None
    """Defaults to the task's priority."""
    kind: Kind = Kind.TEXT


class TaskSpec(BaseModel):
    goal: str = Field(min_length=1)
    priority: int = Priority.NORMAL
    subtasks: list[SubtaskSpec] = Field(min_length=1)
    retry: RetryPolicy = Field(default_factory=RetryPolicy)
    """The task's retry budgets, kept with it: whoever recovers or judges its subtasks (any
    worker, any run) charges them by this policy."""
    tools: Tools = Tools.GRANTED

    @model_validator(mode="after")
    def _dependencies_form_a_dag(self) -> Self:
        count = len(self.subtasks)
        for i, sub in enumerate(self.subtasks):
            sub.depends_on = list(dict.fromkeys(sub.depends_on))  # a repeated edge is one edge
            for dep in sub.depends_on:
                if not 0 <= dep < count:
                    raise ValueError(f"subtask {i} depends on unknown subtask {dep}")
                if dep == i:
                    raise ValueError(f"subtask {i} depends on itself")
        # Kahn's algorithm: if some subtasks never become free of dependencies, there is a cycle.
        remaining = {i: set(sub.depends_on) for i, sub in enumerate(self.subtasks)}
        while free := [i for i, deps in remaining.items() if not deps]:
            for i in free:
                del remaining[i]
            for deps in remaining.values():
                deps.difference_update(free)
        if remaining:
            raise ValueError(f"dependency cycle among subtasks {sorted(remaining)}")
        return self


class Subtask(BaseModel):
    """A subtask as stored in the database."""

    model_config = ConfigDict(frozen=True)

    id: str
    task_id: str
    seq: int
    description: str
    acceptance_criteria: str
    inputs: dict[str, Any]
    priority: int
    status: SubtaskStatus
    lease_owner: str | None
    lease_token: str | None
    lease_until: str | None
    resume_status: SubtaskStatus | None
    paused_until: str | None
    checkpoint: dict[str, Any] | None
    attempts: Attempts
    accepted_attempt: int | None = None
    parent_id: str | None = None
    """Set on the parts of a subtask that was split (`Store.decompose`)."""
    review_owner: str | None = None
    review_until: str | None = None
    """A review in progress: who reserved it and until when (`Store.reserve_review`)."""
    queued_at: str | None = None
    """When it joined the queue it is in (READY or REVIEW); the scheduler ages it from then."""
    kind: Kind = Kind.TEXT

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Subtask:
        return cls(
            id=row["id"],
            task_id=row["task_id"],
            seq=row["seq"],
            description=row["description"],
            acceptance_criteria=row["acceptance_criteria"],
            inputs=json.loads(row["inputs_json"]),
            priority=row["priority"],
            status=SubtaskStatus(row["status"]),
            lease_owner=row["lease_owner"],
            lease_token=row["lease_token"],
            lease_until=row["lease_until"],
            resume_status=row["resume_status"],
            paused_until=row["paused_until"],
            checkpoint=json.loads(row["checkpoint_json"]) if row["checkpoint_json"] else None,
            attempts=Attempts(quality=row["attempts_quality"], transient=row["attempts_transient"]),
            accepted_attempt=row["accepted_attempt"],
            parent_id=row["parent_id"],
            review_owner=row["review_owner"],
            review_until=row["review_until"],
            queued_at=row["queued_at"],
            kind=Kind(row["kind"]),
        )
