"""Contracts for tasks and subtasks: what callers submit and what the store returns."""

from __future__ import annotations

import json
import sqlite3
from enum import IntEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from desman.core.retry import Attempts
from desman.core.states import SubtaskStatus


class Priority(IntEnum):
    """Named priority levels; any integer works, higher runs first."""

    LOW = 0
    NORMAL = 50
    HIGH = 100
    CRITICAL = 200


class SubtaskSpec(BaseModel):
    description: str = Field(min_length=1)
    acceptance_criteria: str = Field(min_length=1)
    inputs: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[int] = Field(default_factory=list)
    """Indices of other subtasks in the same TaskSpec."""
    priority: int | None = None
    """Defaults to the task's priority."""


class TaskSpec(BaseModel):
    goal: str = Field(min_length=1)
    priority: int = Priority.NORMAL
    subtasks: list[SubtaskSpec] = Field(min_length=1)

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
        )
