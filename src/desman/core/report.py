"""Read models for the operator: what `desman status`, `list`, `run` and `resume` print.

Everything is read in one snapshot, so a report never mixes states from before and after a
concurrent commit. Lease tokens are never included: holding one lets you act as the executor.
"""

from __future__ import annotations

import json
import sqlite3

from pydantic import BaseModel, ConfigDict, Field

from desman.core import db
from desman.core.retry import Attempts
from desman.core.states import SubtaskStatus, TaskStatus, derive_task_status
from desman.core.store import Store


class SubtaskReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    seq: int
    description: str
    status: SubtaskStatus
    attempts: Attempts
    depends_on: list[int]
    lease_owner: str | None
    lease_until: str | None
    paused_until: str | None
    reason: str | None
    """Why it reached its current status, from the audit log (e.g. the escalation reason)."""


class TaskReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    goal: str
    priority: int
    created_at: str
    status: TaskStatus
    done: int
    total: int
    subtasks: list[SubtaskReport] = Field(default_factory=list)
    """Filled by `task_report`; empty in `task_summaries`."""


def task_report(store: Store, task_id: str) -> TaskReport:
    """One task with all its subtasks. Raises NotFoundError for an unknown id."""
    with db.read_snapshot(store.conn):
        row = store.find_task(task_id)
        subtasks = store.list_subtasks(row["id"])
        seq_of = {s.id: s.seq for s in subtasks}
        reports = []
        for sub in subtasks:
            last = store.conn.execute(
                """SELECT details_json FROM audit_log WHERE subtask_id = ? AND to_status = ?
                   ORDER BY seq DESC LIMIT 1""",
                (sub.id, sub.status.value),
            ).fetchone()
            reason = json.loads(last[0]).get("reason") if last else None
            reports.append(
                SubtaskReport(
                    id=sub.id,
                    seq=sub.seq,
                    description=sub.description,
                    status=sub.status,
                    attempts=sub.attempts,
                    depends_on=sorted(seq_of[d] for d in store.dependencies(sub.id)),
                    lease_owner=sub.lease_owner,
                    lease_until=sub.lease_until,
                    paused_until=sub.paused_until,
                    reason=reason if isinstance(reason, str) else None,
                )
            )
        return _report(row, [s.status for s in subtasks], reports)


def task_summaries(
    store: Store, *, status: TaskStatus | None = None, limit: int = 20
) -> list[TaskReport]:
    """Tasks newest first, without their subtasks."""
    with db.read_snapshot(store.conn):
        statuses: dict[str, list[SubtaskStatus]] = {}
        for task_id, value in store.conn.execute("SELECT task_id, status FROM subtasks"):
            statuses.setdefault(task_id, []).append(SubtaskStatus(value))
        rows = store.conn.execute(
            "SELECT * FROM tasks ORDER BY created_at DESC, rowid DESC"
        ).fetchall()
    summaries = []
    for row in rows:
        report = _report(row, statuses.get(row["id"], []), [])
        if status is None or report.status is status:
            summaries.append(report)
            if len(summaries) >= limit:
                break
    return summaries


def _report(
    row: sqlite3.Row, statuses: list[SubtaskStatus], subtasks: list[SubtaskReport]
) -> TaskReport:
    return TaskReport(
        id=row["id"],
        goal=row["goal"],
        priority=row["priority"],
        created_at=row["created_at"],
        status=derive_task_status(statuses),
        done=sum(1 for s in statuses if s is SubtaskStatus.DONE),
        total=len(statuses),
        subtasks=subtasks,
    )
