"""Read models for the operator: what `desman status`, `list`, `run` and `resume` print.

Everything is read in one snapshot, so a report never mixes states from before and after a
concurrent commit. Lease tokens are never included: holding one lets you act as the executor.
"""

from __future__ import annotations

import json
import sqlite3

from pydantic import BaseModel, ConfigDict, Field

from desman.core import db
from desman.core.models import Subtask
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
        by_id = {s.id: s for s in subtasks}
        reports = []
        for sub in subtasks:
            reason = _reason(store, sub, by_id)
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
                    reason=reason,
                )
            )
        return _report(row, [s.status for s in subtasks], reports)


def task_summaries(
    store: Store, *, status: TaskStatus | None = None, limit: int = 20
) -> list[TaskReport]:
    """Tasks newest first, without their subtasks.

    Read page by page: status is derived from subtasks, so a filter cannot be pushed into SQL,
    but without one only `limit` tasks are read.
    """
    page = limit if status is None else max(limit, 100)
    summaries: list[TaskReport] = []
    with db.read_snapshot(store.conn):
        offset = 0
        while len(summaries) < limit:
            rows = store.conn.execute(
                "SELECT * FROM tasks ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?",
                (page, offset),
            ).fetchall()
            if not rows:
                break
            offset += len(rows)
            ids = [row["id"] for row in rows]
            statuses: dict[str, list[SubtaskStatus]] = {}
            placeholders = ", ".join("?" for _ in ids)
            for task_id, value in store.conn.execute(
                f"SELECT task_id, status FROM subtasks WHERE task_id IN ({placeholders})", ids
            ):
                statuses.setdefault(task_id, []).append(SubtaskStatus(value))
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


class InboxItem(BaseModel):
    """A subtask only the operator can move on."""

    model_config = ConfigDict(frozen=True)

    ref: str
    """How to name it in `desman retry/accept/fail`: "<unique task prefix>/<seq>"."""
    version: int
    """`Store.status_version`: pass it as `--if-version` to act only on this exact situation."""
    task_id: str
    goal: str
    subtask_id: str
    seq: int
    description: str
    status: SubtaskStatus
    attempts: Attempts
    reason: str | None


INBOX_STATUSES = (SubtaskStatus.NEEDS_HUMAN, SubtaskStatus.FAILED)
"""BLOCKED is not listed: it clears by itself once what blocks it is resolved."""


def inbox(store: Store) -> list[InboxItem]:
    """Everything waiting for the operator, oldest task first."""
    items = []
    with db.read_snapshot(store.conn):
        rows = store.conn.execute(
            """SELECT s.task_id, t.goal FROM subtasks s JOIN tasks t ON t.id = s.task_id
               WHERE s.status IN (?, ?) GROUP BY s.task_id ORDER BY MIN(t.created_at)""",
            [s.value for s in INBOX_STATUSES],
        ).fetchall()
        for task_id, goal in rows:
            short = store.short_id(task_id)
            subtasks = store.list_subtasks(task_id)
            by_id = {s.id: s for s in subtasks}
            for sub in subtasks:
                if sub.status not in INBOX_STATUSES:
                    continue
                items.append(
                    InboxItem(
                        ref=f"{short}/{sub.seq}",
                        version=store.status_version(sub.id),
                        task_id=task_id,
                        goal=goal,
                        subtask_id=sub.id,
                        seq=sub.seq,
                        description=sub.description,
                        status=sub.status,
                        attempts=sub.attempts,
                        reason=_reason(store, sub, by_id),
                    )
                )
    return items


def _reason(store: Store, sub: Subtask, by_id: dict[str, Subtask]) -> str | None:
    """Why `sub` reached its current status, from the last audit entry that put it there."""
    last = store.conn.execute(
        """SELECT details_json FROM audit_log WHERE subtask_id = ? AND to_status = ?
           ORDER BY seq DESC LIMIT 1""",
        (sub.id, sub.status.value),
    ).fetchone()
    details = json.loads(last[0]) if last else {}
    if sub.status is SubtaskStatus.BLOCKED:
        root = by_id.get(details.get("cause", ""))
        if root is not None:  # blocking names the failed or cancelled root, not a reason
            return f"blocked by [{root.seq}] {root.description} ({root.status})"
    reason = details.get("reason")
    return reason if isinstance(reason, str) else None
