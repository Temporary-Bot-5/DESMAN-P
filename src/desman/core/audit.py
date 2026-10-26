"""Reading the audit log: a task's full history, and a check that the history is complete (D7).

The store writes an entry for every status change (all of them pass through `Store._apply` or
`claim`), every reviewer verdict, checkpoint, interrupted review and discarded piece of work.
The triggers of migration 3 make each entry name its task. `check_history` replays the log:
every subtask starts PENDING, each status change must leave the status the previous one
reached, and the last one must match the subtask's current status. A gap means a status was
changed without being recorded, which is a bug.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from pydantic import BaseModel, ConfigDict

from desman.core.states import Actor, SubtaskStatus


class AuditEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    seq: int
    ts: str
    task_id: str
    subtask_id: str | None
    actor: Actor
    action: str
    from_status: SubtaskStatus | None
    to_status: SubtaskStatus | None
    details: dict[str, Any]

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> AuditEntry:
        return cls(
            seq=row["seq"],
            ts=row["ts"],
            task_id=row["task_id"],
            subtask_id=row["subtask_id"],
            actor=Actor(row["actor"]),
            action=row["action"],
            from_status=row["from_status"],
            to_status=row["to_status"],
            details=json.loads(row["details_json"]),
        )


def history(conn: sqlite3.Connection, task_id: str) -> list[AuditEntry]:
    """Every entry of a task and its subtasks, in the order they were written."""
    rows = conn.execute("SELECT * FROM audit_log WHERE task_id = ? ORDER BY seq", (task_id,))
    return [AuditEntry.from_row(row) for row in rows]


def check_history(conn: sqlite3.Connection, task_id: str) -> list[str]:
    """Problems found replaying the task's history; empty when it accounts for every change."""
    entries = history(conn, task_id)
    current = {
        row["id"]: SubtaskStatus(row["status"])
        for row in conn.execute("SELECT id, status FROM subtasks WHERE task_id = ?", (task_id,))
    }
    problems = []
    if not any(e.action == "task_created" for e in entries):
        problems.append("no task_created entry")
    replayed = dict.fromkeys(current, SubtaskStatus.PENDING)
    for entry in entries:
        if entry.to_status is None:
            continue
        if entry.subtask_id not in replayed:
            problems.append(f"#{entry.seq}: status change of unknown subtask {entry.subtask_id}")
            continue
        before = replayed[entry.subtask_id]
        if entry.from_status is not before:
            problems.append(
                f"#{entry.seq}: {entry.subtask_id} left {entry.from_status} but history had it "
                f"{before} (an unrecorded change)"
            )
        replayed[entry.subtask_id] = entry.to_status
    for subtask_id, status in current.items():
        if replayed[subtask_id] is not status:
            problems.append(
                f"{subtask_id} is {status} but its history ends at {replayed[subtask_id]}"
            )
    return problems
