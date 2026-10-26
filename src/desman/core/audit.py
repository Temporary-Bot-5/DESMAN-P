"""Reading the audit log: a task's full history, and a check that the history is complete (D7).

The store writes an entry for every status change (all of them pass through `Store._apply` or
`claim`), every reviewer verdict, checkpoint, interrupted review and discarded piece of work.
The triggers of migration 3 make each entry name its task. `check_history` replays the log:
the subtasks are those `task_created` listed, each starting PENDING; each status change must
leave the status the previous one reached, and the last one must match the subtask's current
status. A gap means something was changed without being recorded, which is a bug. All reads
happen in one snapshot, so a worker committing meanwhile cannot cause a false alarm.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from pydantic import BaseModel, ConfigDict

from desman.core import db
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
    with db.read_snapshot(conn):
        entries = history(conn, task_id)
        rows = conn.execute("SELECT id, status FROM subtasks WHERE task_id = ?", (task_id,))
        current = {row["id"]: SubtaskStatus(row["status"]) for row in rows}
    problems = []
    created = next((e for e in entries if e.action == "task_created"), None)
    if created is None:
        problems.append("no task_created entry")
    recorded = created.details.get("subtasks") if created else None
    if isinstance(recorded, list):  # ids; entries written before 1.9 hold only a count
        for subtask_id in sorted(current.keys() - set(recorded)):
            problems.append(f"subtask {subtask_id} was never recorded as created")
        for subtask_id in sorted(set(recorded) - current.keys()):
            problems.append(f"subtask {subtask_id} was recorded as created but is missing")
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
