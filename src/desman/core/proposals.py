"""The Improvement Analyst's proposals as the database keeps them (ROADMAP 5.4), and the
failures an analysis reads.

An analysis is an ordinary task (`desman.improve` says what it asks and how its result becomes
a diff). Its proposal is a row from the moment the task is created: `analysing` while the task
runs, then settled once, when its result is read: `proposed` (with the diff), `nothing` (the
analyst found nothing worth changing) or `unusable` (the task ended without a result fit to
use, and `reason` says why). A settled proposal never changes (a trigger refuses it), so the
diff the person reads is the one that was proposed. Nothing here applies anything.

The task is marked in its `task_created` entry (`PURPOSE`), so its own failures are never
evidence for the next analysis, and a task created by a run that died before its row was
written is still found (`adopt`).
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
from enum import StrEnum
from typing import Any

from pydantic import BaseModel

from desman.core import db
from desman.core.models import TaskSpec
from desman.core.store import Store

PURPOSE = "improvement_analysis"
"""`purpose` in the `task_created` details of an analysis task."""

FAILURES = 30
"""The most failures one analysis reads (the newest)."""
PLAN_FALLBACKS = 10
"""The most plans that fell back to the template it reads."""
TEXT_CHARS = 400
"""What it reads of one text (a description, a reason): enough to see what went wrong."""


class Status(StrEnum):
    ANALYSING = "analysing"
    PROPOSED = "proposed"
    NOTHING = "nothing"
    UNUSABLE = "unusable"


class Proposal(BaseModel):
    id: int
    task_id: str
    created_at: str
    since: str | None
    """The start of the window the analysis read; None: all time."""
    status: Status
    settled_at: str | None
    summary: str | None
    """The analyst's own account: what it proposes and why (or why nothing)."""
    reason: str | None
    """Why the proposal is unusable."""
    details: dict[str, Any]
    """Proposed: `changes` (target, file, old and new value), `evidence`, `expected_effect`,
    `base` (each changed file's SHA-256 when the diff was made)."""
    diff: str | None


_PURPOSE_SQL = """SELECT task_id FROM audit_log
                  WHERE action = 'task_created' AND json_extract(details_json, '$.purpose') = ?"""


def start(store: Store, spec: TaskSpec, since: str | None) -> Proposal:
    """Create the analysis task for `spec` and its proposal, `analysing`."""
    task_id = store.create_task(spec, details={"purpose": PURPOSE, "since": since})
    return _insert(store.conn, task_id, since)


def adopt(conn: sqlite3.Connection) -> None:
    """A row for every analysis task that has none (its run died between the two writes)."""
    rows = conn.execute(
        """SELECT task_id, json_extract(details_json, '$.since') AS since FROM audit_log
           WHERE action = 'task_created' AND json_extract(details_json, '$.purpose') = ?
             AND task_id NOT IN (SELECT task_id FROM proposals)""",
        (PURPOSE,),
    ).fetchall()
    for row in rows:
        with contextlib.suppress(sqlite3.IntegrityError):  # adopted by another process
            _insert(conn, row["task_id"], row["since"])


def _insert(conn: sqlite3.Connection, task_id: str, since: str | None) -> Proposal:
    with db.transaction(conn):
        cursor = conn.execute(
            "INSERT INTO proposals (task_id, since) VALUES (?, ?)", (task_id, since)
        )
    found = get(conn, int(cursor.lastrowid or 0))
    assert found is not None
    return found


def settle(
    conn: sqlite3.Connection,
    proposal_id: int,
    status: Status,
    *,
    summary: str | None = None,
    reason: str | None = None,
    details: dict[str, Any] | None = None,
    diff: str | None = None,
) -> bool:
    """Settle an `analysing` proposal; False when someone else already did."""
    if status is Status.ANALYSING:
        raise ValueError("settling means leaving analysing")
    if (diff is not None) != (status is Status.PROPOSED):
        raise ValueError("a diff comes with a proposed change, and only with one")
    with db.transaction(conn):
        cursor = conn.execute(
            """UPDATE proposals SET status = ?, settled_at = ?, summary = ?, reason = ?,
                                    details_json = ?, diff = ?
               WHERE id = ? AND status = 'analysing'""",
            (
                status.value,
                db.now_iso(),
                summary,
                reason,
                json.dumps(details or {}, ensure_ascii=False),
                diff,
                proposal_id,
            ),
        )
    return cursor.rowcount == 1


def get(conn: sqlite3.Connection, proposal_id: int) -> Proposal | None:
    row = conn.execute("SELECT * FROM proposals WHERE id = ?", (proposal_id,)).fetchone()
    return None if row is None else _proposal(row)


def listing(conn: sqlite3.Connection, limit: int | None = None) -> list[Proposal]:
    """Newest first."""
    rows = conn.execute(
        "SELECT * FROM proposals ORDER BY id DESC LIMIT ?", (-1 if limit is None else limit,)
    ).fetchall()
    return [_proposal(r) for r in rows]


def analysing(conn: sqlite3.Connection) -> list[Proposal]:
    rows = conn.execute("SELECT * FROM proposals WHERE status = 'analysing' ORDER BY id").fetchall()
    return [_proposal(r) for r in rows]


def _proposal(row: sqlite3.Row) -> Proposal:
    return Proposal(
        id=row["id"],
        task_id=row["task_id"],
        created_at=row["created_at"],
        since=row["since"],
        status=Status(row["status"]),
        settled_at=row["settled_at"],
        summary=row["summary"],
        reason=row["reason"],
        details=json.loads(row["details_json"]),
        diff=row["diff"],
    )


# ── What an analysis reads ───────────────────────────────────────────────────


def failures(
    conn: sqlite3.Connection, since: str | None, limit: int = FAILURES
) -> list[dict[str, Any]]:
    """The newest failed attempts that say something about how the work is done: rejected by
    the reviewer or unusable (quality), impossible to retry (permanent), escalated by the
    reviewer. Interruptions and limits say nothing about prompts. Analysis tasks are left out:
    the analyst does not study itself."""
    where, args = ("a.ts >= ?", [since]) if since is not None else ("1", [])
    rows = conn.execute(
        f"""
        SELECT a.ts, a.actor, a.action, a.details_json,
               s.description, s.acceptance_criteria, s.kind, t.goal
        FROM audit_log a
        JOIN subtasks s ON s.id = a.subtask_id
        JOIN tasks t ON t.id = a.task_id
        WHERE {where}
          AND a.task_id NOT IN ({_PURPOSE_SQL})
          AND (json_extract(a.details_json, '$.failure') IN ('quality', 'permanent')
               OR (a.action = 'verdict'
                   AND json_extract(a.details_json, '$.verdict') = 'escalate'))
        ORDER BY a.seq DESC LIMIT ?
        """,
        [*args, PURPOSE, limit],
    ).fetchall()
    found = []
    for row in rows:
        details = json.loads(row["details_json"])
        said = details.get("cause") or details.get("feedback") or details.get("reason") or ""
        found.append(
            {
                "at": row["ts"],
                "goal": _cut(row["goal"]),
                "subtask": _cut(row["description"]),
                "acceptance_criteria": _cut(row["acceptance_criteria"]),
                "kind": row["kind"],
                "by": row["actor"],
                "what": details.get("verdict") or row["action"],
                "failure": details.get("failure"),
                "said": _cut(str(said)),
            }
        )
    return found


def plan_fallbacks(
    conn: sqlite3.Connection, since: str | None, limit: int = PLAN_FALLBACKS
) -> list[dict[str, Any]]:
    """The newest goals whose plan by the model failed, so the template was used instead."""
    where, args = ("ts >= ?", [since]) if since is not None else ("1", [])
    rows = conn.execute(
        f"""SELECT ts, details_json FROM audit_log
            WHERE action = 'task_created' AND {where}
              AND json_extract(details_json, '$.plan_error') IS NOT NULL
            ORDER BY seq DESC LIMIT ?""",
        [*args, limit],
    ).fetchall()
    found = []
    for row in rows:
        details = json.loads(row["details_json"])
        found.append(
            {
                "at": row["ts"],
                "goal": _cut(str(details.get("goal", ""))),
                "error": _cut(str(details["plan_error"])),
            }
        )
    return found


def _cut(text: str, limit: int = TEXT_CHARS) -> str:
    return text if len(text) <= limit else text[:limit] + " …(cut)"
