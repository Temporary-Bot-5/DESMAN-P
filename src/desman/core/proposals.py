"""The Improvement Analyst's proposals as the database keeps them (ROADMAP 5.4), and the
failures an analysis reads.

An analysis is an ordinary task (`desman.improve` says what it asks and how its result becomes
a diff). Its proposal is a row from the moment the task is created: `analysing` while the task
runs, then settled once, when its result is read: `proposed` (with the diff), `nothing` (the
analyst found nothing worth changing) or `unusable` (the task ended without a result fit to
use, and `reason` says why). A settled proposal never changes (a trigger refuses it), so the
diff the person reads is the one that was proposed. Applying it is the person's
(`desman.applying`); here it is only recorded.

The task is marked in its `task_created` entry (`PURPOSE`), so its own failures are never
evidence for the next analysis. The task and its row are written in one transaction, which
also makes sure only one analysis is under way at a time.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
from enum import StrEnum
from typing import Any

from pydantic import BaseModel

from desman.core import db, operator
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


class UnderWay(Exception):
    """An analysis is already under way: no second one is started."""

    def __init__(self, proposal_id: int) -> None:
        super().__init__(f"proposal {proposal_id} is still being analysed")
        self.proposal_id = proposal_id


def start(store: Store, spec: TaskSpec, since: str | None) -> Proposal:
    """Create the analysis task for `spec` and its proposal, `analysing`, in one transaction:
    raises `UnderWay` (and creates nothing) when another analysis is under way, so two
    processes never run two at once, and no task is ever without its row."""

    def record(conn: sqlite3.Connection, task_id: str) -> None:
        row = conn.execute("SELECT id FROM proposals WHERE status = 'analysing'").fetchone()
        if row is not None:
            raise UnderWay(row["id"])
        conn.execute("INSERT INTO proposals (task_id, since) VALUES (?, ?)", (task_id, since))

    task_id = store.create_task(spec, details={"purpose": PURPOSE, "since": since}, also=record)
    row = store.conn.execute("SELECT id FROM proposals WHERE task_id = ?", (task_id,)).fetchone()
    found = get(store.conn, row["id"])
    assert found is not None
    return found


def adopt(conn: sqlite3.Connection) -> None:
    """A row for every analysis task that has none (made before task and row were written in
    one transaction, by a development revision)."""
    rows = conn.execute(
        """SELECT task_id, json_extract(details_json, '$.since') AS since FROM audit_log
           WHERE action = 'task_created' AND json_extract(details_json, '$.purpose') = ?
             AND task_id NOT IN (SELECT task_id FROM proposals)""",
        (PURPOSE,),
    ).fetchall()
    for row in rows:
        with contextlib.suppress(sqlite3.IntegrityError):  # adopted by another process
            _insert(conn, row["task_id"], row["since"])


def _insert(conn: sqlite3.Connection, task_id: str, since: str | None) -> None:
    with db.transaction(conn):
        conn.execute("INSERT INTO proposals (task_id, since) VALUES (?, ?)", (task_id, since))


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


# ── Evaluations (ROADMAP 5.5) ────────────────────────────────────────────────


class Evaluation(BaseModel):
    """A proposal run on the benchmark (`desman.evaluation`), as it was recorded."""

    id: int
    proposal_id: int
    created_at: str
    mode: str
    """"fake" (the scripted model) or "real" (the person's models)."""
    commit: str
    """The commit both runs started from."""
    verdict: str
    """better, worse, same; failed: it could not be finished (`reason`)."""
    reason: str | None
    base: dict[str, Any] | None
    """The current version's run (`desman.bench.Result`); None if it did not report."""
    candidate: dict[str, Any] | None
    """The run with the proposal."""


def record_evaluation(
    conn: sqlite3.Connection,
    proposal_id: int,
    *,
    mode: str,
    commit: str,
    verdict: str,
    reason: str | None = None,
    base: dict[str, Any] | None = None,
    candidate: dict[str, Any] | None = None,
) -> Evaluation:
    with db.transaction(conn):
        cursor = conn.execute(
            """INSERT INTO evaluations (proposal_id, mode, git_commit, verdict, reason,
                                        base_json, candidate_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                proposal_id,
                mode,
                commit,
                verdict,
                reason,
                None if base is None else json.dumps(base, ensure_ascii=False),
                None if candidate is None else json.dumps(candidate, ensure_ascii=False),
            ),
        )
    found = evaluations(conn, proposal_id, evaluation_id=cursor.lastrowid)
    return found[0]


def evaluations(
    conn: sqlite3.Connection, proposal_id: int, *, evaluation_id: int | None = None
) -> list[Evaluation]:
    """The proposal's evaluations, newest first."""
    rows = conn.execute(
        """SELECT * FROM evaluations WHERE proposal_id = ? AND (? IS NULL OR id = ?)
           ORDER BY id DESC""",
        (proposal_id, evaluation_id, evaluation_id),
    ).fetchall()
    return [
        Evaluation(
            id=r["id"],
            proposal_id=r["proposal_id"],
            created_at=r["created_at"],
            mode=r["mode"],
            commit=r["git_commit"],
            verdict=r["verdict"],
            reason=r["reason"],
            base=None if r["base_json"] is None else json.loads(r["base_json"]),
            candidate=None if r["candidate_json"] is None else json.loads(r["candidate_json"]),
        )
        for r in rows
    ]


# ── Applying and reverting (ROADMAP 5.6) ─────────────────────────────────────


class Application(BaseModel):
    """A proposal applied or reverted by the person (`desman.applying`), as it was recorded."""

    id: int
    proposal_id: int
    created_at: str
    action: str
    """applied, or reverted (in turn, starting with applied)."""
    commit: str | None
    """The commit made; None when only settings changed."""
    base: str | None
    """The HEAD it was made on."""
    branch: str | None
    """The branch the checkout was on (None: a detached HEAD, or no commit)."""
    details: dict[str, Any]
    """`files` the commit changed; `settings` changed ({setting, old, new}); `left`: settings
    a revert left as they were, because the person had changed them since."""


def record_application(
    conn: sqlite3.Connection,
    proposal_id: int,
    action: str,
    *,
    commit: str | None,
    base: str | None,
    branch: str | None,
    details: dict[str, Any],
    session_id: str | None = None,
) -> Application:
    """Record what was done, with its entry in the operator's journal, in one transaction. The
    database refuses it unless the proposal is proposed and `action` comes in turn."""
    with db.transaction(conn):
        cursor = conn.execute(
            """INSERT INTO applications (proposal_id, action, git_commit, base_commit, branch,
                                         details_json)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                proposal_id,
                action,
                commit,
                base,
                branch,
                json.dumps(details, ensure_ascii=False),
            ),
        )
        operator.write(
            conn,
            f"operator_proposal_{action}",
            {"proposal": proposal_id, "commit": commit, "settings": details.get("settings", [])},
            session_id=session_id,
        )
    found = applications(conn, proposal_id)
    assert found and found[0].id == cursor.lastrowid
    return found[0]


def applications(conn: sqlite3.Connection, proposal_id: int) -> list[Application]:
    """What was done with the proposal, newest first."""
    rows = conn.execute(
        "SELECT * FROM applications WHERE proposal_id = ? ORDER BY id DESC", (proposal_id,)
    ).fetchall()
    return [
        Application(
            id=r["id"],
            proposal_id=r["proposal_id"],
            created_at=r["created_at"],
            action=r["action"],
            commit=r["git_commit"],
            base=r["base_commit"],
            branch=r["branch"],
            details=json.loads(r["details_json"]),
        )
        for r in rows
    ]


def last_application(conn: sqlite3.Connection, proposal_id: int) -> Application | None:
    found = applications(conn, proposal_id)
    return found[0] if found else None


class Update(BaseModel):
    """A release installed or reverted by the person."""

    id: int
    release: int
    created_at: str
    action: str
    commit: str
    base: str
    branch: str
    details: dict[str, Any]


def updates(conn: sqlite3.Connection) -> list[Update]:
    return [
        Update(
            id=r["id"],
            release=r["release"],
            created_at=r["created_at"],
            action=r["action"],
            commit=r["git_commit"],
            base=r["base_commit"],
            branch=r["branch"],
            details=json.loads(r["details_json"]),
        )
        for r in conn.execute(
            "SELECT * FROM applications WHERE release IS NOT NULL ORDER BY id DESC"
        )
    ]


def last_update(conn: sqlite3.Connection) -> Update | None:
    found = updates(conn)
    return found[0] if found else None


def record_update(
    conn: sqlite3.Connection,
    release: int,
    action: str,
    *,
    commit: str,
    base: str,
    branch: str,
    details: dict[str, Any],
    session_id: str | None = None,
) -> Update:
    from desman.core import netlog

    with db.transaction(conn):
        cursor = conn.execute(
            "INSERT INTO applications (release, action, git_commit, base_commit, branch, details_json)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (release, action, commit, base, branch, json.dumps(details, ensure_ascii=False)),
        )
        entry = {"release": release, "commit": commit}
        operator.write(conn, f"operator_update_{action}", entry, session_id=session_id)
        netlog.write(conn, f"update_{action}", entry)
    found = last_update(conn)
    assert found is not None and found.id == cursor.lastrowid
    return found
