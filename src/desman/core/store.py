"""Task store: every read and write of task state.

Status changes are compare-and-swap updates (`... WHERE id = ? AND status = <expected>`), checked
by the state machine first and recorded in the audit log in the same transaction. If the row
moved on in the meantime (another worker, an expired lease) the change is refused with
StaleStateError instead of overwriting newer state.

Readiness is computed in SQL: a PENDING subtask becomes READY once all its dependencies are DONE,
and only READY subtasks can be claimed. A high-priority subtask waiting for its dependencies
therefore never hides work that is ready to run.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from desman.core import db
from desman.core.models import Subtask, TaskSpec
from desman.core.states import Actor, SubtaskStatus
from desman.core.states import transition as check_transition

S = SubtaskStatus
Clock = Callable[[], datetime]

DEFAULT_LEASE_SECONDS = 300


class StaleStateError(RuntimeError):
    """The subtask is no longer in the expected state (or no longer held by this executor)."""


class NotFoundError(LookupError):
    pass


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _new_id() -> str:
    return uuid.uuid4().hex


class Store:
    def __init__(self, conn: sqlite3.Connection, clock: Clock = _utcnow) -> None:
        self.conn = conn
        self.clock = clock

    def _now(self) -> str:
        return db.to_iso(self.clock())

    # ── Tasks ────────────────────────────────────────────────────────────────

    def create_task(self, spec: TaskSpec, actor: Actor = Actor.HUMAN) -> str:
        """Store a task and its subtask graph; returns the task id."""
        task_id = _new_id()
        now = self._now()
        ids = [_new_id() for _ in spec.subtasks]
        with db.transaction(self.conn):
            self.conn.execute(
                "INSERT INTO tasks (id, goal, priority, created_at) VALUES (?, ?, ?, ?)",
                (task_id, spec.goal, spec.priority, now),
            )
            for seq, (subtask_id, sub) in enumerate(zip(ids, spec.subtasks, strict=True)):
                self.conn.execute(
                    """INSERT INTO subtasks (id, task_id, seq, description, acceptance_criteria,
                                             inputs_json, priority, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        subtask_id,
                        task_id,
                        seq,
                        sub.description,
                        sub.acceptance_criteria,
                        json.dumps(sub.inputs, ensure_ascii=False),
                        spec.priority if sub.priority is None else sub.priority,
                        now,
                        now,
                    ),
                )
            for subtask_id, sub in zip(ids, spec.subtasks, strict=True):
                self.conn.executemany(
                    "INSERT INTO subtask_deps (subtask_id, depends_on) VALUES (?, ?)",
                    [(subtask_id, ids[dep]) for dep in sub.depends_on],
                )
            self._audit(
                actor,
                "task_created",
                task_id=task_id,
                details={"goal": spec.goal, "priority": spec.priority, "subtasks": len(ids)},
            )
            self._promote_ready()
        return task_id

    # ── Subtasks ─────────────────────────────────────────────────────────────

    def get_subtask(self, subtask_id: str) -> Subtask:
        row = self.conn.execute("SELECT * FROM subtasks WHERE id = ?", (subtask_id,)).fetchone()
        if row is None:
            raise NotFoundError(f"subtask {subtask_id} not found")
        return Subtask.from_row(row)

    def list_subtasks(self, task_id: str) -> list[Subtask]:
        rows = self.conn.execute(
            "SELECT * FROM subtasks WHERE task_id = ? ORDER BY seq", (task_id,)
        ).fetchall()
        return [Subtask.from_row(r) for r in rows]

    def dependencies(self, subtask_id: str) -> list[str]:
        rows = self.conn.execute(
            """SELECT d.depends_on FROM subtask_deps d JOIN subtasks s ON s.id = d.depends_on
               WHERE d.subtask_id = ? ORDER BY s.seq""",
            (subtask_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def claim(
        self,
        owner: str,
        *,
        task_id: str | None = None,
        lease_seconds: int = DEFAULT_LEASE_SECONDS,
    ) -> Subtask | None:
        """Atomically take the most urgent READY subtask; None if nothing is ready.

        Order: priority (high first), then age, then position in the task.
        """
        check_transition(S.READY, S.RUNNING, Actor.EXECUTOR)
        lease_until = db.to_iso(self.clock() + timedelta(seconds=lease_seconds))
        scope, params = ("AND task_id = ?", (task_id,)) if task_id else ("", ())
        with db.transaction(self.conn):
            self._promote_ready()
            row = self.conn.execute(
                f"""UPDATE subtasks
                    SET status = 'running', lease_owner = ?, lease_until = ?, updated_at = ?
                    WHERE id = (SELECT id FROM subtasks WHERE status = 'ready' {scope}
                                ORDER BY priority DESC, created_at, seq LIMIT 1)
                    RETURNING *""",
                (owner, lease_until, self._now(), *params),
            ).fetchone()
            if row is None:
                return None
            subtask = Subtask.from_row(row)
            self._audit(
                Actor.EXECUTOR,
                "claim",
                subtask=subtask,
                from_status=S.READY,
                to_status=S.RUNNING,
                details={"owner": owner, "lease_until": lease_until},
            )
        return subtask

    def transition(
        self,
        subtask_id: str,
        src: SubtaskStatus,
        dst: SubtaskStatus,
        actor: Actor,
        *,
        owner: str | None = None,
        reason: str | None = None,
        paused_until: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> Subtask:
        """Move a subtask from `src` to `dst` if it is still in `src`.

        Leaving RUNNING as the executor requires `owner` to match the lease holder, so an executor
        whose lease expired and was handed to someone else cannot overwrite the new holder's work.
        Lease and pause bookkeeping follows the status automatically. Claiming (-> RUNNING) goes
        through `claim()`.
        """
        if dst is S.RUNNING:
            raise ValueError("use claim() to start running a subtask")
        check_transition(src, dst, actor)
        if src is S.RUNNING and actor is Actor.EXECUTOR and owner is None:
            raise ValueError("executor must pass owner= to leave RUNNING")

        sets = ["status = ?", "updated_at = ?"]
        params: list[Any] = [dst.value, self._now()]
        if src is S.RUNNING:
            sets += ["lease_owner = NULL", "lease_until = NULL"]
        if dst is S.PAUSED:
            sets += ["resume_status = ?", "paused_until = ?"]
            params += [(S.READY if src is S.RUNNING else S.REVIEW).value, paused_until]
        if src is S.PAUSED:
            sets += ["resume_status = NULL", "paused_until = NULL"]

        where = ["id = ?", "status = ?"]
        params += [subtask_id, src.value]
        if src is S.RUNNING and owner is not None:
            where.append("lease_owner = ?")
            params.append(owner)
        if src is S.PAUSED:
            where.append("resume_status = ?")  # resume where it stopped
            params.append(dst.value)

        with db.transaction(self.conn):
            row = self.conn.execute(
                f"UPDATE subtasks SET {', '.join(sets)} WHERE {' AND '.join(where)} RETURNING *",
                params,
            ).fetchone()
            if row is None:
                raise StaleStateError(self._stale_reason(subtask_id, src, dst, owner))
            subtask = Subtask.from_row(row)
            audit_details = dict(details or {})
            if reason:
                audit_details["reason"] = reason
            if owner:
                audit_details["owner"] = owner
            self._audit(
                actor,
                "transition",
                subtask=subtask,
                from_status=src,
                to_status=dst,
                details=audit_details,
            )
            if dst is S.DONE:
                self._promote_ready()
        return subtask

    def _stale_reason(
        self, subtask_id: str, src: SubtaskStatus, dst: SubtaskStatus, owner: str | None
    ) -> str:
        current = self.get_subtask(subtask_id)
        if current.status is not src:
            return f"subtask {subtask_id} is {current.status}, expected {src} (wanted -> {dst})"
        if src is S.RUNNING and owner is not None and current.lease_owner != owner:
            return f"subtask {subtask_id} is leased to {current.lease_owner}, not {owner}"
        return f"subtask {subtask_id} paused to resume as {current.resume_status}, not {dst}"

    def _promote_ready(self) -> None:
        """PENDING -> READY for every subtask whose dependencies are all DONE. Needs a transaction."""
        rows = self.conn.execute(
            """UPDATE subtasks SET status = 'ready', updated_at = ?
               WHERE status = 'pending' AND NOT EXISTS (
                   SELECT 1 FROM subtask_deps d JOIN subtasks dep ON dep.id = d.depends_on
                   WHERE d.subtask_id = subtasks.id AND dep.status != 'done')
               RETURNING id, task_id""",
            (self._now(),),
        ).fetchall()
        for row in rows:
            check_transition(S.PENDING, S.READY, Actor.SCHEDULER)
            self._write_audit(
                Actor.SCHEDULER,
                "dependencies_met",
                task_id=row["task_id"],
                subtask_id=row["id"],
                from_status=S.PENDING,
                to_status=S.READY,
            )

    # ── Audit ────────────────────────────────────────────────────────────────

    def audit_log(self, task_id: str) -> list[sqlite3.Row]:
        """Full history of a task and its subtasks, in order."""
        return self.conn.execute(
            "SELECT * FROM audit_log WHERE task_id = ? ORDER BY seq", (task_id,)
        ).fetchall()

    def _audit(
        self,
        actor: Actor,
        action: str,
        *,
        task_id: str | None = None,
        subtask: Subtask | None = None,
        from_status: SubtaskStatus | None = None,
        to_status: SubtaskStatus | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self._write_audit(
            actor,
            action,
            task_id=subtask.task_id if subtask else task_id,
            subtask_id=subtask.id if subtask else None,
            from_status=from_status,
            to_status=to_status,
            details=details,
        )

    def _write_audit(
        self,
        actor: Actor,
        action: str,
        *,
        task_id: str | None,
        subtask_id: str | None,
        from_status: SubtaskStatus | None = None,
        to_status: SubtaskStatus | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.conn.execute(
            """INSERT INTO audit_log (ts, task_id, subtask_id, actor, action, from_status,
                                      to_status, details_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                self._now(),
                task_id,
                subtask_id,
                actor.value,
                action,
                from_status.value if from_status else None,
                to_status.value if to_status else None,
                json.dumps(details or {}, ensure_ascii=False),
            ),
        )
