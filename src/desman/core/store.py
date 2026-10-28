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
from typing import Any, NoReturn

from desman.core import db
from desman.core.audit import TASK_HISTORY_SQL
from desman.core.context import DependencyResult, Feedback
from desman.core.models import Subtask, TaskSpec
from desman.core.retry import Attempts, FailureKind, RetryPolicy, decide
from desman.core.states import TERMINAL, Actor, SubtaskStatus, TaskStatus, derive_task_status
from desman.core.states import transition as check_transition

S = SubtaskStatus
Clock = Callable[[], datetime]

DEFAULT_LEASE_SECONDS = 300
DEFAULT_PAUSE_SECONDS = 60
"""Pause used when a provider hits its limits without saying when they reset."""


class StaleStateError(RuntimeError):
    """The subtask is no longer in the expected state (or no longer held by this executor)."""


class PauseNotOverError(StaleStateError):
    """A paused subtask cannot resume before its `paused_until`."""


class NotFoundError(LookupError):
    pass


class AmbiguousIdError(LookupError):
    pass


MIN_ID_PREFIX = 4


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

    def create_task(
        self, spec: TaskSpec, actor: Actor = Actor.HUMAN, details: dict[str, Any] | None = None
    ) -> str:
        """Store a task and its subtask graph; returns the task id. `details` go into the
        `task_created` audit entry (e.g. how the plan was made)."""
        task_id = _new_id()
        ids = [_new_id() for _ in spec.subtasks]
        with db.transaction(self.conn):
            now = self._now()  # after the lock: creation order matches commit order
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
                details={  # extra details never replace the task's own fields
                    **(details or {}),
                    "goal": spec.goal,
                    "priority": spec.priority,
                    "subtasks": ids,
                },
            )
            self._promote_ready()
        return task_id

    def find_task(self, prefix: str) -> sqlite3.Row:
        """The task whose id is `prefix` or starts with it (at least MIN_ID_PREFIX characters),
        as typed by an operator."""
        return self._find_by_prefix("tasks", "task", prefix)

    def find_subtask(self, ref: str) -> Subtask:
        """A subtask as an operator names it: "<task id or prefix>/<seq>" (as `desman status`
        numbers them) or its own id or unique prefix."""
        task_part, slash, seq = ref.rpartition("/")
        if slash:
            if not seq.isdigit():
                raise NotFoundError(f"subtask {ref!r}: expected <task>/<number>")
            task_id = self.find_task(task_part)["id"]
            row = self.conn.execute(
                "SELECT * FROM subtasks WHERE task_id = ? AND seq = ?", (task_id, int(seq))
            ).fetchone()
            if row is None:
                raise NotFoundError(f"task {task_id} has no subtask {seq}")
            return Subtask.from_row(row)
        return Subtask.from_row(self._find_by_prefix("subtasks", "subtask", ref))

    def short_id(self, task_id: str, minimum: int = 8) -> str:
        """The shortest prefix (at least `minimum` characters) naming only this task."""
        for length in range(min(minimum, len(task_id)), len(task_id)):
            prefix = task_id[:length]
            count = self.conn.execute(
                "SELECT COUNT(*) FROM tasks WHERE substr(id, 1, ?) = ?", (length, prefix)
            ).fetchone()[0]
            if count == 1:
                return prefix
        return task_id

    def _find_by_prefix(self, table: str, kind: str, prefix: str) -> sqlite3.Row:
        exact: sqlite3.Row | None = self.conn.execute(
            f"SELECT * FROM {table} WHERE id = ?", (prefix,)
        ).fetchone()
        if exact is not None:
            return exact
        if len(prefix) < MIN_ID_PREFIX:
            raise NotFoundError(
                f"{kind} {prefix!r} not found (give at least {MIN_ID_PREFIX} characters)"
            )
        rows = self.conn.execute(
            f"SELECT * FROM {table} WHERE substr(id, 1, ?) = ? ORDER BY created_at LIMIT 6",
            (len(prefix), prefix),
        ).fetchall()
        if not rows:
            raise NotFoundError(f"{kind} {prefix!r} not found")
        if len(rows) > 1:
            shown = ", ".join(r["id"] for r in rows[:5]) + (", …" if len(rows) > 5 else "")
            raise AmbiguousIdError(f"{kind} id {prefix!r} is ambiguous: {shown}")
        row: sqlite3.Row = rows[0]
        return row

    def task_status(self, task_id: str) -> TaskStatus:
        rows = self.conn.execute(
            "SELECT status FROM subtasks WHERE task_id = ?", (task_id,)
        ).fetchall()
        if not rows:
            raise NotFoundError(f"task {task_id} not found")
        return derive_task_status(SubtaskStatus(r[0]) for r in rows)

    def cancel_task(self, task_id: str, reason: str) -> list[Subtask]:
        """Operator stops a whole task: every subtask not finished yet becomes CANCELLED.

        A running executor finds out at its next heartbeat or submit (its lease is gone).
        """
        cancelled: list[Subtask] = []
        with db.transaction(self.conn):
            ids = [
                r[0]
                for r in self.conn.execute(
                    "SELECT id FROM subtasks WHERE task_id = ? ORDER BY seq", (task_id,)
                )
            ]
            if not ids:
                raise NotFoundError(f"task {task_id} not found")
            for subtask_id in ids:
                # Re-read each time: cancelling one subtask blocks its dependents.
                current = self.get_subtask(subtask_id)
                if current.status in TERMINAL:
                    continue
                cancelled.append(
                    self._apply(
                        subtask_id,
                        current.status,
                        S.CANCELLED,
                        Actor.HUMAN,
                        lease=None,
                        attempts=None,
                        reason=reason,
                        paused_until=None,
                        details={"scope": "task"},
                        action="cancel",
                    )
                )
        return cancelled

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

        Order: priority (high first), then age, then position in the task. The returned subtask
        carries a fresh `lease_token`; the executor must present it to leave RUNNING.
        """
        check_transition(S.READY, S.RUNNING, Actor.EXECUTOR)
        lease_token = _new_id()
        scope, params = ("AND task_id = ?", (task_id,)) if task_id else ("", ())
        with db.transaction(self.conn):
            # Times are taken after the write lock is held: waiting for it must not eat the lease.
            lease_until = db.to_iso(self.clock() + timedelta(seconds=lease_seconds))
            self._promote_ready()
            row = self.conn.execute(
                f"""UPDATE subtasks
                    SET status = 'running', lease_owner = ?, lease_token = ?, lease_until = ?,
                        updated_at = ?
                    WHERE id = (SELECT id FROM subtasks WHERE status = 'ready' {scope}
                                ORDER BY priority DESC, created_at, seq LIMIT 1)
                    RETURNING *""",
                (owner, lease_token, lease_until, self._now(), *params),
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
                details={"owner": owner, "lease": lease_token, "lease_until": lease_until},
            )
        return subtask

    def transition(
        self,
        subtask_id: str,
        src: SubtaskStatus,
        dst: SubtaskStatus,
        actor: Actor,
        *,
        lease: str | None = None,
        attempts: Attempts | None = None,
        reason: str | None = None,
        paused_until: str | None = None,
        details: dict[str, Any] | None = None,
        action: str = "transition",
        expected_version: int | None = None,
    ) -> Subtask:
        """Move a subtask from `src` to `dst` if it is still in `src`.

        - With `expected_version` (from `status_version()`), also only if its status has not
          changed at all since: a subtask that left `src` and came back is a different situation
          (e.g. a newer escalation), and a decision taken on the old one must not apply to it.
        - Leaving RUNNING as the executor requires `lease`, the token from `claim()`: an executor
          whose claim expired cannot touch a later claim, even one taken under the same name.
        - `attempts` replaces the failure counters in the same update; a human sending the
          subtask back to READY resets them (a fresh budget is the point of a restart).
        - The scheduler can only take RUNNING away from an executor whose lease has expired.
        - A pause without `paused_until` lasts DEFAULT_PAUSE_SECONDS. Resuming from PAUSED goes
          back where the subtask stopped and not before `paused_until`.
        - Lease and pause fields follow the status. Claiming (-> RUNNING) goes through `claim()`.
        """
        with db.transaction(self.conn):
            return self._apply(
                subtask_id,
                src,
                dst,
                actor,
                lease=lease,
                attempts=attempts,
                reason=reason,
                paused_until=paused_until,
                details=details,
                action=action,
                expected_version=expected_version,
            )

    def status_version(self, subtask_id: str) -> int:
        """Grows with every status change of the subtask (the audit seq of the latest one)."""
        row = self.conn.execute(
            """SELECT COALESCE(MAX(seq), 0) FROM audit_log
               WHERE subtask_id = ? AND to_status IS NOT NULL""",
            (subtask_id,),
        ).fetchone()
        version: int = row[0]
        return version

    def record_failure(
        self,
        subtask_id: str,
        src: SubtaskStatus,
        actor: Actor,
        kind: FailureKind,
        *,
        lease: str | None = None,
        reason: str | None = None,
        paused_until: str | None = None,
        policy: RetryPolicy | None = None,
        action: str = "failure",
        details: dict[str, Any] | None = None,
    ) -> Subtask:
        """Apply the retry policy to a failed attempt: retry, pause or escalate to the operator.

        Counters are read and written in one transaction, so concurrent failures cannot lose a count.
        A transient failure during REVIEW is the reviewer's, not the work's: the subtask stays in
        REVIEW with its artifact (only the counter grows), so the work is not run again.
        """
        with db.transaction(self.conn):
            current = self.get_subtask(subtask_id)
            decision = decide(kind, current.attempts, policy)
            if src is S.REVIEW and kind is FailureKind.TRANSIENT and decision.status is S.READY:
                return self._count_review_interruption(
                    current,
                    decision.attempts,
                    f"{reason}: {decision.reason}" if reason else decision.reason,
                    details,
                )
            return self._apply(
                subtask_id,
                src,
                decision.status,
                actor,
                lease=lease,
                attempts=decision.attempts,
                reason=f"{reason}: {decision.reason}" if reason else decision.reason,
                paused_until=paused_until,
                # `cause` is the raw reason (e.g. reviewer feedback), passed to the next attempt.
                details={
                    **(details or {}),
                    "failure": kind.value,
                    **({"cause": reason} if reason else {}),
                },
                action=action,
            )

    def _count_review_interruption(
        self,
        current: Subtask,
        attempts: Attempts,
        reason: str,
        details: dict[str, Any] | None = None,
    ) -> Subtask:
        """Bump the transient counter of a subtask that stays in REVIEW. Needs a transaction."""
        if current.status is not S.REVIEW:
            raise StaleStateError(
                f"subtask {current.id} is {current.status}, expected {S.REVIEW} (review interrupted)"
            )
        row = self.conn.execute(
            """UPDATE subtasks SET attempts_quality = ?, attempts_transient = ?, updated_at = ?
               WHERE id = ? AND status = 'review' RETURNING *""",
            (attempts.quality, attempts.transient, self._now(), current.id),
        ).fetchone()
        subtask = Subtask.from_row(row)
        self._audit(
            Actor.REVIEWER,
            "review_interrupted",
            subtask=subtask,
            details={
                **(details or {}),
                "reason": reason,
                "attempts": attempts.model_dump(),
                "failure": "transient",
            },
        )
        return subtask

    def submit(
        self, subtask_id: str, lease: str, content: dict[str, Any], *, kind: str = "result"
    ) -> Subtask:
        """Executor hands in its result: stores the artifact and moves RUNNING -> REVIEW atomically.

        If the lease is no longer valid nothing is stored.
        """
        with db.transaction(self.conn):
            attempt = (
                1
                + self.conn.execute(
                    "SELECT COUNT(*) FROM artifacts WHERE subtask_id = ?", (subtask_id,)
                ).fetchone()[0]
            )
            subtask = self._apply(
                subtask_id,
                S.RUNNING,
                S.REVIEW,
                Actor.EXECUTOR,
                lease=lease,
                attempts=None,
                reason=None,
                paused_until=None,
                details={"artifact_attempt": attempt},
                action="submit",
            )
            self.conn.execute(
                """INSERT INTO artifacts (id, subtask_id, attempt, kind, content_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    _new_id(),
                    subtask_id,
                    attempt,
                    kind,
                    json.dumps(content, ensure_ascii=False),
                    self._now(),
                ),
            )
        return subtask

    def artifacts(self, subtask_id: str) -> list[sqlite3.Row]:
        """All artifacts of a subtask, oldest attempt first."""
        return self.conn.execute(
            "SELECT * FROM artifacts WHERE subtask_id = ? ORDER BY attempt", (subtask_id,)
        ).fetchall()

    def latest_artifact(self, subtask_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT content_json FROM artifacts WHERE subtask_id = ? ORDER BY attempt DESC LIMIT 1",
            (subtask_id,),
        ).fetchone()
        content: dict[str, Any] | None = json.loads(row[0]) if row else None
        return content

    def dependency_results(self, subtask_id: str) -> list[DependencyResult]:
        """Accepted results of the subtask's direct dependencies, in task order.

        Only the artifact the DONE verdict accepted is passed on (`accepted_attempt`), never a
        rejected attempt.
        """
        rows = self.conn.execute(
            """SELECT dep.id, dep.description,
                      (SELECT a.content_json FROM artifacts a
                       WHERE a.subtask_id = dep.id AND a.attempt = dep.accepted_attempt)
                      AS content_json
               FROM subtask_deps d JOIN subtasks dep ON dep.id = d.depends_on
               WHERE d.subtask_id = ? AND dep.status = 'done'
               ORDER BY dep.seq""",
            (subtask_id,),
        ).fetchall()
        return [
            DependencyResult(
                subtask_id=r["id"],
                description=r["description"],
                content=json.loads(r["content_json"]) if r["content_json"] else None,
            )
            for r in rows
        ]

    def last_feedback(self, subtask_id: str) -> Feedback | None:
        """Why the subtask was last sent back to READY by the reviewer or the operator."""
        rows = self.conn.execute(
            """SELECT actor, details_json FROM audit_log
               WHERE subtask_id = ? AND to_status = 'ready' AND actor IN ('reviewer', 'human')
               ORDER BY seq DESC LIMIT 1""",
            (subtask_id,),
        ).fetchall()
        if not rows:
            return None
        details = json.loads(rows[0]["details_json"])
        text = details.get("cause") or details.get("reason")
        return Feedback(source=rows[0]["actor"], text=text) if text else None

    # ── Leases and recovery ──────────────────────────────────────────────────

    def heartbeat(
        self,
        subtask_id: str,
        lease: str,
        *,
        lease_seconds: int = DEFAULT_LEASE_SECONDS,
        checkpoint: dict[str, Any] | None = None,
    ) -> Subtask:
        """Extend the lease and, optionally, save a checkpoint in the same update.

        Works while the claim is still this executor's (token matches), even if the deadline has
        just passed and nobody has requeued the subtask yet. Once it was requeued or reclaimed,
        raises StaleStateError: the executor must stop and drop its work.
        """
        sets = ["lease_until = ?", "updated_at = ?"]
        if checkpoint is not None:
            sets.append("checkpoint_json = ?")
        with db.transaction(self.conn):
            # Deadline from after the write lock is acquired, or a slow lock wait could commit
            # an already expired lease that recover_expired() would immediately take away.
            lease_until = db.to_iso(self.clock() + timedelta(seconds=lease_seconds))
            params: list[Any] = [lease_until, self._now()]
            if checkpoint is not None:
                params.append(json.dumps(checkpoint, ensure_ascii=False))
            row = self.conn.execute(
                f"""UPDATE subtasks SET {", ".join(sets)}
                    WHERE id = ? AND status = 'running' AND lease_token = ? RETURNING *""",
                (*params, subtask_id, lease),
            ).fetchone()
            if row is None:
                self._raise_stale(subtask_id, S.RUNNING, S.RUNNING, lease, self._now())
            subtask = Subtask.from_row(row)
            if checkpoint is not None:
                # Plain heartbeats are not audited (too frequent). Checkpoints are, with their
                # content: the column only keeps the latest one, the audit log keeps them all.
                self._audit(
                    Actor.EXECUTOR,
                    "checkpoint",
                    subtask=subtask,
                    details={"owner": subtask.lease_owner, "checkpoint": checkpoint},
                )
        return subtask

    def recover_expired(
        self,
        policy: RetryPolicy | None = None,
        *,
        task_id: str | None = None,
        abandoned: Callable[[str], bool] | None = None,
    ) -> list[Subtask]:
        """Requeue RUNNING subtasks whose executor is gone: the lease ran out, or `abandoned`
        says its owner no longer exists (e.g. a process on this machine that was killed).

        Each recovery counts as a transient failure, so a subtask that keeps killing its executor
        ends up with the operator instead of looping forever. The checkpoint is kept, so the next
        executor resumes from it. `policy` is charged to every subtask recovered, so pass
        `task_id` when the policy belongs to one task's run. `abandoned` must only answer True
        when it is certain: a live executor that loses its lease has its work discarded.
        """
        scope, params = ("AND task_id = ?", (task_id,)) if task_id else ("", ())
        recovered: list[Subtask] = []
        with db.transaction(self.conn):
            now = self._now()
            rows = self.conn.execute(
                f"""SELECT * FROM subtasks WHERE status = 'running' {scope}
                    ORDER BY lease_until""",
                params,
            ).fetchall()
            for row in rows:
                stuck = Subtask.from_row(row)
                expired = stuck.lease_until is not None and stuck.lease_until < now
                gone = (
                    not expired
                    and abandoned is not None
                    and stuck.lease_owner is not None
                    and abandoned(stuck.lease_owner)
                )
                if not (expired or gone):
                    continue
                decision = decide(FailureKind.TRANSIENT, stuck.attempts, policy)
                why = (
                    f"lease expired at {stuck.lease_until}"
                    if expired
                    else f"lease owner {stuck.lease_owner} is gone"
                )
                recovered.append(
                    self._apply(
                        stuck.id,
                        S.RUNNING,
                        decision.status,
                        Actor.SCHEDULER,
                        # Taking a live lease: only this exact one (compare-and-swap on the token).
                        lease=stuck.lease_token if gone else None,
                        attempts=decision.attempts,
                        reason=f"{why}: {decision.reason}",
                        paused_until=None,
                        details={
                            "failure": FailureKind.TRANSIENT.value,
                            "lease": stuck.lease_token,
                        },
                        action="lease_expired" if expired else "lease_abandoned",
                        abandoned_lease=gone,
                    )
                )
        return recovered

    def resume_due(self, *, task_id: str | None = None) -> list[Subtask]:
        """Resume PAUSED subtasks whose pause is over, back to where each one stopped."""
        scope, params = ("AND task_id = ?", (task_id,)) if task_id else ("", ())
        resumed: list[Subtask] = []
        with db.transaction(self.conn):
            now = self._now()
            rows = self.conn.execute(
                f"""SELECT id, resume_status FROM subtasks
                    WHERE status = 'paused' AND (paused_until IS NULL OR paused_until <= ?) {scope}
                    ORDER BY paused_until""",
                (now, *params),
            ).fetchall()
            for row in rows:
                resumed.append(
                    self._apply(
                        row["id"],
                        S.PAUSED,
                        SubtaskStatus(row["resume_status"]),
                        Actor.SCHEDULER,
                        lease=None,
                        attempts=None,
                        reason="pause over",
                        paused_until=None,
                        details=None,
                        action="pause_over",
                    )
                )
        return resumed

    def pending_reviews(self, limit: int = 10, *, task_id: str | None = None) -> list[Subtask]:
        """Subtasks waiting for a verdict, most urgent first.

        Review needs no lease: it only reads the submitted result, and the verdict is a
        compare-and-swap from REVIEW. If a reviewer dies, the subtask simply stays here and is
        reviewed again without re-running the work.
        """
        scope, params = ("AND task_id = ?", (task_id,)) if task_id else ("", ())
        rows = self.conn.execute(
            f"""SELECT * FROM subtasks WHERE status = 'review' {scope}
                ORDER BY priority DESC, updated_at, seq LIMIT ?""",
            (*params, limit),
        ).fetchall()
        return [Subtask.from_row(r) for r in rows]

    def sweep(
        self,
        policy: RetryPolicy | None = None,
        *,
        task_id: str | None = None,
        abandoned: Callable[[str], bool] | None = None,
    ) -> tuple[list[Subtask], list[Subtask]]:
        """Scheduler housekeeping: (requeued or escalated expired leases, resumed pauses)."""
        recovered = self.recover_expired(policy, task_id=task_id, abandoned=abandoned)
        return recovered, self.resume_due(task_id=task_id)

    def _apply(
        self,
        subtask_id: str,
        src: SubtaskStatus,
        dst: SubtaskStatus,
        actor: Actor,
        *,
        lease: str | None = None,
        attempts: Attempts | None = None,
        reason: str | None = None,
        paused_until: str | None = None,
        details: dict[str, Any] | None = None,
        action: str = "transition",
        abandoned_lease: bool = False,
        expected_version: int | None = None,
    ) -> Subtask:
        """Compare-and-swap status change plus audit entry. Needs a transaction.

        `abandoned_lease` lets the scheduler take back the live lease `lease` because its owner
        is gone; only recover_expired sets it, after asking its `abandoned` predicate.

        Every status change except claiming goes through here, including the ones the store makes
        on its own (readiness, blocking, unblocking): one place checks the state machine, keeps
        lease/pause fields consistent and writes the audit entry.
        """
        if dst is S.RUNNING:
            raise ValueError("use claim() to start running a subtask")
        check_transition(src, dst, actor)
        if src is S.RUNNING and actor is Actor.EXECUTOR and lease is None:
            raise ValueError("executor must pass lease= (the token from claim()) to leave RUNNING")

        now = self._now()
        sets = ["status = ?", "updated_at = ?"]
        params: list[Any] = [dst.value, now]
        if src is S.RUNNING:
            sets += ["lease_owner = NULL", "lease_token = NULL", "lease_until = NULL"]
        if dst is S.PAUSED:
            if paused_until is None:
                paused_until = db.to_iso(self.clock() + timedelta(seconds=DEFAULT_PAUSE_SECONDS))
            else:
                paused_until = db.normalize_iso(paused_until)  # may come from a provider
            sets += ["resume_status = ?", "paused_until = ?"]
            params += [(S.READY if src is S.RUNNING else S.REVIEW).value, paused_until]
        if src is S.PAUSED:
            sets += ["resume_status = NULL", "paused_until = NULL"]
        if actor is Actor.HUMAN and dst is S.READY:
            attempts = Attempts()
        if attempts is not None:
            sets += ["attempts_quality = ?", "attempts_transient = ?"]
            params += [attempts.quality, attempts.transient]
        if dst is S.DONE:
            sets.append("accepted_attempt = ?")
            params.append(self._accepted_attempt(subtask_id, src))

        where = ["id = ?", "status = ?"]
        params += [subtask_id, src.value]
        if src is S.RUNNING and lease is not None:
            where.append("lease_token = ?")
            params.append(lease)
        if src is S.RUNNING and actor is Actor.SCHEDULER and not abandoned_lease:
            where.append("lease_until < ?")  # never take a live lease away
            params.append(now)
        if expected_version is not None:
            where.append(
                """NOT EXISTS (SELECT 1 FROM audit_log WHERE subtask_id = subtasks.id
                               AND to_status IS NOT NULL AND seq > ?)"""
            )
            params.append(expected_version)
        if src is S.PAUSED and actor is not Actor.HUMAN:
            # The scheduler resumes where it stopped, once the pause is over. The operator may
            # decide at any time (cancel a paused task, retry it now).
            where.append("resume_status = ?")
            where.append("(paused_until IS NULL OR paused_until <= ?)")
            params += [dst.value, now]

        before = self.conn.execute(
            "SELECT lease_owner FROM subtasks WHERE id = ?", (subtask_id,)
        ).fetchone()
        row = self.conn.execute(
            f"UPDATE subtasks SET {', '.join(sets)} WHERE {' AND '.join(where)} RETURNING *",
            params,
        ).fetchone()
        if row is None:
            self._raise_stale(subtask_id, src, dst, lease, now, expected_version, actor)
        subtask = Subtask.from_row(row)

        audit_details = dict(details or {})
        if reason:
            audit_details["reason"] = reason
        if src is S.RUNNING and before is not None:
            audit_details["owner"] = before["lease_owner"]
        if attempts is not None:
            audit_details["attempts"] = attempts.model_dump()
        self._audit(
            actor,
            action,
            subtask=subtask,
            from_status=src,
            to_status=dst,
            details=audit_details,
        )
        if dst in (S.FAILED, S.CANCELLED):
            self._block_dependents(subtask_id)
        if src is S.FAILED:
            self._unblock_dependents(actor)
        if dst is S.DONE or src is S.FAILED:
            self._promote_ready()
        return subtask

    def _accepted_attempt(self, subtask_id: str, src: SubtaskStatus) -> int | None:
        """The artifact a DONE verdict accepts.

        The reviewer accepts what it reviewed: the latest attempt. The operator resolving an
        escalation accepts the latest attempt only if it was under review when escalated; if the
        executor escalated (e.g. after earlier attempts were rejected), the latest artifact is a
        rejected one and nothing is accepted - the subtask was done by other means.
        """
        latest: int | None = self.conn.execute(
            "SELECT MAX(attempt) FROM artifacts WHERE subtask_id = ?", (subtask_id,)
        ).fetchone()[0]
        if src is S.REVIEW:
            return latest
        escalated_from = self.conn.execute(
            """SELECT from_status FROM audit_log
               WHERE subtask_id = ? AND to_status = 'needs_human' ORDER BY seq DESC LIMIT 1""",
            (subtask_id,),
        ).fetchone()
        return latest if escalated_from and escalated_from[0] == S.REVIEW.value else None

    def _raise_stale(
        self,
        subtask_id: str,
        src: SubtaskStatus,
        dst: SubtaskStatus,
        lease: str | None,
        now: str,
        expected_version: int | None = None,
        actor: Actor | None = None,
    ) -> NoReturn:
        current = self.get_subtask(subtask_id)
        if current.status is not src:
            raise StaleStateError(
                f"subtask {subtask_id} is {current.status}, expected {src} (wanted -> {dst})"
            )
        if expected_version is not None and self.status_version(subtask_id) > expected_version:
            raise StaleStateError(
                f"subtask {subtask_id} changed since version {expected_version} "
                f"(now {self.status_version(subtask_id)}), although it is {src} again"
            )
        if src is S.RUNNING and lease is not None and current.lease_token != lease:
            raise StaleStateError(
                f"subtask {subtask_id}: lease {lease} is no longer valid "
                f"(now held by {current.lease_owner})"
            )
        if src is S.RUNNING:
            raise StaleStateError(
                f"subtask {subtask_id} is leased to {current.lease_owner} "
                f"until {current.lease_until} (now {now})"
            )
        if src is S.PAUSED and actor is Actor.HUMAN:  # nothing else could have stopped it
            raise StaleStateError(f"subtask {subtask_id} changed meanwhile (wanted -> {dst})")
        if current.resume_status is not dst:
            raise StaleStateError(
                f"subtask {subtask_id} paused to resume as {current.resume_status}, not {dst}"
            )
        raise PauseNotOverError(
            f"subtask {subtask_id} is paused until {current.paused_until} (now {now})"
        )

    def _block_dependents(self, subtask_id: str) -> None:
        """PENDING -> BLOCKED for everything downstream of a failed or cancelled subtask.

        The audit names the root cause, not the immediate parent: that is what needs fixing.
        """
        frontier = [subtask_id]
        while frontier:
            placeholders = ", ".join("?" for _ in frontier)
            ids = [
                r[0]
                for r in self.conn.execute(
                    f"""SELECT DISTINCT s.id FROM subtasks s JOIN subtask_deps d ON d.subtask_id = s.id
                        WHERE s.status = 'pending' AND d.depends_on IN ({placeholders})""",
                    frontier,
                )
            ]
            for dependent in ids:
                self._apply(
                    dependent,
                    S.PENDING,
                    S.BLOCKED,
                    Actor.SCHEDULER,
                    details={"cause": subtask_id},
                    action="dependency_failed",
                )
            frontier = ids

    def _unblock_dependents(self, actor: Actor) -> None:
        """BLOCKED -> PENDING where no dependency is failed, cancelled or blocked any more.

        Runs as part of the operator reviving a failed subtask, hence the operator's authority.
        Repeats so that a whole blocked chain comes back, not just its first link.
        """
        while ids := [
            r[0]
            for r in self.conn.execute(
                """SELECT id FROM subtasks s WHERE status = 'blocked' AND NOT EXISTS (
                       SELECT 1 FROM subtask_deps d JOIN subtasks dep ON dep.id = d.depends_on
                       WHERE d.subtask_id = s.id
                         AND dep.status IN ('failed', 'cancelled', 'blocked'))"""
            )
        ]:
            for subtask_id in ids:
                self._apply(subtask_id, S.BLOCKED, S.PENDING, actor, action="dependency_revived")

    def _promote_ready(self) -> None:
        """PENDING -> READY for every subtask whose dependencies are all DONE. Needs a transaction."""
        ids = [
            r[0]
            for r in self.conn.execute(
                """SELECT id FROM subtasks s WHERE status = 'pending' AND NOT EXISTS (
                       SELECT 1 FROM subtask_deps d JOIN subtasks dep ON dep.id = d.depends_on
                       WHERE d.subtask_id = s.id AND dep.status != 'done')"""
            )
        ]
        for subtask_id in ids:
            self._apply(subtask_id, S.PENDING, S.READY, Actor.SCHEDULER, action="dependencies_met")

    # ── Audit ────────────────────────────────────────────────────────────────

    def audit_log(self, task_id: str) -> list[sqlite3.Row]:
        """Full history of a task and its subtasks, in order."""
        return self.conn.execute(TASK_HISTORY_SQL, {"task": task_id}).fetchall()

    def note(
        self, subtask_id: str, actor: Actor, action: str, details: dict[str, Any] | None = None
    ) -> None:
        """Record something that happened to a subtask without changing it (e.g. work that was
        thrown away because its lease was lost), so the history explains every outcome."""
        with db.transaction(self.conn):
            self._audit(actor, action, subtask=self.get_subtask(subtask_id), details=details)

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
