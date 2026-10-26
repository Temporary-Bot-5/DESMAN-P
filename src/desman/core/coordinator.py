"""Coordinator: drives one task until the system has nothing left to do on its own.

The loop per tick:
    1. housekeeping: requeue expired leases, end pauses that are over
    2. review whatever waits for a verdict (no re-execution after a reviewer crash)
    3. claim the most urgent ready subtask, execute it, submit the result or record the failure

It stops as soon as the task is no longer ACTIVE (done, failed, cancelled or waiting for the
operator). Termination is guaranteed: every quality or transient failure spends a bounded budget
and ends in NEEDS_HUMAN, and blocked dependents are settled immediately. The one open-ended wait
is a provider pause; by default the coordinator returns instead of sleeping through it.

Executor and reviewer are plain callables, so the loop is tested with scripted fakes and later
driven by real LLM-backed agents.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from desman.core import db
from desman.core.context import DEFAULT_BUDGET_TOKENS, Context, build_context
from desman.core.models import Subtask
from desman.core.retry import FailureKind, RetryPolicy
from desman.core.states import Actor, SubtaskStatus, TaskStatus
from desman.core.store import DEFAULT_LEASE_SECONDS, StaleStateError, Store

S = SubtaskStatus


class Success(BaseModel):
    model_config = ConfigDict(frozen=True)

    content: dict[str, Any]


class Failure(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: FailureKind
    reason: str
    paused_until: str | None = None
    """For LIMIT: when the provider's quota resets. Any timezone-aware ISO-8601 timestamp."""

    @field_validator("paused_until", mode="before")
    @classmethod
    def _normalize_deadline(cls, value: object) -> object:
        # Provider data: reject garbage here rather than crash the loop later.
        if isinstance(value, datetime):
            return db.to_iso(value)
        if isinstance(value, str):
            return db.normalize_iso(value)
        return value


Outcome = Success | Failure


class Verdict(StrEnum):
    ACCEPT = "accept"
    RETRY = "retry"
    ESCALATE = "escalate"


class Review(BaseModel):
    model_config = ConfigDict(frozen=True)

    verdict: Verdict
    feedback: str = ""


class Work(BaseModel):
    """What an executor gets: the subtask, its context and a way to keep its lease alive."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    subtask: Subtask
    context: Context
    """Inputs, checkpoint, accepted results of dependencies (fitted to the budget), feedback on
    the previous attempt. Everything the executor may build on."""
    heartbeat: Callable[[dict[str, Any] | None], None]
    """Call during long work; pass a checkpoint to save progress. Raises StaleStateError if the
    lease was lost: stop and drop the work."""


Executor = Callable[[Work], Outcome]
Reviewer = Callable[[Subtask, dict[str, Any] | None], Review | Failure]


class RunResult(BaseModel):
    task_id: str
    status: TaskStatus
    ticks: int
    paused_until: str | None = None
    """Set when the run returned because only paused work was left."""
    events: list[str] = Field(default_factory=list)


class Coordinator:
    def __init__(
        self,
        store: Store,
        executor: Executor,
        reviewer: Reviewer,
        *,
        owner: str = "coordinator",
        policy: RetryPolicy | None = None,
        lease_seconds: int = DEFAULT_LEASE_SECONDS,
        context_budget_tokens: int = DEFAULT_BUDGET_TOKENS,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.store = store
        self.executor = executor
        self.reviewer = reviewer
        self.owner = owner
        self.policy = policy
        self.lease_seconds = lease_seconds
        self.context_budget_tokens = context_budget_tokens
        self.sleep = sleep

    def run(
        self, task_id: str, *, wait_for_pauses: bool = False, max_ticks: int = 10_000
    ) -> RunResult:
        """Work on `task_id` until it is no longer ACTIVE.

        With `wait_for_pauses` the coordinator sleeps until the earliest pause ends; otherwise it
        returns, and a later `run()` resumes (the state is all in the database). `max_ticks` is
        a safety net against bugs, not part of normal termination.
        """
        result = RunResult(task_id=task_id, status=self.store.task_status(task_id), ticks=0)
        while result.status is TaskStatus.ACTIVE:
            if result.ticks >= max_ticks:
                raise RuntimeError(f"task {task_id} still active after {max_ticks} ticks")
            result.ticks += 1
            progressed = self.tick(task_id, result.events)
            result.status = self.store.task_status(task_id)
            if progressed or result.status is not TaskStatus.ACTIVE:
                continue
            # Nothing could move. Either someone else's lease is still running (another worker,
            # or our own previous process that crashed) or only paused work is left.
            lease_until = self._earliest(task_id, "running", "lease_until")
            pause_until = self._earliest(task_id, "paused", "paused_until")
            if lease_until is not None and (pause_until is None or lease_until <= pause_until):
                self.sleep(self._seconds_until(lease_until) + 0.001)  # bounded by the lease
                continue
            if pause_until is None:
                raise RuntimeError(f"task {task_id} is active but nothing can move")  # a bug
            if not wait_for_pauses:
                result.paused_until = pause_until
                return result
            self.sleep(self._seconds_until(pause_until))
        return result

    def tick(self, task_id: str, events: list[str]) -> bool:
        """One pass; returns whether anything moved."""
        # Only this task: `policy` belongs to this run, other tasks may have their own.
        recovered, resumed = self.store.sweep(self.policy, task_id=task_id)
        events += [f"recovered {s.id}" for s in recovered]
        events += [f"resumed {s.id}" for s in resumed]
        progressed = bool(recovered or resumed)

        for subtask in self.store.pending_reviews(task_id=task_id):
            progressed |= self._review(subtask, events)

        claimed = self.store.claim(self.owner, task_id=task_id, lease_seconds=self.lease_seconds)
        if claimed is not None:
            self._execute(claimed, events)
            progressed = True
        return progressed

    def _execute(self, subtask: Subtask, events: list[str]) -> None:
        lease = subtask.lease_token
        assert lease is not None

        def heartbeat(checkpoint: dict[str, Any] | None = None) -> None:
            self.store.heartbeat(
                subtask.id, lease, lease_seconds=self.lease_seconds, checkpoint=checkpoint
            )

        outcome: Outcome
        try:
            context = build_context(
                subtask,
                self.store.dependency_results(subtask.id),
                self.store.last_feedback(subtask.id),
                self.context_budget_tokens,
            )
            returned = self.executor(Work(subtask=subtask, context=context, heartbeat=heartbeat))
            if not isinstance(returned, Success | Failure):
                raise TypeError(f"executor returned {type(returned).__name__}, not Success/Failure")
            outcome = returned
        except StaleStateError:
            events.append(f"lost lease {subtask.id}")
            return
        except Exception as e:  # an executor bug or crash: the attempt was interrupted
            outcome = Failure(
                kind=FailureKind.TRANSIENT, reason=f"executor raised {type(e).__name__}: {e}"
            )

        try:
            if isinstance(outcome, Success):
                self.store.submit(subtask.id, lease, outcome.content)
                events.append(f"submitted {subtask.id}")
            else:
                moved = self.store.record_failure(
                    subtask.id,
                    S.RUNNING,
                    Actor.EXECUTOR,
                    outcome.kind,
                    lease=lease,
                    reason=outcome.reason,
                    paused_until=outcome.paused_until,
                    policy=self.policy,
                )
                events.append(f"{outcome.kind} {subtask.id} -> {moved.status}")
        except StaleStateError:
            events.append(f"lost lease {subtask.id}")  # e.g. cancelled while running

    def _review(self, subtask: Subtask, events: list[str]) -> bool:
        review: Review | Failure
        try:
            returned = self.reviewer(subtask, self.store.latest_artifact(subtask.id))
            # Validate at the boundary: e.g. a verdict parsed from an LLM arrives as a string.
            review = returned if isinstance(returned, Failure) else Review.model_validate(returned)
        except (Exception, ValidationError) as e:
            review = Failure(
                kind=FailureKind.TRANSIENT, reason=f"reviewer raised {type(e).__name__}: {e}"
            )
        try:
            if isinstance(review, Failure):
                # Counts against the subtask's budget, so a reviewer that keeps failing cannot
                # loop forever. A transient one keeps the subtask in REVIEW: the submitted work is
                # reviewed again, not run again.
                moved = self.store.record_failure(
                    subtask.id,
                    S.REVIEW,
                    Actor.REVIEWER,
                    review.kind,
                    reason=review.reason,
                    paused_until=review.paused_until,
                    policy=self.policy,
                )
            elif review.verdict is Verdict.ACCEPT:
                moved = self.store.transition(
                    subtask.id, S.REVIEW, S.DONE, Actor.REVIEWER, reason=review.feedback or None
                )
            elif review.verdict is Verdict.RETRY:
                moved = self.store.record_failure(
                    subtask.id,
                    S.REVIEW,
                    Actor.REVIEWER,
                    FailureKind.QUALITY,
                    reason=review.feedback or "rejected by reviewer",
                    policy=self.policy,
                )
            elif review.verdict is Verdict.ESCALATE:
                moved = self.store.transition(
                    subtask.id,
                    S.REVIEW,
                    S.NEEDS_HUMAN,
                    Actor.REVIEWER,
                    reason=review.feedback or "escalated by reviewer",
                )
        except StaleStateError:
            return False  # another reviewer got there first
        events.append(f"reviewed {subtask.id} -> {moved.status}")
        return True

    def _earliest(self, task_id: str, status: str, column: str) -> str | None:
        row = self.store.conn.execute(
            f"SELECT MIN({column}) FROM subtasks WHERE task_id = ? AND status = ?",
            (task_id, status),
        ).fetchone()
        moment: str | None = row[0]
        return moment

    def _seconds_until(self, moment: str) -> float:
        target = db.parse_iso(moment)
        return max(0.0, (target - self.store.clock()).total_seconds())
