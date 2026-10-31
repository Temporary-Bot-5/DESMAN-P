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
driven by real LLM-backed agents. An executor acts on nothing itself: it asks for tool calls
through `Work.tools` (core/tools.py), which checks, guards, audits and journals each one, so
that a rerun of an interrupted execution does not repeat them (core/effects.py).
"""

from __future__ import annotations

import functools
import json
import time
from collections.abc import Callable
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from desman.core import calls, db
from desman.core.context import DEFAULT_BUDGET_TOKENS, Context, build_context
from desman.core.effects import Effects
from desman.core.models import Subtask, SubtaskSpec
from desman.core.policy import Access
from desman.core.retry import FailureKind, RetryPolicy, Strategies, Strategy, strategy_for
from desman.core.states import Actor, SubtaskStatus, TaskStatus
from desman.core.store import DEFAULT_LEASE_SECONDS, StaleStateError, Store
from desman.core.tools import Registry, Toolbox
from desman.core.workspace import PathEscapeError, Workspace, workspace_root_for
from desman.llm.base import ProviderError

S = SubtaskStatus


class Success(BaseModel):
    model_config = ConfigDict(frozen=True)

    content: dict[str, Any]
    """Stored as the artifact: must be strict JSON (no NaN/Infinity, only JSON types)."""

    @field_validator("content")
    @classmethod
    def _strict_json(cls, content: dict[str, Any]) -> dict[str, Any]:
        try:
            json.dumps(content, allow_nan=False)
        except (TypeError, ValueError) as e:
            raise ValueError(f"result is not storable as JSON: {e}") from e
        return content


class Failure(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: FailureKind
    reason: str
    paused_until: str | None = None
    """For LIMIT: when the provider's quota resets. Any timezone-aware ISO-8601 timestamp."""
    unfit_provider: str | None = None
    """The provider whose model cannot answer this kind of request (`ProviderError.unfit_model`):
    the task's next attempts go to another model, if there is one."""

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

ONE_MODEL: Strategies = functools.partial(strategy_for, other_model=False)
"""The coordinator's default: no other model to switch to, so none is claimed in the audit."""

ERROR_TYPE = "error"
"""A result `{"type": "error", "reason": ...}` is the executor saying it could not do the work."""
REASON_EXCERPT = 300


def is_error_report(content: dict[str, Any] | None) -> bool:
    """An artifact that reports a failure instead of a result: never accepted (ROADMAP 2.8)."""
    return content is not None and content.get("type") == ERROR_TYPE


def _error_reason(content: dict[str, Any]) -> str:
    reason = content.get("reason") or content.get("message") or "no reason given"
    return f"executor reported an error: {str(reason)[:REASON_EXCERPT]}"


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
    tools: Toolbox
    """The only way to act: tool calls on the task's own workspace (and, once granted, more).
    A refused or mistaken call comes back as a failed ToolResult; every call is audited."""


Executor = Callable[[Work], Outcome]
Decomposer = Callable[[Work], list[SubtaskSpec] | None]
"""Splits the subtask of `Work` into smaller ones (ROADMAP 2.9b); None if it cannot. Provider
errors propagate like the executor's."""
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
        workspaces: Path | None = None,
        tools: Registry | None = None,
        access: Access | None = None,
        abandoned: Callable[[str], bool] | None = None,
        strategies: Strategies = ONE_MODEL,
        providers: tuple[str, str | None] | None = None,
        decomposer: Decomposer | None = None,
        sleep: Callable[[float], None] = time.sleep,
        watch: Callable[[], None] | None = None,
    ) -> None:
        self.store = store
        self.executor = executor
        self.reviewer = reviewer
        self.owner = owner
        self.policy = policy
        self.lease_seconds = lease_seconds
        self.context_budget_tokens = context_budget_tokens
        self.workspaces = workspaces if workspaces is not None else workspace_root_for(store.conn)
        self.tools = tools
        """The registry executors may call into; the built-in tools by default."""
        self.access = access
        """What the executor may use: its tool policy within the human's permissions. By
        default the shipped policy with every switch off (the workspace tools only)."""
        self.abandoned = abandoned
        """Tells whether a lease owner is certainly gone, so its work is taken back at once
        instead of after the lease expires (see core/owner.py)."""
        self.strategies = strategies
        """Attempt number -> strategy. The executor's owner says what it can do: by default
        there is no other model to switch to, so the audit never claims a switch that did not
        happen; pass `strategy_for` when the executor has one."""
        self.providers = providers
        """The names of the providers the executor asks: the main one and the other model of
        `strategies` (None if there is none). Without them no model is avoided as unfit."""
        self.decomposer = decomposer
        """Without one, subtasks are never split."""
        self.sleep = sleep
        self.watch = watch
        """Called whenever the task may look different (housekeeping, a claim, an execution, a
        review), so progress can be shown while `run()` works. It reads the state itself; the
        model calls in between are watched through `calls.watching`."""

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
        recovered, resumed = self.store.sweep(
            self.policy, task_id=task_id, abandoned=self.abandoned
        )
        events += [f"recovered {s.id}" for s in recovered]
        events += [f"resumed {s.id}" for s in resumed]
        progressed = bool(recovered or resumed)
        if progressed:
            self._changed()

        for subtask in self.store.pending_reviews(task_id=task_id):
            progressed |= self._review(subtask, events)
            self._changed()

        claimed = self.store.claim(self.owner, task_id=task_id, lease_seconds=self.lease_seconds)
        if claimed is not None:
            self._changed()
            self._execute(claimed, events)
            self._changed()
            progressed = True
        return progressed

    def _changed(self) -> None:
        if self.watch is not None:
            self.watch()

    def _for(self, role: calls.Role, subtask: Subtask) -> calls.Purpose:
        return calls.Purpose(role, task_id=subtask.task_id, subtask_id=subtask.id)

    def _execute(self, subtask: Subtask, events: list[str]) -> None:
        lease = subtask.lease_token
        assert lease is not None
        journal = Effects(self.store, subtask.id, lease)  # a rerun replays, never repeats

        def heartbeat(checkpoint: dict[str, Any] | None = None) -> None:
            self.store.heartbeat(
                subtask.id,
                lease,
                lease_seconds=self.lease_seconds,
                checkpoint=checkpoint,
                # A run resuming from the checkpoint skips the calls before it: so does the replay.
                effects=None if checkpoint is None else journal.position,
            )

        outcome: Outcome
        try:
            attempt = subtask.attempts.quality + 1
            avoid = self._avoid(subtask.task_id)
            if self.store.parts(subtask.id):
                strategy = Strategy.COMBINE  # its parts are done: their results are its inputs
            else:
                can_split = self.decomposer is not None and subtask.parent_id is None
                strategy = self.strategies(attempt, split=can_split, **avoid)
            context = build_context(
                subtask,
                self.store.dependency_results(subtask.id),
                self.store.last_feedback(subtask.id),
                self.context_budget_tokens,
                strategy,
            )
            self._started(subtask, context)
            work = Work(
                subtask=subtask,
                context=context,
                heartbeat=heartbeat,
                tools=self._toolbox(subtask, journal, heartbeat),
            )
            if strategy is Strategy.DECOMPOSE:
                if self._decompose(work, lease, events):
                    return
                # No usable split: try once more the way the attempt before did.
                context = context.model_copy(
                    update={
                        "strategy": self.strategies(attempt, split=False, **avoid)
                    }
                )
                work = work.model_copy(update={"context": context})
                self._started(subtask, context)
            with calls.for_(self._for(calls.Role.EXECUTOR, subtask)):
                returned = self.executor(work)
            if not isinstance(returned, Success | Failure):
                raise TypeError(f"executor returned {type(returned).__name__}, not Success/Failure")
            outcome = returned
        except StaleStateError:
            self._discard(subtask, "lease lost during execution", events)
            return
        except ProviderError as e:  # agents may let these propagate: the error knows its kind
            outcome = self._provider_failure(e)
        except PathEscapeError as e:
            # The attempt asked for a forbidden path: try again, differently.
            outcome = Failure(kind=FailureKind.QUALITY, reason=f"path refused: {e}")
        except Exception as e:  # an executor bug or crash: the attempt was interrupted
            outcome = Failure(
                kind=FailureKind.TRANSIENT, reason=f"executor raised {type(e).__name__}: {e}"
            )

        if isinstance(outcome, Success) and is_error_report(outcome.content):
            # Not a result: a failed attempt, never something a reviewer could accept.
            outcome = Failure(kind=FailureKind.QUALITY, reason=_error_reason(outcome.content))
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
                    action="execution_failed",
                    details=(
                        None
                        if outcome.unfit_provider is None
                        else {"unfit_provider": outcome.unfit_provider}
                    ),
                )
                events.append(f"{outcome.kind} {subtask.id} -> {moved.status}")
        except StaleStateError:  # e.g. cancelled while running
            self._discard(subtask, f"lease lost before {type(outcome).__name__.lower()}", events)

    def _toolbox(
        self,
        subtask: Subtask,
        journal: Effects,
        heartbeat: Callable[[dict[str, Any] | None], None],
    ) -> Toolbox:
        def record(action: str, details: dict[str, Any]) -> None:
            self.store.note(subtask.id, Actor.EXECUTOR, action, details)

        return Toolbox(
            Workspace(self.workspaces, subtask.task_id),
            self.tools,
            access=self.access,
            record=record,
            guard=lambda: heartbeat(None),  # a lost lease stops the work before any effect
            journal=journal,
        )

    def _started(self, subtask: Subtask, context: Context) -> None:
        self.store.note(
            subtask.id,
            Actor.EXECUTOR,
            "attempt_started",
            {"attempt": context.attempt, "strategy": context.strategy.value},
        )

    def _decompose(self, work: Work, lease: str, events: list[str]) -> bool:
        """Split the subtask; False (recorded) if the decomposer could not. A lost lease or a
        provider error propagates to `_execute`, as from the executor."""
        assert self.decomposer is not None
        subtask = work.subtask
        with calls.for_(self._for(calls.Role.DECOMPOSER, subtask)):
            parts = self.decomposer(work)
        if parts:
            try:
                created = self.store.decompose(subtask.id, lease, parts)
            except ValueError as e:  # not a valid graph of parts
                reason = str(e)
            else:
                events.append(f"split {subtask.id} into {len(created)}")
                return True
        else:
            reason = "the decomposer returned no parts"
        self.store.note(subtask.id, Actor.EXECUTOR, "decomposition_failed", {"reason": reason})
        return False

    def _review(self, subtask: Subtask, events: list[str]) -> bool:
        review: Review | Failure
        artifact = self.store.latest_artifact(subtask.id)
        try:
            with calls.for_(self._for(calls.Role.REVIEWER, subtask)):
                returned = self.reviewer(subtask, artifact)
            # Validate at the boundary: e.g. a verdict parsed from an LLM arrives as a string.
            review = returned if isinstance(returned, Failure) else Review.model_validate(returned)
        except ProviderError as e:
            review = self._provider_failure(e)
        except (Exception, ValidationError) as e:
            review = Failure(
                kind=FailureKind.TRANSIENT, reason=f"reviewer raised {type(e).__name__}: {e}"
            )
        if (
            isinstance(review, Review)
            and review.verdict is Verdict.ACCEPT
            and is_error_report(artifact)
        ):
            # Whatever the reviewer says: an error report is not a result.
            assert artifact is not None
            review = Review(
                verdict=Verdict.RETRY,
                feedback=f"{_error_reason(artifact)} (accepted by the reviewer, overruled)",
            )
        verdict: dict[str, Any] = {}
        if isinstance(review, Review):
            verdict = {"verdict": review.verdict.value, "feedback": review.feedback}
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
                    action="review_failed",
                )
            elif review.verdict is Verdict.ACCEPT:
                moved = self.store.transition(
                    subtask.id,
                    S.REVIEW,
                    S.DONE,
                    Actor.REVIEWER,
                    reason=review.feedback or None,
                    details=verdict,
                    action="verdict",
                )
            elif review.verdict is Verdict.RETRY:
                moved = self.store.record_failure(
                    subtask.id,
                    S.REVIEW,
                    Actor.REVIEWER,
                    FailureKind.QUALITY,
                    reason=review.feedback or "rejected by reviewer",
                    policy=self.policy,
                    details=verdict,
                    action="verdict",
                )
            elif review.verdict is Verdict.ESCALATE:
                moved = self.store.transition(
                    subtask.id,
                    S.REVIEW,
                    S.NEEDS_HUMAN,
                    Actor.REVIEWER,
                    reason=review.feedback or "escalated by reviewer",
                    details=verdict,
                    action="verdict",
                )
        except StaleStateError:
            return False  # another reviewer got there first
        events.append(f"reviewed {subtask.id} -> {moved.status}")
        return True

    def _discard(self, subtask: Subtask, why: str, events: list[str]) -> None:
        """The executor's work is thrown away: say so in the history, not just in `events`."""
        events.append(f"lost lease {subtask.id}")
        self.store.note(
            subtask.id, Actor.EXECUTOR, "work_discarded", {"reason": why, "owner": self.owner}
        )

    def _provider_failure(self, error: ProviderError) -> Failure:
        until = error.paused_until(self.store.clock())
        return Failure(
            kind=error.kind,
            reason=f"{type(error).__name__}: {error}",
            paused_until=None if until is None else db.to_iso(until),
            unfit_provider=error.provider if error.unfit_model else None,
        )

    def _avoid(self, task_id: str) -> dict[str, bool]:
        """Which of the executor's models an execution of the task found unfit (C.9): the
        other one is then not switched to; the main one is not asked while the other is fit."""
        if self.providers is None:
            return {}
        unfit = self.store.unfit_providers(task_id)
        main, other = self.providers
        avoid_other = other is not None and other in unfit
        return {"avoid_main": main in unfit and not avoid_other, "avoid_other": avoid_other}

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
