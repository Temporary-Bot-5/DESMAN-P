"""Subtask state machine.

Every status change of a subtask goes through `transition()`, which checks the edge against
`TRANSITIONS` and the actor allowed to take it. The store applies the result with a
compare-and-swap (`UPDATE ... WHERE status = <from>`), so an illegal or stale change can never
reach the database.

Checkpoints, retry counters and lease deadlines are data on the subtask, not statuses: a crashed
subtask goes back to READY and keeps its checkpoint; "retry" is the RUNNING/REVIEW -> READY edge.
Which edge to take after a failure is decided by `desman.core.retry`.

The system works autonomously until it runs out of retries, then hands over to the operator.
It never gives up on its own: FAILED is a human verdict.

    happy path         PENDING -> READY -> RUNNING -> REVIEW -> DONE
    retry / crash      RUNNING, REVIEW -> READY
    out of quota       RUNNING, REVIEW -> PAUSED -> READY | REVIEW   (attempt not spent)
    escalate           RUNNING, REVIEW -> NEEDS_HUMAN -(human)-> READY | DONE | FAILED
    revive             FAILED -(human)-> READY
    dependency failed  PENDING -> BLOCKED -(human)-> PENDING
    cancel             any non-terminal -(human)-> CANCELLED
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import StrEnum
from typing import Final

__all__ = [
    "AWAITING_HUMAN",
    "TERMINAL",
    "TRANSITIONS",
    "Actor",
    "InvalidTransitionError",
    "SubtaskStatus",
    "TaskStatus",
    "allowed_targets",
    "can_transition",
    "derive_task_status",
    "transition",
]


class SubtaskStatus(StrEnum):
    """Lifecycle status of a subtask."""

    PENDING = "pending"  # created; waiting for dependencies
    READY = "ready"  # dependencies done; waiting to be claimed
    RUNNING = "running"  # claimed by an executor holding a lease
    REVIEW = "review"  # result submitted; waiting for the reviewer's verdict
    DONE = "done"  # accepted
    PAUSED = "paused"  # provider limits hit; resumes when the quota resets
    NEEDS_HUMAN = "needs_human"  # retries exhausted or unretryable; waits for the operator
    FAILED = "failed"  # the operator gave up on it; only the operator can revive it
    BLOCKED = "blocked"  # a dependency failed or was cancelled
    CANCELLED = "cancelled"  # stopped by a human


class Actor(StrEnum):
    """Who requests a transition."""

    SCHEDULER = "scheduler"  # dependency tracking, lease expiry, quota resets
    EXECUTOR = "executor"  # claims and performs the work
    REVIEWER = "reviewer"  # judges submitted results
    HUMAN = "human"  # the operator; final authority


S = SubtaskStatus
A = Actor

_NON_TERMINAL: Final = (
    S.PENDING,
    S.READY,
    S.RUNNING,
    S.REVIEW,
    S.PAUSED,
    S.NEEDS_HUMAN,
    S.FAILED,
    S.BLOCKED,
)

TRANSITIONS: Final[Mapping[tuple[SubtaskStatus, SubtaskStatus], frozenset[Actor]]] = {
    # Dependency tracking
    (S.PENDING, S.READY): frozenset({A.SCHEDULER}),
    (S.PENDING, S.BLOCKED): frozenset({A.SCHEDULER}),
    # Unblocking is part of the operator reviving the failed dependency (FAILED -> READY).
    (S.BLOCKED, S.PENDING): frozenset({A.HUMAN}),
    # Execution
    (S.READY, S.RUNNING): frozenset({A.EXECUTOR}),  # claim
    (S.RUNNING, S.REVIEW): frozenset({A.EXECUTOR}),  # submit result
    (S.RUNNING, S.READY): frozenset({A.EXECUTOR, A.SCHEDULER}),  # retry / lease expired
    (S.RUNNING, S.PAUSED): frozenset({A.EXECUTOR}),  # provider limits
    (S.RUNNING, S.NEEDS_HUMAN): frozenset({A.EXECUTOR, A.SCHEDULER}),  # out of retries
    # Review
    (S.REVIEW, S.DONE): frozenset({A.REVIEWER}),
    (S.REVIEW, S.READY): frozenset({A.REVIEWER}),  # retry with feedback
    (S.REVIEW, S.PAUSED): frozenset({A.REVIEWER}),  # provider limits
    (S.REVIEW, S.NEEDS_HUMAN): frozenset({A.REVIEWER}),  # out of retries
    # Quota reset: back to where the subtask was paused
    (S.PAUSED, S.READY): frozenset({A.SCHEDULER}),
    (S.PAUSED, S.REVIEW): frozenset({A.SCHEDULER}),
    # Operator decisions
    (S.NEEDS_HUMAN, S.READY): frozenset({A.HUMAN}),
    (S.NEEDS_HUMAN, S.DONE): frozenset({A.HUMAN}),
    (S.NEEDS_HUMAN, S.FAILED): frozenset({A.HUMAN}),
    (S.FAILED, S.READY): frozenset({A.HUMAN}),
    **{(status, S.CANCELLED): frozenset({A.HUMAN}) for status in _NON_TERMINAL},
}

TERMINAL: Final = frozenset(S) - {src for src, _ in TRANSITIONS}
"""Statuses with no way out: DONE and CANCELLED."""

AWAITING_HUMAN: Final = frozenset({S.FAILED, S.BLOCKED, S.NEEDS_HUMAN})
"""Statuses only a human can move a subtask out of (CANCELLED aside)."""


class InvalidTransitionError(ValueError):
    """Raised when a status change is not in `TRANSITIONS` or not allowed for the actor."""

    def __init__(self, src: SubtaskStatus, dst: SubtaskStatus, actor: Actor) -> None:
        self.src = src
        self.dst = dst
        self.actor = actor
        allowed = TRANSITIONS.get((src, dst))
        if allowed is None:
            reason = "no such transition"
        else:
            reason = f"allowed for {', '.join(sorted(allowed))}"
        super().__init__(f"{actor} cannot move subtask {src} -> {dst} ({reason})")


def can_transition(src: SubtaskStatus, dst: SubtaskStatus, actor: Actor) -> bool:
    """Whether `actor` may move a subtask from `src` to `dst`."""
    return actor in TRANSITIONS.get((src, dst), frozenset())


def transition(src: SubtaskStatus, dst: SubtaskStatus, actor: Actor) -> SubtaskStatus:
    """Validate a status change and return the new status, or raise InvalidTransitionError."""
    if not can_transition(src, dst, actor):
        raise InvalidTransitionError(src, dst, actor)
    return dst


def allowed_targets(src: SubtaskStatus, actor: Actor) -> frozenset[SubtaskStatus]:
    """Statuses `actor` may move a subtask to from `src`."""
    return frozenset(
        dst for (s, dst), actors in TRANSITIONS.items() if s == src and actor in actors
    )


# ── Task status ──────────────────────────────────────────────────────────────


class TaskStatus(StrEnum):
    """Status of a whole task, derived from its subtasks (it is never stored)."""

    ACTIVE = "active"  # the system can still make progress on its own
    NEEDS_HUMAN = "needs_human"  # nothing moves until the operator decides an escalation
    DONE = "done"  # every subtask accepted
    FAILED = "failed"  # the operator gave up on a subtask; the rest is blocked or finished
    CANCELLED = "cancelled"  # stopped by the operator


_ACTIVE: Final = frozenset({S.READY, S.RUNNING, S.REVIEW, S.PAUSED})


def derive_task_status(statuses: Iterable[SubtaskStatus]) -> TaskStatus:
    """Task status from its subtasks' statuses.

    PENDING alone does not count as progress: a pending subtask only moves when one of its
    dependencies does, so if nothing is ACTIVE the pending ones are waiting on the operator too.
    Anything but ACTIVE means the coordinator has nothing left to do for this task.
    """
    present = set(statuses)
    if present & _ACTIVE:
        return TaskStatus.ACTIVE
    if not present or present == {S.DONE}:
        return TaskStatus.DONE
    if S.NEEDS_HUMAN in present:
        return TaskStatus.NEEDS_HUMAN
    if S.FAILED in present:
        return TaskStatus.FAILED
    if present & {S.CANCELLED, S.BLOCKED}:
        return TaskStatus.CANCELLED
    # Only PENDING (and DONE) left with nothing active: impossible for a valid DAG, since a
    # pending subtask whose dependencies are all DONE is promoted to READY in the same transaction.
    raise ValueError(f"inconsistent subtask statuses: {sorted(present)}")
