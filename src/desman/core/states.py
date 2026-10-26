"""Subtask state machine.

Every status change of a subtask goes through `transition()`, which checks the edge against
`TRANSITIONS` and the actor allowed to take it. The store applies the result with a
compare-and-swap (`UPDATE ... WHERE status = <from>`), so an illegal or stale change can never
reach the database.

Checkpoints, retry counters and lease deadlines are data on the subtask, not statuses: a crashed
subtask goes back to READY and keeps its checkpoint; "retry" is the RUNNING/REVIEW -> READY edge.

    happy path         PENDING -> READY -> RUNNING -> REVIEW -> DONE
    retry / crash      RUNNING, REVIEW -> READY
    give up            RUNNING, REVIEW -> FAILED -(human)-> READY
    escalate           RUNNING, REVIEW -> NEEDS_HUMAN -(human)-> READY | DONE | FAILED
    dependency failed  PENDING -> BLOCKED -(human)-> PENDING
    cancel             any non-terminal -(human)-> CANCELLED
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Final

__all__ = [
    "AWAITING_HUMAN",
    "TERMINAL",
    "TRANSITIONS",
    "Actor",
    "InvalidTransitionError",
    "SubtaskStatus",
    "allowed_targets",
    "can_transition",
    "transition",
]


class SubtaskStatus(StrEnum):
    """Lifecycle status of a subtask."""

    PENDING = "pending"  # created; waiting for dependencies
    READY = "ready"  # dependencies done; waiting to be claimed
    RUNNING = "running"  # claimed by an executor holding a lease
    REVIEW = "review"  # result submitted; waiting for the reviewer's verdict
    DONE = "done"  # accepted
    FAILED = "failed"  # retries exhausted or unrecoverable; only a human can revive it
    BLOCKED = "blocked"  # a dependency failed or was cancelled
    NEEDS_HUMAN = "needs_human"  # escalated; waits for an operator decision
    CANCELLED = "cancelled"  # stopped by a human


class Actor(StrEnum):
    """Who requests a transition."""

    SCHEDULER = "scheduler"  # dependency tracking, lease expiry, retry budget
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
    S.FAILED,
    S.BLOCKED,
    S.NEEDS_HUMAN,
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
    (S.RUNNING, S.READY): frozenset({A.EXECUTOR, A.SCHEDULER}),  # release / lease expired
    (S.RUNNING, S.FAILED): frozenset({A.EXECUTOR, A.SCHEDULER}),  # fatal / retries exhausted
    (S.RUNNING, S.NEEDS_HUMAN): frozenset({A.EXECUTOR}),
    # Review
    (S.REVIEW, S.DONE): frozenset({A.REVIEWER}),
    (S.REVIEW, S.READY): frozenset({A.REVIEWER}),  # retry with feedback
    (S.REVIEW, S.FAILED): frozenset({A.REVIEWER}),
    (S.REVIEW, S.NEEDS_HUMAN): frozenset({A.REVIEWER}),
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
