"""Retry policy: what happens to a subtask after a failed attempt.

Failures are classified, because retrying only helps for some of them:

- QUALITY    the work was done badly (reviewer rejected it, output did not parse). Retry, each time
             differently (the attempt number picks the `Strategy`), up to `max_attempts`.
- TRANSIENT  the attempt was interrupted (network, 5xx, timeout, executor crashed). Retry; this
             says nothing about the subtask itself, so it has its own, larger budget.
- LIMIT      a provider's rate limit or quota is exhausted. Retrying now only burns resources:
             pause until the quota resets. No attempt is spent.
- PERMANENT  retrying cannot help (bad credentials, policy denial, invalid configuration).
             Hand over to the operator at once.

When a budget runs out the subtask goes to NEEDS_HUMAN, never to FAILED: giving up is the
operator's call. When the operator sends it back to READY the store resets the counters.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from desman.core.states import SubtaskStatus


class FailureKind(StrEnum):
    QUALITY = "quality"
    TRANSIENT = "transient"
    LIMIT = "limit"
    PERMANENT = "permanent"


class RetryPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_attempts: int = Field(default=4, ge=1)
    """Total attempts allowed before quality failures escalate to the operator: by default one
    of each `Strategy`."""
    max_transient: int = Field(default=5, ge=1)
    """Interruptions tolerated before escalating (a subtask that keeps crashing its executor)."""


class Strategy(StrEnum):
    """How an attempt differs from the one before it (ROADMAP 2.9). Repeating a rejected attempt
    the same way mostly repeats the mistake."""

    FIRST = "first"
    FEEDBACK = "feedback"  # the same model again, told what the reviewer found wrong
    REPHRASE = "rephrase"  # the instructions worded differently, at a higher temperature
    OTHER_MODEL = "other_model"  # another model or provider (with the feedback)


def strategy_for(attempt: int, *, other_model: bool = True) -> Strategy:
    """The strategy of attempt number `attempt` (1 is the first try). Later attempts keep using
    another model; splitting the subtask comes in ROADMAP 2.9b. Without `other_model` (nothing
    to switch to) those attempts are rephrased instead."""
    if attempt <= 1:
        return Strategy.FIRST
    if attempt == 2:
        return Strategy.FEEDBACK
    if attempt == 3 or not other_model:
        return Strategy.REPHRASE
    return Strategy.OTHER_MODEL


class Attempts(BaseModel):
    """Failure counters stored on the subtask."""

    model_config = ConfigDict(frozen=True)

    quality: int = Field(default=0, ge=0)
    transient: int = Field(default=0, ge=0)


class RetryDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: SubtaskStatus
    attempts: Attempts
    reason: str


def decide(
    kind: FailureKind, attempts: Attempts, policy: RetryPolicy | None = None
) -> RetryDecision:
    """Next status and updated counters after a failed attempt."""
    policy = policy or RetryPolicy()

    if kind is FailureKind.LIMIT:
        return RetryDecision(
            status=SubtaskStatus.PAUSED,
            attempts=attempts,
            reason="provider limits reached; paused until quota resets",
        )

    if kind is FailureKind.PERMANENT:
        return RetryDecision(
            status=SubtaskStatus.NEEDS_HUMAN,
            attempts=attempts,
            reason="failure cannot be fixed by retrying",
        )

    if kind is FailureKind.QUALITY:
        used = attempts.quality + 1
        updated = attempts.model_copy(update={"quality": used})
        budget = policy.max_attempts
    else:
        used = attempts.transient + 1
        updated = attempts.model_copy(update={"transient": used})
        budget = policy.max_transient

    if used >= budget:
        return RetryDecision(
            status=SubtaskStatus.NEEDS_HUMAN,
            attempts=updated,
            reason=f"{kind} failures: {used}/{budget}, retries exhausted",
        )
    return RetryDecision(
        status=SubtaskStatus.READY,
        attempts=updated,
        reason=f"{kind} failure {used}/{budget}, retrying",
    )
