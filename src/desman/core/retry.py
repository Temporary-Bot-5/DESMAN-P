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
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from desman.core.states import SubtaskStatus


class FailureKind(StrEnum):
    QUALITY = "quality"
    TRANSIENT = "transient"
    LIMIT = "limit"
    PERMANENT = "permanent"


class RetryPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_attempts: int = Field(default=5, ge=1)
    """Total attempts allowed before quality failures escalate to the operator: by default one
    of each `Strategy` up to the split; the attempt that combines the parts is the fifth too."""
    max_transient: int = Field(default=5, ge=1)
    """Interruptions tolerated before escalating (a subtask that keeps crashing its executor)."""


class Strategy(StrEnum):
    """How an attempt differs from the one before it (ROADMAP 2.9). Repeating a rejected attempt
    the same way mostly repeats the mistake."""

    FIRST = "first"
    FEEDBACK = "feedback"  # the same model again, told what the reviewer found wrong
    REPHRASE = "rephrase"  # the instructions worded differently, at a higher temperature
    OTHER_MODEL = "other_model"  # another model or provider (with the feedback)
    DECOMPOSE = "decompose"  # split into smaller subtasks instead of trying again (no execution)
    COMBINE = "combine"  # a split subtask, once its parts are done: the result from theirs


def strategy_for(
    attempt: int, *, other_model: bool = True, split: bool = True, avoid_main: bool = False
) -> Strategy:
    """The strategy of attempt number `attempt` (1 is the first try): first, feedback, rephrase,
    other model, then decompose. Without `other_model` (nothing to switch to) the fourth attempt
    is rephrased; when the subtask may not be split (`split`), later attempts repeat the fourth
    one. With `avoid_main` (the main model proved unable to answer this task, ROADMAP C.9) every
    attempt that would ask it goes to the other model instead, if there is one. COMBINE is not
    picked by number: it is what a split subtask does next."""
    if attempt >= 5 and split:
        return Strategy.DECOMPOSE
    if avoid_main and other_model:
        return Strategy.OTHER_MODEL
    if attempt <= 1:
        return Strategy.FIRST
    if attempt == 2:
        return Strategy.FEEDBACK
    if attempt == 3 or not other_model:
        return Strategy.REPHRASE
    return Strategy.OTHER_MODEL


class Strategies(Protocol):
    """Attempt number -> strategy, like `strategy_for`; `split` says whether this subtask may
    still be split and `avoid_main` whether the main model failed this task as unfit (the
    coordinator knows both, the caller who narrows the choice does not)."""

    def __call__(
        self, attempt: int, *, split: bool = True, avoid_main: bool = False
    ) -> Strategy: ...


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
