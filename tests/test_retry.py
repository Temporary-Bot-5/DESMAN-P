"""Tests for the retry policy."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from desman.core.retry import Attempts, FailureKind, RetryPolicy, decide
from desman.core.states import SubtaskStatus

S = SubtaskStatus
K = FailureKind


def run(kinds: list[FailureKind], policy: RetryPolicy | None = None) -> list[SubtaskStatus]:
    attempts = Attempts()
    statuses = []
    for kind in kinds:
        decision = decide(kind, attempts, policy)
        attempts = decision.attempts
        statuses.append(decision.status)
    return statuses


def test_quality_failures_retry_then_escalate() -> None:
    assert run([K.QUALITY] * 3) == [S.READY, S.READY, S.NEEDS_HUMAN]


def test_single_attempt_policy_escalates_immediately() -> None:
    assert run([K.QUALITY], RetryPolicy(max_attempts=1)) == [S.NEEDS_HUMAN]


def test_transient_has_own_budget() -> None:
    policy = RetryPolicy(max_attempts=2, max_transient=3)
    # Interruptions do not eat into the quality budget and vice versa.
    assert run([K.TRANSIENT, K.TRANSIENT, K.QUALITY], policy) == [S.READY, S.READY, S.READY]
    assert run([K.TRANSIENT] * 3, policy) == [S.READY, S.READY, S.NEEDS_HUMAN]


def test_limits_pause_without_spending_attempts() -> None:
    attempts = Attempts(quality=2, transient=4)
    decision = decide(K.LIMIT, attempts)
    assert decision.status is S.PAUSED
    assert decision.attempts == attempts
    # However many times quotas run out, the subtask is never escalated for it.
    assert run([K.LIMIT] * 50) == [S.PAUSED] * 50


def test_permanent_failure_goes_to_human_at_once() -> None:
    decision = decide(K.PERMANENT, Attempts())
    assert decision.status is S.NEEDS_HUMAN
    assert decision.attempts == Attempts()


@pytest.mark.parametrize("kind", list(K))
def test_policy_never_gives_up_on_its_own(kind: FailureKind) -> None:
    assert S.FAILED not in run([kind] * 10)


def test_counters_and_reason() -> None:
    decision = decide(K.QUALITY, Attempts(quality=1))
    assert decision.attempts == Attempts(quality=2)
    assert "2/3" in decision.reason


def test_policy_validation() -> None:
    with pytest.raises(ValidationError):
        RetryPolicy(max_attempts=0)
