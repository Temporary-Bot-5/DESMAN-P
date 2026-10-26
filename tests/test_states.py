"""Tests for the subtask state machine."""

from __future__ import annotations

from collections import deque

import pytest

from desman.core.states import (
    AWAITING_HUMAN,
    TERMINAL,
    TRANSITIONS,
    Actor,
    InvalidTransitionError,
    SubtaskStatus,
    allowed_targets,
    can_transition,
    transition,
)

S = SubtaskStatus
A = Actor


def walk(path: list[tuple[SubtaskStatus, Actor]], start: SubtaskStatus = S.PENDING) -> None:
    status = start
    for dst, actor in path:
        status = transition(status, dst, actor)
    assert status == path[-1][0]


def reachable(start: SubtaskStatus, actors: frozenset[Actor] = frozenset(A)) -> set[SubtaskStatus]:
    seen = {start}
    queue = deque([start])
    while queue:
        src = queue.popleft()
        for (s, dst), allowed in TRANSITIONS.items():
            if s == src and allowed & actors and dst not in seen:
                seen.add(dst)
                queue.append(dst)
    return seen


# ── Scenarios ────────────────────────────────────────────────────────────────


def test_happy_path() -> None:
    walk(
        [
            (S.READY, A.SCHEDULER),
            (S.RUNNING, A.EXECUTOR),
            (S.REVIEW, A.EXECUTOR),
            (S.DONE, A.REVIEWER),
        ]
    )


def test_review_rejection_retries() -> None:
    walk([(S.READY, A.REVIEWER), (S.RUNNING, A.EXECUTOR)], start=S.REVIEW)


def test_crashed_executor_is_requeued_by_scheduler() -> None:
    walk([(S.READY, A.SCHEDULER), (S.RUNNING, A.EXECUTOR)], start=S.RUNNING)


def test_failed_dependency_blocks_and_human_revives() -> None:
    walk([(S.BLOCKED, A.SCHEDULER)])
    walk([(S.READY, A.HUMAN)], start=S.FAILED)
    walk([(S.PENDING, A.HUMAN), (S.READY, A.SCHEDULER)], start=S.BLOCKED)


@pytest.mark.parametrize("dst", [S.READY, S.DONE, S.FAILED])
def test_human_resolves_escalation(dst: SubtaskStatus) -> None:
    walk([(S.NEEDS_HUMAN, A.REVIEWER), (dst, A.HUMAN)], start=S.REVIEW)


# ── Forbidden moves ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("src", "dst", "actor"),
    [
        (S.RUNNING, S.DONE, A.EXECUTOR),  # executor cannot accept its own work
        (S.REVIEW, S.DONE, A.EXECUTOR),
        (S.PENDING, S.RUNNING, A.EXECUTOR),  # cannot skip dependency check
        (S.READY, S.RUNNING, A.SCHEDULER),  # only executors claim
        (S.REVIEW, S.CANCELLED, A.REVIEWER),  # only humans cancel
        (S.NEEDS_HUMAN, S.READY, A.SCHEDULER),  # only humans resolve escalations
        (S.FAILED, S.READY, A.SCHEDULER),  # failures are not silently retried
        (S.BLOCKED, S.PENDING, A.SCHEDULER),  # nor silently unblocked
        (S.RUNNING, S.RUNNING, A.EXECUTOR),  # no self-transitions
    ],
)
def test_forbidden(src: SubtaskStatus, dst: SubtaskStatus, actor: Actor) -> None:
    assert not can_transition(src, dst, actor)
    with pytest.raises(InvalidTransitionError) as exc_info:
        transition(src, dst, actor)
    err = exc_info.value
    assert (err.src, err.dst, err.actor) == (src, dst, actor)
    assert str(src) in str(err) and str(dst) in str(err)


def test_error_message_names_allowed_actor() -> None:
    with pytest.raises(InvalidTransitionError, match="allowed for reviewer"):
        transition(S.REVIEW, S.DONE, A.EXECUTOR)
    with pytest.raises(InvalidTransitionError, match="no such transition"):
        transition(S.DONE, S.READY, A.HUMAN)


def test_invalid_transition_is_value_error() -> None:
    with pytest.raises(ValueError):
        transition(S.PENDING, S.DONE, A.HUMAN)


# ── Structural invariants ────────────────────────────────────────────────────


def test_terminal_statuses() -> None:
    assert frozenset({S.DONE, S.CANCELLED}) == TERMINAL
    for status in TERMINAL:
        for actor in A:
            assert allowed_targets(status, actor) == frozenset()


def test_no_self_transitions() -> None:
    assert all(src != dst for src, dst in TRANSITIONS)


def test_every_status_reachable_from_pending() -> None:
    assert reachable(S.PENDING) == set(S)


@pytest.mark.parametrize("status", sorted(frozenset(S) - TERMINAL))
def test_human_can_always_cancel(status: SubtaskStatus) -> None:
    assert can_transition(status, S.CANCELLED, A.HUMAN)


@pytest.mark.parametrize("status", sorted(frozenset(S) - TERMINAL))
def test_no_dead_ends(status: SubtaskStatus) -> None:
    # From any live status the work can still be finished, possibly with human help.
    assert S.DONE in reachable(status)


def test_system_alone_cannot_leave_awaiting_human() -> None:
    machine = frozenset({A.SCHEDULER, A.EXECUTOR, A.REVIEWER})
    for status in AWAITING_HUMAN:
        assert reachable(status, machine) == {status}


@pytest.mark.parametrize("status", sorted(frozenset(S) - TERMINAL - AWAITING_HUMAN))
def test_system_alone_can_finish_live_work(status: SubtaskStatus) -> None:
    machine = frozenset({A.SCHEDULER, A.EXECUTOR, A.REVIEWER})
    assert S.DONE in reachable(status, machine)


def test_only_reviewer_or_human_marks_done() -> None:
    actors = {
        actor for (_, dst), allowed in TRANSITIONS.items() if dst == S.DONE for actor in allowed
    }
    assert actors == {A.REVIEWER, A.HUMAN}


def test_allowed_targets() -> None:
    assert allowed_targets(S.RUNNING, A.EXECUTOR) == {S.REVIEW, S.READY, S.FAILED, S.NEEDS_HUMAN}
    assert allowed_targets(S.RUNNING, A.SCHEDULER) == {S.READY, S.FAILED}
    assert allowed_targets(S.PENDING, A.EXECUTOR) == frozenset()
