"""Tests for the task store: task creation, readiness, atomic claim, compare-and-swap transitions."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from desman.core import db
from desman.core.models import Priority, Subtask, SubtaskSpec, TaskSpec
from desman.core.retry import Attempts, FailureKind, RetryPolicy
from desman.core.states import Actor, InvalidTransitionError, SubtaskStatus
from desman.core.store import PauseNotOverError, StaleStateError, Store

S = SubtaskStatus
A = Actor


class FakeClock:
    def __init__(self) -> None:
        self.now = datetime(2026, 1, 1, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def store(clock: FakeClock) -> Iterator[Store]:
    conn = db.open_db(":memory:")
    yield Store(conn, clock=clock)
    conn.close()


def sub(name: str, *deps: int, priority: int | None = None) -> SubtaskSpec:
    return SubtaskSpec(
        description=name, acceptance_criteria="ok", depends_on=list(deps), priority=priority
    )


def spec(*subtasks: SubtaskSpec, priority: int = Priority.NORMAL) -> TaskSpec:
    return TaskSpec(goal="goal", priority=priority, subtasks=list(subtasks))


def finish(store: Store, claimed: Subtask) -> None:
    store.transition(claimed.id, S.RUNNING, S.REVIEW, A.EXECUTOR, lease=claimed.lease_token)
    store.transition(claimed.id, S.REVIEW, S.DONE, A.REVIEWER)


def claim(store: Store, owner: str = "w1") -> Subtask:
    claimed = store.claim(owner)
    assert claimed is not None
    return claimed


# ── TaskSpec validation ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("subtasks", "message"),
    [
        ([sub("a", 1)], "unknown subtask"),
        ([sub("a", 0)], "depends on itself"),
        ([sub("a", 1), sub("b", 0)], "cycle"),
        ([sub("a"), sub("b", 2), sub("c", 1)], "cycle"),
        ([], "at least 1"),
    ],
)
def test_invalid_graphs_are_rejected(subtasks: list[SubtaskSpec], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        TaskSpec(goal="g", subtasks=subtasks)


def test_diamond_graph_is_valid() -> None:
    spec(sub("a"), sub("b", 0), sub("c", 0), sub("d", 1, 2))


# ── Task creation ────────────────────────────────────────────────────────────


def test_create_task_stores_graph(store: Store) -> None:
    task_id = store.create_task(spec(sub("a"), sub("b", 0), sub("c", 0, 1, priority=7)))
    a, b, c = store.list_subtasks(task_id)
    assert [x.description for x in (a, b, c)] == ["a", "b", "c"]
    assert store.dependencies(c.id) == [a.id, b.id]
    # Roots are ready at once; the rest wait.
    assert (a.status, b.status, c.status) == (S.READY, S.PENDING, S.PENDING)
    # Task priority is inherited unless the subtask sets its own.
    assert (a.priority, c.priority) == (Priority.NORMAL, 7)


def test_create_task_is_audited(store: Store) -> None:
    task_id = store.create_task(spec(sub("a")))
    actions = [(r["actor"], r["action"]) for r in store.audit_log(task_id)]
    assert actions == [("human", "task_created"), ("scheduler", "dependencies_met")]
    assert json.loads(store.audit_log(task_id)[0]["details_json"])["subtasks"] == 1


# ── Readiness and claim ──────────────────────────────────────────────────────


def test_claim_respects_dependencies(store: Store) -> None:
    task_id = store.create_task(spec(sub("a"), sub("b", 0)))
    a, b = store.list_subtasks(task_id)
    claimed = store.claim("w1")
    assert claimed is not None and claimed.id == a.id
    assert store.claim("w1") is None  # b waits for a
    finish(store, claimed)
    assert store.get_subtask(b.id).status is S.READY
    nxt = store.claim("w1")
    assert nxt is not None and nxt.id == b.id


def test_blocked_high_priority_does_not_starve_ready_work(store: Store) -> None:
    # The old prototype picked the highest-priority PENDING subtask, found its dependency
    # unfinished and gave up — forever. Only READY subtasks are candidates now.
    task_id = store.create_task(
        spec(sub("slow root", priority=Priority.LOW), sub("urgent", 0, priority=Priority.CRITICAL))
    )
    other_id = store.create_task(spec(sub("other")))
    first = store.claim("w1")
    second = store.claim("w2")
    assert first is not None and second is not None
    assert {first.task_id, second.task_id} == {task_id, other_id}
    assert store.claim("w3") is None


def test_claim_order_priority_then_age(store: Store, clock: FakeClock) -> None:
    old_low = store.create_task(spec(sub("old low"), priority=Priority.LOW))
    clock.advance(1)
    old_high = store.create_task(spec(sub("old high"), priority=Priority.HIGH))
    clock.advance(1)
    new_high = store.create_task(spec(sub("new high"), priority=Priority.HIGH))
    order = []
    while (claimed := store.claim("w")) is not None:
        order.append(claimed.task_id)
    assert order == [old_high, new_high, old_low]


def test_claim_scoped_to_task(store: Store) -> None:
    store.create_task(spec(sub("a")))
    mine = store.create_task(spec(sub("b"), priority=Priority.LOW))
    claimed = store.claim("w", task_id=mine)
    assert claimed is not None and claimed.task_id == mine


def test_claim_sets_lease_and_audits(store: Store, clock: FakeClock) -> None:
    task_id = store.create_task(spec(sub("a")))
    claimed = store.claim("w1", lease_seconds=60)
    assert claimed is not None
    assert claimed.status is S.RUNNING
    assert claimed.lease_owner == "w1"
    assert claimed.lease_until == db.to_iso(clock.now + timedelta(seconds=60))
    last = store.audit_log(task_id)[-1]
    assert (last["action"], last["from_status"], last["to_status"]) == ("claim", "ready", "running")


def test_concurrent_claims_never_hand_out_a_subtask_twice(tmp_path: Path) -> None:
    path = tmp_path / "d.db"
    setup = Store(db.open_db(path))
    for _ in range(4):
        setup.create_task(spec(*(sub(f"s{i}") for i in range(10))))
    setup.conn.close()

    claimed: list[str] = []
    lock = threading.Lock()

    def worker(name: str) -> None:
        store = Store(db.open_db(path))
        while (subtask := store.claim(name)) is not None:
            with lock:
                claimed.append(subtask.id)
        store.conn.close()

    threads = [threading.Thread(target=worker, args=(f"w{i}",)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(claimed) == 40
    assert len(set(claimed)) == 40


# ── Compare-and-swap transitions ─────────────────────────────────────────────


def test_transition_clears_lease_and_records_history(store: Store) -> None:
    task_id = store.create_task(spec(sub("a")))
    claimed = store.claim("w1")
    assert claimed is not None
    moved = store.transition(
        claimed.id, S.RUNNING, S.REVIEW, A.EXECUTOR, lease=claimed.lease_token, reason="submitted"
    )
    assert moved.status is S.REVIEW
    assert (moved.lease_owner, moved.lease_token, moved.lease_until) == (None, None, None)
    last = store.audit_log(task_id)[-1]
    assert (last["actor"], last["from_status"], last["to_status"]) == (
        "executor",
        "running",
        "review",
    )
    assert json.loads(last["details_json"]) == {"reason": "submitted", "owner": "w1"}
    assert claimed.lease_token is not None


def test_stale_transition_is_refused(store: Store) -> None:
    store.create_task(spec(sub("a")))
    claimed = store.claim("w1")
    assert claimed is not None
    store.transition(claimed.id, S.RUNNING, S.REVIEW, A.EXECUTOR, lease=claimed.lease_token)
    with pytest.raises(StaleStateError, match="is review, expected running"):
        store.transition(claimed.id, S.RUNNING, S.READY, A.SCHEDULER)


def test_only_lease_holder_can_leave_running(store: Store) -> None:
    store.create_task(spec(sub("a")))
    claimed = claim(store)
    with pytest.raises(StaleStateError, match="no longer valid"):
        store.transition(claimed.id, S.RUNNING, S.REVIEW, A.EXECUTOR, lease="forged")
    with pytest.raises(ValueError, match="lease="):
        store.transition(claimed.id, S.RUNNING, S.REVIEW, A.EXECUTOR)
    assert store.get_subtask(claimed.id).status is S.RUNNING


def test_stale_claim_under_same_name_is_rejected(store: Store) -> None:
    # Executor "w1" stalls, its subtask is requeued and claimed again by a restarted "w1".
    # The stalled process must not be able to submit into the new claim.
    store.create_task(spec(sub("a")))
    old = claim(store, "w1")
    store.transition(old.id, S.RUNNING, S.READY, A.SCHEDULER, reason="lease expired")
    new = claim(store, "w1")
    assert new.id == old.id and new.lease_token != old.lease_token
    with pytest.raises(StaleStateError, match="no longer valid"):
        store.transition(old.id, S.RUNNING, S.REVIEW, A.EXECUTOR, lease=old.lease_token)
    store.transition(new.id, S.RUNNING, S.REVIEW, A.EXECUTOR, lease=new.lease_token)


def test_invalid_transition_never_reaches_database(store: Store) -> None:
    task_id = store.create_task(spec(sub("a")))
    claimed = store.claim("w1")
    assert claimed is not None
    before = len(store.audit_log(task_id))
    with pytest.raises(InvalidTransitionError):
        store.transition(claimed.id, S.RUNNING, S.DONE, A.EXECUTOR, lease=claimed.lease_token)
    with pytest.raises(ValueError, match="claim"):
        store.transition(claimed.id, S.READY, S.RUNNING, A.EXECUTOR)
    assert store.get_subtask(claimed.id).status is S.RUNNING
    assert len(store.audit_log(task_id)) == before


@pytest.mark.parametrize(
    ("src", "actor", "resume"),
    [(S.RUNNING, A.EXECUTOR, S.READY), (S.REVIEW, A.REVIEWER, S.REVIEW)],
)
def test_pause_resumes_where_it_stopped(
    store: Store, clock: FakeClock, src: SubtaskStatus, actor: Actor, resume: SubtaskStatus
) -> None:
    store.create_task(spec(sub("a")))
    claimed = claim(store)
    if src is S.REVIEW:
        store.transition(claimed.id, S.RUNNING, S.REVIEW, A.EXECUTOR, lease=claimed.lease_token)
    until = db.to_iso(clock.now + timedelta(hours=1))
    paused = store.transition(
        claimed.id, src, S.PAUSED, actor, lease=claimed.lease_token, paused_until=until
    )
    assert (paused.resume_status, paused.paused_until) == (resume, until)
    wrong = S.REVIEW if resume is S.READY else S.READY
    with pytest.raises(StaleStateError, match=f"resume as {resume}"):
        store.transition(claimed.id, S.PAUSED, wrong, A.SCHEDULER)
    clock.advance(3600)
    resumed = store.transition(claimed.id, S.PAUSED, resume, A.SCHEDULER)
    assert (resumed.status, resumed.resume_status, resumed.paused_until) == (resume, None, None)


def test_done_unlocks_dependents_across_diamond(store: Store) -> None:
    task_id = store.create_task(spec(sub("a"), sub("b", 0), sub("c", 0), sub("d", 1, 2)))
    a, b, c, d = store.list_subtasks(task_id)
    for expected in ([a.id], [b.id, c.id], [d.id]):
        got = []
        while (claimed := store.claim("w")) is not None:
            got.append(claimed)
        assert sorted(x.id for x in got) == sorted(expected)
        for claimed in got:
            finish(store, claimed)
    assert all(s.status is S.DONE for s in store.list_subtasks(task_id))


def test_duplicate_dependency_is_one_edge(store: Store) -> None:
    task_id = store.create_task(spec(sub("a"), sub("b", 0, 0)))
    a, b = store.list_subtasks(task_id)
    assert store.dependencies(b.id) == [a.id]


def test_pause_cannot_end_early(store: Store, clock: FakeClock) -> None:
    store.create_task(spec(sub("a")))
    claimed = claim(store)
    until = db.to_iso(clock.now + timedelta(minutes=30))
    store.transition(
        claimed.id, S.RUNNING, S.PAUSED, A.EXECUTOR, lease=claimed.lease_token, paused_until=until
    )
    clock.advance(29 * 60)
    with pytest.raises(PauseNotOverError, match="paused until"):
        store.transition(claimed.id, S.PAUSED, S.READY, A.SCHEDULER)
    assert store.get_subtask(claimed.id).status is S.PAUSED
    clock.advance(60)
    assert store.transition(claimed.id, S.PAUSED, S.READY, A.SCHEDULER).status is S.READY


# ── Retry policy through the store ───────────────────────────────────────────


def reject(store: Store) -> Subtask:
    claimed = claim(store)
    store.transition(claimed.id, S.RUNNING, S.REVIEW, A.EXECUTOR, lease=claimed.lease_token)
    return store.record_failure(claimed.id, S.REVIEW, A.REVIEWER, FailureKind.QUALITY)


def test_repeated_rejections_escalate_to_human(store: Store) -> None:
    task_id = store.create_task(spec(sub("a")))
    assert reject(store).status is S.READY
    assert reject(store).status is S.READY
    third = reject(store)
    assert third.status is S.NEEDS_HUMAN
    assert third.attempts == Attempts(quality=3)
    last = json.loads(store.audit_log(task_id)[-1]["details_json"])
    assert last["failure"] == "quality" and "retries exhausted" in last["reason"]


def test_human_restart_resets_attempts(store: Store) -> None:
    store.create_task(spec(sub("a")))
    for _ in range(3):
        escalated = reject(store)
    restarted = store.transition(escalated.id, S.NEEDS_HUMAN, S.READY, A.HUMAN)
    assert restarted.attempts == Attempts()
    assert reject(store).status is S.READY  # a fresh budget


def test_executor_failures_use_lease_and_transient_budget(store: Store) -> None:
    store.create_task(spec(sub("a")))
    policy = RetryPolicy(max_transient=2)
    first = claim(store)
    with pytest.raises(ValueError, match="lease="):
        store.record_failure(first.id, S.RUNNING, A.EXECUTOR, FailureKind.TRANSIENT, policy=policy)
    retried = store.record_failure(
        first.id,
        S.RUNNING,
        A.EXECUTOR,
        FailureKind.TRANSIENT,
        lease=first.lease_token,
        policy=policy,
    )
    assert (retried.status, retried.attempts) == (S.READY, Attempts(transient=1))
    second = claim(store)
    escalated = store.record_failure(
        second.id,
        S.RUNNING,
        A.EXECUTOR,
        FailureKind.TRANSIENT,
        lease=second.lease_token,
        policy=policy,
    )
    assert escalated.status is S.NEEDS_HUMAN


def test_limit_failure_pauses_without_spending_attempts(store: Store, clock: FakeClock) -> None:
    store.create_task(spec(sub("a")))
    claimed = claim(store)
    until = db.to_iso(clock.now + timedelta(minutes=5))
    paused = store.record_failure(
        claimed.id,
        S.RUNNING,
        A.EXECUTOR,
        FailureKind.LIMIT,
        lease=claimed.lease_token,
        paused_until=until,
        reason="429 from provider",
    )
    assert (paused.status, paused.paused_until, paused.attempts) == (S.PAUSED, until, Attempts())


def test_failed_record_failure_changes_nothing(store: Store) -> None:
    store.create_task(spec(sub("a")))
    claimed = claim(store)
    with pytest.raises(StaleStateError):
        store.record_failure(claimed.id, S.RUNNING, A.EXECUTOR, FailureKind.QUALITY, lease="forged")
    assert store.get_subtask(claimed.id).attempts == Attempts()
