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
from desman.core.models import Priority, SubtaskSpec, TaskSpec
from desman.core.states import Actor, InvalidTransitionError, SubtaskStatus
from desman.core.store import StaleStateError, Store

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


def finish(store: Store, subtask_id: str, owner: str) -> None:
    store.transition(subtask_id, S.RUNNING, S.REVIEW, A.EXECUTOR, owner=owner)
    store.transition(subtask_id, S.REVIEW, S.DONE, A.REVIEWER)


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
    finish(store, a.id, "w1")
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
        claimed.id, S.RUNNING, S.REVIEW, A.EXECUTOR, owner="w1", reason="submitted"
    )
    assert moved.status is S.REVIEW
    assert (moved.lease_owner, moved.lease_until) == (None, None)
    last = store.audit_log(task_id)[-1]
    assert (last["actor"], last["from_status"], last["to_status"]) == (
        "executor",
        "running",
        "review",
    )
    assert json.loads(last["details_json"]) == {"reason": "submitted", "owner": "w1"}


def test_stale_transition_is_refused(store: Store) -> None:
    store.create_task(spec(sub("a")))
    claimed = store.claim("w1")
    assert claimed is not None
    store.transition(claimed.id, S.RUNNING, S.REVIEW, A.EXECUTOR, owner="w1")
    with pytest.raises(StaleStateError, match="is review, expected running"):
        store.transition(claimed.id, S.RUNNING, S.READY, A.SCHEDULER)


def test_only_lease_holder_can_leave_running(store: Store) -> None:
    store.create_task(spec(sub("a")))
    claimed = store.claim("w1")
    assert claimed is not None
    with pytest.raises(StaleStateError, match="leased to w1, not w2"):
        store.transition(claimed.id, S.RUNNING, S.REVIEW, A.EXECUTOR, owner="w2")
    with pytest.raises(ValueError, match="owner"):
        store.transition(claimed.id, S.RUNNING, S.REVIEW, A.EXECUTOR)
    assert store.get_subtask(claimed.id).status is S.RUNNING


def test_invalid_transition_never_reaches_database(store: Store) -> None:
    task_id = store.create_task(spec(sub("a")))
    claimed = store.claim("w1")
    assert claimed is not None
    before = len(store.audit_log(task_id))
    with pytest.raises(InvalidTransitionError):
        store.transition(claimed.id, S.RUNNING, S.DONE, A.EXECUTOR, owner="w1")
    with pytest.raises(ValueError, match="claim"):
        store.transition(claimed.id, S.READY, S.RUNNING, A.EXECUTOR)
    assert store.get_subtask(claimed.id).status is S.RUNNING
    assert len(store.audit_log(task_id)) == before


@pytest.mark.parametrize(
    ("src", "actor", "resume"),
    [(S.RUNNING, A.EXECUTOR, S.READY), (S.REVIEW, A.REVIEWER, S.REVIEW)],
)
def test_pause_resumes_where_it_stopped(
    store: Store, src: SubtaskStatus, actor: Actor, resume: SubtaskStatus
) -> None:
    store.create_task(spec(sub("a")))
    claimed = store.claim("w1")
    assert claimed is not None
    if src is S.REVIEW:
        store.transition(claimed.id, S.RUNNING, S.REVIEW, A.EXECUTOR, owner="w1")
    paused = store.transition(
        claimed.id, src, S.PAUSED, actor, owner="w1", paused_until="2026-01-01T01:00:00.000Z"
    )
    assert (paused.resume_status, paused.paused_until) == (resume, "2026-01-01T01:00:00.000Z")
    wrong = S.REVIEW if resume is S.READY else S.READY
    with pytest.raises(StaleStateError, match=f"resume as {resume}"):
        store.transition(claimed.id, S.PAUSED, wrong, A.SCHEDULER)
    resumed = store.transition(claimed.id, S.PAUSED, resume, A.SCHEDULER)
    assert (resumed.status, resumed.resume_status, resumed.paused_until) == (resume, None, None)


def test_done_unlocks_dependents_across_diamond(store: Store) -> None:
    task_id = store.create_task(spec(sub("a"), sub("b", 0), sub("c", 0), sub("d", 1, 2)))
    a, b, c, d = store.list_subtasks(task_id)
    for expected in ([a.id], [b.id, c.id], [d.id]):
        got = []
        while (claimed := store.claim("w")) is not None:
            got.append(claimed.id)
        assert sorted(got) == sorted(expected)
        for subtask_id in got:
            finish(store, subtask_id, "w")
    assert all(s.status is S.DONE for s in store.list_subtasks(task_id))
