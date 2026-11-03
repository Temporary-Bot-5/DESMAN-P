"""Tests for the global scheduler (ROADMAP 4.3): urgency across tasks, aging against starvation,
critical work first, reviews and executions in one order."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

import pytest

from conftest import FakeClock
from desman.core import db
from desman.core.coordinator import Coordinator, Review, Success, Verdict, Work
from desman.core.models import Priority, Subtask, SubtaskSpec, TaskSpec
from desman.core.scheduler import DEFAULT_AGING, Aging
from desman.core.states import Actor, SubtaskStatus
from desman.core.store import Store

S = SubtaskStatus
A = Actor


def task(store: Store, name: str, priority: int) -> str:
    """A one-subtask task; the subtask is described by `name`."""
    return store.create_task(
        TaskSpec(
            goal=name,
            priority=priority,
            subtasks=[SubtaskSpec(description=name, acceptance_criteria="ok")],
        )
    )


def claim(store: Store) -> Subtask:
    claimed = store.claim("w1")
    assert claimed is not None
    return claimed


def finish(store: Store, claimed: Subtask) -> None:
    store.transition(claimed.id, S.RUNNING, S.REVIEW, A.EXECUTOR, lease=claimed.lease_token)
    store.transition(claimed.id, S.REVIEW, S.DONE, A.REVIEWER)


# ── Urgency ──────────────────────────────────────────────────────────────────


def test_urgency_grows_with_the_wait_up_to_just_below_critical(clock: FakeClock) -> None:
    queued = db.to_iso(clock.now)

    def at(seconds: float) -> datetime:
        return clock.now + timedelta(seconds=seconds)

    aging = DEFAULT_AGING
    assert aging.urgency(Priority.LOW, queued, at(0)) == 0
    assert aging.urgency(Priority.LOW, queued, at(59.999)) == 0
    assert aging.urgency(Priority.LOW, queued, at(60)) == 10
    assert aging.urgency(Priority.NORMAL, queued, at(600)) == 150
    assert aging.urgency(Priority.LOW, queued, at(86_400)) == Priority.CRITICAL - 1
    # Critical is first already: it does not age, and nothing aged reaches it.
    assert aging.urgency(Priority.CRITICAL, queued, at(86_400)) == Priority.CRITICAL
    assert aging.urgency(300, queued, at(86_400)) == 300
    # Not queued, or queued "in the future" by another machine's clock: no bonus.
    assert aging.urgency(Priority.LOW, None, at(600)) == 0
    assert aging.urgency(Priority.LOW, queued, at(-600)) == 0


def test_aging_rejects_nonsense_values() -> None:
    with pytest.raises(ValueError, match="aging"):
        Aging(step=1, every_seconds=0)
    with pytest.raises(ValueError, match="aging"):
        Aging(step=-1)


@pytest.mark.parametrize("aging", [DEFAULT_AGING, Aging(step=7, every_seconds=45)])
@pytest.mark.parametrize("priority", [-5, 0, 50, 100, 199, 200, 300])
@pytest.mark.parametrize(
    "waited", [None, -30.0, 0.0, 44.999, 45.0, 59.999, 60.0, 60.001, 119.999, 600.0, 86_400.0]
)
@pytest.mark.parametrize("origin", ["local", "network"])
def test_sql_and_python_agree(
    clock: FakeClock, aging: Aging, priority: int, waited: float | None, origin: str
) -> None:
    """The claim orders in SQL, the coordinator compares in Python: they must never differ,
    not even on a step boundary; nor for a job from the network (G.6b)."""
    now = clock.now + timedelta(days=3)
    queued = None if waited is None else db.to_iso(now - timedelta(seconds=waited))
    conn = db.open_db(":memory:")
    conn.execute(
        "INSERT INTO tasks (id, goal, priority, created_at, max_attempts, max_transient, origin)"
        " VALUES ('t', 'g', 0, ?, 3, 3, ?)",
        (db.to_iso(now), origin),
    )
    (in_sql,) = conn.execute(
        f"SELECT {aging.urgency_sql()} FROM"
        " (SELECT ? AS priority, ? AS queued_at, 't' AS task_id) AS subtasks",
        (db.to_iso(now), priority, queued),
    ).fetchone()
    assert in_sql == aging.urgency(priority, queued, now, network=origin == "network")


# ── Claim order across tasks ─────────────────────────────────────────────────


def test_a_low_priority_task_is_not_starved_by_a_stream_of_urgent_ones(
    store: Store, clock: FakeClock
) -> None:
    """A new HIGH task arrives every minute and is worked on at once. Without aging the LOW
    task would wait forever; with it, it is taken after ten minutes (equal urgency then, and it
    has waited longer)."""
    task(store, "low", Priority.LOW)
    taken_at = None
    for minute in range(30):
        task(store, f"high {minute}", Priority.HIGH)
        claimed = claim(store)
        if claimed.description == "low":
            taken_at = minute
            break
        finish(store, claimed)
        clock.advance(60)
    assert taken_at == 10


def test_without_aging_the_stream_would_starve_it(clock: FakeClock) -> None:
    store = Store(db.open_db(":memory:"), clock=clock, aging=Aging(step=0))
    task(store, "low", Priority.LOW)
    for minute in range(30):
        task(store, f"high {minute}", Priority.HIGH)
        claimed = claim(store)
        assert claimed.description != "low"
        finish(store, claimed)
        clock.advance(60)


def test_critical_work_goes_before_any_aged_work(store: Store, clock: FakeClock) -> None:
    task(store, "low", Priority.LOW)
    clock.advance(1)
    task(store, "high", Priority.HIGH)
    clock.advance(86_400)
    # A day's wait lifts both to the ceiling: then the one waiting longer goes first.
    task(store, "critical", Priority.CRITICAL)
    assert [claim(store).description for _ in range(3)] == ["critical", "low", "high"]


def test_among_equals_the_longest_waiting_goes_first(store: Store, clock: FakeClock) -> None:
    first = task(store, "first", Priority.NORMAL)
    clock.advance(1)
    task(store, "second", Priority.NORMAL)
    assert claim(store).task_id == first


def test_waiting_starts_over_in_each_queue(store: Store, clock: FakeClock) -> None:
    (ready,) = store.list_subtasks(task(store, "a", Priority.NORMAL))
    assert ready.queued_at == db.to_iso(clock.now)
    clock.advance(10)
    running = claim(store)
    assert running.queued_at is None
    clock.advance(10)
    in_review = store.transition(
        running.id, S.RUNNING, S.REVIEW, A.EXECUTOR, lease=running.lease_token
    )
    assert in_review.queued_at == db.to_iso(clock.now)
    clock.advance(10)
    again = store.transition(running.id, S.REVIEW, S.READY, A.REVIEWER)
    assert again.queued_at == db.to_iso(clock.now)


def test_the_claim_records_how_urgent_the_work_was(store: Store, clock: FakeClock) -> None:
    task_id = task(store, "low", Priority.LOW)
    clock.advance(180)
    claim(store)
    (entry,) = [r for r in store.audit_log(task_id) if r["action"] == "claim"]
    assert json.loads(entry["details_json"])["urgency"] == 30


def test_next_ready_is_what_claim_would_take(store: Store, clock: FakeClock) -> None:
    task(store, "normal", Priority.NORMAL)
    task(store, "high", Priority.HIGH)
    peeked = store.next_ready()
    assert peeked is not None and peeked.description == "high"
    assert store.urgency(peeked) == Priority.HIGH
    assert claim(store).id == peeked.id


# ── Reviews and executions in one order ──────────────────────────────────────


class Log:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def execute(self, work: Work) -> Success:
        self.calls.append(f"execute {work.subtask.description}")
        return Success(content={})

    def review(self, subtask: Subtask, artifact: dict[str, Any] | None) -> Review:
        self.calls.append(f"review {subtask.description}")
        return Review(verdict=Verdict.ACCEPT)


def in_review(store: Store, name: str, priority: int) -> None:
    task(store, name, priority)
    claimed = store.claim("earlier", task_id=None)
    assert claimed is not None and claimed.description == name
    store.transition(claimed.id, S.RUNNING, S.REVIEW, A.EXECUTOR, lease=claimed.lease_token)


def test_urgent_work_runs_before_a_less_urgent_review(store: Store, clock: FakeClock) -> None:
    in_review(store, "low", Priority.LOW)
    task(store, "critical", Priority.CRITICAL)
    log = Log()
    coordinator = Coordinator(store, log.execute, log.review, sleep=clock.sleep)
    assert coordinator.tick(None, [])
    assert log.calls == ["execute critical"]
    assert coordinator.tick(None, [])
    assert log.calls == ["execute critical", "review critical", "review low"]


def test_a_review_as_urgent_as_the_ready_work_goes_first(store: Store, clock: FakeClock) -> None:
    in_review(store, "high", Priority.HIGH)
    task(store, "low", Priority.LOW)
    log = Log()
    coordinator = Coordinator(store, log.execute, log.review, sleep=clock.sleep)
    assert coordinator.tick(None, [])
    assert log.calls == ["review high", "execute low"]


def test_work_a_verdict_unlocks_goes_before_less_urgent_reviews(
    store: Store, clock: FakeClock
) -> None:
    """Accepting `first` makes its critical dependent ready: it runs before the LOW review."""
    store.create_task(
        TaskSpec(
            goal="g",
            priority=Priority.HIGH,
            subtasks=[
                SubtaskSpec(description="first", acceptance_criteria="ok"),
                SubtaskSpec(
                    description="then",
                    acceptance_criteria="ok",
                    depends_on=[0],
                    priority=Priority.CRITICAL,
                ),
            ],
        )
    )
    claimed = claim(store)
    store.transition(claimed.id, S.RUNNING, S.REVIEW, A.EXECUTOR, lease=claimed.lease_token)
    in_review(store, "low", Priority.LOW)
    log = Log()
    coordinator = Coordinator(store, log.execute, log.review, sleep=clock.sleep)
    assert coordinator.tick(None, [])
    assert log.calls == ["review first", "execute then"]
