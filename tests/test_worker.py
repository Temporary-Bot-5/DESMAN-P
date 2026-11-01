"""Tests for workers (ROADMAP 4.1): `desman worker`, tasks that keep their own retry policy,
reviews reserved so that two workers never pay for the same verdict."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from conftest import FakeClock
from desman.cli import cli
from desman.core import db
from desman.core.audit import check_history
from desman.core.coordinator import Coordinator, Review, Success, Verdict, Work
from desman.core.models import Priority, Subtask, SubtaskSpec, TaskSpec, Tools
from desman.core.retry import FailureKind, RetryPolicy
from desman.core.states import Actor, SubtaskStatus, TaskStatus
from desman.core.store import DEFAULT_LEASE_SECONDS, DEFAULT_REVIEW_SECONDS, Store

S = SubtaskStatus


def spec(*names: str, retry: RetryPolicy | None = None, **fields: Any) -> TaskSpec:
    return TaskSpec(
        goal="g",
        subtasks=[
            SubtaskSpec(description=n, acceptance_criteria="ok", depends_on=[i - 1] if i else [])
            for i, n in enumerate(names)
        ],
        retry=retry or RetryPolicy(),
        **fields,
    )


def ok(work: Work) -> Success:
    return Success(content={"by": work.subtask.description})


def accept(subtask: Subtask, artifact: dict[str, Any] | None) -> Review:
    return Review(verdict=Verdict.ACCEPT)


def worker(store: Store, clock: FakeClock, owner: str, **kwargs: Any) -> Coordinator:
    kwargs.setdefault("executor", ok)
    kwargs.setdefault("reviewer", accept)
    return Coordinator(store, owner=owner, sleep=clock.sleep, **kwargs)


def submitted(store: Store, task_id: str) -> Subtask:
    claimed = store.claim("w", task_id=task_id)
    assert claimed is not None and claimed.lease_token is not None
    return store.submit(claimed.id, claimed.lease_token, {"x": 1})


# ── The retry policy belongs to the task ─────────────────────────────────────


def test_a_task_keeps_its_retry_policy(store: Store) -> None:
    task_id = store.create_task(spec("a", retry=RetryPolicy(max_attempts=2, max_transient=7)))
    assert store.policy(task_id) == RetryPolicy(max_attempts=2, max_transient=7)
    created = store.audit_log(task_id)[0]
    assert json.loads(created["details_json"])["retry"] == {"max_attempts": 2, "max_transient": 7}
    assert store.tools(task_id) is Tools.GRANTED


def test_the_column_defaults_are_the_default_policy() -> None:
    # A task stored before 4.1 gets the budgets a task made now gets by default.
    columns = db.MIGRATIONS[15]
    default = RetryPolicy()
    assert f"max_attempts INTEGER NOT NULL DEFAULT {default.max_attempts}" in columns
    assert f"max_transient INTEGER NOT NULL DEFAULT {default.max_transient}" in columns


def test_a_sweep_of_every_task_charges_each_by_its_own_policy(
    store: Store, clock: FakeClock
) -> None:
    fragile = store.create_task(spec("fragile", retry=RetryPolicy(max_transient=1)))
    sturdy = store.create_task(spec("sturdy"))
    for task_id in (fragile, sturdy):
        assert store.claim("dead-worker", task_id=task_id) is not None
    clock.advance(DEFAULT_LEASE_SECONDS + 1)
    recovered, _ = store.sweep()  # no task_id: whoever sweeps, whatever its own settings
    assert {s.task_id: s.status for s in recovered} == {fragile: S.NEEDS_HUMAN, sturdy: S.READY}


def test_a_failure_is_judged_by_the_tasks_policy(store: Store) -> None:
    task_id = store.create_task(spec("a", retry=RetryPolicy(max_attempts=1)))
    claimed = store.claim("w", task_id=task_id)
    assert claimed is not None
    moved = store.record_failure(
        claimed.id, S.RUNNING, Actor.EXECUTOR, FailureKind.QUALITY, lease=claimed.lease_token
    )
    assert moved.status is S.NEEDS_HUMAN


# ── Reviews are reserved ─────────────────────────────────────────────────────


def test_a_review_is_reserved_by_one_reviewer(store: Store, clock: FakeClock) -> None:
    task_id = store.create_task(spec("a"))
    sub = submitted(store, task_id)
    assert store.reserve_review(sub.id, "w1")
    assert not store.reserve_review(sub.id, "w2")
    assert store.reserve_review(sub.id, "w1")  # its own, again (renewed)
    held = store.get_subtask(sub.id)
    assert held.review_owner == "w1" and held.review_until is not None
    clock.advance(DEFAULT_REVIEW_SECONDS + 1)
    assert store.reserve_review(sub.id, "w2")  # ran out: taken over


def test_a_gone_reviewers_reservation_is_taken_over_at_once(store: Store) -> None:
    sub = submitted(store, store.create_task(spec("a")))
    assert store.reserve_review(sub.id, "dead")
    assert not store.reserve_review(sub.id, "w2", abandoned=lambda owner: False)
    assert store.reserve_review(sub.id, "w2", abandoned=lambda owner: owner == "dead")


def test_the_verdict_ends_the_reservation(store: Store) -> None:
    sub = submitted(store, store.create_task(spec("a")))
    assert store.reserve_review(sub.id, "w1")
    done = store.transition(sub.id, S.REVIEW, S.DONE, Actor.REVIEWER, action="verdict")
    assert (done.review_owner, done.review_until) == (None, None)
    assert not store.reserve_review(sub.id, "w2")  # not in REVIEW any more


def test_an_interrupted_review_frees_the_reservation(store: Store) -> None:
    sub = submitted(store, store.create_task(spec("a")))
    assert store.reserve_review(sub.id, "w1")
    kept = store.record_failure(sub.id, S.REVIEW, Actor.REVIEWER, FailureKind.TRANSIENT)
    assert kept.status is S.REVIEW and kept.review_owner is None
    assert store.reserve_review(sub.id, "w2")
    store.release_review(sub.id, "w1")  # not the holder: changes nothing
    assert store.get_subtask(sub.id).review_owner == "w2"
    store.release_review(sub.id, "w2")
    assert store.get_subtask(sub.id).review_owner is None


def test_the_schema_allows_a_reservation_only_in_review(store: Store) -> None:
    task_id = store.create_task(spec("a"))
    [sub] = store.list_subtasks(task_id)
    with pytest.raises(Exception, match="CHECK"):
        store.conn.execute(
            "UPDATE subtasks SET review_owner = 'w', review_until = '2026-01-01T00:00:00.000Z' "
            "WHERE id = ?",
            (sub.id,),
        )


def test_two_workers_never_review_the_same_result(store: Store, clock: FakeClock) -> None:
    task_id = store.create_task(spec("a"))
    submitted(store, task_id)
    second_reviews: list[str] = []

    def second_reviewer(subtask: Subtask, artifact: dict[str, Any] | None) -> Review:
        second_reviews.append(subtask.id)
        return Review(verdict=Verdict.ACCEPT)

    second = worker(store, clock, "w2", reviewer=second_reviewer)

    def first_reviewer(subtask: Subtask, artifact: dict[str, Any] | None) -> Review:
        # While this review is under way, the other worker looks for work.
        assert second.tick(None, []) is False
        return Review(verdict=Verdict.ACCEPT)

    first = worker(store, clock, "w1", reviewer=first_reviewer)
    assert first.tick(None, []) is True
    assert second_reviews == []
    assert store.task_status(task_id) is TaskStatus.DONE
    assert (first.reviewed, second.reviewed) == (1, 0)


def test_a_crashed_review_is_released(store: Store, clock: FakeClock) -> None:
    sub = submitted(store, store.create_task(spec("a")))

    def interrupted(subtask: Subtask, artifact: dict[str, Any] | None) -> Review:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        worker(store, clock, "w1", reviewer=interrupted).tick(None, [])
    assert store.get_subtask(sub.id).review_owner is None  # another worker need not wait


def test_a_run_waits_for_a_review_someone_else_holds(store: Store, clock: FakeClock) -> None:
    task_id = store.create_task(spec("a"))
    sub = submitted(store, task_id)
    assert store.reserve_review(sub.id, "other-worker")  # alive, but never finishes
    result = worker(store, clock, "me").run(task_id)
    assert result.status is TaskStatus.DONE
    assert clock.slept == pytest.approx(DEFAULT_REVIEW_SECONDS, abs=0.01)


def test_a_run_looks_again_soon_while_a_worker_holds_its_subtask(
    store: Store, clock: FakeClock
) -> None:
    task_id = store.create_task(spec("a"))
    claimed = store.claim("busy-worker", task_id=task_id)
    assert claimed is not None and claimed.lease_token is not None
    naps: list[float] = []

    def sleep(seconds: float) -> None:
        naps.append(seconds)
        clock.sleep(seconds)
        if len(naps) == 3:  # the worker finishes meanwhile
            assert claimed.lease_token is not None
            store.submit(claimed.id, claimed.lease_token, {"by": "busy-worker"})

    me = Coordinator(store, ok, accept, owner="me", sleep=sleep, poll_seconds=2)
    assert me.run(task_id).status is TaskStatus.DONE
    assert naps == [2, 2, 2]  # not the whole lease
    assert store.latest_artifact(claimed.id) == {"by": "busy-worker"}


# ── Working on every task ────────────────────────────────────────────────────


def test_a_worker_does_every_task_most_urgent_first(store: Store, clock: FakeClock) -> None:
    order: list[str] = []

    def executor(work: Work) -> Success:
        order.append(work.subtask.description)
        return ok(work)

    low = store.create_task(spec("low", priority=Priority.LOW))
    high = store.create_task(spec("high-1", "high-2", priority=Priority.HIGH))
    result = worker(store, clock, "w1", executor=executor).work(until_idle=True)
    assert order == ["high-1", "high-2", "low"]
    assert (result.executed, result.reviewed) == (3, 3)
    for task_id in (low, high):
        assert store.task_status(task_id) is TaskStatus.DONE
        assert check_history(store.conn, task_id) == []


def test_an_idle_worker_waits_for_new_work(store: Store, clock: FakeClock) -> None:
    created: list[str] = []

    def sleep(seconds: float) -> None:
        clock.sleep(seconds)
        if not created:
            created.append(store.create_task(spec("late")))

    w = Coordinator(store, ok, accept, owner="w1", sleep=sleep, poll_seconds=3)
    result = w.work(max_ticks=3)  # idle, sleep; execute; review
    assert clock.slept == 3 and result.executed == 1
    assert store.task_status(created[0]) is TaskStatus.DONE


def test_until_idle_still_waits_for_work_another_worker_holds(
    store: Store, clock: FakeClock
) -> None:
    task_id = store.create_task(spec("a"))
    assert store.claim("dead-elsewhere", task_id=task_id) is not None  # another machine
    result = worker(store, clock, "w1").work(until_idle=True)
    assert result.executed == 1  # recovered once the lease ran out, then done
    assert store.task_status(task_id) is TaskStatus.DONE
    assert clock.slept >= DEFAULT_LEASE_SECONDS


def test_a_task_without_tools_gets_none_from_any_worker(store: Store, clock: FakeClock) -> None:
    seen: dict[str, list[str]] = {}

    def executor(work: Work) -> Success:
        seen[work.subtask.description] = [tool.name for tool in work.tools.available()]
        return ok(work)

    store.create_task(spec("summary", tools=Tools.NONE))
    store.create_task(spec("ordinary"))
    worker(store, clock, "w1", executor=executor).work(until_idle=True)
    assert seen["summary"] == []
    assert seen["ordinary"]  # the workspace tools at least


# ── `desman worker` ──────────────────────────────────────────────────────────


def write(path: Path, data: Any) -> Path:
    path.write_text(json.dumps(data))
    return path


def fake_script(tmp_path: Path, latency: float = 0.0) -> Path:
    reply = {"type": "reply", "text": '{"done": true}', "latency": latency}
    routes = [
        {"match": "Execute", "steps": [reply], "repeat_last": True},
        {"match": "Review", "steps": ['{"verdict": "accept"}'], "repeat_last": True},
    ]
    return write(tmp_path / "fake.json", {"routes": routes})


def tasks_in(path: Path, count: int, steps: int = 1) -> list[str]:
    store = Store(db.open_db(path))
    try:
        names = [f"step {n}" for n in range(steps)]
        return [store.create_task(spec(*names)) for _ in range(count)]
    finally:
        store.conn.close()


def test_the_worker_command_works_until_idle(tmp_path: Path) -> None:
    path = tmp_path / "desman.db"
    ids = tasks_in(path, 2, steps=2)
    args = ["worker", "--until-idle", "--fake-llm", str(fake_script(tmp_path)), "--db", str(path)]
    result = CliRunner().invoke(cli, [*args, "--json"])
    assert result.exit_code == 0, result.output
    out = json.loads(result.output)
    assert (out["executed"], out["reviewed"], out["interrupted"]) == (4, 4, False)
    store = Store(db.open_db(path))
    try:
        assert {store.task_status(i) for i in ids} == {TaskStatus.DONE}
    finally:
        store.conn.close()

    shown = CliRunner().invoke(cli, args)  # nothing left: starts and stops
    assert shown.exit_code == 0, shown.output
    assert "started" in shown.output and "Executions: 0, reviews: 0." in shown.output


@pytest.mark.skipif(os.name != "posix", reason="separate processes on one database")
def test_several_worker_processes_share_one_database(tmp_path: Path) -> None:
    # Real processes and real time, like tests/test_crash.py: each subtask is executed and
    # reviewed exactly once, whichever worker took it.
    path = tmp_path / "desman.db"
    ids = tasks_in(path, 6, steps=2)
    script = fake_script(tmp_path, latency=0.05)
    command = [sys.executable, "-m", "desman", "worker", "--until-idle", "--poll-seconds", "0.1"]
    command += ["--fake-llm", str(script), "--db", str(path), "--json"]
    started = time.monotonic()
    workers = [
        subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE) for _ in range(3)
    ]
    outputs = []
    for proc in workers:
        out, err = proc.communicate(timeout=120)
        assert proc.returncode == 0, err.decode()
        outputs.append(json.loads(out))
    assert time.monotonic() - started < 120
    assert len({o["owner"] for o in outputs}) == 3
    assert sum(o["executed"] for o in outputs) == 12
    assert sum(o["reviewed"] for o in outputs) == 12

    conn = db.open_db(path)
    try:
        store = Store(conn)
        assert {store.task_status(i) for i in ids} == {TaskStatus.DONE}
        actions = [r[0] for r in conn.execute("SELECT action FROM audit_log")]
        assert actions.count("claim") == 12 and actions.count("verdict") == 12
        assert all(check_history(conn, i) == [] for i in ids)
    finally:
        conn.close()
