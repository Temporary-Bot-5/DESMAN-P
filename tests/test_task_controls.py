"""Operator pauses use preemption and preserve durable task history."""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import FakeClock
from desman.core import db
from desman.core.models import SubtaskSpec, TaskSpec
from desman.core.report import task_summaries
from desman.core.states import Actor
from desman.core.states import SubtaskStatus as S
from desman.core.store import Store


def task(store: Store) -> str:
    return store.create_task(
        TaskSpec(goal="goal", subtasks=[SubtaskSpec(description="work", acceptance_criteria="ok")])
    )


def test_upgrade_previous_schema() -> None:
    conn = db.connect(":memory:")
    db.migrate(conn, db.MIGRATIONS[:-1])
    task(Store(conn))
    db.migrate(conn)
    row = conn.execute("SELECT pinned, hidden_at, paused_at FROM tasks").fetchone()
    assert tuple(row) == (0, None, None)
    db.verify_schema(conn)
    conn.close()


def test_pin_bulk_pause_and_resume(store: Store) -> None:
    pinned, other = task(store), task(store)
    store.pin_task(pinned)
    assert store.pause_unfinished() == 1
    claimed = store.claim("worker")
    assert claimed and claimed.task_id == pinned
    assert store.reserve_review(store.list_subtasks(other)[0].id, "reviewer") is None
    store.resume_task(other)
    second = store.claim("second")
    assert second and second.task_id == other
    assert [r["action"] for r in store.audit_log(other) if r["action"].startswith("task_")] == [
        "task_created",
        "task_paused",
        "task_resumed",
    ]


def test_pause_running_preserves_checkpoint_and_attempt(store: Store) -> None:
    ref = task(store)
    running = store.claim("worker")
    assert running and running.lease_token
    store.heartbeat(running.id, running.lease_token, checkpoint={"step": 3})
    store.pause_task(ref)
    assert store.yield_paused(running.id, running.lease_token)
    waiting = store.get_subtask(running.id)
    assert waiting.status is S.READY
    assert waiting.checkpoint == {"step": 3}
    assert waiting.attempts == running.attempts
    assert store.claim("other") is None
    store.resume_task(ref)
    resumed = store.claim("other")
    assert resumed and resumed.checkpoint == {"step": 3}


def test_hide_finished_keeps_history_and_pin(store: Store) -> None:
    finished, pinned, unfinished = task(store), task(store), task(store)
    for ref in (finished, pinned):
        sub = store.claim("worker", task_id=ref)
        assert sub
        store.transition(sub.id, S.RUNNING, S.REVIEW, Actor.EXECUTOR, lease=sub.lease_token)
        store.transition(sub.id, S.REVIEW, S.DONE, Actor.REVIEWER)
    store.pin_task(pinned)
    with pytest.raises(ValueError, match="unfinished"):
        store.hide_task(unfinished)
    assert store.hide_finished() == 1
    assert {r.id for r in task_summaries(store)} == {pinned, unfinished}
    assert {r.id for r in task_summaries(store, all=True)} == {pinned, unfinished, finished}
    assert store.find_task(finished)["hidden_at"]
    assert store.audit_log(finished)[-1]["action"] == "task_hidden"


@pytest.mark.parametrize("checkpointed", [False, True])
def test_pause_other_worker_then_resume_without_duplicate_effects(
    tmp_path: Path, checkpointed: bool
) -> None:
    import threading

    from desman.core.audit import check_history
    from desman.core.coordinator import Coordinator, Review, Success, Verdict, Work
    from desman.core.tools import ToolCall
    from test_effects import ACCESS, BumpArgs, bump_tool, registry

    path = tmp_path / "workers.db"
    conn = db.open_db(path)
    store = Store(conn)
    ref = task(store)
    entered, proceed = threading.Event(), threading.Event()
    ran: list[int] = []
    failures: list[BaseException] = []
    tools = registry(bump_tool(ran))

    def execute(work: Work) -> Success:
        assert work.tools is not None
        if work.context.checkpoint is None:
            work.tools.call(ToolCall(tool="bump", args=BumpArgs(n=1).model_dump()))
            if checkpointed:
                work.heartbeat({"stage": "after effect"})
            entered.set()
            assert proceed.wait(10)
            work.heartbeat(None)
        else:
            assert work.context.checkpoint == {"stage": "after effect"}
        return Success(content={"ok": True})

    def worker() -> None:
        own = db.open_db(path)
        try:
            Coordinator(
                Store(own),
                execute,
                lambda *_: Review(verdict=Verdict.ACCEPT),
                tools=tools,
                access=ACCESS,
                workspaces=tmp_path / "workspaces",
            ).run(ref)
        except BaseException as exc:
            failures.append(exc)
        finally:
            own.close()

    thread = threading.Thread(target=worker)
    thread.start()
    try:
        assert entered.wait(10)
        store.pause_task(ref)
    finally:
        proceed.set()
        thread.join(10)
    assert not thread.is_alive() and not failures
    assert store.list_subtasks(ref)[0].status is S.READY
    assert store.claim("another") is None
    store.resume_task(ref)
    worker()
    assert not failures and ran == [1]
    assert store.task_status(ref).value == "done"
    assert check_history(store.conn, ref) == []
    conn.close()


def test_manual_pause_blocks_review_reservations(store: Store) -> None:
    ref = task(store)
    running = store.claim("executor")
    assert running
    store.transition(running.id, S.RUNNING, S.REVIEW, Actor.EXECUTOR, lease=running.lease_token)
    store.pause_task(ref)
    assert store.pending_reviews() == []
    assert store.reserve_review(running.id, "reviewer") is None
    store.resume_task(ref)
    assert [s.id for s in store.pending_reviews()] == [running.id]
    assert store.reserve_review(running.id, "reviewer") is not None


def test_manual_pause_blocks_provider_timed_resume(store: Store, clock: FakeClock) -> None:
    from datetime import timedelta

    ref = task(store)
    running = store.claim("executor")
    assert running
    store.transition(
        running.id,
        S.RUNNING,
        S.PAUSED,
        Actor.EXECUTOR,
        lease=running.lease_token,
        paused_until=db.to_iso(clock() + timedelta(seconds=1)),
    )
    store.pause_task(ref)
    clock.advance(2)
    assert store.resume_due() == []
    assert store.get_subtask(running.id).status is S.PAUSED
    store.resume_task(ref)
    assert [s.id for s in store.resume_due()] == [running.id]
    assert store.claim("executor") is not None
