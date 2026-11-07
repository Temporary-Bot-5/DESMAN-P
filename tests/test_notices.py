"""Tests for telling the operator what waits for them (ROADMAP 5.1): `InboxWatch`, a worker's
terminal and the chat."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from conftest import FakeClock
from desman import actions
from desman.chat import Chat
from desman.cli import cli
from desman.config import ConfigManager, default_config_path
from desman.core import db
from desman.core.coordinator import Coordinator
from desman.core.models import SubtaskSpec, TaskSpec
from desman.core.report import InboxWatch, inbox
from desman.core.retry import FailureKind
from desman.core.sessions import Role, Sessions
from desman.core.states import Actor, SubtaskStatus, TaskStatus
from desman.core.store import Store
from helpers import make_config
from test_worker import accept, ok

S = SubtaskStatus


def spec(goal: str = "the goal") -> TaskSpec:
    return TaskSpec(goal=goal, subtasks=[SubtaskSpec(description="step", acceptance_criteria="ok")])


def escalate(store: Store, task_id: str, reason: str = "401 bad key") -> str:
    """The task's ready subtask taken and escalated to the operator, as a refused key does."""
    claimed = store.claim("w", task_id=task_id)
    assert claimed is not None and claimed.lease_token is not None
    store.record_failure(
        claimed.id,
        S.RUNNING,
        Actor.EXECUTOR,
        FailureKind.PERMANENT,
        lease=claimed.lease_token,
        reason=reason,
    )
    return claimed.id


def test_a_watch_tells_each_situation_once(store: Store) -> None:
    watch = InboxWatch(store)
    assert watch.new() == []
    task_id = store.create_task(spec())
    sub_id = escalate(store, task_id)
    [item] = watch.new()
    assert (item.subtask_id, item.status) == (sub_id, S.NEEDS_HUMAN)
    assert item.reason is not None and item.reason.startswith("401 bad key")
    assert watch.new() == []  # told already

    store.transition(sub_id, S.NEEDS_HUMAN, S.READY, Actor.HUMAN, reason="try again")
    assert watch.new() == []
    escalate(store, task_id, reason="401 again")  # back again: a new situation
    [again] = watch.new()
    assert again.reason is not None and again.reason.startswith("401 again")


def test_what_was_shown_another_way_is_not_new(store: Store) -> None:
    escalate(store, store.create_task(spec("retold")))
    escalate(store, store.create_task(spec("not yet")))
    watch = InboxWatch(store)
    watch.told([i for i in inbox(store) if i.goal == "retold"])
    assert [i.goal for i in watch.new()] == ["not yet"]


def test_an_idle_worker_still_tells_what_others_sent_to_the_person(
    store: Store, clock: FakeClock
) -> None:
    """Another process escalates while this worker has nothing to do: it hears of it."""
    task_id = store.create_task(spec("elsewhere"))
    claimed = store.claim("other", task_id=task_id)  # held by another worker
    assert claimed is not None and claimed.lease_token is not None
    watch = InboxWatch(store)
    told: list[str] = []

    def idle() -> None:
        told.extend(i.goal for i in watch.new())

    def sleep(seconds: float) -> None:
        clock.sleep(seconds)
        if store.task_status(task_id) is not TaskStatus.NEEDS_HUMAN:
            store.record_failure(
                claimed.id,
                S.RUNNING,
                Actor.EXECUTOR,
                FailureKind.PERMANENT,
                lease=claimed.lease_token,
                reason="401",
            )

    coordinator = Coordinator(store, ok, accept, owner="w", sleep=sleep)
    coordinator.work(max_ticks=2, on_idle=idle)  # idle twice: nothing of its own to do
    assert told == ["elsewhere"]


def fake_script(tmp_path: Path) -> Path:
    routes = [{"match": "Execute", "steps": [{"type": "rejected", "status": 401}]}]
    path = tmp_path / "fake.json"
    path.write_text(json.dumps({"routes": routes}))
    return path


def test_a_worker_tells_what_waits_and_what_comes_to_wait(tmp_path: Path) -> None:
    path = tmp_path / "desman.db"
    store = Store(db.open_db(path))
    try:
        escalate(store, store.create_task(spec("waiting already")))
        store.create_task(spec("refused on the way"))
    finally:
        store.conn.close()
    args = ["worker", "--until-idle", "--fake-llm", str(fake_script(tmp_path)), "--db", str(path)]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    out = result.output
    started = out.index("started")
    first = out.index("1 subtask waits for your decision:", started)
    assert out.index("waiting already", first) < out.index("Executions: 1", first)
    second = out.index("1 subtask waits for your decision:", first + 1)
    assert "refused on the way" in out[second:]
    assert out.count("waiting already") == 1  # each told once
    assert "Decide with: desman retry|accept|fail" in out

    quiet = CliRunner().invoke(cli, [*args, "--json"])  # machine output stays plain JSON
    assert quiet.exit_code == 0, quiet.output
    json.loads(quiet.output)


def test_the_chat_tells_what_waits_then_what_comes_from_elsewhere(
    clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    conn = db.open_db(":memory:")
    sessions = Sessions(conn, clock=clock)
    store = Store(conn, clock=clock)
    session_id = sessions.add(None, Role.USER, "hello").session_id
    escalate(store, store.create_task(spec("from before"), session_id=session_id))
    own = store.create_task(spec("asked here"), session_id=session_id)
    other = store.create_task(spec("from a worker"))
    said = iter(["/help", "first", None])

    def read() -> str | None:
        line = next(said)
        if line == "/help":  # while the person types: one of theirs, one from elsewhere
            escalate(store, own)
            escalate(store, other)
        return line

    told: list[str] = []
    monkeypatch.setattr(
        actions, "print_waiting", lambda items, hint: told.extend(i.goal for i in items)
    )
    try:
        Chat(sessions, read=read).run()
    finally:
        conn.close()
    # Without a model the chat retells nothing, so its own task is told as a notice too.
    assert told[0] == "from before"
    assert sorted(told[1:]) == ["asked here", "from a worker"]  # both told after /help


def test_the_chat_shows_what_waits_when_it_opens() -> None:
    ConfigManager(default_config_path()).save(make_config(language="en", configured=True))
    store = Store(db.open_db(db.default_db_path()))
    try:
        escalate(store, store.create_task(spec("left from yesterday")))
    finally:
        store.conn.close()
    result = CliRunner().invoke(cli, [], input="/exit\n")
    assert result.exit_code == 0, result.output
    assert "1 new in General Terminal (/terminal)" in result.output
    assert "left from yesterday" not in result.output
    conn = db.open_db(db.default_db_path())
    try:
        sessions = Sessions(conn)
        text = "\n".join(message.text for message in sessions.messages(sessions.terminal()))
        assert "1 subtask waits for your decision:" in text
        assert (
            "left from yesterday" in text
            and "Decide with `inbox`, or `retry`, `accept`, `fail` REF." in text
        )
    finally:
        conn.close()
