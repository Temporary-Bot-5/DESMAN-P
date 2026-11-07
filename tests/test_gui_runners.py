"""Event-gated model turns: independent chats, FIFO queues and human windows."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from threading import Event, get_ident
from typing import Any

import pytest
from click.testing import CliRunner

from conftest import FakeClock
from desman import runs, ui
from desman.cli import cli
from desman.commands import TuningArgs
from desman.config import Config, Tuning, open_config, upgrade
from desman.core import db
from desman.core.sessions import Role, Sessions
from desman.gui.front import WebFront
from desman.gui.panels import snapshot
from desman.gui.runners import PageChat, Runners
from desman.llm.base import Request
from desman.llm.fake import FakeLLM
from desman.settings_actions import Limits
from helpers import make_config


class Harness:
    def __init__(self, path: Path, clock: FakeClock, monkeypatch: pytest.MonkeyPatch) -> None:
        self.path = path
        self.conn = db.open_db(path)
        self.sessions = Sessions(self.conn, clock=clock)
        self.front = WebFront(clock=lambda: clock.slept)
        self.runners = Runners(path, self.front, clock=lambda: clock.slept, session_clock=clock)
        self.chat = PageChat(self.sessions, self.runners)
        self.entered: dict[str, Event] = {}
        self.gates: dict[str, Event] = {}
        self.outcomes: dict[str, dict[str, Any]] = {}
        self.models: list[tuple[int, sqlite3.Connection, FakeLLM]] = []
        self.requests: list[tuple[str, dict[str, Any]]] = []
        open_config().save(make_config(language="en", configured=True))

        def models(config: Config, conn: sqlite3.Connection, **kwargs: Any) -> runs.Models:
            fake = FakeLLM(clock=clock, sleep=clock.sleep).on(
                "Chat turn", self.reply, repeat_last=True
            )
            self.models.append((get_ident(), conn, fake))
            fake.on("Plan task", self.plan, repeat_last=True)
            fake.on("Retell task", self.retell, repeat_last=True)
            return runs.Models(fake, None)

        monkeypatch.setattr(runs, "models", models)

    def reply(self, request: Request) -> str:
        data = json.loads(request.prompt.split("\n", 1)[1])
        text = next(m["text"] for m in reversed(data["conversation"]) if m["from"] == "person")
        self.requests.append((text, data))
        self.entered.setdefault(text, Event()).set()
        if text in self.gates:
            assert self.gates[text].wait(10), f"unreleased gate: {text}"
        return json.dumps(self.outcomes.get(text, {"type": "reply", "text": f"reply:{text}"}))

    def retell(self, request: Request) -> str:
        self.entered.setdefault("retell", Event()).set()
        if "retell" in self.gates:
            assert self.gates["retell"].wait(10)
        return '{"text": "retold result"}'

    def plan(self, request: Request) -> str:
        self.entered.setdefault("plan", Event()).set()
        if "plan" in self.gates:
            assert self.gates["plan"].wait(10)
        return json.dumps({"subtasks": [{"description": "Do it", "acceptance_criteria": "Done"}]})

    def gate(self, text: str) -> Event:
        self.entered[text] = Event()
        self.gates[text] = Event()
        return self.gates[text]

    def send(self, text: str) -> str:
        with ui.use(self.front):
            assert self.chat.dispatch(text)
        assert self.chat.session_id is not None
        return self.chat.session_id

    def new(self) -> None:
        with ui.use(self.front):
            self.chat._new()

    def wait(self, predicate: Any) -> None:
        with self.front.condition:
            assert self.front.condition.wait_for(predicate, timeout=10)

    def done(self, session: str) -> None:
        self.wait(lambda: self.runners.status(session)["state"] is None)

    def answers(self, session: str) -> list[str]:
        return [m.text for m in self.sessions.messages(session) if m.role is Role.INTERLOCUTOR]

    def close(self) -> None:
        for event in self.gates.values():
            event.set()
        self.runners.close()
        self.conn.close()


@pytest.fixture
def page(tmp_path: Path, clock: FakeClock, monkeypatch: pytest.MonkeyPatch) -> Iterator[Harness]:
    harness = Harness(tmp_path / "page.db", clock, monkeypatch)
    try:
        yield harness
    finally:
        harness.close()


def test_a_blocked_model_leaves_slash_commands_and_b_free(page: Harness) -> None:
    gate = page.gate("A")
    a = page.send("A")
    assert page.entered["A"].wait(10)
    with ui.use(page.front):
        assert not page.chat.dispatch("/help")
        assert page.chat._slash("/help")
    assert any("help" in event.get("html", "") for event in page.front.events)
    page.new()
    b = page.send("B")
    page.done(b)
    assert page.answers(b) == ["reply:B"] and not page.answers(a)
    assert len({id(conn) for _, conn, _ in page.models}) == 2
    assert all(thread != get_ident() for thread, _, _ in page.models)
    gate.set()
    page.done(a)
    assert page.answers(a) == ["reply:A"]


def test_same_session_is_ordered_without_blocking_an_eligible_b(page: Harness) -> None:
    gate = page.gate("A1")
    a = page.send("A1")
    assert page.entered["A1"].wait(10)
    assert page.send("A2") == a
    assert page.runners.status(a)["queued"] == 1
    assert [m.text for m in page.sessions.messages(a) if m.role is Role.USER] == ["A1", "A2"]
    page.new()
    b = page.send("B")
    page.done(b)
    assert not page.answers(a)
    gate.set()
    page.done(a)
    assert page.answers(a) == ["reply:A1", "reply:A2"]
    first = next(data for text, data in page.requests if text == "A1")
    assert {"from": "person", "text": "A2"} not in first["conversation"]
    history = next(data for text, data in page.requests if text == "A2")
    assert {"from": "you", "text": "reply:A1"} in history["conversation"]


def test_cap_one_waits_fifo_and_reports_position(page: Harness) -> None:
    config = open_config().load()
    config.tuning.chat_turns = 1
    open_config().save(config)
    gate = page.gate("A")
    a = page.send("A")
    assert page.entered["A"].wait(10)
    page.new()
    b = page.send("B")
    page.wait(lambda: page.runners.status(b)["state"] == "queued")
    assert page.runners.status(b)["ahead"] == 1
    assert page.runners.status(b)["queued"] == 1
    poll = page.front.poll(0, timeout=0)
    assert poll["turn"]["state"] == "queued" and poll["turn"]["ahead"] == 1
    assert not page.answers(b)
    gate.set()
    page.done(a)
    page.done(b)
    assert [text for text, _ in page.requests] == ["A", "B"]


def test_pending_confirmation_is_session_owned_and_releases_cap(page: Harness) -> None:
    config = open_config().load()
    config.tuning.chat_turns = 1
    open_config().save(config)
    page.outcomes["A"] = {"type": "command", "name": "tuning", "args": {"max_subtasks": 3}}
    a = page.send("A")
    page.wait(lambda: page.runners.status(a)["state"] == "pending")
    mailbox = page.runners.mailbox(a)
    assert mailbox is not None and mailbox.prompt is not None
    prompt = dict(mailbox.prompt)
    page.new()
    b = page.send("B")
    page.done(b)
    assert page.answers(b) == ["reply:B"]
    assert page.front.poll(0, timeout=0)["pending"] is None
    assert not page.front.answer({"id": prompt["id"], "value": True})
    selected = page.sessions.get(a)
    assert selected is not None
    with ui.use(page.front):
        page.chat._open_session(selected)
    assert page.front.poll(0, timeout=0)["pending"]["id"] == prompt["id"]
    assert page.front.answer({"id": prompt["id"], "value": True})
    page.done(a)
    assert not page.front.answer({"id": prompt["id"], "value": True})
    assert open_config().load().tuning.max_subtasks == 3
    rows = page.conn.execute(
        "SELECT * FROM operator_log WHERE session_id = ? AND action = 'operator_command'", (a,)
    ).fetchall()
    assert len(rows) == 1 and json.loads(rows[0]["details_json"])["command"] == "tuning"


def test_stop_drops_late_reply_and_clears_only_a_queue(page: Harness) -> None:
    gate = page.gate("A1")
    a = page.send("A1")
    assert page.entered["A1"].wait(10)
    page.send("A2")
    page.front.stop()
    page.new()
    b = page.send("B")
    page.done(b)
    gate.set()
    page.done(a)
    assert not page.answers(a) and page.answers(b) == ["reply:B"]
    assert [text for text, _ in page.requests] == ["A1", "B"]
    assert any("reply that came later was dropped" in m.text for m in page.sessions.messages(a))
    # A later turn gets a fresh cancellation event.
    selected = page.sessions.get(a)
    assert selected is not None
    with ui.use(page.front):
        page.chat._open_session(selected)
    page.send("A3")
    page.done(a)
    assert page.answers(a) == ["reply:A3"]
    request = next(data for text, data in page.requests if text == "A3")
    assert all(m["text"] not in {"A1", "A2"} for m in request["conversation"])
    assert not page.sessions.pending_turns()
    stopped = [m for m in page.sessions.messages(a) if m.cancelled_at]
    assert [m.text for m in stopped] == ["A1", "A2"]
    assert "Stopped" in "".join(page.front.poll(0, timeout=0)["transcript"])


def test_background_reply_unread_clears_on_open(page: Harness) -> None:
    gate = page.gate("B")
    b = page.send("B")
    assert page.entered["B"].wait(10)
    page.new()
    a = page.send("A")
    page.done(a)
    gate.set()
    page.done(b)
    data = snapshot(page.path, "sessions", page.front)
    assert next(s for s in data["sessions"] if s["id"] == b)["unread"]
    selected = page.sessions.get(b)
    assert selected is not None
    with ui.use(page.front):
        page.chat._open_session(selected)
    data = snapshot(page.path, "sessions", page.front)
    assert not next(s for s in data["sessions"] if s["id"] == b)["unread"]
    poll = page.front.poll(0, timeout=0)
    assert "reply:B" in "".join(poll["transcript"])
    assert "reply:A" not in "".join(poll["transcript"])


def test_shutdown_cancels_windows_and_finishes_late_reply(page: Harness) -> None:
    page.outcomes["A"] = {"type": "command", "name": "tuning", "args": {"max_subtasks": 3}}
    a = page.send("A")
    page.wait(lambda: page.runners.status(a)["state"] == "pending")
    page.runners.close()
    assert page.runners.closed and not page.runners.runners
    assert open_config().load().tuning.max_subtasks is None
    page.runners.submit(a, "never")
    assert not page.runners.runners


def test_shutdown_abandons_inflight_without_dropping_durable_reply(page: Harness) -> None:
    gate = page.gate("A")
    a = page.send("A")
    assert page.entered["A"].wait(10)
    page.runners.close(timeout=0)
    assert page.runners.runners
    gate.set()
    page.done(a)
    assert page.answers(a) == ["reply:A"]


def test_schema_seven_upgrade_and_all_chat_turn_bounds() -> None:
    config = Config.model_validate(upgrade({"schema_version": 7, "tuning": {"workers": 2}}))
    assert config.schema_version == 8
    assert config.tuning.workers == 2 and config.tuning.chat_turns is None
    for n in (0, 9, True):
        with pytest.raises(ValueError):
            Tuning(chat_turns=n)
    assert runs.tuned(Tuning(), "chat_turns") == 4
    assert (
        TuningArgs(chat_turns=1, models=None, keys=None).change().applied(Tuning()).chat_turns == 1
    )
    assert Limits(section="limits", name="chat_turns", value=8).value == 8
    open_config().save(make_config(language="en", configured=True))
    result = CliRunner().invoke(cli, ["tuning", "--chat-turns", "2"])
    assert result.exit_code == 0, result.output
    assert open_config().load().tuning.chat_turns == 2


def test_planning_progress_and_settings_remain_in_their_sessions(page: Harness) -> None:
    gate = page.gate("plan")
    page.outcomes["A"] = {"type": "task", "title": "Task A", "goal": "Do task A"}
    a = page.send("A")
    assert page.entered["plan"].wait(10)
    assert "Planning" in page.front.poll(0, timeout=0)["session_live"]
    page.new()
    with ui.use(page.front):
        action = ui.SettingsAction(
            type="settings",
            id="plan-setting",
            change=Limits(section="limits", name="max_parts", value=3),
        )
        page.chat._action(action)
    assert open_config().load().tuning.max_parts == 3
    b = page.send("B")
    page.done(b)
    assert not page.front.poll(0, timeout=0).get("session_live")
    assert not page.conn.execute("SELECT id FROM tasks WHERE session_id = ?", (a,)).fetchall()
    gate.set()
    page.done(a)
    assert len(page.conn.execute("SELECT id FROM tasks WHERE session_id = ?", (a,)).fetchall()) == 1


def test_hidden_input_never_reaches_sessions_events_or_model_requests(page: Harness) -> None:
    secret = "gsk_" + "k" * 48
    page.outcomes["A"] = {"type": "command", "name": "keys add", "args": {"provider": "groq"}}
    a = page.send("A")
    page.wait(lambda: page.runners.status(a)["state"] == "pending")
    mailbox = page.runners.mailbox(a)
    assert mailbox is not None and mailbox.prompt is not None
    prompt = dict(mailbox.prompt)
    assert prompt["kind"] == "secret"
    assert page.front.answer({"id": prompt["id"], "value": secret})
    page.done(a)
    assert any(k.value == secret for k in open_config().load().keys("GROQ_API_KEY"))
    stored = " ".join(row[0] for row in page.conn.execute("SELECT text FROM messages"))
    stored += " ".join(row[0] for row in page.conn.execute("SELECT details_json FROM operator_log"))
    assert secret not in stored + repr(list(page.front.events)) + repr(list(mailbox.events))
    assert all(secret not in request.prompt for _, _, fake in page.models for request in fake.calls)


def test_stop_queued_session_does_not_consume_a_slot(page: Harness) -> None:
    config = open_config().load()
    config.tuning.chat_turns = 1
    open_config().save(config)
    gate = page.gate("A")
    a = page.send("A")
    assert page.entered["A"].wait(10)
    page.new()
    b = page.send("B")
    page.wait(lambda: page.runners.status(b)["state"] == "queued")
    page.front.stop()
    page.done(b)
    gate.set()
    page.done(a)
    assert page.answers(a) == ["reply:A"] and not page.answers(b)
    assert page.runners.active == 0 and not page.runners.tickets


def test_temporary_session_is_erased_after_its_inflight_turn(page: Harness) -> None:
    with ui.use(page.front):
        page.chat._new(temporary=True)
    gate = page.gate("temporary")
    a = page.send("temporary")
    assert page.entered["temporary"].wait(10)
    page.new()
    assert page.sessions.messages(a)
    gate.set()
    page.done(a)
    assert not page.sessions.messages(a)


def test_retelling_is_dispatched_once_and_stop_does_not_restart_it(page: Harness) -> None:
    from desman.core.models import SubtaskSpec, TaskSpec

    a = page.send("A")
    page.done(a)
    task = page.chat.store.create_task(
        TaskSpec(
            goal="Background task",
            subtasks=[SubtaskSpec(description="Do it", acceptance_criteria="Done")],
        ),
        session_id=a,
    )
    page.chat.store.cancel_task(task, "test complete")
    gate = page.gate("retell")
    with ui.use(page.front):
        page.chat._notices()
    assert page.entered["retell"].wait(10)
    # Repeated idle checks cannot create another retelling behind the blocked one.
    with ui.use(page.front):
        page.chat.background_outcomes()
        page.chat.background_outcomes()
    assert page.runners.status(a)["queued"] == 0
    page.front.stop()
    gate.set()
    page.done(a)
    with ui.use(page.front):
        page.chat.background_outcomes()
    assert not page.runners.runners
    assert page.answers(a) == ["reply:A"]
    assert not page.sessions.changed_tasks(a)


def test_raising_cap_starts_the_waiter_without_another_message(page: Harness) -> None:
    config = open_config().load()
    config.tuning.chat_turns = 1
    open_config().save(config)
    gate = page.gate("A")
    a = page.send("A")
    assert page.entered["A"].wait(10)
    page.new()
    b = page.send("B")
    page.wait(lambda: page.runners.status(b)["state"] == "queued")
    config.tuning.chat_turns = 2
    open_config().save(config)
    with ui.use(page.front):
        page.chat._notices()
    page.done(b)
    assert page.answers(b) == ["reply:B"] and not page.answers(a)
    gate.set()
    page.done(a)


def test_concurrent_plain_and_rich_output_have_separate_recordings() -> None:
    import sys
    from threading import Barrier, Thread

    import click

    fronts = [WebFront(), WebFront()]
    barrier = Barrier(3)
    records: dict[int, str] = {}
    failures: list[BaseException] = []
    original = sys.stdout

    def write(index: int) -> None:
        try:
            with ui.use(fronts[index]), fronts[index].capture(), ui.recording() as rec:
                barrier.wait(10)
                click.echo(f"plain-{index}")
                ui.warn(f"rich-{index}")
                records[index] = rec.text
        except BaseException as exc:
            failures.append(exc)

    threads = [Thread(target=write, args=(i,)) for i in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait(10)
    for thread in threads:
        thread.join(10)
        assert not thread.is_alive()
    assert not failures
    assert sys.stdout is original
    for i, front in enumerate(fronts):
        captured = repr(list(front.events))
        assert f"plain-{i}" in captured and f"rich-{i}" in captured
        assert f"plain-{1 - i}" not in captured and f"rich-{1 - i}" not in captured
        assert records[i].count(f"plain-{i}") == 1
        assert records[i].count(f"rich-{i}") == 1
        assert f"plain-{1 - i}" not in records[i] and f"rich-{1 - i}" not in records[i]


def test_claiming_first_slot_wakes_other_eligible_sessions(
    page: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    from desman.gui.runners import runner_chat

    initializing = Event()
    allow_initialization = Event()
    original_factory = runner_chat
    first = True
    waiter = Event()
    original_wait = page.runners.condition.wait_for

    def wait_for(predicate: Any, timeout: float | None = None) -> Any:
        if not predicate():
            waiter.set()
        return original_wait(predicate, timeout)

    monkeypatch.setattr(page.runners.condition, "wait_for", wait_for)

    def factory(sessions: Sessions) -> Any:
        nonlocal first
        with page.runners.condition:
            block = first
            first = False
        if block:
            initializing.set()
            assert allow_initialization.wait(10)
        return original_factory(sessions)

    page.runners.factory = factory
    gate = page.gate("A")
    a = page.send("A")
    assert initializing.wait(10)
    page.new()
    b = page.send("B")
    assert waiter.wait(10)
    try:
        allow_initialization.set()
        assert page.entered["A"].wait(10)
        page.done(b)
        assert page.answers(b) == ["reply:B"] and not page.answers(a)
    finally:
        allow_initialization.set()
        gate.set()
    page.done(a)


def test_failed_runner_initialization_does_not_strand_fifo_tickets(page: Harness) -> None:
    from desman.gui.runners import runner_chat

    def broken(sessions: Sessions) -> Any:
        raise RuntimeError("private details must not be shown")

    page.runners.factory = broken
    a = page.send("A")
    page.done(a)
    assert not page.runners.tickets and page.runners.active == 0
    assert any("RuntimeError" in m.text for m in page.sessions.messages(a))
    assert not any("private details" in m.text for m in page.sessions.messages(a))
    page.runners.factory = runner_chat
    page.new()
    b = page.send("B")
    page.done(b)
    assert page.answers(b) == ["reply:B"]


def test_dispatcher_loop_reads_help_and_b_while_a_is_blocked(page: Harness) -> None:
    lines = iter(["A", "/help", "/new", "B", "/exit"])
    gate = page.gate("A")
    a: str | None = None
    step = 0

    def read() -> str:
        nonlocal step, a
        if step == 1:
            a = page.chat.session_id
            assert page.entered["A"].wait(10)
        if step == 2:
            assert any("help" in event.get("html", "") for event in page.front.events)
        if step == 4:
            assert page.chat.session_id is not None
            page.done(page.chat.session_id)
            assert page.answers(page.chat.session_id) == ["reply:B"]
            assert a is not None and not page.answers(a)
        step += 1
        return next(lines)

    page.chat.read = read
    try:
        with ui.use(page.front), page.front.capture():
            page.chat.run()
    finally:
        gate.set()
    assert step == 5 and a is not None
    page.done(a)


def test_queued_input_is_redacted_before_a_key_can_be_removed(page: Harness) -> None:
    secret = "configured-only<&opaque>credential"
    config = make_config(language="en", configured=True, api_keys={"GROQ_API_KEY": secret})
    config.tuning.chat_turns = 1
    open_config().save(config)
    gate = page.gate("A")
    a = page.send("A")
    assert page.entered["A"].wait(10)
    page.new()
    b = page.send(f"B {secret}")
    page.wait(lambda: page.runners.status(b)["state"] == "queued")
    config.api_keys.clear()
    open_config().save(config)
    gate.set()
    page.done(a)
    page.done(b)
    assert all(secret not in request.prompt for _, _, fake in page.models for request in fake.calls)
    assert secret not in " ".join(m.text for m in page.sessions.messages(b))
    assert page.answers(b)


def test_restart_recovers_queued_inputs_and_waits_for_late_reply(page: Harness) -> None:
    config = open_config().load()
    config.tuning.chat_turns = 1
    open_config().save(config)
    gate = page.gate("A")
    a = page.send("A")
    assert page.entered["A"].wait(10)
    page.new()
    b = page.send("B")
    page.wait(lambda: page.runners.status(b)["state"] == "queued")
    old = page.runners
    old.close(timeout=0)
    front = WebFront()
    new = Runners(page.path, front, cap=1)
    try:
        new.recover(page.sessions)
        gate.set()
        with front.condition:
            assert front.condition.wait_for(lambda: not new.runners, timeout=10)
        assert page.answers(a) == ["reply:A"]
        assert page.answers(b) == ["reply:B"]
        assert [text for text, _ in page.requests] == ["A", "B"]
        new.recover(page.sessions)
        assert not new.runners
    finally:
        gate.set()
        new.close()


def test_clear_persists_screen_boundary_without_changing_context(page: Harness) -> None:
    a = page.send("hello")
    page.done(a)
    with ui.use(page.front):
        assert page.chat._slash("/clear")
    page.send("again")
    page.done(a)
    screen = "".join(page.front.poll(0, timeout=0)["transcript"])
    assert "hello" not in screen and "again" in screen
    assert screen.count("Earlier messages hidden") == 1
    history = page.sessions.history(a)
    assert "hello" in [m.text for m in history.messages]
    request = next(data for text, data in page.requests if text == "again")
    assert {"from": "person", "text": "hello"} in request["conversation"]
    reload = WebFront()
    reload.runners = page.runners
    reload.database = page.path
    reload.session(a, False)
    assert "hello" not in "".join(reload.poll(0, timeout=0)["transcript"])


def test_handed_off_command_saved_once_before_terminal_output(page: Harness) -> None:
    page.send("/run example")
    terminal = page.chat.session_id
    assert terminal is not None
    page.done(terminal)
    messages = page.sessions.messages(terminal)
    assert messages[0].role is Role.USER and messages[0].text == "/run example"
    assert len([m for m in messages if m.role is Role.USER]) == 1
    assert any(m.role is Role.DESMAN for m in messages[1:])
    reload = WebFront()
    reload.runners = page.runners
    reload.database = page.path
    reload.session(terminal, False)
    screen = "".join(reload.poll(0, timeout=0)["transcript"])
    assert screen.count("/run example") == 1


def test_previous_schema_upgrade_preserves_messages_and_recovers_only_live(
    clock: FakeClock,
) -> None:
    conn = db.connect(":memory:")
    try:
        db.migrate(conn, db.MIGRATIONS[:-1])
        conn.execute("INSERT INTO sessions (id, created_at) VALUES ('old', '2026-01-01')")
        conn.execute(
            "INSERT INTO messages (session_id, at, role, kind, text) "
            "VALUES ('old', '2026-01-01', 'user', 'message', 'pending')"
        )
        conn.execute("INSERT INTO chat_turns (user_seq) VALUES (1)")
        db.migrate(conn)
        assert (
            conn.execute("SELECT task_id FROM chat_turns WHERE user_seq = 1").fetchone()[0] is None
        )
        sessions = Sessions(conn, clock=clock)
        pending = sessions.pending_turns()
        assert [m.text for m in pending] == ["pending"]
        sessions.settle_turn(pending[0].seq, cancelled=True)
        assert not sessions.pending_turns() and not sessions.history("old").messages
        assert sessions.messages("old")[0].text == "pending"
    finally:
        conn.close()


def test_old_outcome_does_not_settle_new_input_and_reply_receipt_is_atomic(
    page: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    from desman.chat import Chat
    from desman.gui.runners import runner_chat

    message = page.sessions.add(None, Role.USER, "new input")
    page.sessions.register_turn(message.seq)
    page.sessions.reply_to = message.seq
    chat = runner_chat(page.sessions)
    chat.session_id = message.session_id

    def catch_up(self: Chat, interlocutor: Any) -> None:
        self.sessions.add(message.session_id, Role.INTERLOCUTOR, "old outcome")

    monkeypatch.setattr(Chat, "_catch_up", catch_up)
    chat._catch_up(None)  # type: ignore[arg-type]
    assert not page.sessions.turn_settled(message.seq)
    chat._say("the answer")
    assert page.sessions.turn_settled(message.seq)
    assert not page.sessions.pending_turns()


def test_task_commit_survives_crash_before_start_event(
    page: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    from desman.chat import Chat

    page.outcomes["task input"] = {"type": "task", "title": "One", "goal": "Do it"}
    original = Chat._event

    def crash(self: Chat, text: str, task_id: str | None = None) -> None:
        if task_id is not None:
            raise RuntimeError("crash after task commit")
        original(self, text, task_id)

    monkeypatch.setattr(Chat, "_event", crash)
    session = page.send("task input")
    page.done(session)
    tasks = page.conn.execute("SELECT id FROM tasks WHERE session_id = ?", (session,)).fetchall()
    assert len(tasks) == 1
    receipt = page.conn.execute("SELECT task_id, settled_at FROM chat_turns").fetchone()
    assert receipt["task_id"] == tasks[0]["id"] and receipt["settled_at"] is not None
    page.runners.recover(page.sessions)
    page.done(session)
    assert (
        len(page.conn.execute("SELECT id FROM tasks WHERE session_id = ?", (session,)).fetchall())
        == 1
    )
    assert [text for text, _ in page.requests] == ["task input"]


def test_summary_commit_does_not_answer_pending_turn(page: Harness) -> None:
    from desman.core.models import SubtaskSpec, TaskSpec
    from desman.core.sessions import Kind
    from desman.gui.runners import runner_chat

    internal_task = page.chat.store.create_task(
        TaskSpec(
            goal="Summarise",
            subtasks=[SubtaskSpec(description="Summarise", acceptance_criteria="Summary")],
        )
    )
    user = page.sessions.add(None, Role.USER, "still needs an answer")
    page.sessions.register_turn(user.seq)
    sessions = runner_chat(page.sessions).sessions
    sessions.reply_to = user.seq
    sessions.add(
        user.session_id,
        Role.DESMAN,
        "summary",
        kind=Kind.SUMMARY,
        covers=user.seq,
        task_id=internal_task,
    )
    sessions.add(
        user.session_id, Role.DESMAN, "internal task output", kind=Kind.EVENT, task_id=internal_task
    )
    assert not sessions.turn_settled(user.seq)
    page.runners.recover(page.sessions)
    page.done(user.session_id)
    assert page.answers(user.session_id) == ["reply:still needs an answer"]
    page.runners.recover(page.sessions)
    assert [text for text, _ in page.requests] == ["still needs an answer"]


def test_stop_after_reply_commit_preserves_question(
    page: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    from desman.chat import Chat

    committed = Event()
    release = Event()
    original = Chat._say

    def reply(self: Chat, text: str, task_id: str | None = None) -> None:
        original(self, text, task_id)
        if text == "reply:answered question":
            committed.set()
            assert release.wait(10)

    monkeypatch.setattr(Chat, "_say", reply)
    session = page.send("answered question")
    assert committed.wait(10)
    try:
        page.runners.stop(session)
    finally:
        release.set()
    page.done(session)
    assert page.sessions.messages(session)[0].cancelled_at is None
    page.send("next question")
    page.done(session)
    request = next(data for text, data in page.requests if text == "next question")
    assert {"from": "person", "text": "answered question"} in request["conversation"]
    assert {"from": "you", "text": "reply:answered question"} in request["conversation"]


def test_turn_lock_entries_are_released(tmp_path: Path) -> None:
    from desman.gui.runners import _turn_locks, turn_lock

    for seq in range(100):
        with turn_lock((tmp_path, seq)):
            assert (tmp_path, seq) in _turn_locks
        assert (tmp_path, seq) not in _turn_locks
    with pytest.raises(RuntimeError), turn_lock((tmp_path, 101)):
        raise RuntimeError("failed turn")
    assert (tmp_path, 101) not in _turn_locks


def test_command_explanation_does_not_settle_until_result(page: Harness) -> None:
    from desman.gui.runners import runner_chat

    user = page.sessions.add(None, Role.USER, "change tuning")
    page.sessions.register_turn(user.seq)
    page.sessions.reply_to = user.seq
    chat = runner_chat(page.sessions)
    chat.session_id = user.session_id
    chat._say("I will change the tuning", settle=False)
    assert not page.sessions.turn_settled(user.seq)
    chat._result("Declined", None, None)
    assert page.sessions.turn_settled(user.seq)
