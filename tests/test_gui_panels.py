"""Panels read committed snapshots and return actions to the chat's open line only."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from html import escape, unescape
from pathlib import Path
from typing import Any

import pytest
from rich.panel import Panel
from rich.text import Text

from desman import providers, ui, wizard
from desman.chat import Chat
from desman.config import open_config
from desman.core import calls, db
from desman.core.models import SubtaskSpec, TaskSpec
from desman.core.report import inbox
from desman.core.sessions import Role, Sessions
from desman.core.states import Actor, SubtaskStatus
from desman.core.store import Store
from desman.gui.front import WebFront
from desman.gui.server import Server
from desman.llm.openai_compat import key_id
from helpers import make_config
from test_gui import headers
from test_live import call


@pytest.fixture
def panel_app(tmp_path: Path) -> Iterator[tuple[Server, Store]]:
    path = tmp_path / "panels.db"
    conn = db.open_db(path)
    app = Server(WebFront(), 12345, poll_timeout=0, database=path)
    yield app, Store(conn)
    conn.close()


@pytest.mark.parametrize("path", ["/sessions", "/tasks", "/action"])
def test_panel_transport_guards(panel_app: tuple[Server, Store], path: str) -> None:
    app, _ = panel_app
    for method in ["GET", "POST"]:
        assert app.handle(method, path, headers(app, Cookie=""), b"{}").status == 403
        assert app.handle(method, path, headers(app, Host="evil"), b"{}").status == 403
    assert app.handle("POST", path, headers(app, Origin="evil"), b"{}").status == 403
    assert (
        app.handle("POST", path, headers(app, **{"Content-Type": "text/plain"}), b"{}").status
        == 415
    )


def test_action_validation_and_line_gate(panel_app: tuple[Server, Store]) -> None:
    app, _ = panel_app

    def send(data: dict[str, Any]) -> int:
        return app.handle("POST", "/action", headers(app), json.dumps(data).encode()).status

    bad_actions: list[dict[str, Any]] = [
        {},
        {"type": "unknown"},
        {"type": "settings", "extra": 1},
        {"type": "new_session", "temporary": "false"},
        {"type": "decide", "ref": "a/1", "version": True, "decision": "accept", "note": ""},
        {"type": "decide", "ref": "a/1", "version": 1, "decision": "fail", "note": " "},
    ]
    for bad in bad_actions:
        assert send(bad) == 400
    assert send({"type": "settings"}) == 409
    got: list[Any] = []
    thread = threading.Thread(target=lambda: got.append(app.front.line()))
    thread.start()
    with app.front.condition:
        assert app.front.condition.wait_for(lambda: app.front.prompt is not None, timeout=2)
    assert send({"type": "settings"}) == 200
    assert send({"type": "settings"}) == 409
    thread.join(2)
    assert got == [ui.SimpleAction(type="settings")]


def test_greeting_reset_and_replay() -> None:
    front = WebFront()
    with front.greeting():
        front.note("hello")
    assert front.poll(0, 0)["events"][0]["greeting"] is True
    thread = threading.Thread(target=front.line)
    thread.start()
    with front.condition:
        assert front.condition.wait_for(lambda: front.prompt is not None, timeout=2)
        assert front.prompt is not None
        id = front.prompt["id"]
    assert front.answer({"id": id, "value": "first"})
    thread.join(2)
    events = front.poll(0, 0)["events"]
    # Reload replays both the greeting and its removal trigger, in order.
    assert events[-1]["kind"] == "line" and events[-1]["echo"] == "first"
    front.reset()
    assert [e["type"] for e in front.poll(0, 0)["events"]] == ["reset"]


def waiting(store: Store, goal: str = "goal") -> str:
    task = store.create_task(
        TaskSpec(goal=goal, subtasks=[SubtaskSpec(description=goal, acceptance_criteria="done")])
    )
    sub = store.claim("test", task_id=task)
    assert sub is not None
    store.transition(
        sub.id,
        SubtaskStatus.RUNNING,
        SubtaskStatus.NEEDS_HUMAN,
        Actor.EXECUTOR,
        lease=sub.lease_token,
        reason=goal,
    )
    return task


def test_snapshots_are_redacted_and_read_only(panel_app: tuple[Server, Store]) -> None:
    app, store = panel_app
    key = "gsk_" + "a" * 48
    open_config().save(make_config(api_keys={"GROQ_API_KEY": key}))
    task = waiting(store, key)
    sub = store.list_subtasks(task)[0]
    with calls.for_(calls.Purpose(calls.Role.EXECUTOR, task_id=task, subtask_id=sub.id)):
        calls.recorder(store.conn)(call(key_id=key_id(key)))
    sessions = Sessions(store.conn)
    session = sessions.add(None, Role.USER, key).session_id
    sessions.pin(session, True)
    app.front.session(session, False)
    before = store.conn.total_changes
    for path in ["/sessions", "/tasks"]:
        response = app.handle("GET", path, headers(app))
        assert response.status == 200
        assert key.encode() not in response.body
    listed = json.loads(app.handle("GET", "/sessions", headers(app)).body)
    assert listed["current"] == session and listed["sessions"][0]["pinned"]
    tasks = json.loads(app.handle("GET", "/tasks", headers(app)).body)["tasks"]
    assert tasks[0]["id"] == task and tasks[0]["needs_human"]
    module = tasks[0]["modules"][0]
    assert module["calls"] == 1 and module["subtask"] == sub.seq
    assert module["key"] and key not in module["key"]
    assert tasks[0]["subtasks"][0]["version"] == inbox(store)[0].version
    assert store.conn.total_changes == before


def test_session_actions_and_confirmation(
    panel_app: tuple[Server, Store], monkeypatch: pytest.MonkeyPatch
) -> None:
    app, store = panel_app
    sessions = Sessions(store.conn)
    id = sessions.add(None, Role.USER, "hello").session_id
    chat = Chat(sessions)
    with ui.use(app.front):
        chat._action(ui.SessionAction(type="open_session", id=id))
        assert chat.session_id == id and app.front.current == id
        assert app.front.events[0]["type"] == "reset"
        chat._action(ui.SessionAction(type="pin", id=id))
        assert sessions.get(id).pinned  # type: ignore[union-attr]
        chat._action(ui.SessionAction(type="unpin", id=id))
        assert not sessions.get(id).pinned  # type: ignore[union-attr]
        monkeypatch.setattr(ui, "confirm", lambda *a, **kw: False)
        chat._action(ui.SessionAction(type="delete_session", id=id))
        assert sessions.get(id) is not None
        monkeypatch.setattr(ui, "confirm", lambda *a, **kw: True)
        chat._action(ui.SessionAction(type="delete_session", id=id))
        assert sessions.get(id) is None and chat.session_id is None
        chat._action(ui.SessionAction(type="open_session", id=id))
        assert chat.session_id is None
        chat._action(ui.NewSessionAction(type="new_session", temporary=True))
        assert chat.temporary
        chat._action(ui.NewSessionAction(type="new_session", temporary=False))
        assert not chat.temporary


@pytest.mark.parametrize(
    "decision,target", [("retry", "ready"), ("accept", "done"), ("fail", "failed")]
)
def test_decisions_are_typed_version_checked(
    panel_app: tuple[Server, Store], decision: str, target: str
) -> None:
    app, store = panel_app
    open_config().save(make_config())
    task = waiting(store)
    item = inbox(store)[0]
    chat = Chat(Sessions(store.conn))
    with ui.use(app.front):
        chat._action(
            ui.ACTIONS.validate_python(
                {
                    "type": "decide",
                    "ref": item.ref,
                    "version": item.version + 1,
                    "decision": decision,
                    "note": "answer",
                }
            )
        )
        assert store.list_subtasks(task)[0].status == SubtaskStatus.NEEDS_HUMAN
        assert "changed since version" in str(app.front.events)
        chat._action(
            ui.ACTIONS.validate_python(
                {
                    "type": "decide",
                    "ref": item.ref,
                    "version": item.version,
                    "decision": decision,
                    "note": "answer",
                }
            )
        )
    assert store.list_subtasks(task)[0].status.value == target
    rows = store.conn.execute("SELECT details_json FROM operator_log").fetchall()
    assert any(json.loads(r[0]).get("typed") is True for r in rows)


def test_simple_actions_reuse_chat_commands(
    panel_app: tuple[Server, Store], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, store = panel_app
    chat = Chat(Sessions(store.conn))
    seen: list[str] = []
    monkeypatch.setattr(chat, "_typed", seen.append)
    monkeypatch.setattr(chat, "_slash", seen.append)
    monkeypatch.setattr(chat, "_delete_all", lambda: seen.append("delete_all"))
    for type in ["settings", "messenger", "delete_all"]:
        chat._action(ui.ACTIONS.validate_python({"type": type}))
    assert seen == ["config", "/messages", "delete_all"]


def test_switch_shows_200_messages_and_erases_temporary(panel_app: tuple[Server, Store]) -> None:
    app, store = panel_app
    sessions = Sessions(store.conn)
    id: str | None = None
    for index in range(205):
        id = sessions.add(id, Role.USER, f"message-{index}").session_id
    assert id is not None
    temporary = sessions.add(None, Role.USER, "private", temporary=True).session_id
    chat = Chat(sessions, temporary=True)
    chat.session_id = temporary
    with ui.use(app.front):
        chat._action(ui.SessionAction(type="open_session", id=id))
    assert len([e for e in app.front.events if "message-" in e.get("html", "")]) == 200
    assert sessions.messages(temporary) == []
    assert chat.session_id == id and not chat.temporary


def test_delete_all_keeps_pinned_and_resets_current(
    panel_app: tuple[Server, Store], monkeypatch: pytest.MonkeyPatch
) -> None:
    app, store = panel_app
    sessions = Sessions(store.conn)
    pinned = sessions.add(None, Role.USER, "keep").session_id
    sessions.pin(pinned, True)
    current = sessions.add(None, Role.USER, "remove").session_id
    chat = Chat(sessions)
    chat.session_id = current
    monkeypatch.setattr(ui, "confirm", lambda *a, **kw: True)
    with ui.use(app.front):
        chat._action(ui.SimpleAction(type="delete_all"))
    assert sessions.get(current) is None and sessions.get(pinned) is not None
    assert chat.session_id is None and app.front.current is None
    assert any(e["type"] == "reset" for e in app.front.events)


def test_resumed_transcript_is_not_greeting(panel_app: tuple[Server, Store]) -> None:
    app, store = panel_app
    sessions = Sessions(store.conn)
    sessions.add(None, Role.USER, "kept message")
    chat = Chat(sessions, read=lambda: None, resume=True)
    with ui.use(app.front):
        chat.run()
    message = next(e for e in app.front.events if "kept message" in e.get("html", ""))
    assert not message.get("greeting")


def test_settings_action_enters_wizard_without_extra_confirmation(
    panel_app: tuple[Server, Store], monkeypatch: pytest.MonkeyPatch
) -> None:
    app, store = panel_app
    open_config().save(make_config())
    seen: list[str] = []
    monkeypatch.setattr(wizard, "run", lambda manager: seen.append("wizard"))
    with ui.use(app.front):
        Chat(Sessions(store.conn))._action(ui.SimpleAction(type="settings"))
    assert seen == ["wizard"]


def test_actions_cannot_answer_a_confirmation(panel_app: tuple[Server, Store]) -> None:
    app, _ = panel_app
    got: list[bool] = []
    thread = threading.Thread(target=lambda: got.append(app.front.confirm("Delete?")))
    thread.start()
    with app.front.condition:
        assert app.front.condition.wait_for(lambda: app.front.prompt is not None, timeout=2)
        assert app.front.prompt is not None
        id = app.front.prompt["id"]
    assert app.handle("POST", "/action", headers(app), b'{"type":"delete_all"}').status == 409
    assert app.front.answer({"id": id, "value": False})
    thread.join(2)
    assert got == [False]


@pytest.mark.parametrize("key", ["gsk_" + "b" * 48, "configured&<key>\"'value"])
def test_opened_transcript_also_redacts_configured_keys(
    panel_app: tuple[Server, Store], key: str
) -> None:
    app, store = panel_app
    open_config().save(make_config(api_keys={"GROQ_API_KEY": key}))
    sessions = Sessions(store.conn)
    id = sessions.add(None, Role.USER, key).session_id
    with ui.use(app.front):
        Chat(sessions)._action(ui.SessionAction(type="open_session", id=id))
    data = app.front.poll(0, 0)
    response = app.handle("GET", "/events?after=0", headers(app))
    assert response.status == 200
    for output in (json.dumps(data), response.body.decode()):
        assert key not in unescape(output)
        assert escape(key) not in output
    assert providers.REDACTED in str(data)


@pytest.mark.parametrize("failed", [False, True])
def test_tasks_keep_old_inbox_items(panel_app: tuple[Server, Store], failed: bool) -> None:
    app, store = panel_app
    oldest = waiting(store, "oldest waiting")
    if failed:
        store.transition(
            store.list_subtasks(oldest)[0].id,
            SubtaskStatus.NEEDS_HUMAN,
            SubtaskStatus.FAILED,
            Actor.HUMAN,
            reason="operator failed",
        )
    newer = [
        store.create_task(
            TaskSpec(
                goal=str(i), subtasks=[SubtaskSpec(description="new", acceptance_criteria="ok")]
            )
        )
        for i in range(30)
    ]
    tasks = json.loads(app.handle("GET", "/tasks", headers(app)).body)["tasks"]
    assert [task["id"] for task in tasks] == [*reversed(newer), oldest]
    old = tasks[-1]
    item = inbox(store)[0]
    assert old["needs_human"]
    assert old["subtasks"][0]["ref"] == item.ref
    assert old["subtasks"][0]["version"] == item.version


@pytest.mark.parametrize("wrapped", [False, True])
def test_output_and_live_redact_html_keys(panel_app: tuple[Server, Store], wrapped: bool) -> None:
    app, _ = panel_app
    key = "configured&<key>\"'value"
    open_config().save(make_config(api_keys={"GROQ_API_KEY": key}))
    message = Text("message " + key)
    app.front.out(Panel(message) if wrapped else message)
    app.front.out("plain " + key)
    app.front.live_update(Panel(message))
    response = app.handle("GET", "/events?after=0", headers(app))
    assert response.status == 200
    data = json.loads(response.body)
    for output in [*(e.get("html", "") for e in data["events"]), data["live"]]:
        assert key not in unescape(output)
        assert escape(key) not in output
        assert providers.REDACTED in output
