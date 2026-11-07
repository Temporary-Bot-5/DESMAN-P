"""Module cards use safe snapshots and the same change bodies as the CLI."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Literal

import pytest
from pydantic import ValidationError

from desman import modules_cli, tuning_cli, ui
from desman.chat import Chat
from desman.config import open_config
from desman.core import calls, choices, db
from desman.core.models import SubtaskSpec, TaskSpec
from desman.core.sessions import Role, Sessions
from desman.core.store import Store
from desman.gui.front import WebFront
from desman.gui.server import Server, strings
from desman.llm.openai_compat import key_id
from helpers import make_config
from test_gui import headers
from test_live import call


@pytest.fixture
def module_app(tmp_path: Path) -> Any:
    path = tmp_path / "modules.db"
    conn = db.open_db(path)
    app = Server(WebFront(), 12345, poll_timeout=0, database=path)
    yield app, Store(conn)
    conn.close()


@pytest.mark.parametrize("name", ["logo", "banner", "module"])
def test_pixel_assets(module_app: Any, name: str) -> None:
    app, _ = module_app
    response = app.handle("GET", f"/img/{name}.png", headers(app))
    assert response.status == 200
    assert response.headers["Content-Type"] == "image/png"
    assert response.body.startswith(b"\x89PNG\r\n\x1a\n")
    assert "default-src 'self'" in response.headers["Content-Security-Policy"]


def test_module_snapshot_and_session_usage(module_app: Any) -> None:
    app, store = module_app
    secret = 'configured&<key>"value'
    config = make_config(api_keys={"GROQ_API_KEY": secret})
    open_config().save(config)
    task = store.create_task(
        TaskSpec(goal=secret, subtasks=[SubtaskSpec(description="go", acceptance_criteria="ok")])
    )
    sessions = Sessions(store.conn)
    session = sessions.add(None, Role.USER, secret).session_id
    model = tuning_cli.role_rows(config)[0]["candidate_models"][0]
    modules_cli.set_module(store.conn, config, "session", session, "interlocutor", model, "groq:#1")
    app.front.session(session, False)
    with calls.for_(calls.Purpose(calls.Role.INTERLOCUTOR, session_id=session)):
        calls.recorder(store.conn)(call(key_id=key_id(secret)))
    before = store.conn.total_changes
    for path in ["/modules", "/sessions", "/tasks", "/strings.json", "/events?after=0"]:
        response = app.handle("GET", path, headers(app))
        assert response.status == 200
        assert secret not in response.body.decode()

        def check(value: Any) -> None:
            if isinstance(value, str):
                assert secret not in value
            elif isinstance(value, dict):
                for item in value.values():
                    check(item)
            elif isinstance(value, list):
                for item in value:
                    check(item)

        check(json.loads(response.body))
    data = json.loads(app.handle("GET", "/modules", headers(app)).body)
    assert data["modules"][0]["task_id"] == task
    assert {r["role"] for r in data["modules"]} == {"text", "reviewer"}
    card = data["interlocutor"]
    assert card["source"] == "chat" and card["scope_id"] == session
    assert card["calls"] == 1 and card["tokens"] > 0
    assert card["key_fingerprint"] == key_id(secret)
    assert card["icon"] == "module"
    assert data["roles"][0]["candidate_keys"][0]["fingerprint"] == key_id(secret)
    assert store.conn.total_changes == before
    app.front.session(None, False)
    fresh = json.loads(app.handle("GET", "/modules", headers(app)).body)["interlocutor"]
    assert fresh["scope_id"] is None and fresh["calls"] == 0


def payload(**fields: Any) -> dict[str, Any]:
    return (
        dict(type="module", role="text", scope="task", scope_id="task-id", model=None, key=None)
        | fields
    )


@pytest.mark.parametrize(
    "fields",
    [
        dict(extra=1),
        dict(scope="global"),
        dict(role="unknown"),
        dict(key="secret"),
        dict(scope="role"),
        dict(scope="session", role="text"),
        dict(scope_id=None),
    ],
)
def test_module_action_is_strict(module_app: Any, fields: dict[str, Any]) -> None:
    app, _ = module_app
    assert (
        app.handle("POST", "/action", headers(app), json.dumps(payload(**fields)).encode()).status
        == 400
    )
    with pytest.raises(ValidationError):
        ui.ACTIONS.validate_python(payload(**fields))


def test_module_action_line_gate(module_app: Any) -> None:
    app, _ = module_app
    body = json.dumps(payload()).encode()
    assert app.handle("POST", "/action", headers(app), body).status == 409
    received: list[Any] = []
    thread = threading.Thread(target=lambda: received.append(app.front.line()))
    thread.start()
    with app.front.condition:
        assert app.front.condition.wait_for(lambda: app.front.prompt is not None, timeout=2)
    assert app.handle("POST", "/action", headers(app), body).status == 200
    assert app.handle("POST", "/action", headers(app), body).status == 409
    thread.join(2)
    assert isinstance(received[0], ui.ModuleAction)


@pytest.mark.parametrize("scope", ["task", "session", "role"])
def test_module_action_uses_bodies_and_console(
    module_app: Any, monkeypatch: pytest.MonkeyPatch, scope: Literal["task", "session", "role"]
) -> None:
    app, store = module_app
    secret = "gsk_" + "a" * 48
    config = make_config(api_keys={"GROQ_API_KEY": secret})
    open_config().save(config)
    task = store.create_task(
        TaskSpec(goal="work", subtasks=[SubtaskSpec(description="go", acceptance_criteria="ok")])
    )
    sessions = Sessions(store.conn)
    session = sessions.add(None, Role.USER, "hello").session_id
    chat = Chat(sessions)
    chat.session_id = session
    role = "interlocutor" if scope == "session" else "text"
    model = next(r for r in tuning_cli.role_rows(config) if r["role"] == role)["candidate_models"][
        0
    ]
    identity = task if scope == "task" else session if scope == "session" else None
    seen: list[str] = []
    original = modules_cli.set_module

    def set_module(*args: Any, **kwargs: Any) -> Any:
        seen.append("modules")
        return original(*args, **kwargs)

    original_tuning = tuning_cli.set_tuning

    def set_tuning(*args: Any, **kwargs: Any) -> Any:
        seen.append("tuning")
        return original_tuning(*args, **kwargs)

    monkeypatch.setattr(modules_cli, "set_module", set_module)
    monkeypatch.setattr(tuning_cli, "set_tuning", set_tuning)
    action = ui.ACTIONS.validate_python(
        payload(role=role, scope=scope, scope_id=identity, model=model, key=key_id(secret))
    )
    with ui.use(app.front):
        ui.session(session, False)
        chat._action(action)
    assert seen == (["tuning"] if scope == "role" else ["modules"])
    if scope == "role":
        saved = open_config().load()
        assert saved.tuning.models.text == model and saved.tuning.keys.text == key_id(secret)
    else:
        assert scope in ("task", "session") and identity is not None
        choice = choices.get(store.conn, scope, identity, role)
        assert choice.model == model and choice.key == key_id(secret)
    assert chat.session_id == session and not chat.in_terminal
    current = json.loads(app.handle("GET", "/sessions", headers(app)).body)
    assert current["current"] == session and current["terminal"]["unread"] == 1
    polled = json.loads(app.handle("GET", "/modules", headers(app)).body)
    updated = (
        polled["interlocutor"]
        if scope == "session"
        else next(
            row for row in polled["modules"] if row["task_id"] == task and row["role"] == role
        )
    )
    assert updated["model"] == model and updated["key_fingerprint"] == key_id(secret)
    assert any(m.kind.value == "event" for m in sessions.messages(sessions.terminal()))
    assert store.conn.execute("SELECT COUNT(*) FROM operator_log").fetchone()[0] > 0
    assert secret not in str(app.front.events)


def test_other_session_cannot_be_edited(module_app: Any) -> None:
    app, store = module_app
    open_config().save(make_config())
    sessions = Sessions(store.conn)
    other = sessions.add(None, Role.USER, "other").session_id
    chat = Chat(sessions)
    with ui.use(app.front):
        chat._action(
            ui.ACTIONS.validate_python(
                payload(role="interlocutor", scope="session", scope_id=other)
            )
        )
    assert choices.get(store.conn, "session", other, "interlocutor") == choices.Choice()
    assert not chat.in_terminal


def test_module_page_strings() -> None:
    assert all(
        strings()[key]
        for key in [
            "change",
            "only_here",
            "everywhere",
            "apply",
            "scope",
            "auto",
            "role_text",
            "role_code",
            "role_analysis",
            "role_interlocutor",
        ]
    )


def test_modules_endpoint_requires_authentication(module_app: Any) -> None:
    app, _ = module_app
    assert app.handle("GET", "/modules", headers(app, Cookie="")).status == 403
    assert app.handle("GET", "/modules", headers(app, Host="evil")).status == 403


def test_unavailable_choice_is_refused(module_app: Any) -> None:
    app, store = module_app
    open_config().save(make_config())
    chat = Chat(Sessions(store.conn))
    with ui.use(app.front):
        chat._action(
            ui.ACTIONS.validate_python(
                payload(scope="role", scope_id=None, model="groq/unavailable")
            )
        )
    assert open_config().load().tuning.models.text is None
    assert store.conn.execute("SELECT COUNT(*) FROM module_choices").fetchone()[0] == 0
    assert "Invalid module choice" in "\n".join(
        m.text for m in chat.sessions.messages(chat.sessions.terminal())
    )


def test_interlocutor_inherits_role_until_session_override(module_app: Any) -> None:
    app, store = module_app
    config = make_config(api_keys={"GROQ_API_KEY": "gsk_" + "b" * 48})
    models = next(r for r in tuning_cli.role_rows(config) if r["role"] == "interlocutor")[
        "candidate_models"
    ]
    config.tuning.models.interlocutor = models[0]
    config.tuning.keys.interlocutor = key_id(config.keys("GROQ_API_KEY")[0].value)
    open_config().save(config)
    session = Sessions(store.conn).add(None, Role.USER, "hello").session_id
    app.front.session(session, False)

    def card() -> Any:
        return json.loads(app.handle("GET", "/modules", headers(app)).body)["interlocutor"]

    assert card()["source"] == "role" and card()["model"] == models[0]
    modules_cli.set_module(store.conn, config, "session", session, "interlocutor", models[-1])
    assert card()["source"] == "chat" and card()["model"] == models[-1]
    assert card()["key_fingerprint"] == config.tuning.keys.interlocutor
    modules_cli.set_module(store.conn, config, "session", session, "interlocutor", "auto", "auto")
    assert card()["source"] == "role"
