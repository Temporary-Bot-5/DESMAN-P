"""Usage windows, explicit replacement recovery and cancellation keep the page usable."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import pytest

from desman import modules_cli, providers, runs, ui
from desman.chat import Chat
from desman.config import open_config
from desman.core import calls, choices, db, holds
from desman.core.models import Kind, SubtaskSpec, TaskSpec
from desman.core.sessions import Role, Sessions
from desman.gui.front import WebFront
from desman.gui.module_view import cards
from desman.gui.server import Server
from desman.llm import router
from desman.llm.openai_compat import key_id
from helpers import make_config
from test_gui import headers, roundtrip
from test_live import call

SECRET = "t2n-private-key-with-a-long-value"


@pytest.fixture
def page(tmp_path: Path) -> Any:
    path = tmp_path / "page.db"
    conn = db.open_db(path)
    cfg = make_config(language="en", configured=True, api_keys={"GROQ_API_KEY": SECRET})
    open_config().save(cfg)
    app = Server(WebFront(), 12345, poll_timeout=0, database=path)
    chat = Chat(Sessions(conn))
    identity = chat.store.create_task(
        TaskSpec(
            goal="Active work",
            subtasks=[SubtaskSpec(description="Write", acceptance_criteria="ok")],
        )
    )
    yield app, chat, cfg, identity
    conn.close()


def test_usage_json_masks_removed_and_current_keys_and_limits(page: Any) -> None:
    app, chat, cfg, _ = page
    assert json.loads(app.handle("GET", "/usage", headers(app)).body)["rows"] == []
    record = calls.recorder(chat.store.conn)
    zero = {"x-ratelimit-limit-requests": "0", "x-ratelimit-remaining-requests": "0"}
    record(call(key_id=key_id(SECRET), limits=zero))
    record(call(key_id="0123456789ab", limits=zero))
    record(call(key_id="0123456789ab"))
    for since in ("24h", "7d", "30d", "all"):
        response = app.handle("GET", "/usage?since=" + since, headers(app))
        assert response.status == 200 and SECRET not in response.body.decode()
        data = json.loads(response.body)
        assert data["removed"]
        assert [r["calls"] for r in data["rows"]] == [1, 2]
        assert data["rows"][0]["key"].startswith("#1 ")
        assert data["rows"][1]["key"] == "removed (0123…)"
        assert data["limits"][0]["exhausted_for_good"]
        assert not data["limits"][1]["limits_current"]
        assert not data["limits"][1]["exhausted_for_good"]
    for since in ("", "bad", "0h", "999999999999999999999999999d"):
        assert app.handle("GET", "/usage?since=" + since, headers(app)).status == 400
    data = json.loads(app.handle("GET", "/modules", headers(app)).body)
    assert "removed" not in data


@pytest.mark.parametrize("entry", ["card", "settings"])
@pytest.mark.parametrize("key", [None, key_id(SECRET)])
def test_choice_restores_removed_sources_and_journals_but_preserves_stops(
    page: Any, entry: str, key: str | None
) -> None:
    app, chat, cfg, identity = page
    model = router.lineup(cfg, calls.Role.EXECUTOR, kind=Kind.TEXT)[0].label
    source = key_id(SECRET)
    removed = holds.Hold(source=source, model=model)
    stopped = holds.Hold(source=source, model=model, scope="task", scope_id=identity)
    holds.remove(chat.store.conn, removed)
    other = holds.Hold(source="a" * 12, model=model)
    holds.remove(chat.store.conn, other)
    holds.stop(chat.store.conn, stopped)
    with ui.use(app.front):
        if entry == "card":
            action: ui.Action = ui.ModuleAction(
                type="module", role="text", model=model, key=key, scope="task", scope_id=identity
            )
        else:
            action = ui.ACTIONS.validate_python(
                {
                    "type": "settings",
                    "id": "restore",
                    "change": {"section": "roles", "role": "text", "model": model, "key": key},
                }
            )
        chat._action(action)
    remaining = holds.listed(chat.store.conn)
    assert any(h.state == "stopped" for h in remaining)
    assert [h.source for h in remaining if h.state == "removed"] == ([other.source] if key else [])
    rows = chat.store.conn.execute(
        "SELECT details_json FROM operator_log WHERE action='operator_module_restore'"
    ).fetchall()
    assert len(rows) == (1 if key else 2)
    assert all(json.loads(row[0])["model"] == model for row in rows)


def test_local_choice_restores_only_matching_scopes_and_sources(page: Any) -> None:
    app, chat, cfg, identity = page
    session = chat.sessions.add(None, Role.USER, "hello").session_id
    chat.session_id = session
    model = router.lineup(cfg, calls.Role.INTERLOCUTOR)[0].label
    scopes: list[tuple[holds.Scope, str]] = [("all", ""), ("session", session), ("task", identity)]
    for scope, scope_id in scopes:
        holds.remove(
            chat.store.conn,
            holds.Hold(source=key_id(SECRET), model=model, scope=scope, scope_id=scope_id),
        )
    with ui.use(app.front):
        chat._action(
            ui.ModuleAction(
                type="module",
                role="interlocutor",
                scope="session",
                scope_id=session,
                model=model,
                key=key_id(SECRET),
            )
        )
    assert [(h.scope, h.scope_id) for h in holds.listed(chat.store.conn)] == [("task", identity)]


LOCAL = "ollama/qwen2.5:7b"


@pytest.mark.parametrize("operation", ["stop", "remove"])
def test_cancel_confirmation_keeps_real_chat_loop_running(page: Any, operation: str) -> None:
    app, chat, cfg, identity = page
    action = ui.ACTIONS.validate_python(
        {
            "type": "module_hold",
            "operation": operation,
            "source": "ollama",
            "model": LOCAL,
            "scope": "all",
            "scope_id": "",
        }
    )
    reads = iter([action, ui.SimpleAction(type="terminal"), None])

    continued = []

    def run() -> None:
        conn = db.open_db(app.database)
        try:
            thread_chat = Chat(Sessions(conn), read=lambda: next(reads))
            with ui.use(app.front):
                thread_chat.run()
            continued.append(thread_chat.in_terminal)
        finally:
            conn.close()

    result, prompt = roundtrip(app.front, run, cancel=True)
    assert result is None and prompt["kind"] == "confirm"
    assert LOCAL in prompt["question"]
    assert continued == [True] and not app.front.ended
    assert not holds.listed(chat.store.conn)


def no_window(*_: Any, **__: Any) -> bool:
    raise AssertionError("a cloud module or a task's ✕ must not ask")


def test_cloud_module_buttons_and_task_remove_act_at_once(
    page: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, chat, cfg, identity = page
    model = router.lineup(cfg, calls.Role.EXECUTOR)[0].label
    monkeypatch.setattr(ui, "confirm", no_window)

    def hold(operation: Literal["stop", "continue", "remove"]) -> None:
        chat._action(
            ui.ModuleHoldAction(
                type="module_hold",
                operation=operation,
                source=key_id(SECRET),
                model=model,
                scope="all",
                scope_id="",
            )
        )

    hold("stop")
    assert [h.state for h in holds.listed(chat.store.conn)] == ["stopped"]
    hold("continue")
    assert not holds.listed(chat.store.conn)
    hold("remove")
    assert [h.state for h in holds.listed(chat.store.conn)] == ["removed"]
    chat._action(ui.TaskControlAction(type="task_control", operation="remove", id=identity))
    assert chat.store.find_task(identity)["hidden_at"] is not None


@pytest.mark.parametrize("operation", ["stop", "continue", "remove"])
def test_local_module_buttons_ask_and_decline_changes_nothing(
    page: Any, monkeypatch: pytest.MonkeyPatch, operation: Literal["stop", "continue", "remove"]
) -> None:
    app, chat, cfg, identity = page
    if operation == "continue":
        modules_cli.hold_change(
            chat.store.conn, cfg, "stop", holds.Hold(source="ollama", model=LOCAL)
        )
    before = holds.listed(chat.store.conn)
    asked: list[str] = []

    def decline(question: str, default: bool = False) -> bool:
        asked.append(question)
        return False

    monkeypatch.setattr(ui, "confirm", decline)
    chat._action(
        ui.ModuleHoldAction(
            type="module_hold",
            operation=operation,
            source="ollama",
            model=LOCAL,
            scope="all",
            scope_id="",
        )
    )
    assert len(asked) == 1 and LOCAL in asked[0]
    if operation == "remove":  # a cloud key is ready: the work may leave this computer
        assert "leave this computer" in asked[0]
    assert holds.listed(chat.store.conn) == before


def test_change_asks_only_when_a_local_model_is_involved(page: Any) -> None:
    _, _, cfg, _ = page
    cloud = router.lineup(cfg, calls.Role.EXECUTOR)[0].label
    assert modules_cli.page_choice_question(cfg, cloud, cloud) is None
    assert modules_cli.page_choice_question(cfg, None, cloud) is None
    assert modules_cli.page_choice_question(cfg, cloud, None) is None
    picking = modules_cli.page_choice_question(cfg, cloud, LOCAL)
    assert picking is not None and LOCAL in picking and "free memory" in picking
    leaving = modules_cli.page_choice_question(cfg, LOCAL, cloud)
    assert leaving is not None and cloud in leaving and "leave this computer" in leaving
    to_auto = modules_cli.page_choice_question(cfg, LOCAL, None)
    assert to_auto is not None and "leave this computer" in to_auto
    assert modules_cli.page_choice_question(cfg, LOCAL, LOCAL) is not None


@pytest.mark.parametrize("scope", ["task", "role"])
def test_pause_change_resume_uses_new_choice(page: Any, scope: Literal["task", "role"]) -> None:
    app, chat, cfg, identity = page
    lineup = router.lineup(cfg, calls.Role.EXECUTOR, kind=Kind.TEXT)
    original, replacement = lineup[:2]
    hold = holds.Hold(source=key_id(SECRET), model=original.label, scope="task", scope_id=identity)
    # Reuse the same model router over the edit, as an already-running worker does.
    models = runs.models(
        cfg, chat.store.conn, cache=False, read_tuning=lambda: open_config().load().tuning
    )
    modules_cli.hold_change(chat.store.conn, cfg, "stop", hold)
    assert chat.store.find_task(identity)["paused_at"]
    with ui.use(app.front):
        chat._action(
            ui.ModuleAction(
                type="module",
                scope=scope,
                scope_id=identity if scope == "task" else None,
                role="text",
                model=replacement.label,
                key=key_id(SECRET),
            )
        )
    if scope == "task":
        assert choices.get(chat.store.conn, "task", identity, "text").model == replacement.label
    else:
        assert open_config().load().tuning.models.text == replacement.label
    assert holds.listed(chat.store.conn)[0].state == "stopped"
    modules_cli.hold_change(chat.store.conn, cfg, "continue", hold)
    assert chat.store.find_task(identity)["paused_at"] is None
    with calls.for_(calls.Purpose(calls.Role.EXECUTOR, task_id=identity)):
        assert runs.label(models.executor(Kind.TEXT)[0]) == replacement.label


def test_cards_include_provider_and_filter_finished_work(page: Any) -> None:
    _, chat, cfg, identity = page
    rows = modules_cli.modules_data(chat.store.conn, cfg)
    active = cards(chat.store.conn, cfg, rows)
    assert any(w["id"] == identity for c in active for w in c["work"])
    assert all(c["provider_name"] == providers.find(c["provider"]).name for c in active)
    chat.store.cancel_task(identity, "finished")
    assert not any(
        w["id"] == identity for c in cards(chat.store.conn, cfg, rows) for w in c["work"]
    )


def test_page_shutdown_is_not_swallowed(page: Any) -> None:
    app, chat, _, identity = page
    app.front.end()
    with ui.use(app.front), pytest.raises(ui.Cancelled):
        chat._action(
            ui.ModuleHoldAction(
                type="module_hold",
                operation="remove",
                source="ollama",
                model=LOCAL,
                scope="all",
                scope_id="",
            )
        )


def test_usage_window_javascript() -> None:
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("node unavailable")
    subprocess.run(
        [node, "tests/js/gui_usage.cjs", "src/desman/gui/static/app.js"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def test_change_from_auto_running_a_local_model_asks(
    page: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, chat, cfg, identity = page
    cloud = router.lineup(cfg, calls.Role.EXECUTOR)[0].label
    asked: list[str] = []

    def decline(question: str, default: bool = False) -> bool:
        asked.append(question)
        return False

    monkeypatch.setattr(ui, "confirm", decline)
    # Auto picks a local model first: leaving it for the cloud asks (Devin, PR #110).
    monkeypatch.setattr(router, "candidates", lambda *_: [LOCAL, cloud])
    chat._action(
        ui.ModuleAction(
            type="module",
            role="text",
            scope="task",
            scope_id=identity,
            model=cloud,
            key=key_id(SECRET),
        )
    )
    assert len(asked) == 1 and LOCAL in asked[0] and "leave this computer" in asked[0]
    assert choices.get(chat.store.conn, "task", identity, "text").model is None
