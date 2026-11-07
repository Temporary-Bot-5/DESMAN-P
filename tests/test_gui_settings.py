"""Settings fields stay human-owned, line-gated, read-only on HTTP and secret-free."""

from __future__ import annotations

import json
import threading
from html import escape
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from desman import config_cli, tuning_cli, ui, wizard
from desman.chat import Chat
from desman.config import open_config
from desman.core import db
from desman.core.profile import Author
from desman.core.sessions import Role, Sessions
from desman.core.store import Store
from desman.gui.front import WebFront
from desman.gui.server import Server
from desman.llm.openai_compat import key_id
from helpers import make_config
from test_gui import headers


@pytest.fixture
def settings_app(tmp_path: Path) -> Any:
    path = tmp_path / "settings.db"
    conn = db.open_db(path)
    app = Server(WebFront(), 12345, poll_timeout=0, database=path)
    yield app, Store(conn)
    conn.close()


def action(change: dict[str, Any]) -> ui.Action:
    return ui.ACTIONS.validate_python({"type": "settings", "id": "test", "change": change})


def snapshot(app: Server) -> dict[str, Any]:
    response = app.handle("GET", "/settings", headers(app))
    assert response.status == 200, response.body
    data: dict[str, Any] = json.loads(response.body)
    return data


def test_settings_snapshot_is_read_only_and_redacted(settings_app: Any) -> None:
    app, store = settings_app
    secret = 'configured&<key>"\\value'
    config = make_config(api_keys={"GROQ_API_KEY": secret}, personalize=True)
    config.behaviour.custom = escape(secret)
    config.settings["OLLAMA_HOST"] = json.dumps(secret)[1:-1]
    config.settings["OLLAMA_MODEL"] = secret
    open_config().save(config)
    chat = Chat(Sessions(store.conn))
    chat.profile.add(secret, Author.PERSON)
    before = store.conn.total_changes
    data = snapshot(app)
    assert set(data) == {
        "general",
        "providers",
        "roles",
        "permissions",
        "personalize",
        "behaviour",
        "limits",
        "network",
        "protected",
        "result",
    }
    assert len(data["general"]["languages"]) == 13
    assert len(data["permissions"]) == 4
    assert len(data["limits"]) == 6
    groq = next(p for p in data["providers"] if p["id"] == "groq")
    assert groq["ready"] and groq["keys"][0]["fingerprint"] == key_id(secret)
    assert "value" not in groq["keys"][0]
    assert len(data["roles"]) == 7
    assert data["personalize"]["consent"] == wizard.personalize_consent()
    assert data["personalize"]["notes"]
    assert data["network"]["enabled"] is False
    for endpoint in [
        "/settings",
        "/sessions",
        "/tasks",
        "/modules",
        "/events?after=0",
        "/strings.json",
        "/",
        "/app.js",
        "/app.css",
    ]:
        response = app.handle("GET", endpoint, headers(app))
        raw = response.body.decode()
        for form in {secret, escape(secret), json.dumps(secret)[1:-1]}:
            assert form not in raw
    assert store.conn.total_changes == before
    assert app.handle("GET", "/settings", headers(app, Cookie="")).status == 403


VALID: list[dict[str, Any]] = [
    {"section": "general", "language": "en"},
    {"section": "wizard"},
    {"section": "keys", "operation": "add", "provider": "groq"},
    {"section": "keys", "operation": "check", "provider": "groq"},
    {"section": "provider", "provider": "ollama", "setting": "OLLAMA_MODEL", "value": "m"},
    {"section": "roles", "role": "planner", "model": None, "key": None},
    {"section": "permissions", "name": "shell_execution", "on": True},
    {"section": "personalize", "on": True},
    {"section": "forget", "note": 1},
    {"section": "behaviour", "tone": "formal"},
    {"section": "limits", "name": "max_parts", "value": 4},
    {"section": "models", "operation": "add", "model": "a/model"},
    {"section": "network", "operation": "join"},
    {"section": "protected", "operation": "show"},
]


@pytest.mark.parametrize("change", VALID)
def test_settings_sections_forbid_extra(change: dict[str, Any]) -> None:
    assert isinstance(action(change), ui.SettingsAction)
    with pytest.raises(ValidationError):
        action(change | {"extra": 1})


@pytest.mark.parametrize(
    "change",
    [
        {"section": "general", "language": "xx"},
        {"section": "keys", "operation": "remove", "provider": "groq", "key": "secret"},
        {"section": "keys", "operation": "remove", "provider": "groq"},
        {"section": "provider", "provider": "ollama", "setting": "a", "value": "v"},
        {"section": "roles", "role": "analyst", "model": None, "key": None},
        {"section": "permissions", "name": "shell_execution", "on": "true"},
        {"section": "personalize", "on": 1},
        {"section": "forget", "note": 0},
        {"section": "behaviour", "tone": "loud"},
        {"section": "behaviour", "custom": "x" * 501},
        {"section": "limits", "name": "max_parts", "value": 7},
        {"section": "limits", "name": "max_parts", "value": True},
        {"section": "models", "operation": "add", "model": "a/m", "server": "other"},
        {"section": "network", "operation": "donate", "provider": "groq", "calls": 0},
        {"section": "network", "operation": "donate"},
        {"section": "protected", "operation": "delete"},
    ],
)
def test_bad_settings_values_are_refused(settings_app: Any, change: dict[str, Any]) -> None:
    app, _ = settings_app
    with pytest.raises(ValidationError):
        action(change)
    body = json.dumps({"type": "settings", "id": "bad", "change": change}).encode()
    assert app.handle("POST", "/action", headers(app), body).status == 400


def test_settings_need_an_open_line(settings_app: Any) -> None:
    app, _ = settings_app
    body = action(VALID[0]).model_dump_json().encode()
    assert app.handle("POST", "/action", headers(app), body).status == 409
    got: list[Any] = []
    thread = threading.Thread(target=lambda: got.append(app.front.line()))
    thread.start()
    with app.front.condition:
        assert app.front.condition.wait_for(lambda: app.front.prompt is not None, timeout=2)
    assert app.handle("POST", "/action", headers(app), body).status == 200
    assert app.handle("POST", "/action", headers(app), body).status == 409
    thread.join(2)
    assert isinstance(got[0], ui.SettingsAction)


@pytest.mark.parametrize(
    "section",
    [
        "general",
        "permissions",
        "behaviour",
        "limits",
        "roles",
        "personalize",
        "keys",
        "provider",
        "forget",
    ],
)
def test_settings_use_bodies_and_keep_chat(
    settings_app: Any, monkeypatch: pytest.MonkeyPatch, section: str
) -> None:
    app, store = settings_app
    secret = "gsk_" + "a" * 48
    config = make_config(api_keys={"GROQ_API_KEY": secret})
    open_config().save(config)
    sessions = Sessions(store.conn)
    session = sessions.add(None, Role.USER, "hello").session_id
    chat = Chat(sessions)
    chat.session_id = session
    seen: list[str] = []
    functions = {
        "general": "set_language",
        "permissions": "set_permissions",
        "behaviour": "set_behaviour",
        "personalize": "set_personalize",
        "keys": "remove_keys",
    }
    target = (
        tuning_cli
        if section in ("limits", "roles")
        else wizard
        if section == "provider"
        else config_cli
    )
    name = (
        "set_tuning"
        if target is tuning_cli
        else "set_provider_setting"
        if target is wizard
        else functions.get(section)
    )
    if name:
        original = getattr(target, name)

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            seen.append("body")
            return original(*args, **kwargs)

        monkeypatch.setattr(target, name, wrapper)
    model = tuning_cli.role_rows(config)[0]["candidate_models"][0]
    changes: dict[str, dict[str, Any]] = {
        "general": {"section": "general", "language": "ru"},
        "permissions": {"section": "permissions", "name": "network_access", "on": True},
        "behaviour": {"section": "behaviour", "tone": "formal"},
        "limits": {"section": "limits", "name": "max_subtasks", "value": 6},
        "roles": {"section": "roles", "role": "planner", "model": model, "key": key_id(secret)},
        "personalize": {"section": "personalize", "on": True},
        "keys": {
            "section": "keys",
            "operation": "remove",
            "provider": "groq",
            "key": key_id(secret),
        },
        "provider": {
            "section": "provider",
            "provider": "ollama",
            "setting": "OLLAMA_MODEL",
            "value": "model",
        },
    }
    if section == "forget":
        config.personalize = True
        open_config().save(config)
        note = chat.profile.add("private", Author.PERSON)
        changes["forget"] = {"section": "forget", "note": note.id}
    with ui.use(app.front):
        ui.session(session, False)
        chat._action(action(changes[section]))
    result = snapshot(app)["result"]
    assert result["id"] == "test" and result["ok"], result
    if name:
        assert seen == ["body"]
    saved = open_config().load()
    if section == "general":
        assert saved.language == "ru"
    if section == "permissions":
        assert saved.permissions.network_access
    if section == "behaviour":
        assert saved.behaviour.tone == "formal"
    if section == "limits":
        assert saved.tuning.max_subtasks == 6
    if section == "roles":
        assert saved.tuning.models.planner == model and saved.tuning.keys.planner == key_id(secret)
    if section == "personalize":
        assert saved.personalize
    if section == "keys":
        assert not saved.keys("GROQ_API_KEY")
    if section == "provider":
        assert saved.settings["OLLAMA_MODEL"] == "model"
    if section == "forget":
        assert not chat.profile.notes()
    assert chat.session_id == session and not chat.in_terminal
    assert app.front.current == session
    assert sessions.messages(sessions.terminal())[-1].kind.value == "event"
    journal = store.conn.execute("SELECT action, details_json FROM operator_log").fetchall()
    assert journal
    expected = {
        "general": "operator_language",
        "permissions": "operator_permission_grant",
        "behaviour": "operator_behaviour",
        "limits": "operator_tuning",
        "roles": "operator_tuning",
        "personalize": "operator_personalize",
        "keys": "operator_key_removed",
        "provider": "operator_setting",
        "forget": "profile_note_forgotten",
    }
    assert expected[section] in {r["action"] for r in journal}
    assert secret not in str([tuple(r) for r in journal])
    assert secret not in str(app.front.events)


def test_add_key_uses_secret_prompt_without_events(settings_app: Any) -> None:
    app, store = settings_app
    open_config().save(make_config())
    chat = Chat(Sessions(store.conn))
    secret = 'secret&<key>"value'

    def respond() -> None:
        with app.front.condition:
            assert app.front.condition.wait_for(lambda: app.front.prompt is not None, timeout=2)
            prompt = app.front.prompt
            assert prompt is not None and prompt["kind"] == "secret"
            assert app.front.answer({"id": prompt["id"], "value": secret})

    thread = threading.Thread(target=respond)
    thread.start()
    with ui.use(app.front):
        chat._action(
            action({"section": "keys", "operation": "add", "provider": "groq", "label": "work"})
        )
    thread.join(2)
    assert not thread.is_alive()
    assert open_config().load().keys("GROQ_API_KEY")[0].value == secret
    assert snapshot(app)["result"]["ok"]
    for form in [secret, escape(secret), json.dumps(secret)[1:-1]]:
        assert form not in json.dumps(app.front.poll(0, 0), ensure_ascii=False)
        assert form not in json.dumps(snapshot(app), ensure_ascii=False)
        assert form not in str([tuple(r) for r in store.conn.execute("SELECT * FROM messages")])


def test_check_results_and_errors_are_inline(
    settings_app: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from desman import providers
    from desman.llm.check import KeyCheck

    app, store = settings_app
    secret = "gsk_" + "a" * 48
    open_config().save(make_config(api_keys={"GROQ_API_KEY": secret}))
    monkeypatch.setattr(
        config_cli,
        "check_keys",
        lambda *_: (
            [providers.find("groq")],
            [KeyCheck("groq", 1, key_id(secret), True, 200, "accepted", models=3)],
        ),
    )
    chat = Chat(Sessions(store.conn))
    with ui.use(app.front):
        chat._action(action({"section": "keys", "operation": "check", "provider": "groq"}))
    result = snapshot(app)["result"]
    assert result["ok"] and "ok" in result["message"]
    assert "GROQ" in result["message"]
    with ui.use(app.front):
        chat._action(
            action({"section": "keys", "operation": "remove", "provider": "groq", "key": "0" * 12})
        )
    assert not snapshot(app)["result"]["ok"]
    assert open_config().load().keys("GROQ_API_KEY")[0].value == secret


def test_models_download_keeps_preview_and_confirmation(
    settings_app: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from desman import local_models
    from desman.llm import hub

    app, store = settings_app
    open_config().save(make_config())
    model = hub.Gguf(hub.parse("owner/model:Q4_K_M"), ("model.gguf",), 1024, False, ("Q4_K_M",))
    monkeypatch.setattr(local_models, "look_up", lambda *_: model)
    monkeypatch.setattr(hub, "memory", lambda: hub.Memory(ram=1024**3, video=None))
    fetched: list[bool] = []
    monkeypatch.setattr(local_models, "fetch", lambda *_: fetched.append(True))
    chat = Chat(Sessions(store.conn))

    def respond() -> None:
        with app.front.condition:
            assert app.front.condition.wait_for(lambda: app.front.prompt is not None, timeout=2)
            prompt = app.front.prompt
            assert prompt is not None and prompt["kind"] == "confirm"
            assert "owner/model" in prompt["question"]
            assert not fetched
            assert app.front.answer({"id": prompt["id"], "value": False})

    thread = threading.Thread(target=respond)
    thread.start()
    with ui.use(app.front):
        chat._action(action({"section": "models", "operation": "add", "model": "owner/model"}))
    thread.join(2)
    assert not fetched and not snapshot(app)["result"]["ok"]
    assert "OLLAMA_MODEL" not in open_config().load().settings


def test_network_lending_uses_existing_bodies_and_both_journals(settings_app: Any) -> None:
    from desman.config import Network

    app, store = settings_app
    config = make_config(settings={"OLLAMA_MODEL": "model"})
    config.network = Network(enabled=True, registry="https://registry.example.org")
    open_config().save(config)
    chat = Chat(Sessions(store.conn))
    with ui.use(app.front):
        chat._action(
            action(
                {
                    "section": "network",
                    "operation": "donate",
                    "provider": "ollama",
                    "calls": 10,
                    "tokens": 1000,
                }
            )
        )
    assert snapshot(app)["result"]["ok"]
    assert open_config().load().network.donated["ollama"].calls_per_day == 10
    with ui.use(app.front):
        chat._action(
            action(
                {
                    "section": "network",
                    "operation": "quota",
                    "provider": "ollama",
                    "calls": 12,
                    "tokens": 2000,
                }
            )
        )
        chat._action(action({"section": "network", "operation": "undonate", "provider": "ollama"}))
    assert not open_config().load().network.donated
    assert store.conn.execute("SELECT COUNT(*) FROM network_log").fetchone()[0] == 3
    assert (
        store.conn.execute(
            "SELECT COUNT(*) FROM operator_log WHERE action = 'operator_network'"
        ).fetchone()[0]
        == 3
    )


def test_code_prompt_defaults_do_not_expose_configured_keys(settings_app: Any) -> None:
    app, _ = settings_app
    secret = 'configured&<key>"\\value'
    open_config().save(make_config(api_keys={"GROQ_API_KEY": secret}))
    got: list[str] = []
    thread = threading.Thread(target=lambda: got.append(app.front.text(secret, escape(secret))))
    thread.start()
    try:
        with app.front.condition:
            assert app.front.condition.wait_for(lambda: app.front.prompt is not None, timeout=2)
            prompt = app.front.prompt
            assert prompt is not None
            raw = app.handle("GET", "/events?after=0", headers(app)).body.decode()
            for form in [secret, escape(secret), json.dumps(secret)[1:-1]]:
                assert form not in raw
            assert app.front.answer({"id": prompt["id"], "value": "safe"})
    finally:
        if app.front.prompt is not None:
            app.front.answer({"id": app.front.prompt["id"], "cancel": True})
        thread.join(2)
    assert got == ["safe"]


def test_settings_autosave_events(tmp_path: Path) -> None:
    """Unchanged/invalid fields stay local; selects and edited text send typed actions."""
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is needed for the browser event regression")
    source = Path("src/desman/gui/static/app.js").read_text(encoding="utf-8")
    helper = source[source.index("function settingAuto(") : source.index("function settingSave(")]
    script = tmp_path / "autosave.cjs"
    script.write_text(
        'const assert = require("node:assert/strict");\n'
        'const crypto = require("node:crypto");\n'
        "let lineOpen = true, sending = false, settingsPending = null, settingsReceipt;\n"
        "const settingsDeferred = new Set(); const sent = []; const act = a => sent.push(a);\n"
        "const text = () => ({isConnected: true});\n"
        + helper
        + """
const events = {};
const control = {tagName: 'INPUT', type: 'number', value: '3', dataset: {},
  checkValidity: () => true, addEventListener: (name, fn) => events[name] = fn};
settingAuto({}, () => ({section: 'limits', name: 'max_parts', value: Number(control.value)}), [control]);
events.blur(); assert.equal(sent.length, 0);
control.value = '4'; events.keydown({key: 'Enter', preventDefault() {}});
assert.equal(sent.length, 1); assert.equal(sent[0].change.value, 4);
settingsPending = null; events.blur(); assert.equal(sent.length, 1);
control.value = '5'; control.checkValidity = () => false;
events.blur(); assert.equal(sent.length, 1);
control.checkValidity = () => true; lineOpen = false;
events.blur(); assert.equal(sent.length, 1);
lineOpen = true; control.tagName = 'SELECT'; control.dataset = {};
settingAuto({}, () => ({section: 'general', language: control.value}), [control]);
control.value = 'ru'; events.change();
assert.equal(sent.at(-1).change.language, 'ru');
""",
        encoding="utf-8",
    )
    subprocess.run([node, str(script)], check=True, capture_output=True, text=True)


@pytest.mark.parametrize("section", ["roles", "keys"])
def test_settings_key_fingerprint_ignores_index_like_label(settings_app: Any, section: str) -> None:
    app, store = settings_app
    first, labelled = "gsk_" + "a" * 48, "gsk_" + "b" * 48
    config = make_config(
        api_keys={
            "GROQ_API_KEY": [
                {"value": first, "label": "first"},
                {"value": labelled, "label": "#1"},
            ]
        }
    )
    open_config().save(config)
    chat = Chat(Sessions(store.conn))
    if section == "roles":
        model = tuning_cli.role_rows(config)[0]["candidate_models"][0]
        change = {"section": section, "role": "planner", "model": model, "key": key_id(first)}
    else:
        change = {
            "section": section,
            "operation": "remove",
            "provider": "groq",
            "key": key_id(first),
        }
    with ui.use(app.front):
        chat._action(action(change))
    assert snapshot(app)["result"]["ok"]
    saved = open_config().load()
    if section == "roles":
        assert saved.tuning.keys.planner == key_id(first)
    else:
        assert [k.value for k in saved.keys("GROQ_API_KEY")] == [labelled]


@pytest.mark.parametrize("key", ["unknown", "other-provider"])
def test_settings_role_refuses_invalid_fingerprint(settings_app: Any, key: str) -> None:
    app, store = settings_app
    secret = "gsk_" + "a" * 48
    other = "other-secret"
    config = make_config(api_keys={"GROQ_API_KEY": secret, "OPENROUTER_API_KEY": other})
    open_config().save(config)
    model = next(
        m for m in tuning_cli.role_rows(config)[0]["candidate_models"] if m.startswith("groq/")
    )
    before = open_config().config_path.read_bytes()
    with ui.use(app.front):
        Chat(Sessions(store.conn))._action(
            action(
                {
                    "section": "roles",
                    "role": "planner",
                    "model": model,
                    "key": key_id(other) if key == "other-provider" else "0" * 12,
                }
            )
        )
    assert not snapshot(app)["result"]["ok"]
    assert open_config().config_path.read_bytes() == before
    assert not store.conn.execute("SELECT * FROM operator_log").fetchall()


@pytest.mark.parametrize("auto", [True, False])
def test_settings_role_keeps_independent_key(settings_app: Any, auto: bool) -> None:
    app, store = settings_app
    secret, other = "groq-secret", "openrouter-secret"
    config = make_config(api_keys={"GROQ_API_KEY": secret, "OPENROUTER_API_KEY": other})
    config.tuning.keys.planner = key_id(other)
    open_config().save(config)
    model = next(
        m for m in tuning_cli.role_rows(config)[0]["candidate_models"] if m.startswith("groq/")
    )
    with ui.use(app.front):
        Chat(Sessions(store.conn))._action(
            action(
                {
                    "section": "roles",
                    "role": "planner",
                    "model": None if auto else model,
                    "key": key_id(other),
                }
            )
        )
    result = snapshot(app)["result"]
    assert result["ok"], result
    saved = open_config().load()
    assert saved.tuning.keys.planner == key_id(other)
    assert saved.tuning.models.planner == (None if auto else model)
    if not auto:
        assert "only applies" in result["message"]


def test_settings_auto_accepts_any_configured_key(settings_app: Any) -> None:
    app, store = settings_app
    config = make_config(api_keys={"GROQ_API_KEY": "groq", "OPENROUTER_API_KEY": "other"})
    open_config().save(config)
    with ui.use(app.front):
        Chat(Sessions(store.conn))._action(
            action({"section": "roles", "role": "planner", "model": None, "key": key_id("other")})
        )
    assert snapshot(app)["result"]["ok"]
    assert open_config().load().tuning.keys.planner == key_id("other")


def test_settings_remote_server_warning_is_inline(settings_app: Any) -> None:
    app, store = settings_app
    open_config().save(make_config())
    sessions = Sessions(store.conn)
    with ui.use(app.front):
        Chat(sessions)._action(
            action(
                {
                    "section": "provider",
                    "provider": "ollama",
                    "setting": "OLLAMA_HOST",
                    "value": "https://models.example.org",
                }
            )
        )
    result = snapshot(app)["result"]
    assert result["ok"]
    terminal = sessions.messages(sessions.terminal())[-1].text
    assert "models.example.org" in open_config().load().settings["OLLAMA_HOST"]
    assert result["message"] != "Saved"
    warning = result["message"].split("\n", 1)[1]
    assert "prompts, context and data will be sent there" in warning
    assert " ".join(warning.split()) in " ".join(terminal.split())


def test_settings_role_key_options(tmp_path: Path) -> None:
    """Shared pickers filter provider choices and preserve open editors during polling."""
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is needed for the browser event regression")
    subprocess.run(
        [node, "tests/js/gui_picker.cjs", "src/desman/gui/static/app.js"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
