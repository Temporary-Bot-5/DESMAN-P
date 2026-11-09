"""Owner regressions for T.2j: new chats, durable terminal screens and panel isolation."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from desman import local_models, runs, tuning_cli, ui
from desman.chat import Chat, _commands
from desman.config import Config, open_config
from desman.core import calls, choices, db
from desman.core.sessions import Kind, Role, Sessions
from desman.gui.front import WebFront
from desman.gui.panels import snapshot
from desman.gui.server import Server, strings
from desman.llm import hub, router
from desman.llm.fake import FakeLLM
from desman.llm.openai_compat import key_id
from helpers import make_config
from test_gui import headers


@pytest.fixture
def sessions(tmp_path: Path) -> Any:
    conn = db.open_db(tmp_path / "gui.db")
    open_config().save(make_config(api_keys={"GROQ_API_KEY": "groq-secret"}))
    yield Sessions(conn)
    conn.close()


def module(**extra: Any) -> ui.ModuleAction:
    return ui.ModuleAction.model_validate(
        dict(
            type="module", role="interlocutor", scope="session", scope_id=None, model=None, key=None
        )
        | extra
    )


def setting(change: dict[str, Any]) -> ui.SettingsAction:
    return ui.SettingsAction.model_validate(dict(type="settings", id="fix", change=change))


@pytest.mark.parametrize("temporary", [False, True])
def test_pending_choice_card_first_call_and_journal(
    sessions: Sessions, temporary: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    front = WebFront()
    chat = Chat(sessions, temporary=temporary)
    config = open_config().load()
    model = tuning_cli.role_rows(config)[-1]["candidate_models"][-1]
    key = key_id("groq-secret")
    fake = FakeLLM('{"type":"reply","text":"first chosen reply"}')
    seen: list[tuple[str | None, choices.Choice]] = []

    class Reply:
        name = "chosen fake"

        def complete(self, request: Any) -> Any:
            purpose = calls.purpose()
            assert purpose.session_id is not None
            choice = choices.get(sessions.conn, "session", purpose.session_id, "interlocutor")
            seen.append((purpose.session_id, choice))
            return fake.complete(request)

    def models(config: Config) -> runs.Models:
        assert config.tuning.models.interlocutor == model
        return runs.Models(fake, None, roles={calls.Role.INTERLOCUTOR: Reply()})

    monkeypatch.setattr(chat, "_models", models)
    with ui.use(front):
        chat._action(module(model=model, key=key))
        assert chat.session_id is None
        assert not sessions.conn.execute("SELECT * FROM module_choices").fetchall()
        assert not sessions.conn.execute("SELECT * FROM operator_log").fetchall()
        path = Path(sessions.conn.execute("PRAGMA database_list").fetchone()["file"])
        card = snapshot(path, "modules", front)["interlocutor"]
        assert card["source"] == "chat" and card["choice_model"] == model
        chat._message("first message")
    assert seen and seen[0][1] == choices.Choice(model=model, key=key)
    assert chat._pending_choice is None and front.pending_choice is None
    assert (
        sessions.conn.execute(
            "SELECT temporary FROM sessions WHERE id = ?", (chat.session_id,)
        ).fetchone()[0]
        == temporary
    )
    entries = sessions.conn.execute(
        "SELECT * FROM operator_log WHERE action = 'operator_module_choice'"
    ).fetchall()
    assert len(entries) == 1
    assert json.loads(entries[0]["details_json"])["scope_id"] == chat.session_id


@pytest.mark.parametrize("invalid", ["model", "key", "foreign", "null-after-start"])
def test_pending_choice_cannot_bypass_validation(sessions: Sessions, invalid: str) -> None:
    chat = Chat(sessions)
    other = sessions.add(None, Role.USER, "other").session_id
    payload: dict[str, Any] = {}
    if invalid == "model":
        payload["model"] = "ollama/forged"
    elif invalid == "key":
        payload["key"] = "0" * 12
    elif invalid == "foreign":
        payload["scope_id"] = other
    else:
        chat.session_id = other
    with ui.use(WebFront()):
        chat._action(module(**payload))
    assert chat._pending_choice is None
    assert not sessions.conn.execute("SELECT * FROM module_choices").fetchall()
    assert not sessions.conn.execute("SELECT * FROM operator_log").fetchall()
    with pytest.raises(ValidationError):
        module(scope="task", role="reviewer")
    with pytest.raises(ValidationError):
        module(role="planner")


@pytest.mark.parametrize("switch", ["new", "temporary", "session"])
def test_pending_choice_dropped_on_navigation(sessions: Sessions, switch: str) -> None:
    chat, front = Chat(sessions), WebFront()
    with ui.use(front):
        chat._action(module(key=key_id("groq-secret")))
        if switch == "session":
            other = sessions.add(None, Role.USER, "other").session_id
            info = sessions.get(other)
            assert info is not None
            chat._open_session(info)
        else:
            chat._new(temporary=switch == "temporary")
    assert chat._pending_choice is None and front.pending_choice is None


def test_clear_upgrade_and_cross_front_screen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "upgrade.db"
    conn = db.connect(path)
    db.migrate(conn, db.MIGRATIONS[:-1])
    sessions = Sessions(conn)
    terminal = sessions.terminal()
    sessions.add(terminal, Role.DESMAN, "old screen", Kind.EVENT)
    conn.close()
    conn = db.open_db(path)
    sessions = Sessions(conn)
    assert sessions.screen(terminal)[0].text == "old screen"
    chat, front = Chat(sessions), WebFront()
    open_config().save(Config(configured=True))
    with ui.use(front), front.capture():
        chat._terminal_command("clear")
        assert not sessions.screen(terminal)
        chat._back()
        chat._terminal_open()
    visible = json.dumps(list(front.events))
    assert "old screen" not in visible
    assert len(sessions.messages(terminal)) >= 2  # append-only originals and clear line
    assert "/clear" in _commands(terminal=True) and "/clear" in _commands()
    assert snapshot(path, "sessions", WebFront())["terminal"]["unread"] == 0
    sessions.add(terminal, Role.DESMAN, "new screen", Kind.EVENT)
    assert snapshot(path, "sessions", WebFront())["terminal"]["unread"] == 1
    assert [m.text for m in sessions.screen(terminal)] == ["new screen"]
    # !clear from an ordinary chat follows the other explicit commands into the terminal.
    with ui.use(front), front.capture():
        chat._back()
        chat._terminal_command("!clear")
    assert not sessions.screen(terminal)
    cleared: list[bool] = []
    monkeypatch.setattr("click.clear", lambda: cleared.append(True))
    with ui.use(ui.TerminalFront()):
        chat._typed("clear")
    assert cleared == [True]
    conn.close()


@pytest.mark.parametrize(
    "change, expected",
    [
        ({"section": "general", "language": "en"}, "Language set to English."),
        (
            {"section": "network", "operation": "donate", "provider": "ollama"},
            "desman-global is off",
        ),
        ({"section": "limits", "name": "max_parts", "value": 4}, "Saved"),
    ],
)
def test_real_front_panel_output_never_enters_chat(
    sessions: Sessions, change: dict[str, Any], expected: str
) -> None:
    if change["section"] == "general":
        config = open_config().load()
        config.language = "ru"
        open_config().save(config)
    front, chat = WebFront(), Chat(sessions)
    current = sessions.add(None, Role.USER, "ordinary chat").session_id
    chat.session_id = current
    with ui.use(front), front.capture():
        chat._action(setting(change))
    assert chat.session_id == current
    assert not any(e["type"] == "out" for e in front.events), list(front.events)
    assert expected in " ".join(m.text for m in sessions.messages(chat._terminal_id))
    assert [m.text for m in sessions.messages(current)] == ["ordinary chat"]
    assert front.settings_outcome is not None


def test_settings_prompt_english_echo_only_in_terminal(sessions: Sessions) -> None:
    front, chat = WebFront(), Chat(sessions)
    failures: list[BaseException] = []

    def answer() -> None:
        try:
            with front.condition:
                assert front.condition.wait_for(lambda: front.prompt is not None, timeout=5)
                prompt = dict(front.prompt or {})
            assert prompt["kind"] == "select" and prompt["terminal"] is True
            assert front.answer({"id": prompt["id"], "value": 0})
            with front.condition:
                assert front.condition.wait_for(
                    lambda: front.prompt is not None and front.prompt["id"] != prompt["id"],
                    timeout=5,
                )
                prompt = dict(front.prompt or {})
            assert front.answer({"id": prompt["id"], "cancel": True})
        except BaseException as exc:
            failures.append(exc)

    thread = threading.Thread(target=answer, daemon=True)
    with ui.use(front), front.capture():
        thread.start()
        chat._action(setting({"section": "wizard"}))
        thread.join(timeout=5)
    assert not thread.is_alive() and not failures
    echoes = [e["echo"] for e in front.events if e["type"] == "prompt_closed"]
    assert echoes and all(e is None for e in echoes)
    assert "English" in " ".join(m.text for m in sessions.messages(chat._terminal_id))
    assert not any(e["type"] == "out" for e in front.events)


def test_local_list_search_and_hf_token_boundary(
    sessions: Sessions, monkeypatch: pytest.MonkeyPatch
) -> None:
    from desman.gui.settings import local_search

    config = open_config().load()
    config.api_keys["HF_TOKEN"] = [
        config.api_keys["GROQ_API_KEY"][0].model_copy(update={"value": "hf-secret"})
    ]
    config.settings.update(OLLAMA_HOST="http://ollama.test:11434")
    local_models._tags_cache.clear()
    requests: list[httpx.Request] = []

    def transport(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.host == "ollama.test":
            assert "authorization" not in request.headers
            assert request.url.path == "/api/tags"
            return httpx.Response(
                200,
                json={
                    "models": [
                        {"name": "local:latest", "size": 1000},
                        {"name": "other:latest", "size": 2000},
                    ]
                },
            )
        assert request.url.host == "huggingface.co" and request.url.path == "/api/models"
        assert request.headers["authorization"] == "Bearer hf-secret"
        assert request.url.params["search"] == "tiny" and request.url.params["filter"] == "gguf"
        assert request.url.params["pipeline_tag"] == "text-generation"
        assert request.url.params.get_list("expand[]") == ["gguf", "downloads", "pipeline_tag"]
        assert request.url.params["limit"] == "20" and request.url.params["sort"] == "downloads"
        return httpx.Response(
            200,
            json=[
                {
                    "id": "author/tiny-GGUF",
                    "gguf": {"total": 7_600_000_000, "context_length": 32768},
                    "downloads": 1234,
                    "pipeline_tag": "text-generation",
                },
                {"id": "author/unknown-GGUF"},
            ],
        )

    original = httpx.Client
    monkeypatch.setattr(
        httpx, "Client", lambda **kw: original(transport=httpx.MockTransport(transport))
    )
    monkeypatch.setattr(hub, "memory", lambda: hub.Memory(ram=10_000_000_000, video=0))
    result = local_search(config, "tiny")
    assert len(requests) == 2 and result["reachable"]
    assert not result["options"][0]["selected"] and result["options"][0]["downloaded"]
    assert "7.6B" in result["options"][2]["hint"]
    assert "Approximately" in result["options"][2]["hint"]
    assert "1234" in result["options"][2]["hint"]
    assert "Fits in memory" in result["options"][2]["hint"]
    assert result["options"][3]["hint"] == "Size unknown"
    assert all(c.provider.id != "ollama" for c in router.lineup(config, calls.Role.INTERLOCUTOR))
    config.tuning.models.interlocutor = "ollama/other:latest"
    assert router.lineup(config, calls.Role.INTERLOCUTOR)[0].label == "ollama/other:latest"
    assert not tuning_cli.unready(config, config.tuning)
    local_only = config.model_copy(deep=True)
    local_only.api_keys = {}
    assert router.lineup(local_only, calls.Role.INTERLOCUTOR)[0].label == "ollama/other:latest"
    open_config().save(config)
    row = tuning_cli.role_rows(config)[0]
    p = next(p for p in row["providers"] if p["id"] == "ollama")
    assert "ollama/local:latest" in p["models"] and not p["keys"]
    assert "hf-secret" not in json.dumps(result) + json.dumps(row)
    front = WebFront()
    chat = Chat(sessions)
    with ui.use(front), front.capture():
        chat._action(setting({"section": "models", "operation": "use", "model": "local:latest"}))
    assert front.settings_outcome and front.settings_outcome["ok"]
    assert open_config().load().settings["OLLAMA_MODEL"] == "local:latest"
    assert sessions.conn.execute("SELECT * FROM operator_log").fetchall()
    config.tuning.models.interlocutor = "ollama/local:latest"
    assert router.lineup(config, calls.Role.INTERLOCUTOR)[0].label == "ollama/local:latest"
    local_models._tags_cache.clear()


def test_local_endpoint_auth_redaction_failures(
    sessions: Sessions, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = Path(sessions.conn.execute("PRAGMA database_list").fetchone()["file"])
    app = Server(WebFront(), 12345, poll_timeout=0, database=path)
    original = httpx.Client

    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("private traceback", request=request)

    monkeypatch.setattr(httpx, "Client", lambda **kw: original(transport=httpx.MockTransport(fail)))
    local_models._tags_cache.clear()
    before = sessions.conn.total_changes
    assert app.handle("GET", "/local-models?q=x", headers(app, Cookie="")).status == 403
    response = app.handle("GET", "/local-models?q=x", headers(app))
    assert response.status == 200
    result = json.loads(response.body)
    assert len(result["notes"]) == 2
    assert "traceback" not in response.body.decode()
    assert sessions.conn.total_changes == before
    local_models._tags_cache.clear()


def test_terminal_name_translated() -> None:
    from desman import i18n

    i18n.set_language("ru")
    try:
        assert strings()["terminal"] == "Общий терминал"
        assert "Общий терминал" in _commands()["/terminal"]
    finally:
        i18n.set_language("en")


@pytest.mark.parametrize("accept", [False, True])
def test_local_download_confirm_then_select(
    sessions: Sessions, monkeypatch: pytest.MonkeyPatch, accept: bool
) -> None:
    config = open_config().load()
    config.settings["OLLAMA_HOST"] = "http://download.test:11434"
    open_config().save(config)
    local_models._tags_cache.clear()
    downloaded: list[str] = []
    previews: list[Any] = []
    original = httpx.Client

    def send(request: httpx.Request) -> httpx.Response:
        if request.url.host == "huggingface.co":
            return httpx.Response(
                200, json={"siblings": [{"rfilename": "small-Q4_K_M.gguf", "size": 1024}]}
            )
        assert request.url.host == "download.test"
        assert "authorization" not in request.headers
        if request.url.path == "/api/tags":
            return httpx.Response(
                200, json={"models": [{"name": name, "size": 1024} for name in downloaded]}
            )
        assert request.url.path == "/api/pull"
        downloaded.append(json.loads(request.content)["model"])
        return httpx.Response(200, content=b'{"status":"success"}\n')

    monkeypatch.setattr(httpx, "Client", lambda **kw: original(transport=httpx.MockTransport(send)))
    monkeypatch.setattr(hub, "memory", lambda: hub.Memory(ram=100000, video=None))
    chat, front = Chat(sessions), WebFront()

    def confirm(proposal: Any, *, in_prompt: bool = False) -> bool:
        assert in_prompt and "Download author/small:Q4_K_M (" in proposal.question
        previews.append(proposal)
        assert not downloaded
        return accept

    monkeypatch.setattr(chat, "_confirmed", confirm)
    with ui.use(front), front.capture():
        chat._action(setting({"section": "models", "operation": "add", "model": "author/small"}))
    assert len(previews) == 1
    assert downloaded == (["hf.co/author/small:Q4_K_M"] if accept else [])
    assert open_config().load().settings.get("OLLAMA_MODEL") == (downloaded[0] if accept else None)
    if accept:
        tags, reachable = local_models.installed(open_config().load())
        assert reachable and tags[0].name == downloaded[0]
    assert not any(e["type"] == "out" for e in front.events)
    local_models._tags_cache.clear()


def test_hf_search_missing_metadata() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.params["pipeline_tag"] == "text-generation"
        assert request.url.params.get_list("expand[]") == ["gguf", "downloads", "pipeline_tag"]
        return httpx.Response(
            200,
            json=[
                {"id": "author/missing"},
                {"id": "author/partial", "gguf": {"context_length": 8192}, "downloads": 0},
                {"id": "author/null", "gguf": None, "downloads": None},
            ],
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        rows = hub.search("small", client)
    assert all(row.gguf() is None for row in rows)
    assert rows[0].metadata is None and rows[0].downloads is None
    assert rows[1].metadata and rows[1].metadata.total is None
    assert rows[1].metadata.context_length == 8192 and rows[1].downloads == 0


def test_hf_search_matches_each_word() -> None:
    asked: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        asked.append(request)
        return httpx.Response(
            200,
            json=[
                {"id": "Qwen/Qwen2.5-32B-Instruct-GGUF"},
                {"id": "Qwen/Qwen2.5-7B-Instruct-GGUF"},
                {"id": "someone/QwQ-32B-GGUF"},
            ],
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        rows = hub.search("  32b Qwen ", client)
        assert [row.id for row in rows] == ["Qwen/Qwen2.5-32B-Instruct-GGUF"]
        assert asked[0].url.params["search"] == "qwen"
        assert asked[0].url.params["limit"] == str(hub.SEARCH_WIDE)
        assert len(hub.search("qwen", client)) == 3  # one word: the hub's own match
        assert asked[1].url.params["limit"] == str(hub.SEARCH_LIMIT)


def test_provider_label_capitalised_in_every_catalog() -> None:
    from importlib import import_module

    assert strings()["provider"] == "Provider"
    for language in ("ru", "zh", "es", "pt", "de", "fr", "ja", "ko", "pl", "tr", "uk", "hi"):
        value = import_module("desman.locales." + language).MESSAGES["Provider"]
        assert isinstance(value, str)
        assert value[0] == value[0].upper()
