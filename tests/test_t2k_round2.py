"""Owner regressions: removal, adaptive context, exact inspection and language proposals."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from click.testing import CliRunner

from desman import commands, local_models, runs, ui
from desman.chat import Chat
from desman.cli import cli
from desman.config import open_config
from desman.core import choices, db, operator
from desman.core.calls import Role as CallRole
from desman.core.models import SubtaskSpec, TaskSpec
from desman.core.report import short_description
from desman.core.sessions import Role, Sessions
from desman.core.store import Store
from desman.gui.front import WebFront
from desman.gui.settings import command
from desman.interlocutor import Interlocutor
from desman.llm import hub, router
from desman.llm.base import Message as LLMMessage
from desman.llm.base import Request
from desman.llm.fake import FakeLLM
from desman.settings_actions import Models
from helpers import make_config


def test_remove_preview_confirm_and_preserve_choices(
    store: Store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config(
        settings={"OLLAMA_HOST": "http://configured.test:11434", "OLLAMA_MODEL": "tiny:latest"}
    )
    config.tuning.models.interlocutor = "ollama/tiny"
    open_config().save(config)
    sessions = Sessions(store.conn)
    ref = sessions.add(None, Role.USER, "hello").session_id
    task = store.create_task(
        TaskSpec(
            goal="test", subtasks=[SubtaskSpec(description="work", acceptance_criteria="done")]
        )
    )
    choices.set_choice(
        store.conn, "session", ref, "interlocutor", choices.Choice(model="ollama/tiny:latest")
    )
    choices.set_choice(store.conn, "task", task, "text", choices.Choice(model="ollama/tiny:latest"))
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        assert request.url.host == "configured.test"
        if request.method == "GET":
            assert request.url.path == "/api/tags"
            return httpx.Response(
                200, json={"models": [{"name": "tiny:latest", "size": 5_000_000_000}]}
            )
        assert request.method == "DELETE" and request.url.path == "/api/delete"
        assert json.loads(request.content) == {"model": "tiny:latest"}
        return httpx.Response(200, json={})

    original = httpx.Client
    monkeypatch.setattr(
        httpx, "Client", lambda **kw: original(transport=httpx.MockTransport(handle), **kw)
    )
    entry, args = command(Models(section="models", operation="remove", model="tiny:latest"), config)
    assert entry.kind is commands.Kind.CHANGE
    context = commands.Context(store, open_config(), lambda *_: None, lambda _: None)
    proposal = entry.handler(context, args)
    assert isinstance(proposal, commands.Proposal)
    assert "tiny:latest" in str(proposal.question) and "GB" in str(proposal.question)
    assert [r.method for r in seen] == ["GET"]
    outcome = proposal.apply()
    assert outcome.data is not None
    assert "session/" + ref + "/interlocutor" in outcome.data["missing_choices"]
    assert "task/" + task + "/text" in outcome.data["missing_choices"]
    assert "role/interlocutor" in outcome.data["missing_choices"]
    assert choices.get(store.conn, "task", task, "text").model == "ollama/tiny:latest"
    assert open_config().load() == config
    assert local_models.cached_installed(config) == ([], False)


def test_remove_cli_journal_and_redirect_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    open_config().save(make_config(settings={"OLLAMA_HOST": "http://configured.test"}))
    seen = []
    redirect = False

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        assert request.url.host == "configured.test"
        if request.method == "GET":
            return httpx.Response(200, json={"models": [{"name": "x", "size": 1}]})
        return (
            httpx.Response(307, headers={"location": "http://foreign.test/api/delete"})
            if redirect
            else httpx.Response(200)
        )

    original = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kw: original(transport=httpx.MockTransport(handle), follow_redirects=True, **kw),
    )
    result = CliRunner().invoke(cli, ["models", "remove", "x"])
    assert result.exit_code == 0, result.output
    conn = db.open_db()
    assert operator.entries(conn)[-1].details["command"] == "models remove"
    redirect = True
    result = CliRunner().invoke(cli, ["models", "remove", "x"])
    assert result.exit_code != 0 and "Could not remove" in result.output
    assert len(operator.entries(conn)) == 1
    conn.close()


@pytest.mark.parametrize(
    "free,maximum,expected,risk",
    [
        (15_000_000_000, 32768, 16384, False),
        (10_000_000_000, 32768, 8192, False),
        (4_000_000_000, 32768, 8192, True),
        (15_000_000_000, 4096, 4096, False),
    ],
)
def test_adaptive_context_and_metadata(
    free: int,
    maximum: int,
    expected: int,
    risk: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config(settings={"OLLAMA_MODEL": "eight"})
    monkeypatch.setattr(hub, "memory", lambda: hub.Memory(free, None))
    seen = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path == "/api/tags":
            return httpx.Response(
                200, json={"models": [{"name": "eight:latest", "size": 5_000_000_000}]}
            )
        if request.url.path == "/api/show":
            return httpx.Response(
                200,
                json={
                    "model_info": {
                        "llama.context_length": maximum,
                        "llama.block_count": 32,
                        "llama.embedding_length": 4096,
                        "llama.attention.head_count": 32,
                        "llama.attention.head_count_kv": 32,
                    }
                },
            )
        assert request.url.path == "/api/chat"
        assert json.loads(request.content)["options"]["num_ctx"] == expected
        return httpx.Response(200, json={"message": {"content": "ok"}})

    [choice] = router.lineup(config, CallRole.INTERLOCUTOR)
    llm = router.connect(config, choice, client=httpx.Client(transport=httpx.MockTransport(handle)))
    assert seen == []  # Routing and connection perform no discovery.
    llm.complete(Request(messages=[LLMMessage(role="user", content="hello")]))
    assert llm.context == expected and llm.memory_risk is risk
    assert seen == ["/api/tags", "/api/show", "/api/chat"]
    config.settings["OLLAMA_CONTEXT"] = "65536"
    assert local_models.ollama_context(config, "eight") == (65536, False)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("First sentence. More detail.", "First sentence."),
        ("Question? Explanation", "Question?"),
        ("Do it! Then continue", "Do it!"),
        ("Heading\nMore text", "Heading"),
        ("x" * 120, "x" * 99 + "…"),
        ("version 1.2 is fine", "version 1.2 is fine"),
    ],
)
def test_short_description(text: str, expected: str) -> None:
    assert short_description(text) == expected


def test_context_exact_text_alias_and_redaction(tmp_path: Path) -> None:
    config = make_config(api_keys={"GROQ_API_KEY": "secret-to-redact"})
    open_config().save(config)
    conn = db.open_db(tmp_path / "context.db")
    sessions = Sessions(conn)
    ref = sessions.add(None, Role.USER, "hello secret-to-redact").session_id
    chat = Chat(sessions)
    chat.session_id = ref
    expected = Interlocutor(FakeLLM(), config.language, None, config.behaviour).build_request(
        sessions.history(ref).model_copy(
            update={
                "messages": [
                    *sessions.history(ref).messages,
                    sessions.messages(ref)[0].model_copy(update={"text": "[your next message]"}),
                ]
            }
        )
    )
    output = []
    for suffix in ("", " full"):
        front = WebFront()
        with ui.use(front):
            chat._slash("/context" + suffix)
        [event] = list(front.events)
        assert event["context"] is True
        text = event["html"]
        assert "secret-to-redact" not in text
        assert "[your next message]" in text
        assert "system" in text and "user" in text
        assert "tokens in total" in text
        # Every exact request message is present after HTML escaping and key redaction.
        from html import escape

        from desman.providers import redact_keys

        for message in expected.messages:
            assert escape(redact_keys(message.content, config)[0]) in text
        output.append(text)
    assert output[0] == output[1]
    conn.close()


def test_language_proposal_decline_is_in_next_fake_request(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeLLM().on(
        "Chat turn",
        '{"type":"command","name":"language","args":{"code":"en"}}',
        '{"type":"reply","text":"Hello again"}',
    )
    monkeypatch.setattr(runs, "models", lambda *_a, **_kw: runs.Models(fake, None))
    open_config().save(make_config(language="ru", configured=True))
    result = CliRunner().invoke(cli, [], input="Hello in English\nn\nContinue in English\n/exit\n")
    assert result.exit_code == 0, result.output
    assert open_config().load().language == "ru"
    assert len(fake.calls) == 2
    assert "language_switch_declined" in fake.calls[1].prompt
    assert "do not propose a language switch again" in fake.calls[1].messages[0].content
    conn = db.open_db()
    assert len([e for e in operator.entries(conn) if e.action == "operator_declined"]) == 1
    conn.close()


def test_memory_warning_survives_wrappers_and_ollama_still_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hub, "memory", lambda: hub.Memory(1_000_000_000, None))
    config = make_config(settings={"OLLAMA_MODEL": "eight"})
    local_models._tags_cache["http://localhost:11434"] = (
        0,
        [hub.InstalledModel(name="eight", size=5_000_000_000)],
        True,
    )
    chat_calls = []

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/show":
            return httpx.Response(404)
        chat_calls.append(json.loads(request.content))
        return httpx.Response(500)

    [choice] = router.lineup(config, CallRole.INTERLOCUTOR)
    api = router.connect(config, choice, client=httpx.Client(transport=httpx.MockTransport(handle)))
    wrapped = router.Backoff(api, attempts=1)
    fallback = router.Fallback([(wrapped, choice), (FakeLLM("ok"), None)])
    fallback.complete(Request(messages=[LLMMessage(role="user", content="hello")]))
    assert chat_calls[0]["options"]["num_ctx"] == 8192
    assert fallback.last_call.skipped[0].reason == "memory"


def test_terminal_completion_bindings_pick_first_then_cycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from prompt_toolkit import PromptSession
    from prompt_toolkit.input import DummyInput
    from prompt_toolkit.keys import Keys
    from prompt_toolkit.output import DummyOutput

    original = PromptSession
    monkeypatch.setattr(ui, "_prompt", None)
    monkeypatch.setattr(
        "prompt_toolkit.PromptSession",
        lambda *args, **kw: original(
            *args,
            input=DummyInput(),
            output=DummyOutput(),
            **kw,
        ),
    )
    prompt = ui._chat_prompt()
    actions = []
    buffer = SimpleNamespace(
        complete_state=None,
        start_completion=lambda **kw: actions.append(kw),
        go_to_completion=lambda index: actions.append(index),
    )
    event = SimpleNamespace(current_buffer=buffer)
    bindings = prompt.key_bindings
    bindings.get_bindings_for_keys((Keys.ControlI,))[0].handler(event)
    assert actions == [{"select_first": True}]
    buffer.complete_state = SimpleNamespace(
        completions=["/config", "/context"], complete_index=None
    )
    bindings.get_bindings_for_keys((Keys.ControlI,))[0].handler(event)
    bindings.get_bindings_for_keys((Keys.BackTab,))[0].handler(event)
    assert actions[-2:] == [0, 1]
    buffer.complete_state.complete_index = 1
    bindings.get_bindings_for_keys((Keys.ControlI,))[0].handler(event)
    assert actions[-1] == 0
    bindings.get_bindings_for_keys((Keys.Escape,))[0].handler(event)
    assert buffer.complete_state is None


def test_command_suggestions_start_config_then_context() -> None:
    from desman.chat import _commands

    options = {**_commands(), **commands.completions()}
    assert [name for name, _ in ui.slash_matches("/con", options)][:2] == ["/config", "/context"]
