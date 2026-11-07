"""Shared model discovery, explicit routing and command/context regressions for T.2k."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from click.testing import CliRunner

from desman import commands, providers, tuning_cli, ui
from desman.chat import Chat, _commands
from desman.cli import cli
from desman.config import open_config
from desman.core import calls, db
from desman.core.sessions import History, Role, Sessions
from desman.gui.front import WebFront
from desman.gui.settings import _MODEL_CACHE, source_models
from desman.interlocutor import Interlocutor
from desman.llm import check, router
from desman.llm.base import Message, Request
from desman.llm.fake import FakeLLM, Rejected
from desman.llm.openai_compat import key_id
from helpers import make_config


def test_lazy_source_listing_auth_filter_order_cache() -> None:
    config = make_config(
        api_keys={"GROQ_API_KEY": [{"value": "first-secret"}, {"value": "second-secret"}]}
    )
    _MODEL_CACHE.clear()
    requests: list[httpx.Request] = []

    def response(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.method == "GET" and request.url.path.endswith("/models")
        assert request.headers["authorization"] == "Bearer second-secret"
        return httpx.Response(
            200,
            json={
                "data": [
                    {"id": x}
                    for x in [
                        "new-chat",
                        "whisper-1",
                        "tts-1",
                        "embed-small",
                        "guard-1",
                        "moderation-1",
                        "rerank-1",
                        "image-generation-1",
                        "vision-generation-1",
                        "openai/gpt-oss-20b",
                    ]
                ]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(response)) as client:
        result = source_models(config, "groq", key_id("second-secret"), client=client)
        again = source_models(config, "groq", key_id("second-secret"), client=client)
    assert result == again and len(requests) == 1
    assert [row["value"] for row in result["models"]] == [
        "groq/openai/gpt-oss-20b",
        "groq/new-chat",
    ]
    assert result["models"][0]["known"] and not result["models"][1]["known"]
    assert "secret" not in json.dumps(result)


def test_listing_failure_uses_catalog_and_safe_note() -> None:
    config = make_config(api_keys={"GROQ_API_KEY": "list-failure-secret"})
    _MODEL_CACHE.clear()
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(404, text="list-failure-secret"))
    ) as client:
        result = source_models(config, "groq", key_id("list-failure-secret"), client=client)
        assert result["models"] and result["note"]
        assert "list-failure-secret" not in json.dumps(result)
        with pytest.raises(ValueError, match="unknown source"):
            source_models(config, "groq", "unknown", client=client)


def test_check_and_picker_share_listing(monkeypatch: pytest.MonkeyPatch) -> None:
    config = make_config(api_keys={"GROQ_API_KEY": "shared-listing-secret"})
    _MODEL_CACHE.clear()
    seen: list[str] = []
    original = check.models_response

    def listing(*args: Any, **kwargs: Any) -> httpx.Response:
        seen.append("listed")
        return original(*args, **kwargs)

    monkeypatch.setattr(check, "models_response", listing)
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"data": []}))
    ) as client:
        assert check.check(config, providers.find("groq"), client)[0].ok
        source_models(config, "groq", key_id("shared-listing-secret"), client=client)
    assert len(seen) == 2


def test_page_role_data_has_no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("role snapshots must not discover models")

    monkeypatch.setattr(httpx.Client, "get", forbidden)
    assert tuning_cli.role_rows(make_config(api_keys={"GROQ_API_KEY": "snapshot-secret"}))


@pytest.mark.parametrize("role", [calls.Role.INTERLOCUTOR, calls.Role.PLANNER, calls.Role.EXECUTOR])
def test_unknown_chosen_model_goes_first(role: calls.Role) -> None:
    config = make_config(api_keys={"GROQ_API_KEY": "routing-secret"})
    name = router.tuning_field(role)
    assert name
    setattr(config.tuning.models, name, "groq/never-in-catalog")
    choices = router.lineup(config, role)
    assert choices[0].label == "groq/never-in-catalog"
    assert len({c.label for c in choices}) == len(choices)
    assert all(c.provider.id == "groq" for c in choices)


def test_fallback_exposes_safe_last_call_and_resets() -> None:
    provider = providers.find("groq")
    first = FakeLLM(Rejected(status=404), '{"ok":true}', name="first")
    second = FakeLLM('{"ok":true}', name="second")
    llm = router.Fallback(
        [(first, router.Choice(provider, "first")), (second, router.Choice(provider, "second"))]
    )
    request = Request(messages=[Message(role="user", content="hello")])
    reply = llm.complete(request)
    assert llm.last_call.answered == f"{reply.provider}/{reply.model}"
    assert llm.last_call.skipped == (router.SkippedModel("groq/first", "refused", 404),)
    llm.complete(request)
    assert not llm.last_call.skipped


def test_context_builder_is_the_real_turn_request() -> None:
    fake = FakeLLM('{"type":"reply","text":"hello"}')
    interlocutor = Interlocutor(fake, "en")
    history = History(summary=None, messages=[])
    expected = interlocutor.build_request(history)
    interlocutor.turn(history)
    assert fake.calls[0] == expected


def test_chat_context_and_clear_keep_history_without_model(tmp_path: Path) -> None:
    conn = db.open_db(tmp_path / "chat.db")
    config = make_config(api_keys={"GROQ_API_KEY": "context-secret"})
    open_config().save(config)
    sessions = Sessions(conn)
    ref = sessions.add(None, Role.USER, "kept message").session_id
    chat = Chat(sessions)
    chat.session_id = ref
    front = WebFront()
    before = sessions.messages(ref)
    with ui.use(front):
        chat._slash("/context full")
        output = " ".join(str(e) for e in front.events)
        assert "kept message" in output and "context-secret" not in output
        assert "groq/" in output and "tokens in total" in output
        chat._slash("/clear")
    assert sessions.messages(ref) == before
    assert any(e["type"] == "reset" for e in front.events)
    assert chat.models is None
    conn.close()


def test_command_names_do_not_collide() -> None:
    roots = {entry.name.split()[0] for entry in commands.CATALOG}
    assert roots & {name[1:] for name in _commands()} == {"help"}
    assert "/task pause" in commands.completions()


def test_catalog_parses_bulk_and_hidden_flags() -> None:
    for line, field in [
        ("list --all", "all"),
        ("task pause --all", "all"),
        ("task hide --finished", "finished"),
    ]:
        parsed = commands.parse_line(line, None)
        assert parsed and getattr(parsed[1], field) is True


def test_shared_help_entries_shell_chat_and_terminal() -> None:
    shell = CliRunner().invoke(cli, ["help"])
    assert shell.exit_code == 0, shell.output
    for terminal in (False, True):
        front = WebFront()
        with ui.use(front):
            commands.print_help(terminal=terminal)
        rendered = " ".join(str(e) for e in front.events)
        for entry in commands.CATALOG:
            assert f"desman {entry.name}" in shell.output
            assert f"/{entry.name}" in rendered
        assert "/clear" in rendered and "/context" in rendered


def test_fallback_note_is_visible_in_chat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from desman import runs

    config = make_config(api_keys={"GROQ_API_KEY": "fallback-note-secret"})
    open_config().save(config)
    conn = db.open_db(tmp_path / "notes.db")
    sessions = Sessions(conn)
    chat = Chat(sessions)
    provider = providers.find("groq")
    first = FakeLLM(Rejected(status=404, message="fallback-note-secret"), name="refused")
    second = FakeLLM('{"type":"reply","text":"hello"}', name="answering")
    llm = router.Fallback(
        [(first, router.Choice(provider, "first")), (second, router.Choice(provider, "second"))]
    )
    monkeypatch.setattr(
        chat, "_models", lambda _: runs.Models(second, None, roles={calls.Role.INTERLOCUTOR: llm})
    )
    front = WebFront()
    with ui.use(front):
        chat._message("hello")
    assert chat.session_id
    messages = sessions.messages(chat.session_id)
    assert messages[-2].text == "hello"
    assert "Answered by answering/" in messages[-1].text
    assert "groq/first: no such model or no access" in messages[-1].text
    assert "fallback-note-secret" not in messages[-1].text
    assert any("Answered by" in str(e) for e in front.events)
    conn.close()


def test_context_profile_is_opt_in_and_new_chat_needs_no_model(tmp_path: Path) -> None:
    from desman.core.profile import Author

    conn = db.open_db(tmp_path / "context.db")
    config = make_config(api_keys={"GROQ_API_KEY": "profile-context-secret"})
    chat = Chat(Sessions(conn))
    chat.profile.add("unique profile preference", Author.PERSON)
    for enabled in (False, True):
        config.personalize = enabled
        open_config().save(config)
        front = WebFront()
        with ui.use(front):
            chat._slash("/context full")
        text = " ".join(str(e) for e in front.events)
        assert ("unique profile preference" in text) is enabled
        assert "profile-context-secret" not in text
        assert chat.session_id is None and chat.models is None
    conn.close()


def test_bulk_preview_does_not_capture_new_tasks_or_ignore_new_pins(tmp_path: Path) -> None:
    from desman.core.store import Store
    from test_task_controls import task

    conn = db.open_db(tmp_path / "bulk.db")
    store = Store(conn)
    original = task(store)
    context = commands.Context(store, open_config(), lambda *_: None, lambda _: None)
    entry = commands.find("task pause")
    assert entry
    preview = entry.handler(context, commands.TaskPauseArgs(all=True))
    assert isinstance(preview, commands.Proposal)
    newcomer = task(store)
    store.pin_task(original)
    outcome = preview.apply()
    assert outcome.data["count"] == 0
    assert store.find_task(newcomer)["paused_at"] is None
    assert store.find_task(original)["paused_at"] is None
    conn.close()


def test_pinned_paused_task_and_last_success_are_in_page(tmp_path: Path) -> None:
    from desman.core.store import Store
    from desman.gui.panels import snapshot
    from test_live import call
    from test_task_controls import task

    path = tmp_path / "panels.db"
    conn = db.open_db(path)
    config = make_config(api_keys={"GROQ_API_KEY": "panel-model-secret"})
    open_config().save(config)
    store = Store(conn)
    ref = task(store)
    store.pin_task(ref)
    store.pause_task(ref)
    with calls.for_(
        calls.Purpose(calls.Role.EXECUTOR, task_id=ref, subtask_id=store.list_subtasks(ref)[0].id)
    ):
        calls.recorder(conn)(call(model="fallback-answer"))
        calls.recorder(conn)(call(model="later-failure", error_kind="transient"))
    data = snapshot(path, "tasks", WebFront())
    [row] = data["tasks"]
    assert row["pinned"] and row["paused_at"]
    text = next(card for card in row["cards"] if card["role"] == "text")
    assert text["pinned"] and text["last_model"].endswith("/fallback-answer")
    conn.close()


def test_discovered_unknown_model_applies_through_both_strict_editors(tmp_path: Path) -> None:
    from desman.core import choices
    from desman.core.store import Store
    from test_task_controls import task

    config = make_config(api_keys={"GROQ_API_KEY": "dynamic-model-secret"})
    open_config().save(config)
    fingerprint = key_id("dynamic-model-secret")
    _MODEL_CACHE.clear()
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json={"data": [{"id": "brand-new-chat"}]})
        )
    ) as client:
        source_models(config, "groq", fingerprint, client=client)
    conn = db.open_db(tmp_path / "dynamic.db")
    chat = Chat(Sessions(conn))
    front = WebFront()
    with ui.use(front):
        chat._settings_action(
            ui.SettingsAction.model_validate(
                {
                    "type": "settings",
                    "id": "dynamic",
                    "change": {
                        "section": "roles",
                        "role": "interlocutor",
                        "model": "groq/brand-new-chat",
                        "key": fingerprint,
                    },
                }
            )
        )
        assert open_config().load().tuning.models.interlocutor == "groq/brand-new-chat"
        ref = task(Store(conn))
        chat._module_action(
            ui.ModuleAction(
                type="module",
                scope="task",
                scope_id=ref,
                role="text",
                model="groq/brand-new-chat",
                key=fingerprint,
            )
        )
    assert choices.get(conn, "task", ref, "text").model == "groq/brand-new-chat"
    conn.close()


def test_unknown_chosen_model_is_tried_before_known_large_context() -> None:
    provider = providers.find("groq")
    first = FakeLLM("chosen", name="chosen")
    second = FakeLLM("fallback", name="fallback")
    llm = router.Fallback(
        [
            (first, router.Choice(provider, "brand-new-chat")),
            (second, router.Choice(provider, "openai/gpt-oss-20b")),
        ]
    )
    request = Request(messages=[Message(role="user", content="x" * 30_000)])
    assert llm.complete(request).text == "chosen"
    assert first.calls and not second.calls


def test_worker_preview_does_not_resume_a_new_operator_pause(tmp_path: Path) -> None:
    from desman.core.store import Store
    from test_task_controls import task

    conn = db.open_db(tmp_path / "worker-pause.db")
    store = Store(conn)
    ref = task(store)
    started: list[str] = []
    context = commands.Context(store, open_config(), lambda *_: None, started.append)
    entry = commands.find("worker")
    assert entry
    preview = entry.handler(context, commands.NoArgs())
    assert isinstance(preview, commands.Proposal)
    store.pause_task(ref)
    preview.apply()
    assert not started
    assert store.find_task(ref)["paused_at"] is not None
    conn.close()


def test_worker_reports_operator_pause_once_while_idle(tmp_path: Path) -> None:
    from desman.cli import _worker_progress
    from desman.core.coordinator import Coordinator, Review, Success, Verdict
    from desman.core.report import InboxWatch
    from desman.core.store import Store
    from test_task_controls import task

    conn = db.open_db(tmp_path / "worker-notice.db")
    store = Store(conn)
    ref = task(store)
    worker = Coordinator(
        store, lambda _: Success(content={}), lambda *_: Review(verdict=Verdict.ACCEPT)
    )
    progress = _worker_progress(worker, InboxWatch(store))
    store.pause_task(ref)
    front = WebFront()
    with ui.use(front):
        progress(None)
        progress(None)
    text = " ".join(str(e) for e in front.events)
    assert text.count("paused") == 1
    assert "desman resume" in text and store.short_id(ref) in text
    conn.close()


def test_context_says_what_the_next_message_changes_first(tmp_path: Path) -> None:
    from desman.core.models import SubtaskSpec, TaskSpec

    conn = db.open_db(tmp_path / "pending.db")
    open_config().save(make_config(api_keys={"GROQ_API_KEY": "pending-secret"}))
    sessions = Sessions(conn)
    ref = sessions.add(None, Role.USER, "make it").session_id
    chat = Chat(sessions)
    chat.session_id = ref
    task = chat.store.create_task(
        TaskSpec(goal="g", subtasks=[SubtaskSpec(description="d", acceptance_criteria="a")]),
        session_id=ref,
    )
    chat.store.cancel_task(task, "settled while the chat looked away")
    sessions.add(ref, Role.INTERLOCUTOR, "x " * 20000)  # over the history budget
    front = WebFront()
    with ui.use(front):
        chat._slash("/context")
    output = " ".join(str(e) for e in front.events)
    assert "what became of these tasks is added" in output
    assert chat.store.short_id(task) in output
    assert "older messages are summarised" in output
