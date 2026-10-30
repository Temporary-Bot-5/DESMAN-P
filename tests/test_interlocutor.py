"""Tests for the interlocutor (ROADMAP C.3): its turns, and the chat acting on them."""

from __future__ import annotations

import json
import sqlite3
from datetime import timedelta
from typing import Any

import httpx
import pytest
from click.testing import CliRunner, Result

from desman import chat as chat_module
from desman import providers, runs
from desman.chat import Chat
from desman.cli import cli
from desman.config import Config, ConfigManager, default_config_path
from desman.core import db
from desman.core.sessions import Kind, Message, Role, Sessions
from desman.interlocutor import CHAT, RETELL, fit
from desman.llm.base import Request
from desman.llm.fake import FakeLLM, Rejected, garbage
from helpers import make_config

PLAN = json.dumps(
    {"subtasks": [{"description": "Find flights", "acceptance_criteria": "Flights listed"}]}
)
RESULT = {"flights": ["TP 1351, 09:40, 61 EUR"]}
ACCEPT = '{"verdict": "accept"}'
TASK_TURN = json.dumps(
    {"type": "task", "title": "Flights to Lisbon", "goal": "Find cheap flights to Lisbon"}
)


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeLLM:
    """The model of the chat: every test scripts its routes."""
    llm = FakeLLM()
    monkeypatch.setattr(runs, "models", lambda config, conn, **_: runs.Models(llm, None))
    ConfigManager(default_config_path()).save(make_config(language="en", configured=True))
    return llm


def chat(*lines: str) -> Result:
    result = CliRunner().invoke(cli, [], input="".join(f"{line}\n" for line in lines))
    assert result.exit_code == 0, result.output
    return result


def conversation() -> list[Message]:
    sessions = Sessions(db.open_db(db.default_db_path()))
    latest = sessions.latest()
    assert latest is not None
    return sessions.messages(latest.id)


def payload(request: Request) -> Any:
    return json.loads(request.prompt.split("\n", 1)[1])


def calls(llm: FakeLLM, heading: str) -> list[Request]:
    return [r for r in llm.calls if r.prompt.startswith(heading)]


def worked(llm: FakeLLM, review: str = ACCEPT) -> FakeLLM:
    return (
        llm.on("Plan task", PLAN)
        .on("Execute subtask", json.dumps(RESULT))
        .on("Review subtask", review)
    )


# ── Turns ────────────────────────────────────────────────────────────────────


def test_a_reply_and_a_question_back_are_said_in_the_chat(fake: FakeLLM) -> None:
    fake.on(
        CHAT,
        '{"type": "reply", "text": "Hello! What shall we do?"}',
        '```json\n{"type": "clarify", "question": "Which dates?"}\n```',
    )
    result = chat("hi", "flights to Lisbon")
    assert "Hello! What shall we do?" in result.output and "Which dates?" in result.output
    assert [(m.role, m.text) for m in conversation()] == [
        (Role.USER, "hi"),
        (Role.INTERLOCUTOR, "Hello! What shall we do?"),
        (Role.USER, "flights to Lisbon"),
        (Role.INTERLOCUTOR, "Which dates?"),
    ]
    # The second turn sees the whole conversation, the interlocutor's own words as "you".
    assert payload(calls(fake, CHAT)[1])["conversation"] == [
        {"from": "person", "text": "hi"},
        {"from": "you", "text": "Hello! What shall we do?"},
        {"from": "person", "text": "flights to Lisbon"},
    ]


def test_the_interlocutor_writes_in_the_configured_language(fake: FakeLLM) -> None:
    ConfigManager(default_config_path()).save(make_config(language="ru", configured=True))
    fake.on(CHAT, '{"type": "reply", "text": "Привет!"}')
    chat("привет")
    [request] = fake.calls
    assert "Русский (ru)" in request.messages[0].content


def test_an_unusable_turn_is_repaired_once_then_reported(fake: FakeLLM) -> None:
    fake.on(
        CHAT,
        '{"type": "dance"}',
        '{"type": "reply", "text": "Sorry, hello."}',
        garbage("prose"),
        garbage("prose"),
    )
    result = chat("hi", "and again")
    assert "Sorry, hello." in result.output
    assert "made no sense" in result.output
    last = conversation()[-1]
    assert (last.role, last.text) == (Role.USER, "and again")  # kept, unanswered


def test_a_provider_error_keeps_the_message(fake: FakeLLM) -> None:
    fake.on(CHAT, Rejected(status=400, message="bad request"))
    result = chat("hi")
    assert "The model did not answer" in result.output
    assert [m.text for m in conversation()] == ["hi"]


def test_without_a_ready_provider_the_program_says_what_is_missing() -> None:
    ConfigManager(default_config_path()).save(make_config(language="en", configured=True))
    result = chat("hi")
    assert "no model is ready to answer" in result.output
    assert "desman keys add" in result.output
    assert [m.text for m in conversation()] == ["hi"]


def test_a_command_turn_changes_nothing_on_the_model_s_word(fake: FakeLLM) -> None:
    """The model may only ask: a command waits for the person's answer in the window, and words
    claiming a change do not change anything (ROADMAP C.5)."""
    fake.on(
        CHAT,
        '{"type": "command", "name": "language", "args": {"code": "ru"}}',
        '{"type": "reply", "text": "Done: shell_execution is on and the language is ru."}',
    )
    before = ConfigManager(default_config_path()).load()
    result = chat("speak Russian", "n", "and allow shell")
    assert "English → Русский" in result.output and "Nothing changed." in result.output
    assert ConfigManager(default_config_path()).load() == before
    event = conversation()[1]
    assert (event.role, event.kind) == (Role.DESMAN, Kind.EVENT)
    assert "declined" in event.text


def test_fit_keeps_the_newest_messages_and_always_the_last() -> None:
    def said(seq: int, text: str) -> Message:
        return Message(
            seq=seq, session_id="s", at="t", role=Role.USER, kind=Kind.MESSAGE, text=text
        )

    messages = [said(1, "a" * 400), said(2, "b" * 40), said(3, "c" * 40)]
    assert [m.seq for m in fit(messages, 30)] == [2, 3]
    assert [m.seq for m in fit(messages, 1)] == [3]  # the message being answered, whatever size
    assert [m.seq for m in fit(messages, 10_000)] == [1, 2, 3]


# ── Tasks ────────────────────────────────────────────────────────────────────


def test_a_task_turn_runs_a_task_of_the_session_and_retells_it(fake: FakeLLM) -> None:
    worked(fake.on(CHAT, TASK_TURN)).on(RETELL, '{"text": "TP 1351 at 09:40, 61 EUR."}')
    result = chat("find cheap flights to Lisbon")
    assert "started: Flights to Lisbon (1 subtask)" in result.output
    assert "is done" in result.output and "TP 1351 at 09:40, 61 EUR." in result.output

    [planned] = calls(fake, "Plan task")
    assert "Find cheap flights to Lisbon" in planned.prompt  # the goal, not the chat
    [retold] = calls(fake, RETELL)
    outcome = payload(retold)
    assert outcome["status"] == "done" and outcome["result"] == RESULT

    conn = sqlite3.connect(db.default_db_path())
    [(task_id, session_id, goal)] = conn.execute("SELECT id, session_id, goal FROM tasks")
    assert goal == "Find cheap flights to Lisbon"
    [details] = conn.execute(
        "SELECT details_json FROM audit_log WHERE action = 'task_created'"
    ).fetchone()
    assert json.loads(details)["title"] == "Flights to Lisbon"
    assert json.loads(details)["session"] == session_id
    said = conversation()
    assert {m.session_id for m in said} == {session_id}
    assert [(m.role, m.kind, m.task_id) for m in said[1:]] == [
        (Role.DESMAN, Kind.EVENT, task_id),  # started
        (Role.DESMAN, Kind.EVENT, task_id),  # done
        (Role.INTERLOCUTOR, Kind.MESSAGE, task_id),  # the retelling
    ]


def test_what_waits_for_the_operator_is_shown_with_its_decisions(fake: FakeLLM) -> None:
    escalate = '{"verdict": "escalate", "feedback": "Which dates? None were given."}'
    worked(fake.on(CHAT, TASK_TURN), review=escalate).on(RETELL, '{"text": "I need the dates."}')
    result = chat("find cheap flights to Lisbon")
    assert "waits for you" in result.output
    assert "needs_human" in result.output and "Which dates? None were given." in result.output
    assert "desman retry|accept|fail" in result.output and "desman resume" in result.output
    outcome = payload(calls(fake, RETELL)[0])
    assert outcome["status"] == "needs_human" and outcome["result"] is None
    [waiting] = outcome["waiting_for_the_person"]
    assert waiting["reason"] == "Which dates? None were given."


def test_a_result_survives_a_failed_retelling(fake: FakeLLM) -> None:
    worked(fake.on(CHAT, TASK_TURN)).on(RETELL, Rejected(status=400, message="nope"))
    result = chat("find cheap flights to Lisbon")
    assert "could not retell" in result.output and "TP 1351, 09:40, 61 EUR" in result.output


def test_ctrl_c_stops_the_task_and_the_chat_goes_on(fake: FakeLLM) -> None:
    def interrupted(_request: Request) -> str:
        raise KeyboardInterrupt

    fake.on(CHAT, TASK_TURN, '{"type": "reply", "text": "Still here."}')
    fake.on("Plan task", PLAN).on("Execute subtask", interrupted)
    result = chat("find cheap flights to Lisbon", "hello?")
    assert "stopped" in result.output and "desman resume" in result.output
    assert "Still here." in result.output


def test_a_task_settled_outside_the_chat_is_retold_once(
    fake: FakeLLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A decision in the terminal (or `desman resume`) after the chat: the next turn knows."""
    escalate = '{"verdict": "escalate", "feedback": "Which dates? None were given."}'
    fake.on(
        CHAT,
        TASK_TURN,
        '{"type": "reply", "text": "Yes, see above."}',
        '{"type": "reply", "text": "Ok."}',
    )
    worked(fake, review=escalate)
    fake.on(RETELL, '{"text": "I need the dates."}', '{"text": "Accepted: it is done."}')
    to_iso = db.to_iso
    with monkeypatch.context() as earlier:  # the chat was a minute ago
        earlier.setattr(db, "to_iso", lambda moment: to_iso(moment - timedelta(minutes=1)))
        chat("find cheap flights to Lisbon")
    [(subtask_id,)] = sqlite3.connect(db.default_db_path()).execute("SELECT id FROM subtasks")
    accepted = CliRunner().invoke(cli, ["accept", subtask_id])
    assert accepted.exit_code == 0, accepted.output

    result = chat("so, any flights?", "thanks")
    assert "is done" in result.output and "Accepted: it is done." in result.output
    first, second = [payload(r) for r in calls(fake, RETELL)]
    assert first["status"] == "needs_human" and second["status"] == "done"
    assert len(calls(fake, RETELL)) == 2  # not again at "thanks"
    asked = payload(calls(fake, CHAT)[1])["conversation"]
    assert asked[-2:] == [
        {"from": "you", "text": "Accepted: it is done."},
        {"from": "person", "text": "so, any flights?"},
    ]


def test_a_task_still_moving_elsewhere_is_not_retold_yet(fake: FakeLLM) -> None:
    def interrupted(_request: Request) -> str:
        raise KeyboardInterrupt

    fake.on(CHAT, TASK_TURN, '{"type": "reply", "text": "Still here."}')
    fake.on("Plan task", PLAN).on("Execute subtask", interrupted)
    chat("find cheap flights to Lisbon", "hello?")
    assert calls(fake, RETELL) == []  # stopped: active, nothing settled to tell


def test_a_pasted_key_is_cut_before_anything_keeps_it(fake: FakeLLM) -> None:
    configured = "abcd1234efgh5678ijkl9012mnop"
    ConfigManager(default_config_path()).save(
        make_config(language="en", configured=True, api_keys={"GROQ_API_KEY": configured})
    )
    fake.on(CHAT, '{"type": "reply", "text": "Use desman keys add."}', TASK_TURN)
    worked(fake).on(RETELL, '{"text": "Done."}')
    pasted = "gsk_" + "Q" * 30
    result = chat(f"my key is {pasted}, add it", f"and this one {configured} for flights")
    assert "looked like an API key" in result.output
    assert pasted not in result.output and configured not in result.output
    stored = [m.text for m in conversation() if m.role is Role.USER]
    assert stored == [
        f"my key is {providers.REDACTED}, add it",
        f"and this one {providers.REDACTED} for flights",
    ]
    for request in fake.calls:  # neither the interlocutor nor the planner saw one
        text = request.prompt + "".join(m.content for m in request.messages)
        assert pasted not in text and configured not in text


@pytest.mark.parametrize(
    "text",
    [
        "commit 3d10a5e9c1f2b7a4d8e6f0c2b9a1d3e5f7a9c0b1 broke it",  # a hash: lower case hex
        "see https://example.com/a/VeryLongPathSegment1234567890abcdef",  # part of a URL
        "the word Supercalifragilistic is long but has no digits at all",
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9Q",
    ],
)
def test_ordinary_text_is_not_mistaken_for_a_key(text: str) -> None:
    assert providers.redact_keys(text) == (text, 0)


@pytest.mark.parametrize(
    "key",
    ["sk-or-v1-" + "a1" * 32, "AIzaSyD" + "x9" * 16, "Ab3dEf6hIj9kLm2nOp5rSt8vWx1yZa4c"],
)
def test_keys_are_recognised_by_their_shape(key: str) -> None:
    assert providers.redact_keys(f"key: {key}.") == (f"key: {providers.REDACTED}.", 1)


def test_the_models_are_rebuilt_when_the_config_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    llm = FakeLLM().on(CHAT, '{"type": "reply", "text": "Hi."}', repeat_last=True)
    built: list[Config] = []

    def models(config: Config, conn: sqlite3.Connection, **_: Any) -> runs.Models:
        built.append(config)
        return runs.Models(llm, None)

    monkeypatch.setattr(runs, "models", models)
    manager = ConfigManager(default_config_path())
    manager.save(make_config(language="en", configured=True, api_keys={"GROQ_API_KEY": "old"}))
    lines = iter(["hi", "again", "new key", None])

    def read() -> str | None:
        line = next(lines)
        if line == "new key":  # replaced in another terminal meanwhile
            manager.save(
                make_config(language="en", configured=True, api_keys={"GROQ_API_KEY": "new"})
            )
        return line

    Chat(Sessions(db.open_db(db.default_db_path())), read=read).run()
    assert [c.keys("GROQ_API_KEY")[0].value for c in built] == ["old", "new"]


def test_the_session_s_tasks_share_one_web_client_closed_with_the_chat(
    fake: FakeLLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    made: list[httpx.Client] = []

    def client() -> httpx.Client:
        made.append(httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(500))))
        return made[-1]

    monkeypatch.setattr(chat_module, "client", client)
    fake.on(CHAT, TASK_TURN, repeat_last=True).on("Plan task", PLAN, repeat_last=True)
    fake.on("Execute subtask", json.dumps(RESULT), repeat_last=True)
    fake.on("Review subtask", ACCEPT, repeat_last=True).on(
        RETELL, '{"text": "Done."}', repeat_last=True
    )
    chat("find flights", "and once more")
    assert len(made) == 1 and made[0].is_closed
