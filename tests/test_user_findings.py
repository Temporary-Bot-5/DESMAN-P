"""Regressions from T.6c's real user sessions, without live provider calls."""

from __future__ import annotations

import io
import json
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

import click
import httpx
import pytest
from click.testing import CliRunner

from conftest import FakeClock
from desman import actions, providers
from desman.agents import llm_executor, llm_reviewer
from desman.cli import cli
from desman.core import db
from desman.core.context import build_context
from desman.core.coordinator import Coordinator, Failure, Work
from desman.core.models import SubtaskSpec, TaskSpec
from desman.core.report import task_report
from desman.core.retry import FailureKind
from desman.core.sessions import Role, Sessions
from desman.core.store import Store
from desman.core.tools import Toolbox
from desman.core.workspace import Workspace, workspace_root_for
from desman.interlocutor import Interlocutor, Reply, wrong_script
from desman.llm.base import Message, ProviderRejected, ProviderUnavailable, Request
from desman.llm.fake import FakeLLM, Rejected
from desman.llm.limiter import Buckets, Limited
from desman.llm.openai_compat import OpenAICompatible
from desman.llm.router import Choice, Fallback
from desman.planner import plan

REQUEST = Request(messages=[Message(role="user", content="hello")])


def spec() -> TaskSpec:
    return TaskSpec(
        goal="Answer", subtasks=[SubtaskSpec(description="work", acceptance_criteria="done")]
    )


def test_model_404_is_shared_and_expires(tmp_path: Path, clock: FakeClock) -> None:
    store = Store(db.open_db(tmp_path / "shared.db"), clock=clock)
    dead = FakeLLM(Rejected(status=404), '{"back": true}')
    alive = FakeLLM('{"ok": true}', '{"ok": true}')
    buckets = Buckets(store.conn, clock)
    choice = Choice(providers.find("groq"), "missing")
    first = Limited(dead, buckets, "groq/missing", 30, sleep=clock.sleep)
    assert Fallback([(first, choice), (alive, None)]).complete(REQUEST).text == '{"ok": true}'
    assert buckets.blocked_until("groq/missing") == clock() + timedelta(hours=24)
    # A new connection and fallback represent a later run in another process.
    path = Path(store.conn.execute("PRAGMA database_list").fetchone()[2])
    conn = db.open_db(path)
    try:
        second = Limited(dead, Buckets(conn, clock), "groq/missing", 30, sleep=clock.sleep)
        Fallback([(second, choice), (alive, None)]).complete(REQUEST)
        assert len(dead.calls) == 1
        clock.sleep(24 * 60 * 60)
        assert second.complete(REQUEST).text == '{"back": true}'
        assert len(dead.calls) == 2
    finally:
        conn.close()


def test_key_401_does_not_block_model(store: Store, clock: FakeClock) -> None:
    llm = Limited(FakeLLM(Rejected(status=401)), Buckets(store.conn, clock), "groq/m", 30)
    with pytest.raises(ProviderRejected):
        llm.complete(REQUEST)
    assert llm.buckets.blocked_until("groq/m") is None


def finish(store: Store, content: dict[str, Any]) -> str:
    task = store.create_task(spec())
    llm = FakeLLM().on("Execute", json.dumps(content)).on("Review", '{"verdict":"accept"}')
    Coordinator(store, llm_executor(llm), llm_reviewer(llm)).run(task)
    return task


def test_done_report_result_files_and_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store = Store(db.open_db(tmp_path / "results.db"))
    task = finish(store, {"content": "The answer"})
    Workspace(workspace_root_for(store.conn), task).write("answer.txt", b"The answer")
    report = task_report(store, task)
    actions.print_report(report, store)
    output = capsys.readouterr().out
    assert "The answer" in output and "Files in " in output and "answer.txt" in output
    path = store.conn.execute("PRAGMA database_list").fetchone()[2]
    result = CliRunner().invoke(cli, ["status", task, "--db", path, "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["result"] == {"content": "The answer"}
    assert data["files"] == [{"path": "answer.txt", "bytes": 10}]
    assert data["folder"] == str(workspace_root_for(store.conn) / task)


def test_pending_report_has_no_result_or_files(
    store: Store, capsys: pytest.CaptureFixture[str]
) -> None:
    task = store.create_task(spec())
    Workspace(workspace_root_for(store.conn), task).write("hidden.txt", b"secret")
    report = task_report(store, task)
    actions.print_report(report, store)
    assert "hidden.txt" not in capsys.readouterr().out
    assert actions.report_output(store, report) == {}


def test_result_line_limit(store: Store, capsys: pytest.CaptureFixture[str]) -> None:
    task = finish(store, {"content": "\n".join(f"answer-{i}" for i in range(45))})
    actions.print_report(task_report(store, task), store)
    output = capsys.readouterr().out
    assert "answer-39" in output and "answer-40" not in output
    assert "Omitted lines: 5." in output


def test_executor_echo_is_quality_failure(store: Store, tmp_path: Path, clock: FakeClock) -> None:
    store.create_task(spec())
    sub = store.claim("w")
    assert sub is not None
    work = Work(
        subtask=sub,
        context=build_context(sub, [], None),
        heartbeat=lambda _: None,
        tools=Toolbox(Workspace(tmp_path, sub.task_id)),
    )
    llm = FakeLLM('{"subtask":"work","acceptance_criteria":"done","inputs":{}}')
    result = llm_executor(llm, clock=clock)(work)
    assert isinstance(result, Failure) and result.kind is FailureKind.QUALITY
    assert result.reason == "the reply repeated the request instead of doing the work"
    assert f"Today is {clock().date()}." in llm.calls[0].messages[0].content


def test_planner_and_reviewer_date_and_language(store: Store, clock: FakeClock) -> None:
    llm = FakeLLM(
        '{"subtasks":[{"description":"work","acceptance_criteria":"done"}]}', '{"verdict":"accept"}'
    )
    plan(llm, "Напиши ответ", clock=clock)
    store.create_task(spec())
    sub = store.claim("w")
    assert sub is not None
    llm_reviewer(llm, evidence=lambda _: {}, clock=clock)(sub, {"content": "answer"})
    assert all(f"Today is {clock().date()}." in r.messages[0].content for r in llm.calls)
    assert "naming the language" in llm.calls[0].messages[0].content
    assert "only the evidence decides" in llm.calls[1].messages[0].content


@pytest.mark.parametrize(
    "message, answer",
    [
        ("Привет мир", "Hello"),
        ("你好世界", "Hello"),
        ("안녕하세요", "Hello"),
        ("नमस्ते दुनिया", "Hello"),
        ("مرحبا بالعالم", "Hello"),
    ],
)
def test_script_mismatch(message: str, answer: str) -> None:
    assert wrong_script(message, answer)
    assert not wrong_script(message, message)
    assert not wrong_script("Hello", "Привет")


def test_interlocutor_retries_language_once(store: Store, clock: FakeClock) -> None:
    sessions = Sessions(store.conn, clock=clock)
    message = sessions.add(None, Role.USER, "Привет мир")
    llm = FakeLLM().on(
        "Chat turn", '{"type":"reply","text":"Hello"}', '{"type":"reply","text":"Привет"}'
    )
    result = Interlocutor(llm, "ru", clock=clock).turn(sessions.history(message.session_id))
    assert isinstance(result, Reply) and result.text == "Привет"
    assert len(llm.calls) == 2
    assert "Answer in the language" in llm.calls[1].messages[-1].content
    assert f"Today is {clock().date()}." in llm.calls[0].messages[0].content


def test_cli_reconfigures_piped_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = io.TextIOWrapper(
        io.BytesIO("Привет".encode()), encoding="cp1252", errors="surrogateescape"
    )
    monkeypatch.setattr(sys, "stdin", stream)
    monkeypatch.setattr(click.Group, "main", lambda *a, **kw: None)
    cli.main([])
    assert stream.encoding == "utf-8" and stream.errors == "replace"
    assert stream.read() == "Привет"


@pytest.mark.parametrize(
    "message",
    [
        {"role": "assistant", "tool_calls": [{"secret": "never print"}]},
        {"role": "assistant", "reasoning_content": "never print", "content": None},
    ],
)
def test_empty_reply_metadata(message: dict[str, Any]) -> None:
    response = {"choices": [{"message": message, "finish_reason": "MAX_TOKENS"}]}
    llm = OpenAICompatible(
        "groq",
        "https://example.test/",
        ["key"],
        "m",
        client=httpx.Client(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=response))
        ),
    )
    with pytest.raises(ProviderUnavailable) as caught:
        llm.complete(REQUEST)
    text = str(caught.value)
    assert "finish_reason=MAX_TOKENS" in text and "message keys:" in text
    assert "never print" not in text
    if "tool_calls" in message:
        assert "a native tool call instead of text" in text


def test_repeated_request_never_reaches_reviewer(store: Store, clock: FakeClock) -> None:
    task = store.create_task(spec())
    llm = (
        FakeLLM()
        .on("Execute", '{"subtask":"work","acceptance_criteria":"done"}', '{"content":"finished"}')
        .on("Review", '{"verdict":"accept"}')
    )
    Coordinator(
        store, llm_executor(llm, clock=clock), llm_reviewer(llm, clock=clock), sleep=clock.sleep
    ).run(task)
    reviews = [r for r in llm.calls if r.prompt.startswith("Review")]
    assert len(reviews) == 1 and "finished" in reviews[0].prompt
    attempts = [r for r in llm.calls if r.prompt.startswith("Execute")]
    assert "the reply repeated the request instead of doing the work" in attempts[1].prompt


@pytest.mark.parametrize("kind, field", [("reply", "text"), ("clarify", "question")])
def test_language_retry_stops_after_one_failure(store: Store, kind: str, field: str) -> None:
    sessions = Sessions(store.conn)
    message = sessions.add(None, Role.USER, "Привет мир")
    llm = FakeLLM(
        json.dumps({"type": kind, field: "Hello"}),
        json.dumps({"type": kind, field: "Still English"}),
    )
    result = Interlocutor(llm, "ru").turn(sessions.history(message.session_id))
    assert getattr(result, field) == "Still English" and len(llm.calls) == 2


def test_non_content_result_is_indented_json(
    store: Store, capsys: pytest.CaptureFixture[str]
) -> None:
    task = finish(store, {"answer": [1, 2]})
    actions.print_report(task_report(store, task), store)
    assert '"answer": [\n    1,\n    2\n  ]' in capsys.readouterr().out


def test_reasoning_is_not_the_reply() -> None:
    response = {
        "choices": [
            {"message": {"content": '{"answer":42}', "reasoning_content": "private reasoning"}}
        ]
    }
    llm = OpenAICompatible(
        "nvidia",
        "https://example.test/",
        ["key"],
        "m",
        client=httpx.Client(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=response))
        ),
    )
    assert llm.complete(REQUEST).text == '{"answer":42}'


def test_usage_reports_active_model_blocks(
    tmp_path: Path,
    clock: FakeClock,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from datetime import datetime

    from desman import config_cli

    class FixedDate(datetime):
        @classmethod
        def now(cls, tz: Any = None) -> FixedDate:
            return cls.fromtimestamp(clock().timestamp(), tz)

    monkeypatch.setattr(config_cli, "datetime", FixedDate)
    path = tmp_path / "limits.db"
    conn = db.open_db(path)
    until = clock() + timedelta(hours=24)
    Buckets(conn, clock).block("nvidia/missing", until)
    conn.close()
    config_cli.print_model_blocks(path)
    output = capsys.readouterr().out
    assert "nvidia/missing" in output and until.date().isoformat() in output
    clock.sleep(24 * 60 * 60)
    config_cli.print_model_blocks(path)
    assert capsys.readouterr().out == ""


def test_session_storage_replaces_lone_surrogates(store: Store) -> None:
    sessions = Sessions(store.conn)
    message = sessions.add(None, Role.USER, "hello \udc8f", styled="hello \udc8f")
    assert message.text == "hello ?" and message.styled == "hello ?"
    assert sessions.messages(message.session_id)[0].text == "hello ?"


def test_language_correction_keeps_looked_up_evidence(store: Store) -> None:
    sessions = Sessions(store.conn)
    message = sessions.add(None, Role.USER, "Расскажи о задачах")
    llm = FakeLLM(
        '{"type":"lookup","nodes":["tasks"]}',
        '{"type":"reply","text":"Tasks are durable"}',
        '{"type":"reply","text":"Задачи сохраняются"}',
    )
    result = Interlocutor(llm, "ru").turn(sessions.history(message.session_id))
    assert isinstance(result, Reply) and result.text == "Задачи сохраняются"
    assert "knowledge" in llm.calls[-1].messages[1].content
    assert "DESMAN/Manual/tasks" in llm.calls[-1].messages[1].content


def test_a_result_is_shown_as_its_text_or_as_indented_json() -> None:
    from desman.actions import result_text

    assert result_text({"content": "hello", "files_written": []}) == "hello"
    assert result_text({"text": "привет"}) == "привет"
    assert result_text({"type": "result", "result": {"answer": "42"}}) == "42"
    assert result_text({"names": ["a"]}) == '{\n  "names": [\n    "a"\n  ]\n}'
