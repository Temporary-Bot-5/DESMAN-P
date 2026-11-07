"""The permanent command console is isolated from conversations and models."""

from __future__ import annotations

import json
import sqlite3
import sys
from collections.abc import Iterator
from pathlib import Path

import click
import pytest
from rich.table import Table

from conftest import FakeClock
from desman import runs, ui
from desman.chat import Chat
from desman.config import Config, open_config
from desman.core import db
from desman.core.sessions import Kind, Role, Sessions
from desman.gui.front import WebFront
from desman.gui.panels import snapshot
from desman.llm.fake import FakeLLM


@pytest.fixture
def sessions(clock: FakeClock) -> Iterator[Sessions]:
    conn = db.open_db(":memory:")
    open_config().save(Config(language="en", configured=True))
    yield Sessions(conn, clock=clock)
    conn.close()


def test_terminal_upgrade_and_constraints(tmp_path: Path) -> None:
    path = tmp_path / "old.db"
    conn = db.connect(path)
    db.migrate(conn, db.MIGRATIONS[:-1])
    sessions = Sessions(conn)
    conn.execute("INSERT INTO sessions (id, created_at) VALUES ('old', '2026-01-01')")
    conn.close()
    conn = db.open_db(path)
    sessions = Sessions(conn)
    with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
        conn.execute(
            "INSERT INTO sessions (id, created_at, terminal, temporary) VALUES ('bad', '', 1, 1)"
        )
    terminal = sessions.terminal()
    assert sessions.terminal() == terminal
    assert conn.execute("SELECT terminal FROM sessions WHERE id = 'old'").fetchone()[0] == 0
    for sql in (
        "INSERT INTO sessions (id, created_at, terminal) VALUES ('second', '', 1)",
        "UPDATE sessions SET temporary = 1 WHERE terminal = 1",
        "UPDATE sessions SET terminal = 2 WHERE terminal = 1",
    ):
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(sql)
    message = sessions.add(terminal, Role.USER, "list")
    for sql in (
        "UPDATE messages SET text = 'x' WHERE seq = ?",
        "DELETE FROM messages WHERE seq = ?",
    ):
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(sql, (message.seq,))
    conn.close()


def test_terminal_is_never_a_conversation(sessions: Sessions) -> None:
    normal = sessions.add(None, Role.USER, "hello").session_id
    terminal = sessions.terminal()
    sessions.add(terminal, Role.USER, "list")
    sessions.rename(terminal, "changed")
    sessions.pin(terminal, True)
    sessions.delete(terminal)
    assert [s.id for s in sessions.recent()] == [normal]
    assert sessions.latest().id == normal  # type: ignore[union-attr]
    assert sessions.get(terminal) is None
    assert sessions.history(terminal).messages == []
    with pytest.raises(ValueError, match="summaries"):
        sessions.add(terminal, Role.DESMAN, "summary", Kind.SUMMARY, covers=1)
    assert sessions.delete_all() == 1
    assert sessions.messages(terminal)
    row = sessions.conn.execute("SELECT * FROM sessions WHERE id = ?", (terminal,)).fetchone()
    assert row["title"] == "General_Terminal" and row["pinned_at"] is None
    assert row["deleted_at"] is None


@pytest.mark.parametrize("temporary", [False, True])
def test_command_switches_and_back_preserves_chat(
    sessions: Sessions, temporary: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    normal = sessions.add(None, Role.USER, "hello", temporary=temporary).session_id
    chat = Chat(sessions, temporary=temporary)
    chat.session_id = normal
    llm = FakeLLM()
    monkeypatch.setattr(runs, "models", lambda *a, **kw: runs.Models(llm, None))
    lines = iter(["!list", "list", "nonsense ordinary text", "/back", "/exit"])
    chat.read = lambda: next(lines, None)
    # Check before final cleanup, which intentionally erases temporary sessions.
    original_back = chat._back

    def back() -> None:
        original_back()
        assert chat.session_id == normal and chat.temporary == temporary
        assert [m.text for m in sessions.messages(normal)] == ["hello"]

    monkeypatch.setattr(chat, "_back", back)
    chat.run()
    messages = sessions.messages(sessions.terminal())
    assert [m.text for m in messages if m.role is Role.USER] == [
        "!list",
        "list",
        "nonsense ordinary text",
        "/back",
    ]
    events = [m for m in messages if m.kind is Kind.EVENT]
    assert len(events) == 3
    assert all("No tasks." in event.text for event in events[:2])
    assert "“nonsense” is not a command." in events[2].text
    assert "help" in events[2].text and "/back" in events[2].text
    assert not llm.calls


def test_notices_stay_out_of_chat_and_pointer_is_one_per_batch(
    sessions: Sessions, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    chat = Chat(sessions)
    normal = sessions.add(None, Role.USER, "hello").session_id
    chat.session_id = normal
    monkeypatch.setattr(chat, "_tell_waiting", lambda: ui.warn("waiting notice"))
    monkeypatch.setattr(chat, "_tell_blocked", lambda: ui.note("blocked notice"))
    chat._notices()
    output = capsys.readouterr().out
    assert output.count("new in General Terminal") == 1
    assert "waiting notice" not in output and "blocked notice" not in output
    assert [m.text for m in sessions.messages(normal)] == ["hello"]
    event = sessions.messages(sessions.terminal())[-1]
    assert (
        event.kind is Kind.EVENT
        and "waiting notice" in event.text
        and "blocked notice" in event.text
    )
    chat._terminal_open()
    capsys.readouterr()
    chat._notices()
    assert "waiting notice" in capsys.readouterr().out


def test_messenger_error_is_kept_in_terminal(sessions: Sessions) -> None:
    chat = Chat(sessions)
    normal = sessions.add(None, Role.USER, "hello").session_id
    chat.session_id = normal
    chat._action(ui.SimpleAction(type="messenger"))
    assert chat.in_terminal
    events = sessions.messages(sessions.terminal())
    assert events[-1].kind is Kind.EVENT and "join it first" in events[-1].text
    assert len(sessions.messages(normal)) == 1


@pytest.mark.parametrize("web", [False, True])
def test_recording_tees_ui_rich_and_plain_output(
    web: bool, capsys: pytest.CaptureFixture[str]
) -> None:
    front = WebFront() if web else ui.TerminalFront()
    with ui.use(front), ui.recording() as rec:
        table = Table("heading")
        table.add_row("rich cell")
        ui.out(table)
        ui.ok("success")
        ui.note("note")
        ui.warn("warning")
        ui.error("error")
        ui.header("header", "subtitle")
        ui.section("section")
        print("\x1b[31mplain stdout\x1b[0m")
        print("plain stderr", file=sys.stderr)
    assert "\x1b" not in rec.text
    for word in (
        "rich cell",
        "success",
        "note",
        "warning",
        "error",
        "header",
        "subtitle",
        "section",
        "plain stdout",
        "plain stderr",
    ):
        assert rec.text.count(word) == 1
    if web:
        assert isinstance(front, WebFront)
        assert front.events
    else:
        assert "rich cell" in capsys.readouterr().out


@pytest.mark.parametrize("web", [False, True])
def test_command_output_and_input_redact_configured_key(
    sessions: Sessions, monkeypatch: pytest.MonkeyPatch, web: bool
) -> None:
    from desman.config import ApiKey

    config = open_config().load()
    secret = "configured-secret-with-special-<>&-characters" * 6
    config.api_keys["groq"] = [ApiKey(value=secret)]
    open_config().save(config)
    chat = Chat(sessions)

    def print_key(line: str) -> None:
        ui.out(secret)
        table = ui.table("key")
        table.add_row(secret)
        ui.out(table)
        click.echo(secret)

    monkeypatch.setattr(chat, "_typed", print_key)
    front = WebFront() if web else ui.TerminalFront()
    with ui.use(front):
        if isinstance(front, WebFront):
            with front.capture():
                chat._terminal_command("!list " + secret)
        else:
            chat._terminal_command("!list " + secret)
    texts = "\n".join(m.text for m in sessions.messages(sessions.terminal()))
    assert secret not in texts and secret[:30] not in texts and "[" in texts


def test_page_terminal_snapshot_unread_and_open_clears(tmp_path: Path) -> None:
    path = tmp_path / "state.db"
    conn = db.open_db(path)
    sessions = Sessions(conn)
    front = WebFront()
    chat = Chat(sessions)
    with ui.use(front):
        sessions.add(sessions.terminal(), Role.DESMAN, "notice", kind=Kind.EVENT)
        data = snapshot(path, "sessions", front)
        assert data["sessions"] == []
        assert data["terminal"] == {"id": sessions.terminal(), "unread": 1}
        chat._terminal_open()
        data = snapshot(path, "sessions", front)
        assert data["current"] == sessions.terminal() and data["terminal"]["unread"] == 0
        chat._back()
        assert snapshot(path, "sessions", front)["terminal"]["unread"] == 0
    conn.close()


def test_terminal_slash_restrictions_and_full_event_return(
    sessions: Sessions, capsys: pytest.CaptureFixture[str]
) -> None:
    chat = Chat(sessions)
    terminal = sessions.terminal()
    sessions.add(terminal, Role.DESMAN, "first line\nsecond line", Kind.EVENT)
    chat._terminal_open()
    assert "second line" in capsys.readouterr().out
    for command in ("/rename test", "/pin", "/compact", "/profile", "/delete", "/delete-all"):
        assert chat._slash(command)
    assert capsys.readouterr().err.count("not available in General Terminal") == 6


@pytest.mark.parametrize("command", ["run", "resume", "worker"])
def test_terminal_task_commands_use_code_report_without_retelling(
    sessions: Sessions, monkeypatch: pytest.MonkeyPatch, command: str, tmp_path: Path
) -> None:
    from desman.config import ApiKey
    from desman.core.models import SubtaskSpec, TaskSpec
    from desman.core.workspace import Workspace
    from desman.interlocutor import RETELL

    secret = "terminal-task-result-secret"
    config = open_config().load()
    config.api_keys["groq"] = [ApiKey(value=secret)]
    open_config().save(config)
    monkeypatch.setattr("desman.chat.workspace_root_for", lambda conn: tmp_path)
    original_work = Chat._work

    def work(self: Chat, task_id: str, *args: object) -> None:
        Workspace(tmp_path, task_id).write("answer.txt", "task file")
        original_work(self, task_id, *args)  # type: ignore[arg-type]

    monkeypatch.setattr(Chat, "_work", work)
    llm = FakeLLM()
    llm.on("Plan task", '{"subtasks": [{"description": "find", "acceptance_criteria": "found"}]}')
    llm.on("Execute subtask", json.dumps({"answer": "terminal answer", "key": secret}))
    llm.on("Review subtask", '{"verdict": "accept"}')
    monkeypatch.setattr(runs, "models", lambda *a, **kw: runs.Models(llm, None))
    chat = Chat(sessions)
    if command == "run":
        line = 'run "find it"'
    else:
        task = chat.store.create_task(
            TaskSpec(
                goal="find it",
                subtasks=[SubtaskSpec(description="find", acceptance_criteria="found")],
            )
        )
        line = "worker" if command == "worker" else f"resume {task}"
    chat._terminal_command(line)
    messages = sessions.messages(sessions.terminal())
    assert any("Goal: find it" in m.text for m in messages if m.kind is Kind.EVENT)
    if command != "worker":
        output = "\n".join(m.text for m in messages if m.kind is Kind.EVENT)
        assert "The result as it is:" in output and "terminal answer" in output
        assert "Files of task" in output and "answer.txt" in output
        assert secret not in output
        assert (
            output.index("Goal: find it")
            < output.index("terminal answer")
            < output.index("answer.txt")
        )
    assert all(not request.prompt.startswith(RETELL) for request in llm.calls)
    assert not any(m.role is Role.INTERLOCUTOR for m in messages)
    if chat._http is not None:
        chat._http.close()


def test_terminal_help_slash_messages_and_error_are_saved(sessions: Sessions) -> None:
    chat = Chat(sessions)
    lines = iter(["/terminal", "/help", "/messages", "/rename wrong", "/exit"])
    chat.read = lambda: next(lines, None)
    chat.run()
    messages = sessions.messages(sessions.terminal())
    assert [m.text for m in messages if m.role is Role.USER] == [
        "/help",
        "/messages",
        "/rename wrong",
        "/exit",
    ]
    assert len([m for m in messages if m.kind is Kind.EVENT]) == 3


def test_leaving_terminal_for_new_chat_erases_suspended_temporary(sessions: Sessions) -> None:
    chat = Chat(sessions, temporary=True)
    temporary = sessions.add(None, Role.USER, "secret chat", temporary=True).session_id
    chat.session_id = temporary
    chat._terminal_open()
    assert sessions.messages(temporary)
    chat._new()
    assert sessions.messages(temporary) == []


def test_ending_while_terminal_open_erases_suspended_temporary(sessions: Sessions) -> None:
    chat = Chat(sessions, temporary=True)
    temporary = sessions.add(None, Role.USER, "secret chat", temporary=True).session_id
    chat.session_id = temporary
    lines = iter(["/terminal", "/exit"])
    chat.read = lambda: next(lines, None)
    chat.run()
    assert sessions.messages(temporary) == []


def test_terminal_analysis_does_not_retell(
    sessions: Sessions, monkeypatch: pytest.MonkeyPatch
) -> None:
    from desman.core.models import SubtaskSpec, TaskSpec
    from desman.core.retry import FailureKind
    from desman.core.states import Actor, SubtaskStatus
    from desman.interlocutor import RETELL

    chat = Chat(sessions)
    chat.store.create_task(
        TaskSpec(
            goal="impossible",
            subtasks=[SubtaskSpec(description="find", acceptance_criteria="found")],
        )
    )
    sub = chat.store.claim("worker")
    assert sub is not None
    chat.store.record_failure(
        sub.id,
        SubtaskStatus.RUNNING,
        Actor.EXECUTOR,
        FailureKind.PERMANENT,
        lease=sub.lease_token,
        reason="criteria need unavailable data",
    )
    llm = FakeLLM()
    llm.on(
        "Execute",
        '{"changes": [], "summary": "No change needed.", "evidence": ["unavailable data"], "expected_effect": "none"}',
    )
    llm.on("Review", '{"verdict": "accept"}')
    monkeypatch.setattr(runs, "models", lambda *a, **kw: runs.Models(llm, None))
    chat._terminal_command("proposals analyze")
    events = [m.text for m in sessions.messages(sessions.terminal()) if m.kind is Kind.EVENT]
    assert any("No change needed." in event for event in events)
    assert llm.calls and all(not request.prompt.startswith(RETELL) for request in llm.calls)
    if chat._http is not None:
        chat._http.close()


def test_unknown_terminal_word_redacts_key(sessions: Sessions) -> None:
    from desman.config import ApiKey

    config = open_config().load()
    secret = "secret-unknown-command-key"
    config.api_keys["groq"] = [ApiKey(value=secret)]
    open_config().save(config)
    chat = Chat(sessions)
    chat._terminal_command(secret)
    messages = sessions.messages(sessions.terminal())
    assert secret not in "\n".join(m.text for m in messages)
    assert "is not a command" in messages[-1].text


def test_plain_cyrillic_terminal_text_is_not_model_input(
    sessions: Sessions, monkeypatch: pytest.MonkeyPatch
) -> None:
    llm = FakeLLM()
    monkeypatch.setattr(runs, "models", lambda *a, **kw: runs.Models(llm, None))
    lines = iter(["/terminal", "привет", "/exit"])
    Chat(sessions, read=lambda: next(lines, None)).run()
    text = "\n".join(m.text for m in sessions.messages(sessions.terminal()) if m.kind is Kind.EVENT)
    assert "“привет” is not a command." in text
    assert "`help`" in text and "/back" in text
    assert not llm.calls


@pytest.mark.parametrize("temporary", [False, True])
def test_panel_decision_keeps_chat_and_output_unread(
    sessions: Sessions, monkeypatch: pytest.MonkeyPatch, temporary: bool
) -> None:
    chat = Chat(sessions, temporary=temporary)
    normal = sessions.add(None, Role.USER, "hello", temporary=temporary).session_id
    chat.session_id = normal
    monkeypatch.setattr(chat, "_dispatch_action", lambda action: ui.ok("decision saved"))
    front = WebFront()
    with ui.use(front):
        ui.session(normal, temporary)
        chat._action(
            ui.DecideAction(type="decide", ref="task/1", version=1, decision="accept", note="")
        )
    assert chat.session_id == normal and chat.temporary == temporary
    assert chat._terminal_seen == 0 and front.terminal_seq == 0
    assert not front.events
    assert sessions.messages(sessions.terminal())[-1].text.strip() == "✓ decision saved"
    assert [m.text for m in sessions.messages(normal)] == ["hello"]


def test_terminal_result_json_never_shows_an_escaped_key() -> None:
    """A key with a quote or backslash is cut before json.dumps escapes it (Devin, PR #97)."""
    from desman import chat as chat_module
    from desman.config import ApiKey, Config

    key = 'gsk_quote"and\\slash_0123456789abcdef'
    config = Config(api_keys={"GROQ_API_KEY": [ApiKey(value=key)]})
    shown = json.dumps(
        chat_module._redacted({"answer": f"the key is {key}", key: [key]}, config),
        ensure_ascii=False,
    )
    assert key not in shown
    assert json.dumps(key)[1:-1] not in shown


def test_terminal_result_keeps_fields_whose_names_held_keys() -> None:
    from desman import chat as chat_module
    from desman.config import ApiKey, Config

    one, two = "gsk_first_0123456789abcdefghij", "gsk_second_0123456789abcdefghi"
    config = Config(api_keys={"GROQ_API_KEY": [ApiKey(value=one), ApiKey(value=two)]})
    shown = chat_module._redacted({one: "a", two: "b"}, config)
    assert sorted(shown.values()) == ["a", "b"]
    assert one not in json.dumps(shown) and two not in json.dumps(shown)


@pytest.mark.parametrize("line", ["help", "/help"])
def test_terminal_help_is_one_catalog_listing(sessions: Sessions, line: str) -> None:
    chat = Chat(sessions)
    lines = iter(["/terminal", line, "/exit"])
    chat.read = lambda: next(lines, None)
    chat.run()
    events = [m.text for m in sessions.messages(sessions.terminal()) if m.kind is Kind.EVENT]
    listing = next(text for text in events if "clear the General Terminal screen" in text)
    assert listing.count("clear the General Terminal screen") == 1
    assert listing.count("return from General Terminal to the previous chat") == 1
    assert listing.count("init-db") == 1
    assert "the chat's commands" not in listing
    assert "Ask for any of them" not in listing


@pytest.mark.parametrize("line", ["help", "/help"])
def test_terminal_help_lists_commands_with_a_broken_config(
    sessions: Sessions, line: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    (Path(os.environ["DESMAN_CONFIG_DIR"]) / "config.json").write_text(
        "{not json", encoding="utf-8"
    )
    chat = Chat(sessions)
    lines = iter(["/terminal", line, "/exit"])
    chat.read = lambda: next(lines, None)
    chat.run()
    events = [m.text for m in sessions.messages(sessions.terminal()) if m.kind is Kind.EVENT]
    assert any("clear the General Terminal screen" in text and "init-db" in text for text in events)
