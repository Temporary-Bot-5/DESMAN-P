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
from desman import runs, task_files, ui
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
    lines = iter(["/terminal", "!list", "list", "nonsense ordinary text", "/back", "/exit"])
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
    assert "No tasks." in events[0].text
    assert "Commands start with /" in events[1].text
    assert "Commands start with /" in events[2].text
    assert "/status" in events[2].text
    assert not llm.calls


@pytest.mark.parametrize("line", ["/list", "!list"])
@pytest.mark.parametrize("temporary", [False, True])
def test_explicit_command_runs_in_place(
    sessions: Sessions,
    line: str,
    temporary: bool,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    normal = sessions.add(None, Role.USER, "hello", temporary=temporary).session_id
    terminal = sessions.terminal()
    sessions.add(terminal, Role.DESMAN, "old terminal output", Kind.EVENT)
    chat = Chat(sessions, temporary=temporary)
    chat.session_id = normal
    chat.models = runs.Models(FakeLLM(), None)
    models = chat.models
    monkeypatch.setattr(chat, "_tell_blocked", lambda: None)
    monkeypatch.setattr(chat, "_tell_waiting", lambda: None)

    if line.startswith("/"):
        assert chat._slash(line)
    else:
        chat._terminal_command(line)

    output = capsys.readouterr().out
    assert "No tasks." in output
    assert "→ General Terminal" not in output and "old terminal output" not in output
    assert chat.session_id == normal and chat.temporary is temporary and chat.models is models
    assert [m.text for m in sessions.messages(normal)] == ["hello"]
    user, event = sessions.messages(terminal)[-2:]
    assert user.role is Role.USER and user.text == line
    assert event.kind is Kind.EVENT and "No tasks." in event.text

    monkeypatch.setattr(chat, "_tell_waiting", lambda: ui.warn("new notice"))
    chat._notices()
    notice = capsys.readouterr().out
    assert notice == ("" if temporary else "1 new in General Terminal (/terminal)\n")


@pytest.mark.parametrize("command", ["/clear", "!clear"])
def test_clear_alias_only_clears_the_chat_screen(
    sessions: Sessions, command: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    chat = Chat(sessions)
    normal = sessions.add(None, Role.USER, "hello").session_id
    chat.session_id = normal
    terminal = sessions.terminal()
    sessions.add(terminal, Role.DESMAN, "terminal output", Kind.EVENT)
    cleared: list[str | None] = []
    monkeypatch.setattr(ui, "clear_screen", lambda: cleared.append(chat.session_id))
    lines = iter([command, "/exit"])
    chat.read = lambda: next(lines, None)

    chat.run()

    assert cleared == [normal] and chat.session_id == normal
    assert [m.text for m in sessions.messages(normal)] == ["hello"]
    assert [m.text for m in sessions.screen(terminal)] == ["terminal output"]
    assert [m.text for m in sessions.messages(terminal)] == ["terminal output"]


def test_help_alias_lists_chat_commands(
    sessions: Sessions, capsys: pytest.CaptureFixture[str]
) -> None:
    lines = iter(["!help", "!exit"])
    chat = Chat(sessions, read=lambda: next(lines, None))
    chat.run()

    output = capsys.readouterr().out
    assert "/rename" in output and "/compact" in output
    assert not sessions.messages(sessions.terminal())


def test_in_place_commands_share_pending_files(
    sessions: Sessions, monkeypatch: pytest.MonkeyPatch
) -> None:
    chat = Chat(sessions)
    terminal = sessions.terminal()
    files = task_files.PendingFiles()

    def typed(runner: Chat, line: str, *, in_place: bool) -> None:
        assert in_place and runner.session_id == terminal
        if line == "list":
            runner.pending_files[terminal] = files
        else:
            assert runner.pending_files[terminal] is files

    monkeypatch.setattr(Chat, "_typed", typed)
    chat._terminal_command("/list")
    chat._terminal_command("/stats")


def test_in_place_command_restores_chat_on_error(
    sessions: Sessions, monkeypatch: pytest.MonkeyPatch
) -> None:
    chat = Chat(sessions, temporary=True)
    normal = sessions.add(None, Role.USER, "hello", temporary=True).session_id
    chat.session_id = normal
    models = runs.Models(FakeLLM(), None)
    config = Config()
    chat.models, chat._models_for = models, config
    chat._master_pending = True

    def typed(line: str, *, in_place: bool) -> None:
        assert in_place and chat.in_terminal and not chat.temporary
        chat.models, chat._models_for = None, None
        raise RuntimeError("command failed")

    monkeypatch.setattr(chat, "_typed", typed)
    with pytest.raises(RuntimeError, match="command failed"):
        chat._terminal_command("/list")

    assert chat.session_id == normal and chat.temporary and chat._master_pending
    assert chat.models is models and chat._models_for is config


@pytest.mark.parametrize("line", ["/foo", "!foo"])
@pytest.mark.parametrize("in_terminal", [False, True])
def test_unknown_explicit_command_message(
    sessions: Sessions, line: str, in_terminal: bool, capsys: pytest.CaptureFixture[str]
) -> None:
    normal = sessions.add(None, Role.USER, "hello").session_id
    terminal = sessions.terminal()
    chat = Chat(sessions)
    chat.session_id = terminal if in_terminal else normal
    lines = iter([line])
    chat.read = lambda: next(lines, None)
    chat.run()

    output = " ".join(capsys.readouterr().err.split())
    user, event = sessions.messages(terminal)
    assert user.text == line and event.kind is Kind.EVENT
    expected = (
        "“foo” is not a command. General Terminal takes only DESMAN's commands "
        "(`/help` lists them); to talk, go back to a chat with /back."
        if in_terminal
        else "“foo” is not a command. /help lists them."
    )
    assert f"✗ {expected}" in output
    assert " ".join(event.text.split()) == f"✗ {expected}"
    assert chat.session_id == (terminal if in_terminal else normal)


def test_in_place_command_keeps_temporary_chat_erasable(sessions: Sessions) -> None:
    temporary = sessions.add(None, Role.USER, "hello", temporary=True).session_id
    chat = Chat(sessions, temporary=True)
    chat.session_id = temporary

    def lines() -> Iterator[str]:
        yield "/list"
        assert chat.session_id == temporary and chat.temporary
        assert [m.text for m in sessions.messages(temporary)] == ["hello"]
        yield "/exit"

    commands = lines()
    chat.read = lambda: next(commands, None)
    chat.run()

    assert sessions.messages(temporary) == []
    user, event = sessions.messages(sessions.terminal())
    assert user.text == "/list" and "No tasks." in event.text


def test_in_place_command_keeps_pending_master_and_previous_unread(
    sessions: Sessions, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    chat = Chat(sessions)
    chat._master_pending = True
    sessions.add(sessions.terminal(), Role.DESMAN, "unseen output", Kind.EVENT)
    chat._terminal_command("/list")
    assert chat.session_id is None and chat.master_mode
    assert "unseen output" not in capsys.readouterr().out
    monkeypatch.setattr(chat, "_tell_waiting", lambda: ui.warn("new notice"))
    monkeypatch.setattr(chat, "_tell_blocked", lambda: None)
    chat._notices()
    assert "2 new in General Terminal" in capsys.readouterr().out


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

    monkeypatch.setattr(Chat, "_typed", lambda self, line, **kwargs: print_key(line))
    front = WebFront() if web else ui.TerminalFront()
    with ui.use(front):
        if isinstance(front, WebFront):
            with front.capture():
                chat._terminal_command("!list " + secret)
        else:
            chat._terminal_command("!list " + secret)
    assert chat.in_terminal is web
    texts = "\n".join(m.text for m in sessions.messages(sessions.terminal()))
    assert secret not in texts and secret[:30] not in texts and "[" in texts
    styled = "\n".join(m.styled or "" for m in sessions.messages(sessions.terminal()))
    assert secret not in styled and secret[:30] not in styled
    assert "[" in styled


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

    def work(self: Chat, task_id: str, *args: object, **kwargs: object) -> None:
        Workspace(tmp_path, task_id).write("answer.txt", "task file")
        original_work(self, task_id, *args)  # type: ignore[arg-type]

    monkeypatch.setattr(Chat, "_work", work)
    llm = FakeLLM()
    llm.on("Plan task", '{"subtasks": [{"description": "find", "acceptance_criteria": "found"}]}')
    llm.on("Execute subtask", json.dumps({"answer": "terminal answer\nsecond line", "key": secret}))
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
        assert "terminal answer\nsecond line" in output
        assert "\\n" not in output
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
    assert "`/help`" in messages[-1].text


def test_plain_cyrillic_terminal_text_is_not_model_input(
    sessions: Sessions, monkeypatch: pytest.MonkeyPatch
) -> None:
    llm = FakeLLM()
    monkeypatch.setattr(runs, "models", lambda *a, **kw: runs.Models(llm, None))
    lines = iter(["/terminal", "привет", "/exit"])
    Chat(sessions, read=lambda: next(lines, None)).run()
    text = "\n".join(m.text for m in sessions.messages(sessions.terminal()) if m.kind is Kind.EVENT)
    assert "Commands start with /" in text
    assert "/status" in text
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


@pytest.mark.parametrize("line", ["!help", "/help"])
def test_terminal_help_is_one_catalog_listing(sessions: Sessions, line: str) -> None:
    chat = Chat(sessions)
    lines = iter(["/terminal", line, "/exit"])
    chat.read = lambda: next(lines, None)
    chat.run()
    events = [m.text for m in sessions.messages(sessions.terminal()) if m.kind is Kind.EVENT]
    listing = next(text for text in events if "clear the screen" in text)
    assert listing.count("clear the screen") == 1
    assert listing.count("return from General Terminal to the previous chat") == 1
    assert listing.count("init-db") == 1
    assert "the chat's commands" not in listing
    assert "Ask for any of them" not in listing


@pytest.mark.parametrize("line", ["!help", "/help"])
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
    assert any("clear the screen" in text and "init-db" in text for text in events)


def test_terminal_help_is_journaled_and_follows_the_language(sessions: Sessions) -> None:
    from desman import i18n

    chat = Chat(sessions)
    lines = iter(["/terminal", "/help", "/exit"])
    chat.read = lambda: next(lines, None)
    try:
        i18n.set_language("ru")
        chat.run()
    finally:
        i18n.set_language("en")
    events = [m.text for m in sessions.messages(sessions.terminal()) if m.kind is Kind.EVENT]
    listing = next(text for text in events if "init-db" in text)
    assert "Create the database or apply pending migrations." not in listing
    rows = sessions.conn.execute(
        "SELECT details_json FROM operator_log WHERE action = 'operator_command'"
    ).fetchall()
    assert any(json.loads(r[0])["command"] == "help" and json.loads(r[0])["typed"] for r in rows)


def test_usage_remains_a_terminal_command(sessions: Sessions) -> None:
    chat = Chat(sessions)
    chat._terminal_command("/usage")
    assert chat.session_id is None
    messages = sessions.messages(sessions.terminal())
    assert messages[-2].role is Role.USER and messages[-2].text == "/usage"
    assert messages[-1].kind is Kind.EVENT
    assert messages[-1].text.strip()


def test_terminal_style_survives_reload(sessions: Sessions) -> None:
    from rich.text import Text

    chat = Chat(sessions)
    chat._terminal_output(lambda: ui.out(Text("success", style="good")))
    [message] = sessions.messages(sessions.terminal())
    assert message.text.strip() == "success"
    assert message.styled is not None and "\x1b[" in message.styled
    assert "#00ff99" in WebFront.render(Text.from_ansi(message.styled))
    from desman.core.sessions import History
    from desman.interlocutor import Interlocutor

    request = Interlocutor(FakeLLM(), "en").build_request(History(summary=None, messages=[message]))
    assert "success" in request.prompt and "\\u001b" not in request.prompt


@pytest.mark.parametrize("note", ["", " "])
def test_empty_fail_action_and_catalog(note: str) -> None:
    from desman.commands import FailArgs

    action = ui.DecideAction(type="decide", ref="abc/0", version=1, decision="fail", note=note)
    assert (
        FailArgs.model_validate({"subtask": action.ref, "note": action.note}).note == note.strip()
    )
    assert FailArgs.model_validate({"subtask": action.ref}).note is None


@pytest.mark.parametrize("raw", [False, True])
@pytest.mark.parametrize("notices", [False, True])
def test_split_key_never_survives_styled_storage(
    sessions: Sessions, monkeypatch: pytest.MonkeyPatch, raw: bool, notices: bool
) -> None:
    from rich.text import Text

    from desman.config import ApiKey

    secret = "split-configured-secret"
    config = open_config().load()
    config.api_keys["groq"] = [ApiKey(value=secret)]
    open_config().save(config)
    chat = Chat(sessions)

    def output() -> None:
        if raw:
            click.echo(secret[:8] + "\x1b[31m" + secret[8:] + "\x1b[0m", color=True)
        else:
            text = Text(secret)
            text.stylize("red", 8, 15)
            ui.out(text)

    if notices:
        monkeypatch.setattr(chat, "_tell_waiting", output)
        monkeypatch.setattr(chat, "_tell_blocked", lambda: None)
        chat._notices()
    else:
        chat._terminal_output(output)
    [message] = sessions.messages(sessions.terminal())
    assert secret not in message.text
    assert secret not in Text.from_ansi(message.styled or "").plain
    if raw:
        assert message.styled is None


def test_old_split_key_falls_back_in_page_transcript(tmp_path: Path) -> None:
    from html import unescape

    from rich.text import Text

    from desman.config import ApiKey
    from desman.gui.runners import Runners

    secret = "old-split-configured-secret"
    config = Config(language="en", configured=True)
    config.api_keys["groq"] = [ApiKey(value=secret)]
    open_config().save(config)
    path = tmp_path / "state.db"
    conn = db.open_db(path)
    try:
        sessions = Sessions(conn)
        terminal = sessions.terminal()
        sessions.add(
            terminal,
            Role.DESMAN,
            "[redacted]\n",
            Kind.EVENT,
            styled=secret[:8] + "\x1b[31m" + secret[8:] + "\x1b[0m\n",
        )
        front = WebFront()
        front.runners = Runners(path, front)
        front.database = path
        front.current = terminal
        front.session_changed(terminal)
        html = "".join(front.poll(0, timeout=0)["transcript"])
        assert secret not in Text.from_markup(unescape(html)).plain
        assert secret[8:] not in unescape(html)
        assert "[redacted]" in html
    finally:
        conn.close()


def test_notice_recording_matches_page_colours() -> None:
    from rich.text import Text

    front = WebFront()
    with ui.use(front), ui.recording() as rec:
        ui.ok("done")
        ui.warn("careful")
        ui.error("failed")
        ui.note("aside")
    assert rec.text == "✓ done\n! careful\n✗ failed\naside\n"
    restored = WebFront.render(Text.from_ansi(rec.styled))
    for event in front.events:
        assert event["html"] in restored


@pytest.mark.parametrize("temporary", [False, True])
def test_terminal_pointer_excludes_historical_events_and_temporary_chats(
    sessions: Sessions,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    temporary: bool,
) -> None:
    terminal = sessions.terminal()
    for _ in range(3):
        sessions.add(terminal, Role.DESMAN, "old notice", kind=Kind.EVENT)
    chat = Chat(sessions, temporary=temporary)
    monkeypatch.setattr(chat, "_tell_waiting", lambda: ui.note("new notice"))
    monkeypatch.setattr(chat, "_tell_blocked", lambda: None)
    chat._notices()
    output = capsys.readouterr().out
    if temporary:
        assert "new in General Terminal" not in output
    else:
        assert "1 new in General Terminal" in output
        assert "4 new" not in output


def test_startup_does_not_reannounce_existing_inbox_but_new_arrivals_are_noticed(
    sessions: Sessions, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from desman.core.models import SubtaskSpec, TaskSpec
    from desman.core.retry import FailureKind
    from desman.core.states import Actor, SubtaskStatus
    from desman.core.store import Store

    store = Store(sessions.conn, clock=sessions.clock)

    def waiting() -> None:
        store.create_task(
            TaskSpec(
                goal="old task",
                subtasks=[SubtaskSpec(description="work", acceptance_criteria="done")],
            )
        )
        sub = store.claim("worker")
        assert sub is not None
        store.record_failure(
            sub.id,
            SubtaskStatus.RUNNING,
            Actor.EXECUTOR,
            FailureKind.PERMANENT,
            lease=sub.lease_token,
            reason="missing tool",
        )

    waiting()
    chat = Chat(sessions)
    monkeypatch.setattr(chat, "_tell_blocked", lambda: None)
    chat._notices()
    assert "new in General Terminal" not in capsys.readouterr().out
    waiting()
    chat._notices()
    assert "1 new in General Terminal" in capsys.readouterr().out


def test_decision_without_models_shows_resume_hint(
    sessions: Sessions, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from desman.core.models import SubtaskSpec, TaskSpec
    from desman.core.retry import FailureKind
    from desman.core.states import Actor, SubtaskStatus

    chat = Chat(sessions)
    task = chat.store.create_task(
        TaskSpec(
            goal="Answer", subtasks=[SubtaskSpec(description="Work", acceptance_criteria="ok")]
        )
    )
    subtask = chat.store.claim("worker", task_id=task)
    assert subtask is not None
    chat.store.record_failure(
        subtask.id,
        SubtaskStatus.RUNNING,
        Actor.EXECUTOR,
        FailureKind.PERMANENT,
        lease=subtask.lease_token,
        reason="Need dates",
    )
    monkeypatch.setattr(runs, "models", lambda *a, **kw: None)
    monkeypatch.setattr(chat, "_confirmed", lambda *a, **kw: True)
    chat._terminal_command(f"retry {subtask.id}")
    assert "Continue with: desman resume" in capsys.readouterr().out
    assert chat.store.get_subtask(subtask.id).status is SubtaskStatus.READY
