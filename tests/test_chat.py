"""Tests for chat sessions (ROADMAP C.2): the store, the migration and bare `desman`."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterator

import pytest
from click.testing import CliRunner, Result

from conftest import FakeClock
from desman import commands, ui
from desman.chat import Chat
from desman.cli import cli
from desman.config import ConfigManager, default_config_path
from desman.core import db
from desman.core.sessions import Kind, Role, Sessions
from helpers import make_config


@pytest.fixture
def sessions(clock: FakeClock) -> Iterator[Sessions]:
    conn = db.open_db(":memory:")
    yield Sessions(conn, clock=clock)
    conn.close()


def configured() -> None:
    ConfigManager(default_config_path()).save(make_config(language="en", configured=True))


def chat(*lines: str, args: tuple[str, ...] = ()) -> Result:
    result = CliRunner().invoke(cli, list(args), input="".join(f"{line}\n" for line in lines))
    assert result.exit_code == 0, result.output
    return result


def again(*lines: str) -> Result:
    """`desman -c`: the chat continuing the last session."""
    return chat(*lines, args=("-c",))


def stored() -> Sessions:
    return Sessions(db.open_db(db.default_db_path()))


# ── Store ────────────────────────────────────────────────────────────────────


def test_a_session_starts_with_its_first_message(sessions: Sessions, clock: FakeClock) -> None:
    assert sessions.latest() is None and sessions.recent() == []
    first = sessions.add(None, Role.USER, "hello")
    clock.advance(1)
    second = sessions.add(first.session_id, Role.DESMAN, "task started", kind=Kind.EVENT)
    assert second.session_id == first.session_id and second.seq > first.seq
    assert [(m.role, m.kind, m.text) for m in sessions.messages(first.session_id)] == [
        (Role.USER, Kind.MESSAGE, "hello"),
        (Role.DESMAN, Kind.EVENT, "task started"),
    ]
    [info] = sessions.recent()
    assert info.messages == 2 and info.opening == "hello" and info.last_at == second.at


def test_the_latest_session_is_the_one_written_to_last(
    sessions: Sessions, clock: FakeClock
) -> None:
    old = sessions.add(None, Role.USER, "old one").session_id
    clock.advance(1)
    new = sessions.add(None, Role.USER, "new one").session_id
    clock.advance(1)
    sessions.add(old, Role.USER, "back to the old one")
    assert [s.id for s in sessions.recent()] == [old, new]
    assert sessions.latest() is not None and sessions.latest().id == old  # type: ignore[union-attr]
    assert [s.id for s in sessions.recent(limit=1)] == [old]


def test_last_messages_keep_their_order(sessions: Sessions) -> None:
    session_id = sessions.add(None, Role.USER, "m0").session_id
    for i in range(1, 5):
        sessions.add(session_id, Role.USER, f"m{i}")
    assert [m.text for m in sessions.messages(session_id, last=2)] == ["m3", "m4"]


def test_messages_are_append_only(sessions: Sessions) -> None:
    message = sessions.add(None, Role.USER, "said once")
    conn = sessions.conn
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute("UPDATE messages SET text = 'rewritten' WHERE seq = ?", (message.seq,))
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute("DELETE FROM messages WHERE seq = ?", (message.seq,))
    with pytest.raises(sqlite3.IntegrityError):  # its session stays too: messages refer to it
        conn.execute("DELETE FROM sessions WHERE id = ?", (message.session_id,))


def test_a_message_names_an_existing_task(sessions: Sessions) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        sessions.add(None, Role.DESMAN, "done", kind=Kind.EVENT, task_id="no-such-task")
    assert sessions.recent() == []  # the session it would have started is not left behind


@pytest.mark.parametrize(("column", "enum"), [("role", Role), ("kind", Kind)])
def test_message_checks_match_the_enums(
    sessions: Sessions, column: str, enum: type[Role] | type[Kind]
) -> None:
    sql = sessions.conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'messages'"
    ).fetchone()[0]
    match = re.search(rf"{column}\s+TEXT NOT NULL CHECK \({column} IN \(([^)]*)\)\)", sql)
    assert match is not None
    assert set(re.findall(r"'(\w+)'", match.group(1))) == {e.value for e in enum}


# ── Bare `desman` ────────────────────────────────────────────────────────────


def test_messages_are_kept_and_every_chat_starts_a_new_session() -> None:
    configured()
    first = chat("find cheap flights to Lisbon", "/exit", "never read")
    assert "New session" in first.output
    assert "no model is ready to answer" in first.output
    [info] = stored().recent()
    assert info.opening == "find cheap flights to Lisbon" and info.messages == 1

    second = chat("and back on Sunday")
    assert "New session" in second.output and "Continuing the session" not in second.output
    assert "find cheap flights" not in second.output
    assert [s.opening for s in stored().recent()] == [
        "and back on Sunday",
        "find cheap flights to Lisbon",
    ]


def test_c_continues_the_last_session() -> None:
    configured()
    chat("find cheap flights to Lisbon")
    result = again("and back on Sunday")
    assert "Continuing the session" in result.output
    assert "find cheap flights to Lisbon" in result.output  # the last messages are shown
    [info] = stored().recent()
    assert info.messages == 2


def test_new_starts_a_session_with_the_next_message() -> None:
    configured()
    chat("first")
    result = again("/new", "/new", "second")
    assert "New session: it starts with your next message" in result.output
    assert "This session is new already" in result.output
    assert [s.opening for s in stored().recent()] == ["second", "first"]


def test_leaving_without_writing_leaves_no_session() -> None:
    configured()
    chat("/new", "/help", "", "   ")
    assert stored().recent() == []


def test_sessions_switches_to_the_chosen_one() -> None:
    configured()
    chat("about flights")
    chat("about trains")
    # Listed newest first: 1 is "about trains" (current), 2 is "about flights".
    result = again("/sessions", "2", "more on flights")
    assert "Which session?" in result.output and "current" in result.output
    assert "Switched to the session" in result.output
    flights, trains = stored().recent()  # the one written to last comes first
    assert flights.opening == "about flights" and flights.messages == 2
    assert trains.messages == 1


def test_sessions_without_any_and_leaving_the_window() -> None:
    configured()
    assert "No sessions yet" in chat("/sessions").output
    chat("hello")
    result = chat("/sessions")  # input ends inside the window: the chat just ends
    assert "Which session?" in result.output


def test_help_and_unknown_commands() -> None:
    configured()
    result = chat("/help", "/frobnicate", "/EXIT")
    for command in ("/help", "/new", "/sessions", "/exit"):
        assert command in result.output
    assert "“frobnicate” is not a command." in result.output
    assert stored().recent() == []  # a command is not a message


def test_subcommands_and_help_do_not_open_the_chat() -> None:
    configured()
    result = CliRunner().invoke(cli, ["--help"], input="hello\n")
    assert result.exit_code == 0 and "Usage" in result.output
    assert "New session" not in result.output
    assert not db.default_db_path().exists()


def test_a_message_is_kept_as_typed() -> None:
    configured()
    chat("    return answer  ")
    [info] = stored().recent()
    assert info.opening == "    return answer  "


def test_every_session_stays_reachable_page_by_page(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("desman.chat.LISTED_SESSIONS", 2)
    configured()
    chat("oldest", "/new", "middle", "/new", "newest")
    # Page 1: newest (current), middle, "earlier sessions…" (3); page 2: oldest (1).
    result = again("/sessions", "3", "1", "back to the oldest")
    assert "earlier sessions…" in result.output
    assert "Switched to the session" in result.output
    assert [s.opening for s in stored().recent()] == ["oldest", "newest", "middle"]
    assert [s.opening for s in stored().recent(limit=1, offset=2)] == ["middle"]


# ── Suggestions for the chat's commands ──────────────────────────────────────

SLASH = {"/help": "list", "/new": "new", "/sessions": "earlier", "/exit": "leave"}


@pytest.mark.parametrize(
    ("typed", "names"),
    [
        ("/", ["/exit", "/help", "/new", "/sessions"]),
        ("/s", ["/sessions"]),
        ("/E", ["/exit"]),
        ("/x", []),
        ("/sessions now", []),  # a space: the name is done
        ("hello /", []),  # a message, not a command
        ("", []),
    ],
)
def test_slash_suggests_the_commands_a_line_may_become(typed: str, names: list[str]) -> None:
    assert [name for name, _ in ui.slash_matches(typed, SLASH)] == names


def test_the_completer_replaces_what_was_typed_and_says_what_a_command_does() -> None:
    from prompt_toolkit.completion import CompleteEvent
    from prompt_toolkit.document import Document

    found = list(ui._completer(SLASH).get_completions(Document("/se"), CompleteEvent()))
    assert [(c.text, c.start_position, c.display_meta_text) for c in found] == [
        ("/sessions", -3, "earlier")
    ]


def test_the_chat_offers_its_own_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    configured()
    offered: list[list[str]] = []

    def line(commands: dict[str, str] | None = None) -> None:
        offered.append(list(commands or {}))
        return None  # the person leaves

    monkeypatch.setattr(ui, "line", line)
    Chat(stored()).run()
    from desman.chat import _commands

    assert offered == [list({**_commands(), **commands.completions()})]


# ── Naming and deleting sessions ─────────────────────────────────────────────


def test_a_session_is_renamed_and_named_back(sessions: Sessions) -> None:
    first = sessions.add(None, Role.USER, "find flights to Oslo").session_id
    sessions.rename(first, "  Trip   to Oslo ")
    named = sessions.get(first)
    assert named is not None and (named.title, named.name) == ("Trip to Oslo", "Trip to Oslo")
    sessions.rename(first, "x" * 500)
    assert len(sessions.get(first).title or "") == 80  # type: ignore[union-attr]
    sessions.rename(first, None)
    back = sessions.get(first)
    assert back is not None and (back.title, back.name) == (None, "find flights to Oslo")


def test_a_deleted_session_leaves_the_lists_but_keeps_its_messages(sessions: Sessions) -> None:
    kept = sessions.add(None, Role.USER, "keep me").session_id
    gone = sessions.add(None, Role.USER, "forget me").session_id
    sessions.delete(gone)
    assert [s.id for s in sessions.recent()] == [kept]
    assert sessions.latest() is not None and sessions.latest().id == kept  # type: ignore[union-attr]
    assert sessions.get(gone) is None
    assert [m.text for m in sessions.messages(gone)] == ["forget me"]


def operator_log() -> list[tuple[str, dict[str, object]]]:
    from desman.core import operator

    conn = db.open_db(db.default_db_path())
    try:
        return [(e.action, e.details) for e in operator.entries(conn)]
    finally:
        conn.close()


def test_rename_names_the_session_in_the_list() -> None:
    configured()
    chat("find flights to Oslo", "/rename Trip to Oslo")
    [session] = stored().recent()
    assert session.title == "Trip to Oslo"
    listed = again("/sessions")
    [row] = [line for line in listed.output.splitlines() if "· current" in line]
    assert "Trip to Oslo" in row and "find flights" not in row
    assert operator_log() == [("operator_session_renamed", {"title": "Trip to Oslo"})]


def test_rename_without_a_name_asks_for_one_and_dash_names_it_back() -> None:
    configured()
    result = chat("find flights to Oslo", "/rename", "Oslo")
    assert "Name of this session" in result.output
    assert stored().recent()[0].title == "Oslo"
    again("/rename -")
    assert stored().recent()[0].title is None


def test_a_key_never_becomes_a_session_name() -> None:
    configured()
    key = "gsk_" + "Abc123" * 8
    result = chat("hello", f"/rename my {key}")
    assert "looked like an API key" in result.output
    title = stored().recent()[0].title or ""
    assert key not in title and key not in str(operator_log())


def test_nothing_to_rename_or_delete_before_the_first_message() -> None:
    configured()
    result = chat("/rename Nothing", "/delete")
    assert result.output.count("nothing to it yet") == 2
    assert operator_log() == []


def test_delete_asks_then_the_chat_starts_afresh() -> None:
    configured()
    chat("find flights to Oslo")
    declined = again("/delete", "n")
    assert "Its messages stay in the database" in declined.output
    assert "Nothing changed." in declined.output and len(stored().recent()) == 1
    deleted = again("/delete", "y", "something else")
    assert "Session deleted." in deleted.output
    [left] = stored().recent()
    assert left.opening == "something else"
    assert [action for action, _ in operator_log()] == ["operator_session_deleted"]
    # The deleted session is never continued, and its messages are still there.
    continued = again()
    assert "find flights" not in continued.output
    conn = db.open_db(db.default_db_path())
    assert (
        conn.execute("SELECT COUNT(*) FROM messages WHERE text LIKE 'find flights%'").fetchone()[0]
        == 1
    )
    conn.close()


# ── Temporary chats (C.11) ───────────────────────────────────────────────────


def temporary(*lines: str) -> Result:
    """`desman -t`."""
    return chat(*lines, args=("-t",))


def texts() -> list[str]:
    conn = db.open_db(db.default_db_path())
    try:
        return [r[0] for r in conn.execute("SELECT text FROM messages ORDER BY seq")]
    finally:
        conn.close()


def test_a_temporary_chat_is_erased_when_it_ends_and_never_listed() -> None:
    configured()
    chat("a kept one")
    result = temporary("a secret plan", "/help")
    assert "Temporary chat" in result.output
    assert texts() == ["a kept one"]  # its messages are gone, the kept session's stay
    assert [s.opening for s in stored().recent()] == ["a kept one"]
    assert "a secret plan" not in again().output  # `-c` continues the kept one
    conn = db.open_db(db.default_db_path())
    [row] = conn.execute("SELECT temporary, owner FROM sessions WHERE temporary = 1")
    assert row["owner"]  # the row stays: tasks and calls of the chat may name it
    conn.close()


def test_a_temporary_chat_has_no_commands_about_kept_sessions() -> None:
    configured()
    result = temporary("/help", "hello", "/sessions", "/rename X", "/compact", "/delete")
    for name in ("/sessions", "/rename", "/compact", "/delete"):
        assert f"{name} is not available in a temporary chat" in result.output
    assert "start a normal session" in result.output
    assert texts() == []


def test_new_in_a_temporary_chat_erases_the_one_before(sessions: Sessions) -> None:
    lines = iter(["first secret", "/new", "second secret"])
    chat_ = Chat(sessions, read=lambda: next(lines, None), temporary=True)
    seen: list[list[str]] = []
    erase = sessions.erase

    def watched(session_id: str) -> None:
        seen.append([m.text for m in sessions.messages(session_id)])
        erase(session_id)

    sessions.erase = watched  # type: ignore[method-assign]
    chat_.run()
    assert seen == [["first secret"]]  # /new switches to a normal, kept session
    assert [m.text for m in sessions.messages(chat_.session_id or "")] == ["second secret"]


def test_what_a_crashed_temporary_chat_left_is_erased_by_the_next_chat(
    sessions: Sessions, monkeypatch: pytest.MonkeyPatch
) -> None:
    import socket

    from desman.core import sessions as sessions_module

    dead = f"{socket.gethostname()}:{2**22 + 12345}"  # above any pid_max: no such process
    monkeypatch.setattr(sessions_module, "current_owner", lambda: dead)
    left = sessions.add(None, Role.USER, "left behind", temporary=True).session_id
    monkeypatch.setattr(sessions_module, "current_owner", lambda: "elsewhere:1")
    alive = sessions.add(None, Role.USER, "another machine's", temporary=True).session_id
    assert sessions.sweep() == 1
    assert sessions.messages(left) == [] and len(sessions.messages(alive)) == 1


def test_only_a_temporary_session_s_messages_may_be_erased(sessions: Sessions) -> None:
    kept = sessions.add(None, Role.USER, "kept").session_id
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        sessions.erase(kept)
    with pytest.raises(sqlite3.IntegrityError, match="temporary from its start"):
        sessions.conn.execute(
            "UPDATE sessions SET temporary = 1, owner = 'x:1' WHERE id = ?", (kept,)
        )
    with pytest.raises(sqlite3.IntegrityError):  # a temporary session always has its owner
        sessions.conn.execute("INSERT INTO sessions (id, temporary) VALUES ('t', 1)")
    assert [m.text for m in sessions.messages(kept)] == ["kept"]


def test_a_temporary_chat_keeps_no_profile_and_no_cached_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from desman import runs
    from desman.llm.fake import FakeLLM

    llm = FakeLLM().on("Chat turn", '{"type": "reply", "text": "Hi!", "profile": [{"add": "x"}]}')
    asked: list[bool] = []

    def models(config: object, conn: object, **kwargs: object) -> runs.Models:
        asked.append(bool(kwargs.get("cache", True)))
        return runs.Models(llm, None)

    monkeypatch.setattr(runs, "models", models)
    config = make_config(language="en", configured=True)
    config.personalize = True
    ConfigManager(default_config_path()).save(config)
    assert "Hi!" in temporary("hello").output
    assert asked == [False]
    [request] = llm.calls
    assert '"profile"' not in request.prompt  # the model was told nothing of the notes
    conn = db.open_db(db.default_db_path())
    assert conn.execute("SELECT COUNT(*) FROM personal_nodes").fetchone()[0] == 0
    conn.close()


@pytest.mark.parametrize(
    "args", [["-t", "list"], ["-c", "status", "x"], ["-c", "-t"]], ids=" ".join
)
def test_c_and_t_are_for_the_chat_alone(args: list[str]) -> None:
    configured()
    result = CliRunner().invoke(cli, args, input="hello\n")
    assert result.exit_code == 2 and "-c" in result.output
    assert not db.default_db_path().exists() or texts() == []


def test_a_temporary_chat_s_command_is_journaled_without_its_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from desman import runs
    from desman.llm.fake import FakeLLM

    turn = '{"type": "command", "name": "run", "args": {"goal": "plan my secret trip"}}'
    llm = FakeLLM().on("Chat turn", turn)
    monkeypatch.setattr(runs, "models", lambda config, conn, **_: runs.Models(llm, None))
    configured()
    temporary("run it", "n")
    assert operator_log() == [
        ("operator_declined", {"command": "run", "args_given": ["goal"], "temporary": True})
    ]
    conn = db.open_db(db.default_db_path())
    dumped = "\n".join(conn.iterdump())
    conn.close()
    assert "secret trip" not in dumped and "run it" not in dumped  # nowhere in the database
