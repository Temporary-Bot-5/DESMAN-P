"""Tests for chat sessions (ROADMAP C.2): the store, the migration and bare `desman`."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterator

import pytest
from click.testing import CliRunner, Result

from conftest import FakeClock
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


def chat(*lines: str) -> Result:
    result = CliRunner().invoke(cli, [], input="".join(f"{line}\n" for line in lines))
    assert result.exit_code == 0, result.output
    return result


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


def test_messages_are_kept_and_the_chat_continues_the_last_session() -> None:
    configured()
    first = chat("find cheap flights to Lisbon", "/exit", "never read")
    assert "New session" in first.output
    assert "no model is ready to answer" in first.output
    [info] = stored().recent()
    assert info.opening == "find cheap flights to Lisbon" and info.messages == 1

    second = chat("and back on Sunday")
    assert "Continuing the session" in second.output
    assert "find cheap flights to Lisbon" in second.output  # the last messages are shown
    [info] = stored().recent()
    assert info.messages == 2


def test_new_starts_a_session_with_the_next_message() -> None:
    configured()
    chat("first")
    result = chat("/new", "/new", "second")
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
    chat("/new", "about trains")
    # Listed newest first: 1 is "about trains" (current), 2 is "about flights".
    result = chat("/sessions", "2", "more on flights")
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
    assert "unknown command /frobnicate" in result.output
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
    result = chat("/sessions", "3", "1", "back to the oldest")
    assert "earlier sessions…" in result.output
    assert "Switched to the session" in result.output
    assert [s.opening for s in stored().recent()] == ["oldest", "newest", "middle"]
    assert [s.opening for s in stored().recent(limit=1, offset=2)] == ["middle"]
