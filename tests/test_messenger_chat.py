"""Only an explicit consent window discloses correspondence to the interlocutor."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import httpx
import pytest
from click.testing import CliRunner

from conftest import FakeClock
from desman import commands, global_cli, messenger_cli, runs
from desman.chat import Chat, _commands
from desman.cli import cli
from desman.config import Config, open_config
from desman.core import db, netlog, operator
from desman.core.sessions import Sessions
from desman.core.store import Store
from desman.interlocutor import CHAT
from desman.llm.fake import FakeLLM
from desman.net import protocol as p
from desman.net import sign
from desman.registry.server import Registry
from test_messenger import call, conn, joined, messenger_enabled, named, registry
from test_registry import Node

pytestmark = pytest.mark.usefixtures("messenger_enabled")

__all__ = ["messenger_enabled", "conn", "joined", "registry"]


@pytest.fixture
def wire(
    registry: Registry, joined: Config, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> list[str]:
    a = Node(registry, clock, sign.node_key(sign.default_node_key_path()))
    a.register()
    call(a, p.MESSENGER_NAME, name="alice")
    b = named(registry, clock, "bobby")
    named(registry, clock, "carol")
    call(a, p.MESSENGER_CONTACT, action="add", name="bobby")
    call(b, p.MESSENGER_CONTACT, action="add", name="alice")
    call(b, p.MESSENGER_SEND, to="alice", text="private incoming marker")
    paths: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return a._handle(request)

    original = messenger_cli.perform

    def perform(manager: Any, conn: Any, body: Any, **kwargs: Any) -> messenger_cli.Response:
        return original(
            manager, conn, body, transport=httpx.MockTransport(respond), clock=clock, **kwargs
        )

    monkeypatch.setattr(messenger_cli, "perform", perform)
    return paths


@pytest.mark.parametrize("temporary", [False, True])
def test_slash_privacy(
    conn: sqlite3.Connection,
    wire: list[str],
    temporary: bool,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    llm = FakeLLM()
    monkeypatch.setattr(runs, "models", lambda *a, **k: runs.Models(llm, None))
    chat = Chat(Sessions(conn), temporary=temporary)
    assert chat._slash("/messages bobby")
    assert wire == [p.MESSENGER_INBOX]
    assert "private incoming marker" in capsys.readouterr().out
    assert conn.execute("SELECT count(*) FROM messages").fetchone()[0] == 0
    assert operator.entries(conn) == []
    assert llm.calls == [] and chat.session_id is None
    assert "/messages" in _commands(temporary)
    chat._help()
    assert "/messages" in capsys.readouterr().out
    chat._slash("/messages Invalid Name")
    assert "Invalid messenger request" in capsys.readouterr().err
    assert wire == [p.MESSENGER_INBOX]


def test_slash_off_continues(
    conn: sqlite3.Connection, wire: list[str], joined: Config, capsys: pytest.CaptureFixture[str]
) -> None:
    joined.network.enabled = False
    open_config().save(joined)
    chat = Chat(Sessions(conn))
    assert chat._slash("/messages")
    assert "is off" in capsys.readouterr().err
    assert chat._slash("/help")
    assert "/messages" in capsys.readouterr().out and wire == []


def test_catalog_defers_fetch(conn: sqlite3.Connection, wire: list[str]) -> None:
    entry = commands.find("global messages")
    assert entry is not None
    args = entry.args.model_validate({"name": "bobby"})
    assert entry.kind_of(args) is commands.Kind.CHANGE
    ctx = commands.Context(Store(conn), open_config(), lambda *_: None, lambda _: False)
    proposal = entry.handler(ctx, args)
    assert isinstance(proposal, commands.Proposal) and wire == []
    assert "Show your messages to the interlocutor?" in proposal.lines
    assert not netlog.entries(conn)
    result = proposal.apply()
    assert wire == [p.MESSENGER_INBOX]
    assert result.data["messages"][0]["untrusted"] == messenger_cli.UNTRUSTED
    assert any(e.action == "messenger_notice" for e in netlog.entries(conn))
    contacts = commands.find("global contacts")
    assert contacts is not None
    state = contacts.handler(ctx, commands.NoArgs())
    assert isinstance(state, commands.Outcome)
    assert list(state.data["fingerprints"]) == ["bobby"]
    assert len(state.data["fingerprints"]["bobby"]) == 12


@pytest.mark.parametrize("answer", ["y", "n"])
def test_chat_consent_and_journal(
    wire: list[str], joined: Config, monkeypatch: pytest.MonkeyPatch, answer: str
) -> None:
    llm = FakeLLM()
    llm.on(
        CHAT,
        json.dumps({"type": "command", "name": "global messages", "args": {"name": "bobby"}}),
        json.dumps({"type": "reply", "text": "Done"}),
    )
    monkeypatch.setattr(runs, "models", lambda *a, **k: runs.Models(llm, None))
    joined.configured = True
    open_config().save(joined)
    result = CliRunner().invoke(
        cli, [], input=f"show my messages\n{answer}\nwhat did they say?\n/exit\n"
    )
    assert result.exit_code == 0, result.output
    assert "Show your messages to the interlocutor?" in result.output
    assert wire == ([p.MESSENGER_INBOX] if answer == "y" else [])
    conn = db.open_db()
    try:
        entries = operator.entries(conn)
        assert any(e.details.get("command") == "global messages" for e in entries)
        assert "private incoming marker" not in str(entries)
        assert ("private incoming marker" in str(llm.calls)) == (answer == "y")
    finally:
        conn.close()


def test_suspended_cli_warning(
    conn: sqlite3.Connection, joined: Config, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    suspended = p.Suspension(
        until=clock(), reason="[bold]operator reason", record="a" * 32, operator="b" * 64
    )
    inbox = p.Inbox(
        name="alice", contacts=[], asked=[], asking=[], blocked=[], messages=[], suspended=suspended
    )
    monkeypatch.setattr(messenger_cli, "perform", lambda *a, **k: inbox)
    result = CliRunner().invoke(cli, ["global", "messages"])
    assert result.exit_code == 0 and "[bold]operator reason" in result.output
    assert "suspended by the registry" in result.output
    result = CliRunner().invoke(cli, ["global", "messages", "--json"])
    assert json.loads(result.output)["suspended"]["record"] == suspended.record


def test_suspended_refusal(conn: sqlite3.Connection, joined: Config, clock: FakeClock) -> None:
    transport = httpx.MockTransport(lambda req: httpx.Response(403, json={"error": p.SUSPENDED}))
    from click import ClickException

    with pytest.raises(ClickException, match="operator has suspended your messaging"):
        messenger_cli.perform(
            open_config(), conn, p.SetName(name="alice"), transport=transport, clock=clock
        )


def test_slash_malformed_config_continues(
    conn: sqlite3.Connection, wire: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    chat = Chat(Sessions(conn))
    open_config().config_path.write_text("{malformed", encoding="utf-8")
    assert chat._slash("/messages")
    assert "not valid JSON" in capsys.readouterr().err
    assert chat._slash("/help")
    assert "/messages" in capsys.readouterr().out
    assert wire == []


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["complain", "2", "text"], "No contact named 2."),
        (["complain", "nobody", "text"], "No contact named nobody."),
        (["complain", "bobby", "text", "--message", "99"], "Message #99 is not from bobby."),
        (["send", "nobody", "text"], "No contact named nobody."),
        (["contact", "add", "nobody"], "No messenger name nobody exists."),
        (["messages", "nobody"], "No contact named nobody."),
        (["complain", "carol", "text"], "You have never corresponded with carol."),
        (["send", "Invalid Name", "text"], "Invalid messenger request; nothing was sent."),
        (["send", "9" * 33, "text"], "Invalid messenger request; nothing was sent."),
    ],
)
def test_cli_messenger_refusal(wire: list[str], args: list[str], expected: str) -> None:
    result = CliRunner().invoke(cli, ["global", *args])
    assert result.exit_code == 1
    assert expected in result.output


def test_notice_is_shared_by_cli_and_chat(wire: list[str]) -> None:
    runner = CliRunner()
    raw = runner.invoke(cli, ["global", "messages", "--json"])
    assert json.loads(raw.output)["warning"] == messenger_cli.WARNING
    contacts = runner.invoke(cli, ["global", "contacts"])
    assert contacts.exit_code == 0 and "Messages are end-to-end encrypted" in contacts.output
    with global_cli.journal(None) as shared, runner.isolation() as (output, *_):
        chat = Chat(Sessions(shared))
        assert chat._slash("/messages bobby")
        assert "Messages are end-to-end encrypted" not in output.getvalue().decode()
        assert [entry.action for entry in netlog.entries(shared)].count("messenger_notice") == 1
    again = runner.invoke(cli, ["global", "messages"])
    assert again.exit_code == 0 and "Messages are end-to-end encrypted" not in again.output


def test_first_send_window_displays_notice_only_once(
    joined: Config, monkeypatch: pytest.MonkeyPatch, clock: FakeClock
) -> None:
    llm = FakeLLM().on(
        CHAT,
        json.dumps(
            {"type": "command", "name": "global send", "args": {"name": "bobby", "text": "hello"}}
        ),
        repeat_last=True,
    )
    monkeypatch.setattr(runs, "models", lambda *a, **k: runs.Models(llm, None))
    monkeypatch.setattr(
        messenger_cli, "perform", lambda *a, **k: p.MessageSent(id=1, sent_at=clock())
    )
    joined.configured = True
    open_config().save(joined)
    result = CliRunner().invoke(cli, [], input="send hello\nn\nsend again\ny\nsend once more\ny\n")
    assert result.exit_code == 0, result.output
    assert result.output.count("Messages are end-to-end encrypted") == 2
    assert result.output.index("Messages are end-to-end encrypted") < result.output.index("(Y/N)")
    assert result.output.count("Message sent as 1.") == 2
    assert messenger_cli.WARNING not in str(llm.calls)


def test_notice_only_after_joined_and_named(
    joined: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    unnamed = p.Inbox(name=None, contacts=[], asked=[], asking=[], blocked=[], messages=[])
    joined.network.enabled = False
    open_config().save(joined)
    runner = CliRunner()
    refused = runner.invoke(cli, ["global", "contacts"])
    assert refused.exit_code == 1 and isinstance(refused.exception, SystemExit)
    assert "Messages are end-to-end encrypted" not in refused.output
    monkeypatch.setattr(messenger_cli, "perform", lambda *a, **kw: unnamed)
    empty = runner.invoke(cli, ["global", "name"])
    assert empty.exit_code == 0 and "No messenger name yet." in empty.output
    assert "Messages are end-to-end encrypted" not in empty.output
    with global_cli.journal(None) as journal:
        assert not any(e.action == "messenger_notice" for e in netlog.entries(journal))


def test_notice_read_avoids_write_lock_and_reports_busy(tmp_path: Path) -> None:
    from click import ClickException

    path = tmp_path / "notice.db"
    reader, writer = db.open_db(path), db.open_db(path)
    try:
        reader.execute("PRAGMA busy_timeout=0")
        with (
            db.transaction(writer),
            pytest.raises(ClickException, match="the local journal is busy"),
        ):
            messenger_cli.take_notice(reader)
        assert messenger_cli.take_notice(reader) is not None
        with db.transaction(writer):
            assert messenger_cli.take_notice(reader) is None
    finally:
        reader.close()
        writer.close()


def test_cli_reports_busy_when_opening_journal(monkeypatch: pytest.MonkeyPatch) -> None:
    def locked(*_args: object, **_kwargs: object) -> sqlite3.Connection:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(db, "open_db", locked)
    result = CliRunner().invoke(cli, ["global", "contacts"])
    assert result.exit_code == 1 and isinstance(result.exception, SystemExit)
    assert "The local journal is busy." in result.output
