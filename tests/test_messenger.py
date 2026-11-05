"""Correspondence stays human data: consent, privacy, limits and signed boundaries."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import suppress
from pathlib import Path
from typing import Any

import httpx
import pytest
from click import ClickException
from click.testing import CliRunner
from pydantic import ValidationError

from conftest import FakeClock
from desman import commands, messenger_cli
from desman.cli import cli
from desman.config import ApiKey, Config, ConfigManager, Network, default_config_path, open_config
from desman.core import db
from desman.core.store import Store
from desman.net import messenger, seal, sign
from desman.net import protocol as p
from desman.registry import store
from desman.registry.server import Registry
from test_registry import Node


@pytest.fixture
def registry(tmp_path: Path, clock: FakeClock) -> Registry:
    path = tmp_path / "registry.db"
    store.open_registry(path).close()
    return Registry(path, clock=clock)


@pytest.fixture
def conn() -> Iterator[sqlite3.Connection]:
    opened = db.open_db(":memory:")
    yield opened
    opened.close()


def call(node: Node, path: str, **body: Any) -> dict[str, Any]:
    # Existing behavioral tests use clear drafts; the wire always carries sealed envelopes.
    if path == p.MESSENGER_NAME:
        with store.connection(node.registry.path) as conn:
            row = conn.execute(
                "SELECT 1 FROM nodes WHERE fingerprint=? AND status='active' AND fingerprint NOT IN (SELECT node FROM messenger_keys)",
                (node.fingerprint,),
            ).fetchone()
        if row:
            node.call(
                "POST", p.MESSENGER_KEY, {"key": seal.publish(node.key).model_dump(mode="json")}
            )
    if path == p.MESSENGER_SEND and "text" in body:
        try:
            draft = messenger_cli.MessageDraft.model_validate(body)
        except ValidationError:
            return dict(node.call("POST", path, body).json())
        with store.connection(node.registry.path) as conn:
            row = conn.execute(
                "SELECT node FROM messenger_names WHERE name=?", (draft.to,)
            ).fetchone()
            public = (
                conn.execute(
                    "SELECT public_key FROM nodes WHERE fingerprint=?", (row[0],)
                ).fetchone()[0]
                if row
                else sign.encode_public(node.key.public)
            )
        recipient = sign.decode_public(public)
        # Only test fixtures hold both private keys; obtain the already published X25519 key.
        with store.connection(node.registry.path) as conn:
            row = conn.execute(
                "SELECT x25519 FROM messenger_keys WHERE node=?", (sign.fingerprint(recipient),)
            ).fetchone()
        x25519 = row[0] if row else seal.publish(node.key).x25519
        body = {
            "to": draft.to,
            "sealed": seal.seal(
                node.key, sign.fingerprint(recipient), x25519, draft.text, node.clock()
            ).model_dump(mode="json"),
        }
    if path == p.MESSENGER_COMPLAIN and body.get("message") is not None:
        with store.connection(node.registry.path) as conn:
            row = conn.execute(
                "SELECT sealed FROM messages WHERE id=?", (body["message"],)
            ).fetchone()
        if row and row[0]:
            with suppress(seal.SealError):
                body["content_key"] = seal.encode(
                    seal.content_key(node.key, seal.Sealed.model_validate_json(row[0]))
                )
    result = dict(node.call("POST", path, body).json())
    if path == p.MESSENGER_INBOX and "messages" in result:
        for message in result["messages"]:
            if message.get("sealed") is not None:
                message["text"] = seal.open_with(
                    seal.content_key(node.key, seal.Sealed.model_validate(message["sealed"])),
                    seal.Sealed.model_validate(message["sealed"]),
                ).letter.text
    return result


def named(registry: Registry, clock: FakeClock, name: str) -> Node:
    node = Node(registry, clock)
    assert node.register().status_code == 200
    assert call(node, p.MESSENGER_NAME, name=name) == {"name": name}
    return node


def connect(a: Node, b: Node, an: str = "alice", bn: str = "bobby") -> None:
    assert call(a, p.MESSENGER_CONTACT, action="add", name=bn)["state"] == "asked"
    assert call(b, p.MESSENGER_CONTACT, action="add", name=an)["state"] == "contact"


def test_names(registry: Registry, clock: FakeClock) -> None:
    a = named(registry, clock, "alice")
    b = named(registry, clock, "bobby")
    assert call(b, p.MESSENGER_NAME, name="alice")["error"] == "name_taken"
    for name in [
        "admin",
        "desman",
        "desman-fake",
        "operator",
        "registry",
        "root",
        "system",
        "support",
    ]:
        assert call(a, p.MESSENGER_NAME, name=name)["error"] == "name_reserved"
    for name in ["Abc", "ab", "аbc", "a b", "abc\n", "a" * 33]:
        assert call(a, p.MESSENGER_NAME, name=name)["error"] == "malformed"
    assert call(a, p.MESSENGER_NAME, name="alice_new")["name"] == "alice_new"
    assert call(b, p.MESSENGER_NAME, name="alice")["name"] == "alice"
    unnamed = Node(registry, clock)
    unnamed.register()
    assert call(unnamed, p.MESSENGER_INBOX)["name"] is None
    assert call(unnamed, p.MESSENGER_CONTACT, action="add", name="alice")["error"] == "no_name"


def test_consent_conversations_and_windows(registry: Registry, clock: FakeClock) -> None:
    a, b = named(registry, clock, "alice"), named(registry, clock, "bobby")
    c = named(registry, clock, "carol")
    assert call(a, p.MESSENGER_CONTACT, action="add", name="bobby")["state"] == "asked"
    assert call(b, p.MESSENGER_INBOX)["asking"] == ["alice"]
    assert call(a, p.MESSENGER_INBOX)["asked"] == ["bobby"]
    assert call(a, p.MESSENGER_SEND, to="bobby", text="hello")["error"] == "not_contact"
    assert call(b, p.MESSENGER_CONTACT, action="add", name="alice")["state"] == "contact"
    connect(a, c, bn="carol")
    first = call(a, p.MESSENGER_SEND, to="bobby", text="first")["id"]
    call(b, p.MESSENGER_SEND, to="alice", text="reply")
    call(a, p.MESSENGER_SEND, to="carol", text="other")
    assert len(call(a, p.MESSENGER_INBOX)["messages"]) == 3
    filtered = call(a, p.MESSENGER_INBOX, after=first, with_name="bobby")["messages"]
    assert [m["text"] for m in filtered] == ["reply"]
    call(b, p.MESSENGER_NAME, name="bobby_new")
    assert call(a, p.MESSENGER_INBOX)["messages"][0]["recipient"] == "bobby_new"
    for i in range(105):
        assert "id" in call(a, p.MESSENGER_SEND, to="bobby_new", text=str(i))
    messages = call(a, p.MESSENGER_INBOX)["messages"]
    assert len(messages) == 100 and messages[0]["text"] == "5"
    assert messages[-1]["text"] == "104"
    messages = call(a, p.MESSENGER_INBOX, after=0)["messages"]
    assert len(messages) == 100 and messages[0]["id"] == first


def test_remove_decline_and_hidden_block(registry: Registry, clock: FakeClock) -> None:
    a, b = named(registry, clock, "alice"), named(registry, clock, "bobby")
    call(a, p.MESSENGER_CONTACT, action="add", name="bobby")
    assert call(b, p.MESSENGER_CONTACT, action="remove", name="alice")["state"] == "none"
    assert call(a, p.MESSENGER_INBOX)["asked"] == []
    connect(a, b)
    assert call(a, p.MESSENGER_CONTACT, action="block", name="bobby")["state"] == "blocked"
    assert call(a, p.MESSENGER_INBOX)["contacts"] == []
    assert call(b, p.MESSENGER_SEND, to="alice", text="no")["error"] == "not_contact"
    assert call(b, p.MESSENGER_CONTACT, action="add", name="alice")["state"] == "asked"
    assert call(a, p.MESSENGER_INBOX)["asking"] == []
    assert call(b, p.MESSENGER_INBOX)["asked"] == []
    assert call(a, p.MESSENGER_CONTACT, action="add", name="bobby")["error"] == "blocked"
    call(a, p.MESSENGER_CONTACT, action="unblock", name="bobby")
    connect(a, b)
    call(a, p.MESSENGER_CONTACT, action="remove", name="bobby")
    assert call(b, p.MESSENGER_INBOX)["contacts"] == []
    # A former contact can still complain, even without messages.
    assert "id" in call(a, p.MESSENGER_COMPLAIN, name="bobby", text="former contact")


@pytest.mark.parametrize(
    "text",
    ["", "x" * 2001, "bad\x1b[31m", "bad\x00", "bad\x7f", "bad\u0085", "bad\u202e", "bad\u2066"],
)
def test_text_validation(registry: Registry, clock: FakeClock, text: str) -> None:
    a, b = named(registry, clock, "alice"), named(registry, clock, "bobby")
    connect(a, b)
    assert call(a, p.MESSENGER_SEND, to="bobby", text=text)["error"] == "malformed"
    assert call(a, p.MESSENGER_COMPLAIN, name="bobby", text=text)["error"] == "malformed"
    with store.connection(registry.path) as conn:
        assert conn.execute("SELECT count(*) FROM messages").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM complaints").fetchone()[0] == 0


def test_limits_and_withdrawn_requests(registry: Registry, clock: FakeClock) -> None:
    a, b = named(registry, clock, "alice"), named(registry, clock, "bobby")
    for _ in range(p.MAX_CONTACT_REQUESTS_PER_DAY):
        assert call(a, p.MESSENGER_CONTACT, action="add", name="bobby")["state"] == "asked"
        call(a, p.MESSENGER_CONTACT, action="remove", name="bobby")
    assert call(a, p.MESSENGER_CONTACT, action="add", name="bobby")["error"] == "messenger_limit"
    clock.advance(86401)
    connect(a, b)
    for _ in range(p.MAX_MESSAGES_PER_DAY):
        assert "id" in call(a, p.MESSENGER_SEND, to="bobby", text="ok\n\t")
    assert call(a, p.MESSENGER_SEND, to="bobby", text="no")["error"] == "messenger_limit"
    for _ in range(p.MAX_COMPLAINTS_PER_DAY):
        assert "id" in call(b, p.MESSENGER_COMPLAIN, name="alice", text="complaint")
    assert call(b, p.MESSENGER_COMPLAIN, name="alice", text="sixth")["error"] == "messenger_limit"
    clock.advance(86401)
    assert "id" in call(a, p.MESSENGER_SEND, to="bobby", text="next day")
    assert "id" in call(b, p.MESSENGER_COMPLAIN, name="alice", text="next day")


def test_complaints_retention_and_journal(registry: Registry, clock: FakeClock) -> None:
    a, b = named(registry, clock, "alice"), named(registry, clock, "bobby")
    c = named(registry, clock, "carol")
    assert call(c, p.MESSENGER_COMPLAIN, name="alice", text="stranger")["error"] == "unknown_name"
    connect(a, b)
    sent = call(a, p.MESSENGER_SEND, to="bobby", text="private message marker")
    assert (
        call(c, p.MESSENGER_COMPLAIN, name="alice", text="wrong recipient", message=sent["id"])[
            "error"
        ]
        == "unknown_name"
    )
    assert (
        call(a, p.MESSENGER_COMPLAIN, name="bobby", text="wrong direction", message=sent["id"])[
            "error"
        ]
        == "unknown_name"
    )
    complaint = call(
        b, p.MESSENGER_COMPLAIN, name="alice", text="private complaint marker", message=sent["id"]
    )
    clock.advance(31 * 86400)
    assert call(a, p.MESSENGER_INBOX)["messages"] == []
    call(b, p.MESSENGER_SEND, to="alice", text="new")
    assert [m["text"] for m in call(a, p.MESSENGER_INBOX)["messages"]] == ["new"]
    with store.connection(registry.path) as conn:
        evidence = conn.execute(
            "SELECT message_text FROM complaints WHERE id = ?", (complaint["id"],)
        ).fetchone()[0]
        assert evidence == "private message marker"
        logs = str([tuple(r) for r in conn.execute("SELECT * FROM registry_log")])
        assert "private message marker" not in logs and "private complaint marker" not in logs


@pytest.fixture
def joined() -> Config:
    config = Config(language="en", network=Network(enabled=True, registry="https://registry.test"))
    ConfigManager(default_config_path()).save(config)
    sign.node_key(sign.default_node_key_path())
    return config


def test_client_registration_and_network_journal(
    registry: Registry, clock: FakeClock, conn: sqlite3.Connection, joined: Config
) -> None:
    key = sign.node_key(sign.default_node_key_path())
    a = Node(registry, clock, key=key)
    transport = httpx.MockTransport(a._handle)
    manager = open_config()

    def perform(body: messenger_cli.Request) -> messenger_cli.Response:
        return messenger_cli.perform(manager, conn, body, transport=transport, clock=clock)

    assert isinstance(perform(p.SetName(name="alice")), p.MessengerName)
    b = named(registry, clock, "bobby")
    perform(p.ContactAction(action="add", name="bobby"))
    call(b, p.MESSENGER_CONTACT, action="add", name="alice")
    perform(messenger_cli.MessageDraft(to="bobby", text="private outbound marker"))
    call(b, p.MESSENGER_SEND, to="alice", text="incoming")
    perform(p.Complain(name="bobby", text="private complaint marker"))
    before = conn.execute("SELECT count(*) FROM network_log").fetchone()[0]
    perform(p.InboxQuery())
    assert conn.execute("SELECT count(*) FROM network_log").fetchone()[0] == before
    logs = [tuple(r) for r in conn.execute("SELECT * FROM network_log")]
    assert "private outbound marker" not in str(logs) and "private complaint marker" not in str(
        logs
    )
    assert len([r for r in logs if r[2] == "registered"]) == 1


def test_registration_retry_once(
    conn: sqlite3.Connection, joined: Config, clock: FakeClock
) -> None:
    paths: list[str] = []

    def respond(req: httpx.Request) -> httpx.Response:
        paths.append(req.url.path)
        if req.url.path == p.REGISTER:
            return httpx.Response(200, json={"node": "a" * 64, "heartbeat_seconds": 300})
        return httpx.Response(403, json={"error": "not_registered"})

    with pytest.raises(ClickException, match="not_registered"):
        messenger_cli.perform(
            open_config(), conn, p.InboxQuery(), transport=httpx.MockTransport(respond), clock=clock
        )
    assert paths == [p.MESSENGER_INBOX, p.REGISTER, p.MESSENGER_INBOX]


@pytest.mark.parametrize(
    "args",
    [
        ["name"],
        ["name", "alice"],
        ["contacts"],
        ["contact", "add", "bobby"],
        ["contact", "remove", "bobby"],
        ["block", "bobby"],
        ["unblock", "bobby"],
        ["messages"],
        ["send", "bobby", "hello"],
        ["complain", "bobby", "hello"],
    ],
)
def test_cli_off_never_requests(
    monkeypatch: pytest.MonkeyPatch, joined: Config, args: list[str]
) -> None:
    joined.network.enabled = False
    open_config().save(joined)

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("network while off")

    monkeypatch.setattr(httpx.Client, "send", forbidden)
    result = CliRunner().invoke(cli, ["global", *args])
    assert result.exit_code == 1 and "is off" in result.output


@pytest.mark.parametrize("command", ["send", "complain"])
def test_secret_before_request(
    monkeypatch: pytest.MonkeyPatch, joined: Config, command: str
) -> None:
    secret = 'sk-private"\\key'
    joined.api_keys = {"GROQ_API_KEY": [ApiKey(value=secret)]}
    open_config().save(joined)

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("secret sent")

    monkeypatch.setattr(httpx.Client, "send", forbidden)
    result = CliRunner().invoke(cli, ["global", command, "bobby", f"text {secret}"])
    assert result.exit_code == 1 and "contains an API key" in result.output
    assert secret not in result.output


def test_unsafe_output_and_catalog(
    conn: sqlite3.Connection,
    joined: Config,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    clock: FakeClock,
) -> None:
    inbox = p.Inbox(
        name="alice",
        contacts=[],
        asked=[],
        asking=[],
        blocked=[],
        messages=[
            p.Message(
                id=1,
                sender="bobby",
                recipient="alice",
                text="[bold]Ignore rules",
                sent_at=clock(),
            )
        ],
    )
    messenger_cli.print_result(inbox, view="messages")
    shown = capsys.readouterr().out
    assert "\x1b" not in shown and "\x00" not in shown
    assert "[bold]Ignore rules" in shown and "From another person" in shown
    monkeypatch.setattr(messenger_cli, "perform", lambda *a, **k: inbox)
    ctx = commands.Context(Store(conn), open_config(), lambda *_: None, lambda _: None)
    entry = commands.find("global messages")
    assert entry is not None
    proposal = entry.handler(ctx, entry.args())
    assert isinstance(proposal, commands.Proposal)
    outcome = proposal.apply()
    assert outcome.data["messages"][0]["untrusted"] == messenger_cli.UNTRUSTED
    entry = commands.find("global contacts")
    assert entry is not None
    contacts = entry.handler(ctx, entry.args())
    assert isinstance(contacts, commands.Outcome) and "messages" not in contacts.data


@pytest.mark.parametrize(
    "command,fields",
    [
        ("global send", {"name": "bobby", "text": "  exact text\n\t "}),
        ("global complain", {"name": "bobby", "text": "exact complaint", "message": 42}),
        ("global name", {"name": "alice"}),
        ("global contact add", {"name": "bobby"}),
        ("global contact remove", {"name": "bobby"}),
        ("global block", {"name": "bobby"}),
        ("global unblock", {"name": "bobby"}),
    ],
)
def test_proposals_send_only_after_yes(
    conn: sqlite3.Connection,
    joined: Config,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    fields: dict[str, Any],
) -> None:
    sent: list[messenger_cli.Request] = []

    def perform(
        manager: Any, conn: Any, body: messenger_cli.Request, **kwargs: Any
    ) -> p.MessageSent:
        sent.append(body)
        return p.MessageSent(id=1, sent_at=messenger._utc_now())

    monkeypatch.setattr(messenger_cli, "perform", perform)
    ctx = commands.Context(Store(conn), open_config(), lambda *_: None, lambda _: None)
    entry = commands.find(command)
    assert entry is not None
    args = entry.args.model_validate(fields)
    assert entry.kind_of(args) is commands.Kind.CHANGE
    proposal = entry.handler(ctx, args)
    assert isinstance(proposal, commands.Proposal) and not sent
    assert fields["name"] in proposal.lines[1]
    if "text" in fields:
        assert proposal.lines[-1] == fields["text"]
    if "message" in fields:
        assert proposal.lines[2] == "Cited message: 42"
        assert "The operator will read the cited message." in proposal.lines
    assert proposal.warnings == [messenger_cli.warning()]
    outcome = proposal.apply()
    if "text" in fields:
        assert fields["text"] not in json.dumps(outcome.data)
    assert len(sent) == 1
    payload = sent[0].model_dump(mode="json")
    assert (payload.get("to") or payload.get("name")) == fields["name"]
    if "text" in fields:
        assert payload["text"] == fields["text"]


def test_upgrade_from_eight(tmp_path: Path) -> None:
    path = tmp_path / "old.db"
    conn = db.connect(path)
    db.migrate(conn, store.MIGRATIONS[:8])
    conn.close()
    conn = store.open_registry(path)
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == len(store.MIGRATIONS)
        assert conn.execute("SELECT count(*) FROM messages").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM reports").fetchone()[0] == 0
    finally:
        conn.close()


def test_missing_registry_and_locked_recheck(
    conn: sqlite3.Connection, joined: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("request should not leave")

    monkeypatch.setattr(httpx.Client, "send", forbidden)
    ctx = commands.Context(Store(conn), open_config(), lambda *_: None, lambda _: None)
    entry = commands.find("global send")
    assert entry is not None
    proposal = entry.handler(
        ctx, entry.args.model_validate({"name": "bobby", "text": "new-secret"})
    )
    assert isinstance(proposal, commands.Proposal)
    joined.api_keys = {"GROQ_API_KEY": [ApiKey(value="new-secret")]}
    open_config().save(joined)
    with pytest.raises(ClickException, match="contains an API key"):
        proposal.apply()
    joined.network.registry = None
    open_config().save(joined)
    monkeypatch.setattr("desman.net.DEFAULT_REGISTRY", None)
    with pytest.raises(ClickException, match="No registry address"):
        messenger_cli.perform(open_config(), conn, p.InboxQuery())


def test_refusals_are_journaled_and_inactive_nodes_refused(
    registry: Registry, clock: FakeClock
) -> None:
    a = named(registry, clock, "alice")
    call(a, p.MESSENGER_NAME, name="root")
    a.call("POST", p.LEAVE)
    requests: list[tuple[str, dict[str, Any]]] = [
        (p.MESSENGER_NAME, {"name": "alice"}),
        (p.MESSENGER_CONTACT, {"action": "add", "name": "bobby"}),
        (
            p.MESSENGER_SEND,
            {
                "to": "bobby",
                "sealed": seal.seal(
                    a.key, a.fingerprint, seal.publish(a.key).x25519, "hello", clock()
                ).model_dump(mode="json"),
            },
        ),
        (p.MESSENGER_INBOX, {}),
        (p.MESSENGER_COMPLAIN, {"name": "bobby", "text": "hello"}),
    ]
    for path, body in requests:
        assert a.call("POST", path, body).json()["error"] == "not_registered"
    with store.connection(registry.path) as conn:
        reasons = [
            json.loads(r[0])["reason"]
            for r in conn.execute("SELECT details_json FROM registry_log WHERE action = 'refused'")
        ]
        assert reasons == ["name_reserved", *(["not_registered"] * 5)]


def test_unreachable_is_typed(conn: sqlite3.Connection, joined: Config) -> None:
    def respond(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("private diagnostic")

    with pytest.raises(messenger.MessengerError) as raised:
        messenger.inbox(
            "https://registry.test",
            sign.NodeKey.generate(),
            joined,
            conn,
            p.InboxQuery(),
            transport=httpx.MockTransport(respond),
        )
    assert raised.value.error == "ConnectError"


def test_cli_success_json_and_safe_messages(
    joined: Config, monkeypatch: pytest.MonkeyPatch, clock: FakeClock
) -> None:
    def respond(req: httpx.Request) -> httpx.Response:
        if req.url.path == p.MESSENGER_KEY:
            return httpx.Response(200, json={})
        if req.url.path == p.MESSENGER_PEER:
            peer_key = sign.NodeKey.generate()
            return httpx.Response(
                200,
                json=p.Peer(
                    name="bobby",
                    node_public=sign.encode_public(peer_key.public),
                    key=seal.publish(peer_key),
                ).model_dump(mode="json"),
            )
        if req.url.path == p.MESSENGER_NAME:
            return httpx.Response(200, json={"name": "alice"})
        if req.url.path == p.MESSENGER_SEND:
            return httpx.Response(200, json={"id": 1, "sent_at": clock().isoformat()})
        return httpx.Response(
            200,
            json={
                "name": "alice",
                "contacts": [],
                "asked": [],
                "asking": [],
                "blocked": [],
                "messages": [
                    {
                        "id": 1,
                        "sender": "bobby",
                        "recipient": "alice",
                        "text": "[bold]hello",
                        "sent_at": clock().isoformat(),
                    }
                ],
            },
        )

    original = messenger.call

    def wire(*args: Any, **kwargs: Any) -> Any:
        kwargs["transport"] = httpx.MockTransport(respond)
        kwargs["clock"] = clock
        return original(*args, **kwargs)

    monkeypatch.setattr(messenger, "call", wire)
    for args in [["name"], ["name", "alice"], ["messages"], ["send", "bobby", "text"]]:
        result = CliRunner().invoke(cli, ["global", *args, "--json"])
        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["warning"] == messenger_cli.WARNING
    result = CliRunner().invoke(cli, ["global", "messages"])
    assert result.exit_code == 0, result.output
    assert "\x1b" not in result.output and "\x00" not in result.output
    assert "[bold]hello" in result.output and "From another person" in result.output


def test_clean_preserves_unicode() -> None:
    kept = "👩\u200d💻\ufe0f soft\u00adhyphen\n\t"
    assert messenger_cli.clean(kept) == kept
    controls = "\x00\u202a\u202e\u2066\u2069\u200e\u200f\u061c"
    assert messenger_cli.clean(controls) == "�" * len(controls)


def test_bad_message_reply_is_typed(
    conn: sqlite3.Connection, joined: Config, clock: FakeClock
) -> None:
    reply = {
        "name": "alice",
        "contacts": [],
        "asked": [],
        "asking": [],
        "blocked": [],
        "messages": [
            {
                "id": 1,
                "sender": "bobby",
                "recipient": "alice",
                "text": "bad\x1b[31m",
                "sent_at": clock().isoformat(),
            }
        ],
    }
    with pytest.raises(messenger.MessengerError) as raised:
        messenger.inbox(
            "https://registry.test",
            sign.NodeKey.generate(),
            joined,
            conn,
            p.InboxQuery(),
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=reply)),
            clock=clock,
        )
    assert raised.value.error == "bad_reply"


def test_name_contracts_are_independent() -> None:
    for body in [
        p.ContactAction(name="alice", action="add"),
        p.ContactState(name="alice", state="asked"),
        p.Complain(name="alice", text="reason"),
    ]:
        assert not isinstance(body, p.SetName)


def test_departed_contact_and_return(registry: Registry, clock: FakeClock) -> None:
    a, b = named(registry, clock, "alice"), named(registry, clock, "bobby")
    connect(a, b)
    assert b.call("POST", p.LEAVE).status_code == 200
    assert call(a, p.MESSENGER_SEND, to="bobby", text="lost")["error"] == p.AWAY
    assert call(a, p.MESSENGER_CONTACT, action="add", name="bobby")["error"] == p.AWAY
    with store.connection(registry.path) as opened:
        assert opened.execute("SELECT count(*) FROM messages").fetchone()[0] == 0
        assert opened.execute("SELECT count(*) FROM contacts").fetchone()[0] == 1
    assert b.register().status_code == 200
    assert "id" in call(a, p.MESSENGER_SEND, to="bobby", text="welcome back")


@pytest.mark.parametrize("accept", [False, True])
def test_contact_add_requires_active_other(
    registry: Registry, clock: FakeClock, accept: bool
) -> None:
    a, b = named(registry, clock, "alice"), named(registry, clock, "bobby")
    if accept:
        call(b, p.MESSENGER_CONTACT, action="add", name="alice")
    assert b.call("POST", p.LEAVE).status_code == 200
    assert call(a, p.MESSENGER_CONTACT, action="add", name="bobby")["error"] == p.AWAY
    with store.connection(registry.path) as opened:
        assert opened.execute("SELECT count(*) FROM contacts").fetchone()[0] == 0
        assert opened.execute("SELECT count(*) FROM contact_events").fetchone()[0] == int(accept)


@pytest.mark.parametrize("command", ["global send", "global complain", "global report"])
@pytest.mark.parametrize("confirmed", [False, True])
def test_private_chat_command_journal_and_event(
    conn: sqlite3.Connection,
    joined: Config,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    confirmed: bool,
) -> None:
    from desman import global_cli, runs
    from desman.chat import Chat
    from desman.core.sessions import Kind, Role, Sessions
    from desman.interlocutor import Command, Interlocutor
    from desman.llm.fake import FakeLLM

    private = "private correspondence marker"
    fields: dict[str, Any] = {"text": private}
    if command != "global report":
        fields["name"] = "bobby"

    def perform(*args: Any, **kwargs: Any) -> p.MessageSent | p.ComplaintReceived:
        if command == "global complain":
            return p.ComplaintReceived(id="complaint")
        return p.MessageSent(id=1, sent_at=messenger._utc_now())

    monkeypatch.setattr(messenger_cli, "perform", perform)
    monkeypatch.setattr(
        global_cli, "send_report", lambda *a, **kw: p.ReportReceived(id="report", kind="bug")
    )
    monkeypatch.setattr(Chat, "_confirmed", lambda self, proposal: confirmed)
    sessions = Sessions(conn)
    chat = Chat(sessions)
    chat.session_id = sessions.add(None, Role.USER, "send it").session_id
    llm = FakeLLM()
    chat._command(
        Command(type="command", name=command, args=fields),
        joined,
        runs.Models(llm, None),
        Interlocutor(llm, "en"),
    )
    rows = conn.execute("SELECT action, details_json FROM operator_log").fetchall()
    assert len(rows) == 1
    assert rows[0][0] == ("operator_command" if confirmed else "operator_declined")
    assert private not in str([tuple(row) for row in rows])
    details = json.loads(rows[0][1])
    assert details["args_private"] == ["text"]
    assert "text" not in details["args"]
    events = [m for m in sessions.messages(chat.session_id) if m.kind is Kind.EVENT]
    assert events and all(private not in m.text for m in events)
