"""Encrypted wire, selective complaint disclosure, durable pins and upgrades."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import httpx
import pytest
from click import ClickException

from conftest import FakeClock
from desman import global_cli, messenger_cli
from desman.config import Config, ConfigManager, Network
from desman.core import db
from desman.core.sessions import Role, Sessions
from desman.net import protocol as p
from desman.net import seal, sign
from desman.registry import store
from desman.registry.server import Registry
from test_messenger import connect, named, registry
from test_registry import Node

__all__ = ["registry"]
ADDRESS = "https://registry.test"


def envelope(a: Node, b: Node, text: str = "secret message") -> seal.Sealed:
    return seal.seal(a.key, b.fingerprint, seal.publish(b.key).x25519, text, a.clock())


def send(a: Node, b: Node, to: str = "bobby", text: str = "secret message") -> dict[str, Any]:
    return dict(
        a.call(
            "POST",
            p.MESSENGER_SEND,
            {"to": to, "sealed": envelope(a, b, text).model_dump(mode="json")},
        ).json()
    )


def test_publish_and_peer(registry: Registry, clock: FakeClock) -> None:
    a, b, stranger = (
        named(registry, clock, "alice"),
        named(registry, clock, "bobby"),
        named(registry, clock, "carol"),
    )
    published = seal.publish(a.key)
    for _ in range(2):
        assert (
            a.call("POST", p.MESSENGER_KEY, {"key": published.model_dump(mode="json")}).json() == {}
        )
    forged = published.model_copy(update={"signature": seal.encode(bytes(64))})
    assert (
        a.call("POST", p.MESSENGER_KEY, {"key": forged.model_dump(mode="json")}).json()["error"]
        == "bad_key"
    )
    # A genuinely signed alternate X25519 key must not replace the first one.
    alternate = seal.publish(sign.NodeKey.generate()).x25519
    body = {"kind": "messenger_key", "node": a.fingerprint, "x25519": alternate}
    changed = seal.MessengerKey(
        node=a.fingerprint,
        x25519=alternate,
        signature=seal.encode(a.key.sign(sign.canonical(body))),
    )
    assert (
        a.call("POST", p.MESSENGER_KEY, {"key": changed.model_dump(mode="json")}).json()["error"]
        == "key_mismatch"
    )
    assert (
        stranger.call("POST", p.MESSENGER_PEER, {"name": "alice"}).json()["error"] == "not_contact"
    )
    a.call("POST", p.MESSENGER_CONTACT, {"action": "add", "name": "bobby"})
    assert a.call("POST", p.MESSENGER_PEER, {"name": "bobby"}).json()["key"] == seal.publish(
        b.key
    ).model_dump(mode="json")
    assert b.call("POST", p.MESSENGER_PEER, {"name": "alice"}).status_code == 200
    with store.connection(registry.path) as conn:
        assert (
            conn.execute(
                "SELECT count(*) FROM registry_log WHERE action='messenger_key' AND node=?",
                (a.fingerprint,),
            ).fetchone()[0]
            == 1
        )
        with pytest.raises(sqlite3.IntegrityError, match="fixed"):
            conn.execute(
                "UPDATE messenger_keys SET x25519=? WHERE node=?", (alternate, a.fingerprint)
            )


def test_wire_and_missing_keys(registry: Registry, clock: FakeClock) -> None:
    a, b = named(registry, clock, "alice"), named(registry, clock, "bobby")
    connect(a, b)
    assert (
        a.call("POST", p.MESSENGER_SEND, {"to": "bobby", "text": "plaintext"}).json()["error"]
        == "malformed"
    )
    with store.connection(registry.path) as conn:
        conn.execute("DELETE FROM messenger_keys WHERE node=?", (b.fingerprint,))
    assert send(a, b)["error"] == "no_key"
    assert a.call("POST", p.MESSENGER_PEER, {"name": "bobby"}).json()["error"] == "no_key"
    with store.connection(registry.path) as conn:
        conn.execute("DELETE FROM messenger_keys WHERE node=?", (a.fingerprint,))
    assert send(a, b)["error"] == "own_key_missing"


def test_storage_and_complaint_evidence(registry: Registry, clock: FakeClock) -> None:
    a, b = named(registry, clock, "alice"), named(registry, clock, "bobby")
    connect(a, b)
    secret = "unique secret never stored by the registry without disclosure"
    sealed = envelope(a, b, secret)
    sent = a.call(
        "POST", p.MESSENGER_SEND, {"to": "bobby", "sealed": sealed.model_dump(mode="json")}
    ).json()
    raw = b.call("POST", p.MESSENGER_INBOX, {}).json()
    assert {peer["name"] for peer in raw["peers"]} == {"alice", "bobby"}
    assert raw["messages"][0]["text"] is None
    with store.connection(registry.path) as conn:
        row = conn.execute("SELECT text,sealed FROM messages").fetchone()
        assert row[0] == "" and secret not in row[1]
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    assert secret.encode() not in registry.path.read_bytes()
    complaint = {"name": "alice", "text": "please review", "message": sent["id"]}
    for invalid_content in (None, seal.encode(bytes(32))):
        assert (
            b.call(
                "POST", p.MESSENGER_COMPLAIN, complaint | {"content_key": invalid_content}
            ).json()["error"]
            == "bad_evidence"
        )
    content = seal.content_key(b.key, sealed)
    accepted = b.call(
        "POST", p.MESSENGER_COMPLAIN, complaint | {"content_key": seal.encode(content)}
    ).json()
    with store.connection(registry.path) as conn:
        row = conn.execute("SELECT * FROM complaints WHERE id=?", (accepted["id"],)).fetchone()
        assert row["message_text"] == secret
        assert seal.encode(content) not in str(tuple(row))
    # Re-encrypt a forged letter under the correct disclosed key. A valid AEAD tag is not evidence of authorship.
    from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

    signed = seal.open_with(content, sealed)
    forged = signed.model_copy(update={"signature": seal.encode(bytes(64))})
    raw_cipher = ChaCha20Poly1305(content).encrypt(
        seal.decode(sealed.nonce),
        sign.canonical(forged.model_dump(mode="json")),
        b"desman message v1",
    )
    broken = sealed.model_copy(update={"ciphertext": seal.encode(raw_cipher)})
    new = a.call(
        "POST", p.MESSENGER_SEND, {"to": "bobby", "sealed": broken.model_dump(mode="json")}
    ).json()
    assert (
        b.call(
            "POST",
            p.MESSENGER_COMPLAIN,
            complaint | {"message": new["id"], "content_key": seal.encode(content)},
        ).json()["error"]
        == "bad_evidence"
    )


def test_legacy_upgrade(tmp_path: Path, clock: FakeClock) -> None:
    path = tmp_path / "legacy.db"
    conn = db.connect(path)
    db.migrate(conn, store.MIGRATIONS[:11])
    a, b = sign.NodeKey.generate(), sign.NodeKey.generate()
    for key, name in ((a, "alice"), (b, "bobby")):
        conn.execute(
            "INSERT INTO nodes(fingerprint,public_key,status,registered_at,last_seen) VALUES (?,?,'active',?,?)",
            (
                key.fingerprint,
                sign.encode_public(key.public),
                db.to_iso(clock()),
                db.to_iso(clock()),
            ),
        )
        conn.execute(
            "INSERT INTO messenger_names VALUES (?,?,?)",
            (key.fingerprint, name, db.to_iso(clock())),
        )
    conn.execute(
        "INSERT INTO messages(sender,recipient,text,sent_at) VALUES (?,?,?,?)",
        (a.fingerprint, b.fingerprint, "legacy plaintext", db.to_iso(clock())),
    )
    conn.close()
    with store.open_registry(path) as conn:
        assert db.schema_version(conn) == 12
        assert conn.execute("SELECT text,sealed FROM messages").fetchone()["sealed"] is None
    registry = Registry(path, clock=clock)
    node = Node(registry, clock, b)
    result = node.call("POST", p.MESSENGER_INBOX, {}).json()
    assert result["messages"][0]["text"] == "legacy plaintext"
    assert result["messages"][0]["sealed"] is None
    assert (
        node.call(
            "POST", p.MESSENGER_COMPLAIN, {"name": "alice", "message": 1, "text": "legacy evidence"}
        ).status_code
        == 200
    )
    with store.connection(path) as conn:
        assert (
            conn.execute("SELECT message_text FROM complaints").fetchone()[0] == "legacy plaintext"
        )


def test_node_upgrade(tmp_path: Path, clock: FakeClock) -> None:
    path = tmp_path / "node.db"
    conn = db.connect(path)
    db.migrate(conn, db.MIGRATIONS[:32])
    session = Sessions(conn).add(None, Role.USER, "kept across upgrade").session_id
    conn.close()
    with db.open_db(path) as conn:
        assert db.schema_version(conn) == db.SCHEMA_VERSION
        assert Sessions(conn).messages(session)[0].text == "kept across upgrade"
        conn.execute(
            "INSERT INTO messenger_peers VALUES (?,?,?,?,?)",
            (ADDRESS, "alice", "a" * 64, seal.encode(bytes(32)), db.to_iso(clock())),
        )
        conn.execute(
            "INSERT INTO messenger_published VALUES (?,?,?)",
            (ADDRESS, "a" * 64, db.to_iso(clock())),
        )


def test_cli_bodies_two_nodes_and_tofu(
    registry: Registry, tmp_path: Path, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    nodes = [Node(registry, clock), Node(registry, clock)]
    conns = [db.open_db(":memory:"), db.open_db(":memory:")]
    managers = [ConfigManager(tmp_path / f"config{i}.json") for i in range(2)]
    for manager in managers:
        manager.save(Config(language="en", network=Network(enabled=True, registry=ADDRESS)))
    paths: list[str] = []

    def run(index: int, body: messenger_cli.Request) -> messenger_cli.Response:
        node = nodes[index]
        monkeypatch.setattr(global_cli, "usable_node_key", lambda *a, **kw: node.key)

        def respond(req: httpx.Request) -> httpx.Response:
            paths.append(req.url.path)
            return node._handle(req)

        return messenger_cli.perform(
            managers[index], conns[index], body, transport=httpx.MockTransport(respond), clock=clock
        )

    try:
        for i, name in enumerate(("alice", "bobby")):
            run(i, p.SetName(name=name))
        publications = paths.count(p.MESSENGER_KEY)
        run(0, p.ContactAction(name="bobby", action="add"))
        run(1, p.ContactAction(name="alice", action="add"))
        assert (
            conns[0].execute("SELECT node FROM messenger_peers WHERE name='bobby'").fetchone()[0]
            == nodes[1].fingerprint
        )
        assert paths.count(p.MESSENGER_KEY) == publications
        run(0, messenger_cli.MessageDraft(to="bobby", text="first secret"))
        run(1, messenger_cli.MessageDraft(to="alice", text="second secret"))
        for i in (0, 1):
            inbox = run(i, p.InboxQuery())
            assert isinstance(inbox, messenger_cli.ReadInbox)
            assert [m.text for m in inbox.messages] == ["first secret", "second secret"]
            assert all(m.encrypted and m.verified for m in inbox.messages)
            assert inbox.fingerprints == {
                "alice": nodes[0].fingerprint,
                "bobby": nodes[1].fingerprint,
            }
            dumped = json.dumps(messenger_cli.data(inbox))
            assert not any(s in dumped for s in ("ciphertext", "x25519", '"sealed"', '"key"'))
        complaint = run(1, p.Complain(name="alice", text="review first", message=1))
        assert isinstance(complaint, p.ComplaintReceived)
        with store.connection(registry.path) as conn:
            assert (
                conn.execute(
                    "SELECT message_text FROM complaints WHERE id=?", (complaint.id,)
                ).fetchone()[0]
                == "first secret"
            )
        # Lost registry publication, but local receipt persists: exactly one republish/retry.
        with store.connection(registry.path) as conn:
            conn.execute("DELETE FROM messenger_keys WHERE node=?", (nodes[0].fingerprint,))
        paths.clear()
        run(0, messenger_cli.MessageDraft(to="bobby", text="recovered publication"))
        assert paths == [p.MESSENGER_PEER, p.MESSENGER_SEND, p.MESSENGER_KEY, p.MESSENGER_SEND]
        # A new owner acquires bobby after the old identity renames itself.
        nodes[1].call("POST", p.MESSENGER_NAME, {"name": "bobby_old"})
        replacement = named(registry, clock, "bobby")
        nodes[0].call("POST", p.MESSENGER_CONTACT, {"name": "bobby", "action": "add"})
        replacement.call("POST", p.MESSENGER_CONTACT, {"name": "alice", "action": "add"})
        with pytest.raises(ClickException, match="key of bobby changed"):
            run(0, messenger_cli.MessageDraft(to="bobby", text="cancelled"))
        run(0, p.ContactAction(name="bobby", action="remove"))
        assert (
            conns[0].execute("SELECT 1 FROM messenger_peers WHERE name='bobby'").fetchone() is None
        )
        run(0, p.ContactAction(name="bobby", action="add"))
        replacement.call("POST", p.MESSENGER_CONTACT, {"name": "alice", "action": "add"})
        run(0, messenger_cli.MessageDraft(to="bobby", text="accepted replacement"))
        run(0, p.ContactAction(name="bobby", action="block"))
        assert (
            conns[0].execute("SELECT 1 FROM messenger_peers WHERE name='bobby'").fetchone() is None
        )
    finally:
        for conn in conns:
            conn.close()


def test_read_failures_are_placeholders(registry: Registry, clock: FakeClock) -> None:
    a, b = named(registry, clock, "alice"), named(registry, clock, "bobby")
    connect(a, b)
    sent = send(a, b)
    inbox = p.Inbox.model_validate(b.call("POST", p.MESSENGER_INBOX, {}).json())
    with db.open_db(":memory:") as conn:
        good = messenger_cli.read_inbox(inbox, b.key, ADDRESS, conn, clock())
        assert good.messages[0].verified
        # A pinned name now has a different, validly self-signed identity.
        alternate = sign.NodeKey.generate()
        changed_peer = p.Peer(
            name="alice",
            node_public=sign.encode_public(alternate.public),
            key=seal.publish(alternate),
        )
        changed = inbox.model_copy(
            update={
                "peers": [changed_peer if peer.name == "alice" else peer for peer in inbox.peers]
            }
        )
        shown = messenger_cli.read_inbox(changed, b.key, ADDRESS, conn, clock())
        assert shown.messages[0].text == "Key changed, not shown."
        assert not shown.messages[0].verified
        ciphertext = inbox.messages[0].sealed
        assert ciphertext is not None
        raw = seal.decode(ciphertext.ciphertext)
        broken = ciphertext.model_copy(
            update={"ciphertext": seal.encode(bytes([raw[0] ^ 1]) + raw[1:])}
        )
        invalid = inbox.model_copy(
            update={"messages": [inbox.messages[0].model_copy(update={"sealed": broken})]}
        )
        shown = messenger_cli.read_inbox(invalid, b.key, ADDRESS, conn, clock())
        assert shown.messages[0].text == "The message could not be decrypted or verified."
        assert not shown.messages[0].verified
        assert "secret message" not in json.dumps(messenger_cli.data(shown))
        assert shown.messages[0].id == sent["id"]
