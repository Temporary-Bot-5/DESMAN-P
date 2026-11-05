"""One-shot signed messenger calls; no message can cause an action (G.10a)."""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TypeVar

import httpx
from pydantic import BaseModel

from desman.config import Config
from desman.core import db, netlog
from desman.net import node, seal, sign
from desman.net import protocol as p

M = TypeVar("M", bound=BaseModel)


class MessengerError(Exception):
    def __init__(self, error: str) -> None:
        self.error = error
        super().__init__(error)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def call(
    registry: str,
    key: sign.NodeKey,
    config: Config,
    conn: sqlite3.Connection,
    path: str,
    body: BaseModel,
    answer: type[M],
    *,
    transport: httpx.BaseTransport | None = None,
    clock: Callable[[], datetime] = _utc_now,
) -> M:
    """Register on the first not_registered only; never retry an uncertain send."""
    try:
        with httpx.Client(
            transport=transport, timeout=node.TIMEOUT, follow_redirects=False
        ) as client:

            def invoke(path: str, body: BaseModel, answer: type[M]) -> M:
                return node._call(
                    client,
                    key,
                    registry,
                    "POST",
                    path,
                    body,
                    answer,
                    clock(),
                    time.monotonic() + node.TIMEOUT,
                )

            try:
                return invoke(path, body, answer)
            except node._Problem as e:
                if e.error == "own_key_missing":
                    node._call(
                        client,
                        key,
                        registry,
                        "POST",
                        p.MESSENGER_KEY,
                        p.PublishKey(key=seal.publish(key)),
                        p.KeyPublished,
                        clock(),
                        time.monotonic() + node.TIMEOUT,
                    )
                    return invoke(path, body, answer)
                if e.error != node.NOT_REGISTERED:
                    raise
            lent = node.capabilities(config)
            node._call(
                client,
                key,
                registry,
                "POST",
                p.REGISTER,
                p.Register(capabilities=lent),
                p.NodeState,
                clock(),
                time.monotonic() + node.TIMEOUT,
            )
            netlog.record(
                conn,
                "registered",
                {
                    "registry": registry,
                    "node": key.fingerprint,
                    "capabilities": [c.model_dump(mode="json") for c in lent],
                },
            )
            return invoke(path, body, answer)
    except (node._Problem, node._Unreachable) as e:
        raise MessengerError(str(e)) from e


def set_name(
    registry: str,
    key: sign.NodeKey,
    config: Config,
    conn: sqlite3.Connection,
    body: p.SetName,
    *,
    transport: httpx.BaseTransport | None = None,
    clock: Callable[[], datetime] = _utc_now,
) -> p.MessengerName:
    publish_key(registry, key, config, conn, transport=transport, clock=clock)
    return call(
        registry,
        key,
        config,
        conn,
        p.MESSENGER_NAME,
        body,
        p.MessengerName,
        transport=transport,
        clock=clock,
    )


def contact(
    registry: str,
    key: sign.NodeKey,
    config: Config,
    conn: sqlite3.Connection,
    body: p.ContactAction,
    *,
    transport: httpx.BaseTransport | None = None,
    clock: Callable[[], datetime] = _utc_now,
) -> p.ContactState:
    if body.action == "add":
        publish_key(registry, key, config, conn, transport=transport, clock=clock)
    result = call(
        registry,
        key,
        config,
        conn,
        p.MESSENGER_CONTACT,
        body,
        p.ContactState,
        transport=transport,
        clock=clock,
    )

    if body.action == "add":
        try:
            peer(
                registry,
                key,
                config,
                conn,
                p.PeerQuery(name=body.name),
                transport=transport,
                clock=clock,
            )
        except MessengerError as e:
            # A legacy peer may not have published yet. A hidden block deliberately
            # looks like a request, without granting access to the other person's key.
            if e.error not in {"no_key", "not_contact"}:
                raise
    if body.action in {"remove", "block"}:
        with db.transaction(conn):
            conn.execute(
                "DELETE FROM messenger_peers WHERE registry = ? AND name = ?", (registry, body.name)
            )
    return result


def send(
    registry: str,
    key: sign.NodeKey,
    config: Config,
    conn: sqlite3.Connection,
    body: p.SendMessage,
    *,
    transport: httpx.BaseTransport | None = None,
    clock: Callable[[], datetime] = _utc_now,
) -> p.MessageSent:
    publish_key(registry, key, config, conn, transport=transport, clock=clock)
    return call(
        registry,
        key,
        config,
        conn,
        p.MESSENGER_SEND,
        body,
        p.MessageSent,
        transport=transport,
        clock=clock,
    )


def inbox(
    registry: str,
    key: sign.NodeKey,
    config: Config,
    conn: sqlite3.Connection,
    body: p.InboxQuery,
    *,
    transport: httpx.BaseTransport | None = None,
    clock: Callable[[], datetime] = _utc_now,
) -> p.Inbox:
    return call(
        registry,
        key,
        config,
        conn,
        p.MESSENGER_INBOX,
        body,
        p.Inbox,
        transport=transport,
        clock=clock,
    )


def complain(
    registry: str,
    key: sign.NodeKey,
    config: Config,
    conn: sqlite3.Connection,
    body: p.Complain,
    *,
    transport: httpx.BaseTransport | None = None,
    clock: Callable[[], datetime] = _utc_now,
) -> p.ComplaintReceived:
    return call(
        registry,
        key,
        config,
        conn,
        p.MESSENGER_COMPLAIN,
        body,
        p.ComplaintReceived,
        transport=transport,
        clock=clock,
    )


def publish_key(
    registry: str,
    key: sign.NodeKey,
    config: Config,
    conn: sqlite3.Connection,
    *,
    transport: httpx.BaseTransport | None = None,
    clock: Callable[[], datetime] = _utc_now,
    force: bool = False,
) -> None:
    if (
        not force
        and conn.execute(
            "SELECT 1 FROM messenger_published WHERE registry = ? AND node = ?",
            (registry, key.fingerprint),
        ).fetchone()
        is not None
    ):
        return
    call(
        registry,
        key,
        config,
        conn,
        p.MESSENGER_KEY,
        p.PublishKey(key=seal.publish(key)),
        p.KeyPublished,
        transport=transport,
        clock=clock,
    )
    with db.transaction(conn):
        conn.execute(
            "INSERT INTO messenger_published VALUES (?, ?, ?) ON CONFLICT(registry) DO UPDATE SET node=excluded.node,published_at=excluded.published_at",
            (registry, key.fingerprint, db.to_iso(clock())),
        )


def peer(
    registry: str,
    key: sign.NodeKey,
    config: Config,
    conn: sqlite3.Connection,
    body: p.PeerQuery,
    *,
    transport: httpx.BaseTransport | None = None,
    clock: Callable[[], datetime] = _utc_now,
) -> p.Peer:
    found = call(
        registry,
        key,
        config,
        conn,
        p.MESSENGER_PEER,
        body,
        p.Peer,
        transport=transport,
        clock=clock,
    )
    if found.name != body.name:
        raise MessengerError("bad_reply")
    pin(conn, registry, found, clock())
    return found


def pin(conn: sqlite3.Connection, registry: str, peer: p.Peer, now: datetime) -> None:
    try:
        seal.verify(peer.key, sign.decode_public(peer.node_public))
    except (seal.SealError, sign.KeyFileError):
        raise MessengerError("bad_peer") from None
    with db.transaction(conn):
        row = conn.execute(
            "SELECT node,x25519 FROM messenger_peers WHERE registry = ? AND name = ?",
            (registry, peer.name),
        ).fetchone()
        if row is not None and tuple(row) != (peer.key.node, peer.key.x25519):
            raise KeyChanged(peer.name)
        conn.execute(
            "INSERT OR IGNORE INTO messenger_peers VALUES (?, ?, ?, ?, ?)",
            (registry, peer.name, peer.key.node, peer.key.x25519, db.to_iso(now)),
        )


class KeyChanged(MessengerError):
    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__("key_changed")
