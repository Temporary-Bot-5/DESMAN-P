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
from desman.core import netlog
from desman.net import node, sign
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
    return call(
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
