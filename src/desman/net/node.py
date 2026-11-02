"""The node in the network (ADR 0004, decisions 2, 3 and 12; ROADMAP G.5).

While `desman worker` runs and desman-global is on, the node keeps in touch with the registry
between the worker's steps (`Node.poll`): it registers with what it lends, says it is alive
as often as the registry asks, and fetches the assignments and revocations it then checks
against the root (`sign.Trust`). Nothing else is sent yet; tasks from the network come with G.6.

- Off is off: the config is read again before every contact, and `net.registry()` has no
  address while desman-global is off, so a `desman global leave` while a worker runs stops it
  at the next contact, before anything is sent.
- The node's own work never waits for the network. A registry that cannot be reached (no
  connection, a timeout, 5xx, 429, a reply that is not what the protocol says) is tried again
  later, each time waiting twice as long (`RETRY_FIRST` up to `RETRY_MAX`), like a model's
  provider; a contact takes at most `TIMEOUT` seconds.
- A refusal by the registry is told by its word (`protocol.Problem.error`): `not_registered`
  (it forgot the node, or the node left elsewhere) registers again at once; `revoked` stops the
  node until the worker is restarted; anything else (a clock that is off, mostly) waits like a
  registry that cannot be reached.
- Everything is journaled in `network_log`: each registration and change of what is lent,
  each contact that failed and why, each refusal, each record received or refused (once per
  record and reason in a worker's life). A heartbeat that changes nothing is not: one every few
  minutes would bury the rest.

`tell_left` is `desman global leave`'s goodbye: sent once, while desman-global is still on.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, Final, TypeVar
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ValidationError

from desman import net, providers
from desman.config import Config, ConfigError
from desman.core import netlog
from desman.llm.openai_compat import key_id
from desman.net import protocol, request, sign

TIMEOUT: Final = 10.0
RETRY_FIRST: Final = timedelta(seconds=30)
RETRY_MAX: Final = timedelta(minutes=30)
OFF_RECHECK: Final = timedelta(minutes=1)
"""How soon a node that is off (or has no usable key) reads the config again."""
HEARTBEAT_MIN: Final = 60
HEARTBEAT_MAX: Final = 3600
"""The registry's `heartbeat_seconds`, kept within these (a registry cannot make the node
spin, nor fall silent for a day)."""

NOT_REGISTERED: Final = "not_registered"
REVOKED: Final = sign.Reason.REVOKED.value

M = TypeVar("M", bound=BaseModel)


def _utc_now() -> datetime:
    return datetime.now(UTC)


class Change(StrEnum):
    """What a poll changed, for the worker to tell the person."""

    CONNECTED = "connected"
    UNREACHABLE = "unreachable"
    REFUSED = "refused"
    REVOKED = "revoked"
    NO_KEY = "no_key"


@dataclass(frozen=True)
class Event:
    change: Change
    registry: str
    detail: str = ""
    retry_in: float | None = None
    """Seconds until the next try (UNREACHABLE, REFUSED)."""


class _Unreachable(Exception):
    """The registry did not answer as the protocol says: try again later."""


class _Problem(Exception):
    """The registry refused the request, with its word for why."""

    def __init__(self, error: str) -> None:
        super().__init__(error)
        self.error = error


def capabilities(config: Config) -> list[protocol.Capability]:
    """What this node lends, as the registry is told: provider, model, the fingerprint of the
    provider's first key (`key_id`, never the key; none for a local server), quotas. A
    provider no longer known or not set up is left out (the person sees why in `global
    status` and `keys check`)."""
    found: list[protocol.Capability] = []
    for provider_id, donation in sorted(config.network.donated.items()):
        try:
            provider = providers.find(provider_id)
        except providers.UnknownProvider:
            continue
        if provider.paid or not providers.can_donate(provider) or not provider.is_ready(config):
            continue
        keys = [] if provider.local or provider.key_env is None else config.keys(provider.key_env)
        if not provider.local and not keys:
            continue
        try:
            found.append(
                protocol.Capability(
                    provider=provider.id,
                    model=provider.model(config) or None,
                    key=key_id(keys[0].value) if keys else None,
                    calls_per_day=donation.calls_per_day,
                    tokens_per_day=donation.tokens_per_day,
                )
            )
        except ValidationError:
            continue  # a model name the protocol cannot carry: not lent
    return found[: protocol.MAX_CAPABILITIES]


def _target(registry: str, path: str) -> str:
    """The path signed and sent: the registry address's own path, then the endpoint's."""
    return urlsplit(registry).path.rstrip("/") + path


def _origin(registry: str) -> str:
    parts = urlsplit(registry)
    return f"{parts.scheme}://{parts.netloc}"


def _call(
    client: httpx.Client,
    key: sign.NodeKey,
    registry: str,
    method: str,
    path: str,
    body: BaseModel | None,
    answer: type[M],
    now: datetime,
) -> M:
    content = b"" if body is None else body.model_dump_json(exclude_none=True).encode("utf-8")
    target = _target(registry, path)
    headers = request.sign_request(key, method, target, content, now=now)
    if content:
        headers["Content-Type"] = "application/json"
    try:
        reply = client.request(method, _origin(registry) + target, content=content, headers=headers)
    except httpx.HTTPError as e:
        raise _Unreachable(type(e).__name__) from e
    if reply.status_code == 429 or reply.status_code >= 500:
        raise _Unreachable(f"HTTP {reply.status_code}")
    if reply.status_code != 200:
        try:
            problem = protocol.Problem.model_validate_json(reply.content)
        except ValidationError:
            raise _Unreachable(f"HTTP {reply.status_code}") from None
        raise _Problem(problem.error[:64])
    try:
        return answer.model_validate_json(reply.content)
    except ValidationError:
        raise _Unreachable("bad_reply") from None


def tell_left(
    registry: str,
    key: sign.NodeKey,
    *,
    transport: httpx.BaseTransport | None = None,
    clock: Callable[[], datetime] = _utc_now,
) -> str | None:
    """Tell the registry this node leaves; None when it heard, otherwise why not."""
    with httpx.Client(transport=transport, timeout=TIMEOUT, follow_redirects=False) as client:
        try:
            _call(client, key, registry, "POST", protocol.LEAVE, None, protocol.Left, clock())
        except _Unreachable as e:
            return str(e)
        except _Problem as e:
            # Never registered, or gone already: the registry does not list it either way.
            return None if e.error == NOT_REGISTERED else e.error
    return None


class Node:
    """This machine in desman-global, for one worker's life."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        config: Callable[[], Config],
        *,
        key_path: Path | None = None,
        transport: httpx.BaseTransport | None = None,
        clock: Callable[[], datetime] = _utc_now,
        root: Path | None = None,
    ) -> None:
        self.conn = conn
        self._config = config
        self._key_path = key_path
        self._transport = transport
        self._clock = clock
        self._root = root
        self._client: httpx.Client | None = None
        self.registry: str | None = None
        """The registry contacted last."""
        self.registered = False
        self.revoked = False
        """The registry refused this node's key as revoked: nothing more until a restart."""
        self.next_at: datetime | None = None
        """No contact before this (None: at the next poll)."""
        self.failures = 0
        self._sent: list[protocol.Capability] | None = None
        self._trust: sign.Trust | None = None
        self._journaled: set[tuple[str, str]] = set()
        self._no_key: str | None = None

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    # ── The loop ─────────────────────────────────────────────────────────────

    def poll(self) -> Event | None:
        """Contact the registry if it is time: what changed for the person, if anything."""
        now = self._clock()
        if self.revoked or (self.next_at is not None and now < self.next_at):
            return None
        try:
            return self._poll(now)
        except sqlite3.OperationalError:
            # The journal is busy (another worker holds the database too long): the worker's
            # own work matters more; look again later.
            self.next_at = now + RETRY_FIRST
            return None

    def _poll(self, now: datetime) -> Event | None:
        try:
            config = self._config()
        except ConfigError:
            self.next_at = now + OFF_RECHECK  # the person sees it in any command; send nothing
            return None
        address = net.registry(config)
        if address is None:
            self._forget()
            self.next_at = now + OFF_RECHECK
            return None
        key = self._key(address)
        if key is None:
            self.next_at = now + OFF_RECHECK
            return self._key_event(address)
        if address != self.registry:
            self._forget()
            self.registry = address
        try:
            return self._contact(config, address, key, now)
        except _Unreachable as e:
            return self._failed(address, str(e), now)
        except _Problem as e:
            return self._refused(address, key, e.error, now)

    def _forget(self) -> None:
        """Another registry, or off: register afresh next time."""
        self.registry = None
        self.registered = False
        self.failures = 0
        self._sent = None
        self._trust = None
        self.close()

    def _key(self, address: str) -> sign.NodeKey | None:
        path = sign.default_node_key_path() if self._key_path is None else self._key_path
        try:
            key = sign.load_node_key(path)
        except (sign.KeyFileError, OSError) as e:
            problem = str(e)
            if self._no_key != problem:
                self._no_key = problem
                netlog.record(self.conn, "node_key_unusable", {"registry": address})
            return None
        self._no_key = None
        return key

    def _key_event(self, address: str) -> Event | None:
        if self._no_key is None or ("no_key", self._no_key) in self._journaled:
            return None
        self._journaled.add(("no_key", self._no_key))
        return Event(Change.NO_KEY, address, self._no_key)

    def _http(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                transport=self._transport, timeout=TIMEOUT, follow_redirects=False
            )
        return self._client

    def _contact(
        self, config: Config, address: str, key: sign.NodeKey, now: datetime
    ) -> Event | None:
        lent = capabilities(config)
        client = self._http()
        first = not self.registered
        if first:
            state = _call(
                client,
                key,
                address,
                "POST",
                protocol.REGISTER,
                protocol.Register(capabilities=lent),
                protocol.NodeState,
                now,
            )
            self.registered = True
            self._sent = lent
            netlog.record(
                self.conn,
                "registered",
                {"registry": address, "node": key.fingerprint, "capabilities": _dump(lent)},
            )
        else:
            changed = lent != self._sent
            state = _call(
                client,
                key,
                address,
                "POST",
                protocol.HEARTBEAT,
                protocol.Heartbeat(capabilities=lent if changed else None),
                protocol.NodeState,
                now,
            )
            if changed:
                self._sent = lent
                netlog.record(
                    self.conn,
                    "capabilities_sent",
                    {"registry": address, "capabilities": _dump(lent)},
                )
        trust = _call(client, key, address, "GET", protocol.TRUST, None, protocol.TrustState, now)
        revoked = self._receive(address, key, trust)
        if revoked:
            return self._stop(address, key, "the trust list revokes this node")
        recovered = self.failures > 0
        self.failures = 0
        seconds = min(max(state.heartbeat_seconds, HEARTBEAT_MIN), HEARTBEAT_MAX)
        self.next_at = now + timedelta(seconds=seconds)
        if recovered:
            netlog.record(self.conn, "reachable", {"registry": address})
        return Event(Change.CONNECTED, address) if first or recovered else None

    def _receive(self, address: str, key: sign.NodeKey, state: protocol.TrustState) -> bool:
        """Check each record against the root; True if this node's own key is revoked."""
        if self._trust is None:
            root = sign.load_root() if self._root is None else sign.load_root(self._root)
            self._trust = sign.Trust(root, clock=self._clock)
        for signed in state.records:
            record = signed.record
            try:
                self._trust.receive(signed.model_dump(mode="json"))
            except sign.Refused as e:
                self._once(
                    (record.id, e.reason.value),
                    "record_refused",
                    {
                        "registry": address,
                        "kind": str(record.kind),
                        "id": record.id,
                        "signer": record.signer,
                        "reason": e.reason.value,
                    },
                )
                continue
            self._once(
                (record.id, "received"),
                "record_received",
                {
                    "registry": address,
                    "kind": str(record.kind),
                    "id": record.id,
                    "signer": record.signer,
                },
            )
        return self._trust.is_revoked(key.fingerprint)

    def _once(self, mark: tuple[str, str], action: str, details: dict[str, Any]) -> None:
        if mark in self._journaled:
            return
        self._journaled.add(mark)
        netlog.record(self.conn, action, details)

    def _retry(self, now: datetime) -> float:
        self.failures += 1
        wait: timedelta = min(RETRY_FIRST * (2 ** min(self.failures - 1, 16)), RETRY_MAX)
        self.next_at = now + wait
        return wait.total_seconds()

    def _failed(self, address: str, error: str, now: datetime) -> Event | None:
        retry_in = self._retry(now)
        netlog.record(
            self.conn,
            "unreachable",
            {"registry": address, "error": error, "attempt": self.failures, "retry_in": retry_in},
        )
        if self.failures > 1:
            return None
        return Event(Change.UNREACHABLE, address, error, retry_in)

    def _refused(self, address: str, key: sign.NodeKey, error: str, now: datetime) -> Event | None:
        if error == REVOKED:
            return self._stop(address, key, "the registry refuses this node's key as revoked")
        if error == NOT_REGISTERED and self.registered:
            # Forgotten by the registry (or left from elsewhere): register again at once.
            self.registered = False
            self._sent = None
            self.next_at = None
            netlog.record(self.conn, "refused_by_registry", {"registry": address, "error": error})
            return None
        retry_in = self._retry(now)
        netlog.record(
            self.conn,
            "refused_by_registry",
            {"registry": address, "error": error, "attempt": self.failures, "retry_in": retry_in},
        )
        if self.failures > 1:
            return None
        return Event(Change.REFUSED, address, error, retry_in)

    def _stop(self, address: str, key: sign.NodeKey, why: str) -> Event:
        self.revoked = True
        self.registered = False
        self.close()
        netlog.record(
            self.conn, "revoked", {"registry": address, "node": key.fingerprint, "why": why}
        )
        return Event(Change.REVOKED, address)


def _dump(lent: list[protocol.Capability]) -> list[dict[str, Any]]:
    return [json.loads(c.model_dump_json()) for c in lent]
