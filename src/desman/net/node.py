"""The node in the network (ADR 0004, decisions 2, 3 and 12; ROADMAP G.5).

While `desman worker` runs and desman-global is on, the node keeps in touch with the registry
between the worker's steps (`Node.poll`): it registers with what it lends, says it is alive
as often as the registry asks, and fetches the assignments and revocations it then checks
against the root (`sign.Trust`).

Work (G.6b, `net.jobs`): a node whose worker can do it (`work`: models lent, a crew for them)
asks for a job only while the worker is idle and no job is held, and while some lent module has
quota left today; an empty offer is asked again after `CLAIM_EVERY`. A job lent is checked first
(the operator's record by `sign.Trust`, never twice; a recheck's result by the signature of the
node that sent it) and becomes a task of this node; a record that fails is journaled and left
alone (its lease runs out at the registry). While the job is worked on, its lease is extended
once less than `EXTEND_WHEN` is left (the worker polls after each model call too, so one long
step does not outlast the lease); once it ends, the signed result or verdict is handed in,
and sent again until the registry answers. A refusal of the lease (`lease_lost`: it ran out, or
another node has the job now) drops the job. Turning desman-global off, or another registry,
drops what is held without a word to anyone (off is off). No job is asked for while the trust
list is not fresh (G.6c, `sign.Trust.fresh`): a registry could be hiding a revocation; a job
already held is still finished.

- Off is off: the config is read again before every contact, and `net.registry()` has no
  address while desman-global is off, so a `desman global leave` while a worker runs stops it
  at the next contact, before anything is sent.
- The node's own work never waits for the network. A registry that cannot be reached (no
  connection, a timeout, 5xx, 429, a reply that is not what the protocol says) is tried again
  later, each time waiting twice as long (`RETRY_FIRST` up to `RETRY_MAX`), like a model's
  provider. A contact (its two requests together) gets `TIMEOUT` seconds: the second request
  only what the first left, and none at all once they are spent; a reply still arriving when
  they are spent is given up, however steadily it trickles in.
- A refusal by the registry is told by its word (`protocol.Problem.error`): `not_registered`
  (it forgot the node, or the node left elsewhere) registers again at once; `revoked` stops the
  node until the worker is restarted; anything else (a clock that is off, mostly) waits like a
  registry that cannot be reached.
- Everything is journaled in `network_log`: each registration and change of what is lent,
  each contact that failed and why, each refusal, each record received or refused (once per
  record and reason in a worker's life). A heartbeat that changes nothing is not: one every few
  minutes would bury the rest.

G.7b checks proposals synchronously while idle, using only the fake benchmark; signed
checks are retried durably before claiming more work.

`tell_left` is `desman global leave`'s goodbye: sent once, after the config was saved off.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, Final, TypeVar
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ValidationError

from desman import evaluation, net, providers
from desman.config import Config, ConfigError
from desman.core import db, holds, netlog
from desman.core.store import Store
from desman.core.workspace import workspace_root_for
from desman.llm.base import LLM, ProviderError
from desman.llm.openai_compat import key_id
from desman.net import checks, cluster, jobs, protocol, request, sign, updates

TIMEOUT: Final = 10.0
RETRY_FIRST: Final = timedelta(seconds=30)
RETRY_MAX: Final = timedelta(minutes=30)
OFF_RECHECK: Final = timedelta(minutes=1)
"""How soon a node that is off (or has no usable key) reads the config again."""
CLAIM_EVERY: Final = timedelta(minutes=1)
"""How soon an idle node asks for work again after none was offered (or none could be taken)."""
EXTEND_WHEN: Final = timedelta(minutes=5)
"""A held job's lease is extended once less than this is left."""
HEARTBEAT_MIN: Final = 60
HEARTBEAT_MAX: Final = 3600
"""The registry's `heartbeat_seconds`, kept within these (a registry cannot make the node
spin, nor fall silent for a day)."""

NOT_REGISTERED: Final = "not_registered"
LEASE_LOST: Final = "lease_lost"
REVOKED: Final = sign.Reason.REVOKED.value

M = TypeVar("M", bound=BaseModel)


def _utc_now() -> datetime:
    return datetime.now(UTC)


class Change(StrEnum):
    """What a poll changed, for the worker to tell the person."""

    RELEASE = "release"
    CONNECTED = "connected"
    UNREACHABLE = "unreachable"
    REFUSED = "refused"
    REVOKED = "revoked"
    NO_KEY = "no_key"
    WORK_TAKEN = "work_taken"
    WORK_HANDED_IN = "work_handed_in"
    WORK_DROPPED = "work_dropped"
    CHECK_HANDED_IN = "check_handed_in"


@dataclass(frozen=True)
class Event:
    change: Change
    registry: str
    detail: str = ""
    """The error (UNREACHABLE, REFUSED, NO_KEY); the stage (WORK_TAKEN); the outcome
    (WORK_HANDED_IN); why (WORK_DROPPED)."""
    retry_in: float | None = None
    """Seconds until the next try (UNREACHABLE, REFUSED)."""


class _Unreachable(Exception):
    """The registry did not answer as the protocol says: try again later."""


class _Problem(Exception):
    """The registry refused the request, with its word for why."""

    def __init__(self, error: str) -> None:
        super().__init__(error)
        self.error = error


class ProposalSendError(Exception):
    """A proposal could not be sent; the message contains no request or key material."""


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


def active_capabilities(config: Config, conn: sqlite3.Connection) -> list[protocol.Capability]:
    return [
        c
        for c in capabilities(config)
        if not holds.blocked(conn, c.key or c.provider, f"{c.provider}/{c.model}", "network")
    ]


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
    deadline: float,
) -> M:
    """One request, given what is left until `deadline` (`time.monotonic()`)."""
    left = deadline - time.monotonic()
    if left <= 0:
        raise _Unreachable("ContactTimeout")
    content = b"" if body is None else body.model_dump_json(exclude_none=True).encode("utf-8")
    target = _target(registry, path)
    headers = request.sign_request(key, method, target, content, now=now)
    if content:
        headers["Content-Type"] = "application/json"
    built = client.build_request(
        method,
        _origin(registry) + target,
        content=content,
        headers=headers,
        timeout=httpx.Timeout(left),
    )
    try:
        reply = client.send(built, stream=True)
        try:
            # httpx times each read, not the whole reply: a registry that dribbles it out
            # would keep the worker waiting, so the deadline is checked after every piece,
            # and once more at the end (a reply's ending can come late, after its last
            # byte). One read may still outlast the deadline, by its own timeout at most.
            received = bytearray()
            for piece in reply.iter_bytes():
                received += piece
                if time.monotonic() >= deadline:
                    raise _Unreachable("ContactTimeout")
            if time.monotonic() >= deadline:
                raise _Unreachable("ContactTimeout")
        finally:
            reply.close()
    except httpx.HTTPError as e:
        raise _Unreachable(type(e).__name__) from e
    body_bytes = bytes(received)
    if reply.status_code == 429 or reply.status_code >= 500:
        raise _Unreachable(f"HTTP {reply.status_code}")
    if reply.status_code == 404 and path == protocol.RELEASES:
        raise _Problem("not_found")
    if reply.status_code != 200:
        try:
            problem = protocol.Problem.model_validate_json(body_bytes)
        except ValidationError:
            raise _Unreachable(f"HTTP {reply.status_code}") from None
        raise _Problem(problem.error[:64])
    try:
        return answer.model_validate_json(body_bytes)
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
            _call(
                client,
                key,
                registry,
                "POST",
                protocol.LEAVE,
                None,
                protocol.Left,
                clock(),
                time.monotonic() + TIMEOUT,
            )
        except _Unreachable as e:
            return str(e)
        except _Problem as e:
            # Never registered, or gone already: the registry does not list it either way.
            return None if e.error == NOT_REGISTERED else e.error
    return None


def submit_proposal(
    registry: str,
    key: sign.NodeKey,
    signed: protocol.SignedProposalSubmission,
    *,
    transport: httpx.BaseTransport | None = None,
    clock: Callable[[], datetime] = _utc_now,
) -> protocol.ProposalReceived:
    """Send the exact proposal body the person approved, signed as a registry request."""
    body = protocol.SubmitProposal(signed=signed)
    if len(body.model_dump_json(exclude_none=True).encode("utf-8")) > protocol.MAX_PROPOSAL_BYTES:
        raise _Problem("proposal_too_large")
    try:
        with httpx.Client(transport=transport, timeout=TIMEOUT, follow_redirects=False) as client:
            return _call(
                client,
                key,
                registry,
                "POST",
                protocol.PROPOSALS,
                body,
                protocol.ProposalReceived,
                clock(),
                time.monotonic() + TIMEOUT,
            )
    except (_Unreachable, _Problem) as e:
        raise ProposalSendError(str(e)) from e


class ReportSendError(Exception):
    """The registry did not accept a report."""


def submit_report(
    registry: str,
    key: sign.NodeKey,
    signed: protocol.SignedReport,
    *,
    transport: httpx.BaseTransport | None = None,
    clock: Callable[[], datetime] = _utc_now,
) -> protocol.ReportReceived:
    """Send the exact report approved by the person."""
    body = protocol.SubmitReport(signed=signed)
    if len(body.model_dump_json(exclude_none=True).encode("utf-8")) > protocol.MAX_REPORT_BYTES:
        raise ReportSendError("report_too_large")
    try:
        with httpx.Client(transport=transport, timeout=TIMEOUT, follow_redirects=False) as client:
            return _call(
                client,
                key,
                registry,
                "POST",
                protocol.REPORTS,
                body,
                protocol.ReportReceived,
                clock(),
                time.monotonic() + TIMEOUT,
            )
    except (_Unreachable, _Problem) as e:
        raise ReportSendError(str(e)) from e


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
        work: bool = False,
        cluster_model: LLM | None = None,
        workspaces: Path | None = None,
    ) -> None:
        self.conn = conn
        self.cluster_model = cluster_model
        self.cluster_at: datetime | None = None
        self.work = work
        """This worker can do jobs (a crew for the models lent): it asks for them."""
        self._store = Store(conn, clock)
        self._seen = jobs.Seen(conn)
        self._workspaces = workspace_root_for(conn) if workspaces is None else workspaces
        self.claim_at: datetime | None = None
        """No asking for work before this."""
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
        self._appointments: dict[str, sign.Signed] = {}
        self._stale: str | None = None
        """Why the trust list was not fresh when last asked (G.6c); None: it was."""
        self._journaled: set[tuple[str, str]] = set()
        self._no_key: str | None = None
        self._pending_release: Event | None = None
        self.suspended = False

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    # ── The loop ─────────────────────────────────────────────────────────────

    def poll(self, *, idle: bool = False) -> Event | None:
        """Contact the registry if it is time, and tend the work held (asking for some when the
        worker is `idle`): what changed for the person, if anything."""
        now = self._clock()
        if self.revoked:
            return None
        try:
            config = self._config()
            current_role = cluster.role(self.conn, now)
            if not config.network.lead and current_role and current_role["role"] == "operator":
                cluster.left(self.conn, now)
            suspended = config.network.enabled and not active_capabilities(config, self.conn)
            if config.network.enabled:
                with db.transaction(self.conn):
                    previous = self.conn.execute(
                        "SELECT action FROM network_log WHERE action IN ('suspended','resumed') ORDER BY seq DESC LIMIT 1"
                    ).fetchone()
                    was_suspended = previous is not None and previous[0] == "suspended"
                    if suspended != was_suspended:
                        netlog.write(
                            self.conn,
                            "suspended" if suspended else "resumed",
                            {"reason": "no module" if suspended else "module available"},
                        )
            if suspended != self.suspended:
                self.suspended = suspended
                self.next_at = None
            if suspended:
                for job in jobs.holding(self.conn):
                    self._drop(job, "abandoned", "no module left")
                for check in checks.pending(self.conn):
                    checks.close(self.conn, check.lease, "lost")
                return None
            event = None
            if self.next_at is None or now >= self.next_at:
                event = self._poll(now)
            if event is not None and event.change is Change.RELEASE:
                self._pending_release = event
                event = None
            if not self.revoked:
                event = self._jobs(now, idle) or event
            if event is None and not self.revoked:
                event = self._pending_release
                self._pending_release = None
            return event
        except ConfigError:
            return None
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
        self._stale = None
        self._pending_release = None
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
        lent = active_capabilities(config, self.conn)
        client = self._http()
        deadline = time.monotonic() + TIMEOUT
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
                deadline,
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
                deadline,
            )
            if changed:
                self._sent = lent
                netlog.record(
                    self.conn,
                    "capabilities_sent",
                    {"registry": address, "capabilities": _dump(lent)},
                )
        trust = _call(
            client, key, address, "GET", protocol.TRUST, None, protocol.TrustState, now, deadline
        )
        revoked = self._receive(address, key, trust)
        if revoked:
            return self._stop(address, key, "the trust list revokes this node")
        release_event = self._releases(client, key, address, now, deadline)
        recovered = self.failures > 0
        self.failures = 0
        seconds = min(max(state.heartbeat_seconds, HEARTBEAT_MIN), HEARTBEAT_MAX)
        self.next_at = now + timedelta(seconds=seconds)
        if recovered:
            netlog.record(self.conn, "reachable", {"registry": address})
        return release_event or (Event(Change.CONNECTED, address) if first or recovered else None)

    def _releases(
        self, client: httpx.Client, key: sign.NodeKey, address: str, now: datetime, deadline: float
    ) -> Event | None:
        assert self._trust is not None
        newest = self.conn.execute("SELECT MAX(number) FROM releases").fetchone()[0] or 0
        event = None
        try:
            received = _call(
                client,
                key,
                address,
                "GET",
                protocol.RELEASES,
                None,
                protocol.Releases,
                now,
                deadline,
            )
        except _Problem as e:
            self._release_unavailable(address, e.error)
            return None
        for signed in received.records:
            try:
                if updates.keep(self.conn, self._trust, signed):
                    number = sign.Release.model_validate(signed.record.body).number
                    if number > newest:
                        newest = number
                        event = Event(Change.RELEASE, address, str(number))
            except sign.Refused as e:
                self._release_unavailable(address, str(e))
        return event

    def _release_unavailable(self, address: str, why: str) -> None:
        self._once(
            ("releases_unavailable", why), "releases_unavailable", {"registry": address, "why": why}
        )

    def _receive(self, address: str, key: sign.NodeKey, state: protocol.TrustState) -> bool:
        """Check each record against the root; True if this node's own key is revoked."""
        if self._trust is None:
            root = sign.load_root() if self._root is None else sign.load_root(self._root)
            self._trust = sign.Trust(root, clock=self._clock, seen=self._seen)
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
            if record.kind is sign.RecordKind.APPOINTMENT:
                self._appointments[record.id] = signed
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
        latest = {
            signed.record.id
            for signed in state.records
            if signed.record.kind is sign.RecordKind.APPOINTMENT
        }
        self._appointments = {
            id: signed for id, signed in self._appointments.items() if id in latest
        }
        for signed in self._appointments.values():
            body = sign.Appointment.model_validate(signed.record.body)
            current = self._trust.appointment(body.cluster)
            if current is not None and key.fingerprint in current.members:
                cluster.joined(self.conn, current, "member", self._clock(), from_trust=True)
        return self._trust.is_revoked(key.fingerprint)

    def _once(self, mark: tuple[str, str], action: str, details: dict[str, Any]) -> None:
        if mark in self._journaled:
            return
        netlog.record(self.conn, action, details)  # a busy journal: tried again next poll
        self._journaled.add(mark)

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
        for check in checks.pending(self.conn):
            if check.registry == address:
                checks.close(self.conn, check.lease, "lost")
                netlog.record(
                    self.conn,
                    "check_lost",
                    {
                        "registry": address,
                        "proposal": check.proposal,
                        "reason": REVOKED,
                    },
                )
        netlog.record(
            self.conn, "revoked", {"registry": address, "node": key.fingerprint, "why": why}
        )
        return Event(Change.REVOKED, address)

    # ── Work (G.6b) ──────────────────────────────────────────────────────────

    def _jobs(self, now: datetime, idle: bool) -> Event | None:
        self._clusters(now, idle)
        held = jobs.holding(self.conn)
        waiting = checks.pending(self.conn)
        if (
            not held
            and not waiting
            and not (idle and (self.claim_at is None or now >= self.claim_at))
        ):
            return None
        try:
            config = self._config()
        except ConfigError:
            return None
        address = net.registry(config)
        for job in held:
            if job.registry != address:  # off, or another registry: nothing is sent
                jobs.close(self._store, job, "abandoned", "desman-global is off or moved")
                netlog.record(self.conn, "work_dropped", _job_details(job, "off"))
        held = [job for job in held if job.registry == address]
        for check in waiting:
            if check.registry != address:
                checks.close(self.conn, check.lease, "lost")
        waiting = [check for check in waiting if check.registry == address]
        if (
            address is None
            or address != self.registry
            or not self.registered
            or self._trust is None
            or self.failures > 0  # the contact's backoff holds for work too
        ):
            return None
        key = self._key(address)
        if key is None:
            return None
        client = self._http()
        deadline = time.monotonic() + TIMEOUT
        try:
            event = None
            for check in waiting:
                event = self._hand_check(client, key, address, check, now, deadline) or event
            for job in held:
                event = self._tend(client, key, address, job, now, deadline) or event
            if checks.pending(self.conn):
                return event
            if (
                idle
                and not jobs.holding(self.conn)
                and (self.claim_at is None or now >= self.claim_at)
                and self._fresh(address)
            ):
                event = self._claim(client, key, config, address, now, deadline) or event
            return event
        except _Unreachable as e:
            return self._failed(address, str(e), now)
        except _Problem as e:
            return self._refused(address, key, e.error, now)

    def _clusters(self, now: datetime, idle: bool) -> None:
        if not idle:
            return
        if self.cluster_at is not None and now < self.cluster_at:
            return
        config = self._config()
        address = net.registry(config)
        if (
            address is None
            or address != self.registry
            or not self.registered
            or not self.work
            or not config.network.lead
            or self.suspended
            or self._trust is None
            or self.failures
            or not self._fresh(address)
        ):
            return
        key = self._key(address)
        if key is None:
            return
        self.cluster_at = now + cluster.CLUSTER_POLL
        try:
            state = _call(
                self._http(),
                key,
                address,
                "GET",
                protocol.CLUSTERS_MINE,
                None,
                protocol.ClusterState,
                now,
                time.monotonic() + TIMEOUT,
            )
            with self._seen.inspect():
                cluster.receive(state, self._trust)
            for record in [
                *state.appointments,
                *([state.appointment] if state.appointment else []),
            ]:
                self._appointments[record.record.id] = record
            if state.cluster is None:
                cluster.left(self.conn, now, ended=True)
                return
            appointment = self._trust.appointment(state.cluster)
            if appointment is None:
                return
            if state.task is None or state.task.record.id != appointment.task:
                raise sign.Refused(sign.Reason.NOT_ALLOWED, "snapshot names another task")
            who = (
                "operator"
                if appointment.fingerprint == key.fingerprint
                else "member"
                if key.fingerprint in appointment.members
                else "none"
            )
            if who == "none":
                return
            cluster.joined(self.conn, appointment, who, now)
            if who != "operator" or not idle or self.cluster_model is None:
                return

            def send(kind: sign.RecordKind, body: BaseModel) -> sign.Signed | None:
                moment = self._clock()
                current = self._config()
                term = self._trust.appointment(appointment.cluster) if self._trust else None
                if (
                    net.registry(current) != address
                    or not current.network.lead
                    or term is None
                    or term.fingerprint != key.fingerprint
                    or state.task is None
                ):
                    return None
                expires = state.task.record.expires_at
                # Authority is checked at issuance; records survive renewals/replacement.
                if moment >= expires:
                    return None
                signed = sign.sign(
                    key,
                    kind,
                    body.model_dump(mode="json"),
                    issued_at=moment,
                    lifetime=expires - moment,
                    id=cluster.plan_id(body, state.note)
                    if isinstance(body, protocol.ClusterPlan)
                    else None,
                )
                try:
                    received = _call(
                        self._http(),
                        key,
                        address,
                        "POST",
                        protocol.CLUSTER_RECORDS,
                        protocol.ClusterHandIn(record=signed),
                        protocol.ClusterReceived,
                        moment,
                        time.monotonic() + TIMEOUT,
                    )
                    if (
                        received.cluster != appointment.cluster
                        or received.record != signed.record.id
                        or received.kind != kind.value
                    ):
                        raise _Unreachable("bad_reply")
                except _Problem as e:
                    if e.error in (sign.Reason.NOT_ALLOWED.value, sign.Reason.MALFORMED.value):
                        if (
                            self.conn.execute(
                                "SELECT 1 FROM network_log WHERE action = 'record_refused' AND "
                                "json_extract(details_json, '$.registry') = ? AND "
                                "json_extract(details_json, '$.id') = ? AND "
                                "json_extract(details_json, '$.reason') = ? LIMIT 1",
                                (address, signed.record.id, e.error),
                            ).fetchone()
                            is None
                        ):
                            self._once(
                                (signed.record.id, e.error),
                                "record_refused",
                                {"registry": address, "id": signed.record.id, "reason": e.error},
                            )
                        raise _ClusterRefused from e
                    if e.error == sign.Reason.CONFLICT.value:
                        return None
                    raise
                netlog.record(
                    self.conn,
                    kind.value,
                    {
                        "cluster": appointment.cluster,
                        "task": appointment.task,
                        "record": signed.record.id,
                        "version": body.model_dump().get("version"),
                        **(
                            {"nodes": len(body.nodes)}
                            if isinstance(body, protocol.ClusterPlan)
                            else {}
                        ),
                    },
                )
                return signed

            def stuck(why: str) -> None:
                if (
                    self.conn.execute(
                        "SELECT 1 FROM network_log WHERE action='cluster_stuck' AND "
                        "json_extract(details_json, '$.task')=? AND "
                        "json_extract(details_json, '$.reason')=? LIMIT 1",
                        (appointment.task, why),
                    ).fetchone()
                    is None
                ):
                    netlog.record(
                        self.conn,
                        "cluster_stuck",
                        {"cluster": appointment.cluster, "task": appointment.task, "reason": why},
                    )

            cluster.supervise(state, self.cluster_model, send, stuck)
        except _ClusterRefused:
            pass
        except ProviderError:
            pass  # Quota/availability may change before the next cluster poll.
        except (ValidationError, sign.KeyFileError):
            self._once(
                ("cluster", "malformed"),
                "record_refused",
                {"registry": address, "reason": sign.Reason.MALFORMED.value},
            )
        except sign.Refused as e:
            self._once(
                ("cluster", e.reason.value),
                "record_refused",
                {"registry": address, "reason": e.reason.value},
            )
        except _Unreachable as e:
            self._failed(address, str(e), now)
        except _Problem as e:
            self._refused(address, key, e.error, now)

    def _fresh(self, address: str) -> bool:
        """The trust list is fresh (G.6c): without it no work is taken from any operator, as
        the registry may be hiding a revocation. Each change is journaled once."""
        assert self._trust is not None
        why = self._trust.fresh()
        if why != self._stale:
            if why is None:
                netlog.record(self.conn, "trust_fresh", {"registry": address})
            else:
                netlog.record(self.conn, "trust_stale", {"registry": address, "why": why})
            self._stale = why
        return why is None

    def _tend(
        self,
        client: httpx.Client,
        key: sign.NodeKey,
        address: str,
        job: jobs.Held,
        now: datetime,
        deadline: float,
    ) -> Event | None:
        record = job.record
        if record is None:
            end = jobs.ending(self._store, job, self._workspaces, now)
            if end is None:
                if job.expires_at - now <= EXTEND_WHEN:
                    try:
                        lease = _call(
                            client,
                            key,
                            address,
                            "POST",
                            protocol.EXTEND,
                            protocol.Extend(lease=job.lease),
                            protocol.Lease,
                            now,
                            deadline,
                        )
                    except _Problem as e:
                        if e.error != LEASE_LOST:
                            raise
                        return self._drop(job, "lost", LEASE_LOST)
                    jobs.extended(self.conn, job.lease, lease.expires_at)
                return None
            if end.body is None:
                return self._drop(job, "abandoned", end.why)
            kind = (
                sign.RecordKind.VERDICT
                if isinstance(end.body, protocol.Verdict)
                else sign.RecordKind.RESULT
            )
            record = sign.sign(
                key,
                kind,
                end.body.model_dump(mode="json"),
                issued_at=now,
                lifetime=jobs.record_lifetime(job, now),
            )
            jobs.keep(self.conn, job.lease, record)
            job = jobs.held_by_task(self.conn, job.task_id) or job
            record = job.record or record
        try:
            received = _call(
                client,
                key,
                address,
                "POST",
                protocol.HAND_IN,
                protocol.HandIn(lease=job.lease, record=record),
                protocol.Received,
                now,
                deadline,
            )
        except _Problem as e:
            if e.error in (NOT_REGISTERED, REVOKED):
                raise
            # The lease ran out, or the record was refused: either way it is not taken.
            return self._drop(job, "lost", e.error)
        outcome = _outcome(record)
        jobs.close(self._store, job, "handed_in", f"handed in ({outcome})")
        netlog.record(
            self.conn,
            "handed_in",
            {
                **_job_details(job),
                "record": record.record.id,
                "kind": str(record.record.kind),
                "outcome": outcome,
                "status": received.status,
            },
        )
        return Event(Change.WORK_HANDED_IN, address, outcome)

    def _drop(self, job: jobs.Held, state: jobs.State, why: str) -> Event:
        jobs.close(self._store, job, state, why)
        netlog.record(self.conn, "work_dropped", _job_details(job, why))
        return Event(Change.WORK_DROPPED, job.registry, why)

    def _claim(
        self,
        client: httpx.Client,
        key: sign.NodeKey,
        config: Config,
        address: str,
        now: datetime,
        deadline: float,
    ) -> Event | None:
        self.claim_at = now + CLAIM_EVERY
        lent = active_capabilities(config, self.conn)
        job_work = bool(self.work) and any(
            jobs.quota_left(self.conn, config, c.provider, now) for c in lent
        )
        proposal_checks = checks.can_check()
        if not job_work and not proposal_checks:
            return None
        offer = _call(
            client,
            key,
            address,
            "POST",
            protocol.CLAIM,
            protocol.Claim(
                jobs=job_work, proposals=proposal_checks, lead=config.network.lead and job_work
            ),
            protocol.Offer,
            now,
            deadline,
        )
        if offer.check is not None:
            return self._check_proposal(client, key, address, offer.check, now)
        work = offer.work
        if work is None:
            return None
        if not self.work:  # claimed with jobs=False: a registry that lends one anyway is wrong
            raise _Unreachable("bad_reply")
        assert self._trust is not None
        task = work.task.record
        if task.kind is sign.RecordKind.CLUSTER_JOB:
            # The claim itself can appoint a successor after our last trust fetch.
            # Learn its membership before consuming this job's one receipt.
            try:
                trust = _call(
                    client,
                    key,
                    address,
                    "GET",
                    protocol.TRUST,
                    None,
                    protocol.TrustState,
                    now,
                    deadline,
                )
            except _Unreachable:
                pass  # The granted lease still needs receipt/refusal using the trust held.
            else:
                if self._receive(address, key, trust):
                    return self._stop(address, key, "the trust list revokes this node")
        try:
            received = self._trust.receive(work.task.model_dump(mode="json"))
            cluster_id = None
            if received.kind is sign.RecordKind.CLUSTER_JOB:
                body = protocol.ClusterJob.model_validate(received.body)
                cluster_id = body.cluster
                appointment = self._trust.appointment(body.cluster)
                operators = {
                    sign.Appointment.model_validate(r.record.body).fingerprint
                    for r in self._appointments.values()
                    if r.record.body.get("cluster") == body.cluster
                }
                if work.stage == jobs.EXECUTE:
                    allowed = (
                        appointment is not None
                        and key.fingerprint in appointment.members
                        and key.fingerprint != appointment.fingerprint
                    )
                else:
                    allowed = key.fingerprint not in operators
                if not allowed:
                    raise sign.Refused(sign.Reason.NOT_ALLOWED, "cluster role", id=task.id)
                job = body.job
            elif received.kind is sign.RecordKind.TASK:
                job = protocol.Job.model_validate(received.body)
            else:
                raise sign.Refused(sign.Reason.MALFORMED, "not a task", id=task.id)
            if work.stage == jobs.RECHECK:
                _check_result(work, task.id, now)
                if work.result is not None and work.result.record.signer == key.fingerprint:
                    raise sign.Refused(sign.Reason.NOT_ALLOWED, "own result", id=task.id)
        except sign.Refused as e:
            netlog.record(
                self.conn,
                "work_refused",
                {
                    "registry": address,
                    "job": task.id,
                    "signer": task.signer,
                    "stage": work.stage,
                    "reason": e.reason.value,
                },
            )
            return None
        except (ValidationError, sign.KeyFileError):
            netlog.record(
                self.conn,
                "work_refused",
                {
                    "registry": address,
                    "job": task.id,
                    "signer": task.signer,
                    "stage": work.stage,
                    "reason": sign.Reason.MALFORMED.value,
                },
            )
            return None
        task_id = jobs.take(self._store, address, work, job)
        netlog.record(
            self.conn,
            "work_taken",
            {
                "registry": address,
                "job": task.id,
                "signer": task.signer,
                "stage": work.stage,
                "task": task_id,
                **({"cluster": cluster_id} if cluster_id else {}),
            },
        )
        return Event(Change.WORK_TAKEN, address, work.stage)

    def _check_proposal(
        self,
        client: httpx.Client,
        key: sign.NodeKey,
        address: str,
        work: protocol.ProposalWork,
        now: datetime,
    ) -> Event | None:
        assert self._trust is not None
        try:
            checks.verify(work, self._trust, now, conn=self.conn, registry=address)
        except sign.Refused as e:
            netlog.record(
                self.conn,
                "check_refused",
                {
                    "registry": address,
                    "proposal": work.proposal,
                    "reason": e.reason.value,
                },
            )
            return None
        try:
            repo = evaluation.repository()
        except evaluation.EvaluationError:
            # The checkout may disappear after capability discovery.
            body = protocol.ProposalCheck(
                proposal=work.proposal,
                lease=work.lease,
                submission=protocol.submission_hash(work.submission.submission),
                commit=work.submission.submission.base_commit,
                outcome="unable",
                reason="DESMAN no longer runs from a git checkout.",
            )
        else:
            body = checks.run(work, repo, now)
        signed = sign.sign(
            key,
            sign.RecordKind.PROPOSAL_CHECK,
            body.model_dump(mode="json"),
            issued_at=self._clock(),
            lifetime=jobs.RECORD_LIFETIME,
        )
        checks.keep(self.conn, address, work, signed, self._clock())
        netlog.record(
            self.conn,
            "proposal_checked",
            {
                "registry": address,
                "proposal": work.proposal,
                "outcome": body.outcome,
                "verdict": body.verdict,
            },
        )
        pending = next((c for c in checks.pending(self.conn) if c.lease == work.lease), None)
        if pending is None:  # another worker already handed it in
            return None
        return self._hand_check(
            client, key, address, pending, self._clock(), time.monotonic() + TIMEOUT
        )

    def _hand_check(
        self,
        client: httpx.Client,
        key: sign.NodeKey,
        address: str,
        check: checks.Pending,
        now: datetime,
        deadline: float,
    ) -> Event | None:
        # A check takes seconds: consent can change while it runs.
        try:
            current = net.registry(self._config())
        except ConfigError:
            return None
        if current != address:
            checks.close(self.conn, check.lease, "lost")
            return None
        try:
            received = _call(
                client,
                key,
                address,
                "POST",
                protocol.CHECK_HAND_IN,
                protocol.HandInCheck(lease=check.lease, record=check.record),
                protocol.CheckReceived,
                now,
                deadline,
            )
            if received.proposal != check.proposal:
                raise _Unreachable("bad_reply")
        except _Problem as e:
            if e.error in (NOT_REGISTERED, REVOKED):
                raise
            checks.close(self.conn, check.lease, "lost")
            netlog.record(
                self.conn,
                "check_lost",
                {
                    "registry": address,
                    "proposal": check.proposal,
                    "reason": e.error,
                },
            )
            return Event(Change.WORK_DROPPED, address, e.error)
        checks.close(self.conn, check.lease, "handed_in")
        return Event(Change.CHECK_HANDED_IN, address, str(check.record.record.body["outcome"]))


def _check_result(work: protocol.Work, job_id: str, now: datetime) -> None:
    """A recheck's result: signed by the key that came with it, about this job, done, in term."""
    assert work.result is not None and work.result_key is not None
    signed = work.result
    sign.check_signature(signed, sign.decode_public(work.result_key))
    record = signed.record
    if record.kind is not sign.RecordKind.RESULT:
        raise sign.Refused(sign.Reason.MALFORMED, "not a result", id=record.id)
    if now >= record.expires_at:
        raise sign.Refused(sign.Reason.EXPIRED, "the result has expired", id=record.id)
    body = protocol.Result.model_validate(record.body)
    if body.job != job_id or body.outcome != "done":
        raise sign.Refused(sign.Reason.MALFORMED, "not a done result of this job", id=record.id)


def _outcome(record: sign.Signed) -> str:
    body = record.record.body
    if record.record.kind is sign.RecordKind.VERDICT:
        return "accept" if body.get("accept") else "reject"
    return str(body.get("outcome", ""))


def _job_details(job: jobs.Held, why: str | None = None) -> dict[str, Any]:
    details: dict[str, Any] = {
        "registry": job.registry,
        "job": job.job,
        "stage": job.stage,
        "task": job.task_id,
    }
    if why is not None:
        details["why"] = why
    return details


def _dump(lent: list[protocol.Capability]) -> list[dict[str, Any]]:
    return [json.loads(c.model_dump_json()) for c in lent]


class _ClusterRefused(Exception):
    """One cluster hand-in was refused; global connectivity remains healthy."""
