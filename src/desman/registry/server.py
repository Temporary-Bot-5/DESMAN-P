"""The registry's HTTP side (ADR 0004; ROADMAP G.4): `ThreadingHTTPServer`, plain HTTP.

G.7b lends proposals and accepts independent signed benchmark checks.

TLS is the reverse proxy's (docs/registry-deploy.md); the server listens on 127.0.0.1 unless
told otherwise. `Registry.handle()` is the whole of it and knows nothing of sockets: tests call
it through `httpx.MockTransport`, the HTTP handler below only reads a request and writes the
reply.

Every request is checked before anything else (`net.request.verify_request`): signature,
time, nonce; then whether its key is revoked; then its body against `net.protocol`. A refusal
is answered with a `protocol.Problem` and written to the journal with the key it claimed.
Only `REGISTER` is open to a key the registry has not seen; everything else needs a node that
takes part (`active`), and a signed request from one that does not is a refusal like any other.

Work (G.6) is `registry/work.py`'s: `CLAIM` lends a job, `EXTEND` keeps the lease, `HAND_IN`
takes a signed result or verdict. A token that names no lease the caller holds now is refused
as `lease_lost` (409), like any refusal journaled; a node that leaves gives its leases back.

What is answered before the signature is read (an unknown path, another method, a body over
`protocol.MAX_BODY`) is not journaled: nobody could be named, and anyone could fill the journal.
"""

from __future__ import annotations

import base64
import binascii
import json
import sqlite3
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Final, TypeVar

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from pydantic import BaseModel, ValidationError

from desman.core import db
from desman.net import protocol, request, sign
from desman.net.sign import Reason, Refused
from desman.registry import checks, releases, store, work

DEFAULT_HOST: Final = "127.0.0.1"
DEFAULT_PORT: Final = 8470
SOCKET_TIMEOUT: Final = 15.0
"""Seconds a connection may stay silent (a slow client cannot hold a thread for long)."""

NOT_REGISTERED: Final = "not_registered"
NOT_FOUND: Final = "not_found"
NOT_ALLOWED_METHOD: Final = "method_not_allowed"
TOO_LARGE: Final = "too_large"
PROPOSAL_LIMIT: Final = protocol.PROPOSAL_LIMIT
INTERNAL: Final = "internal"

M = TypeVar("M", bound=BaseModel)

_STATUS: Final = {
    Reason.MALFORMED: HTTPStatus.BAD_REQUEST,
    Reason.REVOKED: HTTPStatus.FORBIDDEN,
    Reason.CONFLICT: HTTPStatus.CONFLICT,
}
"""Refusals that are not 401 (who sent it could not be told, or must not be served)."""


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class Reply:
    status: int
    body: dict[str, Any]

    def content(self) -> bytes:
        return json.dumps(self.body, ensure_ascii=False, sort_keys=True).encode("utf-8")


def _problem(status: int, error: str, detail: str = "") -> Reply:
    return Reply(status, protocol.Problem(error=error, detail=detail).model_dump())


class _NotRegistered(Exception):
    """A signed request from a key that does not take part (never registered, or left)."""


def _ok(model: BaseModel) -> Reply:
    return Reply(HTTPStatus.OK, model.model_dump(mode="json"))


Handler = Callable[[sqlite3.Connection, request.Caller, bytes, datetime], Reply]


class Registry:
    """The registry's answers, one database connection per request."""

    def __init__(
        self,
        path: Path,
        *,
        clock: Callable[[], datetime] = _utc_now,
        heartbeat_seconds: int = protocol.HEARTBEAT_SECONDS,
        root: Path | None = None,
    ) -> None:
        self.path = path
        self._clock = clock
        self._heartbeat = heartbeat_seconds
        self._root_path = root
        self._root: Ed25519PublicKey | None = None
        self._routes: dict[str, tuple[str, Handler]] = {
            protocol.REGISTER: ("POST", self._register),
            protocol.HEARTBEAT: ("POST", self._alive),
            protocol.LEAVE: ("POST", self._leave),
            protocol.TRUST: ("GET", self._trust),
            protocol.RELEASES: ("GET", self._releases),
            protocol.CLAIM: ("POST", self._claim),
            protocol.EXTEND: ("POST", self._extend),
            protocol.HAND_IN: ("POST", self._hand_in),
            protocol.CHECK_HAND_IN: ("POST", self._hand_in_check),
            protocol.PROPOSALS: ("POST", self._submit_proposal),
            protocol.REPORTS: ("POST", self._submit_report),
        }

    @property
    def root(self) -> Ed25519PublicKey:
        """The root nodes check jobs against (`net/root.pub` unless told otherwise)."""
        if self._root is None:
            self._root = sign.load_root(
                sign.ROOT_PUB if self._root_path is None else self._root_path
            )
        return self._root

    def handle(
        self, method: str, target: str, headers: Iterable[tuple[str, str]], body: bytes
    ) -> Reply:
        route = self._routes.get(target.split("?", 1)[0])
        if route is None:
            return _problem(HTTPStatus.NOT_FOUND, NOT_FOUND)
        wanted, handler = route
        if method != wanted:
            return _problem(HTTPStatus.METHOD_NOT_ALLOWED, NOT_ALLOWED_METHOD, wanted)
        if len(body) > protocol.MAX_BODY:
            return _problem(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, TOO_LARGE)
        now = self._clock()
        with store.connection(self.path) as conn:
            try:
                caller = request.verify_request(
                    method, target, body, headers, now=now, seen=store.NonceStore(conn)
                )
                if store.is_revoked(conn, caller.fingerprint):
                    raise Refused(Reason.REVOKED, "this key is revoked", signer=caller.fingerprint)
                return handler(conn, caller, body, now)
            except Refused as e:
                self._refused(conn, e.signer, e.reason.value, method, target)
                return _problem(_STATUS.get(e.reason, HTTPStatus.UNAUTHORIZED), e.reason.value)
            except _NotRegistered:
                self._refused(conn, caller.fingerprint, NOT_REGISTERED, method, target)
                return _problem(HTTPStatus.FORBIDDEN, NOT_REGISTERED)
            except work.LeaseLost:
                self._refused(conn, caller.fingerprint, work.LEASE_LOST, method, target)
                return _problem(HTTPStatus.CONFLICT, work.LEASE_LOST)

    def _refused(
        self, conn: sqlite3.Connection, signer: str | None, reason: str, method: str, target: str
    ) -> None:
        with db.transaction(conn):
            store.log(
                conn, "refused", signer, {"reason": reason, "method": method, "path": target[:200]}
            )

    # ── Routes ───────────────────────────────────────────────────────────────

    def _state(self, fingerprint: str) -> Reply:
        return _ok(protocol.NodeState(node=fingerprint, heartbeat_seconds=self._heartbeat))

    def _register(
        self, conn: sqlite3.Connection, caller: request.Caller, body: bytes, now: datetime
    ) -> Reply:
        asked = _body(protocol.Register, body, caller)
        moment = db.to_iso(now)
        with db.transaction(conn):
            known = store.node(conn, caller.fingerprint)
            conn.execute(
                "INSERT INTO nodes (fingerprint, public_key, status, capabilities_json,"
                " registered_at, last_seen) VALUES (?, ?, 'active', ?, ?, ?)"
                " ON CONFLICT (fingerprint) DO UPDATE SET status = 'active',"
                " capabilities_json = excluded.capabilities_json, last_seen = excluded.last_seen",
                (
                    caller.fingerprint,
                    caller.public,
                    store.capabilities_json(asked.capabilities),
                    moment,
                    moment,
                ),
            )
            store.log(
                conn,
                "registered",
                caller.fingerprint,
                {
                    "again": known is not None,
                    "capabilities": [c.model_dump(mode="json") for c in asked.capabilities],
                },
            )
        return self._state(caller.fingerprint)

    def _active(self, conn: sqlite3.Connection, caller: request.Caller) -> store.Node:
        found = store.node(conn, caller.fingerprint)
        if found is None or found.status != "active":
            raise _NotRegistered
        return found

    def _alive(
        self, conn: sqlite3.Connection, caller: request.Caller, body: bytes, now: datetime
    ) -> Reply:
        asked = _body(protocol.Heartbeat, body, caller)
        with db.transaction(conn):
            found = self._active(conn, caller)
            conn.execute(
                "UPDATE nodes SET last_seen = ? WHERE fingerprint = ?",
                (db.to_iso(now), caller.fingerprint),
            )
            if asked.capabilities is not None and asked.capabilities != found.capabilities:
                conn.execute(
                    "UPDATE nodes SET capabilities_json = ? WHERE fingerprint = ?",
                    (store.capabilities_json(asked.capabilities), caller.fingerprint),
                )
                store.log(
                    conn,
                    "capabilities",
                    caller.fingerprint,
                    {"capabilities": [c.model_dump(mode="json") for c in asked.capabilities]},
                )
        return self._state(caller.fingerprint)

    def _leave(
        self, conn: sqlite3.Connection, caller: request.Caller, body: bytes, now: datetime
    ) -> Reply:
        _no_body(body, caller)
        with db.transaction(conn):
            found = store.node(conn, caller.fingerprint)
            if found is None:
                raise _NotRegistered
            if found.status == "left":
                return _ok(protocol.Left(node=caller.fingerprint, changed=False))
            conn.execute(
                "UPDATE nodes SET status = 'left', capabilities_json = '[]', last_seen = ?"
                " WHERE fingerprint = ?",
                (db.to_iso(now), caller.fingerprint),
            )
            store.log(conn, "left", caller.fingerprint)
            work.withdraw(conn, caller.fingerprint, now)
        return _ok(protocol.Left(node=caller.fingerprint, changed=True))

    def _trust(
        self, conn: sqlite3.Connection, caller: request.Caller, body: bytes, now: datetime
    ) -> Reply:
        _no_body(body, caller)
        with db.read_snapshot(conn):
            self._active(conn, caller)
            records = store.handed_out(conn, now)
        return _ok(protocol.TrustState(records=records))

    def _releases(
        self, conn: sqlite3.Connection, caller: request.Caller, body: bytes, now: datetime
    ) -> Reply:
        _no_body(body, caller)
        with db.read_snapshot(conn):
            self._active(conn, caller)
            records = releases.handed_out(conn)
        return _ok(protocol.Releases(records=records))

    def _claim(
        self, conn: sqlite3.Connection, caller: request.Caller, body: bytes, now: datetime
    ) -> Reply:
        asked = _body(protocol.Claim, body, caller)
        found = self._active(conn, caller)
        offered = work.claim(
            conn, found, self.root, now, jobs=asked.jobs, proposals=asked.proposals
        )
        return _ok(
            protocol.Offer(check=offered)
            if isinstance(offered, protocol.ProposalWork)
            else protocol.Offer(work=offered)
        )

    def _extend(
        self, conn: sqlite3.Connection, caller: request.Caller, body: bytes, now: datetime
    ) -> Reply:
        asked = _body(protocol.Extend, body, caller)
        found = self._active(conn, caller)
        return _ok(work.extend(conn, found, asked.lease, now))

    def _hand_in(
        self, conn: sqlite3.Connection, caller: request.Caller, body: bytes, now: datetime
    ) -> Reply:
        given = _body(protocol.HandIn, body, caller)
        found = self._active(conn, caller)
        return _ok(work.hand_in(conn, found, given, now))

    def _hand_in_check(
        self, conn: sqlite3.Connection, caller: request.Caller, body: bytes, now: datetime
    ) -> Reply:
        """Keep an independent proposal benchmark check (G.7b)."""
        given = _body(protocol.HandInCheck, body, caller)
        return _ok(checks.hand_in(conn, self._active(conn, caller), given, now))

    def _submit_proposal(
        self, conn: sqlite3.Connection, caller: request.Caller, body: bytes, now: datetime
    ) -> Reply:
        """Store the proposal body authenticated by this request's node signature."""
        asked = _body(protocol.SubmitProposal, body, caller)
        self._active(conn, caller)
        if len(body) > protocol.MAX_PROPOSAL_BYTES:
            raise Refused(Reason.MALFORMED, "proposal is too large", signer=caller.fingerprint)
        if asked.signed.signer != caller.fingerprint:
            raise Refused(
                Reason.UNKNOWN_SIGNER,
                "proposal signer is not the request node",
                signer=caller.fingerprint,
            )
        try:
            caller.key.verify(
                base64.b64decode(asked.signed.signature, validate=True),
                sign.canonical(asked.signed.submission.model_dump(mode="json")),
            )
        except (InvalidSignature, binascii.Error) as e:
            raise Refused(Reason.FORGED, "bad proposal signature", signer=caller.fingerprint) from e
        try:
            found, _ = store.submit_proposal(conn, caller.fingerprint, asked.signed, now)
        except store.ProposalLimit:
            self._refused(conn, caller.fingerprint, PROPOSAL_LIMIT, "POST", protocol.PROPOSALS)
            return _problem(HTTPStatus.FORBIDDEN, PROPOSAL_LIMIT)
        except ValueError as e:
            raise Refused(Reason.CONFLICT, str(e), signer=caller.fingerprint) from e
        return _ok(
            protocol.ProposalReceived(
                id=found.id,
                proposal=found.submission.proposal,
                evaluation=found.submission.evaluation,
                status="submitted",
            )
        )

    def _submit_report(
        self, conn: sqlite3.Connection, caller: request.Caller, body: bytes, now: datetime
    ) -> Reply:
        """Store the report body authenticated by this request's node signature."""
        asked = _body(protocol.SubmitReport, body, caller)
        self._active(conn, caller)
        if len(body) > protocol.MAX_REPORT_BYTES:
            raise Refused(Reason.MALFORMED, "report is too large", signer=caller.fingerprint)
        if asked.signed.signer != caller.fingerprint:
            raise Refused(
                Reason.UNKNOWN_SIGNER,
                "report signer is not the request node",
                signer=caller.fingerprint,
            )
        try:
            caller.key.verify(
                base64.b64decode(asked.signed.signature, validate=True),
                protocol.report_content(asked.signed.report),
            )
        except (InvalidSignature, binascii.Error) as e:
            raise Refused(Reason.FORGED, "bad report signature", signer=caller.fingerprint) from e
        try:
            found, _ = store.submit_report(conn, caller.fingerprint, asked.signed, now)
        except store.ReportLimit:
            self._refused(conn, caller.fingerprint, protocol.REPORT_LIMIT, "POST", protocol.REPORTS)
            return _problem(HTTPStatus.FORBIDDEN, protocol.REPORT_LIMIT)
        except ValueError as e:
            raise Refused(Reason.CONFLICT, str(e), signer=caller.fingerprint) from e
        return _ok(found)


def _body(model: type[M], body: bytes, caller: request.Caller) -> M:
    try:
        return model.model_validate_json(body or b"{}")
    except ValidationError as e:
        raise Refused(
            Reason.MALFORMED, f"{e.error_count()} problem(s) in the body", signer=caller.fingerprint
        ) from e


def _no_body(body: bytes, caller: request.Caller) -> None:
    if body:
        raise Refused(Reason.MALFORMED, "this request has no body", signer=caller.fingerprint)


# ── HTTP ─────────────────────────────────────────────────────────────────────


def handler_for(registry: Registry) -> type[BaseHTTPRequestHandler]:
    class RegistryHandler(BaseHTTPRequestHandler):
        server_version = "desman-registry"

        def version_string(self) -> str:
            return self.server_version  # no Python version on the wire

        timeout = SOCKET_TIMEOUT
        # HTTP/1.0: one request per connection, so a refused body never lingers on it.

        def _serve(self) -> None:
            if self.headers.get("Transfer-Encoding"):
                self._reply(_problem(HTTPStatus.LENGTH_REQUIRED, "length_required"))
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = -1
            if length < 0:
                self._reply(_problem(HTTPStatus.BAD_REQUEST, Reason.MALFORMED.value))
                return
            if length > protocol.MAX_BODY:
                self._reply(_problem(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, TOO_LARGE))
                return
            body = self.rfile.read(length) if length else b""
            try:
                reply = registry.handle(self.command, self.path, self.headers.items(), body)
            except Exception as e:
                self.log_error("%s", f"{type(e).__name__}: {e}")
                reply = _problem(HTTPStatus.INTERNAL_SERVER_ERROR, INTERNAL)
            self._reply(reply)

        def _reply(self, reply: Reply) -> None:
            content = reply.content()
            self.send_response(reply.status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

        do_GET = _serve
        do_POST = _serve
        do_PUT = _serve
        do_DELETE = _serve
        do_PATCH = _serve

    return RegistryHandler


def make_server(
    registry: Registry, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT
) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), handler_for(registry))
    server.daemon_threads = True
    return server
