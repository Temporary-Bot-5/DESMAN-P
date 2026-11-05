"""desman-registry (ROADMAP G.4): signed requests, nodes, records, the journal, the CLI."""

from __future__ import annotations

import base64
import json
import sqlite3
import subprocess
import sys
import threading
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
from click.testing import CliRunner

from conftest import FakeClock
from desman.core import db
from desman.net import protocol, request, sign
from desman.net.sign import MemorySeen, Reason, Refused
from desman.registry import admin, store
from desman.registry.cli import cli
from desman.registry.server import Registry, make_server

API_KEY = "sk-very-secret-0123456789"


@pytest.fixture(autouse=True)
def registry_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Never the owner's real registry directory (it holds the root key)."""
    path = tmp_path / "registry"
    monkeypatch.setenv(store.DIR_ENV, str(path))
    return path


@pytest.fixture
def db_path(registry_dir: Path) -> Path:
    path = registry_dir / store.DB_NAME
    store.open_registry(path).close()
    return path


@pytest.fixture
def root() -> sign.NodeKey:
    return sign.NodeKey.generate()


@pytest.fixture
def root_pub(tmp_path: Path, root: sign.NodeKey, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "root.pub"
    path.write_bytes(sign.public_pem(root.public))
    monkeypatch.setattr(sign, "ROOT_PUB", path)
    return path


@pytest.fixture
def registry(db_path: Path, clock: FakeClock) -> Registry:
    return Registry(db_path, clock=clock)


class Node:
    """A node talking to the registry in process, through httpx.MockTransport."""

    def __init__(self, registry: Registry, clock: FakeClock, key: sign.NodeKey | None = None):
        self.registry = registry
        self.clock = clock
        self.key = sign.NodeKey.generate() if key is None else key
        self.client = httpx.Client(
            transport=httpx.MockTransport(self._handle), base_url="https://registry.test"
        )
        self.last: httpx.Request | None = None

    def _handle(self, req: httpx.Request) -> httpx.Response:
        self.last = req
        reply = self.registry.handle(
            req.method, req.url.raw_path.decode("ascii"), req.headers.multi_items(), req.content
        )
        return httpx.Response(reply.status, json=reply.body)

    @property
    def fingerprint(self) -> str:
        return self.key.fingerprint

    def call(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        *,
        headers: dict[str, str] | None = None,
        content: bytes | None = None,
    ) -> httpx.Response:
        data = (b"" if body is None else json.dumps(body).encode()) if content is None else content
        signed = request.sign_request(self.key, method, path, data, now=self.clock())
        return self.client.request(method, path, content=data, headers=signed | (headers or {}))

    def register(self, capabilities: list[dict[str, Any]] | None = None) -> httpx.Response:
        return self.call("POST", protocol.REGISTER, {"capabilities": capabilities or []})

    def heartbeat(self, capabilities: list[dict[str, Any]] | None = None) -> httpx.Response:
        body = {} if capabilities is None else {"capabilities": capabilities}
        return self.call("POST", protocol.HEARTBEAT, body)

    def replay(self) -> httpx.Response:
        assert self.last is not None
        return self.client.send(self.last)


GROQ = {"provider": "groq", "model": "llama-3.3-70b", "key": "0123456789ab", "calls_per_day": 50,
        "tokens_per_day": 50_000}  # fmt: skip
LOCAL = {"provider": "ollama", "model": "qwen2.5:7b", "calls_per_day": 200, "tokens_per_day": 0}


def log_actions(path: Path) -> list[tuple[str, str | None]]:
    with store.connection(path) as conn:
        return [(e.action, e.node) for e in store.entries(conn)]


# ── Signed requests ──────────────────────────────────────────────────────────


def signed(key: sign.NodeKey, clock: FakeClock, body: bytes = b"{}") -> dict[str, str]:
    return request.sign_request(key, "POST", "/v1/nodes/register", body, now=clock())


def verify(
    headers: dict[str, str],
    clock: FakeClock,
    *,
    method: str = "POST",
    path: str = "/v1/nodes/register",
    body: bytes = b"{}",
    seen: MemorySeen | None = None,
) -> request.Caller:
    return request.verify_request(
        method, path, body, headers.items(), now=clock(), seen=seen or MemorySeen()
    )


def test_a_signed_request_names_its_node(clock: FakeClock) -> None:
    key = sign.NodeKey.generate()
    caller = verify(signed(key, clock), clock)
    assert caller.fingerprint == key.fingerprint
    assert caller.time == clock().replace(microsecond=0)


@pytest.mark.parametrize(
    "change",
    [
        {"method": "PUT"},
        {"path": "/v1/nodes/heartbeat"},
        {"path": "/v1/nodes/register?x=1"},
        {"body": b'{"capabilities":[]}'},
    ],
)
def test_anything_changed_after_signing_is_forged(clock: FakeClock, change: dict[str, Any]) -> None:
    key = sign.NodeKey.generate()
    with pytest.raises(Refused) as caught:
        verify(signed(key, clock), clock, **change)
    assert caught.value.reason is Reason.FORGED
    assert caught.value.signer == key.fingerprint


def test_another_key_cannot_claim_a_signature(clock: FakeClock) -> None:
    mine, theirs = sign.NodeKey.generate(), sign.NodeKey.generate()
    headers = signed(mine, clock) | {request.KEY: sign.encode_public(theirs.public)}
    with pytest.raises(Refused) as caught:
        verify(headers, clock)
    assert caught.value.reason is Reason.FORGED
    assert caught.value.signer == theirs.fingerprint


def test_a_signed_time_and_nonce_cannot_be_rewritten(clock: FakeClock) -> None:
    key = sign.NodeKey.generate()
    later = request.format_time(clock() + timedelta(seconds=60))
    for header, value in ((request.TIME, later), (request.NONCE, "f" * 32)):
        with pytest.raises(Refused) as caught:
            verify(signed(key, clock) | {header: value}, clock)
        assert caught.value.reason is Reason.FORGED


def test_a_request_is_good_once(clock: FakeClock) -> None:
    key, seen = sign.NodeKey.generate(), MemorySeen()
    headers = signed(key, clock)
    verify(headers, clock, seen=seen)
    with pytest.raises(Refused) as caught:
        verify(headers, clock, seen=seen)
    assert caught.value.reason is Reason.REPLAY


def test_a_request_out_of_its_time_is_refused(clock: FakeClock) -> None:
    key = sign.NodeKey.generate()
    early = signed(key, clock)
    clock.advance(sign.SKEW.total_seconds() + 1)
    with pytest.raises(Refused) as caught:
        verify(early, clock)
    assert caught.value.reason is Reason.EXPIRED

    future = request.sign_request(
        key, "POST", "/v1/nodes/register", b"{}", now=clock() + sign.SKEW + timedelta(seconds=1)
    )
    with pytest.raises(Refused) as caught:
        verify(future, clock)
    assert caught.value.reason is Reason.NOT_YET


def test_a_request_within_the_skew_is_accepted(clock: FakeClock) -> None:
    key = sign.NodeKey.generate()
    headers = signed(key, clock)
    clock.advance(sign.SKEW.total_seconds() - 1)
    verify(headers, clock)


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {request.KEY: "not a key"},
        {request.TIME: "2026-01-01 00:00:00"},
        {request.NONCE: "short"},
        {request.SIGNATURE: "x"},
    ],
)
def test_malformed_headers_are_refused(clock: FakeClock, headers: dict[str, str]) -> None:
    key = sign.NodeKey.generate()
    found = signed(key, clock) | headers
    if not headers:
        del found[request.NONCE]
    with pytest.raises(Refused) as caught:
        verify(found, clock)
    assert caught.value.reason is Reason.MALFORMED


def test_a_header_given_twice_is_refused(clock: FakeClock) -> None:
    key = sign.NodeKey.generate()
    headers = list(signed(key, clock).items()) + [(request.NONCE.lower(), "a" * 32)]
    with pytest.raises(Refused) as caught:
        request.verify_request(
            "POST", "/v1/nodes/register", b"{}", headers, now=clock(), seen=MemorySeen()
        )
    assert caught.value.reason is Reason.MALFORMED


def test_header_names_are_case_insensitive(clock: FakeClock) -> None:
    key = sign.NodeKey.generate()
    headers = {name.lower(): value for name, value in signed(key, clock).items()}
    assert verify(headers, clock).fingerprint == key.fingerprint


# ── Nodes ────────────────────────────────────────────────────────────────────


def test_a_node_registers_with_what_it_lends(
    registry: Registry, clock: FakeClock, db_path: Path
) -> None:
    node = Node(registry, clock)
    answer = node.register([GROQ, LOCAL])
    assert answer.status_code == 200
    assert answer.json() == {
        "node": node.fingerprint,
        "heartbeat_seconds": protocol.HEARTBEAT_SECONDS,
    }
    with store.connection(db_path) as conn:
        found = store.node(conn, node.fingerprint)
    assert found is not None and found.status == "active" and not found.revoked
    assert [c.provider for c in found.capabilities] == ["groq", "ollama"]
    assert found.public_key == sign.encode_public(node.key.public)
    assert log_actions(db_path) == [("registered", node.fingerprint)]


def test_heartbeat_keeps_the_node_seen_and_logs_only_changes(
    registry: Registry, clock: FakeClock, db_path: Path
) -> None:
    node = Node(registry, clock)
    node.register([GROQ])
    clock.advance(300)
    assert node.heartbeat().status_code == 200
    assert node.heartbeat([GROQ]).status_code == 200
    assert node.heartbeat([GROQ, LOCAL]).status_code == 200
    with store.connection(db_path) as conn:
        found = store.node(conn, node.fingerprint)
    assert found is not None
    assert found.last_seen == db.to_iso(clock())
    assert len(found.capabilities) == 2
    assert log_actions(db_path) == [
        ("registered", node.fingerprint),
        ("capabilities", node.fingerprint),
    ]


def test_only_a_registered_node_is_served(
    registry: Registry, clock: FakeClock, db_path: Path
) -> None:
    node = Node(registry, clock)
    for answer in (
        node.heartbeat(),
        node.call("GET", protocol.TRUST),
        node.call("POST", protocol.LEAVE),
    ):
        assert answer.status_code == 403
        assert answer.json()["error"] == "not_registered"
    with store.connection(db_path) as conn:
        refused = store.entries(conn)
    assert [(e.action, e.node, e.details["reason"]) for e in refused] == [
        ("refused", node.fingerprint, "not_registered")
    ] * 3


def test_leaving_takes_back_what_was_lent(
    registry: Registry, clock: FakeClock, db_path: Path
) -> None:
    node = Node(registry, clock)
    node.register([GROQ])
    answer = node.call("POST", protocol.LEAVE)
    assert answer.json() == {"node": node.fingerprint, "changed": True}
    assert node.call("POST", protocol.LEAVE).json()["changed"] is False
    with store.connection(db_path) as conn:
        found = store.node(conn, node.fingerprint)
    assert found is not None and found.status == "left" and found.capabilities == []
    assert node.heartbeat().status_code == 403

    assert node.register().status_code == 200
    assert node.heartbeat().status_code == 200
    with store.connection(db_path) as conn:
        entries = store.entries(conn)
    assert [e.action for e in entries] == ["registered", "left", "refused", "registered"]
    assert entries[-1].details["again"] is True


def test_the_body_carries_modules_never_keys(
    registry: Registry, clock: FakeClock, db_path: Path
) -> None:
    node = Node(registry, clock)
    for capability in (GROQ | {"api_key": API_KEY}, GROQ | {"key": API_KEY}):
        answer = node.register([capability])
        assert answer.status_code == 400
        assert answer.json()["error"] == "malformed"
    assert API_KEY not in db_path.read_bytes().decode("latin-1")
    assert log_actions(db_path) == [("refused", node.fingerprint)] * 2


@pytest.mark.parametrize(
    "body",
    [
        b"not json",
        b'{"capabilities": "groq"}',
        json.dumps({"capabilities": [GROQ, GROQ]}).encode(),
        json.dumps({"capabilities": [GROQ | {"calls_per_day": "50"}]}).encode(),
        json.dumps({"capabilities": [LOCAL] * (protocol.MAX_CAPABILITIES + 1)}).encode(),
    ],
)
def test_a_malformed_body_is_refused(registry: Registry, clock: FakeClock, body: bytes) -> None:
    answer = Node(registry, clock).call("POST", protocol.REGISTER, content=body)
    assert answer.status_code == 400


def test_a_get_has_no_body(registry: Registry, clock: FakeClock) -> None:
    node = Node(registry, clock)
    node.register()
    assert node.call("GET", protocol.TRUST, content=b"{}").status_code == 400


def test_refusals_are_journaled(registry: Registry, clock: FakeClock, db_path: Path) -> None:
    node = Node(registry, clock)
    node.register()
    assert node.replay().status_code == 401
    forged = node.call("POST", protocol.HEARTBEAT, {}, headers={request.NONCE: "e" * 32})
    assert forged.status_code == 401
    unsigned = node.client.post(protocol.HEARTBEAT, content=b"{}")
    assert unsigned.status_code == 400
    with store.connection(db_path) as conn:
        refused = [e for e in store.entries(conn) if e.action == "refused"]
    assert [(e.details["reason"], e.node) for e in refused] == [
        ("replay", node.fingerprint),
        ("forged", node.fingerprint),
        ("malformed", None),
    ]
    assert refused[0].details["path"] == protocol.REGISTER


def test_routes_methods_and_sizes(registry: Registry, clock: FakeClock, db_path: Path) -> None:
    node = Node(registry, clock)
    assert node.call("GET", "/v1/elsewhere").status_code == 404
    assert node.call("GET", protocol.REGISTER).status_code == 405
    huge = b"x" * (protocol.MAX_BODY + 1)
    assert node.call("POST", protocol.REGISTER, content=huge).status_code == 413
    assert log_actions(db_path) == []  # nothing was read: nothing to journal


def test_nonces_outlive_a_restart(db_path: Path, clock: FakeClock) -> None:
    node = Node(Registry(db_path, clock=clock), clock)
    node.register()
    node.registry = Registry(db_path, clock=clock)
    assert node.replay().json()["error"] == "replay"


def test_old_nonces_are_forgotten(db_path: Path, clock: FakeClock) -> None:
    node = Node(Registry(db_path, clock=clock), clock)
    node.register()
    clock.advance(sign.SKEW.total_seconds() * 3)
    node.heartbeat()
    with store.connection(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM nonces").fetchone()[0] == 1


# ── Records ──────────────────────────────────────────────────────────────────


def test_assignments_and_revocations_reach_nodes(
    registry: Registry, clock: FakeClock, db_path: Path, root: sign.NodeKey
) -> None:
    operator, bad = sign.NodeKey.generate(), sign.NodeKey.generate()
    with store.connection(db_path) as conn:
        admin.assign(conn, root, operator.public, {sign.Scope.TASKS}, days=30, now=clock())
        admin.revoke(conn, root, bad.fingerprint, "spam", now=clock())
    node = Node(registry, clock)
    node.register()
    answer = node.call("GET", protocol.TRUST)
    assert answer.status_code == 200
    state = protocol.TrustState.model_validate(answer.json())
    assert [s.record.kind for s in state.records] == ["assignment", "revocation", "trust_list"]

    believed = sign.Trust(root.public, clock=clock)
    for record in state.records:
        believed.receive(record.model_dump(mode="json"))
    found = believed.operator(operator.fingerprint)
    assert found is not None and found.scopes == {sign.Scope.TASKS}
    assert believed.is_revoked(bad.fingerprint)

    with store.connection(db_path) as conn:
        entries = store.entries(conn)
    assert believed.fresh() is None
    entries = [e for e in entries if e.action != "trust_list"]
    assert [(e.action, e.node) for e in entries[:2]] == [
        ("assignment", operator.fingerprint),
        ("revocation", bad.fingerprint),
    ]
    assert entries[0].details["scopes"] == ["tasks"]
    assert entries[1].details["reason"] == "spam"


def test_an_expired_assignment_is_not_handed_out(
    registry: Registry, clock: FakeClock, db_path: Path, root: sign.NodeKey
) -> None:
    operator = sign.NodeKey.generate()
    with store.connection(db_path) as conn:
        admin.assign(conn, root, operator.public, set(sign.Scope), days=1, now=clock())
    clock.advance(timedelta(days=2).total_seconds())
    node = Node(registry, clock)
    node.register()
    records = node.call("GET", protocol.TRUST).json()["records"]
    assert [r["record"]["kind"] for r in records] == ["trust_list"]  # signed with it, still on


def test_a_revoked_node_is_refused_everything(
    registry: Registry, clock: FakeClock, db_path: Path, root: sign.NodeKey
) -> None:
    node = Node(registry, clock)
    node.register([GROQ])
    with store.connection(db_path) as conn:
        admin.revoke(conn, root, node.fingerprint, "abuse", now=clock())
    for answer in (node.heartbeat(), node.register(), node.call("GET", protocol.TRUST)):
        assert answer.status_code == 403
        assert answer.json()["error"] == "revoked"
    with store.connection(db_path) as conn:
        found = store.node(conn, node.fingerprint)
    assert found is not None and found.revoked
    assert found.as_json()["status"] == "revoked"


def test_a_key_revoked_before_it_registers_cannot(
    registry: Registry, clock: FakeClock, db_path: Path, root: sign.NodeKey
) -> None:
    node = Node(registry, clock)
    with store.connection(db_path) as conn:
        admin.revoke(conn, root, node.fingerprint, "", now=clock())
    assert node.register().status_code == 403


def test_what_a_node_would_refuse_is_never_kept(
    db_path: Path, clock: FakeClock, root: sign.NodeKey
) -> None:
    operator = sign.NodeKey.generate()
    with store.connection(db_path) as conn:
        with pytest.raises(admin.AdminError, match="refuse"):
            admin.revoke(conn, root, root.fingerprint, "", now=clock())
        admin.revoke(conn, root, operator.fingerprint, "", now=clock())
        with pytest.raises(admin.AdminError, match="refuse"):
            admin.assign(conn, root, operator.public, set(sign.Scope), days=30, now=clock())
        assert [s.record.kind for s in store.all_records(conn)] == ["revocation"]
        assert [e.action for e in store.entries(conn)] == ["revocation", "trust_list"]


def test_records_and_the_journal_are_append_only(
    db_path: Path, clock: FakeClock, root: sign.NodeKey
) -> None:
    node = sign.NodeKey.generate()
    with store.connection(db_path) as conn:
        admin.revoke(conn, root, node.fingerprint, "", now=clock())
        for statement in (
            "UPDATE records SET kind = 'assignment'",
            "DELETE FROM records",
            "UPDATE registry_log SET action = 'x'",
            "DELETE FROM registry_log",
        ):
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                conn.execute(statement)


def test_only_the_root_of_root_pub_signs(
    tmp_path: Path, root: sign.NodeKey, root_pub: Path
) -> None:
    good, other = tmp_path / "root.key", tmp_path / "other.key"
    sign.create_node_key(other)
    with pytest.raises(admin.AdminError, match="not the root key"):
        admin.load_root_key(other)
    _write_key(good, root)
    assert admin.load_root_key(good).fingerprint == root.fingerprint


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX key modes; registry runs on Linux")
def test_root_key_must_be_private(tmp_path: Path, root: sign.NodeKey, root_pub: Path) -> None:
    good = tmp_path / "root.key"
    _write_key(good, root)
    good.chmod(0o644)
    with pytest.raises(admin.AdminError, match="cannot be used"):
        admin.load_root_key(good)


def _write_key(path: Path, key: sign.NodeKey) -> None:
    from cryptography.hazmat.primitives import serialization

    pem = key._private.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pem)
    path.chmod(0o600)


# ── Backup ───────────────────────────────────────────────────────────────────


def test_backup_is_a_consistent_private_copy(
    registry: Registry, clock: FakeClock, db_path: Path, tmp_path: Path
) -> None:
    Node(registry, clock).register([GROQ])
    dest = tmp_path / "copy.db"
    admin.backup(db_path, dest)
    if sys.platform != "win32":  # POSIX modes; still check backup contents on Windows
        assert dest.stat().st_mode & 0o777 == 0o600
    copy = store.open_registry(dest)
    try:
        assert len(store.nodes(copy)) == 1
    finally:
        copy.close()
    with pytest.raises(admin.AdminError, match="exists already"):
        admin.backup(db_path, dest)
    with pytest.raises(admin.AdminError, match="No database"):
        admin.backup(tmp_path / "nowhere.db", tmp_path / "copy2.db")


# ── CLI ──────────────────────────────────────────────────────────────────────


def registry_cli(*args: str) -> Any:
    return CliRunner().invoke(cli, list(args))


def test_cli_assign_revoke_nodes_log(
    registry_dir: Path, root: sign.NodeKey, root_pub: Path, tmp_path: Path
) -> None:
    _write_key(registry_dir / store.ROOT_KEY, root)
    done = registry_cli("assign", "--scope", "tasks", "--days", "7")
    assert done.exit_code == 0, done.output
    operator = sign.load_node_key(registry_dir / store.OPERATOR_KEY)
    assert operator.fingerprint in done.output

    other = tmp_path / "other.pub"
    other.write_bytes(sign.public_pem(sign.NodeKey.generate().public))
    assert registry_cli("assign", str(other)).exit_code == 0

    gone = sign.NodeKey.generate().fingerprint
    revoked = registry_cli("revoke", gone, "--reason", "abuse")
    assert revoked.exit_code == 0, revoked.output
    assert registry_cli("revoke", "nothex").exit_code == 2
    assert registry_cli("revoke", gone, "--reason", "x" * 501).exit_code == 2

    log = registry_cli("log", "--json")
    assert [e["action"] for e in json.loads(log.output)] == [
        "assignment",
        "trust_list",
        "assignment",
        "trust_list",
        "revocation",
        "trust_list",
    ]
    assert json.loads(registry_cli("nodes", "--json").output) == []
    assert "No node has registered yet." in registry_cli("nodes").output


def test_cli_refuses_a_root_key_that_is_not_the_root(registry_dir: Path, root_pub: Path) -> None:
    sign.create_node_key(registry_dir / store.ROOT_KEY)
    result = registry_cli("assign")
    assert result.exit_code == 1
    assert "is not the root key" in result.output
    assert not (registry_dir / store.OPERATOR_KEY).exists()


def test_cli_nodes_and_backup(
    registry: Registry, clock: FakeClock, db_path: Path, tmp_path: Path
) -> None:
    node = Node(registry, clock)
    node.register([GROQ])
    shown = registry_cli("nodes")
    assert node.fingerprint[:16] in shown.output and "groq/llama-3.3-70b" in shown.output
    assert "registered" in registry_cli("log").output
    dest = tmp_path / "b.db"
    assert registry_cli("backup", str(dest)).exit_code == 0
    assert registry_cli("backup", str(dest)).exit_code == 1


def test_cli_speaks_the_registry_language(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DESMAN_REGISTRY_LANGUAGE", "ru")
    result = registry_cli("log")
    assert result.exit_code == 0
    assert "Журнал пуст." in result.output


# ── Boundaries ───────────────────────────────────────────────────────────────


def test_the_registry_reads_no_config() -> None:
    code = (
        "import sys, desman.registry.cli, desman.registry.server;"
        "bad = [m for m in ('desman.config', 'desman.providers', 'desman.runs') if m in sys.modules];"
        "print(bad); sys.exit(1 if bad else 0)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_schema_is_checked_on_open(db_path: Path) -> None:
    with store.connection(db_path) as conn:
        conn.execute("DROP TRIGGER nodes_no_delete")
    with pytest.raises(db.SchemaMismatchError, match="desman-registry serve"):
        store.open_registry(db_path)


@pytest.fixture
def served(registry: Registry) -> Iterator[str]:
    httpd = make_server(registry, "127.0.0.1", 0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_the_http_server_answers_signed_requests(served: str, clock: FakeClock) -> None:
    key = sign.NodeKey.generate()
    body = json.dumps({"capabilities": [LOCAL]}).encode()
    headers = request.sign_request(key, "POST", protocol.REGISTER, body, now=clock())
    with httpx.Client(base_url=served) as client:
        answer = client.post(protocol.REGISTER, content=body, headers=headers)
        assert answer.status_code == 200
        assert answer.json()["node"] == key.fingerprint
        assert client.post(protocol.REGISTER, content=body, headers=headers).status_code == 401
        assert client.get("/nothing").status_code == 404
        big = client.post(protocol.REGISTER, content=b"x" * (protocol.MAX_BODY + 1))
        assert big.status_code == 413


def test_cli_trust_sign_and_show(registry_dir: Path, root: sign.NodeKey, root_pub: Path) -> None:
    shown = registry_cli("trust", "show")
    assert shown.exit_code == 0 and "No trust list" in shown.output
    assert json.loads(registry_cli("trust", "show", "--json").output) is None
    _write_key(registry_dir / store.ROOT_KEY, root)
    assert registry_cli("revoke", sign.NodeKey.generate().fingerprint).exit_code == 0
    signed = registry_cli("trust", "sign", "--days", "3")
    assert signed.exit_code == 0, signed.output
    found = json.loads(registry_cli("trust", "show", "--json").output)
    assert found["records"] == 1
    assert "Trust list" in registry_cli("trust", "show").output
    assert registry_cli("trust", "sign", "--days", "31").exit_code == 2
    assert registry_cli("trust", "sign", "--days", "0").exit_code == 2


def test_trust_lists_are_append_only(db_path: Path, clock: FakeClock, root: sign.NodeKey) -> None:
    with store.connection(db_path) as conn:
        admin.sign_trust_list(conn, root, days=7, now=clock())
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute("DELETE FROM trust_lists")
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute("UPDATE trust_lists SET expires_at = 'x'")


def test_a_change_signs_a_list_that_is_newer_and_never_shorter(
    db_path: Path, clock: FakeClock, root: sign.NodeKey
) -> None:
    with store.connection(db_path) as conn:
        long = admin.sign_trust_list(conn, root, days=30, now=clock())
        admin.revoke(
            conn, root, sign.NodeKey.generate().fingerprint, "", now=clock()
        )  # same second
        found = admin.trust_list(conn, clock())
        assert found is not None and found.id != long.id
        assert found.issued_at > long.issued_at
        assert found.expires_at == long.expires_at  # the person's 30 days, not 7
        handed = store.handed_out(conn, clock())
    believed = sign.Trust(root.public, clock=clock)
    for signed in handed:
        believed.receive(signed.model_dump(mode="json"))
    assert believed.fresh() is None


def proposal_upload(number: int = 7) -> protocol.ProposalSubmission:
    run = protocol.BenchmarkSnapshot(
        version=1,
        mode="fake",
        source="/checkout/src/desman",
        cases=[protocol.BenchmarkCase(id="case", status="done", plan="llm", subtasks=1)],
        stats={
            "tasks": {"done": 1},
            "roles": [{"calls": 2, "prompt_tokens": 10, "completion_tokens": 5}],
        },
    )
    return protocol.ProposalSubmission(
        proposal=number,
        evaluation=3,
        base_commit="a" * 40,
        summary="Tighten a planning prompt.",
        details=protocol.ProposalDetails(
            changes=[
                protocol.ProposalChange(
                    target="planner.rules",
                    file="src/desman/planner.py",
                    old="old text",
                    new="new text",
                )
            ],
            base={"src/desman/planner.py": "b" * 64},
        ),
        diff="--- a/src/desman/planner.py\n+++ b/src/desman/planner.py\n",
        verdict="better",
        mode="fake",
        base=run,
        candidate=run,
    )


def signed_proposal(node: Node, submission: protocol.ProposalSubmission) -> dict[str, Any]:
    signed = protocol.SignedProposalSubmission(
        submission=submission,
        signer=node.fingerprint,
        signature=base64.b64encode(
            node.key.sign(sign.canonical(submission.model_dump(mode="json")))
        ).decode("ascii"),
    )
    return {"signed": signed.model_dump(mode="json")}


def test_a_registered_node_can_submit_a_proposal_once(
    registry: Registry, db_path: Path, clock: FakeClock
) -> None:
    node = Node(registry, clock)
    assert node.register().status_code == 200
    submission = proposal_upload()
    body = signed_proposal(node, submission)

    first = node.call("POST", protocol.PROPOSALS, body)
    again = node.call("POST", protocol.PROPOSALS, body)
    assert first.status_code == again.status_code == 200
    received = protocol.ProposalReceived.model_validate(first.json())
    assert protocol.ProposalReceived.model_validate(again.json()).id == received.id
    with store.connection(db_path) as conn:
        found = store.proposal(conn, received.id)
        assert found is not None
        assert found.node == node.fingerprint
        assert found.submission == submission
        assert found.status == "submitted"
        assert len(store.proposals(conn)) == 1
        assert store.entries(conn)[-1].action == "proposal_submitted"


def test_a_proposal_upload_cannot_be_replaced_or_submitted_by_an_unregistered_node(
    registry: Registry, db_path: Path, clock: FakeClock
) -> None:
    node = Node(registry, clock)
    submission = proposal_upload()
    body = signed_proposal(node, submission)
    assert node.call("POST", protocol.PROPOSALS, body).status_code == 403
    assert node.register().status_code == 200
    assert node.call("POST", protocol.PROPOSALS, body).status_code == 200
    changed = submission.model_copy(update={"summary": "different"})
    response = node.call("POST", protocol.PROPOSALS, signed_proposal(node, changed))
    assert response.status_code == 409
    with store.connection(db_path) as conn:
        assert len(store.proposals(conn)) == 1
        assert store.proposals(conn)[0].submission.summary == submission.summary


def test_a_proposal_body_must_match_its_node_signature(
    registry: Registry, clock: FakeClock
) -> None:
    node = Node(registry, clock)
    assert node.register().status_code == 200
    body = signed_proposal(node, proposal_upload())
    body["signed"]["submission"]["summary"] = "tampered after signing"
    response = node.call("POST", protocol.PROPOSALS, body)
    assert response.status_code == 401
    assert response.json()["error"] == Reason.FORGED


def test_a_node_submits_at_most_five_proposals_a_day(
    registry: Registry, db_path: Path, clock: FakeClock
) -> None:
    node = Node(registry, clock)
    assert node.register().status_code == 200
    bodies = [signed_proposal(node, proposal_upload(n)) for n in range(1, 8)]
    for body in bodies[: protocol.MAX_PROPOSALS_PER_DAY]:
        assert node.call("POST", protocol.PROPOSALS, body).status_code == 200
    over = node.call("POST", protocol.PROPOSALS, bodies[5])
    assert over.status_code == 403
    assert over.json()["error"] == protocol.PROPOSAL_LIMIT
    # A retry of one already kept is still answered: the person's command may have lost it.
    assert node.call("POST", protocol.PROPOSALS, bodies[0]).status_code == 200
    clock.advance(24 * 3600 + 1)
    assert node.call("POST", protocol.PROPOSALS, bodies[5]).status_code == 200
    with store.connection(db_path) as conn:
        assert len(store.proposals(conn)) == protocol.MAX_PROPOSALS_PER_DAY + 1
    assert ("refused", node.fingerprint) in log_actions(db_path)


def test_a_submission_signature_is_marked_as_one() -> None:
    signed = sign.canonical(proposal_upload().model_dump(mode="json"))
    assert json.loads(signed)["kind"] == protocol.PROPOSAL_KIND
    with pytest.raises(ValueError):
        protocol.ProposalSubmission.model_validate(
            proposal_upload().model_dump(mode="json") | {"kind": "result"}
        )


def test_a_submission_carries_texts_never_settings_or_evidence() -> None:
    data = proposal_upload().model_dump(mode="json")
    for change in (
        {"diff": None},
        {"details": data["details"] | {"settings": [{"setting": "x"}]}},
        {"details": data["details"] | {"evidence": ["a failed task said ..."]}},
    ):
        with pytest.raises(ValueError):
            protocol.ProposalSubmission.model_validate(data | change)


def test_the_operator_lists_and_shows_submissions(
    registry: Registry, db_path: Path, clock: FakeClock
) -> None:
    node = Node(registry, clock)
    assert node.register().status_code == 200
    reply = node.call("POST", protocol.PROPOSALS, signed_proposal(node, proposal_upload()))
    received = protocol.ProposalReceived.model_validate(reply.json())

    listed = registry_cli("proposal", "list", "--json")
    assert listed.exit_code == 0, listed.output
    [item] = json.loads(listed.output)
    assert item["id"] == received.id and item["node"] == node.fingerprint
    assert item["proposal"] == 7 and item["status"] == "submitted"
    assert (
        json.loads(registry_cli("proposal", "list", "--status", "accepted", "--json").output) == []
    )

    shown = registry_cli("proposal", "show", received.id[:8])
    assert shown.exit_code == 0, shown.output
    assert "Current version" in shown.output
    assert "Proposed version" in shown.output
    assert shown.output.count("case: done (subtasks 1)") == 2
    assert shown.output.count("Calls: 2") == 2
    assert "Prompt tokens: 10" in shown.output
    assert "Completion tokens: 5" in shown.output
    assert "Tighten a planning prompt." in shown.output
    assert "+++ b/src/desman/planner.py" in shown.output
    data = json.loads(registry_cli("proposal", "show", received.id, "--json").output)
    assert data["submission"]["details"]["changes"][0]["target"] == "planner.rules"

    missing = registry_cli("proposal", "show", "ffff")
    assert missing.exit_code != 0


def test_trust_show_warns_when_the_list_is_too_old(
    db_path: Path, root: sign.NodeKey, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    from desman.registry import cli as cli_module

    monkeypatch.setattr(cli_module, "_now", clock)
    with store.connection(db_path) as conn:
        admin.sign_trust_list(conn, root, days=7, now=clock())
    assert "more than 2 days old" not in registry_cli("trust", "show").output
    clock.advance(sign.TRUST_LIST_MAX_AGE.total_seconds())
    shown = registry_cli("trust", "show")
    assert shown.exit_code == 0, shown.output
    assert "more than 2 days old" in shown.output
    assert "nodes take no work" in shown.output
