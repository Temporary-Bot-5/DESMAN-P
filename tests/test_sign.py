"""Tests for the signatures of desman-global (ROADMAP G.2, ADR 0004 decision 3)."""

from __future__ import annotations

import json
import os
import stat
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from conftest import FakeClock
from desman.net import sign
from desman.net.sign import (
    NodeKey,
    Reason,
    RecordKind,
    Refused,
    Scope,
    Signed,
    Trust,
)

DAY = timedelta(days=1)
HOUR = timedelta(hours=1)


@pytest.fixture
def root() -> NodeKey:
    return NodeKey.generate()


@pytest.fixture
def trust(root: NodeKey, clock: FakeClock) -> Trust:
    return Trust(root.public, clock=clock)


def task(signer: NodeKey, clock: FakeClock, **kw: Any) -> Signed:
    kw.setdefault("lifetime", HOUR)
    return sign.sign(signer, RecordKind.TASK, {"goal": "sum"}, issued_at=clock(), **kw)


def operator(
    trust: Trust, root: NodeKey, clock: FakeClock, scopes: set[Scope] | None = None
) -> NodeKey:
    key = NodeKey.generate()
    scopes = {Scope.TASKS, Scope.MODERATION} if scopes is None else scopes
    trust.receive(sign.assign(root, key.public, scopes, issued_at=clock(), lifetime=30 * DAY))
    return key


def refused(trust: Trust, data: object) -> Reason:
    with pytest.raises(Refused) as e:
        trust.receive(data)
    return e.value.reason


def wire(signed: Signed) -> str:
    return json.dumps(signed.model_dump(mode="json"))


# ── Canonical form and records ───────────────────────────────────────────────


def test_canonical_json_is_sorted_compact_utf8() -> None:
    assert sign.canonical({"b": 1, "a": ["é", {"d": None, "c": True}]}) == (
        '{"a":["é",{"c":true,"d":null}],"b":1}'.encode()
    )
    with pytest.raises(ValueError):
        sign.canonical({"x": float("nan")})


def test_a_record_signed_by_the_root_is_accepted_from_its_wire_form(
    trust: Trust, root: NodeKey, clock: FakeClock
) -> None:
    signed = task(root, clock)
    record = trust.receive(wire(signed))
    assert record == signed.record
    assert record.signer == root.fingerprint
    assert record.body == {"goal": "sum"}


def test_a_record_must_have_every_field_and_no_other() -> None:
    root = NodeKey.generate()
    data = json.loads(wire(task(root, FakeClock())))
    for missing in ("kind", "id", "issued_at", "expires_at", "signer"):
        broken = json.loads(json.dumps(data))
        del broken["record"][missing]
        with pytest.raises(Refused) as e:
            sign.parse(broken)
        assert e.value.reason is Reason.MALFORMED
    data["record"]["extra"] = 1
    with pytest.raises(Refused):
        sign.parse(data)
    with pytest.raises(Refused):
        sign.parse("not json")


def test_a_record_ends_after_it_begins(root: NodeKey, clock: FakeClock) -> None:
    with pytest.raises(ValueError):
        sign.sign(root, RecordKind.TASK, {}, issued_at=clock(), lifetime=timedelta(0))


# ── Forgery ──────────────────────────────────────────────────────────────────


def test_a_changed_body_is_forged(trust: Trust, root: NodeKey, clock: FakeClock) -> None:
    data = json.loads(wire(task(root, clock)))
    data["record"]["body"]["goal"] = "rm -rf"
    assert refused(trust, data) is Reason.FORGED


def test_a_changed_term_is_forged(trust: Trust, root: NodeKey, clock: FakeClock) -> None:
    data = json.loads(wire(task(root, clock)))
    data["record"]["expires_at"] = "2030-01-01T00:00:00Z"
    assert refused(trust, data) is Reason.FORGED


def test_a_signature_by_another_key_under_the_roots_name_is_forged(
    trust: Trust, root: NodeKey, clock: FakeClock
) -> None:
    mallory = NodeKey.generate()
    fake = task(mallory, clock)
    data = fake.model_dump(mode="json")
    data["record"]["signer"] = root.fingerprint
    assert refused(trust, data) is Reason.FORGED


def test_a_record_signed_by_a_foreign_root_is_refused(trust: Trust, clock: FakeClock) -> None:
    other_root = NodeKey.generate()
    assert refused(trust, task(other_root, clock)) is Reason.UNKNOWN_SIGNER
    # Nor can it name operators of its own.
    op = NodeKey.generate()
    assignment = sign.assign(other_root, op.public, {Scope.TASKS}, issued_at=clock(), lifetime=DAY)
    assert refused(trust, assignment) is Reason.UNKNOWN_SIGNER
    assert refused(trust, task(op, clock)) is Reason.UNKNOWN_SIGNER


# ── Operators ────────────────────────────────────────────────────────────────


def test_an_assigned_operator_signs_within_its_scopes(
    trust: Trust, root: NodeKey, clock: FakeClock
) -> None:
    op = operator(trust, root, clock, {Scope.TASKS})
    assert trust.receive(task(op, clock)).signer == op.fingerprint
    moderation = sign.sign(op, RecordKind.MODERATION, {}, issued_at=clock(), lifetime=HOUR)
    assert refused(trust, moderation) is Reason.NOT_ALLOWED
    node = NodeKey.generate()
    revocation = sign.revoke(op, node.fingerprint, "spam", issued_at=clock(), lifetime=HOUR)
    assert refused(trust, revocation) is Reason.NOT_ALLOWED


@pytest.mark.parametrize("kind", [RecordKind.RELEASE, RecordKind.ASSIGNMENT])
def test_releases_and_assignments_are_the_roots_only(
    trust: Trust, root: NodeKey, clock: FakeClock, kind: RecordKind
) -> None:
    op = operator(trust, root, clock)
    other = NodeKey.generate()
    body = {"key": sign.encode_public(other.public), "scopes": ["tasks"]}
    record = sign.sign(op, kind, body, issued_at=clock(), lifetime=HOUR)
    assert refused(trust, record) is Reason.NOT_ALLOWED
    assert trust.operator(other.fingerprint) is None
    release = sign.sign(
        root, RecordKind.RELEASE, {"commit": "abc"}, issued_at=clock(), lifetime=DAY
    )
    assert trust.receive(release).kind is RecordKind.RELEASE


def test_an_expired_assignment_signs_nothing(trust: Trust, root: NodeKey, clock: FakeClock) -> None:
    op = operator(trust, root, clock)
    clock.advance((30 * DAY).total_seconds() - 60)
    late = task(op, clock, lifetime=DAY)  # issued within the term, still running after it
    clock.advance(120)
    assert refused(trust, late) is Reason.EXPIRED
    assert refused(trust, task(op, clock)) is Reason.NOT_ALLOWED


def test_a_newer_assignment_extends_the_term(trust: Trust, root: NodeKey, clock: FakeClock) -> None:
    op = operator(trust, root, clock, {Scope.TASKS})
    clock.advance(DAY.total_seconds())
    renewed = sign.assign(root, op.public, {Scope.TASKS}, issued_at=clock(), lifetime=90 * DAY)
    trust.receive(renewed)
    clock.advance((60 * DAY).total_seconds())
    assert trust.receive(task(op, clock)).signer == op.fingerprint


def test_an_assignment_must_carry_a_key_and_a_scope(
    trust: Trust, root: NodeKey, clock: FakeClock
) -> None:
    for body in ({"key": "nope", "scopes": ["tasks"]}, {"key": "A" * 43 + "=", "scopes": []}):
        record = sign.sign(root, RecordKind.ASSIGNMENT, body, issued_at=clock(), lifetime=DAY)
        assert refused(trust, record) is Reason.MALFORMED
    rooted = sign.assign(root, root.public, {Scope.TASKS}, issued_at=clock(), lifetime=DAY)
    assert refused(trust, rooted) is Reason.NOT_ALLOWED


# ── Revocations ──────────────────────────────────────────────────────────────


def test_a_revoked_operator_breaks_the_chain(trust: Trust, root: NodeKey, clock: FakeClock) -> None:
    op = operator(trust, root, clock)
    pending = task(op, clock)  # signed before the revocation, delivered after it
    trust.receive(sign.revoke(root, op.fingerprint, "stolen", issued_at=clock(), lifetime=DAY))
    assert trust.is_revoked(op.fingerprint)
    assert refused(trust, pending) is Reason.REVOKED
    assert refused(trust, task(op, clock)) is Reason.REVOKED
    # Assigning the key again does not bring it back.
    again = sign.assign(root, op.public, {Scope.TASKS}, issued_at=clock(), lifetime=DAY)
    assert refused(trust, again) is Reason.REVOKED


def test_an_operator_revokes_nodes_but_not_operators_nor_the_root(
    trust: Trust, root: NodeKey, clock: FakeClock
) -> None:
    op = operator(trust, root, clock)
    other = operator(trust, root, clock)
    node = NodeKey.generate()
    trust.receive(sign.revoke(op, node.fingerprint, "abuse", issued_at=clock(), lifetime=DAY))
    assert trust.is_revoked(node.fingerprint)
    for target in (other.fingerprint, root.fingerprint):
        record = sign.revoke(op, target, "coup", issued_at=clock(), lifetime=DAY)
        assert refused(trust, record) is Reason.NOT_ALLOWED
    by_root = sign.revoke(root, root.fingerprint, "", issued_at=clock(), lifetime=DAY)
    assert refused(trust, by_root) is Reason.NOT_ALLOWED
    assert not trust.is_revoked(other.fingerprint)
    assert not trust.is_revoked(root.fingerprint)


def test_a_revocation_holds_after_its_own_expiry(
    trust: Trust, root: NodeKey, clock: FakeClock
) -> None:
    op = operator(trust, root, clock)
    revocation = sign.revoke(root, op.fingerprint, "", issued_at=clock(), lifetime=HOUR)
    clock.advance(DAY.total_seconds())
    trust.receive(revocation)
    assert refused(trust, task(op, clock)) is Reason.REVOKED


# ── Time and replay ──────────────────────────────────────────────────────────


def test_an_expired_record_is_refused(trust: Trust, root: NodeKey, clock: FakeClock) -> None:
    signed = task(root, clock)
    clock.advance(HOUR.total_seconds())
    assert refused(trust, signed) is Reason.EXPIRED


def test_a_record_from_the_future_is_refused(trust: Trust, root: NodeKey, clock: FakeClock) -> None:
    ahead = sign.sign(root, RecordKind.TASK, {}, issued_at=clock() + HOUR, lifetime=HOUR)
    assert refused(trust, ahead) is Reason.NOT_YET
    slightly = sign.sign(root, RecordKind.TASK, {}, issued_at=clock() + sign.SKEW, lifetime=HOUR)
    assert trust.receive(slightly).id == slightly.record.id


def test_a_record_is_accepted_once(trust: Trust, root: NodeKey, clock: FakeClock) -> None:
    signed = task(root, clock)
    trust.receive(signed)
    assert refused(trust, wire(signed)) is Reason.REPLAY
    clock.advance(60)
    assert refused(trust, signed) is Reason.REPLAY


def test_the_same_state_again_is_not_a_replay(
    trust: Trust, root: NodeKey, clock: FakeClock
) -> None:
    op = NodeKey.generate()
    assignment = sign.assign(root, op.public, {Scope.TASKS}, issued_at=clock(), lifetime=DAY)
    revocation = sign.revoke(
        root, NodeKey.generate().fingerprint, "", issued_at=clock(), lifetime=DAY
    )
    for _ in range(3):  # every poll hands them out again
        trust.receive(wire(assignment))
        trust.receive(wire(revocation))
    other = sign.revoke(root, op.fingerprint, "", issued_at=clock(), lifetime=DAY)
    clash = other.model_dump(mode="json")
    clash["record"]["id"] = revocation.record.id
    assert refused(trust, clash) is Reason.CONFLICT
    assert not trust.is_revoked(op.fingerprint)


def test_seen_ids_are_forgotten_once_they_expired() -> None:
    seen = sign.MemorySeen()
    clock = FakeClock()
    assert seen.add("a" * 32, clock() + HOUR, clock())
    assert not seen.add("a" * 32, clock() + HOUR, clock())
    clock.advance(HOUR.total_seconds())
    assert seen.add("b" * 32, clock() + HOUR, clock())
    assert seen._ids.keys() == {"b" * 32}


# ── The node's key ───────────────────────────────────────────────────────────


@pytest.mark.skipif(os.name == "nt", reason="POSIX permissions")
def test_the_node_key_is_created_once_readable_by_its_owner_only(tmp_path: Path) -> None:
    path = tmp_path / "data" / sign.NODE_KEY
    key = sign.node_key(path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert sign.node_key(path).fingerprint == key.fingerprint
    assert sign.load_node_key(path).fingerprint == key.fingerprint
    with pytest.raises(FileExistsError):
        sign.create_node_key(path)


@pytest.mark.skipif(os.name == "nt", reason="POSIX permissions")
def test_a_node_key_others_may_read_is_refused(tmp_path: Path) -> None:
    path = tmp_path / sign.NODE_KEY
    sign.node_key(path)
    path.chmod(0o644)
    with pytest.raises(sign.KeyFileError, match="chmod 600"):
        sign.node_key(path)


def test_a_node_key_file_must_hold_an_ed25519_key(tmp_path: Path) -> None:
    path = tmp_path / sign.NODE_KEY
    path.write_bytes(b"garbage")
    path.chmod(0o600)
    with pytest.raises(sign.KeyFileError):
        sign.load_node_key(path)


def test_the_node_key_lives_next_to_the_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DESMAN_DATA_DIR", str(tmp_path))
    assert sign.default_node_key_path() == tmp_path / sign.NODE_KEY


def test_a_key_shows_its_fingerprint_never_itself(tmp_path: Path) -> None:
    key = sign.node_key(tmp_path / sign.NODE_KEY)
    secret = (tmp_path / sign.NODE_KEY).read_text()
    body = "".join(line for line in secret.splitlines() if not line.startswith("-----"))
    for shown in (repr(key), str(key), f"{key!r}"):
        assert key.fingerprint in shown
        assert body not in shown
    assert len(key.fingerprint) == 64


def test_a_public_key_round_trips(tmp_path: Path) -> None:
    key = NodeKey.generate()
    assert sign.fingerprint(sign.decode_public(sign.encode_public(key.public))) == key.fingerprint
    path = tmp_path / "root.pub"
    path.write_bytes(sign.public_pem(key.public))
    assert sign.fingerprint(sign.load_root(path)) == key.fingerprint
    with pytest.raises(sign.KeyFileError, match="missing"):
        sign.load_root(tmp_path / "none.pub")
    with pytest.raises(sign.KeyFileError):
        sign.decode_public("not base64!")


def test_check_signature_names_the_key_it_expects(clock: FakeClock) -> None:
    node, other = NodeKey.generate(), NodeKey.generate()
    signed = task(node, clock)
    sign.check_signature(signed, node.public)
    with pytest.raises(Refused) as e:
        sign.check_signature(signed, other.public)
    assert e.value.reason is Reason.UNKNOWN_SIGNER


def test_the_shipped_root_is_an_ed25519_key() -> None:
    assert len(sign.fingerprint(sign.load_root())) == 64
