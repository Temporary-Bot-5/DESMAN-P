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


def release_body() -> dict[str, Any]:
    return sign.Release(
        number=1,
        commit="a" * 40,
        parent="b" * 40,
        branch="master",
        proposal="c" * 32,
        submission="d" * 64,
        checks=["e" * 32],
        summary="Release",
        files=["src/desman/planner.py"],
    ).model_dump(mode="json")


@pytest.mark.parametrize("case", ["missing_commit", "extra", "zero", "proposal", "checks"])
def test_malformed_release_is_refused(
    trust: Trust, root: NodeKey, clock: FakeClock, case: str
) -> None:
    body = release_body()
    if case == "missing_commit":
        del body["commit"]
    elif case == "extra":
        body["extra"] = True
    elif case == "zero":
        body["number"] = 0
    elif case == "proposal":
        body["proposal"] = "bad"
    else:
        body["checks"] = ["bad"]
    signed = sign.sign(root, RecordKind.RELEASE, body, issued_at=clock(), lifetime=DAY)
    assert refused(trust, signed) is Reason.MALFORMED


@pytest.mark.parametrize("kind", [RecordKind.RELEASE, RecordKind.ASSIGNMENT])
def test_releases_and_assignments_are_the_roots_only(
    trust: Trust, root: NodeKey, clock: FakeClock, kind: RecordKind
) -> None:
    op = operator(trust, root, clock)
    other = NodeKey.generate()
    body = {"key": sign.encode_public(other.public), "scopes": ["tasks"]}
    if kind is RecordKind.RELEASE:
        body = release_body()
    record = sign.sign(op, kind, body, issued_at=clock(), lifetime=HOUR)
    assert refused(trust, record) is Reason.NOT_ALLOWED
    assert trust.operator(other.fingerprint) is None
    release = sign.sign(root, RecordKind.RELEASE, release_body(), issued_at=clock(), lifetime=DAY)
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


# ── Found in review ──────────────────────────────────────────────────────────


def test_a_failed_write_leaves_no_key_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / sign.NODE_KEY

    def broken(fd: int, data: bytes) -> int:
        raise OSError("disk full")

    with monkeypatch.context() as m:
        m.setattr(os, "write", broken)
        with pytest.raises(OSError, match="disk full"):
            sign.node_key(path)
    assert list(tmp_path.iterdir()) == []
    assert sign.node_key(path).fingerprint == sign.load_node_key(path).fingerprint


@pytest.mark.skipif(os.name == "nt", reason="POSIX links")
def test_a_node_key_behind_a_link_is_refused(tmp_path: Path) -> None:
    real = tmp_path / "elsewhere.key"
    sign.node_key(real)
    path = tmp_path / sign.NODE_KEY
    path.symlink_to(real)
    with pytest.raises(sign.KeyFileError, match="plain file"):
        sign.node_key(path)


def test_a_revoked_operators_assignment_received_again_is_refused(
    trust: Trust, root: NodeKey, clock: FakeClock
) -> None:
    op = NodeKey.generate()
    assignment = sign.assign(root, op.public, {Scope.TASKS}, issued_at=clock(), lifetime=DAY)
    trust.receive(assignment)
    trust.receive(sign.revoke(root, op.fingerprint, "", issued_at=clock(), lifetime=DAY))
    assert refused(trust, assignment) is Reason.REVOKED
    assert trust.operator(op.fingerprint) is None


def test_the_state_signed_by_a_revoked_operator_is_refused_again(
    trust: Trust, root: NodeKey, clock: FakeClock
) -> None:
    op = operator(trust, root, clock)
    node = NodeKey.generate()
    by_op = sign.revoke(op, node.fingerprint, "", issued_at=clock(), lifetime=DAY)
    trust.receive(by_op)
    trust.receive(sign.revoke(root, op.fingerprint, "", issued_at=clock(), lifetime=DAY))
    assert refused(trust, by_op) is Reason.REVOKED


def test_an_operator_cannot_revoke_an_operator_whose_assignment_comes_later(
    trust: Trust, root: NodeKey, clock: FakeClock
) -> None:
    op = operator(trust, root, clock)
    other = NodeKey.generate()
    early = sign.revoke(op, other.fingerprint, "coup", issued_at=clock(), lifetime=DAY)
    trust.receive(early)  # not an operator yet: as far as this node knows, a node
    assert trust.is_revoked(other.fingerprint)
    trust.receive(sign.assign(root, other.public, {Scope.TASKS}, issued_at=clock(), lifetime=DAY))
    assert not trust.is_revoked(other.fingerprint)
    assert trust.receive(task(other, clock)).signer == other.fingerprint
    assert refused(trust, early) is Reason.NOT_ALLOWED
    assert not trust.is_revoked(other.fingerprint)


@pytest.mark.parametrize("order", [(0, 1), (1, 0), (1, 0, 1, 0)])
def test_assignments_of_the_same_second_grant_only_what_both_allow(
    root: NodeKey, clock: FakeClock, order: tuple[int, ...]
) -> None:
    op = NodeKey.generate()
    both = {Scope.TASKS, Scope.MODERATION}
    records = [
        sign.assign(root, op.public, both, issued_at=clock(), lifetime=30 * DAY),
        sign.assign(root, op.public, {Scope.TASKS}, issued_at=clock(), lifetime=DAY),
    ]
    trust = Trust(root.public, clock=clock)
    for i in order:
        trust.receive(records[i])
    found = trust.operator(op.fingerprint)
    assert found is not None
    assert found.scopes == {Scope.TASKS}
    assert found.expires_at == clock() + DAY


def test_an_older_assignment_received_again_changes_nothing(
    trust: Trust, root: NodeKey, clock: FakeClock
) -> None:
    op = NodeKey.generate()
    both = {Scope.TASKS, Scope.MODERATION}
    old = sign.assign(root, op.public, both, issued_at=clock(), lifetime=DAY)
    trust.receive(old)
    clock.advance(1)
    trust.receive(sign.assign(root, op.public, {Scope.TASKS}, issued_at=clock(), lifetime=DAY))
    trust.receive(old)
    found = trust.operator(op.fingerprint)
    assert found is not None and found.scopes == {Scope.TASKS}


# ── The trust list (G.6c) ────────────────────────────────────────────────────


def listing(*signed: Signed) -> list[sign.Listed]:
    return [
        sign.Listed(
            id=s.record.id,
            until=None if s.record.kind is RecordKind.REVOCATION else s.record.expires_at,
        )
        for s in signed
    ]


def test_without_a_trust_list_nothing_is_fresh(trust: Trust) -> None:
    assert trust.fresh() == "no trust list received"


def test_a_list_naming_every_record_received_is_fresh_until_it_expires(
    trust: Trust, root: NodeKey, clock: FakeClock
) -> None:
    assigned = sign.assign(
        root, NodeKey.generate().public, {Scope.TASKS}, issued_at=clock(), lifetime=30 * DAY
    )
    revoked = sign.revoke(root, "ab" * 32, "lost", issued_at=clock(), lifetime=DAY)
    listed = sign.trust_list(root, listing(assigned, revoked), issued_at=clock(), lifetime=7 * DAY)
    for signed in (assigned, revoked, listed):
        trust.receive(wire(signed))
    assert trust.fresh() is None
    trust.receive(wire(listed))  # handed out again on every poll: no replay
    clock.advance((7 * DAY).total_seconds())
    assert trust.fresh() is not None and "expired" in (trust.fresh() or "")


def test_a_hidden_revocation_leaves_the_list_stale(
    trust: Trust, root: NodeKey, clock: FakeClock
) -> None:
    revoked = sign.revoke(root, "ab" * 32, "spam", issued_at=clock(), lifetime=DAY)
    listed = sign.trust_list(root, listing(revoked), issued_at=clock(), lifetime=7 * DAY)
    trust.receive(wire(listed))  # the registry kept the revocation to itself
    assert revoked.record.id in (trust.fresh() or "")
    clock.advance((DAY + HOUR).total_seconds())  # a revocation is needed past its own term
    assert trust.fresh() is not None
    trust.receive(wire(revoked))  # honoured past its term
    assert trust.fresh() is None


def test_an_assignment_past_its_term_is_no_longer_required(
    trust: Trust, root: NodeKey, clock: FakeClock
) -> None:
    assigned = sign.assign(
        root, NodeKey.generate().public, {Scope.TASKS}, issued_at=clock(), lifetime=DAY
    )
    listed = sign.trust_list(root, listing(assigned), issued_at=clock(), lifetime=7 * DAY)
    trust.receive(wire(listed))
    assert trust.fresh() is not None
    clock.advance((DAY - sign.SKEW).total_seconds())  # the registry stops handing it out
    assert trust.fresh() is None


def test_an_older_list_handed_out_again_does_not_replace_the_newest(
    trust: Trust, root: NodeKey, clock: FakeClock
) -> None:
    old = sign.trust_list(root, [], issued_at=clock(), lifetime=7 * DAY)
    clock.advance(HOUR.total_seconds())
    revoked = sign.revoke(root, "ab" * 32, "spam", issued_at=clock(), lifetime=DAY)
    new = sign.trust_list(root, listing(revoked), issued_at=clock(), lifetime=7 * DAY)
    trust.receive(wire(new))
    trust.receive(wire(old))  # a registry rolling back to before the revocation
    assert revoked.record.id in (trust.fresh() or "")


def test_only_the_root_signs_a_trust_list(trust: Trust, root: NodeKey, clock: FakeClock) -> None:
    signer = operator(trust, root, clock)
    forged = sign.trust_list(signer, [], issued_at=clock(), lifetime=7 * DAY)
    assert refused(trust, wire(forged)) is Reason.NOT_ALLOWED
    stranger = sign.trust_list(NodeKey.generate(), [], issued_at=clock(), lifetime=7 * DAY)
    assert refused(trust, wire(stranger)) is Reason.UNKNOWN_SIGNER
    assert trust.fresh() == "no trust list received"


def test_a_list_in_term_must_be_younger_than_two_days(
    trust: Trust, root: NodeKey, clock: FakeClock
) -> None:
    trust.receive(sign.trust_list(root, [], issued_at=clock(), lifetime=7 * DAY))
    clock.advance(sign.TRUST_LIST_MAX_AGE.total_seconds() - 1)
    assert trust.fresh() is None
    clock.advance(1)
    assert "more than 2 days ago" in (trust.fresh() or "")
    trust.receive(sign.trust_list(root, [], issued_at=clock(), lifetime=7 * DAY))
    assert trust.fresh() is None
