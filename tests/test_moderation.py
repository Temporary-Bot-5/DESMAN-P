"""Messaging moderation is signed, reversible and never part of the trust feed."""

from __future__ import annotations

import json
import sqlite3
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner
from pydantic import ValidationError

from conftest import FakeClock
from desman.core import db
from desman.net import protocol as p
from desman.net import sign
from desman.registry import admin, moderation, store
from desman.registry.cli import cli
from desman.registry.server import Registry
from test_messenger import call, connect, named
from test_registry import db_path, registry, registry_dir, root, root_pub

__all__ = ["db_path", "registry", "registry_dir", "root", "root_pub"]


@pytest.fixture
def operator(db_path: Path, root: sign.NodeKey, root_pub: Path, clock: FakeClock) -> sign.NodeKey:
    key = sign.NodeKey.generate()
    with store.open_registry(db_path) as conn:
        admin.assign(conn, root, key.public, {sign.Scope.MODERATION}, days=30, now=clock())
    return key


def evidence(registry: Registry, clock: FakeClock) -> tuple[str, str, str]:
    a, b = named(registry, clock, "alice"), named(registry, clock, "bobby")
    connect(a, b)
    message = call(b, p.MESSENGER_SEND, to="alice", text="[bold]quoted message")["id"]
    id = call(a, p.MESSENGER_COMPLAIN, name="bobby", message=message, text="[red]complaint")["id"]
    return str(id), a.fingerprint, b.fingerprint


def test_complaints_and_dismiss(registry: Registry, db_path: Path, clock: FakeClock) -> None:
    id, a, b = evidence(registry, clock)
    with store.open_registry(db_path) as conn:
        [row] = moderation.complaints(conn, all=False)
        assert (row.from_node, row.about_node, row.from_name, row.about_name, row.count) == (
            a,
            b,
            "alice",
            "bobby",
            1,
        )
        found = moderation.complaint(conn, id[:8])
        assert found.text == "[red]complaint" and found.message_text == "[bold]quoted message"
        assert found.decisions == []
        moderation.dismiss(conn, id[:8], "no violation", clock())
        assert moderation.complaints(conn, all=False) == []
        assert len(moderation.complaints(conn, all=True)) == 1
        [decision] = moderation.complaint(conn, id).decisions
        assert (
            decision.decision == "dismissed"
            and decision.note == "no violation"
            and decision.record is None
        )
        with pytest.raises(admin.AdminError, match="already settled"):
            moderation.dismiss(conn, id, None, clock())
        with pytest.raises(admin.AdminError, match="No complaint"):
            moderation.find_complaint(conn, "unknown")
        # Make an ambiguous prefix without relying on random UUID collisions.
        conn.execute(
            "INSERT INTO complaints SELECT ?,from_node,about_node,message_id,message_text,text,created_at,'open' FROM complaints WHERE id=?",
            (id[:8] + "0" * 24, id),
        )
        with pytest.raises(admin.AdminError, match="More than one"):
            moderation.find_complaint(conn, id[:8])


def test_suspend_enforcement_and_lift(
    registry: Registry, db_path: Path, operator: sign.NodeKey, clock: FakeClock
) -> None:
    a, b = named(registry, clock, "alice"), named(registry, clock, "bobby")
    connect(a, b)
    id = call(a, p.MESSENGER_COMPLAIN, name="bobby", text="complaint")["id"]
    with store.open_registry(db_path) as conn:
        added = moderation.suspend(
            conn, operator, "bobby", "[bold]reason", days=5, complaints=[id], now=clock()
        )
        assert moderation.complaint(conn, id).decisions[0].record == added.id
        assert conn.execute("SELECT 1 FROM records WHERE id=?", (added.id,)).fetchone() is None
        for table in ["moderations", "complaint_decisions"]:
            for query in [
                f"UPDATE {table} SET {'id=id' if table == 'moderations' else 'note=note'}",
                f"DELETE FROM {table}",
            ]:
                with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                    conn.execute(query)
    for path, fields in [
        (p.MESSENGER_SEND, {"to": "alice", "text": "no"}),
        (p.MESSENGER_CONTACT, {"action": "add", "name": "alice"}),
        (p.MESSENGER_NAME, {"name": "new_name"}),
    ]:
        assert call(b, path, **fields)["error"] == p.SUSPENDED
    inbox = call(b, p.MESSENGER_INBOX)
    assert (
        inbox["suspended"]["record"] == added.id and inbox["suspended"]["reason"] == "[bold]reason"
    )
    assert "id" in call(a, p.MESSENGER_SEND, to="bobby", text="still received")
    assert "id" in call(b, p.MESSENGER_COMPLAIN, name="alice", text="allowed")
    for action in ["remove", "block", "unblock"]:
        assert "state" in call(b, p.MESSENGER_CONTACT, action=action, name="alice")
    with store.open_registry(db_path) as conn:
        refusals = [r for r in store.entries(conn) if r.action == "refused"]
        assert len([r for r in refusals if r.details.get("reason") == p.SUSPENDED]) == 3
        lifted = moderation.lift(conn, operator, added.id[:12], "resolved", now=clock())
        assert lifted.lifts == added.id
        assert moderation.suspension(conn, b.fingerprint, clock()) is None
        with pytest.raises(admin.AdminError, match="No suspension"):
            moderation.lift(conn, operator, "bobby", "again", now=clock())
        assert next(r for r in moderation.moderations(conn, now=clock()) if r.id == added.id).lifted
    assert call(b, p.MESSENGER_INBOX)["suspended"] is None
    assert call(b, p.MESSENGER_NAME, name="new_name")["name"] == "new_name"


@pytest.mark.parametrize(
    "bad", ["other", "settled", "unknown", "duplicate", "scope", "days", "reason"]
)
def test_invalid_suspend_is_atomic(
    registry: Registry,
    db_path: Path,
    operator: sign.NodeKey,
    root: sign.NodeKey,
    clock: FakeClock,
    bad: str,
) -> None:
    id, a, b = evidence(registry, clock)
    with store.open_registry(db_path) as conn:
        if bad == "settled":
            moderation.dismiss(conn, id, None, clock())
        if bad == "scope":
            operator = sign.NodeKey.generate()
            admin.assign(conn, root, operator.public, {sign.Scope.TASKS}, days=10, now=clock())
        before = [
            conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            for table in ["moderations", "complaint_decisions", "registry_log"]
        ]
        with pytest.raises(admin.AdminError):
            moderation.suspend(
                conn,
                operator,
                a if bad == "other" else b,
                "\x00" if bad == "reason" else "reason",
                days=0 if bad == "days" else 1,
                complaints=["unknown"]
                if bad == "unknown"
                else [id, id]
                if bad == "duplicate"
                else [id],
                now=clock(),
            )
        assert before == [
            conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            for table in ["moderations", "complaint_decisions", "registry_log"]
        ]


def test_term_revocation_and_latest(
    registry: Registry, db_path: Path, operator: sign.NodeKey, root: sign.NodeKey, clock: FakeClock
) -> None:
    b = named(registry, clock, "bobby")
    with store.open_registry(db_path) as conn:
        first = moderation.suspend(
            conn, operator, b.fingerprint[:12], "long", days=365, complaints=[], now=clock()
        )
        assert db.parse_iso(first.expires_at) == clock() + timedelta(days=30)
        clock.advance(1)
        second = moderation.suspend(
            conn, operator, "bobby", "short", days=1, complaints=[], now=clock()
        )
        assert moderation.suspension(conn, b.fingerprint, clock()).record == second.id  # type: ignore[union-attr]
        clock.advance(86401)
        assert moderation.suspension(conn, b.fingerprint, clock()).record == first.id  # type: ignore[union-attr]
        moderation.lift(conn, operator, "bobby", "lift long", now=clock())
        assert moderation.suspension(conn, b.fingerprint, clock()) is None
        moderation.suspend(conn, operator, "bobby", "new", days=2, complaints=[], now=clock())
        admin.revoke(conn, root, operator.fingerprint, "revoked", now=clock())
        assert moderation.suspension(conn, b.fingerprint, clock()) is None


def test_expiry(
    registry: Registry, db_path: Path, operator: sign.NodeKey, clock: FakeClock
) -> None:
    b = named(registry, clock, "bobby")
    with store.open_registry(db_path) as conn:
        moderation.suspend(conn, operator, "bobby", "reason", days=1, complaints=[], now=clock())
    clock.advance(86400)
    assert call(b, p.MESSENGER_INBOX)["suspended"] is None


def test_upgrade_nine_with_complaint(tmp_path: Path, clock: FakeClock) -> None:
    path = tmp_path / "old.db"
    conn = db.connect(path)
    db.migrate(conn, store.MIGRATIONS[:9])
    # Use the previous schema for real messenger data, before opening it with migration 10.
    for who, name in [("a" * 64, "alice"), ("b" * 64, "bobby")]:
        conn.execute(
            "INSERT INTO nodes(fingerprint,public_key,status,registered_at,last_seen,capabilities_json) VALUES (?,?,'active',?,?,'[]')",
            (
                who,
                sign.encode_public(sign.NodeKey.generate().public),
                db.to_iso(clock()),
                db.to_iso(clock()),
            ),
        )
        conn.execute("INSERT INTO messenger_names VALUES (?,?,?)", (who, name, db.to_iso(clock())))
    conn.execute(
        "INSERT INTO complaints(id,from_node,about_node,text,created_at) VALUES (?,?,?,?,?)",
        ("c" * 32, "a" * 64, "b" * 64, "kept", db.to_iso(clock())),
    )
    conn.close()
    conn = store.open_registry(path)
    try:
        assert moderation.complaint(conn, "c" * 8).text == "kept"
        moderation.dismiss(conn, "c" * 8, None, clock())
        assert moderation.complaint(conn, "c" * 8).decisions[0].decision == "dismissed"
    finally:
        conn.close()


def test_upgrade_ten_preserves_moderations(tmp_path: Path, clock: FakeClock) -> None:
    path = tmp_path / "ten.db"
    conn = db.connect(path)
    db.migrate(conn, store.MIGRATIONS[:10])
    values = ("s", "node", "suspend", None, "signer", db.to_iso(clock()), db.to_iso(clock()), "{}")
    conn.execute("INSERT INTO moderations VALUES (?,?,?,?,?,?,?,?)", values)
    conn.close()
    with store.open_registry(path) as conn:
        assert tuple(conn.execute("SELECT * FROM moderations").fetchone()) == values
        assert [row[2] for row in conn.execute("PRAGMA index_info(moderations_lifts)")] == ["lifts"]


def test_listing_builds_trust_once_and_writes_skip_listing(
    registry: Registry,
    db_path: Path,
    operator: sign.NodeKey,
    clock: FakeClock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    named(registry, clock, "bobby")
    with store.open_registry(db_path) as conn:
        original_list = moderation.moderations

        def unexpected_list(*args: Any, **kwargs: Any) -> list[moderation.ModerationRow]:
            pytest.fail("writing a moderation must not list history")

        monkeypatch.setattr(moderation, "moderations", unexpected_list)
        first = moderation.suspend(conn, operator, "bobby", "reason", complaints=[], now=clock())
        second = moderation.suspend(conn, operator, "bobby", "reason", complaints=[], now=clock())
        lifted = moderation.lift(conn, operator, first.id, "reason", now=clock())
        assert not first.lifted and not second.lifted and not lifted.lifted
        original_trust = store.trust
        calls = 0

        def counted_trust(*args: Any, **kwargs: Any) -> sign.Trust:
            nonlocal calls
            calls += 1
            return original_trust(*args, **kwargs)

        monkeypatch.setattr(store, "trust", counted_trust)
        rows = {row.id: row for row in original_list(conn, now=clock())}
        assert calls == 1
        assert rows[first.id].lifted
        assert not rows[second.id].lifted and not rows[lifted.id].lifted
        # Checking the same signed record twice must not consume its replay id.
        believed = original_trust(conn, sign.load_root(sign.ROOT_PUB), clock())
        row = conn.execute("SELECT * FROM moderations WHERE id=?", (lifted.id,)).fetchone()
        assert moderation._valid(row, believed, clock()) is not None
        assert moderation._valid(row, believed, clock()) is not None


def test_protocol() -> None:
    bodies: list[dict[str, Any]] = [
        {"action": "lift"},
        {"action": "suspend", "lifts": "a" * 32},
        {"action": "suspend", "reason": "\x1b"},
        {"action": "suspend", "complaints": ["a"] * 21},
    ]
    for body in bodies:
        with pytest.raises(ValidationError):
            p.Moderation.model_validate({"node": "a" * 64, "reason": "reason", **body})


def test_cli(
    registry: Registry,
    db_path: Path,
    registry_dir: Path,
    operator: sign.NodeKey,
    clock: FakeClock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    id, _, _ = evidence(registry, clock)
    monkeypatch.setattr("desman.registry.cli._now", clock)
    key_path = registry_dir / store.OPERATOR_KEY
    # Persist only this test's generated operator key using the existing key helper.
    from cryptography.hazmat.primitives import serialization

    key_path.write_bytes(
        operator._private.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    key_path.chmod(0o600)
    runner = CliRunner()

    def run(*args: str) -> Any:
        result = runner.invoke(cli, list(args))
        assert result.exit_code == 0, result.output
        return result

    assert json.loads(run("complaints", "--json").output)[0]["id"] == id
    shown = run("complaint", "show", id[:8]).output
    assert (
        "Quoted complaint" in shown
        and "[red]complaint" in shown
        and "[bold]quoted message" in shown
    )
    assert (
        json.loads(run("complaint", "show", id, "--json").output)["message_text"]
        == "[bold]quoted message"
    )
    run("moderation", "suspend", "bobby", "--reason", "reason", "--complaint", id[:8])
    rows = json.loads(run("moderation", "list", "--json").output)
    assert rows[0]["action"] == "suspend"
    run("moderation", "lift", rows[0]["id"][:8], "--reason", "resolved")
    assert json.loads(run("complaints", "--json").output) == []
    assert len(json.loads(run("complaints", "--all", "--json").output)) == 1
    other = named(registry, clock, "carol")
    # An independent complaint exercises the dismiss command.
    a = named(registry, clock, "alice_new")
    connect(a, other, "alice_new", "carol")
    new_id = call(a, p.MESSENGER_COMPLAIN, name="carol", text="complaint")["id"]
    run("complaint", "dismiss", new_id[:8], "--note", "reviewed")
    assert runner.invoke(cli, ["complaint", "show", "unknown"]).exit_code == 1


def test_assignment_expiry_and_explicit_server_root(
    registry: Registry,
    db_path: Path,
    operator: sign.NodeKey,
    root_pub: Path,
    clock: FakeClock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    b = named(registry, clock, "bobby")
    with store.open_registry(db_path) as conn:
        added = moderation.suspend(
            conn,
            operator,
            "bobby",
            "reason",
            days=365,
            complaints=[],
            now=clock(),
            root_pub=root_pub,
        )
    # A server's explicit root must win over the package default on every messenger route.
    other = root_pub.with_name("other.pub")
    other.write_bytes(sign.public_pem(sign.NodeKey.generate().public))
    monkeypatch.setattr(sign, "ROOT_PUB", other)
    server = Registry(db_path, clock=clock, root=root_pub)
    from test_registry import Node

    node = Node(server, clock, b.key)
    assert call(node, p.MESSENGER_INBOX)["suspended"]["record"] == added.id
    assert call(node, p.MESSENGER_NAME, name="new_name")["error"] == p.SUSPENDED
    clock.advance(30 * 86400)
    assert call(node, p.MESSENGER_INBOX)["suspended"] is None
    assert call(node, p.MESSENGER_NAME, name="new_name")["name"] == "new_name"


def test_citations_are_one_atomic_decision(
    registry: Registry, db_path: Path, operator: sign.NodeKey, clock: FakeClock
) -> None:
    id, _, _ = evidence(registry, clock)
    with store.open_registry(db_path) as conn:
        with pytest.raises(admin.AdminError, match="No complaint"):
            moderation.suspend(
                conn, operator, "bobby", "reason", days=1, complaints=[id, "missing"], now=clock()
            )
        assert moderation.complaint(conn, id).status == "open"
        assert moderation.complaint(conn, id).decisions == []
        assert moderation.moderations(conn) == []


def test_revoked_lift_restores_suspension(
    registry: Registry, db_path: Path, operator: sign.NodeKey, root: sign.NodeKey, clock: FakeClock
) -> None:
    b = named(registry, clock, "bobby")
    other = sign.NodeKey.generate()
    with store.open_registry(db_path) as conn:
        admin.assign(conn, root, other.public, {sign.Scope.MODERATION}, days=30, now=clock())
        added = moderation.suspend(
            conn, operator, "bobby", "reason", days=5, complaints=[], now=clock()
        )
        moderation.lift(conn, other, added.id, "resolved", now=clock())
        assert moderation.suspension(conn, b.fingerprint, clock()) is None
        assert next(r for r in moderation.moderations(conn, now=clock()) if r.id == added.id).lifted
        admin.revoke(conn, root, other.fingerprint, "revoked", now=clock())
        assert moderation.suspension(conn, b.fingerprint, clock()).record == added.id  # type: ignore[union-attr]
        assert not next(
            r for r in moderation.moderations(conn, now=clock()) if r.id == added.id
        ).lifted
        # A revoked lift must not prevent a currently authorized operator lifting it again.
        moderation.lift(conn, operator, added.id, "resolved again", now=clock())
        assert moderation.suspension(conn, b.fingerprint, clock()) is None


def test_lift_lasts_until_original_suspension_end(
    registry: Registry, db_path: Path, operator: sign.NodeKey, clock: FakeClock
) -> None:
    b = named(registry, clock, "bobby")
    with store.open_registry(db_path) as conn:
        added = moderation.suspend(
            conn, operator, "bobby", "reason", days=5, complaints=[], now=clock()
        )
        clock.advance(3600)
        lifted = moderation.lift(conn, operator, added.id, "resolved", now=clock())
        assert lifted.expires_at == added.expires_at
        clock.advance(2 * 86400)
        assert moderation.suspension(conn, b.fingerprint, clock()) is None
        assert next(r for r in moderation.moderations(conn, now=clock()) if r.id == added.id).lifted
        with pytest.raises(admin.AdminError, match="No suspension"):
            moderation.lift(conn, operator, added.id, "again", now=clock())
        clock.advance(3 * 86400 - 3601)
        assert moderation.suspension(conn, b.fingerprint, clock()) is None
        clock.advance(1)
        assert moderation.suspension(conn, b.fingerprint, clock()) is None
        assert not next(
            r for r in moderation.moderations(conn, now=clock()) if r.id == added.id
        ).lifted


def test_root_signed_suspension_counts(
    registry: Registry, db_path: Path, root: sign.NodeKey, root_pub: Path, clock: FakeClock
) -> None:
    named(registry, clock, "bobby")
    with store.open_registry(db_path) as conn:
        added = moderation.suspend(conn, root, "bobby", "by the root", complaints=[], now=clock())
        found = moderation.suspension(conn, added.node, clock(), root=root.public)
        assert found is not None and found.record == added.id
        moderation.lift(conn, root, added.id, "done", now=clock())
        assert moderation.suspension(conn, added.node, clock(), root=root.public) is None


def test_future_dated_suspension_waits_for_its_time(
    registry: Registry, db_path: Path, operator: sign.NodeKey, clock: FakeClock
) -> None:
    named(registry, clock, "bobby")
    with store.open_registry(db_path) as conn:
        ahead = clock() + timedelta(hours=1)
        added = moderation.suspend(conn, operator, "bobby", "later", complaints=[], now=ahead)
        root_key = sign.load_root(sign.ROOT_PUB)
        assert moderation.suspension(conn, added.node, clock(), root=root_key) is None
        assert moderation.suspension(conn, added.node, ahead, root=root_key) is not None
