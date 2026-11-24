"""The operator API uses real signatures and storage through MockTransport."""

from __future__ import annotations

import os
import stat
from datetime import timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
from click.testing import CliRunner

from conftest import FakeClock
from desman.cli import cli
from desman.config import open_config
from desman.core import db, netlog
from desman.net import operator, operator_records, sign
from desman.net import protocol as p
from desman.registry import admin, store, work
from desman.registry.server import Registry
from test_global_reports import signed_bug
from test_messenger import call, connect, named
from test_registry import Node, db_path, registry, registry_dir, root, root_pub
from test_registry_work import claim, hand_in, node

__all__ = ["db_path", "registry", "registry_dir", "root", "root_pub"]

URL = "https://registry.test"
TASKS = sign.Scope.TASKS
MODERATION = sign.Scope.MODERATION


def assigned(
    db_path: Path, root: sign.NodeKey, clock: FakeClock, *scopes: sign.Scope
) -> sign.NodeKey:
    key = sign.NodeKey.generate()
    with store.connection(db_path) as conn:
        admin.assign(conn, root, key.public, set(scopes), days=2, now=clock())
    return key


@pytest.fixture
def key(db_path: Path, root: sign.NodeKey, root_pub: Path, clock: FakeClock) -> sign.NodeKey:
    return assigned(db_path, root, clock, TASKS, MODERATION)


def client(registry: Registry, key: sign.NodeKey, clock: FakeClock) -> operator.OperatorClient:
    peer = Node(registry, clock, key)
    return operator.OperatorClient(
        URL, key, transport=httpx.MockTransport(peer._handle), clock=clock
    )


@pytest.mark.parametrize(
    "path,scope",
    [
        (p.OPERATOR_WHOAMI, None),
        (p.OPERATOR_JOB_ADD, TASKS),
        (p.OPERATOR_JOB_SETTLE, TASKS),
        (p.OPERATOR_CLUSTER_ADD, TASKS),
        (p.OPERATOR_CLUSTER_SETTLE, TASKS),
        (p.OPERATOR_OWNER, TASKS),
        (p.OPERATOR_COMPLAINTS, MODERATION),
        (p.OPERATOR_COMPLAINT, MODERATION),
        (p.OPERATOR_COMPLAINT_DISMISS, MODERATION),
        (p.OPERATOR_MODERATION, MODERATION),
        (p.OPERATOR_SUSPENSIONS, MODERATION),
    ],
)
@pytest.mark.parametrize("authority", ["absent", "expired", "revoked", "appoint", "wrong", "right"])
def test_operator_guard(
    registry: Registry,
    db_path: Path,
    root: sign.NodeKey,
    root_pub: Path,
    clock: FakeClock,
    path: str,
    scope: sign.Scope | None,
    authority: str,
) -> None:
    scopes = [scope or TASKS]
    if authority == "wrong":
        scopes = [TASKS if scope == MODERATION else MODERATION]
    if authority == "appoint":
        scopes = [sign.Scope.APPOINT]
    key = (
        sign.NodeKey.generate()
        if authority == "absent"
        else assigned(db_path, root, clock, *scopes)
    )
    if authority == "expired":
        clock.advance(2 * 86400)
    if authority == "revoked":
        clock.advance(1)
        with store.connection(db_path) as conn:
            admin.revoke(conn, root, key.fingerprint, "revoked", now=clock())
    peer = Node(registry, clock, key)
    reply = peer.call("POST", path, {})
    if authority in {"absent", "expired", "revoked", "appoint"}:
        assert (reply.status_code, reply.json()["error"]) == (403, "not_operator")
    elif authority == "wrong" and scope is not None:
        assert (reply.status_code, reply.json()["error"]) == (403, "scope")
    elif path in {p.OPERATOR_WHOAMI, p.OPERATOR_SUSPENSIONS}:
        assert reply.status_code == 200, reply.text
    else:
        # The right scope gets past authorization and reaches body validation.
        assert (reply.status_code, reply.json()["error"]) == (400, "malformed")
    if reply.status_code != 200:
        with store.connection(db_path) as conn:
            last = store.entries(conn, 1)[0]
        assert (last.action, last.node, last.details["reason"]) == (
            "refused",
            key.fingerprint,
            reply.json()["error"],
        )


def test_operator_polling_leaves_journal_unchanged(
    registry: Registry, db_path: Path, root: sign.NodeKey, root_pub: Path, clock: FakeClock
) -> None:
    for scope in (TASKS, MODERATION):
        remote = client(registry, assigned(db_path, root, clock, scope), clock)
        before = remote.log()
        for _ in range(3):
            assert remote.whoami().scopes == [scope]
            assert remote.inbox().jobs == []
        assert remote.jobs().jobs == []
        assert remote.clusters().clusters == []
        assert remote.nodes().nodes == []
        assert remote.reports().reports == []
        assert remote.log() == before
        with store.connection(db_path) as conn:
            assert not any(entry.action == "operator" for entry in store.entries(conn))


def test_newest_assignment_controls_access(
    registry: Registry, db_path: Path, root: sign.NodeKey, key: sign.NodeKey, clock: FakeClock
) -> None:
    remote = client(registry, key, clock)
    assert remote.whoami().scopes == [MODERATION, TASKS]
    clock.advance(1)
    with store.connection(db_path) as conn:
        admin.assign(conn, root, key.public, {TASKS}, days=1, now=clock())
    assert remote.whoami().scopes == [TASKS]
    clock.advance(86400)
    assert remote.whoami().scopes == [MODERATION, TASKS]
    clock.advance(86400)
    with pytest.raises(operator.OperatorError, match="not_operator"):
        remote.whoami()


def test_assignment_skips_future_and_forged_records(
    registry: Registry, db_path: Path, root: sign.NodeKey, key: sign.NodeKey, clock: FakeClock
) -> None:
    remote = client(registry, key, clock)
    with store.connection(db_path) as conn:
        future = sign.assign(
            root,
            key.public,
            {TASKS},
            issued_at=clock() + timedelta(seconds=2),
            lifetime=timedelta(days=1),
        )
        forged = sign.assign(
            root,
            key.public,
            {MODERATION},
            issued_at=clock() + timedelta(seconds=3),
            lifetime=timedelta(days=1),
        ).model_copy(update={"signature": future.signature})
        store.add_record(conn, future)
        store.add_record(conn, forged)
    assert remote.whoami().scopes == [MODERATION, TASKS]
    clock.advance(3)
    assert remote.whoami().scopes == [TASKS]


def test_routes_require_signed_post_json(
    registry: Registry, key: sign.NodeKey, clock: FakeClock
) -> None:
    peer = Node(registry, clock, key)
    path = p.OPERATOR_WHOAMI
    assert peer.call("GET", path).status_code == 405
    assert peer.call("POST", path).status_code == 400
    assert peer.call("POST", path, {"extra": 1}).status_code == 400
    assert peer.call("POST", path, {}).status_code == 200
    assert peer.replay().json()["error"] == "replay"


def test_remote_job_can_be_claimed_and_settled(
    registry: Registry, key: sign.NodeKey, clock: FakeClock
) -> None:
    remote = client(registry, key, clock)
    who = remote.whoami()
    job = p.Job(description="Sum 2 and 3", acceptance_criteria="Says 5")
    signed = operator_records.job(key, job, days=7, now=clock(), until=who.until)
    row = remote.job_add(signed)
    assert remote.jobs("execute").jobs == [row]
    assert row.job == signed.record.id
    assert row.task == job
    with pytest.raises(operator.OperatorError, match="replay"):
        remote.job_add(signed)

    a, b = node(registry, clock), node(registry, clock)
    lent = claim(a)
    assert lent is not None and sign.parse(lent["task"]) == signed
    assert hand_in(a, lent).status_code == 200
    check = claim(b)
    assert check is not None
    assert hand_in(b, check, accept=False, reason="disputed").status_code == 200
    assert [j.job for j in remote.inbox().jobs] == [row.job]
    detail = remote.job(row.job[:8])
    assert {item.outcome for item in detail.handed_in} == {"done", "reject"}
    assert remote.job_settle(row.job, "accept").status == "accepted"
    assert remote.inbox().jobs == []
    with pytest.raises(operator.OperatorError) as err:
        remote.job_settle(row.job, "accept")
    assert err.value.reason == "already_settled"
    assert (
        Node(registry, clock, key)
        .call("POST", p.OPERATOR_JOB_SETTLE, {"id": row.job, "decision": "accept"})
        .status_code
        == 409
    )
    assert any(
        entry.node == who.fingerprint
        and entry.details == {"path": p.OPERATOR_JOB_SETTLE, "id": row.job}
        for entry in remote.log().entries
    )


def test_cluster_and_owner(registry: Registry, key: sign.NodeKey, clock: FakeClock) -> None:
    remote = client(registry, key, clock)
    task = p.ClusterTask(goal="Review a design", acceptance_criteria="Give a reason")
    signed = operator_records.cluster_task(
        key, task, days=3, now=clock(), until=remote.whoami().until
    )
    row = remote.cluster_add(signed)
    assert remote.clusters("waiting").clusters[0].task == signed.record.id
    assert remote.cluster(row.task[:8]).goal == task
    assert remote.cluster_settle(row.task, "drop", note="internal text").status == "dropped"
    with pytest.raises(operator.OperatorError, match="already_settled"):
        remote.cluster_settle(row.task, "drop")
    n = node(registry, clock)
    assert remote.owner(n.fingerprint[:12], "test owner").owner == "test owner"
    assert remote.owner(n.fingerprint, None).owner is None
    assert remote.nodes().nodes[0].node == n.fingerprint
    actions = {
        entry.details.get("path") for entry in remote.log().entries if entry.node == key.fingerprint
    }
    assert {p.OPERATOR_CLUSTER_ADD, p.OPERATOR_CLUSTER_SETTLE, p.OPERATOR_OWNER} <= actions


def test_complaint_and_moderation(
    registry: Registry, db_path: Path, key: sign.NodeKey, clock: FakeClock
) -> None:
    remote = client(registry, key, clock)
    a, b = named(registry, clock, "alice"), named(registry, clock, "bobby")
    connect(a, b)
    sent = call(b, p.MESSENGER_SEND, to="alice", text="private message")
    reported = call(
        a, p.MESSENGER_COMPLAIN, name="bobby", text="private wording", message=sent["id"]
    )
    id = str(reported["id"])
    assert remote.complaints().complaints[0].id == id
    detail = remote.complaint(id[:8])
    assert detail.text == "private wording"
    assert detail.message_text == "private message"
    with store.connection(db_path) as conn:
        conn.execute(
            "INSERT INTO complaints SELECT ?,from_node,about_node,message_id,message_text,text,"
            "created_at,'open' FROM complaints WHERE id=?",
            (id[:8] + "a" * 24, id),
        )
    with pytest.raises(operator.OperatorError, match="ambiguous"):
        remote.complaint(id[:8])
    with pytest.raises(operator.OperatorError, match="not_found"):
        remote.complaint("no such complaint")
    assert remote.complaint_dismiss(id).id == id
    with pytest.raises(operator.OperatorError, match="already_settled"):
        remote.complaint_dismiss(id)

    until = remote.whoami().until
    suspended = remote.moderation(
        operator_records.moderation(
            key,
            p.Moderation(node=b.fingerprint, action="suspend", reason="moderation reason"),
            lifetime=timedelta(days=2),
            now=clock(),
            until=until,
        )
    )
    assert call(b, p.MESSENGER_SEND, to="alice", text="held")["error"] == p.SUSPENDED
    lifted = remote.moderation(
        operator_records.moderation(
            key,
            p.Moderation(node=b.fingerprint, action="lift", reason="resolved", lifts=suspended.id),
            lifetime=timedelta(days=2),
            now=clock(),
            until=until,
        )
    )
    assert "id" in call(b, p.MESSENGER_SEND, to="alice", text="allowed")
    assert {row.id: row.lifted for row in remote.suspensions().suspensions} == {
        lifted.id: False,
        suspended.id: True,
    }
    journal = remote.log()
    [read] = [entry for entry in journal.entries if entry.action == "operator_read_complaint"]
    assert (read.node, read.details) == (key.fingerprint, {"id": id})
    assert {
        (entry.details["path"], entry.details["id"])
        for entry in journal.entries
        if entry.action == "operator" and entry.node == key.fingerprint
    } == {
        (p.OPERATOR_COMPLAINT_DISMISS, id),
        (p.OPERATOR_MODERATION, suspended.id),
        (p.OPERATOR_MODERATION, lifted.id),
    }
    entries = journal.model_dump_json()
    assert all(
        text not in entries for text in ("private wording", "private message", "moderation reason")
    )


def test_local_key_files(tmp_path: Path) -> None:
    assert operator.load_key() is None
    source = tmp_path / "existing-operator.key"
    first = sign.create_node_key(source)
    assert operator.import_key(source) == first.fingerprint
    assert operator.import_key(source) == first.fingerprint
    path = operator.operator_key_path()
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    other_source = tmp_path / "other-operator.key"
    other = sign.create_node_key(other_source)
    with pytest.raises(FileExistsError):
        operator.import_key(other_source)
    with pytest.raises(FileExistsError):
        operator.create_key()
    assert operator.import_key(other_source, replace=True) == other.fingerprint
    with pytest.raises(sign.KeyFileError):
        bad_source = tmp_path / "bad.key"
        bad_source.write_text("invalid")
        operator.import_key(bad_source, replace=True)
    assert path.read_bytes() == other_source.read_bytes()
    operator.remove_key()
    assert operator.load_key() is None
    public = operator.create_key()
    assert sign.load_node_key(path).fingerprint == sign.fingerprint(admin.assignment_key(public))


def test_cli_status_import_create_and_remove(
    registry: Registry,
    key: sign.NodeKey,
    clock: FakeClock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = CliRunner()
    missing = runner.invoke(cli, ["operator", "status"])
    assert missing.exit_code == 0 and "desman operator key import FILE" in missing.output
    source = tmp_path / "registry.key"
    monkeypatch.setattr(operator, "operator_key_path", lambda: source)
    operator.install_key(key)
    installed = tmp_path / "installed.key"
    monkeypatch.setattr(operator, "operator_key_path", lambda: installed)
    result = runner.invoke(cli, ["operator", "key", "import", str(source), "--registry", URL])
    assert result.exit_code == 0, result.output
    assert key.fingerprint in result.output
    config = open_config().load()
    assert config.network.registry == URL and not config.network.enabled

    remote = client(registry, key, clock)

    def connect_operator(
        address: str, installed: sign.NodeKey, **kwargs: Any
    ) -> operator.OperatorClient:
        assert address == URL and installed.fingerprint == key.fingerprint
        return remote

    monkeypatch.setattr(operator, "OperatorClient", connect_operator)
    result = runner.invoke(cli, ["operator", "status"])
    assert result.exit_code == 0, result.output
    assert all(word in result.output for word in [key.fingerprint, URL, "moderation", "tasks"])
    assert sign.encode_public(key.public) in result.output.splitlines()
    assert "PRIVATE KEY" not in result.output
    assert runner.invoke(cli, ["operator", "key", "remove"]).exit_code == 0
    assert installed.is_file()
    assert runner.invoke(cli, ["operator", "key", "remove", "--yes"]).exit_code == 0
    assert not installed.exists()
    created = runner.invoke(cli, ["operator", "key", "create"])
    assert created.exit_code == 0, created.output
    generated = sign.load_node_key(installed)
    assert f"desman-registry assign {sign.encode_public(generated.public)}" in created.output
    with db.open_db() as conn:
        entries = [(e.action, e.details) for e in netlog.entries(conn)]
    assert entries == [
        ("operator_key_import", {"fingerprint": key.fingerprint}),
        ("operator_status", {"fingerprint": key.fingerprint}),
        ("operator_key_remove", {"fingerprint": key.fingerprint}),
        ("operator_key_create", {"fingerprint": generated.fingerprint}),
    ]


def test_reports_and_validation(registry: Registry, key: sign.NodeKey, clock: FakeClock) -> None:
    n = node(registry, clock)
    signed = signed_bug(n.key, "report text")
    response = n.call("POST", p.REPORTS, {"signed": signed.model_dump(mode="json")})
    assert response.status_code == 200
    id = response.json()["id"]
    remote = client(registry, key, clock)
    assert remote.reports("memory").reports == []
    [found] = remote.reports("bug").reports
    assert found.id == id and found.node == n.fingerprint
    assert remote.report(id).report == signed
    with pytest.raises(operator.OperatorError, match="not_found"):
        remote.report(id[:8])
    peer = Node(registry, clock, key)
    for limit in (0, 501, "1"):
        response = peer.call("POST", p.OPERATOR_LOG, {"limit": limit})
        assert response.status_code == 400 and response.json()["error"] == "malformed"
    malformed = sign.sign(
        key, sign.RecordKind.MODERATION, {}, issued_at=clock(), lifetime=timedelta(hours=1)
    )
    with pytest.raises(operator.OperatorError, match="malformed"):
        remote.moderation(malformed)


@pytest.mark.parametrize(
    "path,kind,max_days",
    [
        (p.OPERATOR_JOB_ADD, sign.RecordKind.TASK, admin.MAX_JOB_DAYS),
        (p.OPERATOR_CLUSTER_ADD, sign.RecordKind.CLUSTER_TASK, admin.CLUSTER_DAYS),
        (p.OPERATOR_MODERATION, sign.RecordKind.MODERATION, p.MAX_MODERATION_DAYS),
    ],
)
def test_record_term_limit(
    registry: Registry,
    db_path: Path,
    root: sign.NodeKey,
    root_pub: Path,
    clock: FakeClock,
    path: str,
    kind: sign.RecordKind,
    max_days: int,
) -> None:
    key = sign.NodeKey.generate()
    with store.connection(db_path) as conn:
        admin.assign(conn, root, key.public, {TASKS, MODERATION}, days=max_days + 2, now=clock())
    n = node(registry, clock)
    bodies = {
        sign.RecordKind.TASK: p.Job(description="Do work", acceptance_criteria="Done"),
        sign.RecordKind.CLUSTER_TASK: p.ClusterTask(goal="Do work", acceptance_criteria="Done"),
        sign.RecordKind.MODERATION: p.Moderation(
            node=n.fingerprint, action="suspend", reason="Test"
        ),
    }
    peer = Node(registry, clock, key)
    for seconds_over, expected in ((1, 400), (0, 200)):
        signed = sign.sign(
            key,
            kind,
            bodies[kind].model_dump(mode="json"),
            issued_at=clock(),
            lifetime=timedelta(days=max_days, seconds=seconds_over),
        )
        reply = peer.call("POST", path, {"record": signed.model_dump(mode="json")})
        assert reply.status_code == expected, reply.text
        if seconds_over:
            assert reply.json()["error"] == "term"
            with store.connection(db_path) as conn:
                entry = store.entries(conn, 1)[0]
            assert (entry.action, entry.node, entry.details["reason"]) == (
                "refused",
                key.fingerprint,
                "term",
            )


@pytest.mark.parametrize("owner", ["", "a" * 65])
def test_invalid_owner_length(
    registry: Registry, db_path: Path, key: sign.NodeKey, clock: FakeClock, owner: str
) -> None:
    n = node(registry, clock)
    peer = Node(registry, clock, key)
    reply = peer.call("POST", p.OPERATOR_OWNER, {"node": n.fingerprint, "owner": owner})
    assert (reply.status_code, reply.json()["error"]) == (400, "malformed")
    with store.connection(db_path) as conn, pytest.raises(admin.AdminError):
        admin.set_owner(conn, n.fingerprint, owner)
    assert client(registry, key, clock).owner(n.fingerprint, "a" * 64).owner == "a" * 64


def test_settlement_from_a_previous_inbox_round(
    registry: Registry, db_path: Path, key: sign.NodeKey, clock: FakeClock
) -> None:
    remote = client(registry, key, clock)
    task = p.Job(description="Do work", acceptance_criteria="Done")
    signed = operator_records.job(key, task, days=1, now=clock(), until=remote.whoami().until)
    id = remote.job_add(signed).job

    def return_without_result() -> None:
        with store.connection(db_path) as conn, store.write_transaction(conn):
            row = work.job(conn, id)
            assert row is not None and row.status == "execute"
            work.to_operator(conn, row, "failures", {})

    return_without_result()
    assert remote.job_settle(id, "again").status == "execute"
    with pytest.raises(operator.OperatorError, match="already_settled"):
        remote.job_settle(id, "accept")
    return_without_result()
    with pytest.raises(operator.OperatorError, match="cannot_settle"):
        remote.job_settle(id, "accept")
    assert remote.job_settle(id, "again").status == "execute"


def test_operator_record_signer_must_match_caller(
    registry: Registry, db_path: Path, root: sign.NodeKey, key: sign.NodeKey, clock: FakeClock
) -> None:
    signer = assigned(db_path, root, clock, TASKS)
    job = p.Job(description="Do work", acceptance_criteria="Done")
    signed = operator_records.job(signer, job, days=1, now=clock(), until=None)
    peer = Node(registry, clock, key)
    reply = peer.call("POST", p.OPERATOR_JOB_ADD, {"record": signed.model_dump(mode="json")})
    assert (reply.status_code, reply.json()["error"]) == (403, "wrong_signer")
    with store.connection(db_path) as conn:
        entry = store.entries(conn, 1)[0]
        assert (entry.action, entry.node, entry.details["reason"]) == (
            "refused",
            key.fingerprint,
            "wrong_signer",
        )
        assert work.job(conn, signed.record.id) is None


def test_automatic_acceptance_is_not_an_operator_settlement(
    registry: Registry, key: sign.NodeKey, clock: FakeClock
) -> None:
    remote = client(registry, key, clock)
    task = p.Job(description="Do work", acceptance_criteria="Done")
    signed = operator_records.job(key, task, days=1, now=clock(), until=remote.whoami().until)
    id = remote.job_add(signed).job
    executor, reviewer = node(registry, clock), node(registry, clock)
    lent = claim(executor)
    assert lent is not None and hand_in(executor, lent).status_code == 200
    check = claim(reviewer)
    assert check is not None and hand_in(reviewer, check).status_code == 200
    assert remote.job(id).status == "accepted"
    with pytest.raises(operator.OperatorError, match="cannot_settle"):
        remote.job_settle(id, "accept")
