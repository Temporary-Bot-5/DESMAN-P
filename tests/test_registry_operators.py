"""Root-side operator activity and revocation (ROADMAP O.4)."""

from __future__ import annotations

import json
import sqlite3
from datetime import timedelta
from pathlib import Path

import pytest

from conftest import FakeClock
from desman.core import db
from desman.net import sign
from desman.registry import admin, cli, clusters, operators, store, work
from desman.registry.server import Registry
from test_registry import _write_key, db_path, registry, registry_cli, registry_dir, root, root_pub
from test_registry_clusters import add_task, appointer, formed, job, mine, plan, say, send
from test_registry_work import add_job, claim, hand_in, node, operator

__all__ = ["appointer", "db_path", "operator", "registry", "registry_dir", "root", "root_pub"]


@pytest.fixture(autouse=True)
def fixed_time(monkeypatch: pytest.MonkeyPatch, clock: FakeClock) -> None:
    monkeypatch.setattr(cli, "_now", clock)


def test_list_operator_states(
    registry_dir: Path, db_path: Path, root: sign.NodeKey, root_pub: Path, clock: FakeClock
) -> None:
    _write_key(registry_dir / store.ROOT_KEY, root)
    assert "No operator keys" in registry_cli("operators").output
    assert json.loads(registry_cli("operators", "--json").output) == []
    assert registry_cli("assign").exit_code == 0
    own = sign.load_node_key(registry_dir / store.OPERATOR_KEY)
    expired, future, revoked, appoint, invalid = (sign.NodeKey.generate() for _ in range(5))
    with store.connection(db_path) as conn:
        for key, scope, when in (
            (expired, sign.Scope.TASKS, clock() - timedelta(days=2)),
            (future, sign.Scope.TASKS, clock() + timedelta(days=2)),
            (revoked, sign.Scope.TASKS, clock()),
            (appoint, sign.Scope.APPOINT, clock()),
        ):
            store.add_record(
                conn,
                sign.assign(root, key.public, {scope}, issued_at=when, lifetime=timedelta(days=1)),
            )
        store.add_record(
            conn,
            sign.revoke(
                root, revoked.fingerprint, "gone", issued_at=clock(), lifetime=timedelta(days=1)
            ),
        )
        store.add_record(
            conn,
            sign.assign(
                sign.NodeKey.generate(),
                invalid.public,
                {sign.Scope.TASKS},
                issued_at=clock(),
                lifetime=timedelta(days=1),
            ),
        )
        store.log(conn, "operator", own.fingerprint)
        store.log(conn, "operator", own.fingerprint)
        store.log(conn, "refused", own.fingerprint)
        store.log(conn, "operator_read_complaint", own.fingerprint)
    result = registry_cli("operators", "--json")
    assert result.exit_code == 0, result.exception
    rows = {row["fingerprint"]: row for row in json.loads(result.output)}
    assert {key: row["state"] for key, row in rows.items()} == {
        own.fingerprint: "active",
        expired.fingerprint: "expired",
        future.fingerprint: "not_yet",
        revoked.fingerprint: "revoked",
        appoint.fingerprint: "active",
        invalid.fingerprint: "invalid",
    }
    assert rows[appoint.fingerprint]["appointer"]
    assert rows[appoint.fingerprint]["scopes"] == ["appoint"]
    row = rows[own.fingerprint]
    assert row["registry_key"] and not row["appointer"]
    assert (row["writes"], row["refusals"], row["complaint_reads"]) == (2, 1, 1)
    assert row["last_request"]["action"] == "operator_read_complaint"
    assert registry_cli("operators").exit_code == 0


def test_show_activity(
    registry: Registry,
    db_path: Path,
    operator: sign.NodeKey,
    appointer: sign.NodeKey,
    root: sign.NodeKey,
    clock: FakeClock,
) -> None:
    single = add_job(db_path, operator, clock)
    worker = node(registry, clock)
    offer = claim(worker)
    assert offer is not None and hand_in(worker, offer).status_code == 200
    reviewer = node(registry, clock)
    offer = claim(reviewer)
    assert offer is not None and hand_in(reviewer, offer).status_code == 200
    task, lead, member, spare = formed(registry, db_path, operator, clock)
    state = mine(lead)
    planned = plan(lead, state)
    assert send(lead, planned).status_code == 200
    signed = job(lead, state, planned.record.id, 0)
    assert send(lead, signed).status_code == 200
    leased = None
    for candidate in (member, spare, worker, reviewer):
        leased = say(candidate)
        if leased is not None:
            break
    assert leased is not None
    dropped = add_job(db_path, operator, clock)
    with store.connection(db_path) as conn:
        work.set_owner(conn, worker.fingerprint, "Alice")
        waiting = work.job(conn, dropped)
        assert waiting is not None
        work.to_operator(conn, waiting, "failed", {})
        work.settle(conn, dropped, "drop", clock())
        store.log(conn, "operator", appointer.fingerprint, {"id": dropped})
        clusters.settle(conn, task, "drop", clock(), note="private settlement note")
        clock.advance(1)
        admin.assign(conn, root, operator.public, {sign.Scope.TASKS}, days=20, now=clock())
        admin.revoke(conn, root, operator.fingerprint, "retired", now=clock())

    prefix = operator.fingerprint[:8].upper()
    output = registry_cli("operators", "show", prefix, "--json")
    assert output.exit_code == 0, output.exception
    report = json.loads(output.output)
    summary = report["summary"]
    assert summary["fingerprint"] == operator.fingerprint
    assert (summary["jobs"], summary["cluster_tasks"]) == (2, 1)
    assert [r["kind"] for r in report["records"]] == ["revocation", "assignment", "assignment"]
    assert report["records"][0]["reason"] == "retired"
    accepted, stopped = report["jobs"]
    assert accepted["status"] == "accepted" and stopped["status"] == "dropped"
    assert (accepted["count"], accepted["leases"], accepted["results"], accepted["verdicts"]) == (
        1,
        2,
        1,
        1,
    )
    assert {n["fingerprint"]: (n["owner"], n["leases"]) for n in accepted["nodes"]} == {
        worker.fingerprint: ("Alice", 1),
        reviewer.fingerprint: (None, 1),
    }
    cluster = report["cluster_tasks"][0]
    assert (cluster["status"], cluster["count"], cluster["leases"]) == ("dropped", 1, 1)
    assert cluster["nodes"][0]["fingerprint"] == candidate.fingerprint
    actions = {entry["action"]: entry["details"] for entry in report["journal"]}
    assert actions["job_accepted"]["job"] == single
    assert actions["job_settled"]["job"] == dropped
    assert actions["job_to_operator"]["job"] == dropped
    assert actions["operator"]["id"] == dropped
    assert actions["cluster_settled"]["task"] == task
    assert actions["cluster_job_dropped"]["job"] == signed.record.id
    assert "private settlement note" not in output.output
    latest = registry_cli("operators", "show", prefix, "--limit", "1", "--json")
    assert json.loads(latest.output)["journal"] == report["journal"][:1]
    terminal = registry_cli("operators", "show", prefix)
    assert terminal.exit_code == 0, terminal.exception
    assert operator.fingerprint in terminal.output
    assert "token counts" in terminal.output


def test_bad_and_ambiguous_operator_prefix(
    db_path: Path, operator: sign.NodeKey, clock: FakeClock
) -> None:
    assert "8 to 64" in registry_cli("operators", "show", "short").output
    assert "No operator matches" in registry_cli("operators", "show", "0" * 64).output
    prefix = operator.fingerprint[:8]
    different = "e" if operator.fingerprint[8] == "f" else "f"
    # Simulate a collision in the indexed subjects without searching for a matching Ed25519 key.
    with store.connection(db_path) as conn:
        conn.execute(
            """INSERT INTO records (id, kind, subject, signer, issued_at, expires_at, signed_json)
            SELECT ?, kind, ?, signer, issued_at, expires_at, signed_json FROM records
            WHERE subject = ?""",
            ("a" * 32, prefix + different + "f" * 55, operator.fingerprint),
        )
    assert (
        "More than one operator"
        in registry_cli("operators", "show", operator.fingerprint[:8]).output
    )


def test_revoke_all_open_work(
    registry: Registry,
    db_path: Path,
    operator: sign.NodeKey,
    appointer: sign.NodeKey,
    root: sign.NodeKey,
    registry_dir: Path,
    clock: FakeClock,
) -> None:
    running, lead, member, spare = formed(registry, db_path, operator, clock)
    state = mine(lead)
    planned = plan(lead, state)
    assert send(lead, planned).status_code == 200
    signed = job(lead, state, planned.record.id, 0)
    assert send(lead, signed).status_code == 200
    cluster_offer = say(member)
    holder = member
    if cluster_offer is None:
        cluster_offer = say(spare)
        holder = spare
    assert cluster_offer is not None

    executing = add_job(db_path, operator, clock)
    worker = node(registry, clock)
    held = claim(worker)
    assert held is not None and held["task"]["record"]["id"] == executing
    by_status = {"execute": executing} | {
        s: add_job(db_path, operator, clock)
        for s in ("recheck", "operator", "accepted", "dropped", "expired")
    }
    task_statuses = {
        s: add_task(db_path, operator, clock)
        for s in ("waiting", "operator", "accepted", "dropped", "expired")
    } | {"running": running}
    with store.connection(db_path) as conn:
        for status, id in by_status.items():
            conn.execute("UPDATE jobs SET status = ? WHERE id = ?", (status, id))
        for status, id in task_statuses.items():
            conn.execute("UPDATE cluster_tasks SET status = ? WHERE id = ?", (status, id))

    _write_key(registry_dir / store.ROOT_KEY, root)
    result = registry_cli("revoke", operator.fingerprint)
    assert result.exit_code == 0, result.exception
    assert "Dropped 4 jobs." in result.output and "Dropped 3 cluster tasks." in result.output
    assert hand_in(worker, held).json()["error"] == "lease_lost"
    assert hand_in(holder, cluster_offer).json()["error"] == "lease_lost"
    assert claim(node(registry, clock)) is None
    with store.connection(db_path) as conn:
        assert store.is_revoked(conn, operator.fingerprint)
        open_jobs = {by_status[s] for s in ("execute", "recheck", "operator")} | {signed.record.id}
        for status, id in [*by_status.items(), ("execute", signed.record.id)]:
            found = work.job(conn, id)
            assert found is not None
            expected = ("dropped", "operator_revoked") if id in open_jobs else (status, None)
            assert (found.status, found.reason) == expected
        open_tasks = {task_statuses[s] for s in ("waiting", "running", "operator")}
        for status, id in task_statuses.items():
            task = clusters.cluster_task(conn, id)
            assert task is not None
            if id in open_tasks:
                assert (task.status, task.reason, task.note) == (
                    "dropped",
                    "operator_revoked",
                    "operator_revoked",
                )
            else:
                assert (task.status, task.reason, task.note) == (status, None, None)
        events = [e for e in store.entries(conn) if e.action == "operator_work_dropped"]
        assert len(events) == 1 and events[0].node == operator.fingerprint
        assert events[0].details["jobs"] == 4 and set(events[0].details["job_ids"]) == open_jobs
        assert events[0].details["cluster_tasks"] == 3
        assert set(events[0].details["cluster_task_ids"]) == open_tasks
        assert {r[0] for r in conn.execute("SELECT ended FROM leases")} == {"withdrawn"}


def test_revoke_node_and_keep_work(
    registry: Registry,
    db_path: Path,
    operator: sign.NodeKey,
    root: sign.NodeKey,
    registry_dir: Path,
    clock: FakeClock,
) -> None:
    id = add_job(db_path, operator, clock)
    task_id = add_task(db_path, operator, clock)
    worker = node(registry, clock)
    _write_key(registry_dir / store.ROOT_KEY, root)
    assert registry_cli("revoke", worker.fingerprint).exit_code == 0
    run = registry_cli("revoke", operator.fingerprint, "--keep-work")
    assert run.exit_code == 0, run.exception
    assert claim(node(registry, clock)) is None  # Trust refuses the still-open job.
    with store.connection(db_path) as conn:
        assert store.is_revoked(conn, worker.fingerprint)
        assert store.is_revoked(conn, operator.fingerprint)
        assert work.settle(conn, id, "drop", clock()).reason is None
        assert clusters.settle(conn, task_id, "drop", clock(), note="manual").note == "manual"
        assert not any(e.action == "operator_work_dropped" for e in store.entries(conn))


def test_revocation_rolls_back_on_failed_drop(
    db_path: Path,
    operator: sign.NodeKey,
    root: sign.NodeKey,
    clock: FakeClock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    id = add_job(db_path, operator, clock)
    task_id = add_task(db_path, operator, clock)
    real_log = store.log

    def fail(
        conn: sqlite3.Connection,
        action: str,
        node: str | None = None,
        details: dict[str, object] | None = None,
    ) -> None:
        if action == "operator_work_dropped":
            raise RuntimeError("journal failure")
        real_log(conn, action, node, details)

    monkeypatch.setattr(store, "log", fail)
    with store.connection(db_path) as conn:
        before = store.entries(conn)
        with pytest.raises(RuntimeError, match="journal failure"):
            admin.revoke(conn, root, operator.fingerprint, "gone", now=clock())
        assert not store.is_revoked(conn, operator.fingerprint)
        assert store.entries(conn) == before
        detail = operators.detail(conn, root.public, clock(), None, operator.fingerprint, 50)
        assert detail.summary.state == "active"
        assert [r.kind for r in detail.records] == ["assignment"]
        assert (
            conn.execute("SELECT status FROM jobs WHERE id = ?", (id,)).fetchone()[0] == "execute"
        )
        assert (
            conn.execute("SELECT status FROM cluster_tasks WHERE id = ?", (task_id,)).fetchone()[0]
            == "waiting"
        )
        assert db.parse_iso(detail.summary.issued_at) == clock()
