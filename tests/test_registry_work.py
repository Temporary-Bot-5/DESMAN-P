"""Work in the registry (ROADMAP G.6a): jobs lent with leases, results, rechecks, the operator."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest

from conftest import FakeClock
from desman.core import db
from desman.net import protocol, sign
from desman.registry import admin, store, work
from desman.registry.server import Registry
from test_registry import LOCAL, Node, db_path, registry, registry_cli, registry_dir, root, root_pub

__all__ = ["db_path", "registry", "registry_dir", "root", "root_pub"]  # fixtures used here

LEASE = work.LEASE.total_seconds()


@pytest.fixture
def operator(db_path: Path, root: sign.NodeKey, root_pub: Path, clock: FakeClock) -> sign.NodeKey:
    key = sign.NodeKey.generate()
    with store.connection(db_path) as conn:
        admin.assign(conn, root, key.public, {sign.Scope.TASKS}, days=30, now=clock())
    return key


def add_job(db_path: Path, operator: sign.NodeKey, clock: FakeClock, **spec: Any) -> str:
    days = spec.pop("days", 7)
    job = protocol.Job(
        description=spec.pop("description", "Sum 2 and 3"),
        acceptance_criteria=spec.pop("acceptance_criteria", "Says 5"),
        **spec,
    )
    with store.connection(db_path) as conn:
        return admin.add_job(conn, operator, job, days=days, now=clock()).id


def node(registry: Registry, clock: FakeClock, *, lends: bool = True) -> Node:
    found = Node(registry, clock)
    assert found.register([LOCAL] if lends else []).status_code == 200
    return found


def claim(n: Node) -> dict[str, Any] | None:
    answer = n.call("POST", protocol.CLAIM, {})
    assert answer.status_code == 200, answer.text
    offer: dict[str, Any] | None = answer.json()["work"]
    return offer


def record(
    n: Node,
    work_: dict[str, Any],
    *,
    kind: sign.RecordKind | None = None,
    key: sign.NodeKey | None = None,
    **body: Any,
) -> sign.Signed:
    task = work_["task"]["record"]["id"]
    if work_["stage"] == "execute":
        kind = kind or sign.RecordKind.RESULT
        content: dict[str, Any] = {"job": task, "lease": work_["lease"], "outcome": "done",
                                   "output": "5"} | body  # fmt: skip
    else:
        kind = kind or sign.RecordKind.VERDICT
        content = {"job": task, "lease": work_["lease"], "accept": True,
                   "result": work_["result"]["record"]["id"]} | body  # fmt: skip
    return sign.sign(key or n.key, kind, content, issued_at=n.clock(), lifetime=timedelta(days=30))


def hand_in(
    n: Node, work_: dict[str, Any], signed: sign.Signed | None = None, **body: Any
) -> httpx.Response:
    signed = signed or record(n, work_, **body)
    return n.call(
        "POST",
        protocol.HAND_IN,
        {"lease": work_["lease"], "record": signed.model_dump(mode="json")},
    )


def status(db_path: Path, job_id: str) -> tuple[str, str | None]:
    with store.connection(db_path) as conn:
        found = work.job(conn, job_id)
    assert found is not None
    return found.status, found.reason


def actions(db_path: Path) -> list[str]:
    with store.connection(db_path) as conn:
        return [e.action for e in store.entries(conn)]


# ── The way through ──────────────────────────────────────────────────────────


def test_a_job_is_done_then_rechecked_by_another_owner(
    registry: Registry, db_path: Path, operator: sign.NodeKey, clock: FakeClock
) -> None:
    job_id = add_job(db_path, operator, clock)
    a, b = node(registry, clock), node(registry, clock)

    lent = claim(a)
    assert lent is not None and lent["stage"] == "execute" and lent["result"] is None
    # What the node gets is the operator's record, as a node checks it.
    task = sign.Trust(sign.load_root(sign.ROOT_PUB), clock=clock)
    with store.connection(db_path) as conn:
        for kept in store.handed_out(conn, clock()):
            task.receive(kept.model_dump(mode="json"))
    assert task.receive(lent["task"]).id == job_id
    assert claim(b) is None  # one lease per job

    answer = hand_in(a, lent)
    assert answer.status_code == 200, answer.text
    assert answer.json() == {"job": job_id, "status": "recheck"}

    assert claim(a) is None  # never its own result
    check = claim(b)
    assert check is not None and check["stage"] == "recheck"
    assert check["result"]["record"]["signer"] == a.fingerprint
    assert check["result"]["record"]["body"]["output"] == "5"
    assert hand_in(b, check).json() == {"job": job_id, "status": "accepted"}
    assert status(db_path, job_id) == ("accepted", None)
    assert claim(a) is None and claim(b) is None

    log = actions(db_path)
    for action in ("job_added", "lent", "handed_in", "job_accepted"):
        assert action in log


def test_nodes_of_one_owner_never_recheck_each_other(
    registry: Registry, db_path: Path, operator: sign.NodeKey, clock: FakeClock
) -> None:
    job_id = add_job(db_path, operator, clock)
    a, b, c = node(registry, clock), node(registry, clock), node(registry, clock)
    with store.connection(db_path) as conn:
        admin.set_owner(conn, a.fingerprint, "alice")
        admin.set_owner(conn, b.fingerprint[:12], "alice")
    lent = claim(a)
    assert lent is not None
    hand_in(a, lent)
    assert claim(b) is None
    check = claim(c)
    assert check is not None and check["stage"] == "recheck"
    assert status(db_path, job_id) == ("recheck", None)


def test_a_rejected_result_goes_to_the_operator(
    registry: Registry, db_path: Path, operator: sign.NodeKey, clock: FakeClock
) -> None:
    job_id = add_job(db_path, operator, clock)
    a, b = node(registry, clock), node(registry, clock)
    lent = claim(a)
    assert lent is not None
    hand_in(a, lent)
    check = claim(b)
    assert check is not None
    answer = hand_in(b, check, accept=False, reason="It says 6")
    assert answer.json()["status"] == "operator"
    assert status(db_path, job_id) == ("operator", "disputed")
    assert claim(a) is None and claim(b) is None  # never settled silently

    with store.connection(db_path) as conn:
        again = admin.settle(conn, job_id[:8], "again", now=clock())
    assert again.status == "execute" and again.result is None
    # A new round, but never to a node lent the job before: it would refuse the record as a
    # replay (G.6b).
    assert claim(a) is None and claim(b) is None
    c = node(registry, clock)
    redo = claim(c)
    assert redo is not None and redo["stage"] == "execute"
    hand_in(c, redo)
    with store.connection(db_path) as conn:
        with pytest.raises(admin.AdminError, match="cannot be settled"):
            admin.settle(conn, job_id, "accept", now=clock())  # it waits for a recheck
        assert admin.settle(conn, job_id, "drop", now=clock()).status == "dropped"
        assert [h.outcome for h in work.handed_in(conn, job_id)] == ["done", "reject", "done"]


def test_failures_and_lost_leases_go_to_the_operator(
    registry: Registry, db_path: Path, operator: sign.NodeKey, clock: FakeClock
) -> None:
    failing = add_job(db_path, operator, clock)
    a, b = node(registry, clock), node(registry, clock)
    for n in (a, b):
        lent = claim(n)
        assert lent is not None and lent["task"]["record"]["id"] == failing
        assert hand_in(n, lent, outcome="failed").status_code == 200
    assert status(db_path, failing) == ("operator", "failed")

    clock.advance(1)
    lost = add_job(db_path, operator, clock)
    for _ in range(work.MAX_LOST):
        holder = node(registry, clock)  # each node is lent the job once at most
        assert claim(holder) is not None
        assert claim(holder) is None  # one lease at a time
        clock.advance(LEASE + 1)
        assert claim(holder) is None  # and never the same job again
    assert claim(node(registry, clock)) is None
    assert status(db_path, lost) == ("operator", "leases_lost")
    with (
        store.connection(db_path) as conn,
        pytest.raises(admin.AdminError, match="cannot be settled"),
    ):
        admin.settle(conn, lost, "accept", now=clock())  # no result to accept
    assert actions(db_path).count("lease_expired") == work.MAX_LOST


# ── Leases ───────────────────────────────────────────────────────────────────


def test_an_expired_lease_goes_to_another_node_and_its_result_is_refused(
    registry: Registry, db_path: Path, operator: sign.NodeKey, clock: FakeClock
) -> None:
    job_id = add_job(db_path, operator, clock)
    a, b = node(registry, clock), node(registry, clock)
    first = claim(a)
    assert first is not None
    clock.advance(LEASE + 1)
    second = claim(b)
    assert second is not None and second["task"]["record"]["id"] == job_id
    late = hand_in(a, first)
    assert late.status_code == 409 and late.json()["error"] == work.LEASE_LOST
    assert hand_in(b, second).json()["status"] == "recheck"
    with store.connection(db_path) as conn:
        refused = [e for e in store.entries(conn) if e.action == "refused"]
        assert [(e.node, e.details["reason"]) for e in refused] == [(a.fingerprint, "lease_lost")]
        assert [h.node for h in work.handed_in(conn, job_id)] == [b.fingerprint]


def test_an_extended_lease_holds(
    registry: Registry, db_path: Path, operator: sign.NodeKey, clock: FakeClock
) -> None:
    add_job(db_path, operator, clock)
    a, b = node(registry, clock), node(registry, clock)
    lent = claim(a)
    assert lent is not None
    for _ in range(3):
        clock.advance(LEASE - 60)
        kept = a.call("POST", protocol.EXTEND, {"lease": lent["lease"]})
        assert kept.status_code == 200, kept.text
    assert claim(b) is None
    assert b.call("POST", protocol.EXTEND, {"lease": lent["lease"]}).status_code == 409
    assert hand_in(a, lent).status_code == 200
    # Handed in: the lease is over and cannot be kept any longer.
    assert a.call("POST", protocol.EXTEND, {"lease": lent["lease"]}).status_code == 409


def test_a_node_that_leaves_gives_its_lease_back(
    registry: Registry, db_path: Path, operator: sign.NodeKey, clock: FakeClock
) -> None:
    add_job(db_path, operator, clock)
    a, b = node(registry, clock), node(registry, clock)
    assert claim(a) is not None
    assert a.call("POST", protocol.LEAVE).status_code == 200
    assert claim(b) is not None
    assert "lease_withdrawn" in actions(db_path)


def test_who_gets_no_work(
    registry: Registry, db_path: Path, operator: sign.NodeKey, clock: FakeClock
) -> None:
    add_job(db_path, operator, clock)
    clock.advance(1)
    add_job(db_path, operator, clock)
    idle = node(registry, clock, lends=False)
    assert claim(idle) is None  # lends nothing
    busy = node(registry, clock)
    assert claim(busy) is not None
    assert claim(busy) is None  # holds a lease already
    stranger = Node(registry, clock)
    assert stranger.call("POST", protocol.CLAIM, {}).status_code == 403  # never registered


# ── What is handed in ────────────────────────────────────────────────────────


def test_only_the_holders_own_signed_record_is_kept(
    registry: Registry, db_path: Path, operator: sign.NodeKey, clock: FakeClock
) -> None:
    job_id = add_job(db_path, operator, clock)
    a = node(registry, clock)
    lent = claim(a)
    assert lent is not None

    def refused(signed: sign.Signed) -> str:
        answer = hand_in(a, lent, signed)
        assert answer.status_code in (400, 401), answer.text
        return str(answer.json()["error"])

    good = record(a, lent)
    tampered = good.model_copy(
        update={
            "record": good.record.model_copy(update={"body": good.record.body | {"output": "6"}})
        }
    )
    assert refused(tampered) == "forged"
    assert refused(record(a, lent, key=sign.NodeKey.generate())) == "unknown_signer"
    assert refused(record(a, lent, kind=sign.RecordKind.VERDICT)) == "malformed"
    assert refused(record(a, lent, job="0" * 32)) == "malformed"
    assert refused(record(a, lent, outcome="maybe")) == "malformed"
    assert refused(record(a, lent, output="x" * (protocol.MAX_RESULT_BYTES + 1))) == "malformed"
    assert status(db_path, job_id) == ("execute", None)

    assert hand_in(a, lent, good).status_code == 200
    again = hand_in(a, lent, good)
    assert again.status_code == 409  # the lease ended with the first


def test_a_verdict_must_name_the_result_lent(
    registry: Registry, db_path: Path, operator: sign.NodeKey, clock: FakeClock
) -> None:
    add_job(db_path, operator, clock)
    a, b = node(registry, clock), node(registry, clock)
    lent = claim(a)
    assert lent is not None
    hand_in(a, lent)
    check = claim(b)
    assert check is not None
    assert hand_in(b, check, result="f" * 32).json()["error"] == "malformed"
    assert hand_in(b, check, kind=sign.RecordKind.RESULT).json()["error"] == "malformed"
    assert hand_in(b, check).status_code == 200


# ── The operator's authority ─────────────────────────────────────────────────


def test_a_job_whose_operator_is_revoked_is_not_lent(
    registry: Registry,
    db_path: Path,
    operator: sign.NodeKey,
    root: sign.NodeKey,
    clock: FakeClock,
) -> None:
    add_job(db_path, operator, clock)
    with store.connection(db_path) as conn:
        admin.revoke(conn, root, operator.fingerprint, "gone", now=clock())
        with pytest.raises(admin.AdminError, match="refuse"):
            admin.add_job(
                conn,
                operator,
                protocol.Job(description="x", acceptance_criteria="y"),
                days=1,
                now=clock(),
            )
    assert claim(node(registry, clock)) is None


def test_an_unassigned_key_cannot_add_jobs(db_path: Path, root_pub: Path, clock: FakeClock) -> None:
    with store.connection(db_path) as conn:
        with pytest.raises(admin.AdminError, match="unknown_signer"):
            admin.add_job(
                conn,
                sign.NodeKey.generate(),
                protocol.Job(description="x", acceptance_criteria="y"),
                days=1,
                now=clock(),
            )
        assert work.jobs(conn) == []


def test_a_job_past_its_term_expires(
    registry: Registry, db_path: Path, operator: sign.NodeKey, clock: FakeClock
) -> None:
    job_id = add_job(db_path, operator, clock, days=1)
    clock.advance(timedelta(days=1).total_seconds() + 1)
    assert claim(node(registry, clock)) is None
    assert status(db_path, job_id) == ("expired", None)


def test_the_tables_are_append_only(
    registry: Registry, db_path: Path, operator: sign.NodeKey, clock: FakeClock
) -> None:
    job_id = add_job(db_path, operator, clock)
    a = node(registry, clock)
    lent = claim(a)
    assert lent is not None
    hand_in(a, lent)
    with store.connection(db_path) as conn:
        for sql in (
            "DELETE FROM jobs",
            "UPDATE jobs SET signed_json = '{}'",
            "DELETE FROM results",
            "UPDATE results SET outcome = 'failed'",
            "DELETE FROM leases",
            "UPDATE leases SET ended = 'expired'",
            "UPDATE leases SET node = 'x'",
        ):
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(sql)
    assert status(db_path, job_id) == ("recheck", None)


def test_a_registry_of_the_first_schema_is_upgraded(tmp_path: Path) -> None:
    path = tmp_path / "old.db"
    conn = db.connect(path)
    db.migrate(conn, store.MIGRATIONS[:1])
    conn.execute(
        "INSERT INTO nodes (fingerprint, public_key, status, registered_at, last_seen)"
        " VALUES (?, 'k', 'active', 't', 't')",
        ("a" * 64,),
    )
    conn.close()
    upgraded = store.open_registry(path)
    try:
        found = store.node(upgraded, "a" * 64)
        assert found is not None and found.owner is None and found.owned_by == "a" * 64
        assert work.jobs(upgraded) == []
    finally:
        upgraded.close()


# ── CLI ──────────────────────────────────────────────────────────────────────


def test_cli_jobs_from_adding_to_settling(
    registry: Registry,
    registry_dir: Path,
    db_path: Path,
    root: sign.NodeKey,
    root_pub: Path,
    clock: FakeClock,
) -> None:
    clock.now = datetime.now(UTC)  # the CLI signs with the real time; the registry agrees
    key = sign.create_node_key(registry_dir / store.OPERATOR_KEY)
    with store.connection(db_path) as conn:
        admin.assign(conn, root, key.public, {sign.Scope.TASKS}, days=30, now=clock())

    assert (
        registry_cli(
            "job", "add", "--description", "x", "--acceptance", "y", "--input", "bad"
        ).exit_code
        == 2
    )
    added = registry_cli(
        "job", "add", "--description", "Sum [bold]2[/bold] and 3", "--acceptance", "Says 5",
        "--kind", "analysis", "--input", "numbers=2,3",
    )  # fmt: skip
    assert added.exit_code == 0, added.output
    listed = json.loads(registry_cli("jobs", "--json").output)
    assert len(listed) == 1 and listed[0]["status"] == "execute"
    job_id = listed[0]["job"]
    assert listed[0]["task"]["inputs"] == {"numbers": "2,3"}
    assert listed[0]["task"]["kind"] == "analysis"
    assert job_id[:8] in registry_cli("jobs").output
    assert "Nothing waits for the operator." in registry_cli("inbox").output

    a, b = node(registry, clock), node(registry, clock)
    lent = claim(a)
    assert lent is not None
    hand_in(a, lent, output="[red]5[/red]")
    check = claim(b)
    assert check is not None
    hand_in(b, check, accept=False, reason="wrong")

    inbox = registry_cli("inbox")
    assert job_id[:8] in inbox.output and "job settle" in inbox.output
    shown = registry_cli("job", "show", job_id[:8])
    assert shown.exit_code == 0, shown.output
    assert "[red]5[/red]" in shown.output and "[bold]2[/bold]" in shown.output  # never markup
    assert "wrong" in shown.output
    detail = json.loads(registry_cli("job", "show", job_id, "--json").output)
    assert [h["outcome"] for h in detail["handed_in"]] == ["done", "reject"]

    assert registry_cli("job", "settle", job_id[:8], "again").exit_code == 0
    refused = registry_cli("job", "settle", job_id[:8], "accept")
    assert refused.exit_code == 1 and "cannot be settled" in refused.output
    assert registry_cli("job", "show", "ffffffff").exit_code == 1

    assert registry_cli("owner", a.fingerprint[:10], "alice").exit_code == 0
    assert registry_cli("owner", a.fingerprint[:10]).exit_code == 2
    assert registry_cli("owner", "0000000000", "bob").exit_code == 1
    nodes = json.loads(registry_cli("nodes", "--json").output)
    assert {n["owner"] for n in nodes} == {"alice", None}
    assert registry_cli("owner", a.fingerprint, "--clear").exit_code == 0
    log = [e["action"] for e in json.loads(registry_cli("log", "--json", "--limit", "500").output)]
    assert log.count("owner") == 2 and "job_settled" in log


def test_cli_job_add_needs_the_operator_key(registry_dir: Path, root_pub: Path) -> None:
    result = registry_cli("job", "add", "--description", "x", "--acceptance", "y")
    assert result.exit_code == 1
    assert "cannot be used" in result.output
    assert not (registry_dir / store.OPERATOR_KEY).exists()  # never created to sign a job


# ── Review round 1 (Devin) ───────────────────────────────────────────────────


def test_a_result_that_validates_fits_a_request_however_it_is_escaped(
    registry: Registry, db_path: Path, operator: sign.NodeKey, clock: FakeClock
) -> None:
    add_job(db_path, operator, clock)
    a = node(registry, clock)
    lent = claim(a)
    assert lent is not None
    common = {"job": "0" * 32, "lease": "0" * 32, "outcome": "done"}
    with pytest.raises(ValueError, match="over"):
        protocol.Result.model_validate(common | {"output": "é" * 50_000})
    with pytest.raises(ValueError, match="over"):
        protocol.Result.model_validate(
            common | {"files": [{"path": "a.txt", "text": "😀" * 14_000}]}
        )
    # As much as validates, sent the most expanded way (json.dumps escapes non-ASCII).
    output = "é" * ((protocol.MAX_RESULT_BYTES - 1000) // 6)
    files = [{"path": "b.txt", "text": "😀" * 70}]
    protocol.Result.model_validate(common | {"output": output, "files": files})
    answer = hand_in(a, lent, output=output, files=files)
    assert answer.status_code == 200, answer.text


def test_the_inbox_sees_a_job_whose_last_lease_lapsed_with_no_node_calling(
    registry: Registry,
    registry_dir: Path,
    db_path: Path,
    root: sign.NodeKey,
    root_pub: Path,
    clock: FakeClock,
) -> None:
    clock.now = datetime.now(UTC) - timedelta(hours=2)
    key = sign.NodeKey.generate()
    with store.connection(db_path) as conn:
        admin.assign(conn, root, key.public, {sign.Scope.TASKS}, days=30, now=clock())
    job_id = add_job(db_path, key, clock)
    for _ in range(work.MAX_LOST):
        assert claim(node(registry, clock)) is not None
        clock.advance(LEASE + 1)
    assert status(db_path, job_id) == ("execute", None)  # nobody swept yet
    shown = registry_cli("inbox")
    assert job_id[:8] in shown.output
    assert status(db_path, job_id) == ("operator", "leases_lost")


def test_again_is_refused_once_the_jobs_record_has_expired(
    registry: Registry, db_path: Path, operator: sign.NodeKey, clock: FakeClock
) -> None:
    job_id = add_job(db_path, operator, clock, days=1)
    a, b = node(registry, clock), node(registry, clock)
    lent = claim(a)
    assert lent is not None
    hand_in(a, lent)
    check = claim(b)
    assert check is not None
    hand_in(b, check, accept=False)
    clock.advance(timedelta(days=1).total_seconds())
    with store.connection(db_path) as conn:
        with pytest.raises(admin.AdminError, match="expired"):
            admin.settle(conn, job_id, "again", now=clock())
        assert admin.settle(conn, job_id, "accept", now=clock()).status == "accepted"


def test_a_jobs_term_ends_with_the_operators_assignment(
    db_path: Path, root: sign.NodeKey, root_pub: Path, clock: FakeClock
) -> None:
    key = sign.NodeKey.generate()
    with store.connection(db_path) as conn:
        done = admin.assign(conn, root, key.public, {sign.Scope.TASKS}, days=1, now=clock())
        added = admin.add_job(
            conn, key, protocol.Job(description="x", acceptance_criteria="y"), days=7, now=clock()
        )
    assert added.signed.record.expires_at == done.expires_at


def test_a_later_owner_does_not_make_the_same_owner_a_reviewer(
    registry: Registry, db_path: Path, operator: sign.NodeKey, clock: FakeClock
) -> None:
    add_job(db_path, operator, clock)
    a, b, c = node(registry, clock), node(registry, clock), node(registry, clock)
    with store.connection(db_path) as conn:
        admin.set_owner(conn, a.fingerprint, "alice")
        admin.set_owner(conn, b.fingerprint, "alice")
    lent = claim(a)
    assert lent is not None
    hand_in(a, lent)
    with store.connection(db_path) as conn:
        admin.set_owner(conn, a.fingerprint, "bob")  # a's result was alice's when it came
    assert claim(b) is None
    with store.connection(db_path) as conn:
        admin.set_owner(conn, c.fingerprint, "bob")
    assert claim(c) is None  # nor does bob review what is now bob's node
    d = node(registry, clock)
    assert claim(d) is not None


def test_a_record_past_its_own_term_is_refused(
    registry: Registry, db_path: Path, operator: sign.NodeKey, clock: FakeClock
) -> None:
    add_job(db_path, operator, clock)
    a = node(registry, clock)
    lent = claim(a)
    assert lent is not None
    stale = sign.sign(
        a.key,
        sign.RecordKind.RESULT,
        {"job": lent["task"]["record"]["id"], "lease": lent["lease"], "outcome": "done"},
        issued_at=clock() - timedelta(minutes=2),
        lifetime=timedelta(minutes=1),
    )
    answer = hand_in(a, lent, stale)
    assert answer.status_code == 401 and answer.json()["error"] == "expired"


def test_a_done_result_must_hold_as_long_as_its_job(
    registry: Registry, db_path: Path, operator: sign.NodeKey, clock: FakeClock
) -> None:
    """Its recheck may come as late as the job's term: a result ending first could not be
    judged then. A failed one is never rechecked: its term is its own."""
    add_job(db_path, operator, clock, days=7)
    a = node(registry, clock)
    lent = claim(a)
    assert lent is not None
    body = {"job": lent["task"]["record"]["id"], "lease": lent["lease"], "outcome": "done"}
    short = sign.sign(
        a.key, sign.RecordKind.RESULT, body, issued_at=clock(), lifetime=timedelta(days=1)
    )
    answer = hand_in(a, lent, short)
    assert answer.status_code == 400 and answer.json()["error"] == "malformed"
    failed = sign.sign(
        a.key,
        sign.RecordKind.RESULT,
        body | {"outcome": "failed"},
        issued_at=clock(),
        lifetime=timedelta(days=1),
    )
    assert hand_in(a, lent, failed).status_code == 200


def test_a_registry_of_the_second_schema_is_upgraded(tmp_path: Path) -> None:
    path = tmp_path / "v2.db"
    conn = db.connect(path)
    db.migrate(conn, store.MIGRATIONS[:2])
    conn.close()
    upgraded = store.open_registry(path)
    try:
        columns = {r["name"] for r in upgraded.execute("PRAGMA table_info(results)")}
        assert "owner" in columns
    finally:
        upgraded.close()


def test_a_registry_of_the_third_schema_is_upgraded(tmp_path: Path) -> None:
    path = tmp_path / "v3.db"
    conn = db.connect(path)
    db.migrate(conn, store.MIGRATIONS[:3])
    conn.close()
    upgraded = store.open_registry(path)
    try:
        assert upgraded.execute("SELECT count(*) FROM trust_lists").fetchone()[0] == 0
    finally:
        upgraded.close()
