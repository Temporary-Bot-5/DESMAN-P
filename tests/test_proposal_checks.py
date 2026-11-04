"""G.7b: independently authenticated, leased fake benchmarks and durable hand-ins."""

from __future__ import annotations

import base64
import sqlite3
from datetime import timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
from click.testing import CliRunner

from conftest import FakeClock
from desman import bench, evaluation, improve
from desman.config import Config, Network, Tuning
from desman.core import db
from desman.net import checks, jobs, node, protocol, sign
from desman.registry import admin, store, work
from desman.registry import checks as registry_checks
from desman.registry.cli import cli
from desman.registry.server import Registry
from test_evaluation import checkout, git, origin, package, rules_change, worktrees
from test_registry import Node, db_path, registry, registry_dir, root, root_pub
from test_registry_work import add_job, claim, hand_in, operator

__all__ = [
    "checkout",
    "origin",
    "db_path",
    "registry",
    "registry_dir",
    "root",
    "root_pub",
    "operator",
]

REGISTRY = "https://registry.test"


def snapshot() -> protocol.BenchmarkSnapshot:
    return protocol.BenchmarkSnapshot(
        version=bench.VERSION, mode="fake", source="/checkout/src/desman", cases=[], stats={}
    )


def submission(
    key: sign.NodeKey, repo: Path | None = None, **overrides: Any
) -> protocol.SignedProposalSubmission:
    details: dict[str, Any]
    diff, details = (
        "diff",
        dict(
            changes=[
                dict(target="planner.rules", file="src/desman/planner.py", old="old", new="new")
            ],
            base={},
        ),
    )
    commit = "a" * 40
    if repo is not None:
        diff, details = improve.render([rules_change(repo)], package(repo))
        commit = git(repo, "rev-parse", "HEAD").strip()
    item = protocol.ProposalSubmission(
        proposal=1,
        evaluation=1,
        summary="Improve planning",
        details=protocol.ProposalDetails.model_validate(details),
        base_commit=commit,
        diff=diff,
        mode="fake",
        verdict="same",
        base=snapshot(),
        candidate=snapshot(),
        **overrides,
    )
    return signed_submission(key, item)


def signed_submission(
    key: sign.NodeKey, item: protocol.ProposalSubmission
) -> protocol.SignedProposalSubmission:
    return protocol.SignedProposalSubmission(
        submission=item,
        signer=key.fingerprint,
        signature=base64.b64encode(key.sign(sign.canonical(item.model_dump(mode="json")))).decode(),
    )


def submit(a: Node, signed: protocol.SignedProposalSubmission | None = None) -> str:
    answer = a.call(
        "POST",
        protocol.PROPOSALS,
        protocol.SubmitProposal(signed=signed or submission(a.key)).model_dump(mode="json"),
    )
    assert answer.status_code == 200, answer.text
    return str(answer.json()["id"])


def ask(a: Node) -> protocol.ProposalWork | None:
    answer = a.call("POST", protocol.CLAIM, {"proposals": True})
    assert answer.status_code == 200, answer.text
    return protocol.Offer.model_validate(answer.json()).check


def checker(registry: Registry, clock: FakeClock) -> Node:
    a = Node(registry, clock)
    assert a.register().status_code == 200
    return a


def body(
    lent: protocol.ProposalWork, outcome: str = "unable", **overrides: Any
) -> protocol.ProposalCheck:
    data: dict[str, Any] = dict(
        proposal=lent.proposal,
        lease=lent.lease,
        submission=protocol.submission_hash(lent.submission.submission),
        commit=lent.submission.submission.base_commit,
        outcome=outcome,
    )
    if outcome == "checked":
        data.update(verdict="same", base=snapshot(), candidate=snapshot(), matches_submission=True)
    return protocol.ProposalCheck.model_validate(data | overrides)


def record(
    a: Node, lent: protocol.ProposalWork, outcome: str = "unable", **overrides: Any
) -> sign.Signed:
    return sign.sign(
        a.key,
        sign.RecordKind.PROPOSAL_CHECK,
        body(lent, outcome, **overrides).model_dump(mode="json"),
        issued_at=a.clock(),
        lifetime=timedelta(days=1),
    )


def give(a: Node, lent: protocol.ProposalWork, signed: sign.Signed) -> httpx.Response:
    return a.call(
        "POST",
        protocol.CHECK_HAND_IN,
        protocol.HandInCheck(lease=lent.lease, record=signed).model_dump(mode="json"),
    )


@pytest.fixture
def fresh(db_path: Path, root: sign.NodeKey, clock: FakeClock, root_pub: Path) -> None:
    with store.connection(db_path) as conn:
        admin.sign_trust_list(conn, root, days=7, now=clock())


def state(path: Path, proposal: str) -> tuple[str, str | None]:
    with store.connection(path) as conn:
        found = store.proposal(conn, proposal)
        assert found is not None
        return found.status, found.reason


def test_lending_requires_another_owner_and_never_repeats(
    registry: Registry, db_path: Path, clock: FakeClock, fresh: None
) -> None:
    a, b = checker(registry, clock), checker(registry, clock)
    pid = submit(a)
    assert ask(a) is None
    with store.connection(db_path) as conn:
        work.set_owner(conn, b.fingerprint, a.fingerprint)
    assert ask(b) is None
    with store.connection(db_path) as conn:
        work.set_owner(conn, b.fingerprint, "another-owner")
    lent = ask(b)
    assert lent is not None and lent.proposal == pid
    assert state(db_path, pid) == ("reviewing", None)
    assert ask(b) is None
    assert give(b, lent, record(b, lent)).status_code == 200
    assert state(db_path, pid) == ("submitted", None)
    assert ask(b) is None


def test_stale_trust_lends_nothing(registry: Registry, clock: FakeClock) -> None:
    a, b = checker(registry, clock), checker(registry, clock)
    submit(a)
    assert ask(b) is None


def test_checks_between_rechecks_and_executions_and_shared_limit(
    registry: Registry, db_path: Path, clock: FakeClock, operator: sign.NodeKey
) -> None:
    from test_registry import LOCAL

    a, b, c = checker(registry, clock), checker(registry, clock), checker(registry, clock)
    for n in (a, b, c):
        assert n.heartbeat([LOCAL]).status_code == 200
    job = add_job(db_path, operator, clock)
    execution = claim(a)
    assert execution is not None
    pid = submit(a)
    assert ask(a) is None  # its job holds the shared slot
    assert hand_in(a, execution).status_code == 200
    # A recheck has precedence over a pending proposal.
    answer = b.call("POST", protocol.CLAIM, {"proposals": True}).json()
    assert answer["work"]["stage"] == "recheck" and answer["check"] is None
    add_job(db_path, operator, clock, description="Another job")
    lent = ask(c)
    assert lent is not None and lent.proposal == pid
    assert claim(c) is None  # its proposal holds the shared slot
    assert give(c, lent, record(c, lent, "checked")).status_code == 200
    assert state(db_path, pid) == ("reviewed", None)
    assert job != pid


@pytest.mark.parametrize("how", ["expired", "unable", "withdrawn"])
def test_lease_recovery_and_limits(
    registry: Registry, db_path: Path, clock: FakeClock, fresh: None, how: str
) -> None:
    a = checker(registry, clock)
    pid = submit(a)
    for attempt in range(3):
        b = checker(registry, clock)
        lent = ask(b)
        assert lent is not None
        if how == "expired":
            clock.advance(work.LEASE.total_seconds())
            with store.connection(db_path) as conn:
                work.expire(conn, clock())
        elif how == "unable":
            assert give(b, lent, record(b, lent)).status_code == 200
        else:
            assert b.call("POST", protocol.LEAVE).status_code == 200
        expected = (
            ("reviewed", "leases_lost" if how == "expired" else "unable")
            if attempt == 2 and how != "withdrawn"
            else ("submitted", None)
        )
        assert state(db_path, pid) == expected
    if how != "withdrawn":
        assert ask(checker(registry, clock)) is None


@pytest.mark.parametrize("problem", ["signer", "lease", "hash", "kind", "expired", "forged"])
def test_bad_hand_ins(
    registry: Registry, db_path: Path, clock: FakeClock, fresh: None, problem: str
) -> None:
    a, b = checker(registry, clock), checker(registry, clock)
    pid = submit(a)
    lent = ask(b)
    assert lent is not None
    signed = record(
        b,
        lent,
        **(
            {"lease": "0" * 32}
            if problem == "lease"
            else {"submission": "0" * 64}
            if problem == "hash"
            else {}
        ),
    )
    if problem == "signer":
        signed = record(a, lent)
    elif problem == "kind":
        signed = sign.sign(
            b.key,
            sign.RecordKind.RESULT,
            signed.record.body,
            issued_at=clock(),
            lifetime=timedelta(days=1),
        )
    elif problem == "forged":
        signed = signed.model_copy(update={"signature": base64.b64encode(b"x" * 64).decode()})
    elif problem == "expired":
        clock.advance(work.LEASE.total_seconds())
    reply = give(b, lent, signed)
    assert reply.status_code != 200
    with store.connection(db_path) as conn:
        assert registry_checks.results(conn, pid) == []


def test_replay_and_operator_display(
    registry: Registry,
    db_path: Path,
    registry_dir: Path,
    clock: FakeClock,
    fresh: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    a, b = checker(registry, clock), checker(registry, clock)
    pid = submit(a)
    lent = ask(b)
    assert lent is not None
    signed = record(b, lent, "checked")
    assert give(b, lent, signed).status_code == 200
    assert give(b, lent, signed).json()["error"] == "replay"
    monkeypatch.setattr("desman.registry.cli._now", clock)
    result = CliRunner().invoke(cli, ["proposal", "show", pid, "--json"])
    assert result.exit_code == 0, result.output
    import json

    assert json.loads(result.output)["checks"][0]["matches_submission"] is True
    result = CliRunner().invoke(cli, ["proposal", "show", pid])
    assert result.exit_code == 0, result.output
    assert "Matches submission" in result.output
    with store.connection(db_path) as conn:
        for sql in ("DELETE FROM proposal_checks", "UPDATE proposal_checks SET outcome = 'unable'"):
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                conn.execute(sql)
        assert all(lent.lease not in str(e.details) for e in store.entries(conn))


@pytest.mark.parametrize("version", [5])
def test_registry_upgrade(tmp_path: Path, version: int) -> None:
    path = tmp_path / "old-registry.db"
    conn = db.connect(path)
    db.migrate(conn, store.MIGRATIONS[:version])
    key = sign.NodeKey.generate()
    signed = submission(key)
    conn.execute(
        "INSERT INTO nodes (fingerprint, public_key, status, registered_at, last_seen) VALUES (?, ?, 'active', 'old', 'old')",
        (key.fingerprint, sign.encode_public(key.public)),
    )
    conn.execute(
        "INSERT INTO proposals (id, node, proposal, evaluation, submission_json) VALUES (?, ?, 1, 1, ?)",
        ("f" * 32, key.fingerprint, signed.model_dump_json()),
    )
    conn.close()
    conn = store.open_registry(path)
    assert db.schema_version(conn) == 6
    assert conn.execute("SELECT count(*) FROM proposal_checks").fetchone()[0] == 0
    found = store.proposal(conn, "f" * 32)
    assert found is not None and found.signed == signed and found.reason is None
    conn.close()


@pytest.mark.parametrize("version", [1, 20, 29])
def test_node_upgrade(tmp_path: Path, version: int) -> None:
    path = tmp_path / "old-node.db"
    conn = db.connect(path)
    db.migrate(conn, db.MIGRATIONS[:version])
    conn.execute("INSERT INTO tasks (id, goal) VALUES ('kept', 'old task')")
    conn.close()
    conn = db.open_db(path)
    assert db.schema_version(conn) == 30
    assert conn.execute("SELECT goal FROM tasks WHERE id = 'kept'").fetchone()[0] == "old task"
    assert checks.pending(conn) == []
    conn.close()


def test_plain_check_refuses_before_running_and_cleans_up(checkout: Path, tmp_path: Path) -> None:
    diff, details = improve.render([rules_change(checkout)], package(checkout))
    spy = tmp_path / "python-spy"
    marker = tmp_path / "called"
    spy.write_text(f"#!/bin/sh\ntouch {marker}\n")
    spy.chmod(0o700)
    result = evaluation.check(
        checkout,
        git(checkout, "rev-parse", "HEAD").strip(),
        changes=details["changes"],
        base=details["base"],
        diff=diff + "bad",
        mode="fake",
        base_tuning=Tuning(),
        candidate_tuning=Tuning(),
        python=str(spy),
    )
    assert result.outcome == "refused" and not marker.exists()
    assert len(worktrees(checkout)) == 1


def test_plain_check_failed_and_done(checkout: Path, tmp_path: Path) -> None:
    diff, details = improve.render([rules_change(checkout)], package(checkout))
    args: dict[str, Any] = dict(
        changes=details["changes"],
        base=details["base"],
        diff=diff,
        mode="fake",
        base_tuning=Tuning(),
        candidate_tuning=Tuning(),
    )
    commit = git(checkout, "rev-parse", "HEAD").strip()
    failed = evaluation.check(checkout, commit, **args, python=str(tmp_path / "missing"))
    assert failed.outcome == "failed" and failed.base is None
    done = evaluation.check(checkout, commit, **args)
    assert done.outcome == "done" and done.verdict == "same"
    assert len(worktrees(checkout)) == 1


@pytest.mark.parametrize(
    "scenario",
    ["honest", "extra_diff", "missing", "forged", "retry", "dishonest", "real", "failed"],
)
def test_node_end_to_end(
    checkout: Path,
    registry: Registry,
    db_path: Path,
    clock: FakeClock,
    root_pub: Path,
    fresh: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
) -> None:
    a = checker(registry, clock)
    signed = submission(a.key, checkout)
    if scenario in ("honest", "retry"):
        item = signed.submission
        honest = evaluation.check(
            checkout,
            item.base_commit,
            changes=[c.model_dump() for c in item.details.changes],
            base=item.details.base,
            diff=item.diff,
            mode="fake",
            base_tuning=Tuning(),
            candidate_tuning=Tuning(),
        )
        assert honest.base is not None and honest.candidate is not None
        signed = signed_submission(
            a.key,
            item.model_copy(
                update={
                    "base": protocol.benchmark_snapshot(honest.base),
                    "candidate": protocol.benchmark_snapshot(honest.candidate),
                }
            ),
        )
    elif scenario == "extra_diff":
        signed = signed_submission(
            a.key,
            signed.submission.model_copy(
                update={
                    "diff": signed.submission.diff + "\n--- a/src/desman/cli.py\nmalicious code"
                }
            ),
        )
        monkeypatch.setattr(
            evaluation, "_run", lambda *args, **kwargs: pytest.fail("benchmark must not run")
        )
    elif scenario == "missing":
        signed = signed_submission(
            a.key, signed.submission.model_copy(update={"base_commit": "0" * 40})
        )
    if scenario == "real":
        item = signed.submission
        signed = signed_submission(
            a.key,
            item.model_copy(
                update={
                    "mode": "real",
                    "base": item.base.model_copy(update={"mode": "real"}),
                    "candidate": item.candidate.model_copy(update={"mode": "real"}),
                }
            ),
        )
    if scenario == "failed":

        def fail(*args: Any, **kwargs: Any) -> Any:
            raise evaluation.RunFailed("untrusted model output must not be quoted")

        monkeypatch.setattr(evaluation, "_run", fail)
    pid = submit(a, signed)
    if scenario == "forged":
        # A compromised registry offers a forged signature, even though ingress normally refuses it.
        signed = signed.model_copy(update={"signature": base64.b64encode(b"x" * 64).decode()})
        original = registry_checks.lend

        def forged(*args: Any, **kwargs: Any) -> protocol.ProposalWork | None:
            offered = original(*args, **kwargs)
            return None if offered is None else offered.model_copy(update={"submission": signed})

        monkeypatch.setattr(registry_checks, "lend", forged)
        monkeypatch.setattr(
            evaluation, "_run", lambda *args, **kwargs: pytest.fail("forged submission ran")
        )
    monkeypatch.setattr(evaluation, "repository", lambda: checkout)
    key_path = tmp_path / "checker.key"
    sign.create_node_key(key_path)
    conn = db.open_db(tmp_path / "checker.db")
    config = Config(network=Network(enabled=True, registry=REGISTRY))
    requests: list[str] = []
    blocked = scenario == "retry"

    def transport(req: httpx.Request) -> httpx.Response:
        requests.append(req.url.path)
        if blocked and req.url.path == protocol.CHECK_HAND_IN:
            raise httpx.ConnectError("down", request=req)
        reply = registry.handle(
            req.method, req.url.raw_path.decode(), req.headers.multi_items(), req.content
        )
        return httpx.Response(reply.status, json=reply.body)

    machine = node.Node(
        conn,
        lambda: config,
        key_path=key_path,
        root=root_pub,
        transport=httpx.MockTransport(transport),
        clock=clock,
    )
    machine.poll(idle=True)
    if scenario == "retry":
        pending = checks.pending(conn)
        assert len(pending) == 1
        record_id = pending[0].record.record.id
        machine.close()
        blocked = False
        clock.advance(node.RETRY_FIRST.total_seconds())
        machine = node.Node(
            conn,
            lambda: config,
            key_path=key_path,
            root=root_pub,
            transport=httpx.MockTransport(transport),
            clock=clock,
        )
        machine.poll(idle=True)
        assert checks.pending(conn) == []
        with store.connection(db_path) as rconn:
            assert registry_checks.results(rconn, pid)[0]["record"]["record"]["id"] == record_id
    if scenario == "forged":
        assert state(db_path, pid) == ("reviewing", None)
        assert protocol.CHECK_HAND_IN not in requests
        assert (
            conn.execute(
                "SELECT count(*) FROM network_log WHERE action = 'check_refused'"
            ).fetchone()[0]
            == 1
        )
    else:
        with store.connection(db_path) as rconn:
            [result] = registry_checks.results(rconn, pid)
        assert (
            result["outcome"]
            == {
                "honest": "checked",
                "retry": "checked",
                "extra_diff": "refused",
                "missing": "unable",
                "dishonest": "checked",
                "real": "checked",
                "failed": "failed",
            }[scenario]
        )
        if scenario in ("honest", "retry"):
            assert result["matches_submission"] is True
        elif scenario == "dishonest":
            assert result["matches_submission"] is False
        elif scenario == "real":
            assert result["matches_submission"] is None
            assert result["mode"] == "fake"
        elif scenario == "failed":
            assert "untrusted model output" not in result["reason"]
        assert state(db_path, pid)[0] == ("submitted" if scenario == "missing" else "reviewed")
    machine.close()
    conn.close()


def local_work(key: sign.NodeKey, clock: FakeClock) -> protocol.ProposalWork:
    return protocol.ProposalWork(
        lease="1" * 32,
        expires_at=clock() + timedelta(minutes=10),
        proposal="2" * 32,
        submission=submission(key),
        submitter_key=sign.encode_public(key.public),
    )


def test_submission_authentication_revocation_and_local_replay(
    tmp_path: Path, clock: FakeClock
) -> None:
    root_key, author = sign.NodeKey.generate(), sign.NodeKey.generate()
    trust = sign.Trust(root_key.public, clock=clock)
    lent = local_work(author, clock)
    checks.verify(lent, trust, clock())
    with pytest.raises(sign.Refused) as error:
        checks.verify(
            lent.model_copy(update={"expires_at": clock().replace(tzinfo=None)}), trust, clock()
        )
    assert error.value.reason is sign.Reason.MALFORMED
    conn = db.open_db(tmp_path / "checks.db")
    signed = sign.sign(
        root_key,
        sign.RecordKind.PROPOSAL_CHECK,
        body(lent).model_dump(mode="json"),
        issued_at=clock(),
        lifetime=timedelta(days=1),
    )
    checks.keep(conn, REGISTRY, lent, signed, clock())
    with pytest.raises(sign.Refused) as error:
        checks.verify(lent, trust, clock(), conn=conn, registry=REGISTRY)
    assert error.value.reason is sign.Reason.REPLAY
    revocation = sign.sign(
        root_key,
        sign.RecordKind.REVOCATION,
        {"key": author.fingerprint},
        issued_at=clock(),
        lifetime=timedelta(days=1),
    )
    trust.receive(revocation.model_dump(mode="json"))
    with pytest.raises(sign.Refused) as error:
        checks.verify(lent, trust, clock())
    assert error.value.reason is sign.Reason.REVOKED
    with pytest.raises(sign.Refused) as error:
        trust.receive(signed.model_dump(mode="json"))
    assert error.value.reason is sign.Reason.NOT_ALLOWED
    conn.close()


@pytest.mark.parametrize(
    "change",
    [
        {"outcome": "checked"},
        {"outcome": "refused", "base": snapshot().model_dump()},
        {"outcome": "unable", "candidate": snapshot().model_dump()},
        {"outcome": "failed", "verdict": "better"},
        {"outcome": "unable", "matches_submission": False},
        {"outcome": "unable", "reason": "a" * 1001},
        {"outcome": "unable", "mode": "real"},
    ],
)
def test_check_contract_rejects_inconsistent_results(
    clock: FakeClock, change: dict[str, Any]
) -> None:
    lent = local_work(sign.NodeKey.generate(), clock)
    with pytest.raises(ValueError):
        protocol.ProposalCheck.model_validate(body(lent).model_dump() | change)
    with pytest.raises(ValueError):
        protocol.Claim.model_validate({"proposals": 1})
    with pytest.raises(ValueError):
        protocol.Claim.model_validate({"unknown": True})


@pytest.mark.parametrize("change", ["off", "moved", "refused"])
def test_waiting_signed_checks_close_without_new_claims(
    tmp_path: Path, clock: FakeClock, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    key = sign.create_node_key(tmp_path / "node.key")
    lent = local_work(sign.NodeKey.generate(), clock)
    signed = sign.sign(
        key,
        sign.RecordKind.PROPOSAL_CHECK,
        body(lent).model_dump(mode="json"),
        issued_at=clock(),
        lifetime=timedelta(days=1),
    )
    conn = db.open_db(tmp_path / "node.db")
    checks.keep(conn, REGISTRY, lent, signed, clock())
    config = Config(
        network=Network(
            enabled=change != "off",
            registry=REGISTRY if change != "moved" else "https://other.test",
        )
    )
    machine = node.Node(conn, lambda: config, key_path=tmp_path / "node.key", clock=clock)
    machine.next_at = clock() + timedelta(hours=1)
    contacted: list[str] = []

    def call(*args: Any, **kwargs: Any) -> Any:
        contacted.append(str(args[5]))
        raise node._Problem(node.LEASE_LOST)

    monkeypatch.setattr(node, "_call", call)
    if change == "refused":
        machine.registry = REGISTRY
        machine.registered = True
        machine._trust = sign.Trust(key.public, clock=clock)
    machine.poll()
    assert checks.pending(conn) == []
    assert conn.execute("SELECT state FROM network_checks").fetchone()[0] == "lost"
    assert len(contacted) == (1 if change == "refused" else 0)
    machine.close()
    conn.close()


@pytest.mark.parametrize("recheck", [False, True])
def test_check_only_claim_never_lends_jobs(
    registry: Registry, db_path: Path, clock: FakeClock, operator: sign.NodeKey, recheck: bool
) -> None:
    from test_registry import LOCAL

    a, b = checker(registry, clock), checker(registry, clock)
    assert a.heartbeat([LOCAL]).status_code == 200
    assert b.heartbeat([LOCAL]).status_code == 200
    add_job(db_path, operator, clock)
    if recheck:
        execution = claim(a)
        assert execution is not None
        assert hand_in(a, execution).status_code == 200
    pid = submit(a)
    answer = b.call("POST", protocol.CLAIM, {"jobs": False, "proposals": True})
    offered = protocol.Offer.model_validate(answer.json())
    assert offered.work is None
    assert offered.check is not None and offered.check.proposal == pid
    assert give(b, offered.check, record(b, offered.check)).status_code == 200
    assert b.call("POST", protocol.CLAIM, {"jobs": False}).json() == {"work": None, "check": None}


@pytest.mark.parametrize("crew", [False, True])
def test_node_claims_checks_without_job_capacity(
    tmp_path: Path, clock: FakeClock, monkeypatch: pytest.MonkeyPatch, crew: bool
) -> None:
    from test_node import Settings

    config = Settings()()
    key = sign.create_node_key(tmp_path / "node.key")
    conn = db.open_db(tmp_path / "node.db")
    machine = node.Node(conn, lambda: config, key_path=tmp_path / "node.key", clock=clock)
    monkeypatch.setattr(machine, "work", object() if crew else None)
    monkeypatch.setattr(jobs, "quota_left", lambda *args: not crew)
    monkeypatch.setattr(checks, "can_check", lambda: True)
    asked: list[protocol.Claim] = []
    lent = local_work(sign.NodeKey.generate(), clock)
    received: list[protocol.ProposalWork] = []
    monkeypatch.setattr(
        machine, "_check_proposal", lambda client, key, address, check, now: received.append(check)
    )

    def call(*args: Any) -> protocol.Offer:
        asked.append(args[5])
        return protocol.Offer(check=lent)

    monkeypatch.setattr(node, "_call", call)
    machine._claim(machine._http(), key, config, REGISTRY, clock(), float("inf"))
    assert asked == [protocol.Claim(jobs=False, proposals=True)]
    assert received == [lent]
    machine.close()
    conn.close()


def test_pending_check_still_extends_held_job(
    tmp_path: Path, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    key = sign.create_node_key(tmp_path / "node.key")
    conn = db.open_db(tmp_path / "node.db")
    config = Config(network=Network(enabled=True, registry=REGISTRY))
    machine = node.Node(conn, lambda: config, key_path=tmp_path / "node.key", clock=clock)
    machine.registry = REGISTRY
    machine.registered = True
    machine._trust = sign.Trust(key.public, clock=clock)
    held = SimpleNamespace(registry=REGISTRY, record=None, expires_at=clock(), lease="a" * 32)
    monkeypatch.setattr(jobs, "holding", lambda conn: [held])
    monkeypatch.setattr(jobs, "ending", lambda *args: None)
    monkeypatch.setattr(checks, "pending", lambda conn: [SimpleNamespace(registry=REGISTRY)])
    monkeypatch.setattr(machine, "_hand_check", lambda *args: None)
    extended: list[str] = []
    monkeypatch.setattr(jobs, "extended", lambda conn, lease, until: extended.append(lease))

    def call(*args: Any) -> protocol.Lease:
        assert args[4] == protocol.EXTEND
        return protocol.Lease(lease="a" * 32, expires_at=clock() + work.LEASE)

    monkeypatch.setattr(node, "_call", call)
    machine._jobs(clock(), idle=True)
    assert extended == ["a" * 32]
    machine.close()
    conn.close()


def test_claim_job_default_and_strictness() -> None:
    assert protocol.Claim().jobs is True
    with pytest.raises(ValueError):
        protocol.Claim.model_validate({"jobs": 1})
