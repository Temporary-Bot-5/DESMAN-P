"""Clusters in the registry (ADR 0006; ROADMAP T.4b): appointed by DESMAN, led through records."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest

from conftest import FakeClock
from desman.core import db
from desman.net import protocol, sign
from desman.registry import admin, clusters, store, work
from desman.registry.server import Registry
from test_registry import LOCAL, Node, db_path, registry, registry_cli, registry_dir, root, root_pub
from test_registry_work import hand_in, operator

__all__ = ["db_path", "operator", "registry", "registry_dir", "root", "root_pub"]  # fixtures

HOUR = timedelta(hours=1)
TERM = sign.APPOINTMENT_TERM.total_seconds()


@pytest.fixture
def appointer(
    registry_dir: Path, db_path: Path, root: sign.NodeKey, root_pub: Path, clock: FakeClock
) -> sign.NodeKey:
    made = admin.appointer_create(registry_dir / store.APPOINTER_KEY)
    key = sign.load_node_key(registry_dir / store.APPOINTER_KEY)
    assert made.fingerprint == key.fingerprint and made.created
    with store.connection(db_path) as conn:
        admin.assign(conn, root, key.public, {sign.Scope.APPOINT}, days=30, now=clock())
    return key


def add_task(db_path: Path, operator: sign.NodeKey, clock: FakeClock, **spec: Any) -> str:
    task = protocol.ClusterTask(
        goal=spec.pop("goal", "Write a short poem and review it"),
        acceptance_criteria=spec.pop("acceptance_criteria", "A poem of four lines"),
        **spec,
    )
    with store.connection(db_path) as conn:
        return admin.add_cluster_task(conn, operator, task, days=7, now=clock()).id


def member(registry: Registry, clock: FakeClock, *, lead: bool = False) -> Node:
    """A node that registered, lends a module and claimed once (saying whether it may lead)."""
    found = Node(registry, clock)
    assert found.register([LOCAL]).status_code == 200
    say(found, lead=lead)
    return found


def say(n: Node, *, lead: bool = False, jobs: bool = True) -> dict[str, Any] | None:
    answer = n.call("POST", protocol.CLAIM, {"jobs": jobs, "lead": lead})
    assert answer.status_code == 200, answer.text
    offer: dict[str, Any] | None = answer.json()["work"]
    return offer


def mine(n: Node) -> dict[str, Any]:
    answer = n.call("GET", protocol.CLUSTERS_MINE)
    assert answer.status_code == 200, answer.text
    state: dict[str, Any] = answer.json()
    return state


def row(db_path: Path, task: str) -> clusters.ClusterTaskRow:
    with store.connection(db_path) as conn:
        found = clusters.cluster_task(conn, task)
    assert found is not None
    return found


def actions(db_path: Path) -> list[str]:
    with store.connection(db_path) as conn:
        return [e.action for e in store.entries(conn)]


def cluster_record(
    lead: Node, kind: sign.RecordKind, body: dict[str, Any], *, issued_at: datetime | None = None
) -> sign.Signed:
    return sign.sign(
        lead.key, kind, body, issued_at=issued_at or lead.clock(), lifetime=timedelta(days=2)
    )


def refusal(answer: httpx.Response) -> str:
    assert answer.status_code != 200, answer.text
    error: str = answer.json()["error"]
    return error


def send(lead: Node, signed: sign.Signed) -> httpx.Response:
    return lead.call("POST", protocol.CLUSTER_RECORDS, {"record": signed.model_dump(mode="json")})


NODES: list[dict[str, Any]] = [
    {"description": "Write a poem of four lines", "acceptance_criteria": "Four lines"},
    {
        "description": "Polish the poem",
        "acceptance_criteria": "Four lines, polished",
        "depends_on": [0],
    },
]


def plan(lead: Node, state: dict[str, Any], *, version: int = 1, nodes: Any = None) -> sign.Signed:
    return cluster_record(
        lead,
        sign.RecordKind.CLUSTER_PLAN,
        {
            "cluster": state["cluster"],
            "task": state["task"]["record"]["id"],
            "version": version,
            "nodes": NODES if nodes is None else nodes,
        },
    )


def job(
    lead: Node, state: dict[str, Any], plan_id: str, node: int, *, version: int = 1, **over: Any
) -> sign.Signed:
    spec = {k: v for k, v in NODES[node].items() if k != "depends_on"} | over
    return cluster_record(
        lead,
        sign.RecordKind.CLUSTER_JOB,
        {
            "cluster": state["cluster"],
            "task": state["task"]["record"]["id"],
            "plan": plan_id,
            "version": version,
            "node": node,
            "job": spec,
        },
    )


def heartbeat(*nodes: Node) -> None:
    for n in nodes:
        assert n.heartbeat().status_code == 200


def formed(
    registry: Registry, db_path: Path, operator: sign.NodeKey, clock: FakeClock
) -> tuple[str, Node, Node, Node]:
    """A cluster task, and the three nodes it was appointed to (leader first)."""
    task = add_task(db_path, operator, clock)
    lead = member(registry, clock, lead=True)
    a = member(registry, clock)
    b = member(registry, clock)
    assert row(db_path, task).status == "running"
    return task, lead, a, b


# ── Forming a cluster ────────────────────────────────────────────────────────


def test_a_cluster_is_formed_once_three_nodes_are_free(
    registry: Registry,
    db_path: Path,
    operator: sign.NodeKey,
    appointer: sign.NodeKey,
    clock: FakeClock,
) -> None:
    task = add_task(db_path, operator, clock)
    a = member(registry, clock)
    lead = member(registry, clock, lead=True)
    assert row(db_path, task).status == "waiting"
    with store.connection(db_path) as conn:
        why = clusters.why_waiting(
            conn, appointer, registry.root, clock(), registry.online, row(db_path, task)
        )
    assert why == ("free", 2)

    b = member(registry, clock)
    found = row(db_path, task)
    assert found.status == "running"
    with store.connection(db_path) as conn:
        current = clusters.newest(conn, found.cluster)
    assert current is not None
    # The only node that may lead leads, though it is not the oldest.
    assert current.operator == lead.fingerprint
    assert sorted(current.members) == sorted([a.fingerprint, b.fingerprint])
    assert "appointed" in actions(db_path)

    state = mine(lead)
    assert state["role"] == "operator" and state["cluster"] == found.cluster
    assert state["task"]["record"]["id"] == task
    assert state["appointment"]["record"]["kind"] == "appointment"
    assert mine(a)["role"] == "member" and mine(a)["appointment"] is not None
    stranger = member(registry, clock)
    assert mine(stranger) == protocol.ClusterState().model_dump(mode="json")

    # A node checks the appointment with what the registry hands out.
    trusted = sign.Trust(sign.load_root(sign.ROOT_PUB), clock=clock)
    for kept in lead.call("GET", protocol.TRUST).json()["records"]:
        trusted.receive(kept)
    appointed = trusted.appointment(found.cluster)
    assert appointed is not None and appointed.fingerprint == lead.fingerprint


def test_the_leader_is_the_free_node_with_most_accepted_results(
    registry: Registry,
    db_path: Path,
    operator: sign.NodeKey,
    appointer: sign.NodeKey,
    clock: FakeClock,
) -> None:
    older = member(registry, clock, lead=True)
    better = member(registry, clock, lead=True)
    reviewer = member(registry, clock)
    # `better` did a job that another node accepted.
    with store.connection(db_path) as conn:
        job_id = admin.add_job(
            conn, operator, protocol.Job(description="x", acceptance_criteria="y"), days=1,
            now=clock(),
        ).id  # fmt: skip
    lent = say(better, lead=True)
    assert lent is not None and lent["task"]["record"]["id"] == job_id
    assert hand_in(better, lent).status_code == 200
    check = say(older, lead=True)
    assert check is not None and check["stage"] == "recheck"
    assert hand_in(older, check).json()["status"] == "accepted"

    task = add_task(db_path, operator, clock)
    say(reviewer)
    with store.connection(db_path) as conn:
        current = clusters.newest(conn, row(db_path, task).cluster)
    assert current is not None and current.operator == better.fingerprint


def test_who_is_not_free(
    registry: Registry,
    db_path: Path,
    operator: sign.NodeKey,
    appointer: sign.NodeKey,
    clock: FakeClock,
) -> None:
    task = add_task(db_path, operator, clock)
    lead = member(registry, clock, lead=True)
    second = member(registry, clock, lead=True)
    # Lends nothing; takes no jobs; silent too long: none of them counts.
    bare = Node(registry, clock)
    assert bare.register([]).status_code == 200
    say(bare)
    idle = Node(registry, clock)
    assert idle.register([LOCAL]).status_code == 200
    say(idle, jobs=False)
    assert row(db_path, task).status == "waiting"
    late = Node(registry, clock)
    assert late.register([LOCAL]).status_code == 200
    clock.advance(registry.online.total_seconds() + 1)
    heartbeat(lead, second)
    say(late)  # takes jobs now, but has not said it is alive for too long
    assert row(db_path, task).status == "waiting"
    heartbeat(late)
    say(late)
    assert row(db_path, task).status == "running"

    # A node in a running cluster is not free for another.
    other = add_task(db_path, operator, clock, goal="Another")
    member(registry, clock, lead=True)
    member(registry, clock)
    assert row(db_path, other).status == "waiting"
    member(registry, clock)
    assert row(db_path, other).status == "running"


def test_no_appointer_no_cluster(
    registry: Registry, db_path: Path, operator: sign.NodeKey, clock: FakeClock
) -> None:
    task = add_task(db_path, operator, clock)
    member(registry, clock, lead=True)
    member(registry, clock)
    member(registry, clock)
    assert row(db_path, task).status == "waiting"
    with store.connection(db_path) as conn:
        why = clusters.why_waiting(
            conn, None, registry.root, clock(), registry.online, row(db_path, task)
        )
    assert why[0] == "no_appointer"


def test_an_appointer_not_assigned_appoint_appoints_nobody(
    registry: Registry,
    registry_dir: Path,
    db_path: Path,
    operator: sign.NodeKey,
    clock: FakeClock,
) -> None:
    admin.appointer_create(registry_dir / store.APPOINTER_KEY)
    task = add_task(db_path, operator, clock)
    member(registry, clock, lead=True)
    member(registry, clock)
    member(registry, clock)
    assert row(db_path, task).status == "waiting"
    key = registry.appointer
    assert key is not None
    with store.connection(db_path) as conn:
        why = clusters.why_waiting(
            conn, key, registry.root, clock(), registry.online, row(db_path, task)
        )
    assert why[0] == "appointer_not_assigned"


# ── The operator's work ──────────────────────────────────────────────────────


def test_the_operator_plans_lends_and_finishes(
    registry: Registry,
    db_path: Path,
    operator: sign.NodeKey,
    appointer: sign.NodeKey,
    clock: FakeClock,
) -> None:
    task, lead, a, b = formed(registry, db_path, operator, clock)
    state = mine(lead)
    first = plan(lead, state)
    answer = send(lead, first)
    assert answer.status_code == 200, answer.text
    assert answer.json()["status"] == "running"
    plan_id = first.record.id

    # A node whose dependency is not accepted yet waits; a job that is not the plan's node too.
    assert refusal(send(lead, job(lead, state, plan_id, 1))) == "not_allowed"
    assert refusal(send(lead, job(lead, state, plan_id, 0, description="Other"))) == "malformed"
    lent0 = job(lead, state, plan_id, 0)
    assert send(lead, lent0).status_code == 200
    assert refusal(send(lead, job(lead, state, plan_id, 0))) == "conflict"  # lent already

    # The operator never executes its own cluster's job; a member does.
    assert say(lead, lead=True) is None
    work_ = say(a)
    assert work_ is not None and work_["task"]["record"]["id"] == lent0.record.id
    assert work_["task"]["record"]["kind"] == "cluster_job"
    assert hand_in(a, work_, output="Roses...").json()["status"] == "recheck"

    # The recheck goes to neither the executor nor the operator.
    assert say(a) is None
    assert say(lead, lead=True) is None
    check = say(b)
    assert check is not None and check["stage"] == "recheck"
    assert hand_in(b, check).json()["status"] == "accepted"

    state = mine(lead)
    assert [j["status"] for j in state["jobs"]] == ["accepted"]
    assert state["jobs"][0]["result"]["record"]["body"]["output"] == "Roses..."
    assert len(state["jobs"][0]["verdicts"]) == 1

    lent1 = job(lead, state, plan_id, 1)
    assert send(lead, lent1).status_code == 200
    # Not a member, not the operator: a stranger never executes a cluster job ...
    stranger = member(registry, clock)
    assert say(stranger) is None
    work_ = say(b)
    assert work_ is not None and work_["task"]["record"]["id"] == lent1.record.id
    hand_in(b, work_, output="Roses, polished")
    # ... but may recheck one: any node other than its executor and the operator.
    check = say(stranger)
    assert check is not None and check["stage"] == "recheck"
    hand_in(stranger, check)

    state = mine(lead)
    by_node = {j["job"]["record"]["body"]["node"]: j for j in state["jobs"]}
    wrong = cluster_record(
        lead,
        sign.RecordKind.CLUSTER_RESULT,
        {"cluster": state["cluster"], "task": task, "plan": plan_id, "version": 1,
         "result": by_node[0]["result"]["record"]["id"]},
    )  # fmt: skip
    assert refusal(send(lead, wrong)) == "not_allowed"  # not the last node's result
    final = cluster_record(
        lead,
        sign.RecordKind.CLUSTER_RESULT,
        {"cluster": state["cluster"], "task": task, "plan": plan_id, "version": 1,
         "result": by_node[1]["result"]["record"]["id"], "summary": "Done"},
    )  # fmt: skip
    assert send(lead, final).json()["status"] == "operator"
    found = row(db_path, task)
    assert (found.status, found.reason) == ("operator", "result")
    # The cluster dissolves: its operator has none now.
    assert mine(lead)["role"] == "none"

    with store.connection(db_path) as conn:
        settled = clusters.settle(conn, task, "accept", clock())
    assert settled.status == "accepted"
    log = actions(db_path)
    for action in ("cluster_task_added", "appointed", "cluster_record", "cluster_to_operator",
                   "cluster_settled"):  # fmt: skip
        assert action in log


def test_records_come_only_from_the_operator_in_force(
    registry: Registry,
    db_path: Path,
    operator: sign.NodeKey,
    appointer: sign.NodeKey,
    clock: FakeClock,
) -> None:
    task, lead, a, _ = formed(registry, db_path, operator, clock)
    state = mine(lead)
    # A member signs no plan; the operator signs no other cluster's.
    assert refusal(send(a, plan(a, state))) == "not_allowed"
    other = dict(state, cluster="e" * 32)
    assert refusal(send(lead, plan(lead, other))) == "not_allowed"
    # Dated back before its appointment.
    backdated = cluster_record(
        lead,
        sign.RecordKind.CLUSTER_PLAN,
        {"cluster": state["cluster"], "task": task, "version": 1, "nodes": NODES},
        issued_at=clock() - HOUR,
    )
    assert refusal(send(lead, backdated)) == "not_allowed"
    # Signed by someone else, handed in by the operator.
    forged = plan(a, state)
    assert refusal(send(lead, forged)) == "unknown_signer"

    first = plan(lead, state)
    assert send(lead, first).status_code == 200
    assert refusal(send(lead, first)) == "replay"
    # A second version needs a job that gave up (or the person's "again").
    assert refusal(send(lead, plan(lead, state, version=2))) == "not_allowed"
    assert refusal(send(lead, plan(lead, state, version=3))) == "malformed"


def test_a_job_that_gives_up_comes_back_to_the_operator_who_splits_it(
    registry: Registry,
    db_path: Path,
    operator: sign.NodeKey,
    appointer: sign.NodeKey,
    clock: FakeClock,
) -> None:
    task, lead, a, b = formed(registry, db_path, operator, clock)
    state = mine(lead)
    first = plan(lead, state)
    send(lead, first)
    lent = job(lead, state, first.record.id, 0)
    send(lead, lent)
    for n in (a, b):
        given = say(n)
        assert given is not None
        hand_in(n, given, outcome="failed", output="")
    with store.connection(db_path) as conn:
        assert work.job(conn, lent.record.id).status == "operator"  # type: ignore[union-attr]
        # Not in the global operator's inbox: the cluster's operator has it.
        assert work.jobs(conn, "operator", cluster=None) == []
    state = mine(lead)
    assert [(j["status"], j["reason"]) for j in state["jobs"]] == [("operator", "failed")]
    assert row(db_path, task).status == "running"

    split: list[dict[str, Any]] = [
        {"description": "Write two lines", "acceptance_criteria": "Two lines"},
        {"description": "Write two more", "acceptance_criteria": "Two lines", "depends_on": [0]},
        NODES[1] | {"depends_on": [1]},
    ]
    second = plan(lead, state, version=2, nodes=split)
    assert send(lead, second).status_code == 200, send(lead, second).text
    with store.connection(db_path) as conn:
        dropped = work.job(conn, lent.record.id)
    assert dropped is not None and (dropped.status, dropped.reason) == ("dropped", "superseded")
    # The old plan's jobs are refused now; the new one's are lent.
    assert refusal(send(lead, job(lead, state, first.record.id, 0))) == "not_allowed"
    piece = cluster_record(
        lead,
        sign.RecordKind.CLUSTER_JOB,
        {"cluster": state["cluster"], "task": task, "plan": second.record.id, "version": 2,
         "node": 0, "job": {k: v for k, v in split[0].items() if k != "depends_on"}},
    )  # fmt: skip
    assert send(lead, piece).status_code == 200


def test_a_job_that_runs_code_and_gives_up_goes_to_the_inbox(
    registry: Registry,
    db_path: Path,
    operator: sign.NodeKey,
    appointer: sign.NodeKey,
    clock: FakeClock,
) -> None:
    task, lead, a, b = formed(registry, db_path, operator, clock)
    state = mine(lead)
    nodes = [{"description": "Run it", "acceptance_criteria": "Exit 0", "runs_code": True}]
    first = plan(lead, state, nodes=nodes)
    send(lead, first)
    lent = cluster_record(
        lead,
        sign.RecordKind.CLUSTER_JOB,
        {"cluster": state["cluster"], "task": task, "plan": first.record.id, "version": 1,
         "node": 0, "job": {k: v for k, v in nodes[0].items()}},
    )  # fmt: skip
    assert send(lead, lent).status_code == 200
    for n in (a, b):
        given = say(n)
        assert given is not None
        hand_in(n, given, outcome="failed", output="")
    mine(lead)
    found = row(db_path, task)
    assert (found.status, found.reason) == ("operator", "job_given_up")

    # The person sends it back: the job is lent again from the start.
    with store.connection(db_path) as conn:
        again = clusters.settle(conn, task, "again", clock(), note="Try once more")
        assert again.status == "waiting" and again.note == "Try once more"
        assert work.job(conn, lent.record.id).status == "execute"  # type: ignore[union-attr]


# ── The role is a lease ──────────────────────────────────────────────────────


def test_the_operator_is_appointed_again_while_it_polls(
    registry: Registry,
    db_path: Path,
    operator: sign.NodeKey,
    appointer: sign.NodeKey,
    clock: FakeClock,
) -> None:
    task, lead, a, b = formed(registry, db_path, operator, clock)
    state = mine(lead)
    first = plan(lead, state)
    send(lead, first)
    lent = job(lead, state, first.record.id, 0)
    send(lead, lent)
    for minute in range(0, 60, 5):
        clock.advance(5 * 60)
        heartbeat(lead, a, b)
        state = mine(lead)
        assert state["role"] == "operator", minute
    assert len(state["appointments"]) > 2
    assert {x["record"]["body"]["key"] for x in state["appointments"]} == {
        sign.encode_public(lead.key.public)
    }
    assert "appointment_renewed" in actions(db_path)
    # A renewal keeps the poll that asked for it as the operator's heartbeat.
    seen = row(db_path, task).operator_seen
    assert seen is not None and db.parse_iso(seen) == clock()
    # A job signed under the first appointment, long past, is still lent and checked.
    given = say(a)
    assert given is not None and given["task"]["record"]["id"] == lent.record.id
    trusted = sign.Trust(sign.load_root(sign.ROOT_PUB), clock=clock)
    for kept in a.call("GET", protocol.TRUST).json()["records"]:
        trusted.receive(kept)
    trusted.receive(given["task"])
    assert row(db_path, task).status == "running"


def test_a_lost_operator_is_replaced_and_its_late_records_refused(
    registry: Registry,
    db_path: Path,
    operator: sign.NodeKey,
    appointer: sign.NodeKey,
    clock: FakeClock,
) -> None:
    task, lead, a, b = formed(registry, db_path, operator, clock)
    spare = member(registry, clock, lead=True)
    state = mine(lead)
    first = plan(lead, state)
    send(lead, first)

    clock.advance(TERM / 2)
    heartbeat(lead, a, b, spare)
    clock.advance(TERM / 2 + 1)  # not seen for one term
    heartbeat(lead, a, b, spare)
    found = row(db_path, task)
    assert found.lost == 1 and found.status == "running"
    successor = mine(spare)
    assert successor["role"] == "operator"
    # The successor continues from the records: the plan is there.
    assert [p["record"]["id"] for p in successor["plans"]] == [first.record.id]
    assert mine(lead)["role"] != "operator"
    late = job(lead, state, first.record.id, 0)
    assert refusal(send(lead, late)) == "not_allowed"
    assert send(spare, job(spare, successor, first.record.id, 0)).status_code == 200
    assert "operator_lost" in actions(db_path)


def test_three_operators_lost_send_the_cluster_task_to_the_inbox(
    registry: Registry,
    db_path: Path,
    operator: sign.NodeKey,
    appointer: sign.NodeKey,
    clock: FakeClock,
) -> None:
    task = add_task(db_path, operator, clock)
    leads = [member(registry, clock, lead=True) for _ in range(3)]
    others = [member(registry, clock) for _ in range(2)]
    seen: set[str] = set()
    for _ in range(3):
        with store.connection(db_path) as conn:
            current = clusters.newest(conn, row(db_path, task).cluster)
        assert current is not None and current.operator not in seen
        seen.add(current.operator)
        for half in range(2):
            clock.advance(TERM / 2 + half)
            heartbeat(*leads, *others)
            for n in (*leads, *others):
                say(n, lead=n in leads)
    found = row(db_path, task)
    assert (found.status, found.reason) == ("operator", "operators_lost")
    assert seen == {n.fingerprint for n in leads}


def test_a_node_that_stops_leading_is_replaced(
    registry: Registry,
    db_path: Path,
    operator: sign.NodeKey,
    appointer: sign.NodeKey,
    clock: FakeClock,
) -> None:
    task, lead, a, b = formed(registry, db_path, operator, clock)
    spare = member(registry, clock, lead=True)
    say(lead, lead=False)
    assert row(db_path, task).lost == 1
    assert mine(spare)["role"] == "operator"


def test_a_dropped_cluster_task_takes_its_jobs_with_it(
    registry: Registry,
    db_path: Path,
    operator: sign.NodeKey,
    appointer: sign.NodeKey,
    clock: FakeClock,
) -> None:
    task, lead, a, _ = formed(registry, db_path, operator, clock)
    state = mine(lead)
    first = plan(lead, state)
    send(lead, first)
    lent = job(lead, state, first.record.id, 0)
    send(lead, lent)
    assert say(a) is not None
    with store.connection(db_path) as conn:
        with pytest.raises(clusters.SettleError):
            clusters.settle(conn, task, "accept", clock())
        clusters.settle(conn, task, "drop", clock())
        assert work.job(conn, lent.record.id).status == "dropped"  # type: ignore[union-attr]
    assert mine(lead)["role"] == "none"


# ── Storage ──────────────────────────────────────────────────────────────────


def test_the_cluster_tables_are_append_only(
    registry: Registry,
    db_path: Path,
    operator: sign.NodeKey,
    appointer: sign.NodeKey,
    clock: FakeClock,
) -> None:
    _, lead, _, _ = formed(registry, db_path, operator, clock)
    send(lead, plan(lead, mine(lead)))
    with store.connection(db_path) as conn:
        for sql in (
            "UPDATE appointments SET operator = operator",
            "DELETE FROM appointments",
            "UPDATE cluster_records SET version = 2",
            "DELETE FROM cluster_records",
            "DELETE FROM cluster_tasks",
            "UPDATE cluster_tasks SET signed_json = '{}'",
        ):
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(sql)


def test_a_registry_of_the_twelfth_schema_is_upgraded(tmp_path: Path) -> None:
    path = tmp_path / "old.db"
    conn = db.connect(path)
    try:
        db.migrate(conn, store.MIGRATIONS[:12])
        conn.execute(
            "INSERT INTO nodes (fingerprint, public_key, status, registered_at, last_seen)"
            " VALUES (?, 'k', 'active', 'then', 'then')",
            ("a" * 64,),
        )
    finally:
        conn.close()
    with closing(store.open_registry(path)) as upgraded:
        found = store.node(upgraded, "a" * 64)
        assert found is not None and not found.takes_jobs and not found.lead
        assert clusters.cluster_tasks(upgraded) == []
        assert work.jobs(upgraded, cluster=None) == []


# ── The command line ─────────────────────────────────────────────────────────


def test_cli_appointer_cluster_add_clusters_inbox_settle(
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

    made = registry_cli("appointer", "create")
    assert made.exit_code == 0, made.output
    appointer_key = sign.load_node_key(registry_dir / store.APPOINTER_KEY)
    public = sign.encode_public(appointer_key.public)
    assert public in made.output and "--scope appoint" in made.output
    again = registry_cli("appointer", "create")
    assert again.exit_code == 0 and appointer_key.fingerprint[:16] in again.output

    added = registry_cli("cluster", "add", "--goal", "A [bold]poem[/bold]", "--acceptance", "Four")
    assert added.exit_code == 0, added.output
    listed = json.loads(registry_cli("clusters", "--json").output)
    assert len(listed) == 1 and listed[0]["status"] == "waiting"
    task = listed[0]["task"]
    assert "appointer is not assigned" in listed[0]["why"]

    with store.connection(db_path) as conn:
        admin.assign(conn, root, appointer_key.public, {sign.Scope.APPOINT}, days=30, now=clock())
    lead = member(registry, clock, lead=True)
    assert "1 of 3" in registry_cli("clusters").output
    member(registry, clock)
    member(registry, clock)
    shown = registry_cli("cluster", "show", task[:8])
    assert shown.exit_code == 0, shown.output
    assert "[bold]poem[/bold]" in shown.output and lead.fingerprint[:16] in shown.output
    assert "Nothing waits for the operator." in registry_cli("inbox").output

    refused = registry_cli("cluster", "settle", task[:8], "accept")
    assert refused.exit_code == 1 and "cannot be settled" in refused.output
    assert registry_cli("cluster", "settle", task[:8], "drop").exit_code == 0
    assert json.loads(registry_cli("clusters", "--json").output)[0]["status"] == "dropped"
    assert registry_cli("cluster", "show", "ffffffff").exit_code == 1
