"""Node clusters against the real in-process registry, plus bounded protocol conversions."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from time import monotonic
from typing import Any, Literal
from unittest import mock

import pytest
from click.testing import CliRunner

from conftest import FakeClock
from desman import commands, global_cli
from desman.cli import cli
from desman.config import Config, ConfigManager, default_config_path, upgrade
from desman.core import db, netlog
from desman.llm.fake import FakeLLM, Quota, RateLimit, ServerError, Timeout
from desman.net import cluster, jobs, node, protocol, sign
from desman.registry import admin, clusters, store, work
from test_jobs import Machine, World, doer, machine, operator, registry_db, root, root_pub, world

__all__ = ["machine", "operator", "registry_db", "root", "root_pub", "world"]

PLAN = json.dumps(
    {
        "subtasks": [
            {"description": "Write a poem", "acceptance_criteria": "Four lines"},
            {
                "description": "Polish the poem",
                "acceptance_criteria": "Polished",
                "depends_on": [0],
            },
        ]
    }
)
SPLIT = json.dumps(
    {
        "subtasks": [
            {"description": "First half", "acceptance_criteria": "First"},
            {"description": "Second half", "acceptance_criteria": "Second", "depends_on": [0]},
        ]
    }
)


@pytest.fixture
def task(
    registry_db: Path, root: sign.NodeKey, operator: sign.NodeKey, clock: FakeClock, root_pub: Path
) -> str:
    path = registry_db.parent / store.APPOINTER_KEY
    sign.create_node_key(path)
    appointer = sign.load_node_key(path)
    with store.connection(registry_db) as conn:
        admin.assign(conn, root, appointer.public, {sign.Scope.APPOINT}, days=30, now=clock())
        return admin.add_cluster_task(
            conn,
            operator,
            protocol.ClusterTask(goal="Write a poem", acceptance_criteria="Four polished lines"),
            days=7,
            now=clock(),
            root_pub=root_pub,
        ).id


def setup(machine: Callable[[str, FakeLLM], Machine], clock: FakeClock) -> list[Machine]:
    machines = [
        machine(
            str(i),
            doer()
            .on("Plan task", PLAN, repeat_last=True)
            .on("Split subtask", SPLIT, repeat_last=True),
        )
        for i in range(3)
    ]
    for m in machines:
        m.node.cluster_model = m.crew.planner
        m.poll()
        clock.advance(1)
    clock.advance(120)
    for m in machines:
        m.poll()
    return machines


def run_round(machines: list[Machine], clock: FakeClock) -> None:
    clock.advance(120)
    for m in machines:
        m.poll()
        m.work()
        m.poll(idle=False)


def snapshot(m: Machine, world: World, clock: FakeClock) -> protocol.ClusterState:
    assert m.node._trust is not None
    return node._call(
        m.node._http(),
        m.key,
        "https://registry.test",
        "GET",
        protocol.CLUSTERS_MINE,
        None,
        protocol.ClusterState,
        clock(),
        __import__("time").monotonic() + 10,
    )


def test_config_upgrade_and_lead_command() -> None:
    old = {"schema_version": 8, "network": {"enabled": True}}
    assert Config.model_validate(upgrade(old)).network.lead
    manager = ConfigManager(default_config_path())
    manager.config_path.parent.mkdir(parents=True, exist_ok=True)
    manager.config_path.write_text(json.dumps(old))
    config = manager.load()
    assert config.schema_version == 9 and config.network.lead
    for state in ("off", "on"):
        result = CliRunner().invoke(cli, ["global", "lead", state])
        assert result.exit_code == 0, result.output
        assert manager.load().network.lead == (state == "on")
    entry = commands.find("global lead")
    assert entry is not None and entry.kind is commands.Kind.CHANGE
    with global_cli.journal(None) as conn:
        assert [e.details["on"] for e in netlog.entries(conn) if e.action == "lead"] == [
            False,
            True,
        ]


def test_operator_end_to_end_and_restart(
    task: str,
    machine: Callable[[str, FakeLLM], Machine],
    world: World,
    registry_db: Path,
    clock: FakeClock,
    root_pub: Path,
) -> None:
    machines = setup(machine, clock)
    leader = machines[0]
    # Restart from signed state after its first plan. There is no local operator checkpoint.
    leader.node.close()
    leader.node = node.Node(
        leader.conn,
        leader.settings,
        key_path=leader.node._key_path,
        transport=world.transport,
        clock=clock,
        root=root_pub,
        work=True,
        cluster_model=leader.crew.planner,
    )
    for _ in range(16):
        run_round(machines, clock)
        with store.connection(registry_db) as conn:
            found = clusters.cluster_task(conn, task)
            assert found is not None
            if found.status == "operator":
                assert found.reason == "result"
                records = clusters.records(conn, found.cluster, sign.RecordKind.CLUSTER_JOB)
                assert len(records) == 2
                assert len({r.record.body["node"] for r in records}) == 2
                break
    else:
        pytest.fail(str([m.log() for m in machines]))
    taken = [d for m in machines for a, d in m.log() if a == "work_taken"]
    assert len(taken) == 4 and all("cluster" in d for d in taken)
    assert not any(a == "work_taken" for a, d in leader.log())
    assert any(a == "cluster_result" for a, d in leader.log())


def test_lead_off_does_not_poll(
    task: str, machine: Callable[[str, FakeLLM], Machine], world: World, clock: FakeClock
) -> None:
    machines = setup(machine, clock)
    leader = machines[0]
    leader.settings.config.network.lead = False
    before = len(world.requests)
    run_round([leader], clock)
    assert ("GET", protocol.CLUSTERS_MINE) not in world.requests[before:]
    assert any(a == "cluster_left" for a, d in leader.log())
    count = sum(a == "cluster_left" for a, d in leader.log())
    run_round([leader], clock)
    assert sum(a == "cluster_left" for a, d in leader.log()) == count
    with store.connection(world.registry.path) as conn:
        assert not store.node(conn, leader.key.fingerprint).lead  # type: ignore[union-attr]


def test_template_and_idle(
    task: str,
    machine: Callable[[str, FakeLLM], Machine],
    world: World,
    clock: FakeClock,
    registry_db: Path,
) -> None:
    machines = [machine(str(i), doer()) for i in range(3)]
    for m in machines:
        m.node.cluster_model = FakeLLM("bad", "bad", repeat_last=True)
        m.poll()
    clock.advance(120)
    machines[0].poll(idle=False)
    assert not machines[0].node.cluster_model.calls  # type: ignore[union-attr]
    for _ in range(3):
        run_round(machines, clock)
    with store.connection(registry_db) as conn:
        found = clusters.cluster_task(conn, task)
        assert found is not None
        plan = clusters.latest_plan(conn, found.cluster)
        assert plan is not None and len(plan.body.nodes) == 1


def test_given_back_job_splits_and_preserves_accepted(
    task: str,
    machine: Callable[[str, FakeLLM], Machine],
    world: World,
    clock: FakeClock,
    registry_db: Path,
) -> None:
    machines = setup(machine, clock)
    for _ in range(12):
        clock.advance(120)
        # Stop immediately after the second node is lent, and give it back.
        machines[0].poll()
        with store.connection(registry_db) as conn:
            found = clusters.cluster_task(conn, task)
            assert found is not None
            rows = work.jobs(conn, cluster=task)
            if len(rows) == 2:
                with db.transaction(conn):
                    work.to_operator(conn, rows[1], "failed", {})
                before = clusters.latest_plan(conn, found.cluster)
                break
        for m in machines[1:]:
            m.poll()
            m.work()
            m.poll(idle=False)
    else:
        pytest.fail("second node was not lent")
    assert before is not None
    clock.advance(120)
    machines[0].poll()
    with store.connection(registry_db) as conn:
        after = clusters.latest_plan(conn, found.cluster)
    assert after is not None and after.body.version == 2
    assert after.body.nodes[0] == before.body.nodes[0]
    assert len(after.body.nodes) == 3


def test_inputs_bound_escaped_bytes_and_files() -> None:
    values = {f"n-{i}": "\U0001f638" * 100_000 for i in range(20)}
    bounded = cluster.inputs(values)
    assert len(bounded) == 16 and all(v.endswith(cluster.MARKER) for v in bounded.values())
    assert protocol.Job(description="d", acceptance_criteria="c", inputs=bounded)


def test_migration_preserves_call_attribution_and_append_only() -> None:
    conn = db.connect(":memory:")
    db.migrate(conn, db.MIGRATIONS[:-1])
    roles = [None, "planner", "executor", "reviewer", "decomposer", "interlocutor"]
    for role in roles:
        conn.execute(
            "INSERT INTO model_calls(provider, model, prompt_hash, latency_ms, role) "
            "VALUES ('p','m','h',1,?)",
            (role,),
        )
    indexes = list(
        conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='index' AND tbl_name='model_calls'"
        )
    )
    db.migrate(conn)
    assert [r[0] for r in conn.execute("SELECT role FROM model_calls ORDER BY seq")] == roles
    assert (
        list(
            conn.execute(
                "SELECT name, sql FROM sqlite_master WHERE type='index' AND tbl_name='model_calls'"
            )
        )
        == indexes
    )
    conn.execute(
        "INSERT INTO model_calls(provider, model, prompt_hash, latency_ms, role, task_id) "
        "VALUES ('p','m','h',1,'network_planner','remote-id')"
    )
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute("UPDATE model_calls SET role='planner'")
    conn.close()


@pytest.mark.parametrize(
    ("stage", "candidate", "allowed", "renewed"),
    [
        ("execute", 0, False, False),
        ("execute", 1, True, False),
        ("execute", 3, False, False),
        ("recheck", 0, False, False),
        ("recheck", 1, False, False),
        ("recheck", 2, True, False),
        ("execute", 3, True, True),
        ("recheck", 0, False, True),
    ],
)
def test_member_role_checks(
    task: str,
    machine: Callable[[str, FakeLLM], Machine],
    world: World,
    clock: FakeClock,
    registry_db: Path,
    root: sign.NodeKey,
    monkeypatch: pytest.MonkeyPatch,
    stage: Literal["execute", "recheck"],
    candidate: int,
    allowed: bool,
    renewed: bool,
) -> None:
    machines = setup(machine, clock)
    clock.advance(120)
    machines[0].poll()  # lend the first plan node
    state = snapshot(machines[0], world, clock)
    assert state.jobs
    job = state.jobs[0].job
    if candidate == 3 or renewed:
        machines.append(machine("outsider", doer()))
        machines[-1].poll()
    m = machines[candidate]
    result = sign.sign(
        machines[1].key,
        sign.RecordKind.RESULT,
        protocol.Result(
            job=job.record.id, lease="a" * 32, outcome="done", output="poem"
        ).model_dump(mode="json"),
        issued_at=clock(),
        lifetime=timedelta(days=1),
    )
    offered = protocol.Work(
        lease="a" * 32,
        expires_at=clock() + timedelta(minutes=10),
        stage=stage,
        task=job,
        result=result if stage == "recheck" else None,
        result_key=sign.encode_public(machines[1].key.public) if stage == "recheck" else None,
    )
    m.node._trust = sign.Trust(root.public, clock=clock)
    with store.connection(registry_db) as conn:
        for signed in store.handed_out(conn, clock()):
            m.node._trust.receive(signed.model_dump(mode="json"))
    m.node._appointments = {a.record.id: a for a in state.appointments}
    if renewed:
        clock.advance(1)
        appointer = sign.load_node_key(registry_db.parent / store.APPOINTER_KEY)
        assert state.cluster is not None
        appointed = sign.appoint(
            appointer,
            machines[0 if stage == "execute" else 3].key.public,
            cluster=state.cluster,
            task=task,
            members=[
                machines[3 if stage == "execute" else 0].key.fingerprint,
                machines[2].key.fingerprint,
            ],
            issued_at=clock(),
            lifetime=sign.APPOINTMENT_TERM,
        )
        m.node._trust.receive(appointed.model_dump(mode="json"))
        m.node._appointments[appointed.record.id] = appointed
    real_call = node._call

    def reply(*args: Any, **kwargs: Any) -> Any:
        if args[4] == protocol.CLAIM:
            return protocol.Offer(work=offered)
        return real_call(*args, **kwargs)

    monkeypatch.setattr(node, "_call", reply)
    before = len(jobs.holding(m.conn))
    m.node._claim(
        m.node._http(), m.key, m.settings(), "https://registry.test", clock(), monotonic() + 10
    )
    assert (len(jobs.holding(m.conn)) > before) is allowed, m.log()
    assert any(a == ("work_taken" if allowed else "work_refused") for a, d in m.log())


def test_exhausted_versions_stuck_and_code_stuck(
    task: str, machine: Callable[[str, FakeLLM], Machine], world: World, clock: FakeClock
) -> None:
    machines = setup(machine, clock)
    clock.advance(120)
    machines[0].poll()
    state = snapshot(machines[0], world, clock)
    first = state.jobs[0].model_copy(update={"status": "operator"})
    for version, code, reason in [(3, False, "versions"), (1, True, "runs_code")]:
        plan = protocol.ClusterPlan.model_validate(state.plans[0].record.body)
        nodes = list(plan.nodes)
        nodes[0] = nodes[0].model_copy(update={"runs_code": code})
        plan = plan.model_copy(update={"version": version, "nodes": nodes})
        signed = sign.sign(
            machines[0].key,
            sign.RecordKind.CLUSTER_PLAN,
            plan.model_dump(mode="json"),
            issued_at=clock(),
            lifetime=timedelta(days=1),
        )
        revised = state.model_copy(update={"plans": [signed], "jobs": [first]})
        sent: list[sign.RecordKind] = []
        stuck: list[str] = []

        def send(kind: sign.RecordKind, body: Any, collected: list[sign.RecordKind] = sent) -> None:
            collected.append(kind)

        cluster.supervise(revised, FakeLLM(), send, stuck.append)
        assert not sent and stuck == [reason]


def test_planner_calls_count_against_lent_quota_and_appear_on_page(
    task: str, machine: Callable[[str, FakeLLM], Machine], world: World, clock: FakeClock
) -> None:
    from desman.core import calls
    from desman.gui.module_view import cards
    from test_jobs import Slow

    machines = [machine(str(i), doer()) for i in range(3)]
    leader = machines[0]
    model = Slow(FakeLLM(PLAN), leader.conn, clock, 0)
    leader.node.cluster_model = jobs.Quota(model, leader.conn, "ollama", leader.settings, clock)
    for m in machines:
        m.poll()
        clock.advance(1)
    clock.advance(120)
    leader.poll()
    row = leader.conn.execute("SELECT role, task_id FROM model_calls").fetchone()
    assert tuple(row) == (calls.Role.NETWORK_PLANNER.value, task)
    assert jobs.used_today(leader.conn, "ollama", clock())[0] == 1
    from desman.config import Donation

    leader.settings.config.network.donated["ollama"] = Donation(
        calls_per_day=1, tokens_per_day=100000
    )
    assert not jobs.quota_left(leader.conn, leader.settings(), "ollama", clock())
    assert leader.conn.execute("SELECT count(*) FROM quota_holds").fetchone()[0] == 0
    result = cards(leader.conn, leader.settings(), [])
    assert any(
        c["section"] == "network" and any(w["role"] == "network_planner" for w in c["work"])
        for c in result
    )


def test_role_expires_and_page_switch_uses_catalog(
    task: str, machine: Callable[[str, FakeLLM], Machine], clock: FakeClock
) -> None:
    from desman.gui import settings
    from desman.settings_actions import Network

    machines = setup(machine, clock)
    leader = machines[0]
    receipt = cluster.role(leader.conn, clock())
    assert receipt is not None and receipt["role"] == "operator"
    assert cluster.role(leader.conn, clock() + sign.APPOINTMENT_TERM) is None
    entry, args = settings.command(
        Network(section="network", operation="lead", state="off"), leader.settings()
    )
    assert (
        entry.name == "global lead" and isinstance(args, commands.LeadArgs) and args.state == "off"
    )


def test_splitting_nonterminal_keeps_accepted_indices() -> None:
    nodes = [
        protocol.PlanNode(description=str(i), acceptance_criteria="ok", depends_on=deps)
        for i, deps in enumerate([[], [], [0], [1, 2]])
    ]
    parts = [
        protocol.PlanNode(description="a", acceptance_criteria="a"),
        protocol.PlanNode(description="b", acceptance_criteria="b", depends_on=[0]),
    ]
    changed = cluster._replace(nodes, 0, parts, {1})
    assert changed is not None and changed[1] == nodes[1]
    assert protocol.ClusterPlan(cluster="a" * 32, task="b" * 32, version=2, nodes=changed)
    assert changed[-1].depends_on == [1, 3]
    assert len(changed) == 5


def test_snapshot_inspection_does_not_mark_work_taken() -> None:
    conn = db.open_db(":memory:")
    seen = jobs.Seen(conn)
    clock = FakeClock()
    with seen.inspect():
        assert seen.add("a" * 32, clock() + timedelta(days=1), clock())
    assert seen.add("a" * 32, clock() + timedelta(days=1), clock())
    with seen.inspect():
        assert seen.add("a" * 32, clock() + timedelta(days=1), clock())
    assert not seen.add("a" * 32, clock() + timedelta(days=1), clock())
    conn.close()


def test_global_operator_note_replans_once_preserving_accepted(
    task: str,
    machine: Callable[[str, FakeLLM], Machine],
    world: World,
    clock: FakeClock,
    registry_db: Path,
) -> None:
    machines = setup(machine, clock)
    for _ in range(16):
        run_round(machines, clock)
        with store.connection(registry_db) as conn:
            found = clusters.cluster_task(conn, task)
            assert found is not None
            if found.status == "operator":
                before = clusters.latest_plan(conn, found.cluster)
                clusters.settle(conn, task, "again", clock(), note="Make it rhyme")
                break
    else:
        pytest.fail("no result")
    assert before is not None
    for _ in range(5):
        run_round(machines, clock)
        with store.connection(registry_db) as conn:
            after = clusters.latest_plan(conn, found.cluster)
        if after is not None and after.body.version == 2:
            break
    assert after is not None and after.body.version == 2
    assert after.body.nodes[:2] == before.body.nodes
    run_round(machines, clock)
    with store.connection(registry_db) as conn:
        current = clusters.latest_plan(conn, found.cluster)
    assert current is not None and current.body.version == 2
    requests = [request for m in machines for request in m.crew.planner.calls]  # type: ignore[union-attr]
    assert any(
        "Make it rhyme" in message.content for request in requests for message in request.messages
    )


def test_lead_off_while_suspended_journals_once(
    task: str, machine: Callable[[str, FakeLLM], Machine], world: World, clock: FakeClock
) -> None:
    leader = setup(machine, clock)[0]
    leader.settings.config.network.lead = False
    leader.settings.config.network.donated.clear()
    before = len(world.requests)
    leader.poll()
    leader.poll()
    assert world.requests[before:] == []
    assert sum(a == "cluster_left" for a, d in leader.log()) == 1


@pytest.mark.parametrize("replan", [False, True])
@pytest.mark.parametrize("failure", [RateLimit(), Quota(), ServerError(), Timeout()])
def test_provider_failure_waits_for_next_poll(
    task: str,
    machine: Callable[[str, FakeLLM], Machine],
    world: World,
    clock: FakeClock,
    replan: bool,
    failure: Any,
) -> None:
    leader = setup(machine, clock)[0]
    state = snapshot(leader, world, clock)
    assert state.cluster is not None
    if replan:
        state = state.model_copy(
            update={
                "note": "Try again",
                "result": sign.sign(
                    leader.key,
                    sign.RecordKind.CLUSTER_RESULT,
                    protocol.ClusterResult(
                        cluster=state.cluster,
                        task=task,
                        plan=state.plans[-1].record.id,
                        version=1,
                        result="a" * 32,
                        summary="redo",
                    ).model_dump(mode="json"),
                    issued_at=clock(),
                    lifetime=timedelta(days=1),
                ),
            }
        )
    else:
        state = state.model_copy(update={"plans": []})
    assert state.cluster is not None
    monkey = pytest.MonkeyPatch()
    sent: list[sign.Signed] = []

    def contact(*args: Any, **kwargs: Any) -> Any:
        if args[3] == "GET":
            return state
        signed = args[5].record
        sent.append(signed)
        return protocol.ClusterReceived(
            cluster=str(state.cluster),
            record=signed.record.id,
            kind=signed.record.kind.value,
            status="active",
        )

    monkey.setattr(node, "_call", contact)
    model = FakeLLM(failure, PLAN, sleep=clock.sleep, clock=clock)
    leader.node.cluster_model = model
    before = leader.log()
    for _ in range(2):
        clock.advance(120)
        leader.node._clusters(clock(), True)
        if len(model.calls) == 1:
            assert leader.log() == before
    monkey.undo()
    assert len(model.calls) == 2
    assert len(sent) == 1
    assert len(sent[0].record.body["nodes"]) >= 2


@pytest.mark.parametrize("failed,accepted", [(1, {0}), (1, {0, 2}), (2, {0, 1})])
def test_replacement_removes_obsolete_slots(failed: int, accepted: set[int]) -> None:
    def n(name: str, deps: list[int]) -> protocol.PlanNode:
        return protocol.PlanNode(description=name, acceptance_criteria=name, depends_on=deps)

    old = [n("A", []), n("B", [0]), n("C", [1])]
    if accepted == {0, 2}:
        old = [n("A", []), n("B", [0]), n("parallel", [0]), n("C", [1, 2])]
    changed = cluster._replace(old, failed, [n("part1", []), n("part2", [0])], accepted)
    assert changed is not None
    assert len(changed) == len(old) + 1
    assert changed[:failed] == old[:failed]
    for i in accepted:
        assert changed[i] == old[i]
    assert len({n.description for n in changed}) == len(changed)
    assert changed[-1].description == ("part2" if failed == len(old) - 1 else "C")
    for i, item in enumerate(changed):
        assert all(d < i for d in item.depends_on)
        if item.description == "C":
            assert (
                next(j for j, n in enumerate(changed) if n.description == "part2")
                in item.depends_on
            )


def test_replacement_stuck_when_accepted_slot_blocks_parts() -> None:
    old = [protocol.PlanNode(description=str(i), acceptance_criteria="ok") for i in range(3)]
    parts = [old[0], old[1].model_copy(update={"depends_on": [0]})]
    # Accepted final result cannot move away from the plan's end.
    assert cluster._replace(old, 0, parts, {2}) is None


def test_dependent_job_also_gets_task_inputs(
    task: str,
    machine: Callable[[str, FakeLLM], Machine],
    world: World,
    clock: FakeClock,
) -> None:
    leader = setup(machine, clock)[0]
    state = snapshot(leader, world, clock)
    assert state.task is not None and state.plans and state.cluster is not None
    task_record = state.task.model_copy(
        update={
            "record": state.task.record.model_copy(
                update={
                    "body": {**state.task.record.body, "inputs": {"original": "context" * 4000}}
                }
            )
        }
    )
    plan = state.plans[-1]
    job = protocol.ClusterJob(
        cluster=state.cluster,
        task=task,
        plan=plan.record.id,
        version=1,
        node=0,
        job=protocol.Job(description="d", acceptance_criteria="a"),
    )
    signed_job = sign.sign(
        leader.key,
        sign.RecordKind.CLUSTER_JOB,
        job.model_dump(mode="json"),
        issued_at=clock(),
        lifetime=timedelta(days=1),
    )
    result = sign.sign(
        leader.key,
        sign.RecordKind.RESULT,
        protocol.Result(
            job=signed_job.record.id, lease="a" * 32, outcome="done", output="dependency" * 10000
        ).model_dump(mode="json"),
        issued_at=clock(),
        lifetime=timedelta(days=1),
    )
    state = state.model_copy(
        update={
            "task": task_record,
            "jobs": [protocol.ClusterJobState(job=signed_job, status="accepted", result=result)],
        }
    )
    sent: list[Any] = []
    cluster.supervise(
        state, FakeLLM(), lambda kind, body: sent.append(body), lambda why: pytest.fail(why)
    )
    assert len(sent) == 1
    assert list(sent[0].job.inputs) == ["node-0", "original"]
    assert sent[0].job.inputs == cluster.inputs(
        {"node-0": "dependency" * 10000, "original": "context" * 4000}
    )


def test_appointment_cache_follows_latest_trust_state(
    task: str,
    machine: Callable[[str, FakeLLM], Machine],
    world: World,
    clock: FakeClock,
) -> None:
    leader = setup(machine, clock)[0]
    state = snapshot(leader, world, clock)
    assert state.appointments
    leader.node._appointments = {a.record.id: a for a in state.appointments}
    records = protocol.TrustState(records=state.appointments)
    leader.node._receive("https://registry.test", leader.key, records)
    assert set(leader.node._appointments) == {a.record.id for a in records.records}
    leader.node._receive("https://registry.test", leader.key, protocol.TrustState(records=[]))
    assert leader.node._appointments == {}


def test_donation_quota_defers_cluster_planning(clock: FakeClock) -> None:
    from desman.config import Donation
    from desman.llm.base import QuotaExhausted

    conn = db.connect(":memory:")
    db.migrate(conn)
    config = Config()
    model = FakeLLM(PLAN)
    limited = jobs.Quota(model, conn, "ollama", lambda: config, clock)
    task = protocol.ClusterTask(goal="goal", acceptance_criteria="criteria")
    for _ in range(2):
        with pytest.raises(QuotaExhausted):
            cluster.nodes(limited, task, None)
    assert model.calls == []
    config.network.donated["ollama"] = Donation(calls_per_day=10, tokens_per_day=10000)
    assert len(cluster.nodes(limited, task, None)) == 2
    conn.close()


def test_overflow_inputs_are_folded_or_visibly_cut() -> None:
    values = {f"input-{i}": f"value-{i}" for i in range(24)}
    fitted = cluster.inputs(values)
    assert len(fitted) == 16
    assert "additional inputs" in fitted["input-15"]
    for i in range(16, 24):
        assert values[f"input-{i}"] in fitted["input-15"]
    values["input-15"] = "x" * 100000
    fitted = cluster.inputs(values)
    assert "additional inputs" in fitted["input-15"]
    assert cluster.MARKER in fitted["input-15"]
    protocol.Job(description="job", acceptance_criteria="ok", inputs=fitted)


def test_busy_operator_does_not_renew(
    task: str, machine: Callable[[str, FakeLLM], Machine], world: World, clock: FakeClock
) -> None:
    leader = setup(machine, clock)[0]
    before = world.requests.count(("GET", protocol.CLUSTERS_MINE))
    for _ in range(3):
        clock.advance(120)
        leader.poll(idle=False)
    assert world.requests.count(("GET", protocol.CLUSTERS_MINE)) == before


def test_ended_cluster_is_not_rejoined_from_trust_after_restart(
    task: str, machine: Callable[[str, FakeLLM], Machine], world: World, clock: FakeClock
) -> None:
    member = setup(machine, clock)[1]
    assert member.node._trust is not None
    details = cluster.role(member.conn, clock())
    assert details is not None
    appointment = member.node._trust.appointment(details["cluster"])
    assert appointment is not None
    cluster.left(member.conn, clock(), ended=True)
    before = member.log()
    for _ in range(3):
        cluster.joined(member.conn, appointment, "member", clock(), from_trust=True)
        cluster.left(member.conn, clock())
    assert member.log() == before
    assert cluster.role(member.conn, clock()) is None
    # A new explicit registry snapshot may reactivate the same cluster after settle again.
    cluster.joined(member.conn, appointment, "member", clock())
    assert cluster.role(member.conn, clock()) is not None
    renewed = replace(appointment, expires_at=appointment.expires_at + timedelta(minutes=15))
    cluster.joined(member.conn, renewed, "member", clock(), from_trust=True)
    current = cluster.role(member.conn, clock())
    assert current is not None
    assert current["expires_at"] == db.to_iso(renewed.expires_at)
    assert sum(a == "cluster_joined" for a, d in member.log()) == 2


@pytest.mark.parametrize("reason", [sign.Reason.NOT_ALLOWED, sign.Reason.MALFORMED])
def test_cluster_handin_refusal_does_not_backoff_global_work(
    task: str,
    machine: Callable[[str, FakeLLM], Machine],
    world: World,
    clock: FakeClock,
    monkeypatch: pytest.MonkeyPatch,
    reason: sign.Reason,
) -> None:
    leader = setup(machine, clock)[0]
    state = snapshot(leader, world, clock).model_copy(
        update={"plans": [], "jobs": [], "note": "retry"}
    )
    signed_ids: list[str] = []

    def contact(*args: Any, **kwargs: Any) -> Any:
        if args[3] == "GET":
            return state
        signed_ids.append(args[5].record.record.id)
        raise node._Problem(reason.value)

    monkeypatch.setattr(node, "_call", contact)
    before = len(leader.log())
    clock.advance(120)
    leader.node._clusters(clock(), True)
    assert len(signed_ids) == 1
    assert leader.node.failures == 0
    assert leader.node.registered
    refused = [(a, d) for a, d in leader.log()[before:] if a == "record_refused"]
    assert len(refused) == 1
    assert refused[0][1]["id"] == signed_ids[0]
    # Restarting the in-memory journal must not duplicate a receipt for the same record.
    leader.node._journaled.clear()
    clock.advance(120)
    leader.node._clusters(clock(), True)
    assert signed_ids == [signed_ids[0], signed_ids[0]]
    assert sum(a == "record_refused" for a, d in leader.log()[before:]) == 1
    assert leader.node.failures == 0


def test_note_on_relent_job_does_not_replan_or_block_lending(
    task: str, machine: Callable[[str, FakeLLM], Machine], world: World, clock: FakeClock
) -> None:
    leader = setup(machine, clock)[0]
    state = snapshot(leader, world, clock).model_copy(update={"note": "retry please", "jobs": []})
    sent: list[sign.RecordKind] = []
    cluster.supervise(state, FakeLLM(), lambda kind, body: sent.append(kind), pytest.fail)
    assert sent == [sign.RecordKind.CLUSTER_JOB]


@pytest.mark.parametrize("failure", [Timeout(), RateLimit(), Quota(), ServerError()])
def test_split_provider_failure_defers_without_stuck(
    task: str,
    machine: Callable[[str, FakeLLM], Machine],
    world: World,
    clock: FakeClock,
    failure: Any,
) -> None:
    from desman.llm.base import ProviderError

    leader = setup(machine, clock)[0]
    state = snapshot(leader, world, clock)
    # First lend a node, then present it as returned to its operator.
    clock.advance(120)
    leader.poll()
    state = snapshot(leader, world, clock)
    assert state.jobs
    state = state.model_copy(
        update={"jobs": [state.jobs[0].model_copy(update={"status": "operator"})]}
    )
    model = FakeLLM(failure, SPLIT, clock=clock, sleep=clock.sleep)
    sent: list[sign.RecordKind] = []
    with pytest.raises(ProviderError):
        cluster.supervise(state, model, lambda kind, body: sent.append(kind), pytest.fail)
    assert sent == []
    cluster.supervise(state, model, lambda kind, body: sent.append(kind), pytest.fail)
    assert sent == [sign.RecordKind.CLUSTER_PLAN]


def test_replan_prefix_nodes_all_feed_the_result(
    task: str, machine: Callable[[str, FakeLLM], Machine], world: World, clock: FakeClock
) -> None:
    leader = setup(machine, clock)[0]
    state = snapshot(leader, world, clock)
    assert state.cluster is not None
    # An independent node at index 1 was accepted while index 0 is still in flight.
    original = protocol.ClusterPlan(
        cluster=state.cluster,
        task=task,
        version=1,
        nodes=[
            protocol.PlanNode(description="running", acceptance_criteria="ok"),
            protocol.PlanNode(description="accepted", acceptance_criteria="ok"),
            protocol.PlanNode(description="combine", acceptance_criteria="ok", depends_on=[0, 1]),
        ],
    )
    old = state.plans[0].model_copy(
        update={
            "record": state.plans[0].record.model_copy(
                update={"body": original.model_dump(mode="json")}
            )
        }
    )
    job = protocol.ClusterJob(
        cluster=state.cluster,
        task=task,
        plan=old.record.id,
        version=1,
        node=1,
        job=protocol.Job(description="accepted", acceptance_criteria="ok"),
    )
    signed_job = sign.sign(
        leader.key,
        sign.RecordKind.CLUSTER_JOB,
        job.model_dump(mode="json"),
        issued_at=clock(),
        lifetime=timedelta(days=1),
    )
    result = sign.sign(
        leader.key,
        sign.RecordKind.CLUSTER_RESULT,
        protocol.ClusterResult(
            cluster=state.cluster,
            task=task,
            plan=old.record.id,
            version=1,
            result="a" * 32,
            summary="redo",
        ).model_dump(mode="json"),
        issued_at=clock(),
        lifetime=timedelta(days=1),
    )
    state = state.model_copy(
        update={
            "plans": [old],
            "result": result,
            "jobs": [protocol.ClusterJobState(job=signed_job, status="accepted")],
        }
    )
    sent: list[Any] = []
    cluster.supervise(state, FakeLLM(PLAN), lambda kind, body: sent.append(body), pytest.fail)
    changed = sent[0].nodes
    assert changed[1] == original.nodes[1]
    ancestors: set[int] = set()

    def visit(index: int) -> None:
        for dep in changed[index].depends_on:
            if dep not in ancestors:
                ancestors.add(dep)
                visit(dep)

    visit(len(changed) - 1)
    assert ancestors == set(range(len(changed) - 1))


@pytest.mark.parametrize("at_cap", [False, True])
@pytest.mark.parametrize("expired", [False, True])
def test_expired_given_up_job_replans_before_lending(
    task: str,
    machine: Callable[[str, FakeLLM], Machine],
    world: World,
    registry_db: Path,
    clock: FakeClock,
    monkeypatch: pytest.MonkeyPatch,
    at_cap: bool,
    expired: bool,
) -> None:
    if at_cap:
        monkeypatch.setattr(protocol, "MAX_PLAN_VERSIONS", 1)
    real_sign = sign.sign

    def short_job(*args: Any, **kwargs: Any) -> sign.Signed:
        if expired and args[1] is sign.RecordKind.CLUSTER_JOB and args[2]["node"] == 1:
            kwargs["lifetime"] = timedelta(seconds=121)
        return real_sign(*args, **kwargs)

    with mock.patch.object(sign, "sign", short_job):
        machines = setup(machine, clock)
        _expire_given_up(task, machines, world, registry_db, clock, at_cap, expired)


def _expire_given_up(
    task: str,
    machines: list[Machine],
    world: World,
    registry_db: Path,
    clock: FakeClock,
    at_cap: bool,
    expired: bool,
) -> None:
    for _ in range(12):
        clock.advance(120)
        machines[0].poll()
        with store.connection(registry_db) as conn:
            rows = work.jobs(conn, cluster=task)
            if len(rows) == 2:
                found = clusters.cluster_task(conn, task)
                assert found is not None
                before = clusters.latest_plan(conn, found.cluster)
                # The person may see an exhausted job even on an early plan (runs_code).
                with db.transaction(conn):
                    work.to_operator(conn, rows[1], "failed", {})
                    clock.advance(122)
                    conn.execute(
                        "UPDATE cluster_tasks SET status='operator', "
                        "reason='job_given_up' WHERE id=?",
                        (task,),
                    )
                with mock.patch.object(clusters, "_untried_members", return_value=False):
                    clusters.settle(conn, task, "again", clock(), note="Use a new ending")
                dropped = work.job(conn, rows[1].id)
                assert dropped is not None
                assert dropped.reason == ("expired" if expired else "no_member_left")
                assert clusters.cluster_task(conn, task).replan == (not at_cap)  # type: ignore[union-attr]
                accepted = rows[0]
                break
        for m in machines[1:]:
            m.poll()
            m.work()
            m.poll(idle=False)
    else:
        pytest.fail("second job was not lent")
    assert before is not None
    if at_cap:
        for _ in range(4):
            run_round(machines, clock)
        with store.connection(registry_db) as conn:
            latest = clusters.latest_plan(conn, found.cluster)
            assert latest is not None and latest.body.version == 1
            replacement = [
                j
                for j in work.jobs(conn, cluster=task)
                if j.id != rows[1].id and j.cluster_job and j.cluster_job.node == 1
            ]
            assert replacement
            assert work.job(conn, accepted.id) == accepted
        return
    for _ in range(5):
        run_round(machines, clock)
        with store.connection(registry_db) as conn:
            after = clusters.latest_plan(conn, found.cluster)
            if after is not None and after.body.version == 2:
                assert not clusters.cluster_task(conn, task).replan  # type: ignore[union-attr]
                assert work.job(conn, accepted.id) == accepted
                assert after.body.nodes[0] == before.body.nodes[0]
                assert after.body.nodes[1].description == "First half"
                assert len(after.body.nodes) == 3
                # No replacement job was signed against the obsolete plan.
                old_jobs = [
                    r
                    for r in clusters.records(conn, found.cluster, sign.RecordKind.CLUSTER_JOB)
                    if r.record.body["version"] == 1
                ]
                assert len(old_jobs) == 2
                break
    else:
        pytest.fail(str([m.log() for m in machines]))
    assert any(
        "Use a new ending" in msg.content
        for m in machines
        for req in m.crew.planner.calls  # type: ignore[union-attr]
        for msg in req.messages
    )


def test_old_registry_result_heuristic(
    task: str, machine: Callable[[str, FakeLLM], Machine], world: World, clock: FakeClock
) -> None:
    leader = setup(machine, clock)[0]
    state = snapshot(leader, world, clock)
    raw = state.model_dump(mode="json")
    raw.pop("replan")
    old = protocol.ClusterState.model_validate(raw)
    result = sign.sign(
        leader.key,
        sign.RecordKind.CLUSTER_RESULT,
        protocol.ClusterResult(
            cluster=state.cluster or "a" * 32,
            task=task,
            plan=old.plans[-1].record.id,
            version=1,
            result="b" * 32,
            summary="Try again",
        ).model_dump(mode="json"),
        issued_at=clock(),
        lifetime=timedelta(days=1),
    )
    sent: list[protocol.ClusterPlan] = []

    def send(kind: sign.RecordKind, body: Any) -> None:
        assert kind is sign.RecordKind.CLUSTER_PLAN
        sent.append(protocol.ClusterPlan.model_validate(body))

    cluster.supervise(
        old.model_copy(update={"result": result}), FakeLLM(PLAN), send, lambda why: pytest.fail(why)
    )
    assert sent[0].version == 2


@pytest.mark.parametrize("role", ["none", "member", "operator"])
def test_cluster_reply_compatible_without_replan(role: str) -> None:
    from pydantic import ConfigDict, create_model

    from desman.registry.server import _ok

    fields: dict[str, Any] = {
        name: (field.annotation, field)
        for name, field in protocol.ClusterState.model_fields.items()
        if name not in {"replan", "dropped_nodes"}
    }
    old = create_model(
        "OldClusterState",
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )
    state = protocol.ClusterState.model_validate({"role": role})
    reply = _ok(state)
    assert "replan" not in reply.body
    assert "dropped_nodes" not in reply.body
    old.model_validate(reply.body)
    changed = _ok(state.model_copy(update={"replan": True, "dropped_nodes": [1]}))
    assert changed.body["replan"] is True
    assert changed.body["dropped_nodes"] == [1]


@pytest.mark.parametrize("failure", ["unreachable", "stale"])
def test_claim_trust_refresh_does_not_abandon_lease(
    task: str,
    machine: Callable[[str, FakeLLM], Machine],
    world: World,
    registry_db: Path,
    clock: FakeClock,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    machines = setup(machine, clock)
    clock.advance(120)
    machines[0].poll()
    member = machines[2]
    real = node._call
    granted: list[protocol.Work] = []

    def call(*args: Any, **kwargs: Any) -> Any:
        if args[4] == protocol.TRUST and granted and failure == "unreachable":
            raise node._Unreachable("offline")
        reply = real(*args, **kwargs)
        if args[4] == protocol.CLAIM and reply.work is not None:
            granted.append(reply.work)
        return reply

    monkeypatch.setattr(node, "_call", call)
    monkeypatch.setattr(member.node, "_fresh", lambda address: False)
    member.node._claim(
        member.node._http(),
        member.key,
        member.settings(),
        "https://registry.test",
        clock(),
        monotonic() + 10,
    )
    assert granted
    assert any(
        action in {"work_taken", "work_refused"} and details["job"] == granted[0].task.record.id
        for action, details in member.log()
    )


def test_lending_uses_one_appointment_snapshot(
    task: str,
    machine: Callable[[str, FakeLLM], Machine],
    world: World,
    registry_db: Path,
    clock: FakeClock,
) -> None:
    machines = setup(machine, clock)
    clock.advance(120)
    machines[0].poll()
    with store.connection(registry_db) as conn:
        job = work.jobs(conn, "execute", cluster=task)[0]
        caller = store.node(conn, machines[1].key.fingerprint)
        assert caller is not None
        queries: list[str] = []
        conn.set_trace_callback(queries.append)
        assert work._cluster_lends_to(conn, job, caller, clock())
        appointment_queries = [q for q in queries if "FROM appointments" in q]
        assert len(appointment_queries) == 1
        assert all(
            field in appointment_queries[0] for field in ("issued_at", "expires_at", "members_json")
        )
