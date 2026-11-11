"""Deterministic cluster faults through real nodes and the in-process registry."""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from time import monotonic
from typing import Any

import pytest

from conftest import FakeClock
from desman import runs  # initialize the shared run/node imports
from desman.llm.fake import FakeLLM
from desman.net import jobs, node, protocol, sign
from desman.registry import admin, clusters, store, work
from test_jobs import Machine, World, doer, machine, operator, registry_db, root, root_pub, world
from test_node_clusters import PLAN, SPLIT, run_round, setup, snapshot, task

__all__ = ["runs", "machine", "operator", "registry_db", "root", "root_pub", "world", "task"]


def offer(
    m: Machine, signed: sign.Signed, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hostile registry offers a signed record; the real node must refuse it."""
    offered = protocol.Work(
        lease="a" * 32, expires_at=clock() + timedelta(minutes=10), stage="execute", task=signed
    )
    with monkeypatch.context() as patch:
        real_call = node._call

        def reply(*args: Any, **kwargs: Any) -> Any:
            return (
                protocol.Offer(work=offered)
                if args[4] == protocol.CLAIM
                else real_call(*args, **kwargs)
            )

        patch.setattr(node, "_call", reply)
        before = len(jobs.holding(m.conn))
        m.node._claim(
            m.node._http(), m.key, m.settings(), "https://registry.test", clock(), monotonic() + 10
        )
        assert len(jobs.holding(m.conn)) == before
    assert any(a == "work_refused" for a, _ in m.log())


@pytest.mark.parametrize("assigned", [False, True])
def test_forged_appointment(
    task: str,
    machine: Callable[[str, FakeLLM], Machine],
    world: World,
    registry_db: Path,
    root: sign.NodeKey,
    clock: FakeClock,
    monkeypatch: pytest.MonkeyPatch,
    assigned: bool,
) -> None:
    machines = setup(machine, clock)
    attacker = machine("attacker", doer())
    attacker.poll()
    state = snapshot(machines[0], world, clock)
    assert state.cluster is not None
    forger = sign.NodeKey.generate()
    with store.connection(registry_db) as conn:
        if assigned:
            admin.assign(conn, root, forger.public, {sign.Scope.TASKS}, days=1, now=clock())
        trust = store.trust(conn, root.public, clock())
        forged = sign.appoint(
            forger,
            attacker.key.public,
            cluster=state.cluster,
            task=task,
            members=[machines[1].key.fingerprint],
            issued_at=clock(),
        )
        with pytest.raises(sign.Refused):
            trust.receive(forged.model_dump(mode="json"))
        signed = sign.sign(
            attacker.key,
            sign.RecordKind.CLUSTER_JOB,
            protocol.ClusterJob(
                cluster=state.cluster,
                task=task,
                plan=state.plans[0].record.id,
                version=1,
                node=0,
                job=protocol.Job(description="bad", acceptance_criteria="bad"),
            ).model_dump(mode="json"),
            issued_at=clock(),
            lifetime=timedelta(hours=1),
        )
        caller = store.node(conn, attacker.key.fingerprint)
        assert caller is not None
        with pytest.raises(sign.Refused):
            clusters.hand_in(
                conn, caller, protocol.ClusterHandIn(record=signed), root.public, clock()
            )
    offer(machines[1], signed, clock, monkeypatch)


def test_appointment_over_thirty_minutes(
    task: str,
    machine: Callable[[str, FakeLLM], Machine],
    world: World,
    registry_db: Path,
    clock: FakeClock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    machines = setup(machine, clock)
    state = snapshot(machines[0], world, clock)
    assert state.cluster is not None
    appointer = sign.load_node_key(registry_db.parent / store.APPOINTER_KEY)
    forged = sign.appoint(
        appointer,
        machines[0].key.public,
        cluster=state.cluster,
        task=task,
        members=[m.key.fingerprint for m in machines[1:]],
        issued_at=clock(),
        lifetime=timedelta(minutes=30, seconds=1),
    )
    m = machines[1]
    assert m.node._trust is not None
    with pytest.raises(sign.Refused):
        m.node._trust.receive(forged.model_dump(mode="json"))
    # The same malformed appointment in the registry's snapshot is refused by Node.poll.
    bad = state.model_copy(update={"appointment": forged, "appointments": [forged]})
    clock.advance(120)
    with monkeypatch.context() as patch:
        real = node._call

        def reply(*args: Any, **kwargs: Any) -> Any:
            return bad if args[4] == protocol.CLUSTERS_MINE else real(*args, **kwargs)

        patch.setattr(node, "_call", reply)
        machines[0].poll()
    assert any(a == "record_refused" for a, _ in machines[0].log())


@pytest.mark.parametrize("attack", ["cluster", "task", "ordinary_task", "appointment"])
def test_operator_outside_its_authority(
    task: str,
    machine: Callable[[str, FakeLLM], Machine],
    world: World,
    registry_db: Path,
    root: sign.NodeKey,
    clock: FakeClock,
    monkeypatch: pytest.MonkeyPatch,
    attack: str,
) -> None:
    machines = setup(machine, clock)
    state = snapshot(machines[0], world, clock)
    assert state.cluster is not None
    body = protocol.ClusterJob(
        cluster="f" * 32 if attack == "cluster" else state.cluster,
        task="f" * 32 if attack == "task" else task,
        plan=state.plans[0].record.id,
        version=1,
        node=0,
        job=protocol.Job(description="bad", acceptance_criteria="bad"),
    )
    kind = sign.RecordKind.CLUSTER_JOB
    value = body.model_dump(mode="json")
    if attack == "ordinary_task":
        kind, value = sign.RecordKind.TASK, body.job.model_dump(mode="json")
    elif attack == "appointment":
        kind = sign.RecordKind.APPOINTMENT
        value = sign.Appointment(
            cluster=state.cluster,
            task=task,
            key=sign.encode_public(machines[1].key.public),
            members=[machines[2].key.fingerprint],
        ).model_dump(mode="json")
    signed = sign.sign(
        machines[0].key, kind, value, issued_at=clock(), lifetime=timedelta(minutes=20)
    )
    with store.connection(registry_db) as conn:
        caller = store.node(conn, machines[0].key.fingerprint)
        assert caller is not None
        with pytest.raises(sign.Refused):
            clusters.hand_in(
                conn, caller, protocol.ClusterHandIn(record=signed), root.public, clock()
            )
    offer(machines[1], signed, clock, monkeypatch)


def test_revoked_appointer_stops_cluster(
    task: str,
    machine: Callable[[str, FakeLLM], Machine],
    world: World,
    registry_db: Path,
    root: sign.NodeKey,
    operator: sign.NodeKey,
    root_pub: Path,
    clock: FakeClock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    machines = setup(machine, clock)
    with store.connection(registry_db) as conn:
        second = admin.add_cluster_task(
            conn,
            operator,
            protocol.ClusterTask(goal="A second cluster", acceptance_criteria="Done"),
            days=7,
            now=clock(),
            root_pub=root_pub,
        ).id
    for i in range(3, 6):
        m = machine(
            str(i),
            doer()
            .on("Plan task", PLAN, repeat_last=True)
            .on("Split subtask", SPLIT, repeat_last=True),
        )
        m.node.cluster_model = m.crew.planner
        machines.append(m)
        m.poll()
        clock.advance(1)
    clock.advance(120)
    machines[0].poll()
    state = snapshot(machines[0], world, clock)
    assert state.cluster is not None and state.jobs
    appointer = sign.load_node_key(registry_db.parent / store.APPOINTER_KEY)
    with store.connection(registry_db) as conn:
        admin.revoke(conn, root, appointer.fingerprint, "test", now=clock())
        count = conn.execute("SELECT count(*) FROM cluster_records").fetchone()[0]
        appointments = conn.execute("SELECT count(*) FROM appointments").fetchone()[0]
        assert len(clusters.cluster_tasks(conn, "running")) == 2
    # Every node's next heartbeat fetches the newly signed trust list before supervision.
    clock.advance(protocol.HEARTBEAT_SECONDS)
    signed_after_revocation: list[sign.RecordKind] = []
    real_sign = sign.sign

    def observe_signing(*args: Any, **kwargs: Any) -> sign.Signed:
        if args[1] in sign.CLUSTER_KINDS:
            signed_after_revocation.append(args[1])
        return real_sign(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(sign, "sign", observe_signing)
        for _ in range(18):
            run_round(machines, clock)
    assert not signed_after_revocation
    with store.connection(registry_db) as conn:
        assert conn.execute("SELECT count(*) FROM cluster_records").fetchone()[0] == count
        assert conn.execute("SELECT count(*) FROM appointments").fetchone()[0] == appointments
        for task_id in (task, second):
            found = clusters.cluster_task(conn, task_id)
            assert found is not None and found.status == "waiting"
            why, _ = clusters.why_waiting(
                conn, appointer, root.public, clock(), clusters.ONLINE, found
            )
            assert why == "appointer_not_assigned"
    # Trust refresh has invalidated the revoked appointer's appointments on each device.
    offer(machines[1], state.jobs[0].job, clock, monkeypatch)
    from click.testing import CliRunner

    from desman.registry import cli as registry_cli

    monkeypatch.setattr(registry_cli, "_now", clock)
    monkeypatch.setattr(registry_cli, "_root", lambda: root.public)
    result = CliRunner().invoke(
        registry_cli.cli, ["clusters"], env={"DESMAN_REGISTRY_DIR": str(registry_db.parent)}
    )
    assert result.exit_code == 0, result.output
    assert task[:8] in result.output

    assert second[:8] in result.output
    # A new root-assigned appointer restores automatic appointment, without manual lending.
    key_path = registry_db.parent / store.APPOINTER_KEY
    key_path.unlink()
    replacement = sign.create_node_key(key_path)
    with store.connection(registry_db) as conn:
        admin.assign(conn, root, replacement.public, {sign.Scope.APPOINT}, days=30, now=clock())
    # Replacing the online key also restarts the server that held the old private key.
    world.registry = type(world.registry)(registry_db, clock=clock, root=root_pub)
    for _ in range(2):
        run_round(machines, clock)
    with store.connection(registry_db) as conn:
        assert conn.execute("SELECT count(*) FROM appointments").fetchone()[0] > appointments


def test_operator_killed_mid_cluster(
    task: str,
    machine: Callable[[str, FakeLLM], Machine],
    world: World,
    registry_db: Path,
    clock: FakeClock,
) -> None:
    machines = setup(machine, clock)
    extra = machine(
        "replacement",
        doer().on("Plan task", PLAN, repeat_last=True).on("Split subtask", SPLIT, repeat_last=True),
    )
    extra.node.cluster_model = extra.crew.planner
    machines.append(extra)
    clock.advance(120)
    machines[0].poll()  # operator already signed the plan and first job
    with store.connection(registry_db) as conn:
        found = clusters.cluster_task(conn, task)
        assert found is not None
        before = clusters.latest_plan(conn, found.cluster)
    clock.advance(sign.APPOINTMENT_TERM.total_seconds() + 1)
    survivors = machines[1:]
    for _ in range(32):
        run_round(survivors, clock)
        with store.connection(registry_db) as conn:
            found = clusters.cluster_task(conn, task)
            assert found is not None
            if found.status == "operator":
                assert found.reason == "result"
                assert clusters.latest_plan(conn, found.cluster) == before
                a = clusters.newest(conn, found.cluster)
                assert a is not None and a.operator != machines[0].key.fingerprint
                records = work.jobs(conn, "accepted", cluster=task)
                assert len(records) == 2
                assert len({j.cluster_job.node for j in records if j.cluster_job}) == 2
                break
    else:
        pytest.fail(str([m.log() for m in machines]))


def test_no_member_lease_between_appointments(
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
        found = clusters.cluster_task(conn, task)
        assert found is not None
        assert work.jobs(conn, cluster=task)
        before = conn.execute("SELECT count(*) FROM leases").fetchone()[0]
    clock.advance(sign.APPOINTMENT_TERM.total_seconds() + 1)
    # Only one device returns: no replacement can be formed yet. A historical job's
    # signature still holds, but there is no current appointment authorizing a lease.
    machines[1].poll()
    with store.connection(registry_db) as conn:
        assert conn.execute("SELECT count(*) FROM leases").fetchone()[0] == before
        assert clusters.cluster_task(conn, task).status == "waiting"  # type: ignore[union-attr]
