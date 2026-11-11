"""T.4 demo: a goal becomes a cluster, reviewed jobs and a result, by polling alone."""

from collections.abc import Callable
from pathlib import Path

import pytest

from conftest import FakeClock
from desman import runs  # initialize the shared run/node imports
from desman.llm.fake import FakeLLM
from desman.net import sign
from desman.registry import clusters, store, work
from test_jobs import Machine, World, doer, machine, operator, registry_db, root, root_pub, world
from test_node_clusters import PLAN, SPLIT, run_round, task

__all__ = ["runs", "machine", "operator", "registry_db", "root", "root_pub", "world", "task"]


def test_cluster_demo(
    task: str,
    machine: Callable[[str, FakeLLM], Machine],
    world: World,
    registry_db: Path,
    clock: FakeClock,
) -> None:
    # Each device has its own database, key, config and donated model. All allow leading.
    devices = [
        machine(
            f"device-{i}",
            doer()
            .on("Plan task", PLAN, repeat_last=True)
            .on("Split subtask", SPLIT, repeat_last=True),
        )
        for i in range(4)
    ]
    for m in devices:
        m.node.cluster_model = m.crew.planner
        m.poll()
        clock.advance(1)

    # No test appointment or lending call: DESMAN forms and drives the cluster itself.
    for _ in range(24):
        run_round(devices, clock)
        with store.connection(registry_db) as conn:
            found = clusters.cluster_task(conn, task)
            assert found is not None
            if found.status == "operator":
                assert found.reason == "result"
                break
    else:
        pytest.fail(str([m.log() for m in devices]))

    with store.connection(registry_db) as conn:
        a = clusters.newest(conn, found.cluster)
        assert a is not None
        assert conn.execute("SELECT 1 FROM registry_log WHERE action='appointed'").fetchone()
        plans = clusters.records(conn, found.cluster, sign.RecordKind.CLUSTER_PLAN)
        assert len(plans) == 1 and plans[0].record.signer == a.operator
        assert len(work.jobs(conn, "accepted", cluster=task)) == 2
        for job in work.jobs(conn, "accepted", cluster=task):
            result = conn.execute("SELECT node FROM results WHERE id=?", (job.result,)).fetchone()
            verdict = conn.execute(
                "SELECT node FROM results WHERE job=? AND kind='verdict' AND outcome='accept'",
                (job.id,),
            ).fetchone()
            assert result and verdict
            assert verdict["node"] not in (result["node"], a.operator)
        assert clusters.records(conn, found.cluster, sign.RecordKind.CLUSTER_RESULT)
        assert clusters.settle(conn, task, "accept", clock()).status == "accepted"
