"""The bodies of cluster records (ADR 0006, ROADMAP T.4a)."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from desman.net import protocol
from desman.net.protocol import ClusterJob, ClusterPlan, ClusterResult, ClusterTask, Job

CLUSTER = "c" * 32
TASK = "d" * 32
PLAN = "e" * 32


def node(*depends_on: int) -> dict[str, Any]:
    return {"description": "do", "acceptance_criteria": "done", "depends_on": list(depends_on)}


def plan(nodes: list[dict[str, Any]], version: int = 1) -> ClusterPlan:
    return ClusterPlan.model_validate(
        {"cluster": CLUSTER, "task": TASK, "version": version, "nodes": nodes}
    )


def test_a_plan_is_a_graph_in_order() -> None:
    made = plan([node(), node(0), node(0, 1)])
    assert [n.depends_on for n in made.nodes] == [[], [0], [0, 1]]
    for nodes in ([node(0)], [node(), node(2), node()], [node(), node(0, 0)], [node(-1)]):
        with pytest.raises(ValidationError):
            plan(nodes)


def test_a_plan_has_limits() -> None:
    with pytest.raises(ValidationError):
        plan([])
    with pytest.raises(ValidationError):
        plan([node() for _ in range(protocol.MAX_PLAN_NODES + 1)])
    with pytest.raises(ValidationError):
        plan([node()], version=protocol.MAX_PLAN_VERSIONS + 1)


def test_a_cluster_task_keeps_its_inputs_small() -> None:
    ClusterTask(goal="g", acceptance_criteria="c", inputs={"a": "x"})
    with pytest.raises(ValidationError):
        ClusterTask(goal="g", acceptance_criteria="c", inputs={"a": "x" * protocol.MAX_INPUT_BYTES})
    with pytest.raises(ValidationError):
        ClusterTask(goal="g", acceptance_criteria="c", extra=1)  # type: ignore[call-arg]


def test_a_cluster_job_and_result_name_their_cluster() -> None:
    job = ClusterJob(
        cluster=CLUSTER,
        task=TASK,
        plan=PLAN,
        version=1,
        node=0,
        job=Job(description="d", acceptance_criteria="c"),
    )
    assert job.job.kind == "text"
    with pytest.raises(ValidationError):
        ClusterJob.model_validate(job.model_dump() | {"node": protocol.MAX_PLAN_NODES})
    ClusterResult(cluster=CLUSTER, task=TASK, plan=PLAN, version=1, result="f" * 32)
    with pytest.raises(ValidationError):
        ClusterResult(cluster=CLUSTER, task=TASK, plan=PLAN, version=1, result="nothex")
