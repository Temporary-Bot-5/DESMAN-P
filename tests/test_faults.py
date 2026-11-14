"""Fault injection (ROADMAP 4.6): random worker and provider failures, repeatable by seed.

See tests/faults.py for the harness. Each seed is one simulated run of several workers on one
database, with crashes, hangs, provider failures and rejections where its random generator puts
them; at the end every invariant must hold. The network's runs (G.6c) are a registry and
several nodes with their own databases, with the registry down, nodes restarting, leases lost
and a revocation hidden. A failing seed is replayed exactly:

    DESMAN_FAULT_SEED=17 pytest tests/test_faults.py

`DESMAN_FAULT_SEEDS=500` runs that many seeds instead of the default few (a longer hunt).
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import pytest

from desman.core.states import TaskStatus
from faults import Faults, Report, Shape, simulate, simulate_network


def _seeds() -> list[int]:
    if os.environ.get("DESMAN_FAULT_SEED"):
        return [int(os.environ["DESMAN_FAULT_SEED"])]
    return list(range(int(os.environ.get("DESMAN_FAULT_SEEDS", "12"))))


def run(
    tmp_path: Path,
    seed: int,
    faults: Faults | None = None,
    shape: Shape | None = None,
    replay: str | None = None,
) -> Report:
    directory = tmp_path / f"seed-{seed}"
    directory.mkdir(parents=True)
    return simulate(directory, seed, faults, shape, replay=replay)


@pytest.mark.parametrize("seed", _seeds())
def test_every_invariant_holds_under_random_faults(tmp_path: Path, seed: int) -> None:
    report = run(tmp_path, seed)
    assert not report.problems, report.explain()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_every_invariant_holds_in_a_storm(tmp_path: Path, seed: int) -> None:
    storm = Faults(crash=0.1, hang=0.05, provider=0.5, retry=0.3)
    replay = f"pytest 'tests/test_faults.py::test_every_invariant_holds_in_a_storm[{seed}]'"
    report = run(tmp_path, seed, storm, Shape(tasks=4, workers=4), replay)
    assert not report.problems, report.explain()
    assert sum(report.faults.values()) > 20


def test_a_seed_replays_step_for_step(tmp_path: Path) -> None:
    first, again, other = run(tmp_path, 7), run(tmp_path / "again", 7), run(tmp_path, 8)
    assert first.trace == again.trace
    assert first.faults == again.faults and first.faults  # something did go wrong
    assert other.trace != first.trace
    # Faults and audit entries in the order they happened: a crash is followed by its recovery.
    crash = next(i for i, line in enumerate(first.trace) if ": crash: " in line)
    assert any("lease_abandoned" in line or "review_" in line for line in first.trace[crash:])
    assert any(" claim " in line or line.endswith(" claim") for line in first.trace[:crash])


def test_the_faults_and_their_recoveries_all_happen(tmp_path: Path) -> None:
    """The harness is not vacuous: across a few seeds, each kind of fault is injected and each
    way of recovering from it is taken."""
    faults: Counter[str] = Counter()
    actions: Counter[str] = Counter()
    statuses: Counter[TaskStatus] = Counter()
    # Plain executor prose is now a result, which changes the simulated fault schedule.
    # Keep enough seeds to exercise every recovery without weakening the assertions.
    for seed in range(16):
        report = run(tmp_path, seed)
        faults += report.faults
        actions += report.actions
        statuses.update(report.statuses.values())
    for point in ("model call", "model reply", "tool call", "step"):
        assert faults[f"crash: {point}"], point
    for kind in ("garbage", "timeout", "rate_limit", "server_error"):
        assert faults[f"provider: {kind}"], kind
    assert faults["hang"]
    for action in (
        "lease_abandoned",  # a dead worker's subtask, taken over at once
        "lease_expired",  # a hung one's, after its lease
        "work_discarded",  # the hung one's late result
        "review_interrupted",
        "tool_replayed",  # a rerun did not repeat the effect
        "pause_over",  # a rate limit waited out
        "preempted",  # the urgent task took over a running subtask
    ):
        assert actions[action], action
    assert statuses[TaskStatus.DONE]


def test_the_harness_catches_a_repeated_effect(tmp_path: Path) -> None:
    """A planted bug (an effect wrongly declared safe to rerun) is found, and named."""
    shape = Shape(planted_bug=True)
    replay = "pytest tests/test_faults.py::test_the_harness_catches_a_repeated_effect"
    found = [run(tmp_path, seed, Faults(crash=0.1), shape, replay) for seed in range(6)]
    problems = [p for report in found for p in report.problems]
    assert any("effect ran twice" in p for p in problems), problems
    caught = next(r for r in found if r.problems)
    assert f"seed {caught.seed}: " in caught.explain()
    assert f"replay: {replay}\n" in caught.explain()


# ── desman-global (G.6c) ─────────────────────────────────────────────────────


def _network_seeds() -> list[int]:
    """A network run is slower (a registry and three databases): fewer seeds by default."""
    if os.environ.get("DESMAN_FAULT_SEED") or os.environ.get("DESMAN_FAULT_SEEDS"):
        return _seeds()
    return list(range(4))


@pytest.mark.parametrize("seed", _network_seeds())
def test_every_network_invariant_holds_under_random_faults(tmp_path: Path, seed: int) -> None:
    report = simulate_network(tmp_path, seed)
    assert not report.problems, report.explain()
    assert sum(report.faults.values()) > 0


def test_the_network_harness_catches_a_node_that_ignores_its_trust_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from desman.net.node import Node

    with monkeypatch.context() as m:
        m.setattr(Node, "_fresh", lambda self, address: True)
        for seed in range(6):
            directory = tmp_path / str(seed)
            directory.mkdir()
            report = simulate_network(directory, seed)
            if any("not fresh" in p for p in report.problems):
                break
        else:
            pytest.fail("no seed caught the node that ignores its trust list")


# ── Cluster operators (T.4d) ────────────────────────────────────────────────


def _cluster_seeds() -> list[int]:
    return (
        _seeds()
        if (os.environ.get("DESMAN_FAULT_SEED") or os.environ.get("DESMAN_FAULT_SEEDS"))
        else [0, 1, 2]
    )


@pytest.fixture(scope="module")
def cluster_reports(tmp_path_factory: pytest.TempPathFactory) -> dict[int, Report]:
    from faults import simulate_clusters

    return {
        seed: simulate_clusters(tmp_path_factory.mktemp(f"clusters-{seed}"), seed)
        for seed in _cluster_seeds()
    }


@pytest.mark.parametrize("seed", _cluster_seeds())
def test_every_cluster_invariant_holds_under_random_faults(
    cluster_reports: dict[int, Report], seed: int
) -> None:
    report = cluster_reports[seed]
    assert not report.problems, report.explain()


def test_cluster_fault_kinds_happen(cluster_reports: dict[int, Report]) -> None:
    counts: Counter[str] = Counter()
    for report in cluster_reports.values():
        counts.update(report.faults)
    if os.environ.get("DESMAN_FAULT_SEED"):
        return  # A replay checks invariants, not cross-seed coverage.
    for kind in (
        "operator killed",
        "member gone",
        "late record",
        "registry down",
        "clock jump",
        "reject",
    ):
        assert counts[kind], kind
    assert sum(n for kind, n in counts.items() if kind.startswith("provider:"))


def test_cluster_seed_replays(tmp_path: Path, cluster_reports: dict[int, Report]) -> None:
    from faults import simulate_clusters

    seed = next(iter(cluster_reports))
    first = cluster_reports[seed]
    again = simulate_clusters(tmp_path, seed)
    assert first.trace == again.trace
    assert first.faults == again.faults
    assert not again.problems, again.explain()


def test_cluster_harness_catches_a_late_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sqlite3
    from datetime import datetime

    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    from desman.net import protocol, sign
    from desman.registry import clusters, store
    from faults import simulate_clusters

    real = clusters.hand_in

    def broken(
        conn: sqlite3.Connection,
        caller: store.Node,
        given: protocol.ClusterHandIn,
        root: Ed25519PublicKey,
        now: datetime,
    ) -> protocol.ClusterReceived:
        try:
            return real(conn, caller, given, root, now)
        except sign.Refused:
            current = clusters.newest(conn, given.record.record.body["cluster"])
            if current is None or current.operator == caller.fingerprint:
                raise
            return protocol.ClusterReceived(
                cluster=given.record.record.body["cluster"],
                record=given.record.record.id,
                kind=given.record.record.kind.value,
                status="running",
            )

    monkeypatch.setattr(clusters, "hand_in", broken)
    report = simulate_clusters(tmp_path, 0)
    assert "late cluster record accepted" in report.problems, report.explain()


@pytest.mark.parametrize("reason", ["versions", "job_given_up"])
def test_cluster_harness_rejects_unexplained_inbox(tmp_path: Path, reason: str) -> None:
    from desman.registry import store
    from faults import ClusterFaults, ClusterShape, ClusterSimulation

    sim = ClusterSimulation(tmp_path, 0, ClusterFaults(), ClusterShape(nodes=0, tasks=1))
    try:
        with store.connection(sim.registry_db) as conn:
            conn.execute("UPDATE cluster_tasks SET status='operator', reason=?", (reason,))
        sim.check()
        assert f"cluster left operator: {reason}" in sim.report.problems
        assert "no cluster result accepted" in sim.report.problems
    finally:
        sim.close()


def test_cluster_observations_are_incremental_and_include_get(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sqlite3
    from collections.abc import Iterator
    from contextlib import contextmanager
    from typing import Any

    import httpx

    from desman.registry import store
    from faults import ClusterFaults, ClusterShape, ClusterSimulation, NetSimulation

    sim = ClusterSimulation(tmp_path, 0, ClusterFaults(), ClusterShape(nodes=0, tasks=0))
    queries: list[str] = []
    real = store.connection

    @contextmanager
    def traced(*args: Any, **kwargs: Any) -> Iterator[sqlite3.Connection]:
        with real(*args, **kwargs) as conn:
            conn.set_trace_callback(queries.append)
            yield conn

    monkeypatch.setattr(store, "connection", traced)
    monkeypatch.setattr(NetSimulation, "_handle", lambda self, req: httpx.Response(200))
    try:
        sim._handle(httpx.Request("GET", "https://registry.test/v1/clusters/mine"))
        first = sim.report.actions.copy()
        sim._handle(httpx.Request("POST", "https://registry.test/v1/claim"))
        assert sim.report.actions == first
        for table, marker in (
            ("leases l", "l.rowid >"),
            ("results r", "r.rowid >"),
            ("cluster_records", "seq >"),
        ):
            selects = [q for q in queries if q.startswith("SELECT") and f"FROM {table}" in q]
            assert len(selects) == 2
            assert all(marker in q for q in selects)
    finally:
        sim.close()


@pytest.mark.parametrize("changed", [False, True])
def test_cluster_harness_catches_acceptance_across_versions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: bool
) -> None:
    from types import SimpleNamespace

    from desman.net import protocol, sign
    from desman.registry import clusters, work
    from faults import ClusterFaults, ClusterShape, ClusterSimulation

    sim = ClusterSimulation(tmp_path, 0, ClusterFaults(), ClusterShape(nodes=0, tasks=0))
    plans = [
        sign.sign(
            sim.operator,
            sign.RecordKind.CLUSTER_PLAN,
            protocol.ClusterPlan(
                cluster="a" * 32,
                task="b" * 32,
                version=version,
                nodes=[
                    protocol.PlanNode(
                        description="changed" if changed and version == 2 else "same",
                        acceptance_criteria="done",
                    )
                ],
            ).model_dump(mode="json"),
            issued_at=sim.clock(),
            lifetime=sign.APPOINTMENT_TERM,
        )
        for version in (1, 2)
    ]
    accepted = [
        SimpleNamespace(
            id=p.record.id,
            result=None,
            round=1,
            cluster_job=protocol.ClusterJob(
                cluster="a" * 32,
                task="b" * 32,
                plan=p.record.id,
                version=p.record.body["version"],
                node=0,
                job=protocol.Job(description="same", acceptance_criteria="done"),
            ),
        )
        for p in plans
    ]
    monkeypatch.setattr(work, "jobs", lambda *args, **kwargs: accepted)
    monkeypatch.setattr(clusters, "records", lambda *args, **kwargs: plans)
    try:
        sim.check()
        assert ("plan node accepted twice" in sim.report.problems) == (not changed)
    finally:
        sim.close()


def test_cluster_harness_owns_its_planning_texts(monkeypatch: pytest.MonkeyPatch) -> None:
    import test_node_clusters
    from desman.llm.base import Message, Request
    from desman.llm.fake import FakeLLM
    from faults import CLUSTER_PLAN, CLUSTER_SPLIT, _cluster_planning

    monkeypatch.setattr(test_node_clusters, "PLAN", "unprotected change")
    monkeypatch.setattr(test_node_clusters, "SPLIT", "unprotected change")
    model = _cluster_planning(FakeLLM())
    for prompt, expected in (("Plan task", CLUSTER_PLAN), ("Split subtask", CLUSTER_SPLIT)):
        assert (
            model.complete(Request(messages=[Message(role="user", content=prompt)])).text
            == expected
        )
