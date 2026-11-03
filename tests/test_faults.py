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
    for seed in range(8):
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
