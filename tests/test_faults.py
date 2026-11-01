"""Fault injection (ROADMAP 4.6): random worker and provider failures, repeatable by seed.

See tests/faults.py for the harness. Each seed is one simulated run of several workers on one
database, with crashes, hangs, provider failures and rejections where its random generator puts
them; at the end every invariant must hold. A failing seed is replayed exactly:

    DESMAN_FAULT_SEED=17 pytest tests/test_faults.py

`DESMAN_FAULT_SEEDS=500` runs that many seeds instead of the default few (a longer hunt).
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import pytest

from desman.core.states import TaskStatus
from faults import Faults, Report, Shape, simulate


def _seeds() -> list[int]:
    if os.environ.get("DESMAN_FAULT_SEED"):
        return [int(os.environ["DESMAN_FAULT_SEED"])]
    return list(range(int(os.environ.get("DESMAN_FAULT_SEEDS", "12"))))


def run(
    tmp_path: Path, seed: int, faults: Faults | None = None, shape: Shape | None = None
) -> Report:
    directory = tmp_path / f"seed-{seed}"
    directory.mkdir(parents=True)
    return simulate(directory, seed, faults, shape)


@pytest.mark.parametrize("seed", _seeds())
def test_every_invariant_holds_under_random_faults(tmp_path: Path, seed: int) -> None:
    report = run(tmp_path, seed)
    assert not report.problems, report.explain()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_every_invariant_holds_in_a_storm(tmp_path: Path, seed: int) -> None:
    storm = Faults(crash=0.1, hang=0.05, provider=0.5, retry=0.3)
    report = run(tmp_path, seed, storm, Shape(tasks=4, workers=4))
    assert not report.problems, report.explain()
    assert sum(report.faults.values()) > 20


def test_a_seed_replays_step_for_step(tmp_path: Path) -> None:
    first, again, other = run(tmp_path, 7), run(tmp_path / "again", 7), run(tmp_path, 8)
    assert first.trace == again.trace
    assert first.faults == again.faults and first.faults  # something did go wrong
    assert other.trace != first.trace


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
    found = [run(tmp_path, seed, Faults(crash=0.1), shape) for seed in range(6)]
    problems = [p for report in found for p in report.problems]
    assert any("effect ran twice" in p for p in problems), problems
    caught = next(r for r in found if r.problems)
    assert f"DESMAN_FAULT_SEED={caught.seed}" in caught.explain()
