"""Tests for evaluating a proposal (ROADMAP 5.5): the benchmark, the comparison, the runs in git
worktrees of a checkout, and `desman proposals evaluate`. Nothing is ever applied.

The worktrees are made in a copy of this checkout (a git repository of its own in a temporary
folder), and the benchmark runs in real subprocesses on the scripted fake model."""

from __future__ import annotations

import ast
import dataclasses
import difflib
import json
import os
import shutil
import socket
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner, Result

from desman import bench, evaluation, improve, runs
from desman.cli import cli
from desman.config import Config, ConfigManager, ModelChoices, Tuning, default_config_path
from desman.core import db, proposals
from desman.core.models import SubtaskSpec, TaskSpec
from desman.core.proposals import Proposal, Status
from desman.core.sandbox import Outcome, Sandbox, SandboxUnavailable
from desman.core.store import Store
from desman.improve import BY_ID, Change

ROOT = Path(__file__).parent.parent
UNCONFINED = evaluation.isolation()
if UNCONFINED is not None and os.environ.get("DESMAN_REQUIRE_SANDBOX"):  # set in CI
    raise RuntimeError(f"DESMAN_REQUIRE_SANDBOX is set, but runs cannot be confined: {UNCONFINED}")
confined = pytest.mark.skipif(UNCONFINED is not None, reason=f"not confined here: {UNCONFINED}")
"""Evaluating runs the benchmark under bubblewrap only (S.3): elsewhere nothing is evaluated."""
GIT_ENV = {
    "GIT_AUTHOR_NAME": "test",
    "GIT_AUTHOR_EMAIL": "test@example.invalid",
    "GIT_COMMITTER_NAME": "test",
    "GIT_COMMITTER_EMAIL": "test@example.invalid",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
}


def desman(*args: str | Path) -> Result:
    return CliRunner().invoke(cli, [str(a) for a in args])


def git(where: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(where), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
        env={**os.environ, **GIT_ENV},
    ).stdout


@pytest.fixture(scope="module")
def origin(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A git repository holding this checkout's package and test harness, as they are now."""
    repo = tmp_path_factory.mktemp("origin")
    shutil.copytree(
        ROOT / "src", repo / "src", ignore=shutil.ignore_patterns("__pycache__", "*.egg-info")
    )
    shutil.copy(ROOT / "pyproject.toml", repo)
    shutil.copy(ROOT / ".gitattributes", repo)
    (repo / "tests").mkdir()
    for name in ("conftest.py", "faults.py", "helpers.py"):  # protected: checked in a checkout
        shutil.copy(ROOT / "tests" / name, repo / "tests")
    git(repo, "init", "-q", "-b", "main")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "release")
    return repo


@pytest.fixture
def checkout(origin: Path, tmp_path: Path) -> Path:
    """A clone of `origin` that a test may change."""
    repo = tmp_path / "checkout"
    git(origin, "clone", "-q", str(origin), str(repo))
    return repo


def package(repo: Path) -> Path:
    return repo / "src" / "desman"


def store() -> Store:
    return Store(db.open_db())


def proposal(s: Store, diff: str, details: dict[str, Any]) -> Proposal:
    """A proposal as an analysis would have settled it."""
    spec = TaskSpec(
        goal=improve.GOAL, subtasks=[SubtaskSpec(description="analyse", acceptance_criteria="ok")]
    )
    started = proposals.start(s, spec, None)
    assert proposals.settle(
        s.conn, started.id, Status.PROPOSED, summary="A change.", details=details, diff=diff
    )
    found = proposals.get(s.conn, started.id)
    assert found is not None
    return found


def rules_change(repo: Path, extra: str = " Prefer one subtask.") -> Change:
    old = improve.current(BY_ID["planner.rules"], package(repo))
    return Change(target="planner.rules", value=old + extra)


def a_proposal(s: Store, repo: Path, changes: list[Change] | None = None) -> Proposal:
    diff, details = improve.render(
        changes or [rules_change(repo)], package(repo), shown=improve.settings_shown(Config())
    )
    return proposal(s, diff, details)


def evaluate(s: Store, p: Proposal, repo: Path, **kwargs: Any) -> proposals.Evaluation:
    return evaluation.evaluate(s.conn, p, mode="fake", tuning=Tuning(), repo=repo, **kwargs)


def worktrees(repo: Path) -> list[str]:
    return [
        line
        for line in git(repo, "worktree", "list", "--porcelain").splitlines()
        if line.startswith("worktree ")
    ]


# ── The benchmark ────────────────────────────────────────────────────────────


def test_the_benchmark_runs_every_case_on_the_fake() -> None:
    s = store()
    result = bench.run(s, "fake", Tuning())
    assert [c.id for c in result.cases] == [c.id for c in bench.CASES]
    assert all(c.status == "done" and c.plan == "llm" and c.subtasks == 2 for c in result.cases)
    numbers = bench.metrics(result)
    assert numbers["tasks_done"] == len(bench.CASES)
    assert numbers["plan_fallbacks"] == numbers["rejected"] == numbers["escalated"] == 0
    assert numbers["first_try"] == numbers["subtasks"] == 2 * len(bench.CASES)
    # Every call is counted, with its tokens: a plan, then a result and a review per subtask.
    assert numbers["calls"] == len(bench.CASES) * 5
    assert numbers["tokens"] > 0
    # Nothing in the benchmark may use a tool.
    tools = {r[0] for r in s.conn.execute("SELECT tools FROM tasks")}
    assert tools == {"none"}
    # Each case's subtasks are of its kind, so the executor's model for that kind is asked.
    kinds = dict(
        s.conn.execute(
            """SELECT json_extract(a.details_json, '$.benchmark'), group_concat(DISTINCT s.kind)
               FROM audit_log a JOIN subtasks s ON s.task_id = a.task_id
               WHERE a.action = 'task_created' GROUP BY a.task_id"""
        ).fetchall()
    )
    assert kinds == {c.id: c.kind.value for c in bench.CASES}


def test_the_fake_does_not_plan_without_the_format() -> None:
    llm = bench.fake()
    from desman.llm.base import Message, Request

    request = Request(
        messages=[
            Message(role="system", content="Split the goal."),
            Message(role="user", content="Plan task:\nA goal"),
        ]
    )
    with pytest.raises(json.JSONDecodeError):
        json.loads(llm.complete(request).text)


def test_the_benchmark_follows_the_tuning() -> None:
    result = bench.run(store(), "fake", Tuning(max_subtasks=1))
    assert result.limits is not None and result.limits["max_subtasks"] == 1
    assert "models" not in result.limits
    legacy = result.model_dump(exclude={"limits"})
    assert bench.Result.model_validate(legacy).limits is None
    # The fake plans within the limit its prompt states, as a model asked for it would.
    assert all(c.plan == "llm" and c.subtasks == 1 for c in result.cases)
    assert bench.metrics(result)["plan_fallbacks"] == 0


def test_real_mode_asks_the_configured_models(monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[dict[str, Any]] = []

    def models(config: Config, conn: Any, **kwargs: Any) -> runs.Models:
        asked.append({"tuning": config.tuning} | kwargs)
        return runs.Models(bench.Recorded(bench.fake(), conn), None)

    monkeypatch.setattr(runs, "models", models)
    # The tuning evaluated, the model each role asks first included, is what picks the models.
    tuning = Tuning(models=ModelChoices(reviewer="groq/other-model"), max_subtasks=3)
    result = bench.run(store(), "real", tuning, config=Config(), provider="groq", model="m")
    assert result.mode == "real" and bench.metrics(result)["tasks_done"] == len(bench.CASES)
    # Its own database is new: nothing cached is reused.
    assert asked == [
        {
            "tuning": tuning,
            "provider": "groq",
            "model": "m",
            "cache": False,
            "master_candidate": None,
        }
    ]


def test_real_mode_without_a_provider_reports_it(capsys: pytest.CaptureFixture[str]) -> None:
    assert bench.main(["--mode", "real"]) == 2
    assert capsys.readouterr().out == ""


def test_a_bad_tuning_is_refused(capsys: pytest.CaptureFixture[str]) -> None:
    assert bench.main(["--mode", "fake", "--tuning", '{"max_subtasks": 99}']) == 2
    assert "bad --tuning" in capsys.readouterr().err


def result_with(**numbers: int) -> bench.Result:
    """A result whose metrics are `numbers` (the rest zero, four cases done)."""
    done = numbers.pop("tasks_done", 4)
    cases = [
        bench.CaseResult(id=str(i), status="done" if i < done else "failed", plan="llm", subtasks=1)
        for i in range(4)
    ]
    stats = {
        "subtasks": {k: numbers.get(k, 0) for k in ("total", "first_try", "escalated")},
        "reviews": {"retry": numbers.get("rejected", 0)},
        "strategies": [{"started": numbers.get("attempts", 0)}],
        "roles": [{"calls": 1, "failed": 0, "prompt_tokens": numbers.get("tokens", 0)}],
    }
    return bench.Result(version=bench.VERSION, mode="fake", source="x", cases=cases, stats=stats)


@pytest.mark.parametrize(
    ("base", "candidate", "verdict"),
    [
        ({}, {}, "same"),
        ({"tasks_done": 3}, {"tasks_done": 4}, "better"),
        # More done decides before anything else, even many more attempts.
        ({"tasks_done": 3, "attempts": 4}, {"tasks_done": 4, "attempts": 20}, "better"),
        ({"escalated": 0}, {"escalated": 1}, "worse"),
        ({"rejected": 3}, {"rejected": 1}, "better"),
        ({"attempts": 8}, {"attempts": 9}, "worse"),
        # Tokens decide only a tie, and only beyond the margin.
        ({"tokens": 1000}, {"tokens": 1030}, "same"),
        ({"tokens": 1000}, {"tokens": 900}, "better"),
        ({"tokens": 1000}, {"tokens": 1100}, "worse"),
    ],
)
def test_compare(base: dict[str, int], candidate: dict[str, int], verdict: str) -> None:
    assert bench.compare(result_with(**base), result_with(**candidate)) == verdict


# ── In worktrees ─────────────────────────────────────────────────────────────


@confined
def test_a_proposal_is_evaluated_in_worktrees_and_recorded(checkout: Path) -> None:
    s = store()
    p = a_proposal(s, checkout)
    sides: list[str] = []
    found = evaluate(s, p, checkout, on_run=sides.append)
    assert sides == ["base", "candidate"]
    assert found.mode == "fake" and found.commit == git(checkout, "rev-parse", "HEAD").strip()
    assert found.reason is None and found.base is not None and found.candidate is not None
    # Each version ran from a worktree of its own, never from this checkout.
    base_source, candidate_source = found.base["source"], found.candidate["source"]
    assert base_source != candidate_source
    assert str(ROOT) not in base_source and str(checkout) not in base_source
    old, new = evaluation.metrics_of(found)
    assert old is not None and new is not None
    assert old["tasks_done"] == new["tasks_done"] == len(bench.CASES)
    # A longer planner prompt costs more tokens: a little more, within the margin.
    assert new["tokens"] > old["tokens"]
    assert found.verdict == "same"
    # Recorded, and the worktrees are gone; the checkout itself is untouched.
    assert proposals.evaluations(s.conn, p.id) == [found]
    assert len(worktrees(checkout)) == 1
    assert git(checkout, "status", "--porcelain") == ""


def broken_rules(monkeypatch: pytest.MonkeyPatch, repo: Path) -> Change:
    """A planner prompt without the reply's format (which the analyst may never propose)."""
    target = dataclasses.replace(BY_ID["planner.rules"], keep=())
    monkeypatch.setitem(improve.BY_ID, "planner.rules", target)
    return Change(target="planner.rules", value="Split the goal into steps.")


@confined
def test_a_proposal_that_breaks_planning_does_worse(
    checkout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    s = store()
    p = a_proposal(s, checkout, [broken_rules(monkeypatch, checkout)])
    found = evaluate(s, p, checkout)
    old, new = evaluation.metrics_of(found)
    assert old is not None and new is not None
    assert old["plan_fallbacks"] == 0 and new["plan_fallbacks"] == len(bench.CASES)
    assert found.verdict == "worse"


@confined
def test_the_proposed_settings_are_what_the_proposed_version_runs_with(checkout: Path) -> None:
    s = store()
    p = a_proposal(s, checkout, [Change(target="tuning.max_subtasks", value=1)])
    assert p.diff == ""  # settings only
    found = evaluate(s, p, checkout)
    old, new = evaluation.metrics_of(found)
    assert old is not None and new is not None
    assert old["subtasks"] == 2 * len(bench.CASES) and new["subtasks"] == len(bench.CASES)
    assert new["plan_fallbacks"] == 0
    # The same goals done in fewer attempts.
    assert found.verdict == "better"


@confined
def test_the_current_version_runs_with_the_persons_tuning(checkout: Path) -> None:
    s = store()
    p = a_proposal(s, checkout)
    found = evaluation.evaluate(
        s.conn, p, mode="fake", tuning=Tuning(max_subtasks=1), repo=checkout
    )
    old, new = evaluation.metrics_of(found)
    assert old is not None and new is not None
    assert old["subtasks"] == new["subtasks"] == len(bench.CASES)


def test_a_proposal_made_from_other_files_is_not_evaluated(checkout: Path) -> None:
    s = store()
    p = a_proposal(s, checkout)
    planner = package(checkout) / "planner.py"
    planner.write_text(
        planner.read_text(encoding="utf-8") + "\n# changed since\n", encoding="utf-8"
    )
    git(checkout, "commit", "-q", "-am", "later")
    with pytest.raises(evaluation.EvaluationError, match="other versions of src/desman/planner"):
        evaluate(s, p, checkout, box=Box())
    assert proposals.evaluations(s.conn, p.id) == []
    assert len(worktrees(checkout)) == 1


def edited_diff(repo: Path, edit: Any) -> tuple[str, dict[str, Any]]:
    """The diff of a real proposal, with `edit(text)` also applied to planner.py."""
    diff, details = improve.render([rules_change(repo)], package(repo))
    name = improve.SOURCE_PREFIX + "planner.py"
    before = (repo / name).read_text(encoding="utf-8")
    after = edit(improve.replace(before, BY_ID["planner.rules"], details["changes"][0]["new"]))
    lines = difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True), f"a/{name}", f"b/{name}"
    )
    return f"diff --git a/{name} b/{name}\n" + "".join(lines), details


def test_a_diff_changing_more_than_the_targets_is_refused(checkout: Path) -> None:
    s = store()
    diff, details = edited_diff(
        checkout, lambda text: text.replace("MAX_SUBTASKS = 8", "MAX_SUBTASKS = 9")
    )
    p = proposal(s, diff, details)
    with pytest.raises(evaluation.EvaluationError, match="not the one the proposal's texts make"):
        evaluate(s, p, checkout, box=Box())
    assert proposals.evaluations(s.conn, p.id) == []


def test_a_diff_giving_other_texts_than_proposed_is_refused(checkout: Path) -> None:
    s = store()
    diff, details = improve.render([rules_change(checkout)], package(checkout))
    details["changes"][0]["new"] += " Something else."
    p = proposal(s, diff, details)
    with pytest.raises(evaluation.EvaluationError, match="not the one the proposal's texts make"):
        evaluate(s, p, checkout, box=Box())


def test_a_diff_touching_another_file_is_refused(checkout: Path) -> None:
    s = store()
    name = "src/desman/agents.py"
    before = (checkout / name).read_text(encoding="utf-8")
    after = before.replace("MAX_TOOL_ROUNDS = 8", "MAX_TOOL_ROUNDS = 80")
    lines = difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True), f"a/{name}", f"b/{name}"
    )
    p = proposal(s, f"diff --git a/{name} b/{name}\n" + "".join(lines), {"changes": []})
    with pytest.raises(evaluation.EvaluationError, match="not the one the proposal's texts make"):
        evaluate(s, p, checkout, box=Box())


def test_a_patch_is_never_applied(checkout: Path) -> None:
    """A diff that git would apply to more than it shows (here: a new file beside the change)
    is not what code builds from the proposal's texts, so nothing of it runs."""
    s = store()
    diff, details = improve.render([rules_change(checkout)], package(checkout))
    extra = (
        "diff --git a/src/desman/sitecustomize.py b/src/desman/sitecustomize.py\n"
        "new file mode 100644\n--- /dev/null\n+++ b/src/desman/sitecustomize.py\n"
        "@@ -0,0 +1 @@\n+raise SystemExit('ran')\n"
    )
    p = proposal(s, diff + extra, details)
    with pytest.raises(evaluation.EvaluationError, match="not the one the proposal's texts make"):
        evaluate(s, p, checkout, box=Box())


def test_a_target_text_that_may_not_be_one_is_refused(checkout: Path) -> None:
    s = store()
    p = a_proposal(s, checkout)
    details = p.details | {"changes": [p.details["changes"][0] | {"new": "no format"}]}
    q = proposal(s, p.diff or "", details)
    with pytest.raises(evaluation.EvaluationError, match="must still contain"):
        evaluate(s, q, checkout, box=Box())


def test_a_run_that_fails_is_recorded_as_failed(checkout: Path) -> None:
    s = store()
    p = a_proposal(s, checkout)
    crashed = Outcome(exit_code=1, stdout="", stderr="Traceback\nValueError: boom")
    found = evaluate(s, p, checkout, box=Box(crashed))
    assert found.verdict == "failed" and found.base is None and found.candidate is None
    assert found.reason is not None and "current version's run failed" in found.reason
    assert "boom" in found.reason
    assert len(worktrees(checkout)) == 1


@confined
def test_a_python_that_does_not_start_confined_records_nothing(
    checkout: Path, tmp_path: Path
) -> None:
    """The machine's problem (a Python missing its libraries there), never the proposal's."""
    s = store()
    p = a_proposal(s, checkout)
    with pytest.raises(evaluation.Unconfined, match="does not start there"):
        evaluate(s, p, checkout, python=str(tmp_path / "no-python"), box=Sandbox())
    assert proposals.evaluations(s.conn, p.id) == []
    assert len(worktrees(checkout)) == 1


@confined
def test_the_runtime_probe_passes_here() -> None:
    evaluation._probe_runtime(Sandbox(), sys.executable, Path("/tmp/desman-probe-test"))


def test_only_a_proposed_change_is_evaluated(checkout: Path) -> None:
    s = store()
    spec = TaskSpec(goal="g", subtasks=[SubtaskSpec(description="d", acceptance_criteria="c")])
    started = proposals.start(s, spec, None)
    with pytest.raises(evaluation.EvaluationError, match="proposes nothing"):
        evaluate(s, started, checkout)


def test_outside_a_checkout_nothing_is_evaluated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(improve, "PACKAGE", tmp_path)
    with pytest.raises(evaluation.EvaluationError, match="git checkout"):
        evaluation.repository()


def test_this_checkout_is_the_repository() -> None:
    assert evaluation.repository() == ROOT.resolve()


def test_evaluations_are_append_only(checkout: Path) -> None:
    s = store()
    p = a_proposal(s, checkout)
    proposals.record_evaluation(s.conn, p.id, mode="fake", commit="c", verdict="failed", reason="r")
    import sqlite3

    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        s.conn.execute("UPDATE evaluations SET verdict = 'better'")
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        s.conn.execute("DELETE FROM evaluations")


def test_a_v22_database_upgrades_to_evaluations(tmp_path: Path) -> None:
    path = tmp_path / "v22.db"
    old = db.connect(path)
    for v, migration in enumerate(db.MIGRATIONS[:22], start=1):
        with db.transaction(old):
            for statement in db._split_sql(migration):
                old.execute(statement)
            old.execute(f"PRAGMA user_version = {v}")
    old.execute("INSERT INTO tasks (id, goal) VALUES ('t1', 'goal')")
    old.execute("INSERT INTO proposals (task_id) VALUES ('t1')")
    old.close()
    conn = db.open_db(path)
    assert db.schema_version(conn) == db.SCHEMA_VERSION
    assert proposals.evaluations(conn, 1) == []


# ── The command ──────────────────────────────────────────────────────────────


@confined
def test_evaluate_command(checkout: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evaluation, "repository", lambda: checkout)
    s = store()
    p = a_proposal(s, checkout)
    result = desman("proposals", "evaluate", str(p.id))
    assert result.exit_code == 0, result.output
    assert "current version" in result.output and "Evaluation 1" in result.output
    assert "no clear difference" in result.output
    shown = desman("proposals", "show", str(p.id))
    assert "Evaluation 1" in shown.output and "tasks done" in shown.output
    data = json.loads(desman("proposals", "show", str(p.id), "--json").output)
    [e] = data["evaluations"]
    assert e["verdict"] == "same" and e["metrics"]["base"]["tasks_done"] == len(bench.CASES)
    again = json.loads(desman("proposals", "evaluate", str(p.id), "--json").output)
    assert again["id"] == 2
    assert "1 earlier evaluation" in desman("proposals", "show", str(p.id)).output


def test_evaluate_command_refusals(checkout: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evaluation, "repository", lambda: checkout)
    s = store()
    p = a_proposal(s, checkout)
    usage = desman("proposals", "evaluate", str(p.id), "--provider", "groq")
    assert usage.exit_code == 2 and "--real" in usage.output
    no_provider = desman("proposals", "evaluate", str(p.id), "--real")
    assert no_provider.exit_code == 1 and "current version" not in no_provider.output
    assert proposals.evaluations(s.conn, p.id) == []
    missing = desman("proposals", "evaluate", "99")
    default_config_path().write_text("{not json", encoding="utf-8")
    broken = desman("proposals", "evaluate", str(p.id))
    assert broken.exit_code == 1 and "not valid JSON" in broken.output
    default_config_path().unlink()
    assert missing.exit_code == 1 and "No proposal 99" in missing.output
    spec = TaskSpec(goal="g", subtasks=[SubtaskSpec(description="d", acceptance_criteria="c")])
    started = proposals.start(s, spec, None)
    refused = desman("proposals", "evaluate", str(started.id))
    assert refused.exit_code == 1 and "Cannot evaluate proposal" in refused.output


# ── What the analyst sees of it ──────────────────────────────────────────────


def test_the_analyst_sees_how_earlier_proposals_did(checkout: Path, tmp_path: Path) -> None:
    from test_improve import a_failure

    a_failure(tmp_path)
    s = store()
    p = a_proposal(s, checkout)
    assert improve.evaluated(s.conn, p) == {}
    result = bench.run(Store(db.open_db(tmp_path / "bench.db")), "fake", Tuning())
    data = result.model_dump(mode="json")
    proposals.record_evaluation(s.conn, p.id, mode="fake", commit="c1", verdict="same", base=data)
    proposals.record_evaluation(
        s.conn, p.id, mode="fake", commit="c2", verdict="worse", base=data, candidate=data
    )
    proposals.record_evaluation(
        s.conn, p.id, mode="real", commit="c2", verdict="failed", reason="it\x1btook too long"
    )
    spec = improve.spec(s, None)
    assert spec is not None
    [earlier] = spec.subtasks[0].inputs["earlier_proposals"]
    real, fake = earlier["evaluation"]
    # The newest of each mode, real models first; a reason without control characters.
    assert real == {"models": "real", "verdict": "failed", "reason": "ittook too long"}
    assert fake["models"] == "fake" and fake["verdict"] == "worse"
    assert fake["current"] == fake["proposed"] == bench.metrics(result)
    assert "settings" not in earlier  # it changed a prompt only


def test_the_analyst_sees_the_values_an_earlier_proposal_gave(
    checkout: Path, tmp_path: Path
) -> None:
    from test_improve import a_failure

    a_failure(tmp_path)
    s = store()
    a_proposal(s, checkout, [Change(target="tuning.max_subtasks", value=3)])
    spec = improve.spec(s, None)
    assert spec is not None
    [earlier] = spec.subtasks[0].inputs["earlier_proposals"]
    assert earlier["settings"] == [{"setting": "tuning.max_subtasks", "old": None, "new": 3}]


class Box(Sandbox):
    """Stands in for the sandbox: keeps what a run would be given, then fails it."""

    def __init__(self, outcome: Outcome | None = None) -> None:
        super().__init__(platform="linux")
        self.calls: list[dict[str, Any]] = []
        self.outcome = outcome

    def unavailable(self) -> str | None:
        return None

    def program(self, command: Sequence[str], **kwargs: Any) -> Any:
        if "-c" in command:  # the runtime's probe passes
            return Outcome(exit_code=0, stdout="", stderr="")
        config = Path(kwargs["env"]["DESMAN_CONFIG_DIR"]) / "config.json"
        kwargs["config"] = json.loads(config.read_text()) if config.exists() else None
        self.calls.append({"command": command, **kwargs})
        if self.outcome is not None:
            return self.outcome
        raise SandboxUnavailable("checked")


def test_a_run_is_confined_and_given_no_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "parent-secret")
    monkeypatch.setenv("HF_TOKEN", "parent-secret")
    tree = tmp_path / "tree"
    tree.mkdir()
    box = Box()
    with pytest.raises(evaluation.Unconfined, match="checked"):
        evaluation._run(
            tree, tmp_path, "fake", Tuning(), provider=None, model=None, python="python", box=box
        )
    [call] = box.calls
    env = call["env"]
    assert "parent-secret" not in json.dumps(env)
    assert env["PYTHONPATH"] == str(tree / "src") and env["PYTHONDONTWRITEBYTECODE"] == "1"
    data = tmp_path / "tree-data"
    for name in ("HOME", "DESMAN_DATA_DIR", "TMPDIR"):
        assert Path(env[name]) == data
    assert Path(env["DESMAN_CONFIG_DIR"]).is_dir()
    assert call["network"] is False and call["config"] is None  # the fake needs no config
    # Read-only: the worktree, the config and the interpreter; written: a folder in memory only.
    assert tree in call["read_only"] and call["cwd"] == tree
    assert call["writable"] == [] and call["scratch"] == [data] and not data.exists()
    assert not any(ROOT.resolve().is_relative_to(p) for p in call["read_only"])
    assert not any(Path.home().resolve().is_relative_to(p) for p in call["read_only"])


def test_a_real_run_gets_the_network_and_only_the_keys_it_may_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = Config(configured=True, language="ru", personalize=True)
    config.add_key("GROQ_API_KEY", "gsk_free-key-000000000000")
    config.add_key("OPENAI_API_KEY", "sk-paid-key-0000000000000")
    config.add_key("HF_TOKEN", "hf_token-00000000000000000")
    config.add_key("TAVILY_API_KEY", "tvly-search-00000000000000")
    config.settings["OLLAMA_HOST"] = "http://127.0.0.1:11434"
    config.permissions.network_access = True
    config.network.enabled = True
    ConfigManager(default_config_path()).save(config)
    tree = tmp_path / "tree"
    tree.mkdir()
    box = Box()
    for provider, folder in ((None, "a"), ("openai", "b")):
        (tmp_path / folder).mkdir()
        with pytest.raises(evaluation.Unconfined):
            evaluation._run(
                tree,
                tmp_path / folder,
                "real",
                Tuning(),
                provider=provider,
                model=None,
                python="python",
                box=box,
            )
    free, named = (Config.model_validate(c["config"]) for c in box.calls)
    assert all(c["network"] for c in box.calls)
    assert set(free.api_keys) == {"GROQ_API_KEY"}
    assert set(named.api_keys) == {"GROQ_API_KEY", "OPENAI_API_KEY"}
    for given in (free, named):
        assert given.language == "ru" and given.settings == config.settings
        assert given.permissions == Config().permissions and given.network == Config().network
        assert not given.personalize
    # The config is given read-only.
    config_dir = Path(box.calls[0]["env"]["DESMAN_CONFIG_DIR"])
    assert config_dir in box.calls[0]["read_only"]
    assert config_dir not in box.calls[0]["writable"]


def test_only_a_provider_named_brings_a_paid_key() -> None:
    """As the router reads them: `--model` alone names a model of a free provider (an id with a
    slash, OpenRouter's way, is no provider); `--provider` names the provider."""
    config = Config()
    config.add_key("OPENAI_API_KEY", "sk-paid-key-0000000000000")
    assert set(evaluation.bench_config(config, "openai").api_keys) == {"OPENAI_API_KEY"}
    assert evaluation.bench_config(config, None, "openai/gpt-x").api_keys == {}
    assert evaluation.bench_config(config, "openrouter", "openai/gpt-x").api_keys == {}


def test_nothing_is_evaluated_where_runs_cannot_be_confined(checkout: Path) -> None:
    s = store()
    p = a_proposal(s, checkout)
    box = Sandbox(platform="darwin", docker="/nonexistent/docker")
    assert evaluation.isolation(box) is not None
    with pytest.raises(evaluation.Unconfined, match="Linux"):
        evaluate(s, p, checkout, box=box)
    assert len(worktrees(checkout)) == 1
    assert proposals.evaluations(s.conn, p.id) == []


HOSTILE = """\
import json, os, socket, sys
found = {}
def attempt(name, action):
    try:
        action()
        found[name] = "done"
    except Exception as e:
        found[name] = type(e).__name__
attempt("write_tree", lambda: open(os.path.join(os.path.dirname(__file__), "x"), "w"))
attempt("read_secret", lambda: open(SECRET).read())
attempt("read_checkout", lambda: open(CHECKOUT).read())
attempt("connect", lambda: socket.create_connection(("127.0.0.1", PORT), 2))
attempt("write_root", lambda: open("/rootfile", "w"))
attempt("write_data", lambda: open(os.path.join(os.environ["DESMAN_DATA_DIR"], "x"), "w"))
found["env_secret"] = "parent-secret" in json.dumps(dict(os.environ))
found["home"] = os.path.expanduser("~")
print(json.dumps(found), file=sys.stderr)
sys.exit(3)
"""


@confined
def test_a_hostile_version_reaches_nothing_of_the_persons(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The benchmark of a version is code that may not be trusted: under bubblewrap it cannot
    write its worktree, read a file outside the folders it was given (the person's, the
    checkout's), reach a server on this machine, or see the caller's environment."""
    monkeypatch.setenv("OPENAI_API_KEY", "parent-secret")
    tree = tmp_path / "tree"
    (tree / "src" / "desman").mkdir(parents=True)
    (tree / "src" / "desman" / "__init__.py").write_text("")
    secret = tmp_path / "secret.txt"
    secret.write_text("secret")
    server = socket.create_server(("127.0.0.1", 0))
    try:
        port = server.getsockname()[1]
        given = f"SECRET = {str(secret)!r}\nCHECKOUT = {str(ROOT / 'pyproject.toml')!r}\n"
        (tree / "src" / "desman" / "bench.py").write_text(given + f"PORT = {port}\n" + HOSTILE)
        with pytest.raises(evaluation.RunFailed) as failed:
            evaluation._run(
                tree,
                tmp_path,
                "fake",
                Tuning(),
                provider=None,
                model=None,
                python=sys.executable,
                timeout=60,
            )
    finally:
        server.close()
    found = json.loads(str(failed.value))
    assert found["write_tree"] in ("OSError", "PermissionError")
    assert found["read_secret"] == "FileNotFoundError"
    assert found["read_checkout"] == "FileNotFoundError"
    assert found["connect"] != "done"
    assert found["env_secret"] is False
    assert found["write_root"] in ("OSError", "PermissionError")
    assert found["write_data"] == "done"  # its own folder, in memory: nothing on the disk
    assert not (tmp_path / "tree-data").exists()
    assert Path(found["home"]) == tmp_path / "tree-data"


# ── A proposal trying to change more than texts (S.3) ────────────────────────


def proposed(checkout: Path, tmp_path: Path, value: str) -> tuple[Path, list[dict[str, Any]]]:
    """The proposed version of `checkout` giving the planner's rules `value`, built as an
    evaluation builds it (from the proposal's texts), and checked against the current one."""
    diff, details = improve.render(
        [Change(target="planner.rules", value=value)],
        package(checkout),
        shown=improve.settings_shown(Config()),
    )
    candidate = tmp_path / "candidate"
    git(checkout, "worktree", "add", "-q", "--detach", str(candidate), "HEAD")
    evaluation._build(candidate, details["changes"], diff)
    evaluation._check_changes(checkout, candidate, details["changes"])
    return candidate, details["changes"]


@pytest.mark.parametrize(
    "tail",
    [
        ' """) ; import os ; ("""',
        " ''' + __import__('os').getcwd() + '''",
        ' " + __import__("os").getcwd() + "',
        " {__import__('os').getcwd()} {0} {{}}",
        " \\N{BULLET} \\x00 \\u2028 \\ end",
        " trailing backslash \\",
        " \U0001f600 é ı",
    ],
)
def test_any_text_stays_a_text(checkout: Path, tmp_path: Path, tail: str) -> None:
    """Quotes, braces, escapes and backslashes in a proposed text never become code: the
    proposed file holds exactly that text, and nothing else of it changed."""
    old = improve.current(BY_ID["planner.rules"], package(checkout))
    candidate, _ = proposed(checkout, tmp_path, old + tail)
    assert improve.current(BY_ID["planner.rules"], package(candidate)) == old + tail
    assert git(candidate, "status", "--porcelain") == " M src/desman/planner.py\n"


@pytest.mark.parametrize("tail", ["\nimport os", " x", "\x00", "\r", "\x1b[2J", "\x85"])
def test_a_text_with_breaks_or_control_characters_is_refused(checkout: Path, tail: str) -> None:
    old = improve.current(BY_ID["planner.rules"], package(checkout))
    changes = [{"target": "planner.rules", "new": old + tail + " more"}]
    with pytest.raises(evaluation.EvaluationError, match="control characters"):
        evaluation._build(checkout, changes, "")


def swapped(text: str, name: str, source: str) -> str:
    """`text` with the statement assigning `name` replaced by `source`."""
    [node] = [n for n in ast.parse(text).body if evaluation._assigned(n) == name]
    assert node.end_lineno is not None
    lines = text.splitlines(keepends=True)
    return "".join([*lines[: node.lineno - 1], source, *lines[node.end_lineno :]])


@pytest.mark.parametrize(
    ("source", "reason"),
    [
        ("PLANNER_RULES = f\"{__import__('os').getcwd()}\"\n", "not a plain value"),
        ("PLANNER_RULES = str(__import__('os').getcwd())\n", "not a plain value"),
        ("PLANNER_RULES = KIND_RULES\n", "not a plain value"),
        ("PLANNER_RULES: str\n", "more than the targets"),
        ("RULES = 'new'\nPLANNER_RULES = 'new'\n", "more than the targets"),
        ("PLANNER_RULE = 'new'\n", "more than the targets"),
        ("PLANNER_RULES = 'new'\nimport os\n", "more than the targets"),
        ("PLANNER_RULES = 'new'\nPLANNER_RULES = PLANNER_RULES.upper()\n", "more than the targets"),
        ("PLANNER_RULES = 'evil'\nPLANNER_RULES = 'new'\n", "more than the targets"),
        ("PLANNER_RULES = KIND_RULES = 'new'\n", "more than the targets"),
        ("PLANNER_RULES = 'new'\nPLANNER_RULES += '!'\n", "more than the targets"),
        ("PLANNER_RULES = ('new'\n", "not valid Python"),
        ("PLANNER_RULES = b'new'\n", "not a text"),
        ("PLANNER_RULES: __import__('os').getcwd() = 'new'\n", "more than the targets"),
    ],
)
def test_a_version_changing_more_than_the_text_is_refused(
    checkout: Path, tmp_path: Path, source: str, reason: str
) -> None:
    """What the proposed version must not be, whoever made its files: code where a text was, an
    assignment renamed or doubled, another statement. Refused with a reason, never a crash."""
    candidate = tmp_path / "candidate"
    shutil.copytree(checkout / "src", candidate / "src")
    path = candidate / "src" / "desman" / "planner.py"
    path.write_text(swapped(path.read_text("utf-8"), "PLANNER_RULES", source), "utf-8")
    changes = [{"target": "planner.rules", "new": "new"}]
    with pytest.raises(evaluation.EvaluationError, match=reason):
        evaluation._check_changes(checkout, candidate, changes)


def test_a_neighbouring_text_changed_is_refused(checkout: Path, tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    shutil.copytree(checkout / "src", candidate / "src")
    path = candidate / "src" / "desman" / "planner.py"
    text = swapped(path.read_text("utf-8"), "PLANNER_RULES", "PLANNER_RULES = 'new'\n")
    path.write_text(swapped(text, "KIND_RULES", "KIND_RULES = 'other'\n"), "utf-8")
    with pytest.raises(evaluation.EvaluationError, match="more than the targets"):
        evaluation._check_changes(checkout, candidate, [{"target": "planner.rules", "new": "new"}])


def test_the_same_text_written_otherwise_is_still_that_text(checkout: Path, tmp_path: Path) -> None:
    """Concatenated or triple-quoted, a text is a text: the check reads values, not spelling."""
    candidate = tmp_path / "candidate"
    shutil.copytree(checkout / "src", candidate / "src")
    path = candidate / "src" / "desman" / "planner.py"
    source = 'PLANNER_RULES = """ne""" "w"\n'
    path.write_text(swapped(path.read_text("utf-8"), "PLANNER_RULES", source), "utf-8")
    evaluation._check_changes(checkout, candidate, [{"target": "planner.rules", "new": "new"}])


def test_a_target_moved_is_refused(checkout: Path, tmp_path: Path) -> None:
    """Its value may change, not where it stands: moved after another statement, an earlier
    reader of the name would see something else."""
    candidate = tmp_path / "candidate"
    shutil.copytree(checkout / "src", candidate / "src")
    path = candidate / "src" / "desman" / "planner.py"
    text = swapped(path.read_text("utf-8"), "PLANNER_RULES", "")
    path.write_text(text + "PLANNER_RULES = 'new'\n", "utf-8")
    with pytest.raises(evaluation.EvaluationError, match="more than the targets"):
        evaluation._check_changes(checkout, candidate, [{"target": "planner.rules", "new": "new"}])


@pytest.mark.skipif(sys.platform == "win32", reason="links need Developer Mode on Windows")
def test_the_runtime_is_shown_also_through_a_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = tmp_path / "real-python"
    real.write_text("")
    link = tmp_path / "python"
    link.symlink_to(real)
    monkeypatch.setattr(sys, "executable", str(link))
    found = evaluation._runtime()
    assert link in found and real.resolve() in found


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="evaluated under bwrap only")
def test_the_runtime_is_only_python(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A Python installed with `--prefix=~/.local` must not show the rest of that prefix
    (DESMAN's own data lives in ~/.local/share): only the interpreter, its library folders and
    its shared library."""
    for path in evaluation._runtime():
        assert not default_config_path().is_relative_to(path)
        assert not db.default_db_path().is_relative_to(path)
        assert not Path.home().is_relative_to(path)
        assert path.is_file() or "python" in str(path).lower() or "site-packages" in str(path)


def test_a_real_run_keeps_how_to_reach_the_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ConfigManager(default_config_path()).save(Config())
    bundle = tmp_path / "ca.pem"
    bundle.write_text("certificates")
    if sys.platform != "win32":
        (tmp_path / "linked.pem").symlink_to(bundle)
        bundle = tmp_path / "linked.pem"
    monkeypatch.setenv("https_proxy", "http://proxy.invalid:3128")
    monkeypatch.setenv("NO_PROXY", "localhost")
    monkeypatch.setenv("SSL_CERT_FILE", str(bundle))
    monkeypatch.setenv("SSL_CERT_DIR", "missing")
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))
    monkeypatch.chdir(tmp_path)
    tree = tmp_path / "tree"
    tree.mkdir()
    box = Box()
    with pytest.raises(evaluation.Unconfined):
        evaluation._run(
            tree, tmp_path, "real", Tuning(), provider=None, model=None, python="py", box=box
        )
    [call] = box.calls
    assert call["env"]["HTTPS_PROXY"] == "http://proxy.invalid:3128"
    assert call["env"]["NO_PROXY"] == "localhost"
    assert call["env"]["SSL_CERT_FILE"] == str(bundle)
    assert "SSL_CERT_DIR" not in call["env"]  # a folder that is not there is not passed
    assert bundle in call["read_only"] and bundle.resolve() in call["read_only"]
    # A local model's slots are the run's own: no folder the person's runs trust is writable.
    assert call["env"]["USER"]
    assert Path(call["env"]["XDG_RUNTIME_DIR"]) == tmp_path / "tree-data"
    assert call["writable"] == []


def test_a_relative_certificate_folder_is_given_by_its_whole_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ConfigManager(default_config_path()).save(Config())
    (tmp_path / "certs").mkdir()
    (tmp_path / "more").mkdir()
    monkeypatch.chdir(tmp_path)
    listed = ["certs", "missing", str(tmp_path / "more"), str(tmp_path)]  # the last holds all
    monkeypatch.setenv("SSL_CERT_DIR", os.pathsep.join(listed))
    tree = tmp_path / "tree"
    tree.mkdir()
    box = Box()
    with pytest.raises(evaluation.Unconfined):
        evaluation._run(
            tree, tmp_path, "real", Tuning(), provider=None, model=None, python="py", box=box
        )
    given = os.pathsep.join([str((tmp_path / "certs").absolute()), str(tmp_path / "more")])
    assert box.calls[0]["env"]["SSL_CERT_DIR"] == given


def test_a_config_that_cannot_be_read_fails_the_run(tmp_path: Path) -> None:
    default_config_path().parent.mkdir(parents=True, exist_ok=True)
    default_config_path().write_text("{not json")
    tree = tmp_path / "tree"
    tree.mkdir()
    box = Box()
    with pytest.raises(evaluation.EvaluationError, match="cannot read the config"):
        evaluation._run(
            tree, tmp_path, "real", Tuning(), provider=None, model=None, python="py", box=box
        )
    assert box.calls == []


def test_an_unavailable_sandbox_is_probed_again(monkeypatch: pytest.MonkeyPatch) -> None:
    now = [100.0]
    monkeypatch.setattr(evaluation, "_monotonic", lambda: now[0])
    box = Sandbox(platform="linux")
    probes: list[str] = []
    monkeypatch.setattr(box, "unavailable", lambda: "no bubblewrap")
    monkeypatch.setattr(box, "reprobe", lambda: probes.append("again"))
    assert evaluation.isolation(box) is not None
    now[0] += evaluation.REPROBE_AFTER - 1
    assert evaluation.isolation(box) is not None and probes == []
    now[0] += 1
    evaluation.isolation(box)
    assert probes == ["again"]


def test_a_sandbox_failing_during_a_run_records_nothing(checkout: Path) -> None:
    """bwrap stopping working after its probe is the machine's failure, not the proposal's."""
    s = store()
    p = a_proposal(s, checkout)
    with pytest.raises(evaluation.Unconfined, match="checked"):
        evaluate(s, p, checkout, box=Box())
    assert proposals.evaluations(s.conn, p.id) == []
    assert len(worktrees(checkout)) == 1


def test_a_coding_line_is_read_as_python_reads_it(checkout: Path, tmp_path: Path) -> None:
    """A comment naming another encoding changes what the file runs: it is checked as decoded
    on import, where this text holds line breaks and code."""
    candidate = tmp_path / "candidate"
    shutil.copytree(checkout / "src", candidate / "src")
    path = candidate / "src" / "desman" / "planner.py"
    hidden = "a+AAo-import os+AAo-"
    text = swapped(path.read_text("utf-8"), "PLANNER_RULES", f"PLANNER_RULES = {hidden!r}\n")
    path.write_bytes(b"# -*- coding: utf-7 -*-\n" + text.encode("ascii", "replace"))
    with pytest.raises(evaluation.EvaluationError, match=r"planner\.py|not valid Python"):
        evaluation._check_changes(checkout, candidate, [{"target": "planner.rules", "new": hidden}])
    # Alone, the same text and file read as UTF-8 would pass: the coding line is what is refused.
    path.write_text(text, "utf-8")
    evaluation._check_changes(checkout, candidate, [{"target": "planner.rules", "new": hidden}])


def test_a_sandbox_failing_a_run_is_not_offered_until_probed_again(
    checkout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = [100.0]
    monkeypatch.setattr(evaluation, "_monotonic", lambda: now[0])
    s = store()
    p = a_proposal(s, checkout)
    box = Box()
    probes: list[str] = []
    monkeypatch.setattr(box, "reprobe", lambda: probes.append("again"))
    assert evaluation.isolation(box) is None
    with pytest.raises(evaluation.Unconfined):
        evaluate(s, p, checkout, box=box)
    assert evaluation.isolation(box) == "checked" and probes == []
    now[0] += evaluation.REPROBE_AFTER
    assert evaluation.isolation(box) is None and probes == ["again"]


def test_the_default_sandbox_is_made_again_to_find_bwrap_installed_since(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [100.0]
    monkeypatch.setattr(evaluation, "_monotonic", lambda: now[0])
    evaluation._sandbox.cache_clear()
    try:
        first = evaluation._sandbox()
        monkeypatch.setattr(first, "platform", "linux")
        monkeypatch.setattr(first, "unavailable", lambda: "bubblewrap (bwrap) is not installed")
        assert evaluation.isolation() is not None
        now[0] += evaluation.REPROBE_AFTER
        evaluation.isolation()
        assert evaluation._sandbox() is not first
    finally:
        evaluation._sandbox.cache_clear()


def test_a_model_the_tuning_puts_first_brings_its_providers_key() -> None:
    """As the router reads it: a model chosen for a role is asked first, paid or not."""
    config = Config()
    config.add_key("OPENAI_API_KEY", "sk-paid-key-0000000000000")
    config.add_key("ANTHROPIC_API_KEY", "sk-ant-paid-key-000000000000")
    tuning = Tuning(models=ModelChoices(code="openai/gpt-x"))
    assert set(evaluation.bench_config(config, None, None, tuning).api_keys) == {"OPENAI_API_KEY"}
    assert evaluation.bench_config(config, None, None, Tuning()).api_keys == {}


def test_a_certificate_folder_holding_the_persons_files_is_not_shown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "eval"
    folder.mkdir()
    certs = tmp_path / "certs"
    certs.mkdir()
    bundle = certs / "ca.pem"
    bundle.write_text("certificates")
    assert evaluation._harmless(certs, folder) and evaluation._harmless(bundle, folder)
    for shown in (ROOT, Path.home(), default_config_path().parent, tmp_path, Path("/")):
        assert not evaluation._harmless(shown, folder)
    default_config_path().parent.mkdir(parents=True, exist_ok=True)
    ConfigManager(default_config_path()).save(Config())
    assert not evaluation._harmless(default_config_path(), folder)


def test_a_python_failing_confined_is_checked_again_with_the_sandbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [100.0]
    monkeypatch.setattr(evaluation, "_monotonic", lambda: now[0])
    box = Box()
    monkeypatch.setattr(box, "reprobe", lambda: None)
    evaluation._broken[box] = ("the Python DESMAN runs on does not start there", now[0])
    now[0] += evaluation.REPROBE_AFTER

    def failing(*args: Any, **kwargs: Any) -> None:
        raise evaluation._unconfined(box, "still failing")

    monkeypatch.setattr(evaluation, "_probe_runtime", failing)
    assert evaluation.isolation(box) is not None and "still failing" in str(
        evaluation.isolation(box)
    )
