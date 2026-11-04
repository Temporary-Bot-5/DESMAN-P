"""Tests for evaluating a proposal (ROADMAP 5.5): the benchmark, the comparison, the runs in git
worktrees of a checkout, and `desman proposals evaluate`. Nothing is ever applied.

The worktrees are made in a copy of this checkout (a git repository of its own in a temporary
folder), and the benchmark runs in real subprocesses on the scripted fake model."""

from __future__ import annotations

import dataclasses
import difflib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner, Result

from desman import bench, evaluation, improve, runs
from desman.cli import cli
from desman.config import Config, ModelChoices, Tuning, default_config_path
from desman.core import db, proposals
from desman.core.models import SubtaskSpec, TaskSpec
from desman.core.proposals import Proposal, Status
from desman.core.store import Store
from desman.improve import BY_ID, Change

ROOT = Path(__file__).parent.parent
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
    assert asked == [{"tuning": tuning, "provider": "groq", "model": "m", "cache": False}]


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
    planner.write_text(planner.read_text() + "\n# changed since\n")
    git(checkout, "commit", "-q", "-am", "later")
    with pytest.raises(evaluation.EvaluationError, match="other versions of src/desman/planner"):
        evaluate(s, p, checkout)
    assert proposals.evaluations(s.conn, p.id) == []
    assert len(worktrees(checkout)) == 1


def edited_diff(repo: Path, edit: Any) -> tuple[str, dict[str, Any]]:
    """The diff of a real proposal, with `edit(text)` also applied to planner.py."""
    diff, details = improve.render([rules_change(repo)], package(repo))
    name = improve.SOURCE_PREFIX + "planner.py"
    before = (repo / name).read_text()
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
        evaluate(s, p, checkout)
    assert proposals.evaluations(s.conn, p.id) == []


def test_a_diff_giving_other_texts_than_proposed_is_refused(checkout: Path) -> None:
    s = store()
    diff, details = improve.render([rules_change(checkout)], package(checkout))
    details["changes"][0]["new"] += " Something else."
    p = proposal(s, diff, details)
    with pytest.raises(evaluation.EvaluationError, match="not the one the proposal's texts make"):
        evaluate(s, p, checkout)


def test_a_diff_touching_another_file_is_refused(checkout: Path) -> None:
    s = store()
    name = "src/desman/agents.py"
    before = (checkout / name).read_text()
    after = before.replace("MAX_TOOL_ROUNDS = 8", "MAX_TOOL_ROUNDS = 80")
    lines = difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True), f"a/{name}", f"b/{name}"
    )
    p = proposal(s, f"diff --git a/{name} b/{name}\n" + "".join(lines), {"changes": []})
    with pytest.raises(evaluation.EvaluationError, match="not the one the proposal's texts make"):
        evaluate(s, p, checkout)


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
        evaluate(s, p, checkout)


def test_a_target_text_that_may_not_be_one_is_refused(checkout: Path) -> None:
    s = store()
    p = a_proposal(s, checkout)
    details = p.details | {"changes": [p.details["changes"][0] | {"new": "no format"}]}
    q = proposal(s, p.diff or "", details)
    with pytest.raises(evaluation.EvaluationError, match="must still contain"):
        evaluate(s, q, checkout)


def test_a_run_that_fails_is_recorded_as_failed(checkout: Path, tmp_path: Path) -> None:
    s = store()
    p = a_proposal(s, checkout)
    found = evaluate(s, p, checkout, python=str(tmp_path / "no-python"))
    assert found.verdict == "failed" and found.base is None and found.candidate is None
    assert found.reason is not None and "current version's run failed" in found.reason
    assert len(worktrees(checkout)) == 1


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
    default_config_path().write_text("{not json")
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


def test_fake_child_environment_has_no_inherited_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "parent-secret")
    monkeypatch.setenv("HF_TOKEN", "parent-secret")
    tree = tmp_path / "tree"
    tree.mkdir()

    def child(*args: Any, **kwargs: Any) -> Any:
        env = kwargs["env"]
        assert set(env) <= {
            "PATH",
            "LANG",
            "LC_ALL",
            "HOME",
            "PYTHONPATH",
            "DESMAN_DATA_DIR",
            "DESMAN_CONFIG_DIR",
            "PYTHONDONTWRITEBYTECODE",
        }
        assert Path(env["HOME"]).is_relative_to(tmp_path)
        assert Path(env["HOME"]).is_dir()
        assert env["PYTHONDONTWRITEBYTECODE"] == "1"
        raise OSError("checked environment")

    monkeypatch.setattr(subprocess, "run", child)
    with pytest.raises(evaluation.RunFailed, match="checked environment"):
        evaluation._run(
            tree, tmp_path, "fake", Tuning(), provider=None, model=None, python="python"
        )
