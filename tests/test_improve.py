"""Tests for the Improvement Analyst (ROADMAP 5.4): what it reads, what it may change, how its
result becomes a diff, and the `desman proposals` commands. Nothing is ever applied."""

from __future__ import annotations

import ast
import json
import shutil
import sqlite3
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner, Result

from desman import improve, runs
from desman.cli import EXIT_DONE, EXIT_NEEDS_HUMAN, cli
from desman.config import ConfigManager, default_config_path
from desman.core import db, proposals
from desman.core.models import SubtaskSpec, TaskSpec, Tools
from desman.core.proposals import Status
from desman.core.store import Store
from desman.improve import BY_ID, TARGETS, Change, Unfit, render
from desman.interlocutor import CHAT, RETELL
from desman.llm.fake import FakeLLM
from helpers import make_config

ACCEPT = '{"verdict": "accept"}'
RETRY = json.dumps({"verdict": "retry", "feedback": "the criteria ask for a live seat map"})
ROOT = Path(__file__).parent.parent


def desman(*args: str | Path, input: str | None = None) -> Result:
    return CliRunner().invoke(cli, [str(a) for a in args], input=input)


def script(tmp_path: Path, execute: list[Any], review: list[Any] | None = None) -> Path:
    path = tmp_path / f"script-{len(list(tmp_path.glob('script-*')))}.json"
    routes = [
        {"match": "Execute", "steps": execute, "repeat_last": True},
        {"match": "Review", "steps": review or [ACCEPT], "repeat_last": True},
    ]
    path.write_text(json.dumps({"routes": routes}))
    return path


def analysis(
    changes: list[dict[str, Any]], summary: str = "Ask for criteria a check can confirm."
) -> str:
    return json.dumps(
        {
            "changes": changes,
            "summary": summary,
            "evidence": ["the criteria ask for a live seat map"],
            "expected_effect": "fewer rejections for live data",
        }
    )


def criteria_change() -> dict[str, Any]:
    old = improve.current(BY_ID["planner.criteria_rules"])
    assert isinstance(old, str)
    return {"target": "planner.criteria_rules", "value": old + " Never ask for a seat map."}


def a_failure(tmp_path: Path) -> None:
    """A task whose first result the reviewer rejected: one quality failure in the history."""
    fake = script(tmp_path, ['{"seats": "?"}'], [RETRY, ACCEPT])
    assert desman("run", "Book a seat", "--fake-llm", fake).exit_code == EXIT_DONE


def store() -> Store:
    return Store(db.open_db())


# ── The targets ──────────────────────────────────────────────────────────────


def test_every_target_is_a_plain_value_outside_the_protected_files() -> None:
    manifest = tomllib.loads((improve.PACKAGE / "protected.toml").read_text(encoding="utf-8"))
    protected = set(manifest["package"])
    for target in TARGETS:
        assert target.path not in protected, target.id
        value = improve.current(target)
        if isinstance(value, int):
            assert target.minimum is not None and target.maximum is not None
            assert target.minimum <= value <= target.maximum, target.id
        else:
            assert all(k in value for k in target.keep), target.id
            assert len(value) <= target.max_chars, target.id
    assert len(BY_ID) == len(TARGETS)


def test_the_planner_prompts_are_made_of_the_targets() -> None:
    from desman import planner

    assert planner.PLANNER_SYSTEM.startswith(planner.PLANNER_RULES)
    assert planner.SPLITTER_SYSTEM.startswith(planner.SPLITTER_RULES)
    assert str(planner.MAX_SUBTASKS) in planner.PLANNER_SYSTEM
    assert str(planner.MAX_PARTS) in planner.SPLITTER_SYSTEM


# ── From changes to a diff ───────────────────────────────────────────────────


def copy_package(tmp_path: Path) -> Path:
    """The targets' files in a checkout-like tree: tmp/src/desman/..."""
    package = tmp_path / "src" / "desman"
    for path in {t.path for t in TARGETS}:
        (package / path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(improve.PACKAGE / path, package / path)
    return package


def apply(tmp_path: Path, diff: str) -> None:
    patch = tmp_path / "p.diff"
    patch.write_text("# a comment line, as in the file DESMAN writes\n\n" + diff)
    subprocess.run(["git", "apply", str(patch)], cwd=tmp_path, check=True)


def test_a_diff_applies_and_makes_exactly_the_change(tmp_path: Path) -> None:
    package = copy_package(tmp_path)
    old_rules = improve.current(BY_ID["planner.rules"], package)
    assert isinstance(old_rules, str)
    new_rules = old_rules.replace("(one is fine", "(one is best")
    changes = [
        Change(target="planner.rules", value=new_rules),
        Change(target="planner.max_subtasks", value=6),
        Change(target="context.budget_tokens", value=6000),
        Change(target="planner.criteria_rules", value=criteria_change()["value"] + " \"q\" 's' \\"),
    ]
    before = {p: (package / p).read_text() for p in ("planner.py", "core/context.py")}
    diff, details = render(changes[:3] + changes[3:], package)
    assert [c["target"] for c in details["changes"]] == [c.target for c in changes]
    assert set(details["base"]) == {"src/desman/planner.py", "src/desman/core/context.py"}
    assert (package / "planner.py").read_text() == before["planner.py"]  # nothing written
    apply(tmp_path, diff)
    for change in changes:
        assert improve.current(BY_ID[change.target], package) == change.value
    # A one-word change in a long text is a one-line change in the diff.
    assert sum(line.startswith("-    ") for line in diff.splitlines()) == 2  # rules + criteria
    ruff = shutil.which("ruff") or str(Path(sys.executable).parent / "ruff")
    if Path(ruff).exists():
        checked = subprocess.run(
            [ruff, "format", "--check", "--config", str(ROOT / "pyproject.toml"), str(package)],
            capture_output=True,
            text=True,
        )
        assert checked.returncode == 0, checked.stdout + checked.stderr


def test_a_glued_text_keeps_its_spaces_at_either_end(tmp_path: Path) -> None:
    package = copy_package(tmp_path)
    old = improve.current(BY_ID["planner.kind_rules"], package)
    assert isinstance(old, str) and old.startswith(" ")
    new = old.strip().replace("so a model", "so that a model")
    _diff, details = render([Change(target="planner.kind_rules", value=new)], package)
    assert details["changes"][0]["new"] == " " + new


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ([{"target": "agents.executor", "value": "x"}], "'agents.executor' is not a target"),
        ([{"target": "planner.max_parts", "value": "4"}], "planner.max_parts is a number"),
        ([{"target": "planner.max_parts", "value": 9}], "must be from 2 to 6"),
        ([{"target": "planner.rules", "value": 3}], "planner.rules is a text"),
        ([{"target": "planner.rules", "value": "Plan well."}], 'must still contain "subtasks"'),
        ([{"target": "planner.max_subtasks", "value": 8}], "would stay as it is"),
        ([{"target": "planner.criteria_rules", "value": "  "}], "cannot be empty"),
        (
            [
                {"target": "planner.max_parts", "value": 3},
                {"target": "planner.max_parts", "value": 5},
            ],
            "changed twice",
        ),
    ],
)
def test_a_change_that_cannot_be_made_is_refused(
    changes: list[dict[str, Any]], reason: str
) -> None:
    with pytest.raises(Unfit, match=reason.replace("(", r"\(")):
        render([Change.model_validate(c) for c in changes])


# ── What an analysis reads ───────────────────────────────────────────────────


def test_nothing_to_analyse_creates_no_task(tmp_path: Path) -> None:
    desman("init-db")
    result = desman("proposals", "analyze", "--fake-llm", script(tmp_path, ["{}"]))
    assert result.exit_code == 0, result.output
    assert "Nothing to analyse" in result.output
    assert store().conn.execute("SELECT count(*) FROM tasks").fetchone()[0] == 0


def test_the_task_reads_the_failures_and_has_no_tools(tmp_path: Path) -> None:
    a_failure(tmp_path)
    s = store()
    proposal = improve.start(s, None)
    assert proposal is not None and proposal.status is Status.ANALYSING
    assert s.tools(proposal.task_id) is Tools.NONE
    [sub] = s.list_subtasks(proposal.task_id)
    inputs = sub.inputs
    [failure] = inputs["failures"]
    assert failure["subtask"] == "Book a seat" and failure["what"] == "retry"
    assert failure["said"] == "the criteria ask for a live seat map"
    assert [t["target"] for t in inputs["targets"]] == [t.id for t in TARGETS]
    assert inputs["stats"]["reviews"] == {"accept": 1, "retry": 1}
    assert sub.kind.value == "analysis"


def test_plan_fallbacks_are_read_too(tmp_path: Path) -> None:
    s = store()
    spec = TaskSpec(goal="g", subtasks=[SubtaskSpec(description="g", acceptance_criteria="c")])
    s.create_task(
        spec, details={"plan": "template", "plan_error": "StructuredOutputError: no JSON"}
    )
    [fallback] = proposals.plan_fallbacks(s.conn, None)
    assert fallback == {
        "at": fallback["at"],
        "goal": "g",
        "error": "StructuredOutputError: no JSON",
    }
    assert improve.spec(s, None) is not None


def test_the_analyst_does_not_study_its_own_failures(tmp_path: Path) -> None:
    a_failure(tmp_path)
    fake = script(tmp_path, ['{"not": "an analysis"}'], [RETRY, ACCEPT])
    assert desman("proposals", "analyze", "--fake-llm", fake).exit_code == EXIT_DONE
    s = store()
    assert len(proposals.failures(s.conn, None)) == 1  # the analysis's rejection is not there
    [p] = proposals.listing(s.conn)
    assert p.status is Status.UNUSABLE and p.reason is not None
    assert p.reason.startswith("not an analysis: changes: Field required")


def test_a_window_reads_only_what_happened_in_it(tmp_path: Path) -> None:
    a_failure(tmp_path)
    assert proposals.failures(store().conn, "2999-01-01T00:00:00.000Z") == []
    result = desman("proposals", "analyze", "--since", "1m", "--fake-llm", script(tmp_path, ["{}"]))
    assert "Nothing to analyse" not in result.output  # it just happened


# ── desman proposals ─────────────────────────────────────────────────────────


def test_analyze_proposes_a_diff_and_applies_nothing(tmp_path: Path) -> None:
    a_failure(tmp_path)
    planner = (improve.PACKAGE / "planner.py").read_text()
    fake = script(tmp_path, [analysis([criteria_change()])])
    result = desman("proposals", "analyze", "--fake-llm", fake)
    assert result.exit_code == EXIT_DONE, result.output
    assert "Analysing the failures as task" in result.output
    assert "Proposal 1: proposed" in result.output
    assert "Never ask for a seat map." in result.output
    assert "Nothing was applied." in result.output
    assert (improve.PACKAGE / "planner.py").read_text() == planner
    [p] = proposals.listing(store().conn)
    assert p.status is Status.PROPOSED and p.diff is not None
    assert p.details["evidence"] == ["the criteria ask for a live seat map"]
    path = db.default_db_path().parent / "proposals" / "1.diff"
    text = path.read_text()
    assert text.startswith("# DESMAN proposal 1") and text.endswith(p.diff)

    listed = desman("proposals")
    assert "proposed  planner.criteria_rules" in listed.output
    shown = desman("proposals", "show", "1", "--json")
    out = json.loads(shown.output)
    assert out["status"] == "proposed" and out["file"] == str(path) and out["diff"] == p.diff
    assert desman("proposals", "show", "7").exit_code == 1


def test_analyze_json(tmp_path: Path) -> None:
    a_failure(tmp_path)
    fake = script(tmp_path, [analysis([], summary="The one failure was the goal's own.")])
    result = desman("proposals", "analyze", "--fake-llm", fake, "--json")
    out = json.loads(result.output)
    assert out["task"]["status"] == "done"
    assert out["proposal"]["status"] == "nothing" and out["proposal"]["diff"] is None
    assert out["proposal"]["summary"] == "The one failure was the goal's own."


def test_a_change_the_code_refuses_is_unusable_with_the_reason(tmp_path: Path) -> None:
    a_failure(tmp_path)
    fake = script(tmp_path, [analysis([{"target": "planner.max_parts", "value": 40}])])
    desman("proposals", "analyze", "--fake-llm", fake)
    shown = desman("proposals", "show", "1")
    assert "Proposal 1: unusable" in shown.output
    assert "planner.max_parts must be from 2 to 6" in shown.output


def test_an_analysis_that_stopped_is_settled_once_it_ends(tmp_path: Path) -> None:
    a_failure(tmp_path)
    escalate = json.dumps({"verdict": "escalate", "feedback": "unclear"})
    stopped = desman(
        "proposals", "analyze", "--fake-llm", script(tmp_path, [analysis([])], [escalate])
    )
    assert stopped.exit_code == EXIT_NEEDS_HUMAN
    assert "has not ended. Continue it: desman resume" in stopped.output
    # Another analyze continues it rather than starting a second one.
    again = desman("proposals", "analyze", "--fake-llm", script(tmp_path, [analysis([])]))
    assert "Continuing the analysis already under way" in again.output
    s = store()
    [p] = proposals.listing(s.conn)
    assert p.status is Status.ANALYSING
    # The person accepts the result as it is; the next look at the proposals reads it.
    [sub] = s.list_subtasks(p.task_id)
    assert desman("accept", sub.id).exit_code == 0
    assert "nothing to propose" in desman("proposals", "list").output


def test_a_settled_proposal_never_changes(tmp_path: Path) -> None:
    a_failure(tmp_path)
    desman("proposals", "analyze", "--fake-llm", script(tmp_path, [analysis([criteria_change()])]))
    conn = store().conn
    with pytest.raises(sqlite3.IntegrityError, match="never changes"):
        conn.execute("UPDATE proposals SET diff = 'x'")
    with pytest.raises(sqlite3.IntegrityError, match="kept"):
        conn.execute("DELETE FROM proposals")
    assert not proposals.settle(conn, 1, Status.NOTHING, summary="again")


def test_an_analysis_task_without_its_row_is_adopted(tmp_path: Path) -> None:
    a_failure(tmp_path)
    s = store()
    task = improve.spec(s, None)
    assert task is not None
    task_id = s.create_task(task, details={"purpose": proposals.PURPOSE, "since": None})
    improve.settle(s)
    [p] = proposals.listing(s.conn)
    assert p.task_id == task_id and p.status is Status.ANALYSING


def test_proposals_with_none_yet() -> None:
    desman("init-db")
    result = desman("proposals")
    assert result.exit_code == 0 and "No proposals yet" in result.output
    assert json.loads(desman("proposals", "list", "--json").output) == []


def test_a_v21_database_upgrades_to_proposals(tmp_path: Path) -> None:
    path = tmp_path / "v21.db"
    old = db.connect(path)
    for v, migration in enumerate(db.MIGRATIONS[:21], start=1):
        with db.transaction(old):
            for statement in db._split_sql(migration):
                old.execute(statement)
            old.execute(f"PRAGMA user_version = {v}")
    old.execute("INSERT INTO tasks (id, goal) VALUES ('t1', 'goal')")
    old.close()
    conn = db.open_db(path)
    assert db.schema_version(conn) == db.SCHEMA_VERSION
    assert proposals.listing(conn) == []
    assert conn.execute("SELECT goal FROM tasks").fetchone()[0] == "goal"


# ── The chat ─────────────────────────────────────────────────────────────────


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeLLM:
    llm = FakeLLM()
    monkeypatch.setattr(runs, "models", lambda config, conn, **_: runs.Models(llm, None))
    ConfigManager(default_config_path()).save(make_config(language="en", configured=True))
    return llm


def test_the_chat_analyses_only_after_the_window(fake: FakeLLM, tmp_path: Path) -> None:
    a_failure(tmp_path)
    ask = json.dumps({"type": "command", "name": "proposals analyze", "args": {}})
    fake.on(CHAT, ask, ask)
    fake.on("Execute", analysis([criteria_change()]))
    fake.on("Review", ACCEPT)
    fake.on(RETELL, json.dumps({"text": "Done: one proposal."}), repeat_last=True)
    declined = desman(input="improve yourself\nn\n")
    assert "Have the Improvement Analyst study every failure kept" in declined.output
    assert proposals.listing(store().conn) == []

    accepted = desman(input="improve yourself\ny\n")
    assert accepted.exit_code == 0, accepted.output
    assert "Proposal 1: proposed" in accepted.output
    [p] = proposals.listing(store().conn)
    assert p.status is Status.PROPOSED


def test_the_diff_is_parseable_python(tmp_path: Path) -> None:
    package = copy_package(tmp_path)
    diff, _ = render([Change(target="planner.max_parts", value=3)], package)
    apply(tmp_path, diff)
    ast.parse((package / "planner.py").read_text())
