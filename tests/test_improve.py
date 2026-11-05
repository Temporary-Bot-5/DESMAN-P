"""Tests for the Improvement Analyst (ROADMAP 5.4): what it reads, what it may change, how its
result becomes a diff, and the `desman proposals` commands. Nothing is ever applied."""

from __future__ import annotations

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
from desman.config import (
    TUNING_LIMITS,
    ConfigManager,
    ModelChoices,
    default_config_path,
    open_config,
)
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
        assert all(k in value for k in target.keep), target.id
        assert len(value) <= target.max_chars, target.id
    assert len(BY_ID) == len(TARGETS)
    assert not set(BY_ID) & set(improve.SETTINGS_BY_ID)


def test_every_setting_is_a_part_of_the_tuning_and_never_a_key_or_permission() -> None:
    names = {s.name for s in improve.SETTINGS}
    assert names == set(TUNING_LIMITS) | set(ModelChoices.model_fields)
    assert all(s.id.startswith("tuning.") for s in improve.SETTINGS)


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
    patch.write_bytes(("# a comment line, as in the file DESMAN writes\n\n" + diff).encode("utf-8"))
    subprocess.run(["git", "apply", str(patch)], cwd=tmp_path, check=True)


def test_a_diff_applies_and_makes_exactly_the_change(tmp_path: Path) -> None:
    package = copy_package(tmp_path)
    old_rules = improve.current(BY_ID["planner.rules"], package)
    assert isinstance(old_rules, str)
    new_rules = old_rules.replace("(one is fine", "(one is best")
    changes = [
        Change(target="planner.rules", value=new_rules),
        Change(target="planner.criteria_rules", value=criteria_change()["value"] + " \"q\" 's' \\"),
    ]
    before = {p: (package / p).read_text() for p in ("planner.py",)}
    diff, details = render(changes, package)
    assert [c["target"] for c in details["changes"]] == [c.target for c in changes]
    assert set(details["base"]) == {"src/desman/planner.py"}
    assert "settings" not in details
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
        ([{"target": "tuning.max_parts", "value": "4"}], "tuning.max_parts is a number"),
        ([{"target": "tuning.max_parts", "value": 9}], "must be from 2 to 6"),
        ([{"target": "tuning.max_parts", "value": None}], "would stay as it is"),
        ([{"target": "tuning.model.planner", "value": "paid/model"}], "one of the models shown"),
        ([{"target": "planner.rules", "value": 3}], "planner.rules is a text"),
        ([{"target": "planner.rules", "value": "Plan well."}], 'must still contain "subtasks"'),
        ([{"target": "planner.rules", "value": None}], "planner.rules is a text"),
        ([{"target": "planner.criteria_rules", "value": "  "}], "cannot be empty"),
        (
            [
                {"target": "tuning.max_parts", "value": 3},
                {"target": "tuning.max_parts", "value": 5},
            ],
            "changed twice",
        ),
    ],
)
def test_a_change_that_cannot_be_made_is_refused(
    changes: list[dict[str, Any]], reason: str
) -> None:
    shown = improve.settings_shown(make_config(api_keys={"GROQ_API_KEY": "gsk_test"}))
    with pytest.raises(Unfit, match=reason.replace("(", r"\(")):
        render([Change.model_validate(c) for c in changes], shown=shown)


def test_a_setting_change_is_checked_against_what_was_shown() -> None:
    config = make_config(api_keys={"GROQ_API_KEY": "gsk_test"})
    shown = improve.settings_shown(config)
    by_id = {s["setting"]: s for s in shown}
    choices = by_id["tuning.model.reviewer"]["choices"]
    assert choices and all(c.startswith("groq/") for c in choices)
    assert by_id["tuning.max_subtasks"] | {"about": ""} == {
        "setting": "tuning.max_subtasks",
        "about": "",
        "current": None,
        "default": 8,
        "limits": {"min": 1, "max": 12},
    }
    diff, details = render(
        [
            Change(target="tuning.max_subtasks", value=5),
            Change(target="tuning.model.reviewer", value=choices[-1]),
        ],
        shown=shown,
    )
    assert diff == ""  # settings are not a diff
    assert details["settings"] == [
        {"setting": "tuning.max_subtasks", "old": None, "new": 5},
        {"setting": "tuning.model.reviewer", "old": None, "new": choices[-1]},
    ]
    assert details["command"] == f"desman tuning --max-subtasks 5 --model reviewer={choices[-1]}"
    # Without having been shown, a setting cannot change.
    with pytest.raises(Unfit, match="not among the settings shown"):
        render([Change(target="tuning.max_subtasks", value=5)], shown=[])
    # Back to the default: auto.
    config.tuning.max_parts = 3
    _diff, details = render(
        [Change(target="tuning.max_parts", value=None)], shown=improve.settings_shown(config)
    )
    assert details["command"] == "desman tuning --max-parts auto"


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
    assert "Not applied. If you agree, apply it: desman proposals apply 1" in result.output
    assert (improve.PACKAGE / "planner.py").read_text() == planner
    [p] = proposals.listing(store().conn)
    assert p.status is Status.PROPOSED and p.diff is not None
    assert p.details["evidence"] == ["the criteria ask for a live seat map"]
    path = db.default_db_path().parent / "proposals" / "1.diff"
    text = path.read_text(encoding="utf-8")
    assert path.read_bytes() == text.encode("utf-8")
    assert text.startswith("# DESMAN proposal 1") and text.endswith(p.diff)
    assert p.summary is not None and p.summary not in text  # the model's words stay in the DB

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
    fake = script(tmp_path, [analysis([{"target": "tuning.max_parts", "value": 40}])])
    desman("proposals", "analyze", "--fake-llm", fake)
    shown = desman("proposals", "show", "1")
    assert "Proposal 1: unusable" in shown.output
    assert "tuning.max_parts must be from 2 to 6" in shown.output


def test_a_proposed_setting_is_made_by_the_person_with_the_command(tmp_path: Path) -> None:
    a_failure(tmp_path)
    change = [{"target": "tuning.max_attempts", "value": 6}, criteria_change()]
    result = desman("proposals", "analyze", "--fake-llm", script(tmp_path, [analysis(change)]))
    assert result.exit_code == EXIT_DONE, result.output
    assert "tuning.max_attempts: default → 6" in result.output
    assert "If you agree, apply it: desman proposals apply 1" in result.output
    [p] = proposals.listing(store().conn)
    assert improve.changed(p) == ["planner.criteria_rules", "tuning.max_attempts"]
    text = (db.default_db_path().parent / "proposals" / "1.diff").read_text()
    assert "# The settings change with: desman tuning --max-attempts 6" in text
    assert open_config().load().tuning.max_attempts is None  # nothing applied
    assert desman("tuning", "--max-attempts", "6").exit_code == 0
    assert open_config().load().tuning.max_attempts == 6


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


def test_the_analyst_sees_the_settings_and_the_models_stats(tmp_path: Path) -> None:
    a_failure(tmp_path)
    s = store()
    config = make_config(api_keys={"GROQ_API_KEY": "gsk_test"})
    config.tuning.max_subtasks = 4
    task = improve.spec(s, None, config)
    assert task is not None
    inputs = task.subtasks[0].inputs
    by_id = {x["setting"]: x for x in inputs["settings"]}
    assert by_id["tuning.max_subtasks"]["current"] == 4
    assert by_id["tuning.model.code"]["choices"]
    assert "models" in inputs["stats"] and "roles" in inputs["stats"]
    assert "api_keys" not in json.dumps(inputs) and "gsk_test" not in json.dumps(inputs)


# ── Review fixes ─────────────────────────────────────────────────────────────


def test_only_one_analysis_is_under_way_and_its_task_never_lacks_its_row(tmp_path: Path) -> None:
    a_failure(tmp_path)
    s = store()
    first = improve.start(s, None)
    assert first is not None
    tasks = s.conn.execute("SELECT count(*) FROM tasks").fetchone()[0]
    with pytest.raises(proposals.UnderWay):
        improve.start(s, None)  # another process, a moment later
    assert s.conn.execute("SELECT count(*) FROM tasks").fetchone()[0] == tasks  # none created
    from desman import proposals_cli

    again = proposals_cli.prepare(s, None)
    assert again is not None and again.proposal.id == first.id and not again.new


def test_a_text_changed_since_the_analysis_read_it_is_not_diffed(tmp_path: Path) -> None:
    change = Change.model_validate(criteria_change())
    old = improve.current(BY_ID["planner.criteria_rules"])
    render([change], read={"planner.criteria_rules": old})  # as read: fine
    with pytest.raises(Unfit, match="changed since the analysis read it"):
        render([change], read={"planner.criteria_rules": old + " (an older text)"})


def test_the_analyst_s_stats_leave_out_its_own_tasks(tmp_path: Path) -> None:
    a_failure(tmp_path)
    fake = script(tmp_path, [analysis([])], [RETRY, ACCEPT])  # the analysis is rejected once
    desman("proposals", "analyze", "--fake-llm", fake)
    from desman.core.stats import collect

    conn = store().conn
    assert collect(conn).reviews == {"accept": 2, "retry": 2}
    assert collect(conn, skip=proposals.PURPOSE).reviews == {"accept": 1, "retry": 1}


def test_control_characters_never_reach_the_terminal_or_the_file(tmp_path: Path) -> None:
    a_failure(tmp_path)
    sneaky = analysis([criteria_change()], summary="Fine.\x1b[2J\x07 Really\nfine.")
    desman("proposals", "analyze", "--fake-llm", script(tmp_path, [sneaky]))
    [p] = proposals.listing(store().conn)
    assert p.summary == "Fine.[2J Really\nfine."
    with pytest.raises(Unfit, match="control characters"):
        render([Change(target="planner.criteria_rules", value="Criteria.\x1b[31m red")])
