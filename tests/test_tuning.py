"""Tests for the person's tuning (ROADMAP 5.4): `desman tuning`, and that each part of it is
honoured — the model a role asks first, the limits of plans and splits, the dependency results
an executor sees, the retry budgets of planned tasks."""

from __future__ import annotations

import json
from typing import Any

import pytest
from click.testing import CliRunner, Result

from desman import planner, runs
from desman.cli import cli
from desman.config import Config, ConfigManager, Tuning, default_config_path, open_config, upgrade
from desman.core import db, operator
from desman.core.calls import Role
from desman.core.models import Kind, SubtaskSpec, TaskSpec
from desman.core.store import Store
from desman.interlocutor import CHAT
from desman.llm import router
from desman.llm.fake import FakeLLM
from helpers import make_config

GROQ = {"GROQ_API_KEY": "gsk_test"}


def desman(*args: str) -> Result:
    return CliRunner().invoke(cli, list(args))


def save(config: Config) -> None:
    ConfigManager(default_config_path()).save(config)


def saved() -> Tuning:
    return open_config().load().tuning


# ── The command ──────────────────────────────────────────────────────────────


def test_tuning_shows_the_defaults() -> None:
    save(make_config(language="en", configured=True))
    result = desman("tuning")
    assert result.exit_code == 0, result.output
    assert "most subtasks in a plan" in result.output and "default (8)" in result.output
    assert "model: planner" in result.output and "DESMAN's choice" in result.output
    out = json.loads(desman("tuning", "--json").output)
    assert out["max_subtasks"] is None and out["defaults"]["max_subtasks"] == 8
    assert out["models"]["code"] is None


def test_tuning_changes_and_goes_back_to_the_defaults() -> None:
    save(make_config(language="en", configured=True, api_keys=GROQ))
    model = router.candidates(open_config().load(), "reviewer")[-1]
    result = desman(
        "tuning", "--max-subtasks", "5", "--context-tokens", "6000", "--model", f"reviewer={model}"
    )
    assert result.exit_code == 0, result.output
    assert "most subtasks in a plan: default (8) → 5" in result.output
    assert f"model: reviewer: DESMAN's choice → {model}" in result.output
    tuning = saved()
    assert (tuning.max_subtasks, tuning.context_tokens, tuning.models.reviewer) == (5, 6000, model)
    assert "Nothing changed." in desman("tuning", "--max-subtasks", "5").output
    desman("tuning", "--max-subtasks", "auto")
    assert saved().max_subtasks is None and saved().context_tokens == 6000
    desman("tuning", "--reset")
    assert saved() == Tuning()


@pytest.mark.parametrize(
    ("args", "error"),
    [
        (["--max-parts", "9"], "less than or equal to 6"),
        (["--max-parts", "many"], "--max-parts takes a whole number or auto"),
        (["--model", "chat=groq/x"], "--model takes ROLE=PROVIDER/MODEL"),
        (["--model", "planner=not a label"], "pattern"),
    ],
)
def test_tuning_refuses_what_does_not_fit(args: list[str], error: str) -> None:
    save(make_config(language="en", configured=True))
    result = desman("tuning", *args)
    assert result.exit_code == 2 and error in result.output, result.output
    assert saved() == Tuning()


def test_a_model_no_ready_provider_serves_is_kept_with_a_warning() -> None:
    save(make_config(language="en", configured=True, api_keys=GROQ))
    result = desman("tuning", "--model", "code=mistral/codestral-latest")
    assert result.exit_code == 0
    assert "is not among the models code can use now" in result.output
    assert saved().models.code == "mistral/codestral-latest"


def test_a_config_from_before_has_no_tuning() -> None:
    config = Config.model_validate(upgrade({"schema_version": 3, "language": "de"}))
    assert config.schema_version == 4 and config.tuning == Tuning()


# ── Honoured ─────────────────────────────────────────────────────────────────


def test_the_chosen_model_goes_first_and_the_rest_stay_behind_it() -> None:
    config = make_config(api_keys=GROQ)
    natural = [c.label for c in router.lineup(config, Role.EXECUTOR, kind=Kind.CODE)]
    assert len(natural) > 1
    config.tuning.models.code = natural[-1]
    chosen = [c.label for c in router.lineup(config, Role.EXECUTOR, kind=Kind.CODE)]
    assert chosen == [natural[-1], *natural[:-1]]
    # Other roles and kinds keep their order; a model nobody serves changes nothing.
    assert [c.label for c in router.lineup(config, Role.EXECUTOR, kind=Kind.TEXT)] == [
        c.label for c in router.lineup(make_config(api_keys=GROQ), Role.EXECUTOR, kind=Kind.TEXT)
    ]
    config.tuning.models.code = "mistral/codestral-latest"
    assert [c.label for c in router.lineup(config, Role.EXECUTOR, kind=Kind.CODE)] == natural
    # A pinned --model wins over the tuning.
    config.tuning.models.code = natural[-1]
    pinned = router.lineup(config, Role.EXECUTOR, model=natural[0].split("/", 1)[1], kind=Kind.CODE)
    assert [c.label for c in pinned] == [natural[0]]


def test_the_planner_and_the_splitter_keep_to_the_tuned_limits() -> None:
    def plan_of(n: int) -> str:
        return json.dumps(
            {"subtasks": [{"description": f"s{i}", "acceptance_criteria": "c"} for i in range(n)]}
        )

    fake = FakeLLM().on("Plan task", plan_of(4), plan_of(4))
    planned = planner.plan(fake, "goal", limit=3)
    assert planned.source == "template"  # four subtasks, twice: over the limit of three
    assert "Never more than 3 subtasks." in fake.calls[0].messages[0].content
    fake = FakeLLM().on("Plan task", plan_of(10))
    assert len(planner.plan(fake, "goal", limit=10).spec.subtasks) == 10
    assert "at most 3." in planner.splitter_system(3)
    assert planner.split_schema(3).model_fields["subtasks"].metadata  # bounded


def test_a_run_s_coordinator_takes_the_tuning() -> None:
    config = make_config()
    config.tuning.context_tokens = 7000
    store = Store(db.open_db(":memory:"))
    coordinator = runs.coordinator(store, runs.Models(FakeLLM(), None), None, config)  # type: ignore[arg-type]
    assert coordinator.context_budget_tokens == 7000
    plain = runs.coordinator(store, runs.Models(FakeLLM(), None), None, make_config())  # type: ignore[arg-type]
    assert plain.context_budget_tokens == runs.TUNING_DEFAULTS["context_tokens"]


def test_planned_tasks_take_the_tuned_retry_budgets(tmp_path: Any) -> None:
    spec = TaskSpec(goal="g", subtasks=[SubtaskSpec(description="d", acceptance_criteria="c")])
    tuned = runs.tuned_spec(spec, Tuning(max_attempts=7))
    assert (tuned.retry.max_attempts, tuned.retry.max_transient) == (7, 5)
    save(make_config(language="en", configured=True, tuning={"max_attempts": 3}))
    script = tmp_path / "s.json"
    script.write_text(
        json.dumps(
            {
                "routes": [
                    {"match": "Execute", "steps": ["{}"]},
                    {"match": "Review", "steps": ['{"verdict": "accept"}']},
                ]
            }
        )
    )
    assert desman("run", "a goal", "--fake-llm", str(script)).exit_code == 0
    store = Store(db.open_db())
    [row] = store.conn.execute("SELECT id FROM tasks").fetchall()
    assert store.policy(row[0]).max_attempts == 3


# ── The chat ─────────────────────────────────────────────────────────────────


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeLLM:
    llm = FakeLLM()
    monkeypatch.setattr(runs, "models", lambda config, conn, **_: runs.Models(llm, None))
    save(make_config(language="en", configured=True))
    return llm


def test_the_chat_tunes_only_on_the_person_s_yes(fake: FakeLLM) -> None:
    ask = json.dumps({"type": "command", "name": "tuning", "args": {"max_parts": 3}})
    fake.on(CHAT, ask, ask, json.dumps({"type": "reply", "text": "Done."}))
    declined = CliRunner().invoke(cli, [], input="split into fewer parts\nn\n")
    assert "most parts of a split subtask: default (4) → 3" in declined.output
    assert saved().max_parts is None
    CliRunner().invoke(cli, [], input="split into fewer parts\ny\nthanks\n")
    assert saved().max_parts == 3
    conn = db.open_db()
    assert ("operator_tuning", {"changed": ["max_parts"]}) in [
        (e.action, e.details) for e in operator.entries(conn)
    ]
