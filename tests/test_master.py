"""The master module (ADR 0015, ROADMAP M.1): a whole goal for one qualified model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner, Result

from desman import runs
from desman.agents import MASTER_INSTRUCTIONS, MASTER_TOOL_ROUNDS, MAX_TOOL_ROUNDS, llm_executor
from desman.cli import EXIT_DONE, cli
from desman.core import db
from desman.core.audit import check_history, history
from desman.core.calls import Role
from desman.core.context import build_context
from desman.core.coordinator import Failure, Success, Work
from desman.core.models import Kind, Origin, SubtaskSpec, TaskSpec
from desman.core.retry import FailureKind
from desman.core.store import Store
from desman.core.tools import Toolbox
from desman.core.workspace import Workspace
from desman.llm import catalog, router
from desman.llm.fake import FakeLLM
from desman.planner import MASTER_CRITERIA, master_plan
from helpers import make_config

GROQ = "gsk_" + "a" * 48
NVIDIA = "nvapi-" + "n" * 60 + "WxYz"
ULTRA = "nvidia/nvidia/nemotron-3-ultra-550b-a55b"


def labels(choices: list[router.Choice]) -> list[str]:
    return [c.label for c in choices]


# ── Who may be a master ──────────────────────────────────────────────────────


def test_every_model_marked_master_meets_what_a_master_needs() -> None:
    marked = [m for models in catalog.CATALOG.values() for m in models if m.master]
    assert marked and all(catalog.qualifies(m) for m in marked)
    assert catalog.model(catalog_provider("nvidia"), "nvidia/nemotron-3-ultra-550b-a55b").master
    # A per-request ceiling ends a long loop at its first big request: never a master.
    assert not any(m.master for m in catalog.CATALOG["groq"])


def catalog_provider(provider_id: str) -> Any:
    from desman import providers

    return providers.find(provider_id)


def test_only_qualified_models_are_in_the_master_lineup() -> None:
    config = make_config(api_keys={"GROQ_API_KEY": GROQ, "NVIDIA_API_KEY": NVIDIA})
    masters = labels(router.lineup(config, Role.EXECUTOR, kind=catalog.MASTER))
    assert masters == [ULTRA]
    # The ordinary lineup is untouched: groq goes first there.
    assert labels(router.lineup(config, Role.EXECUTOR))[0].startswith("groq/")


def test_without_a_qualified_model_there_is_no_master() -> None:
    config = make_config(api_keys={"GROQ_API_KEY": GROQ})
    with pytest.raises(router.NoProvider, match="master module"):
        router.lineup(config, Role.EXECUTOR, kind=catalog.MASTER)
    # A model given with --model is a master only if the catalog marks it.
    with pytest.raises(router.NoProvider):
        router.lineup(config, Role.EXECUTOR, "groq", "openai/gpt-oss-120b", kind=catalog.MASTER)


def local(context: int, *, named: bool) -> Any:
    settings = {"LLAMA_CPP_MODEL": "big", "LLAMA_CPP_CONTEXT": str(context)}
    data: dict[str, Any] = {"settings": settings}
    if named:
        data["tuning"] = {"models": {"master": "llamacpp/big"}}
    return make_config(**data)


def test_a_local_model_is_a_master_only_when_named_and_large_enough() -> None:
    assert labels(router.lineup(local(65_536, named=True), Role.EXECUTOR, kind=catalog.MASTER)) == [
        "llamacpp/big"
    ]
    with pytest.raises(router.NoProvider):  # not named by the person
        router.lineup(local(65_536, named=False), Role.EXECUTOR, kind=catalog.MASTER)
    with pytest.raises(router.NoProvider):  # named, but its context cannot hold a long loop
        router.lineup(local(8_192, named=True), Role.EXECUTOR, kind=catalog.MASTER)


def test_the_persons_master_goes_first_and_an_unqualified_one_is_ignored() -> None:
    keys = {"NVIDIA_API_KEY": NVIDIA, "MISTRAL_API_KEY": "m" * 32}
    chosen = make_config(
        api_keys=keys, tuning={"models": {"master": "mistral/mistral-medium-latest"}}
    )
    assert labels(router.lineup(chosen, Role.EXECUTOR, kind=catalog.MASTER))[0] == (
        "mistral/mistral-medium-latest"
    )
    unqualified = make_config(
        api_keys=keys, tuning={"models": {"master": "mistral/mistral-small-latest"}}
    )
    masters = labels(router.lineup(unqualified, Role.EXECUTOR, kind=catalog.MASTER))
    assert "mistral/mistral-small-latest" not in masters and masters


def test_the_run_has_master_models_only_when_one_qualifies(tmp_path: Path) -> None:
    conn = db.open_db(tmp_path / "d.db")
    try:
        ready = runs.models(
            make_config(api_keys={"GROQ_API_KEY": GROQ, "NVIDIA_API_KEY": NVIDIA}), conn
        )
        assert ready.master is not None and ready.master_choice is not None
        assert ready.master_choice.label == ULTRA and ready.no_master is None
        none = runs.models(make_config(api_keys={"GROQ_API_KEY": GROQ}), conn)
        assert none.master is None and none.no_master and "master module" in none.no_master
        # desman-global's crew never has a master (ADR 0015, decision 6).
        lent = runs.models(
            make_config(api_keys={"GROQ_API_KEY": GROQ, "NVIDIA_API_KEY": NVIDIA}),
            conn,
            only={"nvidia"},
        )
        assert lent.master is None
    finally:
        conn.close()


# ── The subtask ──────────────────────────────────────────────────────────────


def test_a_master_subtask_is_stored_and_a_job_is_never_one(store: Store) -> None:
    task_id = store.create_task(master_plan("Write the program"))
    [sub] = store.list_subtasks(task_id)
    assert sub.master and sub.acceptance_criteria == MASTER_CRITERIA
    [created] = [e for e in history(store.conn, task_id) if e.action == "task_created"]
    assert created.details["master"] == [sub.id]
    with pytest.raises(ValueError, match="master"):
        store.create_task(master_plan("a job"), origin=Origin.NETWORK)
    plain = store.create_task(
        TaskSpec(goal="g", subtasks=[SubtaskSpec(description="d", acceptance_criteria="c")])
    )
    assert not store.list_subtasks(plain)[0].master


def test_the_parts_of_a_split_master_subtask_are_ordinary(store: Store) -> None:
    store.create_task(master_plan("Big goal"))
    claimed = store.claim("w")
    assert claimed is not None and claimed.master and claimed.lease_token is not None
    parts = store.decompose(
        claimed.id,
        claimed.lease_token,
        [
            SubtaskSpec(description="one", acceptance_criteria="c", kind=Kind.CODE),
            SubtaskSpec(description="two", acceptance_criteria="c", depends_on=[0]),
        ],
    )
    assert [p.master for p in parts] == [False, False]
    assert all(p.parent_id == claimed.id for p in parts)


def test_a_database_before_masters_upgrades_with_ordinary_subtasks(tmp_path: Path) -> None:
    path = tmp_path / "old.db"
    conn = db.connect(path)
    for version, script in enumerate(db.MIGRATIONS[:-1], start=1):
        with db.transaction(conn):
            for statement in db._split_sql(script):
                conn.execute(statement)
            conn.execute(f"PRAGMA user_version = {version}")
    old = Store(conn)
    conn.execute(
        "INSERT INTO tasks (id, goal, priority, created_at, max_attempts, max_transient, tools, "
        "origin) VALUES ('t', 'g', 50, '2026-11-21', 5, 5, 'granted', 'local')"
    )
    conn.execute(
        "INSERT INTO subtasks (id, task_id, seq, description, acceptance_criteria, inputs_json, "
        "priority, kind, runs_code, knowledge_json, created_at, updated_at) VALUES "
        "('s', 't', 0, 'd', 'c', '{}', 50, 'text', 0, '[]', '2026-11-21', '2026-11-21')"
    )
    conn.commit()
    del old
    conn.close()
    upgraded = Store(db.open_db(path))
    try:
        assert upgraded.get_subtask("s").master is False
    finally:
        upgraded.conn.close()


# ── The executor ─────────────────────────────────────────────────────────────


def no_heartbeat(checkpoint: dict[str, Any] | None = None) -> None:
    pass


def list_files() -> str:
    return json.dumps({"tool_calls": [{"tool": "list_files", "args": {}}]})


def work_of(store: Store, tmp_path: Path, spec: TaskSpec) -> Work:
    store.create_task(spec)
    claimed = store.claim("w")
    assert claimed is not None
    return Work(
        subtask=claimed,
        context=build_context(claimed, [], None),
        heartbeat=no_heartbeat,
        tools=Toolbox(Workspace(tmp_path, claimed.task_id)),
    )


def test_a_master_takes_many_more_tool_rounds(store: Store, tmp_path: Path) -> None:
    rounds = MAX_TOOL_ROUNDS + 6
    llm = FakeLLM().on("Execute", *([list_files()] * rounds), '{"text": "done"}')
    work = work_of(store, tmp_path, master_plan("Do it all"))
    outcome = llm_executor(llm, format_final=False)(work)
    assert isinstance(outcome, Success), outcome
    assert len(llm.calls) == rounds + 1
    assert MASTER_INSTRUCTIONS.format(rounds=MASTER_TOOL_ROUNDS)[:60] in (
        llm.calls[0].messages[0].content
    )


def test_a_master_still_has_a_limit(store: Store, tmp_path: Path) -> None:
    llm = FakeLLM().on("Execute", list_files(), repeat_last=True)
    work = work_of(store, tmp_path, master_plan("Never ends"))
    outcome = llm_executor(llm, format_final=False)(work)
    assert isinstance(outcome, Failure) and f"after {MASTER_TOOL_ROUNDS} rounds" in outcome.reason
    assert len(llm.calls) == MASTER_TOOL_ROUNDS + 1


def test_an_ordinary_subtask_keeps_its_rounds_and_instructions(
    store: Store, tmp_path: Path
) -> None:
    llm = FakeLLM().on("Execute", list_files(), repeat_last=True)
    spec = TaskSpec(goal="g", subtasks=[SubtaskSpec(description="d", acceptance_criteria="c")])
    outcome = llm_executor(llm, format_final=False)(work_of(store, tmp_path, spec))
    assert isinstance(outcome, Failure) and len(llm.calls) == MAX_TOOL_ROUNDS + 1
    assert "plan your own steps" not in llm.calls[0].messages[0].content


def test_a_master_subtask_without_a_qualified_model_waits_for_the_person(
    store: Store, tmp_path: Path
) -> None:
    config = make_config(api_keys={"GROQ_API_KEY": GROQ})
    models = runs.models(config, store.conn)
    assert models.master is None
    outcome = runs._executor(models, store)(work_of(store, tmp_path, master_plan("g")))
    assert isinstance(outcome, Failure) and outcome.kind is FailureKind.PERMANENT
    assert outcome.reason == runs.NO_MASTER


def test_the_master_models_serve_a_master_subtask(store: Store) -> None:
    main, other, master = FakeLLM(), FakeLLM(), FakeLLM()
    models = runs.Models(main, other, master=(master, None))
    assert models.executor(Kind.TEXT, master=True) == (master, None)
    assert models.executor(Kind.TEXT) == (main, other)


# ── The command ──────────────────────────────────────────────────────────────


def script(tmp_path: Path, execute: list[Any]) -> Path:
    path = tmp_path / "script.json"
    routes = [
        {"match": "Execute", "steps": execute, "repeat_last": False},
        {"match": "Review", "steps": ['{"verdict": "accept"}'], "repeat_last": True},
    ]
    path.write_text(json.dumps({"routes": routes}), encoding="utf-8")
    return path


def desman(*args: str | Path) -> Result:
    return CliRunner().invoke(cli, [str(a) for a in args])


def test_run_master_gives_the_goal_whole_to_one_subtask(tmp_path: Path) -> None:
    fake = script(tmp_path, [list_files(), '{"text": "All done"}'])
    result = desman("run", "--master", "Build and test the parser", "--fake-llm", fake)
    assert result.exit_code == EXIT_DONE, result.output
    s = Store(db.open_db())
    try:
        [row] = s.conn.execute("SELECT id FROM tasks").fetchall()
        [sub] = s.list_subtasks(row[0])
        assert sub.master and sub.description == "Build and test the parser"
        [created] = [e for e in history(s.conn, row[0]) if e.action == "task_created"]
        assert created.details["plan"] == "master"
        assert check_history(s.conn, row[0]) == []
    finally:
        s.conn.close()


def test_run_master_refuses_a_spec_or_a_plan_mode(tmp_path: Path) -> None:
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps({"goal": "g", "subtasks": [{"description": "d", "acceptance_criteria": "c"}]})
    )
    assert desman("run", "--master", "--spec", spec).exit_code == 2
    assert desman("run", "--master", "g", "--plan", "llm").exit_code == 2


def test_run_master_without_a_qualified_model_creates_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from desman.config import open_config

    manager = open_config()
    manager.save(make_config(api_keys={"GROQ_API_KEY": GROQ}))
    result = desman("run", "--master", "A goal")
    assert result.exit_code == 1 and "master module" in result.output
    conn = db.open_db()
    try:
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0
    finally:
        conn.close()


def test_a_config_before_masters_upgrades_with_no_master_chosen(tmp_path: Path) -> None:
    from desman.config import SCHEMA_VERSION, upgrade

    upgraded = upgrade({"schema_version": 10, "tuning": {"models": {"text": "groq/x"}}})
    assert upgraded["schema_version"] == SCHEMA_VERSION == 12
    config = make_config(tuning=upgraded["tuning"])
    assert config.tuning.models.master is None and config.tuning.keys.master is None


# ── Review findings (M.1) ────────────────────────────────────────────────────

MISTRAL = "m" * 32
ANTHROPIC = "sk-ant-" + "z" * 90


def test_the_master_follows_its_own_key_and_never_the_analysis_choices() -> None:
    from desman.llm.openai_compat import key_id

    keys = {"NVIDIA_API_KEY": NVIDIA, "MISTRAL_API_KEY": MISTRAL}
    natural = labels(router.lineup(make_config(api_keys=keys), Role.EXECUTOR, kind=catalog.MASTER))
    assert natural == ["mistral/mistral-medium-latest", ULTRA]
    # The analyst's choice for analysis work does not decide who the master asks first.
    analysis = make_config(api_keys=keys, tuning={"models": {"analysis": ULTRA}})
    assert labels(router.lineup(analysis, Role.EXECUTOR, kind=catalog.MASTER)) == natural
    # The master's own key does: its provider goes first.
    by_key = make_config(api_keys=keys, tuning={"keys": {"master": key_id(NVIDIA)}})
    assert labels(router.lineup(by_key, Role.EXECUTOR, kind=catalog.MASTER))[0] == ULTRA


def test_a_paid_master_only_with_its_provider_named() -> None:
    keys = {"NVIDIA_API_KEY": NVIDIA, "ANTHROPIC_API_KEY": ANTHROPIC}
    chosen = make_config(api_keys=keys, tuning={"models": {"master": "anthropic/claude-haiku-5-5"}})
    assert all(
        not c.provider.paid for c in router.lineup(chosen, Role.EXECUTOR, kind=catalog.MASTER)
    )
    named = router.lineup(chosen, Role.EXECUTOR, "anthropic", kind=catalog.MASTER)
    assert labels(named) == ["anthropic/claude-haiku-5-5"]


def test_a_master_named_while_the_run_goes_on_serves_the_next_attempt(tmp_path: Path) -> None:
    from desman.config import Tuning

    conn = db.open_db(tmp_path / "d.db")
    try:
        config = make_config(
            api_keys={"GROQ_API_KEY": GROQ},
            settings={"LLAMA_CPP_MODEL": "big", "LLAMA_CPP_CONTEXT": "65536"},
        )
        tuning = [config.tuning]
        models = runs.models(config, conn, read_tuning=lambda: tuning[0])
        assert models.master is None and models.current_master() is None
        tuning[0] = Tuning.model_validate({"models": {"master": "llamacpp/big"}})
        assert models.current_master() is not None
        found = models.find_master
        assert found is not None and found().choice is not None
        assert found().choice.label == "llamacpp/big"  # type: ignore[union-attr]
    finally:
        conn.close()


def test_combining_the_parts_of_a_split_master_is_ordinary_work(
    store: Store, tmp_path: Path
) -> None:
    from desman.core.retry import Strategy

    llm = FakeLLM().on("Execute", list_files(), repeat_last=True)
    work = work_of(store, tmp_path, master_plan("Big"))
    combine = work.model_copy(
        update={"context": build_context(work.subtask, [], None, strategy=Strategy.COMBINE)}
    )
    outcome = llm_executor(llm, format_final=False)(combine)
    assert isinstance(outcome, Failure) and len(llm.calls) == MAX_TOOL_ROUNDS + 1
    assert "plan your own steps" not in llm.calls[0].messages[0].content


def test_the_modules_view_names_a_master_subtask_by_its_role(store: Store) -> None:
    from desman.core.report import task_report

    task_id = store.create_task(master_plan("g"))
    assert [s.master for s in task_report(store, task_id).subtasks] == [True]


def test_tuning_warns_that_an_unqualified_master_is_never_asked() -> None:
    from desman.config import open_config

    open_config().save(make_config(api_keys={"GROQ_API_KEY": GROQ}))
    result = desman("tuning", "--model", "master=groq/openai/gpt-oss-120b")
    assert result.exit_code == 0, result.output
    assert "cannot be the master module" in result.output
    assert "nvidia/nvidia/nemotron-3-ultra-550b-a55b" in result.output


def test_the_master_row_offers_only_models_that_can_be_master() -> None:
    from desman import tuning_cli

    config = make_config(api_keys={"GROQ_API_KEY": GROQ, "NVIDIA_API_KEY": NVIDIA})
    [row] = [r for r in tuning_cli.role_rows(config) if r["role"] == "master"]
    offered = [m for p in row["providers"] for m in p["models"]]
    assert ULTRA in offered and not any(m.startswith("groq/") for m in offered)


def test_the_no_master_message_tells_an_ollama_user_about_its_context() -> None:
    config = make_config(settings={"OLLAMA_MODEL": "qwen3:32b"})
    with pytest.raises(router.NoProvider, match="OLLAMA_CONTEXT"):
        router.lineup(config, Role.EXECUTOR, kind=catalog.MASTER)
