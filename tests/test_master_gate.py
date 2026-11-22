"""M.3 measured admissions, live tuning, and task-mode statistics."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import httpx
import pytest
from click.testing import CliRunner, Result
from pydantic import ValidationError

from conftest import FakeClock
from desman import bench, commands, improve, runs, tuning_cli
from desman.cli import cli
from desman.config import Config, MasterChoices, Tuning, open_config, upgrade
from desman.core import calls, db, operator
from desman.core.coordinator import Failure, Success
from desman.core.models import TaskSpec, Tools
from desman.core.retry import FailureKind
from desman.core.states import Actor, SubtaskStatus
from desman.core.store import Store
from desman.llm import catalog, router
from desman.llm.fake import FakeLLM
from desman.llm.openai_compat import CallRecord
from desman.planner import master_plan
from desman.settings_actions import Limits
from helpers import make_config
from test_db import _db_at_version
from test_master import NVIDIA, ULTRA, list_files, work_of
from test_stats import desman, script, stats_json

CANDIDATE = "nvidia/nvidia/nemotron-3.5-lightning-30b-a3b"
CANDIDATE_ID = CANDIDATE.partition("/")[2]


@pytest.mark.parametrize("rounds", [7, 49, True, "24"])
def test_master_round_bounds(rounds: object) -> None:
    with pytest.raises(ValidationError):
        Tuning.model_validate({"master_rounds": rounds})


def test_master_rounds_are_live_and_person_only(store: Store, tmp_path: Path) -> None:
    rounds = 26
    tuning = Tuning()
    work = work_of(store, tmp_path, master_plan("Do it all"))

    def master() -> FakeLLM:
        return FakeLLM().on("Execute", *([list_files()] * rounds), '{"text": "Done."}')

    llm = master()
    models = runs.Models(llm, None, find_master=lambda: runs.Master((llm, None)))
    executor = runs._executor(models, store, lambda: tuning, master_check=False)
    outcome = executor(work)
    assert isinstance(outcome, Failure) and "after 24 rounds" in outcome.reason

    result = desman("tuning", "--master-rounds", str(rounds), "--json")
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["master_rounds"] == rounds
    tuning = open_config().load().tuning
    llm = master()
    work = work_of(store, tmp_path, master_plan("A second goal"))
    assert isinstance(executor(work), Success)
    assert "tuning.master_rounds" not in improve.SETTINGS_BY_ID
    parsed = commands.parse_line("tuning --master-rounds 26", Config())
    assert parsed and parsed[1].model_dump()["master_rounds"] == rounds
    assert Limits(section="limits", name="master_rounds", value=rounds).value == rounds


def test_config_upgrade_from_11_and_label_validation() -> None:
    config = Config.model_validate(upgrade({"schema_version": 11, "configured": True}))
    assert config.schema_version == 12 and config.configured
    assert config.master == MasterChoices() and config.tuning.master_rounds is None
    assert MasterChoices(admitted=[CANDIDATE, CANDIDATE]).admitted == [CANDIDATE]
    with pytest.raises(ValidationError):
        MasterChoices(revoked=["missing-model", "provider/bad label"])


def test_live_admission_replaces_the_marked_model(store: Store) -> None:
    config = make_config(api_keys={"NVIDIA_API_KEY": NVIDIA})
    current = config.master
    models = runs.models(config, store.conn, read_master=lambda: current)
    assert models.master_status().choice == router.master_candidate(config, ULTRA)
    current = MasterChoices(admitted=[CANDIDATE], revoked=[ULTRA])
    assert models.master_status().choice == router.master_candidate(config, CANDIDATE)
    updated = config.model_copy(update={"master": current})
    [row] = [r for r in tuning_cli.role_rows(updated) if r["role"] == "master"]
    assert CANDIDATE in row["candidate_models"]
    assert ULTRA not in row["candidate_models"] and row["revoked_models"] == [ULTRA]
    current = MasterChoices(revoked=[ULTRA])
    assert models.current_master() is None
    assert not router.may_be_master(ULTRA, config.model_copy(update={"master": current}))


def test_limits_apply_to_admitted_models_and_admitted_local_needs_no_preference() -> None:
    from desman.providers import find

    small = router.Choice(find("cloudflare"), "@cf/meta/llama-3.1-8b-instruct")
    capped = router.Choice(find("groq"), "openai/gpt-oss-20b")
    config = make_config(
        master={"admitted": [small.label, capped.label, "llamacpp/big"]},
        settings={"LLAMA_CPP_MODEL": "big", "LLAMA_CPP_CONTEXT": "32768"},
    )
    assert not router.qualified(config, small)
    assert not router.qualified(config, capped)
    assert [c.label for c in router.lineup(config, calls.Role.EXECUTOR, kind=catalog.MASTER)] == [
        "llamacpp/big"
    ]


@pytest.mark.parametrize("passed", [True, False])
def test_bench_uses_one_master_and_no_planner(
    store: Store, monkeypatch: pytest.MonkeyPatch, passed: bool
) -> None:
    reply = '{"text":"Complete."}' if passed else '{"type":"error","reason":"cannot do this"}'
    fake = (
        FakeLLM()
        .on("Execute", reply, repeat_last=True)
        .on("Review", '{"verdict":"accept"}', repeat_last=True)
    )
    monkeypatch.setattr(bench, "fake", lambda: fake)
    result = bench.run(store, "fake", Tuning(), master=CANDIDATE)
    assert bench.master_passed(result) is passed
    assert result.version == bench.VERSION
    assert result.limits and result.limits["master_rounds"] == 24
    assert bench.metrics(result)["plan_fallbacks"] == 0
    assert all(case.plan == "master" and case.subtasks == 1 for case in result.cases)
    assert {role["role"] for role in result.stats["roles"]} <= {"executor", "reviewer"}
    for row in store.conn.execute("SELECT id, tools FROM tasks"):
        assert row["tools"] == Tools.NONE.value
        [subtask] = store.list_subtasks(row["id"])
        assert subtask.master


def test_cli_check_admit_revoke_and_latest_version(
    monkeypatch: pytest.MonkeyPatch,
    clock: FakeClock,
) -> None:
    config = make_config(
        api_keys={"NVIDIA_API_KEY": NVIDIA},
        tuning={"max_attempts": 2},
        master={"revoked": [CANDIDATE]},
    )
    open_config().save(config)
    used: list[tuple[str, str]] = []
    available = True
    passing = True

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": CANDIDATE_ID}] if available else []})
        data = json.loads(request.content)
        purpose = calls.purpose().role
        assert purpose in (calls.Role.EXECUTOR, calls.Role.REVIEWER)
        used.append((purpose.value, data["model"]))
        if purpose is calls.Role.REVIEWER:
            message = '{"verdict":"accept"}'
        elif passing:
            message = '{"text":"Complete."}'
        else:
            message = '{"type":"error","reason":"cannot do this"}'
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": message}}],
                "usage": {"prompt_tokens": 25, "completion_tokens": 5},
            },
        )

    transport = httpx.MockTransport(respond)
    build_models = runs.models
    monkeypatch.setattr(
        runs,
        "models",
        lambda *args, **kwargs: build_models(*args, clock=clock, sleep=clock.sleep, **kwargs),
    )
    http_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **_kw: http_client(transport=transport))

    def command(*args: str) -> Result:
        return CliRunner().invoke(cli, ["master", *args])

    result = command("admit", CANDIDATE)
    assert result.exit_code == 1 and "master check" in result.output

    available = False
    result = command("check", CANDIDATE)
    assert result.exit_code == 1 and "does not list" in result.output
    available = True

    # Mark the first completed run with yesterday's benchmark version.
    with monkeypatch.context() as old:
        old.setattr(bench, "VERSION", bench.VERSION - 1)
        result = command("check", CANDIDATE, "--json")
        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout)["passed"] is True
        assert "spends" in result.stderr
    assert command("admit", CANDIDATE).exit_code == 1
    assert {model for role, model in used if role == "executor"} == {CANDIDATE_ID}

    result = command("check", CANDIDATE, "--json")
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["done"] == len(bench.CASES)
    assert command("admit", CANDIDATE).exit_code == 0
    assert open_config().load().master == MasterChoices(admitted=[CANDIDATE])

    passing = False
    result = command("check", CANDIDATE, "--json")
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["passed"] is False
    assert command("admit", CANDIDATE).exit_code == 1
    assert command("revoke", CANDIDATE).exit_code == 0
    assert open_config().load().master == MasterChoices(revoked=[CANDIDATE])

    listed = command("--json")
    assert listed.exit_code == 0, listed.output
    [row] = [m for m in json.loads(listed.stdout)["models"] if m["model"] == CANDIDATE]
    assert row["revoked"] and not row["last_check"]["passed"]
    with closing(db.open_db()) as conn:
        assert conn.execute("SELECT COUNT(*) FROM master_checks").fetchone()[0] == 3
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0
        assert [(e.action, e.details) for e in operator.entries(conn)] == [
            ("operator_master_admit", {"model": CANDIDATE}),
            ("operator_master_revoke", {"model": CANDIDATE}),
        ]


def test_catalog_routes_checks_and_decisions_through_confirmation() -> None:
    for name, kind in (
        ("list", commands.Kind.READ),
        ("check", commands.Kind.CHANGE),
        ("admit", commands.Kind.CHANGE),
        ("revoke", commands.Kind.CHANGE),
    ):
        args = "" if name == "list" else f" {CANDIDATE}"
        parsed = commands.parse_line(f"master {name}{args}", Config())
        assert parsed is not None
        entry, given = parsed
        assert entry.kind_of(given) is kind


def test_mode_stats_include_calls_and_past_escalation(tmp_path: Path) -> None:
    ordinary = script(tmp_path, "cluster", ['{"text":"Complete."}'], ['{"verdict":"accept"}'])
    assert desman("run", "Cluster", "--fake-llm", ordinary).exit_code == 0
    assert "Master vs cluster" not in desman("stats").output
    with closing(db.open_db()) as conn:
        store = Store(conn)
        master = store.create_task(master_plan("Master"))
        subtask = store.claim("worker")
        assert subtask and subtask.lease_token
        store.record_failure(
            subtask.id,
            SubtaskStatus.RUNNING,
            Actor.EXECUTOR,
            FailureKind.PERMANENT,
            lease=subtask.lease_token,
            reason="test",
        )
        record = calls.recorder(conn)
        with calls.for_(calls.Purpose(calls.Role.EXECUTOR, task_id=master)):
            record(CallRecord("p", "m", None, "p", None, 7, 4, 0, 429, "limit", None))
    assert desman("accept", subtask.id).exit_code == 0
    data = stats_json()["modes"]
    assert data["master"] == {
        "tasks": 1,
        "done": 1,
        "escalated": 1,
        "calls": 1,
        "failed_calls": 1,
        "prompt_tokens": 7,
        "completion_tokens": 4,
        "subtasks": 1,
    }
    assert data["cluster"]["tasks"] == data["cluster"]["done"] == 1
    assert "Master vs cluster" in desman("stats").output


def test_migration_preserves_existing_tasks_and_guards_checks(tmp_path: Path) -> None:
    path = tmp_path / "v51.db"
    with closing(_db_at_version(path, 51)) as conn:
        task_id = Store(conn).create_task(
            TaskSpec.model_validate(
                {
                    "goal": "Old goal",
                    "subtasks": [{"description": "Old work", "acceptance_criteria": "Complete"}],
                }
            )
        )
    with closing(db.open_db(path)) as conn:
        assert not Store(conn).list_subtasks(task_id)[0].master
        conn.execute(
            "INSERT INTO master_checks (model, version, passed, result_json) VALUES (?, 1, 1, '{}')",
            (CANDIDATE,),
        )
        for sql in ("UPDATE master_checks SET passed = 0", "DELETE FROM master_checks"):
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                conn.execute(sql)
