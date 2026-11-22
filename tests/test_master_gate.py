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
from desman import bench, commands, improve, local_models, providers, runs, tuning_cli
from desman.cli import cli
from desman.config import Config, MasterChoices, Tuning, open_config, upgrade
from desman.core import calls, db, operator, proposals
from desman.core.coordinator import Failure, Success
from desman.core.models import Origin, TaskSpec, Tools
from desman.core.retry import FailureKind
from desman.core.states import Actor, SubtaskStatus
from desman.core.store import Store
from desman.llm import catalog, hub, router
from desman.llm.fake import FakeLLM
from desman.llm.openai_compat import CallRecord, key_id
from desman.planner import master_plan
from desman.settings_actions import Limits
from helpers import make_config
from test_db import _db_at_version
from test_master import NVIDIA, ULTRA, list_files, work_of
from test_stats import desman, script, stats_json

CANDIDATE = "nvidia/nvidia/nemotron-3.5-lightning-30b-a3b"
CANDIDATE_ID = CANDIDATE.partition("/")[2]
CANDIDATE_ALIAS = f"NVIDIA_API_KEY/{CANDIDATE_ID}"


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
    assert config.schema_version == 13 and config.configured
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
        master={"admitted": [small.label, capped.label, "llamacpp/big", "llamacpp/old"]},
        tuning={"models": {"master": "llamacpp/old"}},
        settings={"LLAMA_CPP_MODEL": "big", "LLAMA_CPP_CONTEXT": "32768"},
    )
    assert not router.qualified(config, small)
    assert not router.qualified(config, capped)
    assert [c.label for c in router.lineup(config, calls.Role.EXECUTOR, kind=catalog.MASTER)] == [
        "llamacpp/big"
    ]


def test_ollama_may_route_to_another_admitted_model() -> None:
    config = make_config(
        master={"admitted": ["ollama/another"]},
        settings={"OLLAMA_MODEL": "default", "OLLAMA_CONTEXT": "32768"},
    )
    assert [c.label for c in router.lineup(config, calls.Role.EXECUTOR, kind=catalog.MASTER)] == [
        "ollama/another"
    ]


def test_master_tuning_lists_local_models_and_warns_about_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        local_models, "installed", lambda *_args, **_kw: ([hub.InstalledModel(name="other")], True)
    )
    config = make_config(settings={"OLLAMA_MODEL": "current", "LLAMA_CPP_MODEL": "served"})
    open_config().save(config)
    [row] = [row for row in tuning_cli.role_rows(config) if row["role"] == "master"]
    assert {"ollama/current", "ollama/other", "llamacpp/served"} <= set(row["candidate_models"])
    result = desman("tuning", "--model", "master=llamacpp/served")
    assert result.exit_code == 0, result.output
    assert "context of at least 32768" in result.output
    result = desman("tuning", "--model", "master=missing/model")
    assert result.exit_code == 0, result.output
    assert "cannot be the master module" in result.output

    # The warning should also apply to a local provider with a different ID.
    find = providers.find
    monkeypatch.setattr(
        providers, "find", lambda name: find("llamacpp") if name == "localvariant" else find(name)
    )
    result = desman("tuning", "--model", "master=localvariant/served")
    assert result.exit_code == 0, result.output
    assert "context of at least 32768" in result.output


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
    reviewer_key = NVIDIA.replace("WxYz", "wXyZ")
    config = make_config(
        api_keys={"NVIDIA_API_KEY": [{"value": NVIDIA}, {"value": reviewer_key}]},
        tuning={
            "max_attempts": 2,
            "max_transient": 3,
            "context_tokens": 1024,
            "max_subtasks": 4,
            "max_parts": 3,
            "master_rounds": 30,
            "models": {"reviewer": ULTRA},
            "keys": {"reviewer": key_id(reviewer_key)},
        },
        master={"revoked": [CANDIDATE]},
    )
    open_config().save(config)
    used: list[tuple[str, str]] = []
    available = True
    passing = True
    limited = False
    failure_reason = f"cannot do this {NVIDIA} " + "because " * 40

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": CANDIDATE_ID}] if available else []})
        data = json.loads(request.content)
        purpose = calls.purpose().role
        assert purpose in (calls.Role.EXECUTOR, calls.Role.REVIEWER)
        used.append((purpose.value, data["model"]))
        if limited and purpose is calls.Role.EXECUTOR:
            return httpx.Response(429, headers={"retry-after": "3600"}, json={})
        if purpose is calls.Role.REVIEWER:
            assert data["model"] == ULTRA.partition("/")[2]
            assert request.headers["Authorization"] == f"Bearer {reviewer_key}"
            message = '{"verdict":"accept"}'
        elif passing:
            message = '{"text":"Complete."}'
        else:
            message = json.dumps({"type": "error", "reason": failure_reason})
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

    # An older benchmark version and a previous set of defaults both expire a passing check.
    with monkeypatch.context() as old:
        old.setattr(bench, "VERSION", bench.VERSION - 1)
        result = command("check", CANDIDATE, "--json")
        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout)["passed"] is True
        assert "spends" in result.stderr
    assert command("admit", CANDIDATE).exit_code == 1
    assert {model for role, model in used if role == "executor"} == {CANDIDATE_ID}
    with monkeypatch.context() as old:
        old.setattr(runs, "TUNING_DEFAULTS", {**runs.TUNING_DEFAULTS, "max_attempts": 3})
        assert json.loads(command("check", CANDIDATE, "--json").stdout)["passed"]
    assert command("admit", CANDIDATE).exit_code == 1

    result = command("check", CANDIDATE_ALIAS, "--json")
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["model"] == CANDIDATE
    assert data["done"] == len(bench.CASES)
    assert data["outcome"] == "passed"
    assert data["result"]["limits"] == runs.TUNING_DEFAULTS
    assert all(case["reason"] is None for case in data["result"]["cases"])
    assert data["usage"]["calls"] == 2 * len(bench.CASES)
    assert json.loads(command("admit", CANDIDATE_ALIAS, "--json").stdout) == {
        "model": CANDIDATE,
        "admitted": True,
        "revoked": False,
        "changed": True,
    }
    assert "already admitted" in command("admit", CANDIDATE).output
    assert json.loads(command("admit", CANDIDATE, "--json").stdout)["changed"] is False
    assert open_config().load().master == MasterChoices(admitted=[CANDIDATE])

    passing = False
    result = command("check", CANDIDATE, "--json")
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["passed"] is False and data["outcome"] == "failed"
    for case in data["result"]["cases"]:
        assert case["status"] == "needs_human"
        assert "cannot do this" in case["reason"]
        assert len(case["reason"]) <= 200 and NVIDIA not in case["reason"]
        assert "[key removed]" in case["reason"]
    assert "already admitted" in command("admit", CANDIDATE_ALIAS).output
    assert json.loads(command("revoke", CANDIDATE_ALIAS, "--json").stdout) == {
        "model": CANDIDATE,
        "admitted": False,
        "revoked": True,
        "changed": True,
    }
    assert "already revoked" in command("revoke", CANDIDATE).output
    assert json.loads(command("revoke", CANDIDATE, "--json").stdout)["changed"] is False
    assert open_config().load().master == MasterChoices(revoked=[CANDIDATE])

    limited = True
    result = command("check", CANDIDATE, "--json")
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["outcome"] == "unfinished" and not data["passed"]
    assert all(case["status"] == "active" and case["reason"] for case in data["result"]["cases"])
    assert command("admit", CANDIDATE).exit_code == 1
    result = command("check", CANDIDATE)
    assert result.exit_code == 0, result.output
    assert "Try again later" in result.output
    assert "Reason" in result.output

    listed = command("--json")
    assert listed.exit_code == 0, listed.output
    [row] = [m for m in json.loads(listed.stdout)["models"] if m["model"] == CANDIDATE]
    assert row["revoked"] and row["last_check"]["outcome"] == "unfinished"
    assert "Unfinished" in command("list").output
    with closing(db.open_db()) as conn:
        assert conn.execute("SELECT COUNT(*) FROM master_checks").fetchone()[0] == 6
        logged = conn.execute(
            "SELECT role, model, task_id FROM model_calls ORDER BY seq"
        ).fetchall()
        assert [(row["role"], row["model"]) for row in logged] == used
        assert all(row["task_id"] is None for row in logged)
        assert conn.execute("SELECT 1 FROM rate_buckets WHERE blocked_until IS NOT NULL").fetchone()
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0
        assert [(e.action, e.details) for e in operator.entries(conn)] == [
            ("operator_master_admit", {"model": CANDIDATE}),
            ("operator_master_revoke", {"model": CANDIDATE}),
        ]


@pytest.mark.parametrize(
    ("review_rejects", "zero_limit"), [(False, True), (True, True), (True, False)]
)
def test_check_provider_stops_and_reviewer_rejection(
    monkeypatch: pytest.MonkeyPatch, clock: FakeClock, review_rejects: bool, zero_limit: bool
) -> None:
    open_config().save(make_config(api_keys={"NVIDIA_API_KEY": NVIDIA}))
    first_case_calls = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal first_case_calls
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": CANDIDATE_ID}]})
        if review_rejects and first_case_calls < 2:
            first_case_calls += 1
            content = (
                '{"text":"Incomplete."}'
                if calls.purpose().role == calls.Role.EXECUTOR
                else '{"verdict":"escalate","feedback":"Incomplete work"}'
            )
            return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})
        if not zero_limit:
            return httpx.Response(429, headers={"retry-after": "3600"}, json={})
        return httpx.Response(
            429,
            headers={"x-ratelimit-limit-requests": "0"},
            json={"error": {"message": "limit is 0"}},
        )

    transport = httpx.MockTransport(respond)
    http_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **_kw: http_client(transport=transport))
    build_models = runs.models
    monkeypatch.setattr(
        runs,
        "models",
        lambda *args, **kwargs: build_models(*args, clock=clock, sleep=clock.sleep, **kwargs),
    )

    result = desman("master", "check", CANDIDATE, "--json")
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["outcome"] == ("failed" if review_rejects else "unfinished")
    cases = data["result"]["cases"]
    assert len(cases) == 4
    if review_rejects:
        assert cases[0]["status"] == "needs_human"
        assert cases[0]["reason"] == "Incomplete work"
    stopped = cases[int(review_rejects) :]
    if zero_limit:
        assert all(case["status"] == "needs_human" for case in stopped)
        assert all("ProviderRejected" in case["reason"] for case in stopped)
    else:
        assert all(case["status"] == "active" for case in stopped)

    if not review_rejects:
        shown = desman("master", "check", CANDIDATE)
        assert shown.exit_code == 0, shown.output
        assert "Unfinished" in shown.output
        assert "The provider refused or limited requests" in shown.output
        assert "Did not pass" not in shown.output


def test_chat_already_admitted_without_a_check(
    store: Store, capsys: pytest.CaptureFixture[str]
) -> None:
    manager = open_config()
    manager.save(make_config(master={"admitted": [CANDIDATE]}))
    parsed = commands.parse_line(f"master admit {CANDIDATE_ALIAS}", manager.load())
    assert parsed is not None
    entry, args = parsed
    ctx = commands.Context(store, manager, lambda *_: None, lambda _: False)
    outcome = entry.handler(ctx, args)
    assert isinstance(outcome, commands.Outcome)
    assert f"{CANDIDATE} is already admitted." in capsys.readouterr().out
    assert manager.load().master.admitted == [CANDIDATE]


@pytest.mark.parametrize("command", ["check", "admit", "revoke"])
def test_master_commands_list_known_providers(command: str) -> None:
    result = desman("master", command, "unknown/model")
    assert result.exit_code == 1
    assert "unknown provider 'unknown' (known:" in result.output
    assert "llamacpp" in result.output


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
        if name in {"admit", "revoke"}:
            with pytest.raises(commands.BadArgs, match="--json is not available in the chat"):
                commands.parse_line(f"master {name}{args} --json", Config())
            with pytest.raises(commands.BadArgs, match="as_json"):
                commands.parse(entry, {"model": CANDIDATE, "as_json": True}, Config())


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
        spec = TaskSpec.model_validate(
            {
                "goal": "Housekeeping",
                "subtasks": [{"description": "Internal work", "acceptance_criteria": "Done"}],
            }
        )
        for details in (
            {"summarises": "session", "covers": 42},
            {"purpose": proposals.PURPOSE},
            {"purpose": "other_internal_purpose"},
            {"benchmark": "explain", "plan": "llm"},
        ):
            store.create_task(spec, details=details)
        store.create_task(spec, origin=Origin.NETWORK)
        store.create_task(master_plan("Internal master"), details={"benchmark": "compare"})
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
            "INSERT INTO master_checks (model, version, outcome, result_json) "
            "VALUES (?, 1, 'passed', '{}')",
            (CANDIDATE,),
        )
        with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
            conn.execute(
                "INSERT INTO master_checks (model, version, outcome, result_json) "
                "VALUES (?, 1, 'invalid', '{}')",
                (CANDIDATE,),
            )
        for sql in ("UPDATE master_checks SET outcome = 'failed'", "DELETE FROM master_checks"):
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                conn.execute(sql)
