"""Tests for the task commands: run, resume, status, list (ROADMAP 1.10)."""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner, Result

from desman.cli import EXIT_ACTIVE, EXIT_DONE, EXIT_NEEDS_HUMAN, EXIT_PAUSED, EXIT_STOPPED, cli
from desman.core import db
from desman.core.audit import check_history, history
from desman.core.models import SubtaskSpec, TaskSpec
from desman.core.states import TaskStatus
from desman.core.store import Store

ACCEPT = '{"verdict": "accept"}'


def script(tmp_path: Path, execute: list[Any], review: list[Any] | None = None) -> Path:
    path = tmp_path / f"script-{len(list(tmp_path.glob('script-*')))}.json"
    routes = [
        {"match": "Execute", "steps": execute, "repeat_last": False},
        {"match": "Review", "steps": review or [ACCEPT], "repeat_last": True},
    ]
    path.write_text(json.dumps({"routes": routes}))
    return path


def desman(*args: str | Path) -> Result:
    return CliRunner().invoke(cli, [str(a) for a in args])


def store() -> Store:
    return Store(db.open_db())


def only_task() -> str:
    conn = db.open_db()
    try:
        [row] = conn.execute("SELECT id FROM tasks").fetchall()
        task_id: str = row[0]
        return task_id
    finally:
        conn.close()


# ── run ──────────────────────────────────────────────────────────────────────


def test_run_goal_end_to_end(tmp_path: Path) -> None:
    fake = script(tmp_path, ['{"haiku": "WAL mode hums"}'])
    result = desman("run", "Write a haiku", "--fake-llm", fake)
    assert result.exit_code == EXIT_DONE, result.output
    task_id = only_task()
    lines = result.output.splitlines()
    assert lines[0] == f"Task {task_id} created (1 subtask)"
    assert lines[1] == f"Task {task_id}  done  (1/1 done)"
    s = store()
    [sub] = s.list_subtasks(task_id)
    assert sub.description == "Write a haiku"
    assert s.latest_artifact(sub.id) == {"haiku": "WAL mode hums"}
    assert check_history(s.conn, task_id) == []
    [claim] = [e for e in history(s.conn, task_id) if e.action == "claim"]
    assert claim.details["owner"] == f"{socket.gethostname()}:{os.getpid()}"


def test_run_json(tmp_path: Path) -> None:
    fake = script(tmp_path, ['{"x": 1}'])
    result = desman("run", "g", "--fake-llm", fake, "--json")
    assert result.exit_code == EXIT_DONE, result.output
    out = json.loads(result.output)  # nothing else on stdout
    assert out["run"]["status"] == "done" and out["run"]["paused_until"] is None
    assert out["task"]["status"] == "done"
    [sub] = out["task"]["subtasks"]
    assert sub["status"] == "done" and "lease_token" not in sub


def test_run_spec_with_dependencies(tmp_path: Path) -> None:
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "goal": "report",
                "priority": 100,
                "subtasks": [
                    {"description": "collect", "acceptance_criteria": "ok"},
                    {"description": "write", "acceptance_criteria": "ok", "depends_on": [0]},
                ],
            }
        )
    )
    fake = script(tmp_path, ['{"data": [1, 2]}', '{"text": "done"}'])
    result = desman("run", "--spec", spec, "--fake-llm", fake, "--priority", "critical")
    assert result.exit_code == EXIT_DONE, result.output
    assert "created (2 subtasks)" in result.output
    assert "[1] done         write  (after 0)" in result.output
    s = store()
    row = s.find_task(only_task())
    assert row["priority"] == 200  # --priority wins over the spec


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ([], "either GOAL or --spec"),
        (["g", "--spec", "SPEC"], "either GOAL or --spec"),
        (["   "], "GOAL is empty"),
        (["--spec", "BAD"], "not a valid task spec"),
        (["g", "--priority", "urgent"], "Invalid value"),
    ],
)
def test_run_refuses_bad_input_without_creating_anything(
    tmp_path: Path, args: list[str], message: str
) -> None:
    good = tmp_path / "spec.json"
    good.write_text(
        json.dumps({"goal": "g", "subtasks": [{"description": "d", "acceptance_criteria": "c"}]})
    )
    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps(
            {
                "goal": "g",
                "subtasks": [{"description": "d", "acceptance_criteria": "c", "depends_on": [0]}],
            }
        )
    )
    resolved = [str(good) if a == "SPEC" else str(bad) if a == "BAD" else a for a in args]
    result = desman("run", *resolved, "--fake-llm", script(tmp_path, ["{}"]))
    assert result.exit_code != 0
    assert message in result.output
    assert not db.default_db_path().exists()


def test_run_with_undecodable_spec(tmp_path: Path) -> None:
    spec = tmp_path / "spec.json"
    spec.write_bytes(b"\xff\xfe{}")
    result = desman("run", "--spec", spec, "--fake-llm", script(tmp_path, ["{}"]))
    assert result.exit_code == 1 and "not a valid task spec" in result.output


def test_run_without_provider_fails_fast() -> None:
    result = desman("run", "g")
    assert result.exit_code == 1
    assert "no LLM provider yet" in result.output and "--fake-llm" in result.output
    assert not db.default_db_path().exists()


def test_run_with_invalid_script(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text('{"steps": [{"type": "explode"}]}')
    result = desman("run", "g", "--fake-llm", bad)
    assert result.exit_code == 1 and "not a valid fake LLM script" in result.output


def test_script_running_out_is_an_error_not_an_outage(tmp_path: Path) -> None:
    fake = script(tmp_path, [])  # no Execute steps
    result = desman("run", "g", "--fake-llm", fake)
    assert result.exit_code == 1
    assert "fake LLM script ran out" in result.output


# ── Outcomes and exit codes ──────────────────────────────────────────────────


def test_rate_limit_returns_paused_then_resume_finishes(tmp_path: Path) -> None:
    fake = script(tmp_path, [{"type": "rate_limit", "retry_after": 3600}])
    result = desman("run", "g", "--fake-llm", fake)
    assert result.exit_code == EXIT_PAUSED, result.output
    task_id = only_task()
    assert f"Continue with: desman resume {task_id[:8]}" in result.output
    status = desman("status", task_id[:8])
    assert status.exit_code == EXIT_PAUSED and "paused until" in status.output

    # The quota has reset (pretend an hour went by).
    conn = db.open_db()
    conn.execute("UPDATE subtasks SET paused_until = '2000-01-01T00:00:00.000Z'")
    conn.close()
    resumed = desman("resume", task_id[:8], "--fake-llm", script(tmp_path, ['{"ok": 1}']))
    assert resumed.exit_code == EXIT_DONE, resumed.output
    assert f"Task {task_id}  done" in resumed.output


def test_rejected_request_waits_for_the_operator(tmp_path: Path) -> None:
    fake = script(tmp_path, [{"type": "rejected", "status": 401, "message": "bad key"}])
    result = desman("run", "g", "--fake-llm", fake)
    assert result.exit_code == EXIT_NEEDS_HUMAN, result.output
    assert "needs_human" in result.output and "bad key" in result.output
    assert desman("status", only_task()).exit_code == EXIT_NEEDS_HUMAN


def test_garbage_then_valid_reply(tmp_path: Path) -> None:
    fake = script(tmp_path, ["Sure! Here it is.", '{"ok": true}'])
    result = desman("run", "g", "--fake-llm", fake)
    assert result.exit_code == EXIT_DONE, result.output
    assert "attempts q1/t0" in result.output


def test_cancelled_task(tmp_path: Path) -> None:
    fake = script(tmp_path, [{"type": "rejected", "status": 403}])
    desman("run", "g", "--fake-llm", fake)
    task_id = only_task()
    store().cancel_task(task_id, reason="operator")
    assert desman("status", task_id).exit_code == EXIT_STOPPED
    resumed = desman("resume", task_id, "--fake-llm", script(tmp_path, []))
    assert resumed.exit_code == EXIT_STOPPED  # nothing to do, no LLM call made
    assert "cancelled" in resumed.output


def test_resume_and_status_errors(tmp_path: Path) -> None:
    assert "no database" in desman("status", "abcd").output
    desman("run", "g", "--fake-llm", script(tmp_path, ["{}"]))
    assert "not found" in desman("status", "zzzzzz").output
    assert "not found" in desman("resume", "zzzzzz", "--fake-llm", script(tmp_path, [])).output
    assert "no LLM provider" in desman("resume", only_task()).output


# ── status and list ──────────────────────────────────────────────────────────


def test_status_json_and_running_subtask() -> None:
    s = store()
    sub = SubtaskSpec(description="line one\nline two", acceptance_criteria="c")
    task_id = s.create_task(TaskSpec(goal="g", subtasks=[sub]))
    claimed = s.claim("worker-7")
    assert claimed is not None and claimed.lease_token is not None
    text = desman("status", task_id)
    assert text.exit_code == EXIT_ACTIVE  # someone is working on it
    assert "line one line two" in text.output and "held by worker-7" in text.output
    out = desman("status", task_id, "--json")
    report = json.loads(out.output)
    assert report["subtasks"][0]["lease_owner"] == "worker-7"
    assert claimed.lease_token not in out.output


def test_list(tmp_path: Path) -> None:
    assert "no database" in desman("list").output
    desman("run", "first goal", "--fake-llm", script(tmp_path, ["{}"]))
    desman(
        "run", "second goal", "--fake-llm", script(tmp_path, [{"type": "rejected", "status": 401}])
    )
    lines = desman("list").output.splitlines()
    assert len(lines) == 2
    assert "second goal" in lines[0] and "needs_human" in lines[0]  # newest first
    assert "first goal" in lines[1] and "1/1" in lines[1]
    assert len(desman("list", "--limit", "1").output.splitlines()) == 1
    done = json.loads(desman("list", "--status", "done", "--json").output)
    assert [t["goal"] for t in done] == ["first goal"]
    assert "subtasks" not in done[0]
    assert json.loads(desman("list", "--status", "failed", "--json").output) == []


def test_task_commands_never_prompt(tmp_path: Path) -> None:
    # Unconfigured, empty stdin: the wizard belongs to bare `desman` and `desman config` only.
    runner = CliRunner()
    ran = runner.invoke(cli, ["run", "g", "--fake-llm", str(script(tmp_path, ["{}"]))], input="")
    assert ran.exit_code == EXIT_DONE, ran.output
    for args in (["list"], ["status", only_task()], ["audit", only_task()]):
        result = runner.invoke(cli, args, input="")
        assert result.exit_code == EXIT_DONE, (args, result.output)
        assert "language" not in result.output.lower(), args


# ── Review regressions (PR #11) ──────────────────────────────────────────────


def test_active_task_has_its_own_exit_code() -> None:
    s = store()
    task_id = s.create_task(
        TaskSpec(goal="g", subtasks=[SubtaskSpec(description="d", acceptance_criteria="c")])
    )
    result = desman("status", task_id)
    assert result.exit_code == EXIT_ACTIVE != EXIT_DONE
    assert "active" in result.output


def test_blocked_subtask_names_what_blocks_it(tmp_path: Path) -> None:
    from desman.core.retry import FailureKind
    from desman.core.states import Actor, SubtaskStatus

    s = store()
    spec = TaskSpec(
        goal="g",
        subtasks=[
            SubtaskSpec(description="fetch data", acceptance_criteria="c"),
            SubtaskSpec(description="analyse", acceptance_criteria="c", depends_on=[0]),
            SubtaskSpec(description="report", acceptance_criteria="c", depends_on=[1]),
        ],
    )
    task_id = s.create_task(spec)
    claimed = s.claim("w")
    assert claimed is not None and claimed.lease_token is not None
    s.record_failure(
        claimed.id,
        SubtaskStatus.RUNNING,
        Actor.EXECUTOR,
        FailureKind.PERMANENT,
        lease=claimed.lease_token,
        reason="403",
    )
    s.transition(
        claimed.id,
        SubtaskStatus.NEEDS_HUMAN,
        SubtaskStatus.FAILED,
        Actor.HUMAN,
        reason="source is gone",
    )
    report = json.loads(desman("status", task_id, "--json").output)
    reasons = [sub["reason"] for sub in report["subtasks"]]
    assert reasons[0] == "source is gone"
    assert reasons[1] == reasons[2] == "blocked by [0] fetch data (failed)"
    assert "blocked by [0] fetch data (failed)" in desman("status", task_id).output


def test_list_limit_does_not_load_every_task(monkeypatch: pytest.MonkeyPatch) -> None:
    s = store()
    for i in range(30):
        s.create_task(
            TaskSpec(
                goal=f"goal {i}", subtasks=[SubtaskSpec(description="d", acceptance_criteria="c")]
            )
        )
    seen: list[str] = []
    s.conn.set_trace_callback(seen.append)
    from desman.core.report import task_summaries

    newest = task_summaries(s, limit=2)
    assert [t.goal for t in newest] == ["goal 29", "goal 28"]
    full_scans = [q for q in seen if "FROM subtasks" in q and "WHERE" not in q]
    assert full_scans == []
    only_done = task_summaries(s, status=TaskStatus.DONE, limit=5)
    assert only_done == []
