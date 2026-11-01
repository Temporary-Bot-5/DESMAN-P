"""Tests for `desman stats` (ROADMAP 5.2): numbers read from subtasks, audit and call log."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from click.testing import CliRunner, Result

from desman.cli import EXIT_DONE, EXIT_NEEDS_HUMAN, cli
from desman.core import calls, db
from desman.core.stats import collect
from desman.llm.openai_compat import CallRecord

ACCEPT = '{"verdict": "accept"}'
RETRY = '{"verdict": "retry", "feedback": "too short"}'


def desman(*args: str | Path) -> Result:
    return CliRunner().invoke(cli, [str(a) for a in args])


def script(tmp_path: Path, name: str, execute: list[Any], review: list[Any]) -> Path:
    routes = [
        {"match": "Execute", "steps": execute, "repeat_last": True},
        {"match": "Review", "steps": review, "repeat_last": True},
    ]
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps({"routes": routes}))
    return path


def three_tasks(tmp_path: Path) -> None:
    """One done at once, one after the reviewer sent it back, one accepted by the operator
    after a refused key."""
    first = script(tmp_path, "first", ['{"ok": 1}'], [ACCEPT])
    assert desman("run", "at once", "--fake-llm", first).exit_code == EXIT_DONE
    second = script(tmp_path, "second", ['{"ok": 2}'], [RETRY, ACCEPT])
    assert desman("run", "sent back", "--fake-llm", second).exit_code == EXIT_DONE
    third = script(tmp_path, "third", [{"type": "rejected", "status": 401}], [ACCEPT])
    assert desman("run", "refused", "--fake-llm", third).exit_code == EXIT_NEEDS_HUMAN
    [waiting] = json.loads(desman("inbox", "--json").output)
    assert desman("accept", waiting["ref"]).exit_code == 0


def stats_json(*args: str) -> dict[str, Any]:
    result = desman("stats", "--json", *args)
    assert result.exit_code == 0, result.output
    out: dict[str, Any] = json.loads(result.output)
    return out


def test_tasks_subtasks_reviews_and_failures(tmp_path: Path) -> None:
    three_tasks(tmp_path)
    out = stats_json()
    assert out["since"] is None
    assert out["tasks"] == {"done": 3}
    assert out["subtasks"] == {
        "total": 3,
        "done": 3,
        "first_try": 1,  # the operator's accept is not the reviewer's
        "retried": 1,
        "escalated": 1,
        "split": 0,
    }
    assert out["reviews"] == {"accept": 2, "retry": 1}
    assert out["failures"] == {"permanent": 1, "quality": 1}
    assert out["strategies"] == [
        {"strategy": "first", "started": 3, "accepted": 1},
        {"strategy": "feedback", "started": 1, "accepted": 1},
    ]


def test_calls_by_role_and_by_model() -> None:
    conn = db.open_db(db.default_db_path())
    record = calls.recorder(conn)

    def call(model: str, tokens: int, latency: int, error: str | None = None) -> CallRecord:
        status = 200 if error is None else 429
        return CallRecord("groq", model, "k", "p", None, tokens, 1, latency, status, error, None)

    try:
        with calls.for_(calls.Purpose(calls.Role.EXECUTOR)):
            record(call("big", 100, 1000))
            record(call("big", 50, 3000, error="limit"))
        with calls.for_(calls.Purpose(calls.Role.REVIEWER)):
            record(call("small", 10, 200))
        record(call("small", 0, 100))  # e.g. `desman keys check`: made for no role
        result = collect(conn)
    finally:
        conn.close()
    assert [(r.role, r.calls, r.failed, r.prompt_tokens) for r in result.roles] == [
        ("executor", 2, 1, 150),
        ("reviewer", 1, 0, 10),
        ("other", 1, 0, 0),
    ]
    assert [(m.model, m.calls, m.failed, m.limited, m.latency_ms) for m in result.models] == [
        ("big", 2, 1, 1, 2000),
        ("small", 2, 0, 0, 150),
    ]


def test_a_window_counts_what_happened_in_it(tmp_path: Path) -> None:
    three_tasks(tmp_path)
    assert stats_json("--since", "1h")["subtasks"]["total"] == 3
    conn = db.open_db(db.default_db_path())
    try:
        later = collect(conn, datetime.now(UTC) + timedelta(hours=1))
    finally:
        conn.close()
    assert later.tasks == {} and later.subtasks.total == 0 and later.reviews == {}
    assert later.strategies == [] and later.roles == []

    bad = desman("stats", "--since", "soon")
    assert bad.exit_code == 2 and "30m, 24h, 7d or all" in bad.output


def test_stats_for_the_person(tmp_path: Path) -> None:
    assert desman("init-db").exit_code == 0
    assert "Nothing has run yet." in desman("stats").output
    three_tasks(tmp_path)
    shown = desman("stats", "--since", "7d")
    assert shown.exit_code == 0, shown.output
    out = shown.output
    assert "Stats (last 7d)" in out and "3 tasks: done 3." in out
    assert "Subtasks: 3. Done: 3, on the first try: 1 (33%)." in out
    assert "Retried: 1, escalated to you: 1, split into parts: 0." in out
    assert "Reviews: accept 2, retry 1." in out and "Failures by kind: permanent 1" in out
    assert "feedback" in out  # the attempts table


def test_stats_without_a_database_says_so() -> None:
    result = desman("stats")
    assert result.exit_code == 1 and "no database" in result.output
