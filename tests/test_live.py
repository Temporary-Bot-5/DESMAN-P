"""Tests for live progress (ROADMAP C.4): calls attributed to what they were for, the modules
of a task read from the call log, and the view the chat shows while a task runs."""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from click.testing import CliRunner
from rich.console import Console

from desman import runs
from desman.cli import cli
from desman.config import Config, ConfigManager, default_config_path
from desman.core import calls, db
from desman.core.coordinator import Coordinator, Review, Success, Verdict, Work
from desman.core.models import Subtask, SubtaskSpec, TaskSpec
from desman.core.report import task_report
from desman.core.store import Store
from desman.interlocutor import CHAT, RETELL
from desman.live import Progress, key_labels
from desman.llm.openai_compat import CallRecord, OpenAICompatible, key_id
from helpers import make_config

KEY = "gsk_" + "k" * 48
OTHER = "gsk_" + "o" * 48


@pytest.fixture
def store() -> Iterator[Store]:
    conn = db.open_db(":memory:")
    yield Store(conn)
    conn.close()


def call(error_type: str | None = None, **fields: Any) -> CallRecord:
    values: dict[str, Any] = {
        "provider": "groq",
        "model": "llama",
        "key_id": key_id(KEY),
        "prompt_hash": "p",
        "response_hash": None if error_type else "r",
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "latency_ms": 5,
        "http_status": 429 if error_type else 200,
        "error_kind": "limit" if error_type else None,
        "error": error_type,
        "error_type": error_type,
    }
    return CallRecord(**{**values, **fields})


def one_task(store: Store, *descriptions: str) -> str:
    return store.create_task(
        TaskSpec(
            goal="Find flights",
            subtasks=[SubtaskSpec(description=d, acceptance_criteria="done") for d in descriptions],
        )
    )


# ── Attribution ──────────────────────────────────────────────────────────────


def test_role_check_matches_the_enum(store: Store) -> None:
    sql = store.conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'model_calls'"
    ).fetchone()[0]
    match = re.search(r"role TEXT\s+CHECK \(role IN \(([^)]*)\)\)", sql)
    assert match is not None
    assert set(re.findall(r"'(\w+)'", match.group(1))) == {r.value for r in calls.Role}


def test_a_call_is_recorded_as_made_for_its_purpose(store: Store) -> None:
    task_id = one_task(store, "Search")
    [sub] = store.list_subtasks(task_id)
    record = calls.recorder(store.conn)
    seen: list[int] = []
    record(call())  # nobody said what for: e.g. `desman keys check`
    with (
        calls.watching(lambda: seen.append(1)),
        calls.for_(calls.Purpose(calls.Role.EXECUTOR, task_id=task_id, subtask_id=sub.id)),
    ):
        record(call("RateLimited"))
    rows = store.conn.execute(
        "SELECT role, task_id, subtask_id, error_type FROM model_calls ORDER BY seq"
    ).fetchall()
    assert [tuple(r) for r in rows] == [
        (None, None, None, None),
        ("executor", task_id, sub.id, "RateLimited"),
    ]
    assert seen == [1]  # the watcher heard the second call only


def test_the_coordinator_attributes_its_roles_and_reports_each_step(store: Store) -> None:
    task_id = one_task(store, "Search")
    record = calls.recorder(store.conn)

    def executor(work: Work) -> Success:
        record(call())
        return Success(content={"flights": []})

    def reviewer(subtask: Subtask, result: dict[str, Any] | None) -> Review:
        record(call())
        return Review(verdict=Verdict.ACCEPT)

    steps: list[int] = []
    Coordinator(store, executor, reviewer, watch=lambda: steps.append(1)).run(task_id)
    [sub] = store.list_subtasks(task_id)
    rows = store.conn.execute("SELECT role, task_id, subtask_id FROM model_calls ORDER BY seq")
    assert [tuple(r) for r in rows] == [
        ("executor", task_id, sub.id),
        ("reviewer", task_id, sub.id),
    ]
    assert len(steps) >= 3  # claimed, executed, reviewed


# ── Modules of a task ────────────────────────────────────────────────────────


def test_modules_of_a_task_from_the_call_log(store: Store) -> None:
    task_id = one_task(store, "Search", "Compare")
    first, second = store.list_subtasks(task_id)
    other_task = one_task(store, "Elsewhere")
    record = calls.recorder(store.conn)

    def made(role: calls.Role, sub: str, task: str = task_id) -> calls.Purpose:
        return calls.Purpose(role, task_id=task, subtask_id=sub)

    with calls.for_(made(calls.Role.EXECUTOR, first.id)):
        record(call(limits={"x-ratelimit-remaining-requests": "9"}))
    with calls.for_(made(calls.Role.REVIEWER, second.id)):
        record(call())  # the latest says what it does now; the limits stay from the one before
    with calls.for_(made(calls.Role.EXECUTOR, first.id)):
        record(call(key_id=key_id(OTHER), error_type="QuotaExhausted", error_kind="limit"))
    with calls.for_(made(calls.Role.EXECUTOR, "x", other_task)):
        record(call())  # another task's: not counted

    groq, other = calls.modules(store.conn, task_id)
    assert (groq.key_id, groq.calls, groq.prompt_tokens + groq.completion_tokens) == (
        key_id(KEY),
        2,
        30,
    )
    assert (groq.role, groq.subtask_id, groq.error_type) == (
        calls.Role.REVIEWER,
        second.id,
        None,
    )
    assert groq.limits == {"x-ratelimit-remaining-requests": "9"}
    assert (other.key_id, other.error_type) == (key_id(OTHER), "QuotaExhausted")


# ── The view ─────────────────────────────────────────────────────────────────


def config_with_keys() -> Config:
    return make_config(
        language="en",
        configured=True,
        api_keys={"GROQ_API_KEY": [{"value": KEY}, {"value": OTHER}]},
    )


def test_keys_are_shown_by_label_and_mask_never_whole() -> None:
    labels = key_labels(config_with_keys())
    assert labels[key_id(KEY)] == "#1 gsk_…kkkk"
    assert all(KEY not in label and OTHER not in label for label in labels.values())


def test_without_a_terminal_a_line_is_printed_for_each_change(
    store: Store, capsys: pytest.CaptureFixture[str]
) -> None:
    task_id = one_task(store, "Search")
    record = calls.recorder(store.conn)
    with Progress(store, task_id, config_with_keys(), terminal=False) as progress:
        claimed = store.claim("me", task_id=task_id)
        assert claimed is not None
        progress.refresh()
        purpose = calls.Purpose(calls.Role.EXECUTOR, task_id=task_id, subtask_id=claimed.id)
        with calls.for_(purpose), calls.watching(progress.refresh):
            record(
                call(
                    limits={
                        "x-ratelimit-limit-requests": "1000",
                        "x-ratelimit-remaining-requests": "998",
                    }
                )
            )
            record(call())  # the same line: not printed again
            record(call("RateLimited"))
            record(call(key_id=key_id(OTHER), error_type="QuotaExhausted"))
    printed = capsys.readouterr().out.splitlines()
    assert printed == [
        "[0] ready  Search",
        "[0] running  Search",
        "groq llama (#1 gsk_…kkkk): executor · [0] · working",
        "groq llama (#1 gsk_…kkkk): executor · [0] · waiting for a limit",
        "groq llama (#2 gsk_…oooo): executor · [0] · quota used up",
    ]


def test_in_a_terminal_the_view_has_the_bar_and_what_each_module_spent(store: Store) -> None:
    task_id = one_task(store, "Search", "Compare")
    record = calls.recorder(store.conn)
    claimed = store.claim("me", task_id=task_id)
    assert claimed is not None
    with calls.for_(calls.Purpose(calls.Role.EXECUTOR, task_id=task_id, subtask_id=claimed.id)):
        record(
            call(
                limits={
                    "x-ratelimit-limit-requests": "1000",
                    "x-ratelimit-remaining-requests": "998",
                }
            )
        )
    progress = Progress(store, task_id, config_with_keys(), terminal=True)

    console = Console(width=200, record=True)
    console.print(progress.render(task_report(store, task_id), calls.modules(store.conn, task_id)))
    shown = console.export_text()
    assert "0/2 subtasks" in shown and "[0] running" in shown and "[1] ready" in shown
    assert "1 call · 15 tokens · requests left 998/1000" in shown
    assert "working" in shown and KEY not in shown


# ── In the chat ──────────────────────────────────────────────────────────────

PLAN = json.dumps({"subtasks": [{"description": "Find flights", "acceptance_criteria": "Listed"}]})
TURNS = {
    CHAT: json.dumps({"type": "task", "title": "Flights", "goal": "Find cheap flights"}),
    "Plan task": PLAN,
    "Execute subtask": json.dumps({"flights": ["TP 1351"]}),
    "Review subtask": '{"verdict": "accept"}',
    RETELL: '{"text": "TP 1351."}',
}


def test_a_chat_task_shows_its_progress_and_every_call_is_attributed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        prompt = json.loads(request.content)["messages"][-1]["content"]
        text = next(reply for heading, reply in TURNS.items() if prompt.startswith(heading))
        return httpx.Response(
            200,
            json={
                "model": "llama",
                "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            },
        )

    def models(config: Config, conn: sqlite3.Connection, **_: Any) -> runs.Models:
        llm = OpenAICompatible(
            "groq",
            "https://api.example/v1",
            [KEY],
            "llama",
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        return runs.Models(runs.attach(llm, conn, cache=False), None)

    monkeypatch.setattr(runs, "models", models)
    ConfigManager(default_config_path()).save(config_with_keys())
    result = CliRunner().invoke(cli, [], input="find cheap flights\n")
    assert result.exit_code == 0, result.output
    assert "[0] running  Find flights" in result.output
    assert "groq llama (#1 gsk_…kkkk): executor · [0] · working" in result.output
    assert "TP 1351." in result.output and KEY not in result.output

    conn = sqlite3.connect(db.default_db_path())
    [(session_id,)] = conn.execute("SELECT id FROM sessions")
    [(task_id,)] = conn.execute("SELECT id FROM tasks")
    [(subtask_id,)] = conn.execute("SELECT id FROM subtasks")
    rows = conn.execute(
        "SELECT role, task_id, subtask_id, session_id FROM model_calls ORDER BY seq"
    )
    assert list(rows) == [
        ("interlocutor", None, None, session_id),
        ("planner", None, None, session_id),
        ("executor", task_id, subtask_id, None),
        ("reviewer", task_id, subtask_id, None),
        ("interlocutor", None, None, session_id),  # the retelling is the session's own
    ]


def test_of_two_modules_on_a_subtask_only_the_later_is_working(
    store: Store, capsys: pytest.CaptureFixture[str]
) -> None:
    """A retry with another model: the first module is no longer at work."""
    task_id = one_task(store, "Search")
    claimed = store.claim("me", task_id=task_id)
    assert claimed is not None
    record = calls.recorder(store.conn)
    with calls.for_(calls.Purpose(calls.Role.EXECUTOR, task_id=task_id, subtask_id=claimed.id)):
        record(call())
        record(call(provider="mistral", key_id=None))
    Progress(store, task_id, config_with_keys(), terminal=False).refresh()
    printed = capsys.readouterr().out.splitlines()
    assert "groq llama (#1 gsk_…kkkk): executor · [0] · idle" in printed
    assert "mistral llama (local model): executor · [0] · working" in printed
