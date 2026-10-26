"""Crash test (ROADMAP 1.11, D3): `desman run` killed with SIGKILL mid-subtask, then `resume`.

These run the real CLI in subprocesses and use real time (polling, a killed process), so they
are the one place besides lock contention where tests wait on the clock. Each finishes in a few
seconds: `resume` takes the dead run's lease over at once instead of waiting it out.
"""

from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from desman.cli import cli
from desman.core import db
from desman.core.audit import check_history, history
from desman.core.owner import current_owner, owner_is_gone
from desman.core.store import DEFAULT_LEASE_SECONDS, Store

pytestmark = pytest.mark.skipif(os.name != "posix", reason="SIGKILL and pid checks are POSIX")

ACCEPT = '{"verdict": "accept"}'
SLOW = 120  # seconds a scripted reply "takes": the run is always killed long before


def write(path: Path, data: Any) -> Path:
    path.write_text(json.dumps(data))
    return path


def fake(tmp_path: Path, name: str, execute: list[Any], review: list[Any]) -> Path:
    routes = [
        {"match": "Execute", "steps": execute},
        {"match": "Review", "steps": review, "repeat_last": len(review) == 1},
    ]
    return write(tmp_path / f"{name}.json", {"routes": routes})


def chain_spec(tmp_path: Path) -> Path:
    return write(
        tmp_path / "spec.json",
        {
            "goal": "three steps",
            "subtasks": [
                {"description": "a", "acceptance_criteria": "ok"},
                {"description": "b", "acceptance_criteria": "ok", "depends_on": [0]},
                {"description": "c", "acceptance_criteria": "ok", "depends_on": [1]},
            ],
        },
    )


def desman(*args: str | Path) -> list[str]:
    return [sys.executable, "-m", "desman", *map(str, args)]


def wait_for(condition: Any, proc: subprocess.Popen[bytes], what: str, timeout: float = 30) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if proc.poll() is not None:
            out = proc.stdout.read().decode() if proc.stdout else ""
            raise AssertionError(f"run exited ({proc.returncode}) before {what}:\n{out}")
        if time.monotonic() > deadline:
            proc.kill()
            raise AssertionError(f"timed out waiting for {what}")
        time.sleep(0.05)


def subtask(path: Path, seq: int) -> dict[str, Any]:
    conn = db.connect(path)
    try:
        row = conn.execute("SELECT * FROM subtasks WHERE seq = ?", (seq,)).fetchone()
        return dict(row) if row else {}
    except sqlite3.OperationalError:  # the run has created the file but not the schema yet
        return {}
    finally:
        conn.close()


def run_and_kill(tmp_path: Path, script: Path, killed_when: Any) -> tuple[Path, int]:
    path = tmp_path / "desman.db"
    proc = subprocess.Popen(
        desman("run", "--spec", chain_spec(tmp_path), "--fake-llm", script, "--db", path),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        wait_for(lambda: path.exists() and killed_when(path, proc.pid), proc, "the kill point")
    finally:
        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=10)  # reaped: the pid is really gone, not a zombie
    assert proc.returncode == -signal.SIGKILL
    return path, proc.pid


def resume(path: Path, script: Path) -> tuple[subprocess.CompletedProcess[str], float]:
    conn = db.connect(path)
    task_id = conn.execute("SELECT id FROM tasks").fetchone()[0]
    conn.close()
    started = time.monotonic()
    done = subprocess.run(
        desman("resume", task_id[:8], "--fake-llm", script, "--db", path, "--json"),
        capture_output=True,
        text=True,
        timeout=60,
    )
    return done, time.monotonic() - started


def artifacts(store: Store) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for task in store.conn.execute("SELECT id FROM tasks"):
        for sub in store.list_subtasks(task[0]):
            out[sub.description] = [json.loads(a["content_json"]) for a in store.artifacts(sub.id)]
    return out


def test_killed_mid_execution_resumes_without_redoing_or_duplicating(tmp_path: Path) -> None:
    first = fake(
        tmp_path,
        "first",
        execute=['{"step": "a"}', {"type": "reply", "text": '{"step": "lost"}', "latency": SLOW}],
        review=[ACCEPT],
    )

    def executing_b(path: Path, pid: int) -> bool:
        b = subtask(path, 1)
        return b.get("status") == "running" and str(b.get("lease_owner", "")).endswith(f":{pid}")

    path, pid = run_and_kill(tmp_path, first, executing_b)
    assert subtask(path, 0)["status"] == "done"
    assert subtask(path, 1)["status"] == "running"  # the dead run still holds b's lease
    assert subtask(path, 2)["status"] == "pending"

    # The second run has steps only for b and c: redoing a would run the script dry.
    second = fake(tmp_path, "second", execute=['{"step": "b"}', '{"step": "c"}'], review=[ACCEPT])
    done, elapsed = resume(path, second)
    assert done.returncode == 0, done.stdout + done.stderr
    assert elapsed < DEFAULT_LEASE_SECONDS / 10  # taken over at once, not after the lease
    assert json.loads(done.stdout)["task"]["status"] == "done"

    store = Store(db.open_db(path))
    task_id = store.conn.execute("SELECT id FROM tasks").fetchone()[0]
    assert artifacts(store) == {
        "a": [{"step": "a"}],  # not redone
        "b": [{"step": "b"}],  # the killed attempt left nothing behind
        "c": [{"step": "c"}],
    }
    b = store.list_subtasks(task_id)[1]
    assert b.attempts.transient == 1  # the crash counts against the subtask's budget
    [taken] = [e for e in history(store.conn, task_id) if e.action == "lease_abandoned"]
    assert taken.subtask_id == b.id and f":{pid} is gone" in taken.details["reason"]
    assert check_history(store.conn, task_id) == []
    store.conn.close()
    audit = CliRunner().invoke(cli, ["audit", task_id, "--db", str(path)])
    assert audit.exit_code == 0, audit.output


def test_killed_mid_review_reviews_again_without_rerunning(tmp_path: Path) -> None:
    first = fake(
        tmp_path,
        "first",
        execute=['{"step": "a"}', '{"step": "b"}'],
        review=[ACCEPT, {"type": "reply", "text": ACCEPT, "latency": SLOW}],
    )

    seen: list[float] = []

    def reviewing_b(path: Path, pid: int) -> bool:
        # Kill a moment after b enters review, so the slow review call is surely under way.
        if not seen and subtask(path, 1).get("status") == "review":
            seen.append(time.monotonic())
        return bool(seen) and time.monotonic() - seen[0] > 0.3

    path, _ = run_and_kill(tmp_path, first, reviewing_b)
    second = fake(tmp_path, "second", execute=['{"step": "c"}'], review=[ACCEPT])
    done, _ = resume(path, second)
    assert done.returncode == 0, done.stdout + done.stderr

    store = Store(db.open_db(path))
    assert artifacts(store) == {"a": [{"step": "a"}], "b": [{"step": "b"}], "c": [{"step": "c"}]}
    task_id = store.conn.execute("SELECT id FROM tasks").fetchone()[0]
    assert check_history(store.conn, task_id) == []
    store.conn.close()


# ── Owner detection ──────────────────────────────────────────────────────────


def test_owner_is_gone_only_when_certain() -> None:
    host = current_owner().rpartition(":")[0]
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    assert owner_is_gone(f"{host}:{child.pid}")  # exited and reaped
    assert not owner_is_gone(current_owner())  # ourselves
    assert not owner_is_gone(f"{host}:{os.getppid()}")  # alive
    assert not owner_is_gone(f"{host}:1")  # alive, another user's (or ours in a container)
    assert not owner_is_gone(f"some-other-host:{child.pid}")  # cannot know
    for garbage in ("", "coordinator", f"{host}:", f"{host}:-5", f"{host}:0", f"{host}:x1", "w"):
        assert not owner_is_gone(garbage), garbage
