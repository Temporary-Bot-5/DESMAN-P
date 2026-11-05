"""Reassignment (ROADMAP 4.2, D4): a worker dies or hangs on a subtask, another one finishes it.

Real `desman worker` processes on one database and real time, like tests/test_crash.py. The
first worker takes the only subtask and is killed (SIGKILL) or frozen (SIGSTOP) while its slow
scripted model call is under way; the second, started after that, does the work:

- a killed worker on this machine is certainly gone, so its lease (or review reservation) is
  taken over at once, without waiting it out;
- a frozen one looks alive, so its lease runs out first; when it wakes up, its late result is
  refused (the lease is taken by token) and recorded as discarded work.

Either way the subtask is done once, with the second worker's result, and the history replays.
"""

from __future__ import annotations

import json
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from desman.core import db
from desman.core.audit import check_history, history
from desman.core.models import SubtaskSpec, TaskSpec
from desman.core.states import TaskStatus
from desman.core.store import DEFAULT_LEASE_SECONDS, DEFAULT_REVIEW_SECONDS, Store

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="SIGKILL, SIGSTOP and pid checks")

ACCEPT = '{"verdict": "accept"}'
SLOW = 120  # seconds a scripted reply "takes": the first worker is always stopped long before


def script(tmp_path: Path, name: str, execute: Any, review: Any) -> Path:
    routes = [
        {"match": "Execute", "steps": [execute], "repeat_last": True},
        {"match": "Review", "steps": [review], "repeat_last": True},
    ]
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps({"routes": routes}))
    return path


def reply(by: str, latency: float = 0.0) -> dict[str, Any]:
    return {"type": "reply", "text": json.dumps({"by": by}), "latency": latency}


def one_task(path: Path) -> str:
    store = Store(db.open_db(path))
    try:
        spec = TaskSpec(
            goal="g", subtasks=[SubtaskSpec(description="job", acceptance_criteria="ok")]
        )
        return store.create_task(spec)
    finally:
        store.conn.close()


def worker(path: Path, fake: Path, *extra: str) -> subprocess.Popen[bytes]:
    command = [sys.executable, "-m", "desman", "worker", "--until-idle", "--poll-seconds", "0.1"]
    command += ["--fake-llm", str(fake), "--db", str(path), "--json", *extra]
    return subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def row(path: Path) -> dict[str, Any]:
    conn = db.connect(path)
    try:
        found = conn.execute("SELECT * FROM subtasks").fetchone()
        return dict(found) if found else {}
    finally:
        conn.close()


def wait_until_holding(path: Path, proc: subprocess.Popen[bytes], column: str) -> None:
    """Until `proc` holds the subtask's lease (`lease_owner`) or review (`review_owner`)."""
    deadline = time.monotonic() + 30
    while not str(row(path).get(column) or "").endswith(f":{proc.pid}"):
        if proc.poll() is not None:
            _, err = proc.communicate()
            raise AssertionError(f"worker exited ({proc.returncode}):\n{err.decode()}")
        if time.monotonic() > deadline:
            proc.kill()
            raise AssertionError(f"timed out waiting for {column}")
        time.sleep(0.05)


def finish(proc: subprocess.Popen[bytes]) -> dict[str, Any]:
    out, err = proc.communicate(timeout=60)
    assert proc.returncode == 0, err.decode()
    result: dict[str, Any] = json.loads(out)
    return result


def kill(proc: subprocess.Popen[bytes]) -> None:
    if sys.platform != "win32":
        proc.send_signal(signal.SIGKILL)
    proc.wait(timeout=10)  # reaped: the pid is really gone, not a zombie


def outcome(path: Path, task_id: str) -> tuple[list[Any], list[str], Store]:
    store = Store(db.open_db(path))
    assert store.task_status(task_id) is TaskStatus.DONE
    [sub] = store.list_subtasks(task_id)
    results = [json.loads(a["content_json"]) for a in store.artifacts(sub.id)]
    assert check_history(store.conn, task_id) == []
    return results, [e.action for e in history(store.conn, task_id)], store


def test_a_killed_workers_subtask_is_done_by_another_at_once(tmp_path: Path) -> None:
    path = tmp_path / "desman.db"
    task_id = one_task(path)
    first = worker(path, script(tmp_path, "first", reply("first", SLOW), ACCEPT))
    try:
        wait_until_holding(path, first, "lease_owner")
    finally:
        kill(first)

    started = time.monotonic()
    second = finish(worker(path, script(tmp_path, "second", reply("second"), ACCEPT)))
    assert time.monotonic() - started < DEFAULT_LEASE_SECONDS / 10  # not after the lease
    assert (second["executed"], second["reviewed"]) == (1, 1)

    results, actions, store = outcome(path, task_id)
    assert results == [{"by": "second"}]  # the killed attempt left nothing behind
    assert actions.count("claim") == 2 and "lease_expired" not in actions
    [taken] = [e for e in history(store.conn, task_id) if e.action == "lease_abandoned"]
    assert f":{first.pid} is gone" in taken.details["reason"]
    [sub] = store.list_subtasks(task_id)
    assert sub.attempts.transient == 1  # the crash counts against the subtask's budget
    store.conn.close()


def test_a_killed_reviewers_review_is_done_by_another_at_once(tmp_path: Path) -> None:
    path = tmp_path / "desman.db"
    task_id = one_task(path)
    slow_review = {"type": "reply", "text": ACCEPT, "latency": SLOW}
    first = worker(path, script(tmp_path, "first", reply("first"), slow_review))
    try:
        wait_until_holding(path, first, "review_owner")
    finally:
        kill(first)

    started = time.monotonic()
    # Its execution would give another result: the work already submitted is reviewed instead.
    second = finish(worker(path, script(tmp_path, "second", reply("second"), ACCEPT)))
    assert time.monotonic() - started < DEFAULT_REVIEW_SECONDS / 10  # not after the reservation
    assert (second["executed"], second["reviewed"]) == (0, 1)

    results, actions, store = outcome(path, task_id)
    assert results == [{"by": "first"}]
    assert actions.count("claim") == 1 and actions.count("verdict") == 1
    store.conn.close()


def test_a_hung_workers_subtask_is_done_by_another_after_its_lease(tmp_path: Path) -> None:
    path = tmp_path / "desman.db"
    task_id = one_task(path)
    lease = 2  # the frozen worker's: the second one waits it out
    first = worker(
        path, script(tmp_path, "first", reply("first", 2 * lease), ACCEPT), "--lease-seconds", "2"
    )
    try:
        wait_until_holding(path, first, "lease_owner")
        if sys.platform != "win32":
            first.send_signal(signal.SIGSTOP)  # alive, but renews nothing
        started = time.monotonic()
        second = finish(worker(path, script(tmp_path, "second", reply("second"), ACCEPT)))
        assert time.monotonic() - started >= lease / 2  # a live owner's lease is waited out
        assert (second["executed"], second["reviewed"]) == (1, 1)
    finally:
        if sys.platform != "win32":
            first.send_signal(signal.SIGCONT)
    late = finish(first)  # wakes up, its reply arrives, the result is refused
    assert (late["executed"], late["reviewed"]) == (1, 0)

    results, actions, store = outcome(path, task_id)
    assert results == [{"by": "second"}]  # the late result was not kept
    assert "lease_abandoned" not in actions
    [expired] = [e for e in history(store.conn, task_id) if e.action == "lease_expired"]
    assert expired.actor == "scheduler"
    [discarded] = [e for e in history(store.conn, task_id) if e.action == "work_discarded"]
    assert discarded.details["owner"].endswith(f":{first.pid}")
    store.conn.close()
