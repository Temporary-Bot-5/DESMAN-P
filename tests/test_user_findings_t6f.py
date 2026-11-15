"""Regressions for the second batch of user findings."""

from __future__ import annotations

import errno
import socket
from typing import Any

import click
import pytest

from desman import actions, manual
from desman.core.report import task_report
from desman.core.states import Actor
from desman.core.store import Store
from desman.core.workspace import Workspace, workspace_root_for
from desman.gui import app as gui_app
from test_user_findings import finish


@pytest.mark.parametrize(
    "key",
    [
        "response_text",
        "response",
        "message",
        "summary",
        "final_answer",
        "answer_text",
        "reply",
        "unusual",
    ],
)
def test_prose_field(key: str) -> None:
    prose = "Стоимость проезда — 75 рублей. Источник: официальный сайт. " * 2
    assert actions.result_text({"fare_amount_rubles": 75, key: prose}) == prose


def test_ambiguous_or_structured_results_keep_fields() -> None:
    assert "changes:" in actions.result_text(
        {"summary": "Proposed", "changes": [{"value": "new rule"}]}
    )
    assert "other:" in actions.result_text({"one": "a" * 80, "other": "b" * 80})
    assert "metadata:" in actions.result_text({"one": "a" * 80, "metadata": {"x": 1}})
    assert actions.result_text({"custom": "line one\nline two", "ok": True}) == "line one\nline two"


def written(store: Store, task: str, path: str, data: bytes, *, ok: bool = True) -> None:
    Workspace(workspace_root_for(store.conn), task).write(path, data)
    store.note(
        store.list_subtasks(task)[-1].id,
        Actor.EXECUTOR,
        "tool_called",
        {"tool": "write_file", "args": {"path": path}, "ok": ok},
    )


def test_report_displays_written_answer(store: Store, capsys: pytest.CaptureFixture[str]) -> None:
    task = finish(store, {"status": "done", "plan_file": "plan.txt", "description": "Saved"})
    written(store, task, "plan.txt", b"First month: practise daily.\nSecond month: read.\n")
    report = task_report(store, task)
    before = actions.report_output(store, report)
    actions.print_report(report, store)
    assert "plan.txt:\nFirst month: practise daily." in capsys.readouterr().out
    assert actions.report_output(store, report) == before
    assert "content" not in before["files"][0]


@pytest.mark.parametrize("data", [b"\xff", b"a\0b", b"x" * 4001, b"x\n" * 41])
def test_ineligible_files(store: Store, data: bytes) -> None:
    task = finish(store, {"file": "answer.txt"})
    written(store, task, "answer.txt", data)
    assert actions.result_file_texts(store, task_report(store, task)) == []


def test_only_two_written_files_and_no_duplicate_text(store: Store) -> None:
    task = finish(store, {"text": "short result"})
    for i in range(3):
        written(store, task, f"{i}.txt", f"Answer {i}".encode())
    written(store, task, "failed.txt", b"failed", ok=False)
    Workspace(workspace_root_for(store.conn), task).write("unwritten.txt", b"not written")
    assert actions.result_file_texts(store, task_report(store, task)) == [
        ("0.txt", "Answer 0"),
        ("1.txt", "Answer 1"),
    ]
    task2 = finish(store, {"text": "Already present"})
    written(store, task2, "answer.txt", b"Already present")
    assert actions.result_file_texts(store, task_report(store, task2)) == []


def test_keys_manual_explains_full_key() -> None:
    topic = next(topic for topic in manual.TOPICS if topic.name == "providers")
    assert "never shows a key in full" in topic.text
    assert "config.json in the project folder" in topic.text
    assert "provider’s site" in topic.text


def test_busy_gui_port_real_socket() -> None:
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        port = occupied.getsockname()[1]
        with pytest.raises(click.ClickException, match=f"Port {port} is busy"):
            gui_app.run(False, False, port, False)


def test_busy_gui_windows_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def busy(app: Any) -> Any:
        exc = OSError("busy")
        exc.__dict__["winerror"] = 10048
        raise exc

    monkeypatch.setattr(gui_app, "http_server", busy)
    with pytest.raises(click.ClickException, match="Use another --port"):
        gui_app.run(False, False, 1234, False)


def test_other_bind_error_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    def denied(app: Any) -> Any:
        raise OSError(errno.EACCES, "denied")

    monkeypatch.setattr(gui_app, "http_server", denied)
    with pytest.raises(OSError, match="denied"):
        gui_app.run(False, False, 1234, False)


def test_file_boundaries_and_named_path_in_long_result(store: Store) -> None:
    task = finish(store, {"file": "answer.txt", "description": "x" * 900})
    text = "я" * 3961 + "\n" * 39
    written(store, task, "answer.txt", text.encode("utf-8"))
    assert actions.result_file_texts(store, task_report(store, task)) == [("answer.txt", text)]


def test_long_result_without_path_does_not_show_extra_files(store: Store) -> None:
    task = finish(store, {"text": "Full answer. " * 100})
    written(store, task, "extra.txt", b"An unrelated extra")
    assert actions.result_file_texts(store, task_report(store, task)) == []


def test_pending_task_never_shows_file_text(store: Store) -> None:
    from test_user_findings import spec

    task = store.create_task(spec())
    written(store, task, "answer.txt", b"Unfinished")
    assert actions.result_file_texts(store, task_report(store, task)) == []
