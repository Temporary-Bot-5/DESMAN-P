"""L.1: files the person gives a task through the CLI, chat, or browser."""

from __future__ import annotations

import codecs
import json
import os
import sys
import threading
from collections.abc import Sequence
from contextlib import closing, suppress
from pathlib import Path
from urllib.parse import quote

import pytest
from click.testing import CliRunner
from pydantic import ValidationError

from conftest import FakeClock
from desman import commands, runs, task_files, ui
from desman.chat import Chat
from desman.cli import cli
from desman.config import open_config
from desman.core import db
from desman.core.sessions import Kind, Sessions
from desman.core.store import Store
from desman.core.workspace import Workspace, workspace_root_for
from desman.gui.front import WebFront
from desman.gui.runners import PageChat, Runners
from desman.gui.server import MAX_BODY, Server
from desman.interlocutor import CHAT
from desman.llm.fake import FakeLLM
from desman.planner import PLAN
from helpers import make_config
from test_cli_tasks import desman, script
from test_gui import headers, next_prompt
from test_tools import tool_calls

TEXT = "Привет, мир!\r\nВторая строка.\rТретья строка.\n"
NORMALIZED = TEXT.replace("\r\n", "\n").replace("\r", "\n")
PLAN_REPLY = '{"subtasks":[{"description":"Read the attachment","acceptance_criteria":"Answer"}]}'
TASK_REPLY = '{"type":"task","title":"Read it","goal":"Read the attached document"}'


@pytest.mark.parametrize(
    ("encoding", "bom"),
    [
        ("utf-8", b""),
        ("utf-8", codecs.BOM_UTF8),
        ("utf-16-le", codecs.BOM_UTF16_LE),
        ("utf-16-be", codecs.BOM_UTF16_BE),
        ("cp1251", b""),
    ],
)
def test_file_decoding_and_storage(tmp_path: Path, encoding: str, bom: bytes) -> None:
    source = tmp_path / "документ.TXT"
    source.write_bytes(bom + TEXT.encode(encoding))
    given = task_files.given_file(source)
    assert (given.name, given.text, given.chars, given.source) == (
        "input/документ.TXT",
        NORMALIZED,
        len(NORMALIZED),
        str(source),
    )
    root = tmp_path / "work"
    task = "a" * 32
    assert task_files.place(root, task, [given]) == [given.name]
    assert Workspace(root, task).read(given.name) == NORMALIZED.encode()


@pytest.mark.parametrize("data", [b"\x89PNG\r\n\x1a\n", b"a\0b", b"\xa9\xa9", b"\xff\xfe\x00"])
def test_binary_or_malformed_text_is_refused(tmp_path: Path, data: bytes) -> None:
    path = tmp_path / "file.txt"
    path.write_bytes(data)
    with pytest.raises(task_files.GivenFileError, match="UTF-8"):
        task_files.given_file(path)


def test_cp1251_heuristic() -> None:
    assert task_files._looks_like_cp1251("Привет! 1\tстрока")
    assert not task_files._looks_like_cp1251("résumé — résumé")
    assert not task_files._looks_like_cp1251("привет\x00\x00")


def test_file_size_format_and_regular_file_checks(tmp_path: Path) -> None:
    with pytest.raises(task_files.GivenFileError, match="regular file"):
        task_files.given_file(tmp_path)
    path = tmp_path / "report.docx"
    path.write_text("text", encoding="utf-8")
    with pytest.raises(task_files.GivenFileError, match="extension"):
        task_files.given_file(path)
    path = tmp_path / "large.txt"
    with path.open("wb") as file:
        file.truncate(task_files.MAX_GIVEN_BYTES + 1)
    with pytest.raises(task_files.GivenFileError, match="too large"):
        task_files.given_file(path)


def test_user_can_choose_a_symlink(tmp_path: Path) -> None:
    source = tmp_path / "original.txt"
    source.write_text("content", encoding="utf-8")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(source)
    except OSError as e:
        pytest.skip(f"can’t create a symlink here: {e}")
    attached = task_files.given_file(link)
    assert attached.name == "input/link.txt" and attached.text == "content"


def test_fifo_and_device_are_refused(tmp_path: Path) -> None:
    if sys.platform == "win32":
        pytest.skip("POSIX special files")
    fifo = tmp_path / "pipe.txt"
    os.mkfifo(fifo)
    with pytest.raises(task_files.GivenFileError, match="regular file"):
        task_files.given_file(fifo)
    with pytest.raises(task_files.GivenFileError, match="regular file"):
        task_files.given_file(Path(os.devnull))


def test_names_and_duplicate_suffixes_use_workspace(tmp_path: Path) -> None:
    given = [
        task_files.given_upload(name, str(i).encode())
        for i, name in enumerate(
            [
                "../../план ?.md",
                r"C:\fakepath\план !.md",
                "ПЛАН !.MD",
                "CON.txt",
                ".desman-tmp-test.txt",
                "字" * 110 + ".txt",
                "字" * 110 + ".txt",
            ]
        )
    ]
    root = tmp_path / "work"
    task = "b" * 32
    paths = task_files.place(root, task, given)
    assert paths[:5] == [
        "input/план__.md",
        "input/план__-2.md",
        "input/ПЛАН__-3.MD",
        "input/_CON.txt",
        "input/_.desman-tmp-test.txt",
    ]
    assert len(set(paths)) == len(paths)
    assert paths[-2].endswith(".txt") and paths[-1].endswith("-2.txt")
    assert all(len(Path(path).name) <= 100 for path in paths)
    workspace = Workspace(root, task)
    assert sorted(paths) == sorted(workspace.files())
    for i, path in enumerate(paths):
        assert workspace.read(path) == str(i).encode()
        assert path in task_files.goal_note(given)


def test_combined_limits_are_checked_before_any_write(tmp_path: Path) -> None:
    task = "c" * 32
    small = task_files.given_upload("file.txt", b"x")
    large = task_files.given_upload("file.txt", b"x" * task_files.MAX_GIVEN_BYTES)
    with pytest.raises(task_files.GivenFileError, match="too many files"):
        task_files.place(tmp_path, task, [small] * (task_files.MAX_GIVEN_FILES + 1))
    with pytest.raises(task_files.GivenFileError, match="large together"):
        task_files.place(
            tmp_path, task, [large] * (task_files.MAX_GIVEN_TOTAL // task_files.MAX_GIVEN_BYTES + 1)
        )
    assert not (tmp_path / task).exists()


@pytest.mark.parametrize("as_json", [False, True])
def test_run_reads_attached_file_using_the_executor_tool(tmp_path: Path, as_json: bool) -> None:
    source = tmp_path / "report.txt"
    secret = "Source material to read"
    source.write_bytes(secret.encode())
    fake = script(
        tmp_path,
        [tool_calls(("read_file", {"path": "input/report.txt"})), '{"text":"I read it."}'],
    )
    args = ["--json"] if as_json else []
    result = desman(
        "run", "Read the report", "--file", source, "--file", source, "--fake-llm", fake, *args
    )
    assert result.exit_code == 0, result.output
    if as_json:
        assert json.loads(result.output)["files"] == ["input/report.txt", "input/report-2.txt"]
    else:
        assert (
            result.output.splitlines()[1] == "Files attached: input/report.txt, input/report-2.txt"
        )
    with closing(db.open_db()) as conn:
        store = Store(conn)
        [(task, goal)] = conn.execute("SELECT id, goal FROM tasks")
        assert f"input/report.txt ({len(secret)} characters)" in goal
        assert f"input/report-2.txt ({len(secret)} characters)" in goal
        assert secret not in goal and str(source) not in goal
        workspace = Workspace(workspace_root_for(conn), task)
        assert (
            workspace.read("input/report.txt")
            == workspace.read("input/report-2.txt")
            == secret.encode()
        )
        [subtask] = store.list_subtasks(task)
        [effect] = store.effects(subtask.id, 0)
        assert json.loads(effect["result_json"])["output"]["content"] == secret


def test_spec_and_bad_file_fail_before_the_model_is_loaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from desman import cli as cli_module

    def unexpected(*args: object, **kwargs: object) -> None:
        pytest.fail("loading models before validating attachments")

    monkeypatch.setattr(cli_module, "_provider", unexpected)
    bad = tmp_path / "binary.txt"
    bad.write_bytes(b"\x00")
    spec = tmp_path / "spec.json"
    spec.write_text("{}", encoding="utf-8")
    fake = script(tmp_path, [])
    conflict = desman("run", "--spec", spec, "--file", bad, "--fake-llm", fake)
    assert conflict.exit_code == 2 and "--file requires GOAL" in conflict.output
    result = desman("run", "Read it", "--file", bad, "--fake-llm", fake)
    assert result.exit_code == 1 and "binary.txt" in result.output and "UTF-8" in result.output
    assert not db.default_db_path().exists()


def test_cli_write_failure_rolls_back_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "report.txt"
    source.write_bytes(b"text")

    def fail(self: Workspace, path: str, data: bytes | str) -> str:
        raise OSError("write failed")

    monkeypatch.setattr(Workspace, "write", fail)
    result = desman("run", "Read the report", "--file", source, "--fake-llm", script(tmp_path, []))
    assert result.exit_code == 1 and "task was not started" in result.output
    with closing(db.open_db()) as conn:
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0


def install_chat_models(clock: FakeClock, monkeypatch: pytest.MonkeyPatch) -> FakeLLM:
    fake = FakeLLM(clock=clock, sleep=clock.sleep)
    fake.on(CHAT, TASK_REPLY, repeat_last=True)
    fake.on(PLAN, PLAN_REPLY, repeat_last=True)
    fake.on("Execute", '{"text":"Read it."}', repeat_last=True)
    fake.on("Review", '{"verdict":"accept"}', repeat_last=True)
    fake.on("Retell", '{"text":"Read it."}', repeat_last=True)
    monkeypatch.setattr(runs, "models", lambda *args, **kwargs: runs.Models(fake, None))
    open_config().save(make_config(language="en", configured=True))
    return fake


@pytest.mark.parametrize("temporary", [False, True])
def test_chat_attaches_to_its_next_task_and_clears_pending(
    tmp_path: Path, clock: FakeClock, monkeypatch: pytest.MonkeyPatch, temporary: bool
) -> None:
    fake = install_chat_models(clock, monkeypatch)
    path = tmp_path / "report draft.md"
    payload = "Only the executor needs these words."
    path.write_text(payload, encoding="utf-8")
    args = ["-t"] if temporary else []
    result = CliRunner().invoke(cli, args, input=f'/file "{path}"\n/file\nRead it\n/file\n/exit\n')
    assert result.exit_code == 0, result.output
    assert "Attached for the next task: input/report_draft.md" in result.output
    assert "Files pending for the next task: input/report_draft.md" in result.output
    assert "received the attached files: input/report_draft.md" in result.output
    assert "No files are pending." in result.output
    (plan,) = [request.prompt for request in fake.calls if request.prompt.startswith(PLAN)]
    assert f"input/report_draft.md ({len(payload)} characters)" in plan and payload not in plan
    (turn,) = [request.prompt for request in fake.calls if request.prompt.startswith(CHAT)]
    attached = f"Attached for the next task: input/report_draft.md ({len(payload)} characters)."
    assert attached in turn
    assert '"pending_files": "input/report_draft.md"' in turn and payload not in turn
    with closing(db.open_db()) as conn:
        [(task, session_id)] = conn.execute("SELECT id, session_id FROM tasks")
        assert (
            Workspace(workspace_root_for(conn), task).read("input/report_draft.md")
            == payload.encode()
        )
        if temporary:
            assert not Sessions(conn).recent()
            assert (
                conn.execute(
                    "SELECT COUNT(*) FROM messages WHERE text LIKE '%report_draft%'"
                ).fetchone()[0]
                == 0
            )
        else:
            messages = Sessions(conn).messages(session_id)
            assert messages[0].kind is Kind.EVENT
            assert messages[0].text == attached
            assert messages[0].styled is None
            [received] = [m for m in messages if "received the attached files:" in m.text]
            assert received.kind is Kind.EVENT and received.styled is None


def test_chat_clear_records_an_event_and_errors_are_localized(tmp_path: Path) -> None:
    open_config().save(make_config(language="ru", configured=True))
    good = tmp_path / "ok.txt"
    good.write_bytes(b"ok")
    bad = tmp_path / "bad.txt"
    bad.write_bytes(b"\0")
    result = CliRunner().invoke(
        cli, [], input=f"/file {bad}\n/file {good}\n/file clear\n/file\n/exit\n"
    )
    assert result.exit_code == 0, result.output
    assert "Не удалось прикрепить" in result.output
    assert "сохраните файл как текст в кодировке UTF-8" in result.output
    assert "Нет файлов для следующей задачи." in result.output
    attached = "К следующей задаче прикреплён файл input/ok.txt (2 символа)"
    cleared = "Список файлов для следующей задачи очищен."
    assert attached in result.output and cleared in result.output
    with closing(db.open_db()) as conn:
        sessions = Sessions(conn)
        info = sessions.latest()
        assert info is not None
        messages = sessions.messages(info.id)
        assert [m.kind for m in messages] == [Kind.EVENT, Kind.EVENT]
        assert [m.text for m in messages] == [attached, cleared]
        assert all(m.styled is None for m in messages)


def test_chat_keeps_files_when_creation_fails(
    tmp_path: Path, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_chat_models(clock, monkeypatch)
    path = tmp_path / "report.txt"
    path.write_bytes(b"input")
    real_place = task_files.place
    writes = 0

    def fail_once(
        root: Path,
        task: str,
        given: list[task_files.Given],
        *,
        prefetched: Sequence[task_files.Given] = (),
    ) -> list[str]:
        nonlocal writes
        writes += 1
        if writes == 1:
            raise OSError("disk full")
        return real_place(root, task, given, prefetched=prefetched)

    monkeypatch.setattr(task_files, "place", fail_once)
    result = CliRunner().invoke(
        cli, [], input=f"/file {path}\nFirst request\n/file\nSecond request\n/file\n/exit\n"
    )
    assert result.exit_code == 0, result.output
    assert "Your files are still pending." in result.output
    assert "Files pending for the next task: input/report.txt" in result.output
    assert "No files are pending." in result.output
    with closing(db.open_db()) as conn:
        [(task,)] = conn.execute("SELECT id FROM tasks")
        assert Workspace(workspace_root_for(conn), task).read("input/report.txt") == b"input"


def test_run_args_accepts_file_limits() -> None:
    files = ["a.txt"] * (task_files.MAX_GIVEN_FILES - 1) + ["x" * 4096]
    assert commands.RunArgs(goal="Read", file=files).file == files


@pytest.mark.parametrize(
    "files", [["a.txt"] * (task_files.MAX_GIVEN_FILES + 1), [""], ["x" * 4097]]
)
def test_run_args_rejects_files_outside_limits(files: list[str]) -> None:
    with pytest.raises(ValidationError, match="file"):
        commands.RunArgs(goal="Read", file=files)


def test_catalog_run_accepts_typed_paths_but_never_reads_a_model_supplied_path(
    tmp_path: Path, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_chat_models(clock, monkeypatch)
    source = tmp_path / "report.txt"
    source.write_bytes(b"exclusive to the person")
    with closing(db.open_db()) as conn:
        chat = Chat(Sessions(conn))
        chat._typed(f'run --file "{source}" "Read"')
        [(task,)] = conn.execute("SELECT id FROM tasks")
        assert (
            Workspace(workspace_root_for(conn), task).read("input/report.txt")
            == source.read_bytes()
        )
    run_signature = next(
        line for line in commands.for_model().splitlines() if line.startswith("- run ")
    )
    assert '"file"' not in run_signature

    reply = json.dumps(
        {"type": "command", "name": "run", "args": {"goal": "read", "file": [str(source)]}}
    )
    # A fresh model for this case ensures the earlier fake's repeated CHAT route cannot match.
    refused = FakeLLM(clock=clock, sleep=clock.sleep).on(CHAT, reply)
    monkeypatch.setattr(runs, "models", lambda *args, **kwargs: runs.Models(refused, None))
    result = CliRunner().invoke(cli, [], input="Read some file\n/exit\n")
    assert "Only you can attach local files." in result.output
    assert len(refused.calls) == 1


def test_gui_upload_preserves_prompt_and_reaches_its_next_task(
    tmp_path: Path, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = install_chat_models(clock, monkeypatch)
    path = tmp_path / "page.db"
    front = WebFront()
    server = Server(front, 12345, poll_timeout=0, database=path)
    ready = threading.Event()

    def page() -> None:
        with closing(db.open_db(path)) as conn, ui.use(front):
            runners = Runners(path, front, session_clock=clock)
            chat = PageChat(Sessions(conn, clock=clock), runners)
            ready.set()
            try:
                with suppress(ui.Cancelled):
                    chat.run()
            finally:
                runners.close()

    thread = threading.Thread(target=page, daemon=True)
    thread.start()
    assert ready.wait(5)
    prompt = next_prompt(front)
    attachment_headers = headers(
        server,
        **{
            "X-Desman-Prompt": str(prompt["id"]),
            "X-Desman-Filename": quote("тест 1.txt"),
            "Content-Type": "application/octet-stream",
        },
    )

    try:
        assert (
            server.handle("POST", "/attach", {"Host": "127.0.0.1:12345"}, b"content").status == 403
        )
        assert (
            server.handle(
                "POST", "/attach", {**attachment_headers, "Origin": "http://other"}, b"content"
            ).status
            == 403
        )
        large = server.handle(
            "POST", "/attach", attachment_headers, b"x" * (task_files.MAX_GIVEN_BYTES + 1)
        )
        assert large.status == 413 and "2 000 000 bytes" in json.loads(large.body)["error"]
        assert server.handle("POST", "/attach", attachment_headers, oversized=True).status == 413
        assert (
            server.handle("POST", "/answer", headers(server), b"x" * (MAX_BODY + 1)).status == 413
        )

        body = b"x" * (MAX_BODY + 1)
        assert server.handle("POST", "/attach", attachment_headers, body).status == 200
        loaded = json.loads(server.handle("GET", "/events", headers(server)).body)
        assert loaded["can_attach"] and loaded["prompt"]["id"] == prompt["id"]
        [first] = loaded["attached_files"]
        assert first["name"] == "input/тест_1.txt" and first["chars"] == len(body)

        duplicate = server.handle("POST", "/attach", attachment_headers, b"second")
        assert duplicate.status == 200
        [first, second] = front.poll(0, timeout=0)["attached_files"]
        assert second["name"] == "input/тест_1-2.txt"
        detach = server.handle(
            "POST", "/detach", {**attachment_headers, "X-Desman-File": first["id"]}
        )
        assert detach.status == 200 and front.poll(0, timeout=0)["attached_files"] == [second]
        original = front.current
        assert original is not None
        assert front.action(ui.NewSessionAction(type="new_session", temporary=False))
        new_prompt = next_prompt(front, prompt["id"])
        assert not front.poll(0, timeout=0)["attached_files"]
        stale = server.handle("POST", "/attach", attachment_headers, b"stale")
        assert stale.status == 409
        assert front.action(ui.SessionAction(type="open_session", id=original))
        prompt = next_prompt(front, new_prompt["id"])
        assert front.poll(0, timeout=0)["attached_files"] == [second]
        assert front.answer({"id": prompt["id"], "value": "Read the attachment"})
        next_prompt(front, prompt["id"])
        assert front.runners is not None and front.current is not None
        with front.condition:
            assert front.condition.wait_for(
                lambda: not front.runners or not front.runners.status(front.current or "")["state"],
                timeout=10,
            )
        assert not front.poll(0, timeout=0)["attached_files"]
        with closing(db.open_db(path)) as conn:
            [(task, session_id)] = conn.execute("SELECT id, session_id FROM tasks")
            workspace = Workspace(workspace_root_for(conn), task)
            assert workspace.files() == [second["name"]]
            assert workspace.read(second["name"]) == b"second"
            messages = Sessions(conn).messages(session_id)
            [removed] = [m for m in messages if m.text == "Removed pending file: input/тест_1.txt"]
            assert removed.kind is Kind.EVENT and removed.styled is None
        (plan,) = [request.prompt for request in fake.calls if request.prompt.startswith(PLAN)]
        assert "input/тест_1-2.txt (6 characters)" in plan
    finally:
        front.end()
        thread.join(10)
        assert not thread.is_alive()
