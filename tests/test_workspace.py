"""Tests for the workspace path guard (B5, first part)."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from conftest import FakeClock
from desman.core import workspace as ws_module
from desman.core.coordinator import Coordinator, Review, Success, Verdict, Work
from desman.core.models import Subtask, SubtaskSpec, TaskSpec
from desman.core.states import SubtaskStatus, TaskStatus
from desman.core.store import Store
from desman.core.workspace import (
    PathEscapeError,
    Workspace,
    check_path,
    default_workspace_root,
)

MALICIOUS = [
    "/etc/passwd",
    "//server/share/x",
    "/",
    "..",
    "../x",
    "../../.bashrc",
    "a/../../x",
    "a/..",
    "a/../b",
    "./../x",
    "C:/Windows/x",
    "c:x",
    "C:",
    "..\\x",
    "a\\..\\..\\b",
    "a\x00b",
    "a\nb",
    "a\x1bb",
    "",
    ".",
    "./",
    "//",
    "file.txt:stream",
    "dir/.desman-tmp-1234",
    "x" * 256,
    "я" * 128,  # 256 bytes in UTF-8
    "a/" * 600,
]


@pytest.mark.parametrize("path", MALICIOUS)
def test_malicious_paths_are_refused(path: str) -> None:
    with pytest.raises(PathEscapeError):
        check_path(path)


@pytest.mark.parametrize("path", [None, b"x", Path("x"), 1])
def test_non_string_paths_are_refused(path: Any) -> None:
    with pytest.raises(PathEscapeError, match="string"):
        check_path(path)


@pytest.mark.parametrize(
    ("path", "parts"),
    [
        ("a.txt", ("a.txt",)),
        ("dir/sub/f.json", ("dir", "sub", "f.json")),
        ("./a//b/", ("a", "b")),
        ("~/x", ("~", "x")),  # no home expansion: a directory literally named "~"
        ("...", ("...",)),
        ("..x/y..", ("..x", "y..")),
        ("x" * 255, ("x" * 255,)),
    ],
)
def test_ordinary_paths_are_accepted(path: str, parts: tuple[str, ...]) -> None:
    assert check_path(path) == parts


@pytest.fixture(params=[True, False], ids=["secure", "fallback"])
def secure(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[bool]:
    """Run each file test both with descriptor-relative calls and with the lstat fallback."""
    if request.param and not ws_module.SECURE:
        pytest.skip("no dir_fd support on this platform")
    monkeypatch.setattr(ws_module, "SECURE", request.param)
    yield request.param


@pytest.fixture
def ws(tmp_path: Path, secure: bool) -> Workspace:
    return Workspace(tmp_path / "workspaces", "task1")


@pytest.fixture
def outside(tmp_path: Path) -> Path:
    target = tmp_path / "outside"
    target.mkdir()
    (target / "secret").write_text("keep me")
    return target


def test_write_read_roundtrip(ws: Workspace) -> None:
    path = ws.write("dir/sub/a.txt", "привет")
    assert path == ws.root / "dir" / "sub" / "a.txt"
    assert ws.read_text("dir/sub/a.txt") == "привет"
    ws.write("dir/sub/a.txt", b"\x00\x01")
    assert ws.read("dir/sub/a.txt") == b"\x00\x01"
    ws.write("b.bin", b"")
    assert ws.files() == ["b.bin", "dir/sub/a.txt"]
    assert ws.exists("b.bin") and not ws.exists("nope") and not ws.exists("../x")


def test_written_files_are_private_and_no_temp_files_remain(ws: Workspace) -> None:
    ws.write("a.txt", "x")
    if os.name == "posix":
        assert (ws.root / "a.txt").stat().st_mode & 0o077 == 0
    assert os.listdir(ws.root) == ["a.txt"]


@pytest.mark.parametrize("path", MALICIOUS)
def test_malicious_paths_never_touch_the_disk(ws: Workspace, tmp_path: Path, path: str) -> None:
    before = sorted(p for p in tmp_path.rglob("*"))
    with pytest.raises(PathEscapeError):
        ws.write(path, "pwned")
    with pytest.raises(PathEscapeError):
        ws.read(path)
    with pytest.raises(PathEscapeError):
        ws.resolve(path)
    assert sorted(p for p in tmp_path.rglob("*")) == before


def test_each_task_has_its_own_directory(tmp_path: Path) -> None:
    a = Workspace(tmp_path, "task_a")
    b = Workspace(tmp_path, "task_b")
    a.write("result.txt", "a")
    assert a.root != b.root
    assert b.files() == []
    assert not b.exists("result.txt")


@pytest.mark.parametrize("task_id", ["", ".", "..", "../x", "a/b", "a\\b", "x" * 65, "a b"])
def test_bad_task_ids_are_refused(tmp_path: Path, task_id: str) -> None:
    with pytest.raises(PathEscapeError):
        Workspace(tmp_path, task_id)
    assert list(tmp_path.iterdir()) == []


def test_task_directory_that_is_a_symlink_is_refused(tmp_path: Path, outside: Path) -> None:
    (tmp_path / "ws").mkdir()
    (tmp_path / "ws" / "task1").symlink_to(outside)
    with pytest.raises(PathEscapeError):
        Workspace(tmp_path / "ws", "task1")


def test_symlinked_directory_pointing_outside_is_not_followed(ws: Workspace, outside: Path) -> None:
    (ws.root / "link").symlink_to(outside)
    with pytest.raises(PathEscapeError, match="symlink"):
        ws.write("link/new", "pwned")
    with pytest.raises(PathEscapeError, match="symlink"):
        ws.write("link/secret", "pwned")
    with pytest.raises(PathEscapeError, match="symlink"):
        ws.read("link/secret")
    with pytest.raises(PathEscapeError, match="symlink"):
        ws.resolve("link/secret")
    assert sorted(os.listdir(outside)) == ["secret"]
    assert (outside / "secret").read_text() == "keep me"
    assert ws.files() == []


def test_symlink_deeper_in_the_path_is_not_followed(ws: Workspace, outside: Path) -> None:
    ws.write("a/b/keep.txt", "ok")
    (ws.root / "a" / "b" / "c").symlink_to(outside)
    with pytest.raises(PathEscapeError, match="symlink"):
        ws.write("a/b/c/d/e.txt", "pwned")
    assert sorted(os.listdir(outside)) == ["secret"]


def test_symlink_pointing_inside_is_not_followed_either(ws: Workspace) -> None:
    ws.write("real/f.txt", "x")
    (ws.root / "alias").symlink_to(ws.root / "real")
    with pytest.raises(PathEscapeError, match="symlink"):
        ws.read("alias/f.txt")
    assert ws.files() == ["real/f.txt"]


def test_symlinked_file_is_not_read(ws: Workspace, outside: Path) -> None:
    (ws.root / "evil").symlink_to(outside / "secret")
    with pytest.raises(PathEscapeError, match="symlink"):
        ws.read("evil")
    assert not ws.exists("evil")


def test_writing_over_a_symlinked_file_replaces_the_link_not_the_target(
    ws: Workspace, outside: Path, secure: bool
) -> None:
    (ws.root / "evil").symlink_to(outside / "secret")
    if secure:
        ws.write("evil", "mine")
        assert not (ws.root / "evil").is_symlink()
        assert ws.read_text("evil") == "mine"
    else:
        with pytest.raises(PathEscapeError, match="symlink"):
            ws.write("evil", "mine")
    assert (outside / "secret").read_text() == "keep me"


def test_writing_over_a_hard_link_does_not_write_through(ws: Workspace, outside: Path) -> None:
    os.link(outside / "secret", ws.root / "hard")
    ws.write("hard", "mine")
    assert ws.read_text("hard") == "mine"
    assert (outside / "secret").read_text() == "keep me"


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="no FIFOs here")
def test_reading_a_fifo_does_not_hang(ws: Workspace, secure: bool) -> None:
    os.mkfifo(ws.root / "pipe")
    with pytest.raises((PathEscapeError, FileNotFoundError)):
        ws.read("pipe")
    assert ws.files() == []


def test_ordinary_os_errors_are_not_disguised(ws: Workspace) -> None:
    ws.write("file.txt", "x")
    ws.write("dir/x", "x")
    with pytest.raises(FileNotFoundError):
        ws.read("missing.txt")
    with pytest.raises(NotADirectoryError):
        ws.write("file.txt/below", "x")
    with pytest.raises(IsADirectoryError):
        ws.write("dir", "x")
    with pytest.raises((IsADirectoryError, FileNotFoundError, PathEscapeError)):
        ws.read("dir")
    assert ws.files() == ["dir/x", "file.txt"]
    assert sorted(os.listdir(ws.root)) == ["dir", "file.txt"]  # no temp file left behind


def test_files_skips_links_and_temp_files(ws: Workspace, outside: Path) -> None:
    ws.write("a.txt", "x")
    (ws.root / "dirlink").symlink_to(outside)
    (ws.root / "filelink").symlink_to(outside / "secret")
    (ws.root / ".desman-tmp-dead").write_text("left by a crash")
    assert ws.files() == ["a.txt"]


def test_default_root_is_next_to_the_database(tmp_path: Path) -> None:
    assert default_workspace_root() == tmp_path / "data" / "workspaces"


# ── Through the coordinator ──────────────────────────────────────────────────


def accept(subtask: Subtask, artifact: dict[str, Any] | None) -> Review:
    return Review(verdict=Verdict.ACCEPT)


def one_step() -> TaskSpec:
    return TaskSpec(goal="g", subtasks=[SubtaskSpec(description="a", acceptance_criteria="ok")])


def test_executor_writes_into_its_task_directory(
    store: Store, clock: FakeClock, tmp_path: Path
) -> None:
    def executor(work: Work) -> Success:
        work.workspace.write("out/report.md", "# done")
        return Success(content={"files": work.workspace.files()})

    task_id = store.create_task(one_step())
    other_id = store.create_task(one_step())
    root = tmp_path / "ws"
    result = Coordinator(store, executor, accept, workspaces=root, sleep=clock.sleep).run(task_id)
    assert result.status is TaskStatus.DONE
    assert (root / task_id / "out" / "report.md").read_text() == "# done"
    assert not (root / other_id).exists()
    [sub] = store.list_subtasks(task_id)
    assert store.latest_artifact(sub.id) == {"files": ["out/report.md"]}


def test_coordinator_defaults_to_the_data_directory(store: Store, clock: FakeClock) -> None:
    def executor(work: Work) -> Success:
        work.workspace.write("x.txt", "x")
        return Success(content={})

    task_id = store.create_task(one_step())
    Coordinator(store, executor, accept, sleep=clock.sleep).run(task_id)
    assert (default_workspace_root() / task_id / "x.txt").is_file()


def test_forbidden_path_is_a_quality_failure(
    store: Store, clock: FakeClock, tmp_path: Path
) -> None:
    # The model asked for a bad path: a different attempt may do better, so it is retried.
    paths = iter(["../../.bashrc", "notes.txt"])

    def executor(work: Work) -> Success:
        work.workspace.write(next(paths), "x")
        return Success(content={})

    task_id = store.create_task(one_step())
    root = tmp_path / "ws"
    result = Coordinator(store, executor, accept, workspaces=root, sleep=clock.sleep).run(task_id)
    assert result.status is TaskStatus.DONE
    [sub] = store.list_subtasks(task_id)
    assert sub.status is SubtaskStatus.DONE
    assert sub.attempts.quality == 1
    assert any("path refused" in event or "quality" in event for event in result.events)
    assert not (tmp_path / ".bashrc").exists()
    assert (root / task_id / "notes.txt").is_file()
