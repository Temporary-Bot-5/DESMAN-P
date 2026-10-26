"""Per-task workspace and the path guard every file operation goes through (B5, first part).

File names come from LLM output, so every path is hostile until checked. The old prototype
wrote `files: {"/etc/x": ...}` or `../../.bashrc` wherever it was told. Here:

- each task gets its own directory, `<root>/<task_id>`, and nothing outside it is reachable;
- a path must be relative, `/`-separated and portable: no absolute paths, drive letters,
  backslashes, `..`, `:` or control characters (`check_path`);
- symlinks are never followed, not even ones that point back inside: the sandbox (3.3) will let
  generated code create links, and a link that is harmless now can be swapped later;
- on POSIX every component is opened relative to its parent's descriptor with `O_NOFOLLOW`, so
  there is no window between checking a path and using it. Writes go to a temporary file that is
  renamed into place: a crash never leaves half a file, and a hard link is replaced, not written
  through.

Elsewhere (Windows) the same rules are checked with `lstat` before each use; that leaves a small
race, acceptable until the sandbox lands.
"""

from __future__ import annotations

import os
import re
import secrets
import stat
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Final

from desman.core import db

WORKSPACES_DIR: Final = "workspaces"
_TASK_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")
_DRIVE = re.compile(r"[A-Za-z]:")
_TMP_PREFIX: Final = ".desman-tmp-"
MAX_NAME_BYTES: Final = 255  # common file system limits, so an over-long name is the caller's
MAX_PATH_BYTES: Final = 1024  # mistake (quality), not an OSError mistaken for an outage
_NOFOLLOW: Final = getattr(os, "O_NOFOLLOW", 0)
_DIRECTORY: Final = getattr(os, "O_DIRECTORY", 0)
_CLOEXEC: Final = getattr(os, "O_CLOEXEC", 0)
_BINARY: Final = getattr(os, "O_BINARY", 0)
_NONBLOCK: Final = getattr(os, "O_NONBLOCK", 0)

# Descriptor-relative calls make the guard race-free; without them we fall back to lstat checks.
SECURE: bool = all(f in os.supports_dir_fd for f in (os.open, os.mkdir, os.rename, os.unlink))


class PathEscapeError(ValueError):
    """A path that would leave the workspace or is not allowed in it."""


def default_workspace_root() -> Path:
    """Next to the database: $DESMAN_DATA_DIR/workspaces or ~/.local/share/desman/workspaces."""
    return db.default_db_path().parent / WORKSPACES_DIR


def check_path(path: str) -> tuple[str, ...]:
    """Validate a workspace-relative path and return its components (`.` and empty dropped)."""
    if not isinstance(path, str):
        raise PathEscapeError(f"path must be a string, not {type(path).__name__}")
    if any(ord(ch) < 32 or ch == "\x7f" for ch in path):
        raise PathEscapeError(f"control character in path {path!r}")
    if "\\" in path:
        raise PathEscapeError(f"backslash in path {path!r}: use '/'")
    if path.startswith("/") or _DRIVE.match(path):
        raise PathEscapeError(f"absolute path {path!r}: paths are relative to the workspace")
    parts = tuple(part for part in path.split("/") if part not in ("", "."))
    if not parts:
        raise PathEscapeError(f"empty path {path!r}")
    if len(path.encode()) > MAX_PATH_BYTES:
        raise PathEscapeError(f"path longer than {MAX_PATH_BYTES} bytes")
    for part in parts:
        if part == "..":
            raise PathEscapeError(f"'..' in path {path!r}")
        if ":" in part:
            raise PathEscapeError(f"':' in path {path!r}")
        if len(part.encode()) > MAX_NAME_BYTES:
            raise PathEscapeError(f"name longer than {MAX_NAME_BYTES} bytes in path {path[:80]!r}…")
        if part.startswith(_TMP_PREFIX):
            raise PathEscapeError(f"reserved name in path {path!r}")
    return parts


class Workspace:
    """The directory of one task. All file access by agents goes through it."""

    def __init__(self, root: Path, task_id: str) -> None:
        if not _TASK_ID.fullmatch(task_id):
            raise PathEscapeError(f"bad task id {task_id!r}")
        self.task_id = task_id
        self.root = root.resolve() / task_id
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.root.is_symlink() or not self.root.is_dir():
            raise PathEscapeError(f"workspace {self.root} is not a plain directory")

    def resolve(self, path: str) -> Path:
        """Absolute location of `path`, refusing anything that is or crosses a symlink."""
        parts = check_path(path)
        self._check_no_links(parts)
        return self.root.joinpath(*parts)

    def write(self, path: str, data: bytes | str) -> Path:
        """Create or replace a file, making parent directories as needed."""
        payload = data.encode() if isinstance(data, str) else data
        parts = check_path(path)
        if not SECURE:
            return self._write_fallback(parts, payload)
        with self._parent(parts, create=True) as parent:
            tmp = f"{_TMP_PREFIX}{secrets.token_hex(8)}"
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW | _CLOEXEC | _BINARY
            fd = os.open(tmp, flags, 0o600, dir_fd=parent)
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(payload)
                    f.flush()
                    os.fsync(f.fileno())
                # rename() replaces the entry itself: a symlink or hard link there is not followed.
                os.rename(tmp, parts[-1], src_dir_fd=parent, dst_dir_fd=parent)
            except BaseException:
                with suppress(FileNotFoundError):
                    os.unlink(tmp, dir_fd=parent)
                raise
        return self.root.joinpath(*parts)

    def read(self, path: str) -> bytes:
        parts = check_path(path)
        if not SECURE:
            self._check_no_links(parts)
            target = self.root.joinpath(*parts)
            if not target.is_file():
                raise FileNotFoundError(f"no file {path!r} in workspace")
            return target.read_bytes()
        with self._parent(parts, create=False) as parent:
            # O_NONBLOCK: a FIFO planted under this name must not hang the open.
            flags = os.O_RDONLY | _NOFOLLOW | _CLOEXEC | _BINARY | _NONBLOCK
            fd = _open(parts[-1], flags, parent, path)
            with os.fdopen(fd, "rb") as f:
                if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):
                    raise PathEscapeError(f"{path!r} is not a regular file")
                return f.read()

    def read_text(self, path: str) -> str:
        return self.read(path).decode()

    def exists(self, path: str) -> bool:
        try:
            return self.resolve(path).is_file()
        except (PathEscapeError, OSError):
            return False

    def files(self) -> list[str]:
        """Regular files in the workspace, as sorted `/`-separated relative paths."""
        found = []
        for dirpath, _dirnames, filenames in os.walk(self.root, followlinks=False):
            base = Path(dirpath)  # followlinks=False: linked directories are not entered
            for name in filenames:
                full = base / name
                if name.startswith(_TMP_PREFIX) or full.is_symlink() or not full.is_file():
                    continue
                found.append(full.relative_to(self.root).as_posix())
        return sorted(found)

    @contextmanager
    def _parent(self, parts: tuple[str, ...], *, create: bool) -> Iterator[int]:
        """Descriptor of the directory holding `parts[-1]`, walked without following links."""
        dir_flags = os.O_RDONLY | _DIRECTORY | _NOFOLLOW | _CLOEXEC
        fd = os.open(self.root, dir_flags)
        try:
            for i, part in enumerate(parts[:-1]):
                if create:
                    with suppress(FileExistsError):
                        os.mkdir(part, 0o700, dir_fd=fd)
                child = _open(part, dir_flags, fd, "/".join(parts[: i + 1]))
                os.close(fd)
                fd = child
            yield fd
        finally:
            os.close(fd)

    def _check_no_links(self, parts: tuple[str, ...]) -> None:
        current = self.root
        for i, part in enumerate(parts):
            current = current / part
            try:
                mode = current.lstat().st_mode
            except FileNotFoundError:
                return  # nothing below a missing component can be a link
            if stat.S_ISLNK(mode):
                raise PathEscapeError(f"symlink in path {'/'.join(parts[: i + 1])!r}")
            if i < len(parts) - 1 and not stat.S_ISDIR(mode):
                raise NotADirectoryError(f"{'/'.join(parts[: i + 1])!r} is not a directory")

    def _write_fallback(self, parts: tuple[str, ...], payload: bytes) -> Path:
        self._check_no_links(parts)
        target = self.root.joinpath(*parts)
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._check_no_links(parts)  # again, now that the parents exist
        tmp = target.parent / f"{_TMP_PREFIX}{secrets.token_hex(8)}"
        try:
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _BINARY, 0o600)
            with os.fdopen(fd, "wb") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, target)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        return target


def _open(name: str, flags: int, dir_fd: int, shown: str) -> int:
    try:
        return os.open(name, flags, dir_fd=dir_fd)
    except OSError as e:
        # O_NOFOLLOW on a link gives ELOOP; O_DIRECTORY on a link to a directory gives ENOTDIR.
        if _is_link(name, dir_fd):
            raise PathEscapeError(f"symlink in path {shown!r}") from e
        raise


def _is_link(name: str, dir_fd: int) -> bool:
    try:
        return stat.S_ISLNK(os.stat(name, dir_fd=dir_fd, follow_symlinks=False).st_mode)
    except OSError:
        return False
