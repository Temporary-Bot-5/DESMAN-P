"""Page-owned worker processes; supervision never uses the chat's connection."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Protocol

import click

from desman import runs
from desman.core import db
from desman.core.store import Store
from desman.i18n import t


class Process(Protocol):
    def poll(self) -> int | None: ...
    def send_signal(self, sig: int) -> None: ...
    def wait(self, timeout: float | None = None) -> int: ...
    def terminate(self) -> None: ...
    def kill(self) -> None: ...


Launcher = Callable[[list[str], dict[str, str], Path], Process]


def launch(argv: list[str], env: dict[str, str], path: Path) -> Process:
    handler = RotatingFileHandler(path, maxBytes=2_000_000, backupCount=2, encoding="utf-8")
    flags = 0
    if sys.platform == "win32":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP
    try:
        process = subprocess.Popen(
            argv,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            creationflags=flags,
            start_new_session=sys.platform != "win32",
        )
    except BaseException:
        handler.close()
        raise

    def copy() -> None:
        assert process.stdout is not None
        try:
            for line in iter(process.stdout.readline, b""):
                handler.emit(
                    logging.LogRecord(
                        "worker",
                        logging.INFO,
                        "",
                        0,
                        line.decode("utf-8", errors="replace").rstrip(),
                        (),
                        None,
                    )
                )
        finally:
            process.stdout.close()
            handler.close()

    threading.Thread(target=copy, daemon=True).start()
    return process


@dataclass
class Slot:
    process: Process | None = None
    started: float = 0
    due: float = 0
    deaths: int = 0
    stopped: bool = False


class Pool:
    def __init__(
        self,
        path: Path,
        config_dir: Path,
        size: int = 3,
        *,
        launcher: Launcher = launch,
        clock: Callable[[], float] = time.monotonic,
        guard: Callable[[], None] | None = None,
    ) -> None:
        if not 1 <= size <= 8:
            raise ValueError("worker count must be 1–8")
        self.path, self.clock, self.launcher = path.resolve(), clock, launcher
        self.directory = self.path.parent / "workers"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.env = {
            **os.environ,
            "DESMAN_DATA_DIR": str(self.path.parent),
            "DESMAN_CONFIG_DIR": str(config_dir.resolve()),
            "PYTHONUNBUFFERED": "1",
        }
        self.argv = [sys.executable, "-m", "desman", "worker", "--db", str(self.path)]
        self.slots = [Slot() for _ in range(size)]
        self.messages: deque[str] = deque()
        self.guard = guard or self._guard
        self.blocked = False
        self.end = threading.Event()
        self.thread: threading.Thread | None = None

    def _guard(self) -> None:
        conn = db.open_db(self.path)
        try:
            runs.guard(Store(conn))
        finally:
            conn.close()

    def tick(self) -> None:
        if self.blocked or self.end.is_set():
            return
        now = self.clock()
        for index, slot in enumerate(self.slots, 1):
            if slot.stopped:
                continue
            path = self.directory / f"worker-{index}.log"
            if slot.process is not None:
                if slot.process.poll() is None:
                    continue
                slot.process = None
                slot.deaths = slot.deaths + 1 if now - slot.started < 60 else 1
                try:
                    self.guard()
                except click.ClickException as exc:
                    self.messages.append(exc.format_message())
                    self.blocked = True
                    return
                if slot.deaths >= 5:
                    slot.stopped = True
                    self.messages.append(
                        t(
                            "Background worker stopped after repeated failures. See {path}.",
                            path=path,
                        )
                    )
                    continue
                slot.due = now + min(60, 2 ** (slot.deaths - 1))
            if now >= slot.due:
                try:
                    self.guard()
                except click.ClickException as exc:
                    self.messages.append(exc.format_message())
                    self.blocked = True
                    return
                try:
                    slot.process = self.launcher(self.argv, self.env, path)
                    slot.started = now
                except OSError:
                    slot.deaths += 1
                    slot.due = now + min(60, 2 ** (slot.deaths - 1))
                    if slot.deaths >= 5:
                        slot.stopped = True
                        self.messages.append(
                            t(
                                "Background worker stopped after repeated failures. See {path}.",
                                path=path,
                            )
                        )

    def start(self) -> None:
        self.tick()

        def supervise() -> None:
            while not self.end.wait(0.2):
                self.tick()

        self.thread = threading.Thread(target=supervise, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.end.set()
        if self.thread is not None:
            self.thread.join()
        processes = [
            s.process for s in self.slots if s.process is not None and s.process.poll() is None
        ]
        stop_signal = signal.SIGINT
        if sys.platform == "win32":
            stop_signal = signal.CTRL_BREAK_EVENT
        for process in processes:
            with suppress(OSError):
                process.send_signal(stop_signal)
        deadline = time.monotonic() + 10
        for process in processes:
            try:
                process.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
