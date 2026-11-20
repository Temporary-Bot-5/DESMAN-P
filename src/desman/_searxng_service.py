"""Entry point executed by the SearXNG virtual environment."""

from __future__ import annotations

import os
import runpy
import sys
import threading
import time
from pathlib import Path


def main(folder: Path, token: str) -> None:
    # The venv has SearXNG's dependencies only. Load DESMAN's stdlib-only lock directly.
    lock = runpy.run_path(str(Path(__file__).parent / "core/filelock.py"))["lock"]
    activation = folder / "activation"

    def active() -> bool:
        for _ in range(3):
            try:
                return activation.read_text(encoding="ascii") == token
            except OSError:
                time.sleep(0.1)
        return False

    def watch() -> None:
        try:
            while active():
                time.sleep(0.1)
        finally:
            os._exit(0)

    with (folder / "service.lock").open("a+b") as handle:
        lock(handle.fileno())
        if active():
            threading.Thread(target=watch, daemon=True).start()
            sys.argv = [sys.argv[0]]
            runpy.run_module("searx.webapp", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2])
