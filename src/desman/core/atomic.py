"""Atomic file replacement and bounded retries for Windows sharing violations."""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

_T = TypeVar("_T")


def retry_sharing(
    operation: Callable[[], _T], *, sleep: Callable[[float], None] | None = None
) -> _T:
    """Allow a concurrent open or replacement to finish before trying again."""
    if sys.platform == "win32":
        wait = time.sleep if sleep is None else sleep
        for _ in range(40):
            try:
                return operation()
            except PermissionError as error:
                if getattr(error, "winerror", None) not in (32, 33, 5):
                    raise
                wait(0.05)
    return operation()


def read_text(path: Path, *, sleep: Callable[[float], None] | None = None) -> str:
    """Read UTF-8 text, retrying Windows sharing errors for up to two seconds."""
    return retry_sharing(lambda: path.read_text(encoding="utf-8"), sleep=sleep)


def replace_file(source: str | Path, target: str | Path) -> None:
    """Retry Windows sharing violations for at most two seconds; never unlink first."""
    retry_sharing(lambda: os.replace(source, target))
