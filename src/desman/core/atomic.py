"""Atomic file replacement, allowing Windows readers to release the destination."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path


def replace_file(source: str | Path, target: str | Path) -> None:
    """Retry Windows sharing violations for at most two seconds; never unlink first."""
    for attempt in range(41):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if sys.platform != "win32" or attempt == 40:
                raise
            time.sleep(0.05)
