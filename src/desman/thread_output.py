"""Scoped stdout/stderr routing without changing another thread's destination."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from threading import RLock, local
from typing import Any

_lock = RLock()


class Routed:
    def __init__(self, fallback: Any) -> None:
        self.fallback = fallback
        self.local = local()
        self.users = 0

    @property
    def target(self) -> Any:
        return getattr(self.local, "target", self.fallback)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.target, name)

    def write(self, text: str) -> int:
        return int(self.target.write(text))

    def flush(self) -> None:
        self.target.flush()


@contextmanager
def redirect(name: str, stream: Any) -> Iterator[None]:
    with _lock:
        current = getattr(sys, name)
        router = current if isinstance(current, Routed) else Routed(current)
        setattr(sys, name, router)
        router.users += 1
        previous = router.target
        router.local.target = stream
    try:
        yield
    finally:
        with _lock:
            router.local.target = previous
            router.users -= 1
            if router.users == 0 and getattr(sys, name) is router:
                setattr(sys, name, router.fallback)


def redirect_stdout(stream: Any) -> AbstractContextManager[None]:
    return redirect("stdout", stream)


def redirect_stderr(stream: Any) -> AbstractContextManager[None]:
    return redirect("stderr", stream)
