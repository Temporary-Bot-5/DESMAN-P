"""Bounded provider waits for model calls made without a task lease."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TypeVar

from desman.llm.base import QuotaExhausted, RateLimited

PROVIDER_WAIT = 15 * 60
T = TypeVar("T")


@dataclass
class Wait:
    sleep: Callable[[float], None]
    clock: Callable[[], datetime]
    announce: Callable[[float], None]
    deadline: datetime
    immediate: int = 0
    limited_until: datetime | None = None


_wait: ContextVar[Wait | None] = ContextVar("provider_wait", default=None)


def enabled() -> bool:
    """The caller holds no lease and allows routing through even slow limit failures."""
    return _wait.get() is not None


@contextmanager
def waiting(
    *,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    announce: Callable[[float], None] = lambda seconds: None,
) -> Iterator[Wait]:
    """Share one wall-clock budget across every call of a lease-free operation."""
    state = Wait(sleep, clock, announce, clock() + timedelta(seconds=PROVIDER_WAIT))
    token = _wait.set(state)
    try:
        yield state
    finally:
        _wait.reset(token)


def retry(call: Callable[[], T]) -> T:
    while True:
        try:
            result = call()
            settings = _wait.get()
            if settings is not None:
                settings.limited_until = None
            return result
        except (RateLimited, QuotaExhausted) as error:
            settings = _wait.get()
            if settings is None or isinstance(error, RateLimited) and not error.automatic_wait:
                raise
            now = settings.clock()
            until = error.paused_until(now)
            settings.limited_until = until
            seconds = (
                error.retry_after
                if isinstance(error, RateLimited)
                else None
                if until is None
                else (until - now).total_seconds()
            )
            remaining = (settings.deadline - now).total_seconds()
            if seconds is None or not math.isfinite(seconds) or remaining <= 0:
                raise
            if seconds <= 0:
                settings.immediate += 1
                if settings.immediate > 3:
                    raise
                continue
            if seconds > remaining:
                raise
            settings.announce(seconds)
            if seconds > (settings.deadline - settings.clock()).total_seconds():
                raise
            settings.sleep(seconds)
