"""Bounded provider waits for model calls made without a task lease."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import TypeVar

from desman import ui
from desman.i18n import t
from desman.llm.base import QuotaExhausted, RateLimited

PROVIDER_WAIT = 15 * 60
T = TypeVar("T")
_wait: ContextVar[tuple[Callable[[float], None], bool, Callable[[], datetime]] | None] = ContextVar(
    "provider_wait", default=None
)


def enabled() -> bool:
    """The caller holds no lease and allows routing through even slow limit failures."""
    return _wait.get() is not None


def announce(seconds: float, as_json: bool = False) -> None:
    text = t("Waiting {seconds} s for a provider limit…", seconds=math.ceil(seconds))
    if as_json:
        ui.warn(text, stderr=True)
    else:
        ui.note(text)


@contextmanager
def waiting(
    *,
    sleep: Callable[[float], None] = time.sleep,
    as_json: bool = False,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> Iterator[None]:
    """Enable waiting only around calls that hold no execution/review lease."""
    token = _wait.set((sleep, as_json, clock))
    try:
        yield
    finally:
        _wait.reset(token)


def retry(call: Callable[[], T]) -> T:
    remaining = float(PROVIDER_WAIT)
    while True:
        try:
            return call()
        except (RateLimited, QuotaExhausted) as error:
            if isinstance(error, RateLimited) and not error.automatic_wait:
                raise
            settings = _wait.get()
            now = datetime.now(UTC) if settings is None else settings[2]()
            until = error.paused_until(now)
            seconds = (
                error.retry_after
                if isinstance(error, RateLimited)
                else None
                if until is None
                else (until - now).total_seconds()
            )
            if (
                settings is None
                or seconds is None
                or not math.isfinite(seconds)
                or seconds <= 0
                or seconds > remaining
            ):
                if settings is not None and until is not None:
                    ui.warn(
                        t("Provider limit until {until}.", until=until.isoformat()),
                        stderr=settings[1],
                    )
                raise
            sleep, as_json, _clock = settings
            announce(seconds, as_json)
            sleep(seconds)
            remaining -= seconds
