"""Time supplied to models, using the caller's clock when available."""

from collections.abc import Callable
from datetime import UTC, datetime


def today(clock: Callable[[], datetime] | None = None) -> str:
    now = clock() if clock is not None else datetime.now(UTC)
    return f" Today is {now.astimezone(UTC).date().isoformat()}."
