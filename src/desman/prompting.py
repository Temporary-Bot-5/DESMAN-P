"""Time supplied to models, using the caller's clock when available."""

from collections.abc import Callable
from datetime import UTC, datetime


def today(clock: Callable[[], datetime] | None = None) -> str:
    now = clock() if clock is not None else datetime.now(UTC)
    date = now.astimezone(UTC).date()
    weekdays = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
    return f" Today is {weekdays[date.weekday()]}, {date.isoformat()} (UTC)."
