"""Time supplied to models, using the caller's clock when available."""

from collections.abc import Callable
from datetime import UTC, datetime, tzinfo


def today(clock: Callable[[], datetime] | None = None, *, tz: tzinfo | None = None) -> str:
    now = clock() if clock is not None else datetime.now(UTC)
    date = now.astimezone(tz).date()
    # No UTC offset: it would tell every provider roughly where the person lives.
    label = "UTC" if tz is UTC else "local date"
    weekdays = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
    return f" Today is {weekdays[date.weekday()]}, {date.isoformat()} ({label})."
