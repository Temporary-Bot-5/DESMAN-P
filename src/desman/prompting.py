"""Time supplied to models, using the caller's clock when available."""

from collections.abc import Callable
from datetime import UTC, datetime, tzinfo


def today(clock: Callable[[], datetime] | None = None, *, tz: tzinfo | None = None) -> str:
    now = clock() if clock is not None else datetime.now(UTC)
    local = now.astimezone(tz)
    date = local.date()
    offset = local.strftime("%z")
    offset = f"{offset[:3]}:{offset[3:]}"
    weekdays = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
    return f" Today is {weekdays[date.weekday()]}, {date.isoformat()} (local time, UTC{offset})."
