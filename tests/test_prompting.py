"""The date shown to models is English and follows the UTC calendar."""

from datetime import UTC, datetime, timedelta, timezone

import pytest

from desman.prompting import today


@pytest.mark.parametrize(
    "day, weekday",
    [
        (16, "Monday"),
        (17, "Tuesday"),
        (18, "Wednesday"),
        (19, "Thursday"),
        (20, "Friday"),
        (21, "Saturday"),
        (22, "Sunday"),
    ],
)
def test_today_weekday(day: int, weekday: str) -> None:
    assert today(lambda: datetime(2026, 11, day, tzinfo=UTC)) == (
        f" Today is {weekday}, 2026-11-{day} (UTC)."
    )


def test_today_uses_utc_date() -> None:
    assert today(lambda: datetime(2026, 11, 17, 1, tzinfo=timezone(timedelta(hours=2)))) == (
        " Today is Monday, 2026-11-16 (UTC)."
    )
