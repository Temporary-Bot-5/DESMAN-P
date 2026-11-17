"""The date shown to models is English and follows the requested local calendar."""

from datetime import UTC, datetime, timedelta, timezone, tzinfo

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
    assert today(lambda: datetime(2026, 11, day, tzinfo=UTC), tz=UTC) == (
        f" Today is {weekday}, 2026-11-{day} (UTC)."
    )


@pytest.mark.parametrize(
    "instant, zone, expected",
    [
        (
            datetime(2026, 11, 16, 22, tzinfo=UTC),
            timezone(timedelta(hours=3)),
            "Tuesday, 2026-11-17 (local date)",
        ),
        (
            datetime(2026, 11, 17, 2, tzinfo=UTC),
            timezone(timedelta(hours=-5)),
            "Monday, 2026-11-16 (local date)",
        ),
        (
            datetime(2026, 11, 17, 1, tzinfo=timezone(timedelta(hours=2))),
            UTC,
            "Monday, 2026-11-16 (UTC)",
        ),
        (
            datetime(2026, 11, 17, 1, tzinfo=UTC),
            timezone(timedelta(hours=5, minutes=30)),
            "Tuesday, 2026-11-17 (local date)",
        ),
    ],
)
def test_today_uses_local_date(instant: datetime, zone: tzinfo, expected: str) -> None:
    assert today(lambda: instant, tz=zone) == f" Today is {expected}."


def test_today_stays_stable_within_a_local_day() -> None:
    zone = timezone(timedelta(hours=3))
    assert today(lambda: datetime(2026, 11, 17, 1, 2, tzinfo=UTC), tz=zone) == today(
        lambda: datetime(2026, 11, 17, 19, 59, tzinfo=UTC), tz=zone
    )


def test_no_utc_offset_reaches_a_model() -> None:
    text = today(
        lambda: datetime(2026, 11, 17, 1, tzinfo=UTC), tz=timezone(timedelta(hours=5, minutes=30))
    )
    assert "+05" not in text and "UTC" not in text


def test_network_crew_tells_models_the_utc_date() -> None:
    import inspect

    from desman import runs

    source = inspect.getsource(runs.network_crew)
    assert source.count("tz=UTC") == 3
