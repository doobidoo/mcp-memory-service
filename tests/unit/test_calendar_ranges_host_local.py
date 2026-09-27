"""Calendar expressions and calendar-date deletion must agree, in every host timezone.

Contract: a calendar day means the *host-local* day. The test asserts agreement
within one process per zone, not invariance across zones.
"""

import os
import time as time_mod
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime, time, timedelta

import pytest

from mcp_memory_service.utils.time_parser import parse_boundary, parse_time_expression

ZONES = ["UTC", "Asia/Tokyo", "America/Los_Angeles", "Pacific/Auckland"]


@contextmanager
def process_timezone(name: str) -> Iterator[None]:
    """Temporarily set the process timezone, restoring it (and tzset) on the way out."""
    if not hasattr(time_mod, "tzset"):
        pytest.skip("time.tzset is unavailable on this platform")

    original = os.environ.get("TZ")
    os.environ["TZ"] = name
    time_mod.tzset()
    try:
        yield
    finally:
        if original is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = original
        time_mod.tzset()


@pytest.mark.parametrize("tz", ZONES)
def test_today_is_the_host_local_day(tz):
    with process_timezone(tz):
        start, end = parse_time_expression("today")
        today = date.today()
        assert start == datetime.combine(today, time.min).timestamp()
        assert end == pytest.approx(datetime.combine(today, time.max).timestamp())


@pytest.mark.parametrize("tz", ZONES)
def test_parser_and_delete_path_agree_on_the_same_day(tz):
    """The bounds delete_by_timeframe() builds must match what "yesterday" parses to."""
    with process_timezone(tz):
        yesterday = date.today() - timedelta(days=1)
        parsed_start, parsed_end = parse_time_expression("yesterday")
        # Same construction as storage/mixins/delete.py:delete_by_timeframe
        delete_start = datetime.combine(yesterday, datetime.min.time()).timestamp()
        delete_end = datetime.combine(yesterday, datetime.max.time()).timestamp()
        assert parsed_start == delete_start
        assert parsed_end == pytest.approx(delete_end)


@pytest.mark.parametrize("tz", ZONES)
def test_bare_date_filter_matches_the_local_day_but_datetimes_stay_utc(tz):
    """after="YYYY-MM-DD" is a calendar day (local); a naive datetime is an instant (UTC)."""
    with process_timezone(tz):
        day = date(2026, 9, 4)
        assert parse_boundary("2026-09-04") == datetime.combine(day, time.min).timestamp()
        assert parse_boundary("2026-09-04T00:00:00") == 1788480000.0  # UTC midnight
        assert parse_boundary("2026-09-04T00:00:00+09:00") == 1788447600.0
