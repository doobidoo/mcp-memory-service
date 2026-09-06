"""Timezone invariants for natural-language time ranges."""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime, timezone

import pytest

from mcp_memory_service.utils.time_parser import (
    get_time_of_day_range,
    parse_time_expression,
)


@contextmanager
def process_timezone(name: str) -> Iterator[None]:
    """Temporarily set the process timezone for timestamp regression coverage."""
    if not hasattr(time, "tzset"):
        pytest.skip("time.tzset is unavailable on this platform")

    original = os.environ.get("TZ")
    os.environ["TZ"] = name
    time.tzset()
    try:
        yield
    finally:
        if original is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = original
        time.tzset()


@pytest.mark.parametrize(
    "expression",
    [
        "2024-06-15",
        "03/15/2024",
        "first half of 2024",
        "third quarter of 2024",
        "today",
        "yesterday",
        "last week",
        "this month",
    ],
)
def test_calendar_ranges_are_independent_of_host_timezone(expression: str) -> None:
    results = []
    for timezone_name in ("Asia/Tokyo", "Europe/Berlin", "America/Los_Angeles"):
        with process_timezone(timezone_name):
            results.append(parse_time_expression(expression))

    assert results[0] == results[1] == results[2]


def test_explicit_day_uses_utc_boundaries() -> None:
    with process_timezone("America/Los_Angeles"):
        start, end = parse_time_expression("2024-06-15")

    assert start == datetime(2024, 6, 15, tzinfo=timezone.utc).timestamp()
    assert (
        end
        == datetime(2024, 6, 15, 23, 59, 59, 999999, tzinfo=timezone.utc).timestamp()
    )


def test_time_of_day_uses_utc_boundaries() -> None:
    expected = (
        datetime(2024, 6, 15, 5, tzinfo=timezone.utc).timestamp(),
        datetime(2024, 6, 15, 11, 59, 59, tzinfo=timezone.utc).timestamp(),
    )

    for timezone_name in ("Asia/Tokyo", "America/Los_Angeles"):
        with process_timezone(timezone_name):
            assert get_time_of_day_range(date(2024, 6, 15), "morning") == expected
