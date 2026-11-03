"""Activity breakdown carries machine-readable periods the dashboard can localize (#1819).

The English ``label`` stays for existing API clients. The dashboard renders
weekdays and hours with ``Intl`` and week labels through the catalog, so the
response needs the weekday, hour and ISO week as numbers.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from mcp_memory_service.web.api.analytics import calculate_activity_time_ranges

STATIC = Path(__file__).resolve().parents[3] / "src" / "mcp_memory_service" / "web" / "static"

# 2026-11-02 is a Monday; 14:30 UTC.
MONDAY_AFTERNOON = datetime(2026, 11, 2, 14, 30, tzinfo=timezone.utc).timestamp()


def test_hourly_items_carry_the_hour():
    breakdown, _, _ = calculate_activity_time_ranges([MONDAY_AFTERNOON], "hourly")
    assert [item.hour for item in breakdown] == list(range(24))
    assert breakdown[14].count == 1
    assert breakdown[14].label == "14:00"


def test_daily_items_carry_the_weekday_and_keep_the_english_label():
    breakdown, _, _ = calculate_activity_time_ranges([MONDAY_AFTERNOON], "daily")
    assert [item.weekday for item in breakdown] == list(range(7))
    assert breakdown[0].label == "Monday"
    assert breakdown[0].count == 1


def test_weekly_items_carry_iso_year_and_week_matching_the_label():
    breakdown, _, _ = calculate_activity_time_ranges([MONDAY_AFTERNOON], "weekly")
    assert len(breakdown) == 12
    for item in breakdown:
        assert item.label == f"Week {item.iso_week} ({item.iso_year})"


def test_other_period_fields_stay_unset():
    hourly, _, _ = calculate_activity_time_ranges([MONDAY_AFTERNOON], "hourly")
    daily, _, _ = calculate_activity_time_ranges([MONDAY_AFTERNOON], "daily")
    assert hourly[0].weekday is None and hourly[0].iso_week is None
    assert daily[0].hour is None and daily[0].iso_year is None


def _catalogs():
    return {p.name: json.loads(p.read_text(encoding="utf-8")) for p in sorted((STATIC / "i18n").glob("*.json"))}


def test_week_label_key_exists_in_every_catalog_with_both_tokens():
    for name, catalog in _catalogs().items():
        value = catalog.get("analytics.weekLabel")
        assert value, f"{name} lacks analytics.weekLabel"
        assert set(re.findall(r"\{(\w+)\}", value)) == {"week", "year"}, name


def test_dashboard_formats_activity_labels_in_the_selected_language():
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "analytics.weekLabel" in js
    assert "peak_periods" in js
    # Month names follow the dashboard language, not the browser locale.
    assert "toLocaleString('default'" not in js
