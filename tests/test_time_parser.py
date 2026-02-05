"""
Unit tests for time_parser module
"""

import os
import sys
from datetime import date, datetime, timedelta

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mcp_memory_service.utils.time_parser import extract_time_expression, parse_time_expression


class TestTimeParser:
    """Test time parsing functionality"""

    def test_relative_days(self):
        """Test parsing relative day expressions"""
        # Test "yesterday"
        start_ts, end_ts = parse_time_expression("yesterday")
        assert start_ts is not None
        assert end_ts is not None

        yesterday = date.today() - timedelta(days=1)
        start_dt = datetime.fromtimestamp(start_ts)
        end_dt = datetime.fromtimestamp(end_ts)

        assert start_dt.date() == yesterday
        assert end_dt.date() == yesterday
        assert start_dt.time() == datetime.min.time()
        assert end_dt.time().hour == 23
        assert end_dt.time().minute == 59

        # Test "3 days ago"
        start_ts, end_ts = parse_time_expression("3 days ago")
        three_days_ago = date.today() - timedelta(days=3)
        start_dt = datetime.fromtimestamp(start_ts)
        assert start_dt.date() == three_days_ago

        # Test "today"
        start_ts, end_ts = parse_time_expression("today")
        start_dt = datetime.fromtimestamp(start_ts)
        assert start_dt.date() == date.today()

    def test_relative_weeks(self):
        """Test parsing relative week expressions"""
        start_ts, end_ts = parse_time_expression("2 weeks ago")
        assert start_ts is not None
        assert end_ts is not None

        start_dt = datetime.fromtimestamp(start_ts)
        end_dt = datetime.fromtimestamp(end_ts)

        # Should be a Monday to Sunday range
        assert start_dt.weekday() == 0  # Monday
        assert end_dt.weekday() == 6  # Sunday

        # Should be roughly 2 weeks ago
        days_ago = (date.today() - start_dt.date()).days
        assert 14 <= days_ago <= 20  # Allow some flexibility for week boundaries

    def test_relative_months(self):
        """Test parsing relative month expressions"""
        start_ts, end_ts = parse_time_expression("1 month ago")
        assert start_ts is not None
        assert end_ts is not None

        start_dt = datetime.fromtimestamp(start_ts)
        end_dt = datetime.fromtimestamp(end_ts)

        # Should be first to last day of the month
        assert start_dt.day == 1
        assert (end_dt + timedelta(days=1)).day == 1  # Next day is first of next month

    def test_specific_dates(self):
        """Test parsing specific date formats"""
        # Test MM/DD/YYYY format with unambiguous date
        start_ts, end_ts = parse_time_expression("03/15/2024")
        assert start_ts is not None

        start_dt = datetime.fromtimestamp(start_ts)
        assert start_dt.year == 2024
        assert start_dt.month == 3
        assert start_dt.day == 15

        # Test YYYY-MM-DD format
        start_ts, end_ts = parse_time_expression("2024-06-15")
        assert start_ts is not None
        start_dt = datetime.fromtimestamp(start_ts)
        assert start_dt.date() == date(2024, 6, 15)

    def test_month_names(self):
        """Test parsing month names"""
        current_year = datetime.now().year
        datetime.now().month

        # Test a past month
        start_ts, end_ts = parse_time_expression("january")
        start_dt = datetime.fromtimestamp(start_ts)

        # Month should be January
        assert start_dt.month == 1
        # Year should be either current year or previous year (implementation-dependent)
        assert start_dt.year in (current_year, current_year - 1)

    def test_seasons(self):
        """Test parsing season names"""
        # Test summer
        start_ts, end_ts = parse_time_expression("last summer")
        assert start_ts is not None
        assert end_ts is not None

        start_dt = datetime.fromtimestamp(start_ts)
        end_dt = datetime.fromtimestamp(end_ts)

        # Summer is roughly June 21 to September 22
        assert start_dt.month == 6
        assert end_dt.month == 9

    def test_holidays(self):
        """Test parsing holiday names"""
        # Test Christmas
        start_ts, end_ts = parse_time_expression("christmas")
        assert start_ts is not None

        start_dt = datetime.fromtimestamp(start_ts)
        end_dt = datetime.fromtimestamp(end_ts)

        # Christmas window should include Dec 25 +/- a few days
        assert start_dt.month == 12
        assert 22 <= start_dt.day <= 25
        assert 25 <= end_dt.day <= 28

    def test_time_of_day(self):
        """Test time of day parsing"""
        # Test "yesterday morning"
        start_ts, end_ts = parse_time_expression("yesterday morning")
        start_dt = datetime.fromtimestamp(start_ts)
        end_dt = datetime.fromtimestamp(end_ts)

        yesterday = date.today() - timedelta(days=1)
        assert start_dt.date() == yesterday
        assert 5 <= start_dt.hour <= 6  # Morning starts at 5 AM
        assert 11 <= end_dt.hour <= 12  # Morning ends before noon

    def test_date_ranges(self):
        """Test date range expressions"""
        start_ts, end_ts = parse_time_expression("between january and march")
        assert start_ts is not None
        assert end_ts is not None

        start_dt = datetime.fromtimestamp(start_ts)
        end_dt = datetime.fromtimestamp(end_ts)

        assert start_dt.month == 1
        assert end_dt.month == 3

    def test_quarters(self):
        """Test quarter expressions"""
        start_ts, end_ts = parse_time_expression("first quarter of 2024")
        assert start_ts is not None

        start_dt = datetime.fromtimestamp(start_ts)
        end_dt = datetime.fromtimestamp(end_ts)

        assert start_dt == datetime(2024, 1, 1, 0, 0, 0)
        assert end_dt.year == 2024
        assert end_dt.month == 3
        assert end_dt.day == 31

    def test_extract_time_expression(self):
        """Test extracting time expressions from queries"""
        # Test extraction with semantic content
        cleaned, (start_ts, end_ts) = extract_time_expression("find meetings from last week about project updates")

        assert "meetings" in cleaned
        assert "project updates" in cleaned
        assert "last week" not in cleaned
        assert start_ts is not None
        assert end_ts is not None

        # Test multiple time expressions
        cleaned, (start_ts, end_ts) = extract_time_expression("yesterday in the morning I had coffee")

        assert "coffee" in cleaned
        assert "yesterday" not in cleaned
        assert "in the morning" not in cleaned

    def test_edge_cases(self):
        """Test edge cases and error handling"""
        # Test empty string
        start_ts, end_ts = parse_time_expression("")
        assert start_ts is None
        assert end_ts is None

        # Test invalid date format
        start_ts, end_ts = parse_time_expression("13/32/2024")  # Invalid month and day
        assert start_ts is None
        assert end_ts is None

        # Test nonsense string
        start_ts, end_ts = parse_time_expression("random gibberish text")
        assert start_ts is None
        assert end_ts is None

    def test_this_period_expressions(self):
        """Test 'this X' period expressions"""
        # This week
        start_ts, end_ts = parse_time_expression("this week")
        start_dt = datetime.fromtimestamp(start_ts)
        end_dt = datetime.fromtimestamp(end_ts)

        # Should include today
        today = date.today()
        assert start_dt.date() <= today <= end_dt.date()

        # This month
        start_ts, end_ts = parse_time_expression("this month")
        start_dt = datetime.fromtimestamp(start_ts)
        assert start_dt.month == datetime.now().month
        assert start_dt.year == datetime.now().year

    def test_recent_expressions(self):
        """Test 'recent' and similar expressions"""
        start_ts, end_ts = parse_time_expression("recently")
        assert start_ts is not None
        assert end_ts is not None

        # Should default to last 7 days
        days_diff = (end_ts - start_ts) / (24 * 3600)
        assert 6 <= days_diff <= 8  # Allow for some time variance


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
