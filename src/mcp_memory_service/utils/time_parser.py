# Copyright 2024 Heinrich Krupp
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Natural language time expression parser for MCP Memory Service.

This module provides utilities to parse and understand various time expressions
for retrieving memories based on when they were stored.
"""
import re
import logging
from datetime import datetime, timedelta, date, time, timezone
from typing import Tuple, Optional, Dict

from ..compat import _sanitize_log_value

logger = logging.getLogger(__name__)

# Named time periods and their approximate date ranges
NAMED_PERIODS = {
    # Holidays (US/Western-centric, would need localization for global use)
    "christmas": {"month": 12, "day": 25, "window": 3},
    "new year": {"month": 1, "day": 1, "window": 3},
    "valentine": {"month": 2, "day": 14, "window": 1},
    "halloween": {"month": 10, "day": 31, "window": 3},
    "thanksgiving": {"month": 11, "day": -1, "window": 3},  # -1 means fourth Thursday
    
    # Seasons (Northern Hemisphere)
    "spring": {"start_month": 3, "start_day": 20, "end_month": 6, "end_day": 20},
    "summer": {"start_month": 6, "start_day": 21, "end_month": 9, "end_day": 22},
    "fall": {"start_month": 9, "start_day": 23, "end_month": 12, "end_day": 20},
    "autumn": {"start_month": 9, "start_day": 23, "end_month": 12, "end_day": 20},
    "winter": {"start_month": 12, "start_day": 21, "end_month": 3, "end_day": 19},
}

# Time of day mappings (24-hour format)
TIME_OF_DAY = {
    "morning": (5, 11),    # 5:00 AM - 11:59 AM
    "noon": (12, 12),      # 12:00 PM
    "afternoon": (13, 17), # 1:00 PM - 5:59 PM
    "evening": (18, 21),   # 6:00 PM - 9:59 PM
    "night": (22, 4),      # 10:00 PM - 4:59 AM (wraps around midnight)
    "midnight": (0, 0),    # 12:00 AM
}

# Regular expressions for various time patterns.
# Note: all patterns are applied to input that is pre-truncated to _MAX_TIME_QUERY_LEN
# characters, which prevents ReDoS on adversarial inputs.
PATTERNS = {
    # Use \d{1,4} instead of \d+ to bound digit repetition and avoid polynomial backtracking
    "relative_days": re.compile(r'(?:(\d{1,4})\s+days?\s+ago)|(?:yesterday)|(?:today)'),
    "relative_weeks": re.compile(r'(\d{1,4})\s+weeks?\s+ago'),
    "relative_months": re.compile(r'(\d{1,4})\s+months?\s+ago'),
    "relative_years": re.compile(r'(\d{1,4})\s+years?\s+ago'),
    "last_n_periods": re.compile(r'last\s+(\d{1,4})\s+(days?|weeks?|months?|years?)'),
    "last_period": re.compile(r'last\s+(day|week|month|year|summer|spring|winter|fall|autumn)'),
    "this_period": re.compile(r'this\s+(day|week|month|year|summer|spring|winter|fall|autumn)'),
    "month_name": re.compile(r'(january|february|march|april|may|june|july|august|september|october|november|december)'),
    # Use \S[^&]*? instead of .+? to prevent backtracking on whitespace-only strings
    "date_range": re.compile(r'between\s+(\S[^&]{0,50}?)\s+and\s+(\S[^&]{0,50}?)(?=\s|$)'),
    "time_of_day": re.compile(r'(morning|afternoon|evening|night|noon|midnight)'),
    "recent": re.compile(r'recent|lately|recently'),
    "specific_date": re.compile(r'(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?'),
    "full_date": re.compile(r'(\d{4})-(\d{1,2})-(\d{1,2})'),
    "named_period": re.compile(r'(spring|summer|winter|fall|autumn|christmas|new\s*year|valentine|halloween|thanksgiving|spring\s*break|summer\s*break|winter\s*break)'),    "half_year": re.compile(r'(first|second)\s+half\s+of\s+(\d{4})'),
    "quarter": re.compile(r'(first|second|third|fourth|1st|2nd|3rd|4th)\s+quarter(?:\s+of\s+(\d{4}))?'),
}

_MAX_TIME_QUERY_LEN = 500  # guard against ReDoS on user-supplied input

def _calculate_season_date_range(
    period: str,
    season_info: Dict[str, int],
    base_year: int,
    current_month: Optional[int] = None
) -> Tuple[datetime, datetime]:
    """
    Calculate start and end dates for a season, handling winter's year boundary.

    Args:
        period: Season name ("winter", "spring", "summer", "fall"/"autumn")
        season_info: Dictionary with start_month, start_day, end_month, end_day
        base_year: The year to use as reference for calculation
        current_month: Current month (1-12) for context-aware winter calculation (optional)

    Returns:
        Tuple of (start_datetime, end_datetime) for the season
    """
    if period == "winter":
        # Winter spans year boundary (Dec -> Mar)
        # Determine start year based on current month context
        if current_month is not None and current_month <= 3:
            # We're in Jan-Mar, so winter started the previous year
            start_year = base_year - 1
            end_year = base_year
        else:
            # We're in any other month, winter starts this year
            start_year = base_year
            end_year = base_year + 1

        start_dt = datetime(start_year, season_info["start_month"], season_info["start_day"])
        end_dt = datetime(end_year, season_info["end_month"], season_info["end_day"], 23, 59, 59)
    else:
        # All other seasons fall within a single calendar year
        start_dt = datetime(base_year, season_info["start_month"], season_info["start_day"])
        end_dt = datetime(base_year, season_info["end_month"], season_info["end_day"], 23, 59, 59)

    return start_dt, end_dt

def parse_time_expression(query: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Parse a natural language time expression and return timestamp range.

    Args:
        query: A natural language query with time expressions

    Returns:
        Tuple of (start_timestamp, end_timestamp), either may be None
    """
    # Truncate excessively long input to prevent ReDoS on pathological strings
    if len(query) > _MAX_TIME_QUERY_LEN:
        query = query[:_MAX_TIME_QUERY_LEN]

    query = query.lower().strip()

    # Check for multiple patterns in a single query
    try:
        # First check for date ranges like "between X and Y"
        date_range_match = PATTERNS["date_range"].search(query)
        if date_range_match:
            start_expr = date_range_match.group(1)
            end_expr = date_range_match.group(2)
            start_ts, _ = parse_time_expression(start_expr)
            _, end_ts = parse_time_expression(end_expr)
            return start_ts, end_ts
        
        # Check for full ISO dates (YYYY-MM-DD) FIRST
        full_date_match = PATTERNS["full_date"].search(query)
        if full_date_match:
            year, month, day = full_date_match.groups()
            try:
                specific_date = date(int(year), int(month), int(day))
                start_dt = datetime.combine(specific_date, time.min)
                end_dt = datetime.combine(specific_date, time.max)
                return start_dt.timestamp(), end_dt.timestamp()
            except ValueError as e:
                logger.warning("Invalid date: %s", _sanitize_log_value(e))
                return None, None
            
        # Check for specific dates (MM/DD/YYYY)
        specific_date_match = PATTERNS["specific_date"].search(query)
        if specific_date_match:
            month, day, year = specific_date_match.groups()
            month = int(month)
            day = int(day)
            current_year = datetime.now().year
            year = int(year) if year else current_year
            # Handle 2-digit years
            if year and year < 100:
                year = 2000 + year if year < 50 else 1900 + year
                
            try:
                specific_date = date(year, month, day)
                start_dt = datetime.combine(specific_date, time.min)
                end_dt = datetime.combine(specific_date, time.max)
                return start_dt.timestamp(), end_dt.timestamp()
            except ValueError as e:
                logger.warning("Invalid date: %s", _sanitize_log_value(e))
                return None, None
        
        # Relative days: "X days ago", "yesterday", "today"
        days_ago_match = PATTERNS["relative_days"].search(query)
        if days_ago_match:
            if "yesterday" in query:
                days = 1
            elif "today" in query:
                days = 0
            else:
                days = int(days_ago_match.group(1))
                
            target_date = date.today() - timedelta(days=days)
            
            # Check for time of day modifiers
            time_of_day_match = PATTERNS["time_of_day"].search(query)
            if time_of_day_match:
                # Narrow the range based on time of day
                return get_time_of_day_range(target_date, time_of_day_match.group(1))
            else:
                # Return the full day
                start_dt = datetime.combine(target_date, time.min)
                end_dt = datetime.combine(target_date, time.max)
                return start_dt.timestamp(), end_dt.timestamp()
        
        # Relative weeks: "X weeks ago"
        weeks_ago_match = PATTERNS["relative_weeks"].search(query)
        if weeks_ago_match:
            weeks = int(weeks_ago_match.group(1))
            target_date = date.today() - timedelta(weeks=weeks)
            # Get the start of the week (Monday)
            start_date = target_date - timedelta(days=target_date.weekday())
            end_date = start_date + timedelta(days=6)
            start_dt = datetime.combine(start_date, time.min)
            end_dt = datetime.combine(end_date, time.max)
            return start_dt.timestamp(), end_dt.timestamp()
        
        # Relative months: "X months ago"
        months_ago_match = PATTERNS["relative_months"].search(query)
        if months_ago_match:
            months = int(months_ago_match.group(1))
            current = datetime.now()
            # Calculate target month
            year = current.year
            month = current.month - months
            
            # Adjust year if month goes negative
            while month <= 0:
                year -= 1
                month += 12
                
            # Get first and last day of the month
            first_day = date(year, month, 1)
            if month == 12:
                last_day = date(year + 1, 1, 1) - timedelta(days=1)
            else:
                last_day = date(year, month + 1, 1) - timedelta(days=1)
                
            start_dt = datetime.combine(first_day, time.min)
            end_dt = datetime.combine(last_day, time.max)
            return start_dt.timestamp(), end_dt.timestamp()
        
        # Relative years: "X years ago"
        years_ago_match = PATTERNS["relative_years"].search(query)
        if years_ago_match:
            years = int(years_ago_match.group(1))
            current_year = datetime.now().year
            target_year = current_year - years
            start_dt = datetime(target_year, 1, 1, 0, 0, 0)
            end_dt = datetime(target_year, 12, 31, 23, 59, 59)
            return start_dt.timestamp(), end_dt.timestamp()

        # "Last N X" expressions (e.g., "last 3 days", "last 2 weeks")
        # Check this BEFORE "last_period" to match more specific pattern first
        last_n_periods_match = PATTERNS["last_n_periods"].search(query)
        if last_n_periods_match:
            n = int(last_n_periods_match.group(1))
            period = last_n_periods_match.group(2)
            return get_last_n_periods_range(n, period)

        # "Last X" expressions (e.g., "last week", "last month")
        last_period_match = PATTERNS["last_period"].search(query)
        if last_period_match:
            period = last_period_match.group(1)
            return get_last_period_range(period)
        
        # "This X" expressions
        this_period_match = PATTERNS["this_period"].search(query)
        if this_period_match:
            period = this_period_match.group(1)
            return get_this_period_range(period)
        
        # Month names
        month_match = PATTERNS["month_name"].search(query)
        if month_match:
            month_name = month_match.group(1)
            return get_month_range(month_name)
        
        # Named periods (holidays, etc.)
        named_period_match = PATTERNS["named_period"].search(query)
        if named_period_match:
            period_name = named_period_match.group(1)  # <-- Just get the matched group without replacing
            return get_named_period_range(period_name)
        
        # Half year expressions
        half_year_match = PATTERNS["half_year"].search(query)
        if half_year_match:
            half = half_year_match.group(1)
            year_str = half_year_match.group(2)
            year = int(year_str) if year_str else datetime.now().year
            
            if half.lower() == "first":
                start_dt = datetime(year, 1, 1, 0, 0, 0)
                end_dt = datetime(year, 6, 30, 23, 59, 59)
            else:  # "second"
                start_dt = datetime(year, 7, 1, 0, 0, 0)
                end_dt = datetime(year, 12, 31, 23, 59, 59)
                
            return start_dt.timestamp(), end_dt.timestamp()
        
        # Quarter expressions
        quarter_match = PATTERNS["quarter"].search(query)
        if quarter_match:
            quarter = quarter_match.group(1).lower()
            year_str = quarter_match.group(2)
            year = int(year_str) if year_str else datetime.now().year
            
            # Map textual quarter to number
            quarter_num = {"first": 1, "1st": 1, "second": 2, "2nd": 2, 
                          "third": 3, "3rd": 3, "fourth": 4, "4th": 4}[quarter]
            
            # Calculate quarter start and end dates
            quarter_month = (quarter_num - 1) * 3 + 1
            start_dt = datetime(year, quarter_month, 1, 0, 0, 0)
            
            if quarter_month + 3 > 12:
                end_dt = datetime(year + 1, 1, 1, 0, 0, 0) - timedelta(seconds=1)
            else:
                end_dt = datetime(year, quarter_month + 3, 1, 0, 0, 0) - timedelta(seconds=1)
                
            return start_dt.timestamp(), end_dt.timestamp()
        
        # Recent/fuzzy time expressions
        recent_match = PATTERNS["recent"].search(query)
        if recent_match:
            # Default to last 7 days for "recent"
            end_dt = datetime.now()
            start_dt = end_dt - timedelta(days=7)
            return start_dt.timestamp(), end_dt.timestamp()
            
        # If no time expression is found, return None for both timestamps
        return None, None
        
    except Exception as e:
        logger.error("Error parsing time expression: %s", _sanitize_log_value(e))
        return None, None

def get_time_of_day_range(target_date: date, time_period: str) -> Tuple[float, float]:
    """Get timestamp range for a specific time of day on a given date."""
    if time_period in TIME_OF_DAY:
        start_hour, end_hour = TIME_OF_DAY[time_period]
        
        # Handle periods that wrap around midnight
        if start_hour > end_hour:  # e.g., "night" = (22, 4)
            # For periods that span midnight, we need to handle specially
            if time_period == "night":
                start_dt = datetime.combine(target_date, time(start_hour, 0))
                end_dt = datetime.combine(target_date + timedelta(days=1), time(end_hour, 59, 59))
            else:
                # Default handling for other wrapping periods
                start_dt = datetime.combine(target_date, time(start_hour, 0))
                end_dt = datetime.combine(target_date + timedelta(days=1), time(end_hour, 59, 59))
        else:
            # Normal periods within a single day
            start_dt = datetime.combine(target_date, time(start_hour, 0))
            if end_hour == start_hour:  # For noon, midnight (specific hour)
                end_dt = datetime.combine(target_date, time(end_hour, 59, 59))
            else:
                end_dt = datetime.combine(target_date, time(end_hour, 59, 59))
                
        return start_dt.timestamp(), end_dt.timestamp()
    else:
        # Fallback to full day
        start_dt = datetime.combine(target_date, time.min)
        end_dt = datetime.combine(target_date, time.max)
        return start_dt.timestamp(), end_dt.timestamp()

def get_last_period_range(period: str) -> Tuple[float, float]:
    """Get timestamp range for 'last X' expressions."""
    now = datetime.now()
    today = date.today()
    
    if period == "day":
        # Last day = yesterday
        yesterday = today - timedelta(days=1)
        start_dt = datetime.combine(yesterday, time.min)
        end_dt = datetime.combine(yesterday, time.max)
    elif period == "week":
        # Last week = previous calendar week (Mon-Sun)
        # Find last Monday
        last_monday = today - timedelta(days=today.weekday() + 7)
        # Find last Sunday
        last_sunday = last_monday + timedelta(days=6)
        start_dt = datetime.combine(last_monday, time.min)
        end_dt = datetime.combine(last_sunday, time.max)
    elif period == "month":
        # Last month = previous calendar month
        first_of_this_month = date(today.year, today.month, 1)
        if today.month == 1:
            last_month = 12
            last_month_year = today.year - 1
        else:
            last_month = today.month - 1
            last_month_year = today.year
            
        first_of_last_month = date(last_month_year, last_month, 1)
        last_of_last_month = first_of_this_month - timedelta(days=1)
        
        start_dt = datetime.combine(first_of_last_month, time.min)
        end_dt = datetime.combine(last_of_last_month, time.max)
    elif period == "year":
        # Last year = previous calendar year
        last_year = today.year - 1
        start_dt = datetime(last_year, 1, 1, 0, 0, 0)
        end_dt = datetime(last_year, 12, 31, 23, 59, 59)
    elif period in ["summer", "spring", "winter", "fall", "autumn"]:
        # Last season
        season_info = NAMED_PERIODS[period]
        current_year = today.year
        
        # Determine if we're currently in this season
        current_month = today.month
        current_day = today.day
        is_current_season = False
        
        # Check if today falls within the season's date range
        if period in ["winter"]:  # Winter spans year boundary
            if (current_month >= season_info["start_month"] or 
                (current_month <= season_info["end_month"] and 
                 current_day <= season_info["end_day"])):
                is_current_season = True
        else:
            if (current_month >= season_info["start_month"] and current_month <= season_info["end_month"] and
                current_day >= season_info["start_day"] if current_month == season_info["start_month"] else True and
                current_day <= season_info["end_day"] if current_month == season_info["end_month"] else True):
                is_current_season = True
        
        # If we're currently in the season, get last year's season
        if is_current_season:
            year = current_year - 1
        else:
            year = current_year
            
        # Calculate season date range (handles winter's year boundary)
        context_month = current_month if is_current_season else None
        start_dt, end_dt = _calculate_season_date_range(period, season_info, year, context_month)
    else:
        # Fallback - last 24 hours
        end_dt = now
        start_dt = end_dt - timedelta(days=1)
        
    return start_dt.timestamp(), end_dt.timestamp()

def get_last_n_periods_range(n: int, period: str) -> Tuple[float, float]:
    """Get timestamp range for 'last N X' expressions (e.g., 'last 3 days')."""
    now = datetime.now()
    today = date.today()

    # Normalize period to singular form
    period = period.rstrip('s')  # Remove trailing 's' if present

    if period == "day":
        # Last N days means from N days ago 00:00 until now
        start_date = today - timedelta(days=n)
        start_dt = datetime.combine(start_date, time.min)
        end_dt = now
    elif period == "week":
        # Last N weeks means from N weeks ago Monday 00:00 until now
        start_date = today - timedelta(weeks=n)
        # Get Monday of that week
        start_date = start_date - timedelta(days=start_date.weekday())
        start_dt = datetime.combine(start_date, time.min)
        end_dt = now
    elif period == "month":
        # Last N months means from N months ago first day 00:00 until now
        current = datetime.now()
        year = current.year
        month = current.month - n

        # Handle year boundary
        while month <= 0:
            month += 12
            year -= 1

        start_date = date(year, month, 1)
        start_dt = datetime.combine(start_date, time.min)
        end_dt = now
    elif period == "year":
        # Last N years means from N years ago Jan 1 00:00 until now
        start_year = today.year - n
        start_dt = datetime(start_year, 1, 1, 0, 0, 0)
        end_dt = now
    else:
        # Fallback - interpret as days
        start_date = today - timedelta(days=n)
        start_dt = datetime.combine(start_date, time.min)
        end_dt = now

    return start_dt.timestamp(), end_dt.timestamp()

def get_this_period_range(period: str) -> Tuple[float, float]:
    """Get timestamp range for 'this X' expressions."""
    now = datetime.now()
    today = date.today()
    
    if period == "day":
        # This day = today
        start_dt = datetime.combine(today, time.min)
        end_dt = datetime.combine(today, time.max)
    elif period == "week":
        # This week = current calendar week (Mon-Sun)
        # Find this Monday
        monday = today - timedelta(days=today.weekday())
        sunday = monday + timedelta(days=6)
        start_dt = datetime.combine(monday, time.min)
        end_dt = datetime.combine(sunday, time.max)
    elif period == "month":
        # This month = current calendar month
        first_of_month = date(today.year, today.month, 1)
        if today.month == 12:
            first_of_next_month = date(today.year + 1, 1, 1)
        else:
            first_of_next_month = date(today.year, today.month + 1, 1)
            
        last_of_month = first_of_next_month - timedelta(days=1)
        
        start_dt = datetime.combine(first_of_month, time.min)
        end_dt = datetime.combine(last_of_month, time.max)
    elif period == "year":
        # This year = current calendar year
        start_dt = datetime(today.year, 1, 1, 0, 0, 0)
        end_dt = datetime(today.year, 12, 31, 23, 59, 59)
    elif period in ["summer", "spring", "winter", "fall", "autumn"]:
        # This season
        season_info = NAMED_PERIODS[period]
        current_year = today.year
        
        # Calculate season date range (handles winter's year boundary)
        start_dt, end_dt = _calculate_season_date_range(period, season_info, current_year, today.month)
    else:
        # Fallback - current 24 hours
        end_dt = now
        start_dt = datetime.combine(today, time.min)
        
    return start_dt.timestamp(), end_dt.timestamp()

def get_month_range(month_name: str) -> Tuple[float, float]:
    """Get timestamp range for a named month."""
    # Map month name to number
    month_map = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12
    }
    
    if month_name in month_map:
        month_num = month_map[month_name]
        current_year = datetime.now().year
        
        # If the month is in the future for this year, use last year
        current_month = datetime.now().month
        year = current_year if month_num <= current_month else current_year - 1
        
        # Get first and last day of the month
        first_day = date(year, month_num, 1)
        if month_num == 12:
            last_day = date(year + 1, 1, 1) - timedelta(days=1)
        else:
            last_day = date(year, month_num + 1, 1) - timedelta(days=1)
            
        start_dt = datetime.combine(first_day, time.min)
        end_dt = datetime.combine(last_day, time.max)
        return start_dt.timestamp(), end_dt.timestamp()
    else:
        return None, None

def get_named_period_range(period_name: str) -> Tuple[Optional[float], Optional[float]]:
    """Get timestamp range for named periods like holidays."""
    period_name = period_name.lower().replace("_", " ")
    current_year = datetime.now().year
    current_month = datetime.now().month
    current_day = datetime.now().day
    
    if period_name in NAMED_PERIODS:
        info = NAMED_PERIODS[period_name]
        # Found matching period
        # Determine if the period is in the past or future for this year
        if "month" in info and "day" in info:
            # Simple fixed-date holiday
            month = info["month"]
            day = info["day"]
            window = info.get("window", 1)  # Default 1-day window
            
            # Special case for Thanksgiving (fourth Thursday in November)
            if day == -1 and month == 11:  # Thanksgiving
                # Find the fourth Thursday in November
                first_day = date(current_year, 11, 1)
                # Find first Thursday
                first_thursday = first_day + timedelta(days=((3 - first_day.weekday()) % 7))
                # Fourth Thursday is 3 weeks later
                thanksgiving = first_thursday + timedelta(weeks=3)
                day = thanksgiving.day
            
            # Check if the holiday has passed this year
            is_past = (current_month > month or 
                        (current_month == month and current_day > day + window))
                        
            year = current_year if not is_past else current_year - 1
            target_date = date(year, month, day)
            
            # Create date range with window
            start_date = target_date - timedelta(days=window)
            end_date = target_date + timedelta(days=window)
            
            start_dt = datetime.combine(start_date, time.min)
            end_dt = datetime.combine(end_date, time.max)
            return start_dt.timestamp(), end_dt.timestamp()
            
        elif "start_month" in info and "end_month" in info:
            # Season or date range
            start_month = info["start_month"]
            start_day = info["start_day"]
            end_month = info["end_month"]
            end_day = info["end_day"]
            
            # Determine year based on current date
            if start_month > end_month:  # Period crosses year boundary
                if current_month < end_month or (current_month == end_month and current_day <= end_day):
                    # We're in the end part of the period that started last year
                    start_dt = datetime(current_year - 1, start_month, start_day)
                    end_dt = datetime(current_year, end_month, end_day, 23, 59, 59)
                else:
                    # The period is either coming up this year or happened earlier this year
                    if current_month > start_month or (current_month == start_month and current_day >= start_day):
                        # Period already started this year
                        start_dt = datetime(current_year, start_month, start_day)
                        end_dt = datetime(current_year + 1, end_month, end_day, 23, 59, 59)
                    else:
                        # Period from last year
                        start_dt = datetime(current_year - 1, start_month, start_day)
                        end_dt = datetime(current_year, end_month, end_day, 23, 59, 59)
            else:
                # Period within a single year
                # Check if period has already occurred this year
                if (current_month > end_month or 
                    (current_month == end_month and current_day > end_day)):
                    # Period already passed this year
                    start_dt = datetime(current_year, start_month, start_day)
                    end_dt = datetime(current_year, end_month, end_day, 23, 59, 59)
                else:
                    # Check if current date is within the period
                    is_within_period = (
                        (current_month > start_month or 
                            (current_month == start_month and current_day >= start_day))
                        and
                        (current_month < end_month or 
                            (current_month == end_month and current_day <= end_day))
                    )
                    
                    if is_within_period:
                        # We're in the period this year
                        start_dt = datetime(current_year, start_month, start_day)
                        end_dt = datetime(current_year, end_month, end_day, 23, 59, 59)
                    else:
                        # Period from last year
                        start_dt = datetime(current_year - 1, start_month, start_day)
                        end_dt = datetime(current_year - 1, end_month, end_day, 23, 59, 59)
            
            return start_dt.timestamp(), end_dt.timestamp()
    
    # If no match found
    return None, None

# Helper function to detect time expressions in a general query
def extract_time_expression(query: str) -> Tuple[str, Tuple[Optional[float], Optional[float]]]:
    """
    Extract time-related expressions from a query and return the timestamps.
    
    Args:
        query: A natural language query that may contain time expressions
        
    Returns:
        Tuple of (cleaned_query, (start_timestamp, end_timestamp))
        The cleaned_query has time expressions removed
    """
    # Check for time expressions
    time_expressions = [
        r'\b\d+\s+days?\s+ago\b',
        r'\byesterday\b',
        r'\btoday\b',
        r'\b\d+\s+weeks?\s+ago\b',
        r'\b\d+\s+months?\s+ago\b',
        r'\b\d+\s+years?\s+ago\b',
        r'\blast\s+(day|week|month|year|summer|spring|winter|fall|autumn)\b',
        r'\bthis\s+(day|week|month|year|summer|spring|winter|fall|autumn)\b',
        r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b',
        r'\bbetween\s+.+?\s+and\s+.+?(?:\s|$)',
        r'\bin\s+the\s+(morning|afternoon|evening|night|noon|midnight)\b',
        r'\brecent|lately|recently\b',
        r'\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b',
        r'\b\d{4}-\d{1,2}-\d{1,2}\b',
        r'\b(spring|summer|winter|fall|autumn|christmas|new\s*year|valentine|halloween|thanksgiving|spring\s*break|summer\s*break|winter\s*break)\b',
        r'\b(first|second)\s+half\s+of\s+\d{4}\b',
        r'\b(first|second|third|fourth|1st|2nd|3rd|4th)\s+quarter(?:\s+of\s+\d{4})?\b',
        r'\bfrom\s+.+\s+to\s+.+\b'
    ]
    
    # Combine all patterns
    combined_pattern = '|'.join(f'({expr})' for expr in time_expressions)
    combined_regex = re.compile(combined_pattern, re.IGNORECASE)
    
    # Find all matches
    matches = list(combined_regex.finditer(query))
    if not matches:
        return query, (None, None)
    
    # Extract the time expressions
    time_expressions = []
    for match in matches:
        span = match.span()
        expression = query[span[0]:span[1]]
        time_expressions.append(expression)
    
    # Parse time expressions to get timestamps
    full_time_expression = ' '.join(time_expressions)
    start_ts, end_ts = parse_time_expression(full_time_expression)
    
    # Remove time expressions from the query
    cleaned_query = query
    for expr in time_expressions:
        cleaned_query = cleaned_query.replace(expr, '')
    
    # Clean up multiple spaces
    cleaned_query = re.sub(r'\s+', ' ', cleaned_query).strip()
    
    return cleaned_query, (start_ts, end_ts)

def parse_boundary(value: str) -> float:
    """Epoch for an ``after=``/``before=`` filter string.

    A bare ``YYYY-MM-DD`` is a calendar day and resolves against the host's local
    day, matching what "today"/"yesterday" and the date-based deletion paths use.
    A datetime with a time component is an instant, and is read as UTC unless it
    carries its own offset.
    """
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is not None:
        return dt.timestamp()
    if len(value.strip()) == 10:  # date only: host-local calendar day
        return dt.timestamp()
    return dt.replace(tzinfo=timezone.utc).timestamp()
