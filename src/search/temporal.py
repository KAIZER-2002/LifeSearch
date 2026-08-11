from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional


@dataclass
class TimeRange:
    start_ts: Optional[str] = None
    end_ts: Optional[str] = None
    precision: str = "unresolved"  # "exact_day" | "week" | "month" | "approximate" | "unresolved"
    original_expression: str = ""
    resolved: bool = False


WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}

MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def _iso_utc(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.isoformat()


def _days_in_month(year: int, month: int) -> int:
    if month in (1, 3, 5, 7, 8, 10, 12):
        return 31
    if month in (4, 6, 9, 11):
        return 30
    # February
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        return 29
    return 28


class TemporalParser:
    def parse(self, expression: str, reference_date: Optional[datetime] = None) -> TimeRange:
        if not expression or not expression.strip():
            return TimeRange(original_expression=expression)

        raw = expression.strip()
        expr_lower = raw.lower()

        if reference_date is None:
            ref = datetime.now(timezone.utc)
        else:
            if reference_date.tzinfo is None:
                ref = reference_date.replace(tzinfo=timezone.utc)
            else:
                ref = reference_date.astimezone(timezone.utc)

        # 1. Today
        if expr_lower == "today":
            start = ref.replace(hour=0, minute=0, second=0, microsecond=0)
            end = ref.replace(hour=23, minute=59, second=59, microsecond=999999)
            return TimeRange(_iso_utc(start), _iso_utc(end), "exact_day", raw, True)

        # 2. Yesterday
        if expr_lower == "yesterday":
            target = ref - timedelta(days=1)
            start = target.replace(hour=0, minute=0, second=0, microsecond=0)
            end = target.replace(hour=23, minute=59, second=59, microsecond=999999)
            return TimeRange(_iso_utc(start), _iso_utc(end), "exact_day", raw, True)

        # 3. Two days ago
        if expr_lower == "two days ago":
            target = ref - timedelta(days=2)
            start = target.replace(hour=0, minute=0, second=0, microsecond=0)
            end = target.replace(hour=23, minute=59, second=59, microsecond=999999)
            return TimeRange(_iso_utc(start), _iso_utc(end), "exact_day", raw, True)

        # 4. Last <weekday> (e.g., "last Tuesday")
        m_weekday = re.match(r"^last\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)$", expr_lower)
        if m_weekday:
            target_weekday = WEEKDAYS[m_weekday.group(1)]
            current_weekday = ref.weekday()
            days_back = (current_weekday - target_weekday) % 7
            if days_back == 0:
                days_back = 7
            target = ref - timedelta(days=days_back)
            start = target.replace(hour=0, minute=0, second=0, microsecond=0)
            end = target.replace(hour=23, minute=59, second=59, microsecond=999999)
            return TimeRange(_iso_utc(start), _iso_utc(end), "exact_day", raw, True)

        # 5. Last week
        if expr_lower == "last week":
            current_monday = ref - timedelta(days=ref.weekday())
            last_monday = current_monday - timedelta(days=7)
            last_sunday = last_monday + timedelta(days=6)
            start = last_monday.replace(hour=0, minute=0, second=0, microsecond=0)
            end = last_sunday.replace(hour=23, minute=59, second=59, microsecond=999999)
            return TimeRange(_iso_utc(start), _iso_utc(end), "week", raw, True)

        # 6. This week
        if expr_lower == "this week":
            current_monday = ref - timedelta(days=ref.weekday())
            start = current_monday.replace(hour=0, minute=0, second=0, microsecond=0)
            end = ref.replace(hour=23, minute=59, second=59, microsecond=999999)
            return TimeRange(_iso_utc(start), _iso_utc(end), "week", raw, True)

        # 7. Last month
        if expr_lower == "last month":
            first_of_this_month = ref.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            last_day_prev_month = first_of_this_month - timedelta(days=1)
            first_day_prev_month = last_day_prev_month.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            end = last_day_prev_month.replace(hour=23, minute=59, second=59, microsecond=999999)
            return TimeRange(_iso_utc(first_day_prev_month), _iso_utc(end), "month", raw, True)

        # 8. "around <month>" e.g., "around May"
        m_around = re.match(r"^around\s+(" + "|".join(MONTHS.keys()) + r")$", expr_lower)
        if m_around:
            month_num = MONTHS[m_around.group(1)]
            year = ref.year
            if month_num > ref.month:
                year -= 1
            # Expand ±6 weeks from May 15
            mid = datetime(year, month_num, 15, 12, 0, 0, tzinfo=timezone.utc)
            start = mid - timedelta(weeks=6)
            end = mid + timedelta(weeks=6)
            return TimeRange(_iso_utc(start), _iso_utc(end), "approximate", raw, True)

        # 9. "in <month>" or "in <month> <year>"
        m_in_month = re.match(r"^in\s+(" + "|".join(MONTHS.keys()) + r")(?:\s+(\d{4}))?$", expr_lower)
        if m_in_month:
            month_num = MONTHS[m_in_month.group(1)]
            year = int(m_in_month.group(2)) if m_in_month.group(2) else ref.year
            if not m_in_month.group(2) and month_num > ref.month:
                year -= 1
            num_days = _days_in_month(year, month_num)
            start = datetime(year, month_num, 1, 0, 0, 0, tzinfo=timezone.utc)
            end = datetime(year, month_num, num_days, 23, 59, 59, 999999, tzinfo=timezone.utc)
            return TimeRange(_iso_utc(start), _iso_utc(end), "month", raw, True)

        # 10. Exact date YYYY-MM-DD
        m_date = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", expr_lower)
        if m_date:
            y, m, d = int(m_date.group(1)), int(m_date.group(2)), int(m_date.group(3))
            start = datetime(y, m, d, 0, 0, 0, tzinfo=timezone.utc)
            end = datetime(y, m, d, 23, 59, 59, 999999, tzinfo=timezone.utc)
            return TimeRange(_iso_utc(start), _iso_utc(end), "exact_day", raw, True)

        # 11. "between 10pm and 11pm" or "between 10pm and 11pm on 2026-08-05"
        m_between = re.match(
            r"^between\s+(\d{1,2})(am|pm)\s+and\s+(\d{1,2})(am|pm)(?:\s+on\s+(\d{4}-\d{2}-\d{2}))?$",
            expr_lower,
        )
        if m_between:
            h1 = int(m_between.group(1))
            ampm1 = m_between.group(2)
            h2 = int(m_between.group(3))
            ampm2 = m_between.group(4)
            date_str = m_between.group(5)

            if ampm1 == "pm" and h1 < 12:
                h1 += 12
            elif ampm1 == "am" and h1 == 12:
                h1 = 0

            if ampm2 == "pm" and h2 < 12:
                h2 += 12
            elif ampm2 == "am" and h2 == 12:
                h2 = 0

            if date_str:
                y, m, d = [int(x) for x in date_str.split("-")]
                target = datetime(y, m, d, tzinfo=timezone.utc)
            else:
                target = ref

            start = target.replace(hour=h1, minute=0, second=0, microsecond=0)
            end = target.replace(hour=h2, minute=0, second=0, microsecond=0)
            return TimeRange(_iso_utc(start), _iso_utc(end), "exact_day", raw, True)

        return TimeRange(original_expression=raw)
