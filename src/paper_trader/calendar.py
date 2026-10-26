"""NYSE session labels and UTC decision times, including holidays and DST.

Daily Alpaca bars use New York calendar days. ``previous_session`` deliberately
excludes today's date even after the regular close, when extended-hours prints
could still change today's daily bar.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd

NEW_YORK = ZoneInfo("America/New_York")


def _date(value: str | date) -> date:
    if isinstance(value, datetime):
        raise ValueError("A session must be an ISO date, without a time.")
    if isinstance(value, date):
        return value
    result = date.fromisoformat(value)
    if result.isoformat() != value:
        raise ValueError("A session must use YYYY-MM-DD format.")
    return result


def utc_now(now: datetime | None = None) -> datetime:
    result = datetime.now(timezone.utc) if now is None else now
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("A timezone-aware datetime is required.")
    return result.astimezone(timezone.utc)


@lru_cache(maxsize=16)
def _calendar(first_year: int, last_year: int):
    # Explicit bounds avoid the library's moving default historical window.
    return xcals.get_calendar(
        "XNYS", start=f"{first_year - 1}-01-01", end=f"{last_year + 1}-12-31"
    )


def sessions(start: str | date, end: str | date) -> list[str]:
    """Return exchange session dates in an inclusive date range."""
    first, last = _date(start), _date(end)
    if first > last:
        raise ValueError("The first session date must not follow the last date.")
    labels = _calendar(first.year, last.year).sessions_in_range(
        pd.Timestamp(first), pd.Timestamp(last)
    )
    return [label.date().isoformat() for label in labels]


def previous_session(now: datetime | None = None) -> str:
    """Last session strictly before today's New York date; safe for daily bars."""
    local_day = utc_now(now).astimezone(NEW_YORK).date()
    previous_day = local_day - timedelta(days=1)
    calendar = _calendar(previous_day.year, local_day.year)
    return calendar.date_to_session(pd.Timestamp(previous_day), direction="previous").date().isoformat()


def session_open(session: str) -> datetime:
    day = _date(session)
    return _calendar(day.year, day.year).session_open(pd.Timestamp(day)).to_pydatetime()


def session_close(session: str) -> datetime:
    day = _date(session)
    return _calendar(day.year, day.year).session_close(pd.Timestamp(day)).to_pydatetime()


def _delay(delay_minutes: int) -> timedelta:
    if isinstance(delay_minutes, bool) or not isinstance(delay_minutes, int) or not 0 <= delay_minutes < 210:
        raise ValueError("Decision delay must be an integer from 0 through 209 minutes.")
    return timedelta(minutes=delay_minutes)


def decision_session(now: datetime | None = None, delay_minutes: int = 5) -> str | None:
    """Today's session only after the decision time and before regular close."""
    instant, delay = utc_now(now), _delay(delay_minutes)
    label = instant.astimezone(NEW_YORK).date().isoformat()
    if not sessions(label, label):
        return None
    if session_open(label) + delay <= instant < session_close(label):
        return label
    return None


def next_run(now: datetime | None = None, delay_minutes: int = 5) -> datetime:
    """First scheduled open-plus-delay strictly after now, returned in UTC."""
    instant, delay = utc_now(now), _delay(delay_minutes)
    local_day = instant.astimezone(NEW_YORK).date()
    for label in sessions(local_day, local_day + timedelta(days=31)):
        scheduled = session_open(label) + delay
        if instant < scheduled < session_close(label):
            return scheduled
    raise RuntimeError("No trading session found within 31 days; update the exchange calendar.")
