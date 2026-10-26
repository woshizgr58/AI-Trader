"""Exchange-calendar checks use fixed historical dates, never live connectivity."""
from datetime import datetime, timezone

import pytest

from paper_trader import calendar


def at(iso):
    return datetime.fromisoformat(iso)


def test_holiday_and_weekend_sessions():
    assert calendar.sessions("2024-07-03", "2024-07-08") == ["2024-07-03", "2024-07-05", "2024-07-08"]
    assert calendar.previous_session(at("2024-07-05T13:35:00+00:00")) == "2024-07-03"
    assert calendar.decision_session(at("2024-07-04T15:00:00+00:00")) is None
    assert calendar.next_run(at("2024-07-03T17:00:00+00:00")) == at("2024-07-05T13:35:00+00:00")


def test_daylight_saving_changes_utc_open_but_not_local_decision_time():
    assert calendar.next_run(at("2024-03-07T22:00:00+00:00")) == at("2024-03-08T14:35:00+00:00")
    assert calendar.next_run(at("2024-03-08T22:00:00+00:00")) == at("2024-03-11T13:35:00+00:00")
    assert calendar.next_run(at("2024-11-01T22:00:00+00:00")) == at("2024-11-04T14:35:00+00:00")


def test_decision_timing_never_uses_todays_unfinished_daily_bar():
    assert calendar.decision_session(at("2024-03-11T13:34:59+00:00")) is None
    assert calendar.decision_session(at("2024-03-11T13:35:00+00:00")) == "2024-03-11"
    assert calendar.previous_session(at("2024-03-11T13:35:00+00:00")) == "2024-03-08"
    assert calendar.decision_session(at("2024-03-11T20:00:00+00:00")) is None
    # The regular close does not complete Alpaca's New York daily aggregate.
    assert calendar.previous_session(at("2024-03-11T23:00:00+00:00")) == "2024-03-08"
    assert calendar.previous_session(at("2024-03-12T04:00:00+00:00")) == "2024-03-11"


def test_early_close_is_respected():
    assert calendar.session_close("2024-11-29") == at("2024-11-29T18:00:00+00:00")
    assert calendar.decision_session(at("2024-11-29T17:59:59+00:00")) == "2024-11-29"
    assert calendar.decision_session(at("2024-11-29T18:00:00+00:00")) is None
    assert calendar.next_run(at("2024-11-29T18:00:00+00:00")) == at("2024-12-02T14:35:00+00:00")


def test_timestamp_and_delay_validation():
    with pytest.raises(ValueError, match="timezone-aware"):
        calendar.previous_session(datetime(2024, 3, 11))
    with pytest.raises(ValueError, match="delay"):
        calendar.next_run(datetime(2024, 3, 11, tzinfo=timezone.utc), -1)
    with pytest.raises(ValueError):
        calendar.sessions("2024-03-12", "2024-03-11")


def test_extraordinary_market_closure():
    assert calendar.sessions("2025-01-08", "2025-01-10") == ["2025-01-08", "2025-01-10"]
