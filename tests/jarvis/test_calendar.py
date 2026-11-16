import pandas as pd
import pytest

from jarvis.calendar import last_completed_session, sessions_until


def trt(text):
    return pd.Timestamp(text, tz="Europe/Istanbul")


def test_after_close_selects_same_day():
    assert last_completed_session(trt("2026-11-13 18:30")) == pd.Timestamp("2026-11-13")


def test_before_close_selects_previous_session():
    assert last_completed_session(trt("2026-11-13 17:00")) == pd.Timestamp("2026-11-12")


def test_weekend_and_monday_morning_select_friday():
    assert last_completed_session(trt("2026-11-14 12:00")) == pd.Timestamp("2026-11-13")
    assert last_completed_session(trt("2026-11-16 09:00")) == pd.Timestamp("2026-11-13")


def test_naive_timestamp_is_rejected():
    with pytest.raises(ValueError):
        last_completed_session(pd.Timestamp("2026-11-13 18:30"))


def test_sessions_until_ends_at_session_and_skips_weekends():
    sessions = sessions_until(pd.Timestamp("2026-11-13"), 30)
    assert len(sessions) == 30
    assert sessions[-1] == pd.Timestamp("2026-11-13")
    assert sessions.tz is None
    assert all(day.dayofweek < 5 for day in sessions)
