"""XIST trading-calendar helpers (timezone-naive, normalized session dates)."""

from __future__ import annotations

import exchange_calendars as xcals
import pandas as pd

from jarvis.config import CALENDAR, TRT

# The closing auction ends ten minutes after the continuous-trading close.
CLOSING_PHASE = pd.Timedelta(minutes=10)


def _calendar(start: pd.Timestamp, end: pd.Timestamp):
    return xcals.get_calendar(
        CALENDAR, start=start.date().isoformat(), end=end.date().isoformat()
    )


def last_completed_session(as_of: pd.Timestamp | None = None) -> pd.Timestamp:
    """The most recent XIST session whose closing phase has ended by `as_of`."""
    now = pd.Timestamp.now(tz=TRT) if as_of is None else pd.Timestamp(as_of)
    if now.tzinfo is None:
        raise ValueError("as_of saat dilimi içermeli (ör. +03:00).")
    now = now.tz_convert(TRT)
    day = now.tz_localize(None).normalize()
    closes = _calendar(day - pd.Timedelta(days=30), day).closes + CLOSING_PHASE
    completed = closes.loc[closes <= now.tz_convert("UTC")]
    if completed.empty:
        raise RuntimeError(f"{now} öncesinde tamamlanmış seans bulunamadı.")
    return pd.Timestamp(completed.index[-1]).tz_localize(None).normalize()


def sessions_until(session: pd.Timestamp, count: int) -> pd.DatetimeIndex:
    """The last `count` XIST sessions ending at `session` (inclusive)."""
    session = pd.Timestamp(session).normalize()
    start = session - pd.Timedelta(days=count * 2 + 30)
    sessions = pd.DatetimeIndex(_calendar(start, session).sessions).tz_localize(None).normalize()
    return sessions[-count:]
