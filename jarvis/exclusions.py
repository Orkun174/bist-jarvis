"""Exclusion list: tickers kept out of training and recommendations by date window."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

COLUMNS = ["ticker", "reason", "start", "end", "source"]


def empty_exclusions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": pd.Series(dtype=str),
            "reason": pd.Series(dtype=str),
            "start": pd.Series(dtype="datetime64[ns]"),
            "end": pd.Series(dtype="datetime64[ns]"),
            "source": pd.Series(dtype=str),
        }
    )


def load_exclusions(path: Path | str) -> pd.DataFrame:
    """Read the hand-maintained CSV; a missing or header-only file excludes nobody."""
    path = Path(path)
    if not path.exists():
        return empty_exclusions()
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    if table.empty:
        return empty_exclusions()
    table = table.reindex(columns=COLUMNS, fill_value="")
    table["ticker"] = table["ticker"].str.strip().str.upper()
    table["start"] = pd.to_datetime(table["start"].replace("", None))
    table["end"] = pd.to_datetime(table["end"].replace("", None))
    return table


def excluded_on(exclusions: pd.DataFrame, day: pd.Timestamp) -> dict[str, str]:
    """Ticker -> reason for exclusions active on `day`; an empty `end` never expires."""
    if exclusions.empty:
        return {}
    day = pd.Timestamp(day).normalize()
    started = exclusions["start"].isna() | (exclusions["start"] <= day)
    not_ended = exclusions["end"].isna() | (exclusions["end"] >= day)
    active = exclusions[started & not_ended]
    return dict(zip(active["ticker"], active["reason"]))
