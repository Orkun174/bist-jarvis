"""Scorecard: how graded predictions actually turned out."""

from __future__ import annotations

import pandas as pd

from jarvis.metrics import daily_scores

WINDOWS = {"20": 20, "60": 60, "all": None}
BUCKETS = [("0.50-0.55", 0.50, 0.55), ("0.55-0.60", 0.55, 0.60), ("0.60+", 0.60, 1.01)]


def _window(graded: pd.DataFrame) -> dict:
    picks = graded[graded["recommended"]]
    scores = daily_scores(graded, cost_bps=0.0)
    rank_ic = scores["rank_ic"].dropna()
    return {
        "recommendations": int(len(picks)),
        "hit_rate": None if picks.empty else float((picks["fwd_excess"] > 0).mean()),
        "mean_excess": None if picks.empty else float(picks["fwd_excess"].mean()),
        "mean_rank_ic": None if rank_ic.empty else float(rank_ic.mean()),
        "days": int(graded["date"].nunique()),
    }


def scorecard(
    ledger: pd.DataFrame,
    session: pd.Timestamp,
    kind: str,
    calendar: pd.DatetimeIndex | None = None,
) -> dict:
    """Summary over the last 20 / 60 / all sessions, plus probability calibration.

    `kind` is "live" (real predictions made on the day) or "simulated"
    (walk-forward replay); the two are never mixed in one scorecard.
    """
    if kind not in ("live", "simulated"):
        raise ValueError("kind must be 'live' or 'simulated'")
    session = pd.Timestamp(session).normalize()
    graded = ledger[ledger["matured"].astype(bool) & (ledger["date"] <= session)]
    if calendar is None:
        calendar = pd.DatetimeIndex(sorted(ledger["date"].unique()))
    sessions = calendar[calendar <= session]

    windows = {}
    for name, length in WINDOWS.items():
        subset = graded
        if length is not None and len(sessions) > 0:
            subset = graded[graded["date"] >= sessions[-length:][0]]
        windows[name] = _window(subset)

    calibration = []
    for label, low, high in BUCKETS:
        bucket = graded[(graded["prob"] >= low) & (graded["prob"] < high)]
        calibration.append(
            {
                "bucket": label,
                "count": int(len(bucket)),
                "hit_rate": None if bucket.empty else float((bucket["fwd_excess"] > 0).mean()),
            }
        )

    return {
        "kind": kind,
        "as_of": session.date().isoformat(),
        "windows": windows,
        "calibration": calibration,
        "pending": int((~ledger["matured"].astype(bool)).sum()),
    }
