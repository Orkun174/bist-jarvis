"""Daily data job: universe -> prices -> quality -> exclusions -> eligibility.

Usage:
    python -m jarvis.pipeline --data-dir data
    python -m jarvis.pipeline --data-dir data --limit 15
    python -m jarvis.pipeline --data-dir data --as-of "2026-11-13 19:00:00+03:00"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Callable

import pandas as pd

from jarvis.calendar import last_completed_session, sessions_until
from jarvis.config import (
    LAG_DEADLINE_HOUR, LAG_MAX_WAIT_SECONDS, LAG_RETRIES, LAG_WAIT_SECONDS, QUALITY_WINDOW,
    SOURCE_LAG_RATIO, TRT,
)
from jarvis.eligibility import eligibility_today
from jarvis.exclusions import COLUMNS, load_exclusions
from jarvis.measures import fetch_exclusions
from jarvis.prices import Downloader, update_prices, yahoo_download
from jarvis.quality import ReferenceCloses, build_report, isyatirim_closes
from jarvis.store import Store
from jarvis.universe import UniverseError, fetch_symbols

LOG = logging.getLogger("jarvis.pipeline")
DEFAULT_EXCLUSIONS = Path(__file__).resolve().parents[1] / "config" / "exclusions.csv"

# Which automatic source produced which exclusion reasons.
SOURCE_REASONS = {
    "kap_markets": ("watchlist_market", "unknown_market"),
    "measures": ("vbts",),
}


def _automatic_exclusions(
    store: Store, session: pd.Timestamp, fresh: pd.DataFrame, status: dict[str, str]
) -> pd.DataFrame:
    """Record today's snapshot; for a failed source fall back to its last good one."""
    history = store.read_table("universe/measures")
    if history.empty:
        history = pd.DataFrame(columns=COLUMNS + ["session"])
    effective = [fresh]
    for source, reasons in SOURCE_REASONS.items():
        if status.get(source) == "ok":
            continue
        earlier = history[history["reason"].isin(reasons) & (history["session"] < session)]
        if earlier.empty:
            continue
        latest = earlier[earlier["session"] == earlier["session"].max()][COLUMNS].copy()
        if source == "kap_markets":  # market membership has no end date of its own
            latest["start"] = session
            latest["end"] = session
        effective.append(latest)

    ok_reasons = [r for s, rs in SOURCE_REASONS.items() if status.get(s) == "ok" for r in rs]
    snapshot = fresh[fresh["reason"].isin(ok_reasons)].copy()
    snapshot["session"] = session
    kept = history[~((history["session"] == session) & history["reason"].isin(ok_reasons))]
    frames = [frame for frame in (kept, snapshot) if not frame.empty]
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined = combined.sort_values(["session", "reason", "ticker"], kind="stable")
        store.write_table("universe/measures", combined)
    non_empty = [frame for frame in effective if not frame.empty]
    return pd.concat(non_empty, ignore_index=True) if non_empty else fresh


def _source_lagging(store: Store, tickers: list[str], session: pd.Timestamp) -> bool:
    missing = 0
    for ticker in tickers:
        frame = store.read_prices(ticker)
        missing += frame.empty or frame.index[-1] < session
    return missing > SOURCE_LAG_RATIO * len(tickers)


def lag_retries(session: pd.Timestamp, now: pd.Timestamp) -> int:
    """How many waits for a lagging Yahoo: until the morning deadline, at least LAG_RETRIES."""
    deadline = (pd.Timestamp(session).normalize() + pd.Timedelta(days=1, hours=LAG_DEADLINE_HOUR))
    budget = min((deadline.tz_localize(TRT) - now.tz_convert(TRT)).total_seconds(),
                 LAG_MAX_WAIT_SECONDS)
    return max(LAG_RETRIES, int(budget // LAG_WAIT_SECONDS))


def run(
    data_dir: Path | str,
    as_of: pd.Timestamp | None = None,
    *,
    symbols: Callable[[], list[str]] = fetch_symbols,
    download: Downloader = yahoo_download,
    reference: ReferenceCloses = isyatirim_closes,
    exclusions: Callable = fetch_exclusions,
    exclusions_path: Path | str = DEFAULT_EXCLUSIONS,
    limit: int | None = None,
    wait: Callable[[float], object] = time.sleep,
) -> dict:
    session = last_completed_session(as_of)
    tickers = symbols()  # raises UniverseError before anything is written
    if limit is not None:
        tickers = tickers[:limit]
    store = Store(data_dir)

    LOG.info("Seans %s: %d hisse için fiyatlar güncelleniyor.", session.date(), len(tickers))
    statuses = update_prices(store, tickers, session, download=download)
    source_lag = _source_lagging(store, tickers, session)
    retries = lag_retries(session, pd.Timestamp.now(tz=TRT) if as_of is None else pd.Timestamp(as_of))
    for attempt in range(retries):
        if not source_lag:
            break
        LOG.warning(
            "Yahoo %s seansını henüz yayınlamamış görünüyor; %d dk sonra yeniden denenecek (%d/%d).",
            session.date(), LAG_WAIT_SECONDS // 60, attempt + 1, retries,
        )
        wait(LAG_WAIT_SECONDS)
        # Tickers that already have the session are "current" and are not downloaded again.
        statuses = update_prices(store, tickers, session, download=download)
        source_lag = _source_lagging(store, tickers, session)
    LOG.info("Fiyatlar tamam; veri kontrolü ve kapanış karşılaştırması başlıyor.")
    report = build_report(
        store, tickers, session, statuses, sessions_until(session, QUALITY_WINDOW),
        reference=reference,
    )

    LOG.info("Veri kontrolü tamam; dışlama listeleri alınıyor.")
    fresh, source_status = exclusions(tickers, session)
    automatic = _automatic_exclusions(store, session, fresh, source_status)
    manual = load_exclusions(exclusions_path)
    merged = [frame for frame in (manual, automatic) if not frame.empty]
    all_exclusions = pd.concat(merged, ignore_index=True) if merged else manual

    today = eligibility_today(store, tickers, session, all_exclusions, set(report["quarantined"]))
    universe = store.read_table("universe/universe")
    if not universe.empty:
        universe = universe[universe["session"] != session]
    frames = [frame for frame in (universe, today) if not frame.empty]
    universe = pd.concat(frames, ignore_index=True).sort_values(["session", "ticker"], kind="stable")
    store.write_table("universe/universe", universe)

    report["exclusion_sources"] = source_status
    report["source_lag"] = source_lag
    report["eligible"] = int(today["eligible"].sum())
    report["reasons"] = {
        str(reason or "eligible"): int(count)
        for reason, count in today["reason"].value_counts().sort_index().items()
    }
    store.write_json(f"quality/{report['session']}", report)

    return {
        "session": report["session"],
        "tickers": len(tickers),
        "eligible": report["eligible"],
        "quarantined": len(report["quarantined"]),
        "excluded": int(today["reason"].str.startswith("excluded:").sum()),
        "training_allowed": report["training_allowed"],
        "exclusion_sources": source_status,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--as-of", default=None, help='ör. "2026-11-13 19:00:00+03:00"')
    parser.add_argument("--limit", type=int, default=None, help="yalnızca ilk N sembol")
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    as_of = pd.Timestamp(arguments.as_of) if arguments.as_of else None
    try:
        summary = run(arguments.data_dir, as_of, limit=arguments.limit)
    except UniverseError as error:
        LOG.error("%s", error)
        return 1
    sys.stdout.write(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
