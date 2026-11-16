"""Small synthetic market used by the training tests (no network, deterministic)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from jarvis.calendar import sessions_until
from jarvis.macro import MACRO_SYMBOLS, macro_store
from jarvis.store import Store

SESSION = pd.Timestamp("2026-11-13")
CALENDAR = sessions_until(SESSION, 400)


def random_ohlcv(seed: int, index: pd.DatetimeIndex = CALENDAR, drift: float = 0.0,
                 volume: float = 2_000_000.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(drift, 0.02, len(index))))
    high = close * (1 + rng.uniform(0, 0.02, len(index)))
    low = close * (1 - rng.uniform(0, 0.02, len(index)))
    open_ = low + (high - low) * rng.uniform(0, 1, len(index))
    vol = volume * rng.uniform(0.5, 1.5, len(index))
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol}, index=index
    )


def macro_histories(index: pd.DatetimeIndex = CALENDAR) -> dict[str, pd.DataFrame]:
    return {name: random_ohlcv(100 + i, index) for i, name in enumerate(MACRO_SYMBOLS)}


def build_market(root, tickers: int = 12, index: pd.DatetimeIndex = CALENDAR):
    """Write `tickers` liquid synthetic stocks plus macro series under `root`."""
    store = Store(root)
    names = [f"S{i:02d}" for i in range(tickers)]
    for i, name in enumerate(names):
        store.write_prices(name, random_ohlcv(i, index))
    macros = macro_histories(index)
    target = macro_store(root)
    for name, frame in macros.items():
        target.write_prices(name, frame)
    return store, names, macros
