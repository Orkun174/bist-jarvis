"""Macro and index series (USD/TRY, Brent, gold, S&P 500, VIX, BIST100, bank index)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from jarvis.prices import Downloader, update_prices, yahoo_download
from jarvis.store import Store

MACRO_SYMBOLS: dict[str, str] = {
    "USDTRY": "USDTRY=X",
    "BRENT": "BZ=F",
    "GOLD": "GC=F",
    "SP500": "^GSPC",
    "VIX": "^VIX",
    "BIST100": "XU100.IS",
    "BISTBANK": "XBANK.IS",
}
# Without these no target (BIST100) or recommendation threshold (VIX) exists.
REQUIRED_MACRO = ("BIST100", "VIX")


def macro_store(data_dir: Path | str) -> Store:
    return Store(Path(data_dir) / "macro")


def update_macro(
    data_dir: Path | str, session: pd.Timestamp, download: Downloader = yahoo_download
) -> dict[str, str]:
    return update_prices(
        macro_store(data_dir), list(MACRO_SYMBOLS), session,
        download=download, symbol_of=MACRO_SYMBOLS.__getitem__,
    )


def load_macro(data_dir: Path | str) -> dict[str, pd.DataFrame]:
    store = macro_store(data_dir)
    return {name: store.read_prices(name) for name in MACRO_SYMBOLS}
