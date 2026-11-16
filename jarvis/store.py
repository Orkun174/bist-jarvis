"""Parquet/JSON storage under one data directory.

Writes are idempotent: writing content that is already on disk leaves the file
untouched, so a second run on the same day produces no git diff.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pandas as pd

from jarvis.config import OHLCV


class Store:
    def __init__(self, root: Path | str):
        self.root = Path(root)

    # -- prices ---------------------------------------------------------

    def _price_path(self, ticker: str) -> Path:
        return self.root / "prices" / f"{ticker}.parquet"

    def read_prices(self, ticker: str) -> pd.DataFrame:
        path = self._price_path(ticker)
        if not path.exists():
            return pd.DataFrame(columns=OHLCV, index=pd.DatetimeIndex([]), dtype=float)
        frame = pd.read_parquet(path)
        frame.index = pd.DatetimeIndex(frame.index)
        return frame[OHLCV]

    def write_prices(self, ticker: str, frame: pd.DataFrame) -> bool:
        """Merge `frame` into the stored history; new rows win on equal dates."""
        incoming = frame[OHLCV].astype(float)
        incoming.index = pd.DatetimeIndex(incoming.index).tz_localize(None).normalize()
        existing = self.read_prices(ticker)
        merged = incoming if existing.empty else pd.concat([existing, incoming])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        merged.index.name = "Date"
        if not existing.empty and merged.equals(existing.rename_axis("Date")):
            return False
        self._write_parquet(self._price_path(ticker), merged)
        return True

    # -- generic tables -------------------------------------------------

    def _table_path(self, name: str) -> Path:
        return self.root / f"{name}.parquet"

    def read_table(self, name: str) -> pd.DataFrame:
        path = self._table_path(name)
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    def write_table(self, name: str, frame: pd.DataFrame) -> bool:
        path = self._table_path(name)
        frame = frame.reset_index(drop=True)
        if path.exists() and pd.read_parquet(path).equals(frame):
            return False
        self._write_parquet(path, frame)
        return True

    # -- json -----------------------------------------------------------

    def write_json(self, name: str, payload: dict) -> None:
        text = json.dumps(payload, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True)
        self._atomic_write(self.root / f"{name}.json", (text + "\n").encode("utf-8"))

    # -- helpers --------------------------------------------------------

    def _write_parquet(self, path: Path, frame: pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        os.close(descriptor)
        try:
            frame.to_parquet(temporary)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)

    def _atomic_write(self, path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)
