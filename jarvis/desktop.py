"""What the PC application shows, computed from a local copy of the `data` branch.

Nothing here imports Streamlit: `jarvis_app.py` only draws what these functions
return, so the logic can be tested without a browser.
"""

from __future__ import annotations

import io
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import requests

from jarvis.features import _wilder, extra_features, local_features
from jarvis.macro import macro_store
from jarvis.store import Store
from jarvis.telegram import format_message

DATA_URL = "https://codeload.github.com/Orkun174/bist-jarvis/zip/refs/heads/data"
MAX_DOWNLOAD_BYTES = 400 * 1024 * 1024
SYNC_MARK = ".synced_at"

REASON_TEXT = {
    "": "Uygun: model bu hisseyi sıralıyor.",
    "low_liquidity": "Düşük hacim: son 20 seansın medyan işlem hacmi 50 milyon TL'nin altında.",
    "short_history": "Kısa geçmiş: 252 seanstan az fiyat verisi var.",
    "limit_moves": "Tavan/taban: son 20 seansta 5'ten fazla tavan ya da taban günü.",
    "erratic": "Düzensiz fiyat davranışı: aşırı oynaklık ya da 120 seansta iki katı aşan yükseliş.",
    "quarantined": "Karantina: veri kontrolünden geçemedi.",
    "excluded:vbts": "Borsa tedbiri (VBTS) altında.",
    "excluded:watchlist_market": "Yakın İzleme / piyasa öncesi pazarında işlem görüyor.",
    "excluded:manual": "Elle dışlandı (config/exclusions.csv).",
}


# -- keeping the local copy current -------------------------------------------

def _download(url: str) -> bytes:
    with requests.get(url, timeout=(5, 60), stream=True) as response:
        response.raise_for_status()
        content = bytearray()
        for chunk in response.iter_content(chunk_size=1 << 16):
            content.extend(chunk)
            if len(content) > MAX_DOWNLOAD_BYTES:
                raise ValueError("download too large")
        return bytes(content)


def last_synced(data_dir: Path | str) -> str | None:
    mark = Path(data_dir) / SYNC_MARK
    return mark.read_text(encoding="utf-8").strip() if mark.exists() else None


def sync_data(data_dir: Path | str, fetch: Callable[[str], bytes] = _download) -> dict:
    """Replace `data_dir` with the current `data` branch.

    On any failure the existing copy is left exactly as it was; the caller
    shows `has_local` / `synced_at` so the user knows how old the data is.
    """
    data_dir = Path(data_dir)
    fresh = data_dir.with_name(data_dir.name + ".new")
    old = data_dir.with_name(data_dir.name + ".old")
    try:
        archive = zipfile.ZipFile(io.BytesIO(fetch(DATA_URL)))
        shutil.rmtree(fresh, ignore_errors=True)
        fresh.mkdir(parents=True)
        root = fresh.resolve()
        for member in archive.infolist():
            parts = Path(member.filename).parts[1:]  # drop the "<repo>-data/" folder
            if member.is_dir() or not parts:
                continue
            target = (fresh.joinpath(*parts)).resolve()
            if root not in target.parents:
                raise ValueError("unsafe path in archive")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(member))
        if not (fresh / "universe" / "universe.parquet").exists():
            raise ValueError("archive has no universe table")
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        (fresh / SYNC_MARK).write_text(stamp, encoding="utf-8")
        shutil.rmtree(old, ignore_errors=True)
        if data_dir.exists():
            data_dir.rename(old)
        fresh.rename(data_dir)
        shutil.rmtree(old, ignore_errors=True)
        return {"ok": True, "synced_at": stamp, "has_local": True}
    except Exception as error:  # network, bad archive, locked files: keep what we have
        shutil.rmtree(fresh, ignore_errors=True)
        if not data_dir.exists() and old.exists():
            old.rename(data_dir)
        return {
            "ok": False, "error": type(error).__name__, "synced_at": last_synced(data_dir),
            "has_local": (data_dir / "universe" / "universe.parquet").exists(),
        }


# -- reading ----------------------------------------------------------------

def _json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def sessions(data_dir: Path | str) -> list[str]:
    """Days that have a daily result, newest first."""
    folder = Path(data_dir) / "daily"
    return sorted((path.stem for path in folder.glob("*.json")), reverse=True)


def day_view(data_dir: Path | str, day: str | None = None) -> dict | None:
    """Everything the "today's picks" screen needs, for `day` or the newest day."""
    data_dir = Path(data_dir)
    days = sessions(data_dir)
    day = day or (days[0] if days else None)
    if day is None:
        return None
    daily = _json(data_dir / "daily" / f"{day}.json")
    quality = _json(data_dir / "quality" / f"{day}.json")
    live = _json(data_dir / "scorecard" / "scorecard.json")
    simulated = _json(data_dir / "scorecard" / "simulated.json")
    return {
        "session": day, "daily": daily, "quality": quality, "live": live, "simulated": simulated,
        "message": format_message(daily, quality, live, simulated, session=day),
    }


def universe(data_dir: Path | str) -> pd.DataFrame:
    """The newest session's row per ticker."""
    table = Store(data_dir).read_table("universe/universe")
    if table.empty:
        return table
    return table[table["session"] == table["session"].max()].sort_values("ticker", ignore_index=True)


def rankings(data_dir: Path | str, day: str | None = None) -> pd.DataFrame:
    """The model's full ranking of eligible stocks on `day` (newest by default)."""
    ledger = Store(data_dir).read_table("ledger/predictions")
    if ledger.empty:
        return ledger
    date = pd.Timestamp(day) if day else ledger["date"].max()
    return ledger[ledger["date"] == date].sort_values("rank", ignore_index=True)


def stock_status(data_dir: Path | str, ticker: str) -> dict:
    """Is the stock eligible, why not, and where the model ranks it today."""
    table = universe(data_dir)
    row = table[table["ticker"] == ticker]
    if row.empty:
        return {"ticker": ticker, "known": False}
    row = row.iloc[0]
    reason = str(row["reason"] or "")
    status = {
        "ticker": ticker, "known": True, "session": pd.Timestamp(row["session"]).date().isoformat(),
        "eligible": bool(row["eligible"]), "reason": reason,
        "reason_text": REASON_TEXT.get(reason, f"Dışlandı ({reason})."),
        "median_turnover": float(row["median_turnover"]), "rank": None, "of": None, "prob": None,
        "recommended": False,
    }
    ranked = rankings(data_dir)
    if not ranked.empty and ranked["date"].max() == pd.Timestamp(row["session"]):
        mine = ranked[ranked["ticker"] == ticker]
        status["of"] = int(len(ranked))
        if not mine.empty:
            status.update(rank=int(mine.iloc[0]["rank"]), prob=float(mine.iloc[0]["prob"]),
                          recommended=bool(mine.iloc[0]["recommended"]))
    return status


def prices(data_dir: Path | str, ticker: str) -> pd.DataFrame:
    return Store(data_dir).read_prices(ticker)


def benchmark_close(data_dir: Path | str) -> pd.Series:
    return macro_store(data_dir).read_prices("BIST100")["Close"]


# -- indicators in plain words --------------------------------------------------

def _rsi(close: pd.Series) -> pd.Series:
    change = close.diff()
    gain, loss = _wilder(change.clip(lower=0)), _wilder(-change.clip(upper=0))
    rsi = 100.0 - 100.0 / (1.0 + gain / loss.replace(0, np.nan))
    return rsi.mask((loss == 0) & (gain > 0), 100.0).mask((loss == 0) & (gain == 0), 50.0)


def _pct(value: float, signed: bool = True) -> str:
    text = f"{value * 100:+.1f}" if signed else f"{value * 100:.1f}"
    return "%" + text.replace(".", ",")


def _num(value: float, decimals: int = 0) -> str:
    return f"{value:.{decimals}f}".replace(".", ",")


def indicator_readings(equity: pd.DataFrame, benchmark: pd.Series) -> list[dict]:
    """The latest bar's indicators as {name, value, reading, tone}.

    `tone` is "up", "down" or "flat": what the indicator by itself points to.
    These describe the chart; none of them is a forecast.
    """
    if len(equity) < 30:
        return []
    local = local_features(equity, benchmark).iloc[-1]
    extra = extra_features(equity, benchmark).iloc[-1]
    rsi = _rsi(equity["Close"]).iloc[-1]
    readings: list[dict] = []

    def add(name: str, value, text: str, reading: str, tone: str) -> None:
        if value is not None and np.isfinite(value):
            readings.append({"name": name, "value": text, "reading": reading, "tone": tone})

    add("RSI (14)", rsi, _num(rsi),
        "aşırı alım" if rsi > 70 else "aşırı satım" if rsi < 30 else "nötr bölge",
        "down" if rsi > 70 else "up" if rsi < 30 else "flat")
    for name, key in (("20 günlük ortalama", "EMA20_Distance"), ("50 günlük ortalama", "EMA50_Distance")):
        value = local[key]
        add(name, value, _pct(value),
            "fiyat ortalamanın üstünde" if value > 0 else "fiyat ortalamanın altında",
            "up" if value > 0 else "down")
    macd = local["MACD_Normalized"]
    add("MACD", macd, _pct(macd), "yukarı ivme" if macd > 0 else "aşağı ivme", "up" if macd > 0 else "down")
    adx = local["ADX_14"]
    add("Trend gücü (ADX)", adx, _num(adx),
        "güçlü trend" if adx > 25 else "yatay / zayıf trend" if adx < 20 else "orta güçte trend", "flat")
    cmf = local["CMF_20"]
    add("Para akışı (CMF)", cmf, _num(cmf, 2),
        "para girişi" if cmf > 0.05 else "para çıkışı" if cmf < -0.05 else "dengeli",
        "up" if cmf > 0.05 else "down" if cmf < -0.05 else "flat")
    surge = local["Volume_Surge"]
    add("Hacim", surge, _num(surge, 1) + "×",
        "20 günlük ortalamanın belirgin üstünde" if surge > 1.5
        else "ortalamanın altında" if surge < 0.7 else "olağan", "flat")
    stochastic = local["StochRSI_K"]
    add("StochRSI", stochastic, _num(stochastic),
        "aşırı alım" if stochastic > 80 else "aşırı satım" if stochastic < 20 else "nötr bölge",
        "down" if stochastic > 80 else "up" if stochastic < 20 else "flat")
    month = extra["Rel_Return_20D"]
    add("Endekse göre 1 ay", month, _pct(month),
        "endeksin önünde" if month > 0 else "endeksin gerisinde", "up" if month > 0 else "down")
    high = extra["High_252_Distance"]
    add("52 hafta zirvesi", high, _pct(high),
        "zirveye yakın" if high > -0.05 else "zirvenin epey altında" if high < -0.30 else "zirvenin altında",
        "flat")
    volatility = extra["Volatility_20D"]
    add("Günlük oynaklık (20 gün)", volatility, _pct(volatility, signed=False),
        "çok oynak" if volatility > 0.045 else "sakin" if volatility < 0.02 else "olağan", "flat")
    return readings


# -- scorecard and model history ------------------------------------------------

def model_history(data_dir: Path | str) -> pd.DataFrame:
    """One row per day: was the model retrained, and which checks passed."""
    path = Path(data_dir) / "models" / "history.jsonl"
    if not path.exists():
        return pd.DataFrame()
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        checks = entry.get("checks") or {}
        records.append({
            "session": entry.get("session"),
            "recipe_id": entry.get("recipe_id"),
            "retrained": bool(entry.get("retrained")),
            "weights_changed": bool(entry.get("weights_changed")),
            "data_ok": (checks.get("data") or {}).get("ok"),
            "health_ok": (checks.get("health") or {}).get("ok"),
            "drift_ok": (checks.get("drift") or {}).get("ok"),
            "rank_ic": (checks.get("health") or {}).get("mean_rank_ic"),
            "train_rows": (checks.get("rows") or {}).get("train_rows"),
        })
    return pd.DataFrame(records).sort_values("session", ascending=False, ignore_index=True)


def recipe_exams(data_dir: Path | str) -> pd.DataFrame:
    """One row per weekly exam: who challenged, and whether it was promoted."""
    folder = Path(data_dir) / "models" / "search"
    records = []
    for path in sorted(folder.glob("*.json"), reverse=True):
        entry = json.loads(path.read_text(encoding="utf-8"))
        exam = entry.get("exam") or {}
        records.append({
            "session": entry.get("session", path.stem),
            "candidates": entry.get("candidates"),
            "promoted": bool(entry.get("promoted")),
            "champion": entry.get("champion_recipe_id"),
            "challenger": entry.get("challenger_recipe_id"),
            "top_excess_diff": exam.get("top_excess_diff"),
            "rank_ic_diff": exam.get("rank_ic_diff"),
            "pick_metric": exam.get("pick_metric", "top_excess"),
            "pick_diff": exam.get("pick_diff", exam.get("top_excess_diff")),
            "p_pick": exam.get("p_pick", exam.get("p_top_excess")),
            "p_rank_ic": exam.get("p_rank_ic"),
            "confirmed": (entry.get("confirmation") or {}).get("ok"),
        })
    return pd.DataFrame(records)


def graded_recommendations(data_dir: Path | str) -> pd.DataFrame:
    """Live recommendations whose 5-session result is known, newest first."""
    ledger = Store(data_dir).read_table("ledger/predictions")
    if ledger.empty:
        return ledger
    done = ledger[ledger["recommended"].astype(bool) & ledger["matured"].astype(bool)]
    columns = ["date", "ticker", "rank", "entry_close", "exit_close", "fwd_return",
               "benchmark_return", "fwd_excess"]
    return done.sort_values(["date", "rank"], ascending=[False, True], ignore_index=True)[columns]
