"""One probe round: does Yahoo already serve the last completed BIST session?

Asks Yahoo for a few BIST symbols in three ways and appends one CSV row per
symbol and way:

  prod    start=session-3d, end=session+1d   (exactly what jarvis.prices does)
  wide    start=session-3d, end=session+3d   (end bound further out)
  period  period="5d", no start/end

If prod misses the session while wide or period has it, the problem is the
request form, not Yahoo's data. Temporary: removed once the empty hours are known.

    python tools/yahoo_probe.py probe/yahoo_probe.csv
"""

import csv
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jarvis.calendar import last_completed_session  # noqa: E402

SYMBOLS = ["THYAO.IS", "GARAN.IS", "ASELS.IS", "ARCLK.IS", "TUKAS.IS", "XU100.IS"]
HEADER = ["probed_at_trt", "session", "way", "symbol", "last_bar", "verdict"]


def last_date(frame) -> str:
    if frame is None or frame.empty:
        return ""
    index = pd.DatetimeIndex(frame.dropna(how="all").index)
    if index.tz is not None:
        index = index.tz_localize(None)
    return str(index.max().date()) if len(index) else ""


def probe() -> list[list]:
    now = pd.Timestamp.now(tz="Europe/Istanbul")
    session = last_completed_session(now)
    day = pd.Timedelta(days=1)
    ways = {
        "prod": dict(start=(session - 3 * day).date().isoformat(),
                     end=(session + day).date().isoformat()),
        "wide": dict(start=(session - 3 * day).date().isoformat(),
                     end=(session + 3 * day).date().isoformat()),
        "period": dict(period="5d"),
    }
    stamp = now.isoformat(timespec="seconds")
    rows = []
    for way, kwargs in ways.items():
        try:
            raw = yf.download(SYMBOLS, auto_adjust=True, group_by="ticker", threads=False,
                              progress=False, **kwargs)
        except Exception as error:  # record and continue
            rows += [[stamp, session.date(), way, s, "", f"error:{type(error).__name__}"]
                     for s in SYMBOLS]
            continue
        for symbol in SYMBOLS:
            multi = isinstance(raw.columns, pd.MultiIndex)
            frame = raw[symbol] if multi and symbol in raw.columns.get_level_values(0) else None
            last = last_date(frame)
            verdict = "has_session" if last and last >= str(session.date()) else "missing"
            rows.append([stamp, session.date(), way, symbol, last, verdict])
    return rows


def main(path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = probe()
    new = not out.exists()
    with out.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if new:
            writer.writerow(HEADER)
        writer.writerows(rows)
    for way in ("prod", "wide", "period"):
        hits = sum(r[5] == "has_session" for r in rows if r[2] == way)
        print(f"{rows[0][0]} session {rows[0][1]} {way}: {hits}/{len(SYMBOLS)}")


if __name__ == "__main__":
    main(sys.argv[1])
