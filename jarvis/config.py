"""Thresholds and shared constants for the daily data pipeline.

Every number here comes from the design document
(docs/superpowers/specs/2026-11-16-self-improving-full-bist-design.md).
"""

TRT = "Europe/Istanbul"
CALENDAR = "XIST"
OHLCV = ["Open", "High", "Low", "Close", "Volume"]

# Price history.
HISTORY_YEARS = 5

# Data quality.
CROSS_CHECK_TOLERANCE = 0.01   # Yahoo vs Is Yatirim close, relative difference
MAX_QUARANTINE_RATIO = 0.20    # above this share of quarantined tickers, no training
SUSPECT_JUMP = 0.25            # |daily return| above this on adjusted data is suspect
QUALITY_WINDOW = 20            # recent sessions inspected by the quality checks

# Eligibility (manipulation filter).
MIN_HISTORY_SESSIONS = 252
LIQUIDITY_WINDOW = 20
MIN_MEDIAN_TURNOVER_TRY = 50_000_000
LIMIT_MOVE = 0.095             # |daily return| at or above this counts as a limit day
MAX_LIMIT_DAYS = 5

# Universe sanity floor: fewer symbols than this means the source page broke.
MIN_UNIVERSE_SIZE = 300
