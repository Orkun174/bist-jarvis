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

# --- Part 2: training, ledger, gate ------------------------------------
HORIZON = 5                    # sessions ahead for the target
EMBARGO_SESSIONS = 7           # extra gap between last training label and prediction day
TRAIN_YEARS = 3
MIN_TRAIN_ROWS = 20_000
TOP_N = 5
COST_BPS = 20.0                # round-trip cost charged once per pick
EXAM_SESSIONS = 60
REFIT_EVERY = 5
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_CONFIDENCE = 0.80
HEALTH_SESSIONS = 120
DRIFT_MIN_RANK_CORR = 0.5
SIM_SESSIONS = 250
RANDOM_SEED = 42
