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

# Yahoo publishing late. More than this share of tickers without the session bar
# means the source has not published it yet (2026-11-19: 626 of 629), not that
# the stocks are bad. Prices and macro series are then retried after a wait.
SOURCE_LAG_RATIO = 0.5
LAG_RETRIES = 3
LAG_WAIT_SECONDS = 15 * 60
# Yahoo also drops the previous session for some hours after midnight UTC
# (2026-11-21: present 03:08 TRT, gone 03:58-06:13, back by 10:38), and GitHub's
# queue often starts the evening job right in that gap. The price job then keeps
# retrying until this hour (TRT) of the day after the session, within a cap that
# keeps the whole job under GitHub's six-hour limit.
LAG_DEADLINE_HOUR = 11
LAG_MAX_WAIT_SECONDS = 285 * 60

# Eligibility (manipulation filter).
MIN_HISTORY_SESSIONS = 252
LIQUIDITY_WINDOW = 20
MIN_MEDIAN_TURNOVER_TRY = 50_000_000
LIMIT_MOVE = 0.095             # |daily return| at or above this counts as a limit day
MAX_LIMIT_DAYS = 5
# Erratic-behaviour rule: a point-in-time proxy for the exchange's volatility
# based measures (VBTS), whose history is not available. Thresholds were chosen
# against the Aug-Sep 2026 measure list only (never against returns): the rule
# flagged 72% of the listed liquid names the day before their measure began,
# at the cost of also excluding about 14% of other liquid names.
VOLATILITY_WINDOW = 20
MAX_VOLATILITY_20 = 0.045      # std of daily returns over the window
RUNUP_WINDOW = 120
MAX_RUNUP_120 = 1.0            # price more than doubled over the window
ERRATIC_COOLDOWN = 20          # sessions the exclusion lasts after the last trigger

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
