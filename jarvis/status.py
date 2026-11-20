"""Has today's daily work already been done? Used by the workflow to skip repeat runs.

Usage:
    python -m jarvis.status --data-dir data      # prints session=YYYY-MM-DD and done=true|false
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from jarvis.calendar import last_completed_session


def daily_done(data_dir: Path | str, session: pd.Timestamp) -> bool:
    """True when the training job ranked stocks for `session`.

    A result file alone is not enough: when the data is stale the job still
    writes one with yesterday's model and no eligible stocks, and the later
    slots must then try again.
    """
    path = Path(data_dir) / "daily" / f"{pd.Timestamp(session).date().isoformat()}.json"
    if not path.exists():
        return False
    try:
        daily = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return False
    return bool((daily.get("model") or {}).get("model_id")) and int(daily.get("eligible") or 0) > 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--as-of", default=None)
    arguments = parser.parse_args(argv)
    as_of = pd.Timestamp(arguments.as_of) if arguments.as_of else None
    session = last_completed_session(as_of)
    done = daily_done(arguments.data_dir, session)
    sys.stdout.write(f"session={session.date().isoformat()}\ndone={'true' if done else 'false'}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
