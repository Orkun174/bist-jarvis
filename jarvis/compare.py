"""Compare a challenger recipe with the champion over a long walk-forward window.

A research tool, not part of the daily job and never a way to promote a recipe:
promotion only happens through the weekly exam. It answers two questions the
60-session exam cannot:

- Does the challenger's advantage hold in periods that played no part in
  choosing it (the sessions before the search's selection window)?
- Does the advantage vanish when the challenger's labels are shuffled within
  each day? If it does not, something is leaking.

Usage:
    python -m jarvis.compare --data-dir data --exam data/models/search/2026-11-16.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from jarvis.config import EXAM_SESSIONS, HORIZON, RANDOM_SEED, SIM_SESSIONS, TRAIN_YEARS
from jarvis.features import FEATURE_SETS, FEATURES
from jarvis.gate import compare_recipes
from jarvis.metrics import daily_scores, summarize
from jarvis.model import Recipe
from jarvis.walkforward import walk_forward

LOG = logging.getLogger("jarvis.compare")


def shuffled_labels(rows: pd.DataFrame, column: str, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """A copy of `rows` whose `column` is permuted among each day's rows."""
    rng = np.random.default_rng(seed)
    result = rows.copy()
    result[column] = rows.groupby("date")[column].transform(lambda values: rng.permutation(values.to_numpy()))
    return result


def periods(scores: pd.DataFrame, calendar: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    """Split daily scores into the exam window, the selection window and everything earlier."""
    graded_end = len(calendar) - 1 - HORIZON
    exam_start = calendar[max(graded_end - EXAM_SESSIONS + 1, 0)]
    selection_start = calendar[max(graded_end - 2 * EXAM_SESSIONS + 1, 0)]
    return {
        "earlier": scores[scores.index < selection_start],
        "selection": scores[(scores.index >= selection_start) & (scores.index < exam_start)],
        "exam": scores[scores.index >= exam_start],
    }


def compare(rows: pd.DataFrame, calendar: pd.DatetimeIndex, champion: Recipe, challenger: Recipe,
            sessions: int = SIM_SESSIONS) -> dict:
    start = calendar[max(len(calendar) - sessions, 0)]
    end = calendar[-1]

    def score(recipe: Recipe, frame: pd.DataFrame) -> pd.DataFrame:
        return daily_scores(walk_forward(frame, recipe, calendar, start, end))

    scores = {
        "champion": score(champion, rows),
        "challenger": score(challenger, rows),
        "challenger_shuffled": score(challenger, shuffled_labels(rows, challenger.target_column)),
    }
    result: dict = {
        "champion": champion.to_dict() | {"recipe_id": champion.recipe_id},
        "challenger": challenger.to_dict() | {"recipe_id": challenger.recipe_id},
        "overall": {name: summarize(frame) for name, frame in scores.items()},
        "overall_verdict": compare_recipes(scores["champion"], scores["challenger"]),
        "periods": {},
    }
    split = {name: periods(frame, calendar) for name, frame in scores.items()}
    for period in ("earlier", "selection", "exam"):
        result["periods"][period] = {
            "champion": summarize(split["champion"][period]),
            "challenger": summarize(split["challenger"][period]),
            "verdict": compare_recipes(split["champion"][period], split["challenger"][period]),
        }
    return result


def recipe_from_exam(path: Path, recipe_id: str | None = None) -> Recipe:
    """The challenger of a saved weekly exam (or the trial with `recipe_id`)."""
    exam = json.loads(Path(path).read_text(encoding="utf-8"))
    wanted = recipe_id or exam["challenger_recipe_id"]
    for trial in exam["trials"]:
        for features in (FEATURES, *FEATURE_SETS):
            recipe = Recipe(params=trial["params"], features=features, target=trial.get("target", "sign"),
                            train_years=int(trial.get("train_years", TRAIN_YEARS)))
            if recipe.recipe_id == wanted:
                return recipe
    raise ValueError(f"recipe {wanted} not found in {path}")


def main(argv: list[str] | None = None) -> int:
    from jarvis.train import _clean, _prepare, load_recipe

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--exam", required=True, help="a models/search/<day>.json file")
    parser.add_argument("--recipe-id", default=None)
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--sessions", type=int, default=SIM_SESSIONS)
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    as_of = pd.Timestamp(arguments.as_of) if arguments.as_of else None
    _, _, _, _, calendar, rows, _ = _prepare(Path(arguments.data_dir), as_of, None)
    challenger = recipe_from_exam(Path(arguments.exam), arguments.recipe_id)
    result = compare(rows, calendar, load_recipe(Path(arguments.data_dir)), challenger, arguments.sessions)
    sys.stdout.write(json.dumps(_clean(result), ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
