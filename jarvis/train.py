"""Daily training job, simulated scorecard and weekly recipe search.

Usage (after `python -m jarvis.pipeline --data-dir data`):
    python -m jarvis.train --data-dir data              # daily refit + predictions
    python -m jarvis.train --data-dir data --simulate   # walk-forward scorecard
    python -m jarvis.train --data-dir data --search     # weekly recipe exam
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from jarvis.calendar import last_completed_session
from jarvis.config import (
    EXAM_SESSIONS, HEALTH_SESSIONS, HORIZON, LAG_RETRIES, LAG_WAIT_SECONDS, MIN_TRAIN_ROWS,
    SIM_SESSIONS,
)
from jarvis.dataset import build_rows, prediction_slice, training_slice
from jarvis.features import FEATURE_SETS, feature_set_name
from jarvis.gate import compare_recipes, confirm_recipes, drift_check, health_check
from jarvis.ledger import LEDGER_COLUMNS, build_entries, empty_ledger, mature, read_ledger, record
from jarvis.macro import REQUIRED_MACRO, load_macro, update_macro
from jarvis.metrics import daily_scores, summarize
from jarvis.model import (
    DEFAULT_RECIPE, Recipe, explain, fit, load_model, predict, save_model, trainable,
)
from jarvis.dataset import TARGET_COLUMNS
from jarvis.prices import Downloader, yahoo_download
from jarvis.scorecard import scorecard
from jarvis.store import Store
from jarvis.walkforward import walk_forward

LOG = logging.getLogger("jarvis.train")
SIMULATED_TABLE = "ledger/simulated"
VIX_MAX_AGE = pd.Timedelta(days=7)
BOTTOM_N = 5
MIN_NAMES_FOR_BOTTOM = 20


# -- shared loading ---------------------------------------------------------

class Skip(Exception):
    """The run cannot proceed today; nothing is changed."""


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_recipe(data_dir: Path) -> Recipe:
    saved = _read_json(data_dir / "models" / "recipe.json")
    return Recipe.from_dict(saved) if saved else DEFAULT_RECIPE


def _macro_ready(macro: dict[str, pd.DataFrame], session: pd.Timestamp) -> bool:
    benchmark, vix = macro["BIST100"], macro["VIX"]
    benchmark_ok = not benchmark.empty and benchmark.index[-1] >= session
    vix_ok = not vix.empty and vix.index[-1] >= session - VIX_MAX_AGE
    return benchmark_ok and vix_ok


def _prepare(data_dir: Path, as_of, download: Downloader | None, wait: Callable = time.sleep):
    """Common start: session, macro series, calendar, universe and rows."""
    session = last_completed_session(as_of)
    store = Store(data_dir)
    report = _read_json(data_dir / "quality" / f"{session.date().isoformat()}.json")
    if report is None:
        raise Skip("no_quality_report")

    if download is not None:
        update_macro(data_dir, session, download=download)
    macro = load_macro(data_dir)
    # When the price job already waited for Yahoo in vain, waiting again only burns the job limit.
    retries = LAG_RETRIES if download is not None and not report.get("source_lag") else 0
    for attempt in range(retries):
        if _macro_ready(macro, session):
            break
        LOG.warning(
            "BIST100/VIX %s seansı için henüz yok; %d dk sonra yeniden denenecek (%d/%d).",
            session.date(), LAG_WAIT_SECONDS // 60, attempt + 1, retries,
        )
        wait(LAG_WAIT_SECONDS)
        update_macro(data_dir, session, download=download)
        macro = load_macro(data_dir)
    if not _macro_ready(macro, session):
        raise Skip("macro_unavailable")
    benchmark = macro["BIST100"]
    assert set(REQUIRED_MACRO) == {"BIST100", "VIX"}

    universe = store.read_table("universe/universe")
    if universe.empty:
        raise Skip("no_universe")
    today = universe[universe["session"] == session]
    if today.empty:
        raise Skip("no_universe")
    tickers = sorted(today["ticker"])
    eligible_today = set(today[today["eligible"].astype(bool)]["ticker"])

    calendar = pd.DatetimeIndex(benchmark.index[benchmark.index <= session])
    LOG.info("Seans %s: %d hisse için satırlar hazırlanıyor.", session.date(), len(tickers))
    rows = build_rows(store, macro, tickers, session)
    return session, store, report, macro, calendar, rows, eligible_today


def _graded_window(calendar: pd.DatetimeIndex, length: int, skip: int = 0):
    """The `length` most recent sessions whose labels are already final."""
    end_position = len(calendar) - 1 - HORIZON - skip
    start_position = max(end_position - length + 1, 0)
    if end_position < 0:
        return None
    return calendar[start_position], calendar[end_position]


def _write_history(data_dir: Path, entry: dict) -> None:
    path = data_dir / "models" / "history.jsonl"
    lines = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip() and json.loads(line).get("session") != entry["session"]:
                lines.append(line)
    lines.append(json.dumps(entry, ensure_ascii=False, sort_keys=True, allow_nan=False))
    lines.sort(key=lambda line: json.loads(line)["session"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _clean(value):
    """Make numpy/pandas scalars and NaN safe for strict JSON."""
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else float(value)
    return value


# -- daily ------------------------------------------------------------------

def run_daily(
    data_dir: Path | str, as_of=None, *, download: Downloader | None = yahoo_download,
    wait: Callable = time.sleep,
) -> dict:
    data_dir = Path(data_dir)
    try:
        session, store, report, macro, calendar, rows, eligible_today = _prepare(
            data_dir, as_of, download, wait
        )
    except Skip as skip:
        LOG.warning("Eğitim atlandı: %s", skip)
        return {"status": "skipped", "reason": str(skip)}
    day = session.date().isoformat()

    matured_today = mature(store, store, macro["BIST100"], calendar, session)
    ledger = read_ledger(store)
    live_card = scorecard(ledger, session, "live", calendar=calendar)
    store.write_json("scorecard/scorecard", _clean(live_card))

    recipe = load_recipe(data_dir)
    train = training_slice(rows, session, calendar, train_years=recipe.train_years)
    checks: dict = {
        "data": {
            "ok": bool(report.get("training_allowed", False)),
            "quarantine_ratio": report.get("quarantine_ratio"),
        },
        "rows": {"ok": bool(len(train) >= MIN_TRAIN_ROWS), "train_rows": int(len(train))},
    }

    window = _graded_window(calendar, HEALTH_SESSIONS)
    health_scores = daily_scores(
        walk_forward(rows, recipe, calendar, window[0], window[1])
    ) if window else daily_scores(pd.DataFrame(columns=["date", "ticker", "prob", "fwd_excess"]))
    checks["health"] = health_check(health_scores)

    today_rows = prediction_slice(rows, session, eligible_today)
    champion = load_model(data_dir)
    old_probs = None
    if champion is not None and not today_rows.empty:
        old_probs = predict(champion[0], champion[1], today_rows).set_axis(today_rows["ticker"])

    retrained = weights_changed = False
    active = champion
    if checks["data"]["ok"] and checks["rows"]["ok"] and trainable(recipe, train):
        LOG.info("Model %d satırla yeniden eğitiliyor.", len(train))
        model = fit(recipe, train)
        retrained = True
        new_probs = predict(model, recipe, today_rows).set_axis(today_rows["ticker"])
        checks["drift"] = drift_check(new_probs, old_probs)
        if checks["drift"]["ok"]:
            meta = {
                "model_id": f"{day}-{recipe.recipe_id}",
                "trained_through": train["label_end"].max().date().isoformat(),
                "train_rows": int(len(train)),
            }
            save_model(data_dir, model, recipe, meta)
            active = load_model(data_dir)
            weights_changed = True

    recommendations: list[dict] = []
    top: list[dict] = []
    bottom: list[dict] = []
    if active is not None and not today_rows.empty:
        model, active_recipe, meta = active
        predictions = pd.DataFrame(
            {
                "ticker": today_rows["ticker"].to_numpy(),
                "prob": predict(model, active_recipe, today_rows).to_numpy(),
                "entry_close": today_rows["close"].to_numpy(),
            }
        )
        record(
            store, session, predictions, meta.get("model_id", "unknown"),
            meta.get("recipe_id", active_recipe.recipe_id),
            allow_recommendations=bool(checks["health"]["ok"]),
        )
        todays = read_ledger(store)
        todays = todays[todays["date"] == session]
        as_item = lambda row: {"ticker": row.ticker, "prob": round(float(row.prob), 4), "rank": int(row.rank)}
        recommendations = [as_item(row) for row in todays[todays["recommended"]].itertuples()]
        if recommendations:
            picked = today_rows.set_index("ticker").loc[[item["ticker"] for item in recommendations]]
            # Macro features are the same for every stock today, and the eligible
            # stocks share much of their profile; name what sets each pick apart.
            stock_specific = tuple(
                name for name in active_recipe.features if not name.startswith("macro_")
            )
            explained = explain(
                model, active_recipe, picked, only=stock_specific, reference=today_rows
            )
            for item, reasons in zip(recommendations, explained):
                item["reasons"] = reasons
        top = [as_item(row) for row in todays.head(10).itertuples()]
        # The lowest ranks, worst first: in simulation this end of the ranking was
        # the model's most reliable. Only with a healthy model and a real ranking.
        if checks["health"]["ok"] and len(todays) >= MIN_NAMES_FOR_BOTTOM:
            bottom = [as_item(row) for row in todays.tail(BOTTOM_N).iloc[::-1].itertuples()]

    model_info = {
        "model_id": active[2].get("model_id") if active else None,
        "recipe_id": active[1].recipe_id if active else None,
        "retrained": retrained,
        "weights_changed": weights_changed,
        "checks": checks,
    }
    store.write_json(
        f"daily/{day}",
        _clean(
            {
                "session": day,
                "recommendations": recommendations,
                "top": top,
                "bottom": bottom,
                "eligible": len(today_rows),
                "model": model_info,
                "scorecard_live": live_card["windows"]["60"],
            }
        ),
    )
    _write_history(data_dir, _clean({"session": day, **model_info}))

    return {
        "status": "ok",
        "session": day,
        "retrained": retrained,
        "weights_changed": weights_changed,
        "recommendations": len(recommendations),
        "matured_today": int(matured_today),
        "train_rows": int(len(train)),
        "health_ok": bool(checks["health"]["ok"]),
    }


# -- simulation -------------------------------------------------------------

def _simulated_ledger(predictions: pd.DataFrame, rows: pd.DataFrame, recipe: Recipe) -> pd.DataFrame:
    closes = rows[["date", "ticker", "close"]].rename(columns={"close": "entry_close"})
    merged = predictions.merge(closes, on=["date", "ticker"], how="left")
    pieces = []
    for day, group in merged.groupby("date", sort=True):
        entries = build_entries(day, group[["ticker", "prob", "entry_close"]], "simulation",
                                recipe.recipe_id)
        excess = group.set_index("ticker")["fwd_excess"].reindex(entries["ticker"]).to_numpy()
        entries["fwd_excess"] = excess
        entries["matured"] = ~np.isnan(excess)
        pieces.append(entries)
    if not pieces:
        return empty_ledger()
    return pd.concat(pieces, ignore_index=True)[LEDGER_COLUMNS]


def run_simulation(data_dir: Path | str, as_of=None, *, download: Downloader | None = yahoo_download) -> dict:
    """Replay the current recipe over the last SIM_SESSIONS sessions (not live results)."""
    data_dir = Path(data_dir)
    try:
        session, store, _, _, calendar, rows, _ = _prepare(data_dir, as_of, download)
    except Skip as skip:
        return {"status": "skipped", "reason": str(skip)}
    recipe = load_recipe(data_dir)
    start = calendar[max(len(calendar) - SIM_SESSIONS, 0)]
    LOG.info("Simülasyon: %s - %s", start.date(), session.date())
    predictions = walk_forward(rows, recipe, calendar, start, session)
    simulated = _simulated_ledger(predictions, rows, recipe)
    store.write_table(SIMULATED_TABLE, simulated)
    card = scorecard(simulated, session, "simulated", calendar=calendar)
    net = summarize(daily_scores(predictions))
    card["top5_net_of_cost"] = net
    card["recipe_id"] = recipe.recipe_id
    card["note"] = "Geçmişe dönük ileriye yürütme; gerçek işlem sonucu değildir."
    store.write_json("scorecard/simulated", _clean(card))
    return {"status": "ok", "session": session.date().isoformat(), "days": net["days"],
            "mean_top_excess": net["mean_top_excess"], "mean_rank_ic": net["mean_rank_ic"],
            "hit_rate": net["hit_rate"], "recommendations": card["windows"]["all"]["recommendations"],
            "recommendation_hit_rate": card["windows"]["all"]["hit_rate"],
            "recommendation_mean_excess": card["windows"]["all"]["mean_excess"]}


# -- weekly recipe search ---------------------------------------------------

def sample_recipe(rng: np.random.Generator, base: Recipe) -> Recipe:
    """A random recipe from the ranges the original trainer searched."""
    params = {
        "n_estimators": int(rng.choice(np.arange(150, 651, 50))),
        "max_depth": int(rng.integers(3, 8)),
        "learning_rate": float(np.exp(rng.uniform(np.log(0.01), np.log(0.1)))),
        "min_child_weight": float(rng.uniform(1.0, 5.0)),
        "subsample": float(rng.uniform(0.6, 1.0)),
        "colsample_bytree": float(rng.uniform(0.6, 1.0)),
        "colsample_bylevel": float(rng.uniform(0.6, 1.0)),
        "reg_alpha": float(rng.uniform(0.0, 1.0)),
        "reg_lambda": float(rng.uniform(0.0, 2.0)),
    }
    return Recipe(
        params=params, features=base.features, train_years=base.train_years, target=base.target
    )


def structural_variants(champion: Recipe) -> list[Recipe]:
    """The champion's own settings with each other target and feature set.

    These ask one question at a time ("does this target help?", "do the extra
    features help?") before random hyperparameters are tried on top.
    """
    variants = []
    for features in dict.fromkeys((champion.features, *FEATURE_SETS)):
        for target in TARGET_COLUMNS:
            recipe = Recipe(
                params=dict(champion.params), features=features,
                train_years=champion.train_years, target=target,
            )
            if recipe.recipe_id != champion.recipe_id:
                variants.append(recipe)
    return variants


def run_search(
    data_dir: Path | str, as_of=None, candidates: int = 18, seed: int | None = None,
    *, download: Downloader | None = yahoo_download,
) -> dict:
    """Try random recipes; one may replace the champion only through the exam.

    Candidates are the champion's structural variants (other targets, the
    extra features) followed by random hyperparameters on top of one of them.
    The best candidate is *chosen* on an earlier 60-session window and then
    *examined* against the champion on the most recent 60 sessions, which
    played no part in choosing it. A challenger that passes is promoted only
    if the sessions before both windows confirm it (gate.confirm_recipes).

    It is chosen by rank IC, not by the return of its top picks: the top-pick
    return of 60 days is mostly noise, while rank IC uses every stock every
    day. The exam still demands both.
    """
    data_dir = Path(data_dir)
    try:
        session, store, _, _, calendar, rows, _ = _prepare(data_dir, as_of, download)
    except Skip as skip:
        return {"status": "skipped", "reason": str(skip)}
    day = session.date().isoformat()
    champion = load_recipe(data_dir)
    exam = _graded_window(calendar, EXAM_SESSIONS)
    selection = _graded_window(calendar, EXAM_SESSIONS, skip=EXAM_SESSIONS)
    rng = np.random.default_rng(int(session.strftime("%Y%m%d")) if seed is None else seed)

    def score(recipe: Recipe, window) -> pd.DataFrame:
        return daily_scores(walk_forward(rows, recipe, calendar, window[0], window[1]))

    variants = structural_variants(champion)
    bases = [champion, *variants]
    trials = []
    best: tuple[float, Recipe] | None = None
    for number in range(candidates):
        if number < len(variants):
            recipe = variants[number]
        else:
            recipe = sample_recipe(rng, bases[int(rng.integers(len(bases)))])
        summary = summarize(score(recipe, selection))
        trials.append({
            "recipe_id": recipe.recipe_id, "params": recipe.params, "target": recipe.target,
            "features": len(recipe.features), "feature_set": feature_set_name(recipe.features),
            "train_years": recipe.train_years, "selection": summary,
        })
        value = summary["mean_rank_ic"]
        LOG.info("Aday %d/%d (%s, %s): seçim penceresi rank IC %s",
                 number + 1, candidates, recipe.target, feature_set_name(recipe.features), value)
        if value is not None and (best is None or value > best[0]):
            best = (value, recipe)

    result: dict = {"status": "ok", "session": day, "candidates": candidates, "promoted": False,
                    "champion_recipe_id": champion.recipe_id}
    if best is not None:
        challenger = best[1]
        champion_scores, challenger_scores = score(champion, exam), score(challenger, exam)
        verdict = compare_recipes(champion_scores, challenger_scores)
        result.update(
            {
                "challenger_recipe_id": challenger.recipe_id,
                "exam": verdict,
                "champion_exam": summarize(champion_scores),
                "challenger_exam": summarize(challenger_scores),
            }
        )
        if verdict["promote"]:
            # The exam can be passed by luck; ask sessions that played no part in
            # choosing or examining the challenger (the ones before both windows).
            earlier = _graded_window(
                calendar, max(SIM_SESSIONS - 2 * EXAM_SESSIONS, EXAM_SESSIONS), skip=2 * EXAM_SESSIONS
            )
            if earlier is None:
                confirmation = {"ok": False, "reason": "no_earlier_sessions"}
            else:
                confirmation = confirm_recipes(score(champion, earlier), score(challenger, earlier))
                confirmation["window"] = [earlier[0].date().isoformat(), earlier[1].date().isoformat()]
            result["confirmation"] = confirmation
        if verdict["promote"] and confirmation["ok"]:
            path = data_dir / "models" / "recipe.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(challenger.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
            result["promoted"] = True
    store.write_json(f"models/search/{day}", _clean({**result, "trials": trials}))
    return _clean(result)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--as-of", default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--simulate", action="store_true")
    mode.add_argument("--search", action="store_true")
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    as_of = pd.Timestamp(arguments.as_of) if arguments.as_of else None
    if arguments.simulate:
        summary = run_simulation(arguments.data_dir, as_of)
    elif arguments.search:
        summary = run_search(arguments.data_dir, as_of)
    else:
        summary = run_daily(arguments.data_dir, as_of)
    sys.stdout.write(json.dumps(_clean(summary), ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
