import json

import pandas as pd
import pytest

import jarvis.train as train_module
import jarvis.walkforward as walkforward
from jarvis.eligibility import eligibility_today
from jarvis.exclusions import empty_exclusions
from jarvis.features import FEATURES
from jarvis.ledger import read_ledger
from jarvis.macro import macro_store
from jarvis.model import Recipe, load_model
from jarvis.store import Store
from jarvis.train import run_daily, run_search, run_simulation
from tests.jarvis.synthetic import CALENDAR, SESSION, build_market

AS_OF = pd.Timestamp("2026-11-13 19:00", tz="Europe/Istanbul")
SMALL = Recipe(params={"n_estimators": 15, "max_depth": 2, "learning_rate": 0.1}, features=FEATURES)


@pytest.fixture(autouse=True)
def small_minimums(monkeypatch):
    monkeypatch.setattr(train_module, "MIN_TRAIN_ROWS", 200)
    monkeypatch.setattr(walkforward, "MIN_TRAIN_ROWS", 200)
    monkeypatch.setattr(train_module, "HEALTH_SESSIONS", 20)
    monkeypatch.setattr(train_module, "SIM_SESSIONS", 20)
    monkeypatch.setattr(train_module, "EXAM_SESSIONS", 10)


def prepare(root, tickers=12, session=SESSION, training_allowed=True, index=CALENDAR):
    """A data folder as jarvis.pipeline would leave it for `session`."""
    store, names, _ = build_market(root, tickers=tickers, index=index)
    today = eligibility_today(store, names, session, empty_exclusions(), set())
    store.write_table("universe/universe", today)
    store.write_json(
        f"quality/{session.date().isoformat()}",
        {"session": session.date().isoformat(), "training_allowed": training_allowed,
         "quarantine_ratio": 0.0},
    )
    (root / "models").mkdir(exist_ok=True)
    (root / "models" / "recipe.json").write_text(json.dumps(SMALL.to_dict()), encoding="utf-8")
    return store, names


def daily(root, **kwargs):
    return run_daily(root, AS_OF, download=None, **kwargs)


def test_first_run_trains_and_records(tmp_path):
    store, names = prepare(tmp_path)
    summary = daily(tmp_path)
    assert summary["status"] == "ok" and summary["session"] == "2026-11-13"
    assert summary["retrained"] is True and summary["weights_changed"] is True
    model, recipe, meta = load_model(tmp_path)
    assert recipe == SMALL and meta["model_id"] == f"2026-11-13-{SMALL.recipe_id}"
    assert meta["trained_through"] <= CALENDAR[-8].date().isoformat()
    ledger = read_ledger(store)
    assert len(ledger) == 12 and (ledger["date"] == SESSION).all()
    daily_file = json.loads((tmp_path / "daily" / "2026-11-13.json").read_text(encoding="utf-8"))
    assert daily_file["model"]["checks"]["drift"] == {"ok": True, "reason": "no_previous_model"}
    assert len(daily_file["top"]) == 10 and daily_file["eligible"] == 12
    assert "threshold" not in daily_file
    history = (tmp_path / "models" / "history.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(history) == 1 and json.loads(history[0])["weights_changed"] is True
    assert (tmp_path / "scorecard" / "scorecard.json").exists()


def test_second_run_same_day_is_idempotent(tmp_path):
    store, _ = prepare(tmp_path)
    daily(tmp_path)
    first = read_ledger(store)
    daily(tmp_path)
    second = read_ledger(store)
    assert len(second) == len(first)
    pd.testing.assert_series_equal(first["prob"], second["prob"])
    history = (tmp_path / "models" / "history.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(history) == 1


def test_skips_without_quality_report(tmp_path):
    prepare(tmp_path)
    (tmp_path / "quality" / "2026-11-13.json").unlink()
    assert daily(tmp_path) == {"status": "skipped", "reason": "no_quality_report"}
    assert load_model(tmp_path) is None


def test_skips_when_benchmark_stale(tmp_path):
    prepare(tmp_path)
    target = macro_store(tmp_path)
    target.write_prices("BIST100", target.read_prices("BIST100").iloc[:-1], replace=True)
    assert daily(tmp_path) == {"status": "skipped", "reason": "macro_unavailable"}
    assert load_model(tmp_path) is None
    assert not (tmp_path / "daily").exists()


def test_training_not_allowed_keeps_champion_but_still_predicts(tmp_path):
    store, _ = prepare(tmp_path)
    daily(tmp_path)
    before = (tmp_path / "models" / "champion.json").read_bytes()
    report = tmp_path / "quality" / "2026-11-13.json"
    report.write_text(json.dumps({"session": "2026-11-13", "training_allowed": False}), encoding="utf-8")
    summary = daily(tmp_path)
    assert summary["retrained"] is False and summary["weights_changed"] is False
    assert (tmp_path / "models" / "champion.json").read_bytes() == before
    assert len(read_ledger(store)) == 12


def test_no_champion_and_training_blocked_records_nothing(tmp_path):
    store, _ = prepare(tmp_path, training_allowed=False)
    summary = daily(tmp_path)
    assert summary["status"] == "ok" and summary["recommendations"] == 0
    assert read_ledger(store).empty
    assert json.loads((tmp_path / "daily" / "2026-11-13.json").read_text(encoding="utf-8"))["model"]["model_id"] is None


def test_failed_health_check_makes_no_recommendations(tmp_path, monkeypatch):
    store, _ = prepare(tmp_path)
    monkeypatch.setattr(train_module, "health_check", lambda scores: {"ok": False, "mean_rank_ic": -0.1, "days": 120})
    summary = daily(tmp_path)
    assert summary["health_ok"] is False and summary["recommendations"] == 0
    assert not read_ledger(store)["recommended"].any()


def test_healthy_model_recommends_top_five_every_day(tmp_path, monkeypatch):
    store, _ = prepare(tmp_path)
    monkeypatch.setattr(train_module, "health_check", lambda scores: {"ok": True, "mean_rank_ic": 0.1, "days": 120})
    summary = daily(tmp_path)
    assert summary["recommendations"] == 5
    assert int(read_ledger(store)["recommended"].sum()) == 5
    daily_file = json.loads((tmp_path / "daily" / "2026-11-13.json").read_text(encoding="utf-8"))
    for item in daily_file["recommendations"]:
        assert 1 <= len(item["reasons"]) <= 2
        assert item["reasons"][0]["feature"] in FEATURES
        assert item["reasons"][0]["direction"] in ("up", "down")


def test_drift_failure_keeps_old_weights(tmp_path, monkeypatch):
    prepare(tmp_path)
    daily(tmp_path)
    before = (tmp_path / "models" / "champion.json").read_bytes()
    monkeypatch.setattr(train_module, "drift_check", lambda new, old: {"ok": False, "rank_corr": 0.1})
    summary = daily(tmp_path)
    assert summary["retrained"] is True and summary["weights_changed"] is False
    assert (tmp_path / "models" / "champion.json").read_bytes() == before


def test_fewer_eligible_than_top_n(tmp_path, monkeypatch):
    store, names = prepare(tmp_path)
    universe = store.read_table("universe/universe")
    universe.loc[~universe["ticker"].isin(names[:3]), ["eligible", "reason"]] = [False, "low_liquidity"]
    store.write_table("universe/universe", universe)
    monkeypatch.setattr(train_module, "health_check", lambda scores: {"ok": True, "mean_rank_ic": 0.1, "days": 120})
    summary = daily(tmp_path)
    assert summary["recommendations"] == 3
    assert len(read_ledger(store)) == 3


def test_matured_predictions_feed_live_scorecard(tmp_path):
    early = CALENDAR[-6]
    store, names = prepare(tmp_path, session=early, index=CALENDAR)
    as_of_early = pd.Timestamp(f"{early.date()} 19:00", tz="Europe/Istanbul")
    run_daily(tmp_path, as_of_early, download=None)
    today = eligibility_today(store, names, SESSION, empty_exclusions(), set())
    store.write_table("universe/universe", pd.concat([store.read_table("universe/universe"), today]))
    store.write_json("quality/2026-11-13", {"session": "2026-11-13", "training_allowed": True})
    summary = daily(tmp_path)
    assert summary["matured_today"] == 12
    card = json.loads((tmp_path / "scorecard" / "scorecard.json").read_text(encoding="utf-8"))
    assert card["kind"] == "live" and card["windows"]["all"]["days"] == 1


def test_simulation_does_not_touch_live_ledger(tmp_path):
    store, _ = prepare(tmp_path)
    summary = run_simulation(tmp_path, AS_OF, download=None)
    assert summary["status"] == "ok" and summary["days"] == 15  # 20 sessions, last 5 unlabelled
    assert read_ledger(store).empty
    simulated = store.read_table("ledger/simulated")
    assert simulated["date"].nunique() == 20 and (simulated["model_id"] == "simulation").all()
    card = json.loads((tmp_path / "scorecard" / "simulated.json").read_text(encoding="utf-8"))
    assert card["kind"] == "simulated" and "gerçek işlem sonucu değildir" in card["note"]
    assert load_model(tmp_path) is None


def test_search_promotes_only_through_gate(tmp_path, monkeypatch):
    prepare(tmp_path)
    recipe_path = tmp_path / "models" / "recipe.json"
    before = recipe_path.read_text(encoding="utf-8")
    monkeypatch.setattr(train_module, "sample_recipe", lambda rng, base: Recipe(
        params={"n_estimators": 10, "max_depth": 2, "learning_rate": float(rng.uniform(0.05, 0.2))},
        features=FEATURES))
    monkeypatch.setattr(train_module, "compare_recipes", lambda a, b: {"promote": False, "days": 10})
    result = run_search(tmp_path, AS_OF, candidates=2, download=None)
    assert result["promoted"] is False and recipe_path.read_text(encoding="utf-8") == before
    monkeypatch.setattr(train_module, "compare_recipes", lambda a, b: {"promote": True, "days": 10})
    monkeypatch.setattr(train_module, "confirm_recipes", lambda a, b: {"ok": False, "days": 15})
    result = run_search(tmp_path, AS_OF, candidates=2, download=None)
    assert result["promoted"] is False and result["confirmation"] == {"ok": False, "days": 15}
    assert recipe_path.read_text(encoding="utf-8") == before  # the long window said no
    monkeypatch.setattr(train_module, "confirm_recipes", lambda a, b: {"ok": True, "days": 15})
    result = run_search(tmp_path, AS_OF, candidates=2, download=None)
    assert result["promoted"] is True
    assert Recipe.from_dict(json.loads(recipe_path.read_text(encoding="utf-8"))).recipe_id == result["challenger_recipe_id"]
    assert (tmp_path / "models" / "search" / "2026-11-13.json").exists()


def test_structural_variants_cover_other_targets_and_extra_features():
    from jarvis.features import FEATURE_SETS
    from jarvis.train import structural_variants

    variants = structural_variants(SMALL)
    described = {(recipe.target, recipe.features) for recipe in variants}
    targets = ("sign", "cs_median", "risk_adjusted")
    expected = {(target, features) for target in targets for features in (FEATURES, *FEATURE_SETS)}
    assert described == expected - {("sign", FEATURES)}
    assert all(recipe.params == SMALL.params for recipe in variants)
    assert SMALL.recipe_id not in {recipe.recipe_id for recipe in variants}


def test_search_tries_structural_variants_first(tmp_path, monkeypatch):
    prepare(tmp_path)
    monkeypatch.setattr(train_module, "compare_recipes", lambda a, b: {"promote": True, "days": 10})
    monkeypatch.setattr(train_module, "confirm_recipes", lambda a, b: {"ok": True, "days": 15})
    result = run_search(tmp_path, AS_OF, candidates=5, download=None)
    saved = json.loads((tmp_path / "models" / "search" / "2026-11-13.json").read_text(encoding="utf-8"))
    assert [trial["target"] for trial in saved["trials"][:2]] == ["cs_median", "risk_adjusted"]
    assert {trial["features"] for trial in saved["trials"]} == {24, 33}
    assert len(saved["trials"]) == 5
    promoted = Recipe.from_dict(json.loads((tmp_path / "models" / "recipe.json").read_text(encoding="utf-8")))
    assert promoted.recipe_id == result["challenger_recipe_id"]
    assert daily(tmp_path)["status"] == "ok"  # the promoted recipe trains and predicts


def test_compare_reports_periods_and_a_shuffled_control(tmp_path):
    from jarvis.compare import compare, recipe_from_exam, shuffled_labels
    from jarvis.train import _prepare

    prepare(tmp_path)
    _, _, _, _, calendar, rows, _ = _prepare(tmp_path, AS_OF, None)
    challenger = Recipe(params=dict(SMALL.params), features=FEATURES, target="cs_median")

    shuffled = shuffled_labels(rows, "target_cs_median")
    assert not shuffled["target_cs_median"].equals(rows["target_cs_median"])
    by_day = lambda frame: frame.groupby("date")["target_cs_median"].sum()
    pd.testing.assert_series_equal(by_day(shuffled), by_day(rows))  # only moved within the day

    result = compare(rows, calendar, SMALL, challenger, sessions=30)
    assert set(result["periods"]) == {"earlier", "selection", "exam"}
    assert set(result["overall"]) == {"champion", "challenger", "challenger_shuffled"}
    assert result["overall"]["champion"]["days"] > 0
    # 30 sessions, the last 5 still ungraded: all 25 fall inside the 60-session exam window.
    assert result["periods"]["exam"]["champion"]["days"] == 25
    assert result["periods"]["earlier"]["champion"]["days"] == 0

    exam = tmp_path / "exam.json"
    exam.write_text(json.dumps({"challenger_recipe_id": challenger.recipe_id, "trials": [
        {"params": challenger.params, "target": "cs_median"}]}), encoding="utf-8")
    assert recipe_from_exam(exam) == challenger
    with pytest.raises(ValueError):
        recipe_from_exam(exam, "missing")


def test_daily_result_carries_the_bottom_of_a_large_enough_ranking(tmp_path, monkeypatch):
    prepare(tmp_path, tickers=24)
    monkeypatch.setattr(train_module, "health_check", lambda scores: {"ok": True})
    assert daily(tmp_path)["status"] == "ok"
    result = json.loads((tmp_path / "daily" / "2026-11-13.json").read_text(encoding="utf-8"))
    ledger = read_ledger(Store(tmp_path))
    worst_first = list(ledger.sort_values("rank", ascending=False)["ticker"].head(5))
    assert [item["ticker"] for item in result["bottom"]] == worst_first
    assert not set(worst_first) & {item["ticker"] for item in result["recommendations"]}


def test_no_bottom_list_for_a_small_ranking(tmp_path, monkeypatch):
    prepare(tmp_path, tickers=12)
    monkeypatch.setattr(train_module, "health_check", lambda scores: {"ok": True})
    daily(tmp_path)
    result = json.loads((tmp_path / "daily" / "2026-11-13.json").read_text(encoding="utf-8"))
    assert result["bottom"] == []
