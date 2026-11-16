import numpy as np
import pandas as pd

from jarvis.features import FEATURES
from jarvis.model import DEFAULT_RECIPE, Recipe, fit, load_model, predict, save_model

SMALL = Recipe(
    params={"n_estimators": 40, "max_depth": 3, "learning_rate": 0.1},
    features=FEATURES,
)


def synthetic_rows(n=2000, seed=0):
    """Target depends on one feature only, so a working model separates it well."""
    rng = np.random.default_rng(seed)
    data = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=list(FEATURES))
    data["target"] = (data["CMF_20"] + rng.normal(0, 0.1, n) > 0).astype(float)
    return data


def auc(labels, scores):
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    positives = labels == 1
    n_pos, n_neg = positives.sum(), (~positives).sum()
    return (ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def test_fit_predict_learns_separable_signal():
    train, test = synthetic_rows(seed=0), synthetic_rows(seed=1)
    model = fit(SMALL, train)
    probabilities = predict(model, SMALL, test)
    assert probabilities.index.equals(test.index)
    assert probabilities.between(0, 1).all()
    assert auc(test["target"].to_numpy(), probabilities.to_numpy()) > 0.9


def test_predict_is_deterministic():
    train, test = synthetic_rows(seed=0), synthetic_rows(seed=1)
    first = predict(fit(SMALL, train), SMALL, test)
    second = predict(fit(SMALL, train), SMALL, test)
    pd.testing.assert_series_equal(first, second)


def test_missing_macro_values_are_accepted():
    train, test = synthetic_rows(seed=0), synthetic_rows(seed=1)
    train.loc[::3, "macro_GOLD_1d_lag"] = np.nan
    test["macro_GOLD_1d_lag"] = np.nan
    assert predict(fit(SMALL, train), SMALL, test).notna().all()


def test_save_load_roundtrip_gives_same_predictions(tmp_path):
    train, test = synthetic_rows(seed=0), synthetic_rows(seed=1)
    model = fit(SMALL, train)
    save_model(tmp_path, model, SMALL, {"trained_through": "2026-11-13", "train_rows": 2000})
    loaded, recipe, meta = load_model(tmp_path)
    assert recipe == SMALL
    assert meta["trained_through"] == "2026-11-13" and meta["train_rows"] == 2000
    assert meta["recipe_id"] == SMALL.recipe_id
    pd.testing.assert_series_equal(predict(model, SMALL, test), predict(loaded, recipe, test))


def test_save_keeps_previous(tmp_path):
    train = synthetic_rows(seed=0)
    save_model(tmp_path, fit(SMALL, train), SMALL, {"trained_through": "2026-11-12"})
    save_model(tmp_path, fit(SMALL, train), SMALL, {"trained_through": "2026-11-13"})
    assert (tmp_path / "models" / "previous.json").exists()
    assert (tmp_path / "models" / "previous.meta.json").read_text(encoding="utf-8").count("2026-11-12") == 1
    assert load_model(tmp_path)[2]["trained_through"] == "2026-11-13"


def test_load_missing_returns_none(tmp_path):
    assert load_model(tmp_path) is None


def test_recipe_roundtrip_and_id():
    other = Recipe(params={**SMALL.params, "max_depth": 5}, features=FEATURES)
    assert other.recipe_id != SMALL.recipe_id
    assert Recipe.from_dict(SMALL.to_dict()) == SMALL
    assert len(SMALL.recipe_id) == 10
    assert DEFAULT_RECIPE.features == FEATURES and DEFAULT_RECIPE.train_years == 3
