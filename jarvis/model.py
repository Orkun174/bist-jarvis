"""Recipe (hyperparameters + features + training window), training and persistence."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from jarvis.config import RANDOM_SEED, TRAIN_YEARS
from jarvis.dataset import TARGET_COLUMNS
from jarvis.features import FEATURE_VERSION, FEATURES


@dataclass(frozen=True)
class Recipe:
    params: dict = field(hash=False)
    features: tuple[str, ...] = FEATURES
    train_years: int = TRAIN_YEARS
    target: str = "sign"

    def __post_init__(self) -> None:
        if self.target not in TARGET_COLUMNS:
            raise ValueError(f"unknown target: {self.target}")

    @property
    def target_column(self) -> str:
        return TARGET_COLUMNS[self.target]

    def to_dict(self) -> dict:
        data = {
            "params": dict(self.params),
            "features": list(self.features),
            "train_years": self.train_years,
        }
        if self.target != "sign":  # left out by default so existing recipe ids stay the same
            data["target"] = self.target
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Recipe":
        return cls(
            params=dict(data["params"]),
            features=tuple(data["features"]),
            train_years=int(data.get("train_years", TRAIN_YEARS)),
            target=data.get("target", "sign"),
        )

    @property
    def recipe_id(self) -> str:
        text = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


DEFAULT_RECIPE = Recipe(
    params={
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.03,
        "min_child_weight": 3.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    },
)


def _matrix(recipe: Recipe, rows: pd.DataFrame) -> pd.DataFrame:
    return rows.loc[:, list(recipe.features)].astype(np.float32)


def new_classifier(recipe: Recipe) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        tree_method="hist",
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        n_jobs=2,
        **recipe.params,
    )


def trainable(recipe: Recipe, train: pd.DataFrame) -> bool:
    """Both outcomes of the recipe's target are present."""
    return train[recipe.target_column].nunique() == 2


def fit(recipe: Recipe, train: pd.DataFrame) -> XGBClassifier:
    train = train[train[recipe.target_column].notna()]
    model = new_classifier(recipe)
    model.fit(_matrix(recipe, train), train[recipe.target_column].astype(int))
    return model


def predict(model: XGBClassifier, recipe: Recipe, rows: pd.DataFrame) -> pd.Series:
    if rows.empty:
        return pd.Series(dtype=float, index=rows.index, name="prob")
    probabilities = model.predict_proba(_matrix(recipe, rows))[:, 1]
    return pd.Series(probabilities.astype(float), index=rows.index, name="prob")


def _paths(root: Path | str, name: str) -> tuple[Path, Path]:
    folder = Path(root) / "models"
    return folder / f"{name}.json", folder / f"{name}.meta.json"


def save_model(root: Path | str, model: XGBClassifier, recipe: Recipe, meta: dict) -> None:
    """Write champion weights + metadata, keeping the replaced pair as `previous`."""
    weights, meta_path = _paths(root, "champion")
    weights.parent.mkdir(parents=True, exist_ok=True)
    if weights.exists() and meta_path.exists():
        previous_weights, previous_meta = _paths(root, "previous")
        shutil.copyfile(weights, previous_weights)
        shutil.copyfile(meta_path, previous_meta)
    temporary = weights.with_suffix(".tmp.json")
    model.save_model(temporary)
    temporary.replace(weights)
    payload = {
        **meta,
        "recipe": recipe.to_dict(),
        "recipe_id": recipe.recipe_id,
        "feature_version": FEATURE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    meta_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_model(root: Path | str) -> tuple[XGBClassifier, Recipe, dict] | None:
    weights, meta_path = _paths(root, "champion")
    if not weights.exists() or not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    recipe = Recipe.from_dict(meta["recipe"])
    model = new_classifier(recipe)
    model.load_model(weights)
    return model, recipe, meta


def _contributions(model: XGBClassifier, recipe: Recipe, rows: pd.DataFrame) -> np.ndarray:
    import xgboost

    matrix = xgboost.DMatrix(_matrix(recipe, rows), feature_names=list(recipe.features))
    return model.get_booster().predict(matrix, pred_contribs=True)[:, :-1]  # drop bias


def explain(
    model: XGBClassifier,
    recipe: Recipe,
    rows: pd.DataFrame,
    top: int = 2,
    only: tuple[str, ...] | None = None,
    reference: pd.DataFrame | None = None,
) -> list[list[dict]]:
    """For each row, the `top` features that moved its score most, with direction.

    Uses XGBoost's own per-prediction contributions (log-odds); "up" means the
    feature raised this stock's score, "down" that it lowered it.

    `only` restricts the answer to a subset of features (e.g. the stock-specific
    ones: macro features are identical for every stock on a given day and cannot
    explain a ranking). `reference` is a set of peer rows; contributions are then
    measured against the peers' average, so the answer names what sets the stock
    apart from them rather than what all of them share.
    """
    if rows.empty:
        return []
    contributions = _contributions(model, recipe, rows)
    if reference is not None and not reference.empty:
        contributions = contributions - _contributions(model, recipe, reference).mean(axis=0)
    allowed = [i for i, name in enumerate(recipe.features) if only is None or name in only]
    result = []
    for values in contributions:
        order = sorted(allowed, key=lambda i: -abs(values[i]))[:top]
        result.append(
            [
                {"feature": recipe.features[i], "direction": "up" if values[i] > 0 else "down"}
                for i in order
                if values[i] != 0
            ]
        )
    return result
