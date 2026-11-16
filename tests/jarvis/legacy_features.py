"""Load the original feature functions from core/ai_analyzer.py for parity tests.

That module imports streamlit and shap at the top, so it cannot be imported in
the light test environment. Only the pure functions and the constants they
depend on are compiled and executed here.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

WANTED_FUNCTIONS = {
    "_wilder",
    "build_stationary_features",
    "_available_on_calendar",
    "build_macro_features",
}
WANTED_NAMES = {
    "BENCHMARK_TICKER",
    "MACRO_TICKERS",
    "VIX_PERCENTILE_COLUMN",
    "VIX_PERCENTILE_WINDOW",
    "VIX_PERCENTILE_MIN_PERIODS",
}


def legacy_functions() -> dict:
    path = Path(__file__).resolve().parents[2] / "core" / "ai_analyzer.py"
    body = []
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.FunctionDef) and node.name in WANTED_FUNCTIONS:
            body.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id in WANTED_NAMES for t in targets):
                body.append(node)
    namespace = {"np": np, "pd": pd, "Any": Any, "Mapping": Mapping, "Sequence": Sequence}
    module = ast.Module(body=body, type_ignores=[])
    exec(compile(module, "ai_analyzer_subset", "exec"), namespace)
    return namespace
