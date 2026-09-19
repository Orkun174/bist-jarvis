"""Shared, causal transformations and persisted probability calibration."""
import numpy as np
import pandas as pd

NORMALIZATION_VERSION = 'price-relative-v1'
TARGET_HORIZON = 5
TARGET_RETURN = 0.03
TARGET_DEFINITION = "Close[t+5] / Close[t] - 1 >= 0.03; 5 trading sessions"
LEGACY_TARGET_DEFINITION = "Close[t+3] / Close[t] - 1 >= 0.02; 3 trading sessions"
RENAMES = {
    'MACD': 'MACD_Pct', 'MACD_Signal': 'MACD_Signal_Pct',
    'EMA_20': 'EMA20_Gap_Pct', 'EMA_50': 'EMA50_Gap_Pct', 'ATR_14': 'ATR_Pct',
}

def normalize_features(raw: pd.DataFrame, close) -> pd.DataFrame:
    """Use current close only; no fitted scaler or future observations."""
    close = pd.Series(np.asarray(close, dtype=float), index=raw.index)
    if not np.isfinite(close).all() or (close <= 0).any():
        raise ValueError('Positive finite close prices required')
    result = raw.copy()
    for name in ('MACD', 'MACD_Signal', 'ATR_14'):
        result[name] = 100.0 * result[name] / close
    for name in ('EMA_20', 'EMA_50'):
        result[name] = 100.0 * (close / result[name] - 1.0)
    return result.rename(columns=RENAMES).astype(np.float32)

def mask_macros(matrix: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope not in ('full', 'local'):
        raise ValueError('Unknown feature scope')
    result = matrix.copy()
    if scope == 'local':
        result.loc[:, [c for c in result if '_Return_' in c]] = 0.0
    return result

def calibrated_probability(probabilities, spec: dict):
    p = np.asarray(probabilities, dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Invalid model probability')
    if spec.get('method', 'identity') == 'identity':
        return p
    if spec.get('method') != 'sigmoid':
        raise ValueError('Unknown probability calibration')
    a, b = float(spec['a']), float(spec['b'])
    if not np.isfinite([a, b]).all() or a < 0:
        raise ValueError('Invalid calibration coefficients')
    clipped = np.clip(p, 1e-6, 1 - 1e-6)
    z = np.clip(a * np.log(clipped / (1 - clipped)) + b, -35, 35)
    return 1 / (1 + np.exp(-z))
