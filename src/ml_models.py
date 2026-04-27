
"""ML-модели для прогнозирования временных рядов через лаговые признаки."""

from __future__ import annotations

import pandas as pd


def make_lag_features(
    df: pd.DataFrame,
    date_col: str = "ds",
    target_col: str = "y",
    lags: list[int] | None = None,
    rolling_windows: list[int] | None = None,
) -> pd.DataFrame:
    """Создает лаговые, rolling и календарные признаки."""
    if lags is None:
        lags = [1, 2, 3, 7, 14]
    if rolling_windows is None:
        rolling_windows = [7, 14, 30]

    out = df.copy().sort_values(date_col)

    for lag in lags:
        out[f"lag_{lag}"] = out[target_col].shift(lag)

    for window in rolling_windows:
        shifted = out[target_col].shift(1)
        out[f"rolling_mean_{window}"] = shifted.rolling(window).mean()
        out[f"rolling_std_{window}"] = shifted.rolling(window).std()

    out["dayofweek"] = out[date_col].dt.dayofweek
    out["month"] = out[date_col].dt.month
    out["day"] = out[date_col].dt.day
    out["is_weekend"] = out["dayofweek"].isin([5, 6]).astype(int)

    return out.dropna().reset_index(drop=True)


def get_default_ml_models(random_state: int = 42) -> dict:
    """Возвращает базовый набор ML-моделей."""
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

    models = {
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            random_state=random_state,
        ),
    }

    try:
        from lightgbm import LGBMRegressor
        models["LightGBM"] = LGBMRegressor(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=31,
            random_state=random_state,
        )
    except Exception:
        pass

    return models
