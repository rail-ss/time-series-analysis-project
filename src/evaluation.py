import numpy as np
import pandas as pd


def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def smape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum((np.abs(y_true) + np.abs(y_pred)) / 2, eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


def mase(y_true, y_pred, insample, season_length=24, eps=1e-8):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    insample = np.asarray(insample, dtype=float)

    if len(insample) <= season_length:
        scale = np.mean(np.abs(np.diff(insample)))
    else:
        scale = np.mean(np.abs(insample[season_length:] - insample[:-season_length]))

    scale = max(float(scale), eps)
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def calculate_metrics(y_true, y_pred, insample=None, season_length=24):
    result = {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
    }
    if insample is not None:
        result["mase"] = mase(y_true, y_pred, insample, season_length=season_length)
    return result

def evaluate_forecast(y_true, y_pred, insample=None, season_length=24):
    """Расчет основных метрик качества прогноза.

    Эта функция используется в моделях как единая точка оценки прогноза.
    """
    return calculate_metrics(
        y_true=y_true,
        y_pred=y_pred,
        insample=insample,
        season_length=season_length,
    )

def summarize_metrics(predictions, insample_by_window=None, season_length=24):
    rows = []
    for (model, window), part in predictions.groupby(["model", "window"]):
        insample = None
        if insample_by_window is not None:
            insample = insample_by_window.get(window)
        metrics = calculate_metrics(part["y"].values, part["y_pred"].values, insample=insample, season_length=season_length)
        rows.append({"model": model, "window": window, **metrics})

    by_window = pd.DataFrame(rows)
    summary = (
        by_window
        .groupby("model", as_index=False)
        .agg({"mae": "mean", "rmse": "mean", "mape": "mean", "smape": "mean", "mase": "mean"})
        .sort_values("smape")
        .reset_index(drop=True)
    )
    summary.insert(0, "rank", range(1, len(summary) + 1))
    return by_window, summary
