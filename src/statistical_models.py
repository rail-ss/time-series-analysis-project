"""Статистические методы прогнозирования временного ряда.

Модуль решает блок задания со статистическими моделями:
- baseline-модели;
- ручные статистические модели;
- автоматические модели через statsforecast;
- backtesting;
- анализ остатков;
- сохранение таблиц, графиков и markdown-отчета.

Код рассчитан на подготовленный файл:
    data/processed/prepared_time_series.csv

Формат ряда:
    unique_id, ds, y
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    BACKTEST_STEP,
    BACKTEST_WINDOWS,
    DAILY_SEASON_LENGTH,
    FIGURES_DIR,
    FORECAST_HORIZON,
    FORECASTS_DIR,
    FREQUENCY,
    MAX_TRAIN_HISTORY,
    PREDICTION_LEVELS,
    PREPARED_DATA_PATH,
    STATISTICAL_REPORT_PATH,
    TABLES_DIR,
    WEEKLY_SEASON_LENGTH,
)
from src.evaluation import evaluate_forecast, summarize_metrics


@dataclass
class ForecastResult:
    model: str
    values: np.ndarray
    lower_80: Optional[np.ndarray] = None
    upper_80: Optional[np.ndarray] = None
    lower_95: Optional[np.ndarray] = None
    upper_95: Optional[np.ndarray] = None
    note: str = ""


def ensure_output_dirs() -> None:
    for path in [FIGURES_DIR, TABLES_DIR, FORECASTS_DIR]:
        Path(path).mkdir(parents=True, exist_ok=True)


def load_prepared_series(path: Path = PREPARED_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Не найден подготовленный ряд: {path}\n"
            "Сначала запустите подготовку данных:\n"
            "python scripts/prepare_eda.py"
        )

    df = pd.read_csv(path)
    required = {"unique_id", "ds", "y"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"В подготовленном файле нет колонок: {sorted(missing)}")

    df = df[["unique_id", "ds", "y"]].copy()
    df["ds"] = pd.to_datetime(df["ds"])
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["ds", "y"]).sort_values("ds").reset_index(drop=True)

    return df


def _repeat_last_season(y: np.ndarray, h: int, season_length: int) -> np.ndarray:
    if len(y) < season_length:
        return np.repeat(y[-1], h)
    last_season = y[-season_length:]
    repeats = int(math.ceil(h / season_length))
    return np.tile(last_season, repeats)[:h]


def _seasonal_window_average(y: np.ndarray, h: int, season_length: int = 24, n_seasons: int = 7) -> np.ndarray:
    values = []
    for i in range(1, h + 1):
        idx = len(y) - season_length + ((i - 1) % season_length)
        seasonal_values = []
        for k in range(n_seasons):
            j = idx - k * season_length
            if 0 <= j < len(y):
                seasonal_values.append(y[j])
        if seasonal_values:
            values.append(float(np.mean(seasonal_values)))
        else:
            values.append(float(y[-1]))
    return np.asarray(values)


def _simple_prediction_interval(pred: np.ndarray, residual_scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Грубые интервалы для ручных baseline-моделей.

    Это не полноценная вероятностная модель. Интервалы нужны как ориентир:
    чем выше разброс последних сезонных ошибок, тем шире интервал.
    """
    scale = float(residual_scale) if np.isfinite(residual_scale) and residual_scale > 0 else 0.1
    lo80 = pred - 1.28 * scale
    hi80 = pred + 1.28 * scale
    lo95 = pred - 1.96 * scale
    hi95 = pred + 1.96 * scale
    return lo80, hi80, lo95, hi95


def _seasonal_residual_scale(y: np.ndarray, season_length: int = 24) -> float:
    if len(y) <= season_length:
        return float(np.std(np.diff(y))) if len(y) > 1 else 0.1
    residuals = y[season_length:] - y[:-season_length]
    return float(np.nanstd(residuals))


def manual_forecasts(train: pd.DataFrame, h: int) -> list[ForecastResult]:
    y = train["y"].to_numpy(dtype=float)
    residual_scale = _seasonal_residual_scale(y, DAILY_SEASON_LENGTH)

    results: list[ForecastResult] = []

    naive = np.repeat(y[-1], h)
    results.append(ForecastResult("naive", naive, *_simple_prediction_interval(naive, residual_scale),
                                  note="Последнее наблюдение переносится на весь горизонт."))

    seasonal_24 = _repeat_last_season(y, h, DAILY_SEASON_LENGTH)
    results.append(ForecastResult("seasonal_naive_24", seasonal_24, *_simple_prediction_interval(seasonal_24, residual_scale),
                                  note="Повторяется профиль последних 24 часов."))

    seasonal_168 = _repeat_last_season(y, h, WEEKLY_SEASON_LENGTH)
    results.append(ForecastResult("seasonal_naive_168", seasonal_168, *_simple_prediction_interval(seasonal_168, residual_scale),
                                  note="Повторяется профиль последней недели."))

    window_average = np.repeat(np.mean(y[-WEEKLY_SEASON_LENGTH:]), h)
    results.append(ForecastResult("window_average_168", window_average, *_simple_prediction_interval(window_average, residual_scale),
                                  note="Среднее значение за последнюю неделю."))

    seasonal_average = _seasonal_window_average(y, h, season_length=DAILY_SEASON_LENGTH, n_seasons=7)
    results.append(ForecastResult("seasonal_window_average", seasonal_average, *_simple_prediction_interval(seasonal_average, residual_scale),
                                  note="Среднее по тем же часам суток за последние 7 дней."))

    if len(y) > 1:
        slope = (y[-1] - y[0]) / (len(y) - 1)
    else:
        slope = 0.0
    drift = y[-1] + slope * np.arange(1, h + 1)
    results.append(ForecastResult("drift", drift, *_simple_prediction_interval(drift, residual_scale),
                                  note="Случайное блуждание с линейным дрейфом."))

    # Ручные статистические модели через statsmodels.
    results.extend(statsmodels_manual_forecasts(train, h))

    return results


def statsmodels_manual_forecasts(train: pd.DataFrame, h: int) -> list[ForecastResult]:
    """Ручные модели с явно заданными параметрами.

    Для скорости берется последние 60 дней. Это достаточно для часового ряда,
    чтобы увидеть суточную сезонность, но не перегружать вычисления.
    """
    results: list[ForecastResult] = []
    y_full = train["y"].astype(float)
    y = y_full.tail(24 * 60).copy()

    # ETS с заданной аддитивной структурой.
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ExponentialSmoothing(
                y,
                trend="add",
                seasonal="add",
                seasonal_periods=DAILY_SEASON_LENGTH,
                initialization_method="estimated",
            ).fit(optimized=True)
            pred = np.asarray(model.forecast(h), dtype=float)
        results.append(ForecastResult(
            "manual_ets_additive",
            pred,
            note="ETS с аддитивным трендом и суточной сезонностью, seasonal_periods=24.",
        ))
    except Exception as exc:
        print(f"[warning] manual_ets_additive не построена: {exc}")

    # ThetaModel с явно заданным периодом.
    try:
        from statsmodels.tsa.forecasting.theta import ThetaModel

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ThetaModel(y, period=DAILY_SEASON_LENGTH).fit()
            pred = np.asarray(model.forecast(h), dtype=float)
        results.append(ForecastResult(
            "manual_theta_24",
            pred,
            note="Theta-модель с явно заданным суточным периодом 24.",
        ))
    except Exception as exc:
        print(f"[warning] manual_theta_24 не построена: {exc}")

    # ARIMA с явно заданными параметрами.
    # Сезонность отдельно учитывается сезонными baseline/ETS/Theta-моделями,
    # поэтому здесь проверяется простой ручной ARIMA-вариант.
    try:
        from statsmodels.tsa.arima.model import ARIMA

        y_short = y_full.tail(24 * 30).copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMA(y_short, order=(1, 1, 1)).fit()
            pred = np.asarray(model.forecast(h), dtype=float)
        results.append(ForecastResult(
            "manual_arima",
            pred,
            note="ARIMA(1,1,1), параметры заданы вручную как простой ARIMA-ориентир.",
        ))
    except Exception as exc:
        print(f"[warning] manual_arima не построена: {exc}")

    return results


def _make_model_with_alias(cls, alias: str, **kwargs):
    """Создает модель statsforecast с alias, а если версия пакета не поддерживает alias — без него."""
    try:
        return cls(alias=alias, **kwargs)
    except TypeError:
        return cls(**kwargs)


def statsforecast_auto_forecasts(train: pd.DataFrame, h: int, levels: Iterable[int] = PREDICTION_LEVELS) -> list[ForecastResult]:
    """Автоматические модели через statsforecast.

    Если пакет не установлен, функция не падает, а возвращает пустой список.
    """
    try:
        from statsforecast import StatsForecast
        from statsforecast.models import AutoARIMA, AutoETS, AutoTheta
    except Exception as exc:
        print(f"[warning] statsforecast недоступен: {exc}")
        print("          Установите зависимости: python -m pip install -r requirements.txt")
        return []

    models = [
        _make_model_with_alias(AutoARIMA, "auto_arima", season_length=DAILY_SEASON_LENGTH),
        _make_model_with_alias(AutoETS, "auto_ets", season_length=DAILY_SEASON_LENGTH),
        _make_model_with_alias(AutoTheta, "auto_theta", season_length=DAILY_SEASON_LENGTH),
    ]

    try:
        sf = StatsForecast(models=models, freq=FREQUENCY, n_jobs=1)
        fcst = sf.forecast(df=train[["unique_id", "ds", "y"]], h=h, level=list(levels))
    except Exception as exc:
        print(f"[warning] Автоматические модели statsforecast не построены: {exc}")
        return []

    # StatsForecast обычно возвращает unique_id, ds и по колонке на модель.
    result_models = []
    for col in fcst.columns:
        if col in {"unique_id", "ds"}:
            continue
        if "-lo-" in col or "-hi-" in col:
            continue

        values = fcst[col].to_numpy(dtype=float)
        lo80 = fcst.get(f"{col}-lo-80")
        hi80 = fcst.get(f"{col}-hi-80")
        lo95 = fcst.get(f"{col}-lo-95")
        hi95 = fcst.get(f"{col}-hi-95")

        normalized_name = {
            "autoarima": "auto_arima",
            "autoets": "auto_ets",
            "autotheta": "auto_theta",
        }.get(str(col).replace("_", "").lower(), str(col).lower())

        result_models.append(
            ForecastResult(
                model=normalized_name,
                values=values,
                lower_80=lo80.to_numpy(dtype=float) if lo80 is not None else None,
                upper_80=hi80.to_numpy(dtype=float) if hi80 is not None else None,
                lower_95=lo95.to_numpy(dtype=float) if lo95 is not None else None,
                upper_95=hi95.to_numpy(dtype=float) if hi95 is not None else None,
                note="Автоматический подбор параметров в statsforecast.",
            )
        )

    return result_models


def backtest_cutoffs(df: pd.DataFrame, h: int, n_windows: int, step: int) -> list[int]:
    n = len(df)
    min_needed = h + (n_windows - 1) * step + max(DAILY_SEASON_LENGTH * 14, h)
    if n < min_needed:
        raise ValueError("Недостаточно наблюдений для выбранной схемы backtesting.")

    cutoffs = []
    first_test_start = n - h - (n_windows - 1) * step
    for i in range(n_windows):
        cutoffs.append(first_test_start + i * step)
    return cutoffs


def run_forecasts_for_window(train_full: pd.DataFrame, h: int) -> list[ForecastResult]:
    # Для скорости ограничиваем историю последних 180 дней.
    train = train_full.tail(MAX_TRAIN_HISTORY).copy()
    results = manual_forecasts(train, h)
    results.extend(statsforecast_auto_forecasts(train, h))
    return results


def run_backtesting(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoffs = backtest_cutoffs(
        df,
        h=FORECAST_HORIZON,
        n_windows=BACKTEST_WINDOWS,
        step=BACKTEST_STEP,
    )

    all_predictions = []
    all_metrics = []

    for window_id, cutoff in enumerate(cutoffs, start=1):
        train_full = df.iloc[:cutoff].copy()
        test = df.iloc[cutoff: cutoff + FORECAST_HORIZON].copy()
        forecasts = run_forecasts_for_window(train_full, FORECAST_HORIZON)

        print(f"Окно {window_id}/{len(cutoffs)}: train={len(train_full)}, test={len(test)}, models={len(forecasts)}")

        for item in forecasts:
            pred = np.asarray(item.values, dtype=float)
            if len(pred) != len(test) or np.any(~np.isfinite(pred)):
                continue

            metrics = evaluate_forecast(
                y_true=test["y"].to_numpy(dtype=float),
                y_pred=pred,
                y_train=train_full["y"].to_numpy(dtype=float),
                seasonality=DAILY_SEASON_LENGTH,
            )
            metrics.update({
                "window": window_id,
                "cutoff": train_full["ds"].iloc[-1],
                "model": item.model,
            })
            all_metrics.append(metrics)

            pred_df = pd.DataFrame({
                "window": window_id,
                "cutoff": train_full["ds"].iloc[-1],
                "ds": test["ds"].to_numpy(),
                "y": test["y"].to_numpy(dtype=float),
                "model": item.model,
                "y_pred": pred,
                "lower_80": item.lower_80 if item.lower_80 is not None else np.nan,
                "upper_80": item.upper_80 if item.upper_80 is not None else np.nan,
                "lower_95": item.lower_95 if item.lower_95 is not None else np.nan,
                "upper_95": item.upper_95 if item.upper_95 is not None else np.nan,
            })
            all_predictions.append(pred_df)

    if not all_predictions:
        raise RuntimeError("Не удалось построить прогнозы ни одной моделью.")

    predictions = pd.concat(all_predictions, ignore_index=True)
    metrics = pd.DataFrame(all_metrics)

    return predictions, metrics


def final_forecast(df: pd.DataFrame) -> pd.DataFrame:
    forecasts = run_forecasts_for_window(df, FORECAST_HORIZON)
    last_ds = df["ds"].max()
    future_ds = pd.date_range(last_ds + pd.Timedelta(hours=1), periods=FORECAST_HORIZON, freq=FREQUENCY)

    rows = []
    for item in forecasts:
        pred = np.asarray(item.values, dtype=float)
        if len(pred) != FORECAST_HORIZON or np.any(~np.isfinite(pred)):
            continue
        rows.append(pd.DataFrame({
            "ds": future_ds,
            "model": item.model,
            "y_pred": pred,
            "lower_80": item.lower_80 if item.lower_80 is not None else np.nan,
            "upper_80": item.upper_80 if item.upper_80 is not None else np.nan,
            "lower_95": item.lower_95 if item.lower_95 is not None else np.nan,
            "upper_95": item.upper_95 if item.upper_95 is not None else np.nan,
        }))

    return pd.concat(rows, ignore_index=True)


def residual_diagnostics(predictions: pd.DataFrame, best_model: str) -> pd.DataFrame:
    df = predictions[predictions["model"] == best_model].copy()
    df["residual"] = df["y"] - df["y_pred"]

    residuals = df["residual"].to_numpy(dtype=float)
    diag = {
        "model": best_model,
        "residual_mean": float(np.mean(residuals)),
        "residual_std": float(np.std(residuals)),
        "residual_mae": float(np.mean(np.abs(residuals))),
        "residual_median": float(np.median(residuals)),
        "residual_lag1_autocorr": float(pd.Series(residuals).autocorr(lag=1)),
    }

    if df["lower_80"].notna().any() and df["upper_80"].notna().any():
        inside_80 = (df["y"] >= df["lower_80"]) & (df["y"] <= df["upper_80"])
        diag["coverage_80"] = float(inside_80.mean())
    else:
        diag["coverage_80"] = float("nan")

    if df["lower_95"].notna().any() and df["upper_95"].notna().any():
        inside_95 = (df["y"] >= df["lower_95"]) & (df["y"] <= df["upper_95"])
        diag["coverage_95"] = float(inside_95.mean())
    else:
        diag["coverage_95"] = float("nan")

    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox

        lb = acorr_ljungbox(residuals, lags=[24], return_df=True)
        diag["ljung_box_pvalue_lag_24"] = float(lb["lb_pvalue"].iloc[0])
    except Exception:
        diag["ljung_box_pvalue_lag_24"] = float("nan")

    return pd.DataFrame([diag])


def build_model_selection_table(metrics_summary: pd.DataFrame) -> pd.DataFrame:
    comments = {
        "naive": "Базовый ориентир: перенос последнего значения. Не учитывает суточную сезонность.",
        "seasonal_naive_24": "Сильный baseline для часового электропотребления: повторяет последние сутки.",
        "seasonal_naive_168": "Учитывает недельный паттерн, полезен при различии будней и выходных.",
        "window_average_168": "Сглаженная модель, устойчива к шуму, но плохо реагирует на быстрые изменения.",
        "seasonal_window_average": "Усредняет одинаковые часы суток за неделю, снижает влияние случайных всплесков.",
        "drift": "Проверяет наличие простого линейного изменения уровня ряда.",
        "manual_ets_additive": "Ручная ETS-модель: явно заданы аддитивный тренд и суточная сезонность.",
        "manual_theta_24": "Ручная Theta-модель с суточным периодом.",
        "manual_arima": "Ручная ARIMA(1,1,1)-модель как простой ARIMA-ориентир.",
        "auto_arima": "Автоматический подбор ARIMA-параметров в statsforecast.",
        "auto_ets": "Автоматический подбор ETS-структуры в statsforecast.",
        "auto_theta": "Автоматический выбор варианта Theta-модели в statsforecast.",
    }

    groups = {
        "naive": "baseline",
        "seasonal_naive_24": "baseline",
        "seasonal_naive_168": "baseline",
        "window_average_168": "statistical",
        "seasonal_window_average": "statistical",
        "drift": "statistical",
        "manual_ets_additive": "statistical_manual",
        "manual_theta_24": "statistical_manual",
        "manual_arima": "statistical_manual",
        "auto_arima": "statistical_auto",
        "auto_ets": "statistical_auto",
        "auto_theta": "statistical_auto",
    }

    table = metrics_summary.copy()
    table.insert(1, "group", table["model"].map(groups).fillna("statistical"))
    table["comment"] = table["model"].map(comments).fillna("Статистическая модель сравнивается по backtesting.")
    table["decision"] = np.where(table["rank"] == 1, "основной кандидат", "сравнительная модель")
    return table


def plot_metrics(metrics_summary: pd.DataFrame) -> None:
    top = metrics_summary.sort_values("smape", ascending=True).copy()
    plt.figure(figsize=(10, max(4, 0.45 * len(top))))
    plt.barh(top["model"], top["smape"])
    plt.gca().invert_yaxis()
    plt.xlabel("sMAPE, %")
    plt.title("Сравнение статистических моделей по sMAPE")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "statistical_metrics.png", dpi=160)
    plt.close()


def plot_backtest_comparison(predictions: pd.DataFrame, best_model: str) -> None:
    last_window = predictions["window"].max()
    df = predictions[(predictions["window"] == last_window) & (predictions["model"] == best_model)].copy()
    plt.figure(figsize=(12, 5))
    plt.plot(df["ds"], df["y"], label="fact")
    plt.plot(df["ds"], df["y_pred"], label=best_model)
    if df["lower_95"].notna().any() and df["upper_95"].notna().any():
        plt.fill_between(df["ds"], df["lower_95"], df["upper_95"], alpha=0.2, label="95% interval")
    plt.title(f"Backtesting: факт и прогноз лучшей модели ({best_model})")
    plt.xlabel("Дата")
    plt.ylabel("Global active power, kW")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "statistical_backtest.png", dpi=160)
    plt.close()


def plot_final_forecast(df: pd.DataFrame, forecast: pd.DataFrame, best_model: str) -> None:
    history = df.tail(24 * 14).copy()
    fcst = forecast[forecast["model"] == best_model].copy()
    plt.figure(figsize=(12, 5))
    plt.plot(history["ds"], history["y"], label="history")
    plt.plot(fcst["ds"], fcst["y_pred"], label=f"forecast: {best_model}")
    if fcst["lower_95"].notna().any() and fcst["upper_95"].notna().any():
        plt.fill_between(fcst["ds"], fcst["lower_95"], fcst["upper_95"], alpha=0.2, label="95% interval")
    plt.title("Прогноз на следующие 24 часа")
    plt.xlabel("Дата")
    plt.ylabel("Global active power, kW")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "statistical_forecast.png", dpi=160)
    plt.close()


def plot_residuals(predictions: pd.DataFrame, best_model: str) -> None:
    df = predictions[predictions["model"] == best_model].copy()
    df["residual"] = df["y"] - df["y_pred"]

    plt.figure(figsize=(12, 5))
    plt.plot(df["ds"], df["residual"])
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.title(f"Остатки лучшей статистической модели: {best_model}")
    plt.xlabel("Дата")
    plt.ylabel("Ошибка прогноза")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "statistical_residuals.png", dpi=160)
    plt.close()

    try:
        from statsmodels.graphics.tsaplots import plot_acf

        plt.figure(figsize=(10, 4))
        plot_acf(df["residual"], lags=48)
        plt.title(f"ACF остатков: {best_model}")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "statistical_residuals_acf.png", dpi=160)
        plt.close()
    except Exception as exc:
        print(f"[warning] Не удалось построить ACF остатков: {exc}")


def save_outputs(
    df: pd.DataFrame,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    metrics_summary: pd.DataFrame,
    selection: pd.DataFrame,
    diagnostics: pd.DataFrame,
    forecast: pd.DataFrame,
) -> None:
    predictions.to_csv(FORECASTS_DIR / "statistical_predictions.csv", index=False)
    forecast.to_csv(FORECASTS_DIR / "statistical_forecast.csv", index=False)

    metrics.to_csv(TABLES_DIR / "statistical_metrics_by_window.csv", index=False)
    metrics_summary.to_csv(TABLES_DIR / "statistical_metrics.csv", index=False)
    selection.to_csv(TABLES_DIR / "model_selection.csv", index=False)
    diagnostics.to_csv(TABLES_DIR / "residual_diagnostics.csv", index=False)


def _format_float(x) -> str:
    if pd.isna(x):
        return "н/д"
    return f"{float(x):.4f}"


def make_report(
    df: pd.DataFrame,
    metrics_summary: pd.DataFrame,
    selection: pd.DataFrame,
    diagnostics: pd.DataFrame,
    best_model: str,
) -> str:
    best_row = metrics_summary[metrics_summary["model"] == best_model].iloc[0]
    diag = diagnostics.iloc[0]

    metrics_md = metrics_summary.to_markdown(index=False)
    selection_md = selection[["rank", "group", "model", "mae", "rmse", "smape", "mase", "comment", "decision"]].to_markdown(index=False)
    diagnostics_md = diagnostics.to_markdown(index=False)

    text = f"""# Статистические методы прогнозирования

## Цель раздела

В этом разделе сравниваются статистические методы прогнозирования временного ряда электропотребления. Цель — выбрать сильную статистическую модель, которую затем можно будет сравнить с ML- и DL-подходами.

Рабочий ряд: `data/processed/prepared_time_series.csv`.

- Целевая переменная: `y`, среднее часовое значение `Global_active_power`.
- Частота ряда: 1 час.
- Горизонт прогноза: 24 часа.
- Основные сезонные периоды: 24 часа и 168 часов.

## Использованные методы

В сравнении участвуют baseline-модели, ручные статистические модели и автоматические модели из `statsforecast`.

| Группа | Модели | Логика |
|---|---|---|
| Baseline | `naive`, `seasonal_naive_24`, `seasonal_naive_168` | Нужны как минимальный уровень качества, который должны превзойти более сложные модели. |
| Простые статистические | `window_average_168`, `seasonal_window_average`, `drift` | Проверяют, достаточно ли простого сглаживания, сезонного усреднения или дрейфа. |
| Ручные модели | `manual_ets_additive`, `manual_theta_24`, `manual_arima` | Параметры задаются явно на основе EDA: суточная сезонность 24 часа, наличие сезонности и автокорреляции. |
| Автоматические модели | `auto_arima`, `auto_ets`, `auto_theta` | Параметры подбираются автоматически средствами `statsforecast`. |

## Схема проверки качества

Для проверки использован backtesting по временным окнам. В каждом окне модель обучается только на прошлом и прогнозирует следующие 24 часа. Такой подход корректнее случайного train/test split, потому что во временных рядах нельзя перемешивать прошлое и будущее.

Метрики:

- `MAE` — средняя абсолютная ошибка;
- `RMSE` — корень из средней квадратичной ошибки;
- `MAPE` — относительная ошибка в процентах, используется осторожно;
- `sMAPE` — симметричная относительная ошибка;
- `MASE` — ошибка относительно сезонного naive-подхода.

## Результаты backtesting

{metrics_md}

## Таблица выбора моделей

{selection_md}

## Выбранная модель

По среднему значению `sMAPE` лучшей моделью стала:

**`{best_model}`**

Ее основные метрики:

- MAE: **{_format_float(best_row["mae"])}**
- RMSE: **{_format_float(best_row["rmse"])}**
- sMAPE: **{_format_float(best_row["smape"])}**
- MASE: **{_format_float(best_row["mase"])}**

Выбор сделан по `sMAPE`, потому что эта метрика удобна для сравнения относительной ошибки на разных уровнях электропотребления. При этом дополнительно учитываются `MAE`, `RMSE` и `MASE`, чтобы не выбирать модель только по одной метрике.

## Анализ остатков

Для лучшей модели рассчитаны остатки на backtesting-окнах:

{diagnostics_md}

Среднее значение остатков показывает наличие или отсутствие систематического смещения. Если среднее близко к нулю, модель в среднем не завышает и не занижает прогноз. Автокорреляция остатков и тест Ljung-Box позволяют проверить, осталась ли во временном ряду ошибки неучтенная структура.

Если остатки имеют заметную автокорреляцию, это означает, что модель не полностью использовала временную структуру ряда. В таком случае на следующих этапах имеет смысл применять ML-модели с лаговыми признаками и DL-модели, которые могут лучше учитывать сложные зависимости.

## Вероятностный прогноз

Для моделей, которые поддерживают интервальные прогнозы, дополнительно сохранены 80% и 95% интервалы. Они используются для оценки неопределенности прогноза. Для baseline-моделей интервалы рассчитаны приближенно через разброс сезонных ошибок, поэтому они используются только как ориентир.

## Сохраненные результаты

Таблицы:

- `reports/tables/statistical_metrics_by_window.csv`
- `reports/tables/statistical_metrics.csv`
- `reports/tables/model_selection.csv`
- `reports/tables/residual_diagnostics.csv`

Прогнозы:

- `outputs/forecasts/statistical_predictions.csv`
- `outputs/forecasts/statistical_forecast.csv`

Графики:

- `reports/figures/statistical_metrics.png`
- `reports/figures/statistical_backtest.png`
- `reports/figures/statistical_forecast.png`
- `reports/figures/statistical_residuals.png`
- `reports/figures/statistical_residuals_acf.png`

## Вывод

Статистические модели дают интерпретируемый базовый уровень качества для задачи прогнозирования электропотребления. Baseline-модели показывают, насколько сильна простая сезонная структура ряда. Автоматические модели `statsforecast` позволяют проверить ARIMA-, ETS- и Theta-подходы без ручного перебора параметров. Результаты этого раздела будут использоваться дальше как статистическая база сравнения для ML- и DL-моделей.
"""
    return text


def run_statistical_analysis() -> dict:
    ensure_output_dirs()

    print("[1/7] Загрузка подготовленного ряда...")
    df = load_prepared_series()
    print(f"Наблюдений: {len(df)}")
    print(f"Период: {df['ds'].min()} — {df['ds'].max()}")

    print("[2/7] Backtesting статистических моделей...")
    predictions, metrics = run_backtesting(df)

    print("[3/7] Сводная таблица метрик...")
    metrics_summary = summarize_metrics(metrics)
    selection = build_model_selection_table(metrics_summary)

    best_model = str(metrics_summary.iloc[0]["model"])
    print(f"Лучшая модель по sMAPE: {best_model}")

    print("[4/7] Финальный прогноз на 24 часа...")
    forecast = final_forecast(df)

    print("[5/7] Анализ остатков...")
    diagnostics = residual_diagnostics(predictions, best_model)

    print("[6/7] Сохранение таблиц и графиков...")
    save_outputs(df, predictions, metrics, metrics_summary, selection, diagnostics, forecast)
    plot_metrics(metrics_summary)
    plot_backtest_comparison(predictions, best_model)
    plot_final_forecast(df, forecast, best_model)
    plot_residuals(predictions, best_model)

    print("[7/7] Создание markdown-отчета...")
    report = make_report(df, metrics_summary, selection, diagnostics, best_model)
    STATISTICAL_REPORT_PATH.write_text(report, encoding="utf-8")

    return {
        "best_model": best_model,
        "report": STATISTICAL_REPORT_PATH,
        "metrics": TABLES_DIR / "statistical_metrics.csv",
        "selection": TABLES_DIR / "model_selection.csv",
        "forecast": FORECASTS_DIR / "statistical_forecast.csv",
    }


if __name__ == "__main__":
    result = run_statistical_analysis()
    print("Готово.")
    print(f"Лучшая модель: {result['best_model']}")
    print(f"Отчет: {result['report']}")
    print(f"Метрики: {result['metrics']}")
    print(f"Таблица выбора моделей: {result['selection']}")
    print(f"Прогноз: {result['forecast']}")
