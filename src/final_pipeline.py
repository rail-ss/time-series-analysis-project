"""Финальный пайплайн прогнозирования временного ряда.

Скрипт собирает результаты предыдущих разделов, выбирает итоговую модель,
строит прогноз на 24 часа и проводит тестирование пайплайна:
- проверка данных;
- backtesting выбранного операционного метода;
- статистические проверки ошибок;
- тестирование производительности.

Запуск из корня проекта:
    python scripts/run_pipeline.py
"""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox

from src.config import (
    PREPARED_DATA_PATH,
    REPORTS_DIR,
    TABLES_DIR,
    FIGURES_DIR,
    FORECASTS_DIR,
    PIPELINE_REPORT_PATH,
    FORECAST_HORIZON,
    DAILY_SEASON_LENGTH,
    WEEKLY_SEASON_LENGTH,
)


@dataclass
class PipelineResult:
    selected_model: str
    selected_group: str
    selected_smape: float
    forecast_path: Path
    report_path: Path


def ensure_dirs() -> None:
    for path in [REPORTS_DIR, TABLES_DIR, FIGURES_DIR, FORECASTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_prepared_series(path: Path = PREPARED_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Не найден подготовленный ряд: {path}. Сначала запустите python scripts/prepare_eda.py"
        )
    df = pd.read_csv(path, parse_dates=["ds"])
    required = {"unique_id", "ds", "y"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"В подготовленном ряде нет обязательных колонок: {sorted(missing)}")
    df = df[["unique_id", "ds", "y"]].copy().sort_values("ds").reset_index(drop=True)
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    return df


def validate_series(df: pd.DataFrame) -> pd.DataFrame:
    tests: list[dict[str, str]] = []
    tests.append({"test": "required_columns", "status": "pass" if {"unique_id", "ds", "y"}.issubset(df.columns) else "fail", "details": "Проверка обязательных колонок unique_id, ds, y."})
    duplicate_count = int(df["ds"].duplicated().sum())
    tests.append({"test": "no_duplicate_timestamps", "status": "pass" if duplicate_count == 0 else "fail", "details": f"Количество дублирующихся временных меток: {duplicate_count}."})
    missing_target = int(df["y"].isna().sum())
    tests.append({"test": "no_missing_target", "status": "pass" if missing_target == 0 else "fail", "details": f"Количество пропусков в y: {missing_target}."})
    if len(df) > 1:
        diffs = df["ds"].diff().dropna()
        expected = pd.Timedelta(hours=1)
        irregular = int((diffs != expected).sum())
        tests.append({"test": "regular_hourly_frequency", "status": "pass" if irregular == 0 else "warn", "details": f"Нерегулярных шагов: {irregular}; ожидаемый шаг: 1 час."})
    else:
        tests.append({"test": "regular_hourly_frequency", "status": "fail", "details": "Недостаточно наблюдений для проверки частоты."})
    min_length = WEEKLY_SEASON_LENGTH + FORECAST_HORIZON
    tests.append({"test": "enough_history", "status": "pass" if len(df) >= min_length else "fail", "details": f"Наблюдений: {len(df)}; минимум для недельной сезонности и горизонта: {min_length}."})
    negative_share = float((df["y"] < 0).mean()) if len(df) else 0.0
    tests.append({"test": "non_negative_target", "status": "pass" if negative_share == 0 else "warn", "details": f"Доля отрицательных значений y: {negative_share:.4f}."})
    return pd.DataFrame(tests)


def read_metrics_table(path: Path, group: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "model" not in df.columns:
        return pd.DataFrame()
    keep = [col for col in ["model", "mae", "rmse", "mape", "smape", "mase"] if col in df.columns]
    df = df[keep].copy()
    df["group"] = group
    for col in ["mae", "rmse", "mape", "smape", "mase"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def collect_model_metrics() -> pd.DataFrame:
    parts = [
        read_metrics_table(TABLES_DIR / "statistical_metrics.csv", "statistical"),
        read_metrics_table(TABLES_DIR / "machine_learning_metrics.csv", "ML"),
        read_metrics_table(TABLES_DIR / "neural_metrics.csv", "DL"),
    ]
    combined = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    if combined.empty:
        raise FileNotFoundError(
            "Не найдены таблицы метрик моделей. Запустите предыдущие разделы: statistical_forecast.py, machine_learning_forecast.py, neural_forecast.py"
        )
    combined = combined.dropna(subset=["smape"]).sort_values("smape").reset_index(drop=True)
    combined.insert(0, "rank", np.arange(1, len(combined) + 1))
    return combined


def _same_hour_week_values(train: pd.DataFrame, future_ds: pd.Timestamp, recent_weeks: int = 8) -> pd.Series:
    mask = (train["ds"].dt.dayofweek == future_ds.dayofweek) & (train["ds"].dt.hour == future_ds.hour)
    return train.loc[mask, "y"].tail(recent_weeks)


def seasonal_window_average_forecast(train: pd.DataFrame, horizon: int = FORECAST_HORIZON, recent_weeks: int = 8) -> pd.DataFrame:
    last_ds = train["ds"].max()
    future_dates = pd.date_range(last_ds + pd.Timedelta(hours=1), periods=horizon, freq="h")
    predictions: list[float] = []
    fallback_daily = train["y"].tail(DAILY_SEASON_LENGTH).mean()
    fallback_value = float(train["y"].iloc[-1]) if len(train) else float("nan")
    for future_ds in future_dates:
        values = _same_hour_week_values(train, future_ds, recent_weeks=recent_weeks)
        if len(values) > 0 and np.isfinite(values.mean()):
            predictions.append(float(values.mean()))
        elif np.isfinite(fallback_daily):
            predictions.append(float(fallback_daily))
        else:
            predictions.append(fallback_value)
    return pd.DataFrame({"unique_id": train["unique_id"].iloc[0] if len(train) else "series_1", "ds": future_dates, "y_pred": predictions, "model": "seasonal_window_average"})


def seasonal_naive_168_forecast(train: pd.DataFrame, horizon: int = FORECAST_HORIZON) -> pd.DataFrame:
    last_ds = train["ds"].max()
    future_dates = pd.date_range(last_ds + pd.Timedelta(hours=1), periods=horizon, freq="h")
    if len(train) >= WEEKLY_SEASON_LENGTH + horizon:
        values = train["y"].iloc[-WEEKLY_SEASON_LENGTH : -WEEKLY_SEASON_LENGTH + horizon].to_numpy()
    elif len(train) >= WEEKLY_SEASON_LENGTH:
        values = train["y"].iloc[-WEEKLY_SEASON_LENGTH : -WEEKLY_SEASON_LENGTH + horizon].to_numpy()
        if len(values) < horizon:
            values = np.resize(values, horizon)
    else:
        values = np.repeat(train["y"].iloc[-1], horizon)
    return pd.DataFrame({"unique_id": train["unique_id"].iloc[0], "ds": future_dates, "y_pred": values[:horizon], "model": "seasonal_naive_168"})


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.abs(y_true) > 1e-8
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def compute_metrics(y_true: Iterable[float], y_pred: Iterable[float], mase_denom: float) -> dict[str, float]:
    y_true = np.asarray(list(y_true), dtype=float)
    y_pred = np.asarray(list(y_pred), dtype=float)
    error = y_true - y_pred
    smape_den = np.abs(y_true) + np.abs(y_pred)
    smape_mask = smape_den > 1e-8
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mape": _safe_mape(y_true, y_pred),
        "smape": float(np.mean(2 * np.abs(error[smape_mask]) / smape_den[smape_mask]) * 100) if smape_mask.any() else float("nan"),
        "mase": float(np.mean(np.abs(error)) / mase_denom) if mase_denom > 0 else float("nan"),
    }


def mase_denominator(series: pd.Series, season_length: int = DAILY_SEASON_LENGTH) -> float:
    diff = np.abs(series.to_numpy()[season_length:] - series.to_numpy()[:-season_length])
    denom = float(np.mean(diff)) if len(diff) else float("nan")
    return denom if np.isfinite(denom) and denom > 0 else 1.0


def run_backtest(df: pd.DataFrame, windows: int = 5, horizon: int = FORECAST_HORIZON) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[pd.DataFrame] = []
    metrics_records: list[dict[str, float | str | int]] = []
    denom = mase_denominator(df["y"], DAILY_SEASON_LENGTH)
    total_required = WEEKLY_SEASON_LENGTH + horizon * (windows + 1)
    if len(df) < total_required:
        windows = max(1, min(windows, (len(df) - WEEKLY_SEASON_LENGTH) // horizon - 1))
    for window in range(windows, 0, -1):
        split = len(df) - window * horizon
        train = df.iloc[:split].copy()
        test = df.iloc[split : split + horizon].copy()
        final_pred = seasonal_window_average_forecast(train, horizon=horizon)
        base_pred = seasonal_naive_168_forecast(train, horizon=horizon)
        for pred_df, model_name in [(final_pred, "seasonal_window_average"), (base_pred, "seasonal_naive_168")]:
            merged = test[["unique_id", "ds", "y"]].merge(pred_df[["ds", "y_pred"]], on="ds", how="left")
            merged["model"] = model_name
            merged["window"] = windows - window + 1
            merged["residual"] = merged["y"] - merged["y_pred"]
            records.append(merged)
            m = compute_metrics(merged["y"], merged["y_pred"], denom)
            m.update({"model": model_name, "window": windows - window + 1})
            metrics_records.append(m)
    predictions = pd.concat(records, ignore_index=True)
    metrics = pd.DataFrame(metrics_records)
    summary = metrics.groupby("model")[["mae", "rmse", "mape", "smape", "mase"]].mean().reset_index().sort_values("smape")
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))
    return predictions, summary


def statistical_tests(backtest_predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str | float]] = []
    final_errors = backtest_predictions.loc[backtest_predictions["model"] == "seasonal_window_average", "residual"].dropna()
    base_errors = backtest_predictions.loc[backtest_predictions["model"] == "seasonal_naive_168", "residual"].dropna()
    if len(final_errors) >= 10:
        t_stat, pvalue = stats.ttest_1samp(final_errors, popmean=0.0)
        rows.append({"test": "bias_ttest", "statistic": float(t_stat), "pvalue": float(pvalue), "status": "pass" if pvalue >= 0.05 else "warn", "interpretation": "Проверка среднего смещения остатков относительно нуля."})
    if len(final_errors) >= 30:
        lag = min(24, max(1, len(final_errors) // 3))
        lb = acorr_ljungbox(final_errors, lags=[lag], return_df=True)
        pvalue = float(lb["lb_pvalue"].iloc[0])
        rows.append({"test": f"ljung_box_lag_{lag}", "statistic": float(lb["lb_stat"].iloc[0]), "pvalue": pvalue, "status": "pass" if pvalue >= 0.05 else "warn", "interpretation": "Проверка автокорреляции остатков. Малое p-value означает, что в ошибках осталась структура."})
    if len(final_errors) == len(base_errors) and len(final_errors) >= 10:
        final_abs = np.abs(final_errors.to_numpy())
        base_abs = np.abs(base_errors.to_numpy())
        t_stat, pvalue = stats.ttest_rel(final_abs, base_abs)
        rows.append({"test": "paired_ttest_vs_seasonal_naive_168", "statistic": float(t_stat), "pvalue": float(pvalue), "status": "pass" if np.mean(final_abs) <= np.mean(base_abs) else "warn", "interpretation": "Парная проверка абсолютных ошибок выбранного метода против недельного seasonal naive."})
    return pd.DataFrame(rows)


def save_plots(df: pd.DataFrame, model_metrics: pd.DataFrame, backtest_predictions: pd.DataFrame, forecast: pd.DataFrame) -> None:
    top = model_metrics.head(12).copy()
    plt.figure(figsize=(11, 5))
    plt.bar(top["model"].astype(str), top["smape"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("sMAPE")
    plt.title("Сравнение моделей по sMAPE")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_comparison.png", dpi=150)
    plt.close()

    bt = backtest_predictions[backtest_predictions["model"] == "seasonal_window_average"].copy()
    plt.figure(figsize=(12, 5))
    plt.plot(bt["ds"], bt["y"], label="Факт")
    plt.plot(bt["ds"], bt["y_pred"], label="Прогноз")
    plt.title("Backtesting итогового пайплайна")
    plt.xlabel("Дата")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pipeline_backtest.png", dpi=150)
    plt.close()

    history = df.tail(14 * 24)
    plt.figure(figsize=(12, 5))
    plt.plot(history["ds"], history["y"], label="История")
    plt.plot(forecast["ds"], forecast["y_pred"], label="Прогноз на 24 часа")
    plt.title("Финальный прогноз пайплайна")
    plt.xlabel("Дата")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pipeline_forecast.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(bt["ds"], bt["residual"])
    plt.axhline(0, linewidth=1)
    plt.title("Остатки итогового пайплайна")
    plt.xlabel("Дата")
    plt.ylabel("Факт - прогноз")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pipeline_residuals.png", dpi=150)
    plt.close()


def make_report(model_metrics: pd.DataFrame, pipeline_metrics: pd.DataFrame, validation: pd.DataFrame, statistical: pd.DataFrame, performance: pd.DataFrame, selected_model: str, selected_group: str) -> str:
    top_models = model_metrics.head(10).to_markdown(index=False)
    pipeline_metrics_md = pipeline_metrics.to_markdown(index=False)
    validation_md = validation.to_markdown(index=False)
    statistical_md = statistical.to_markdown(index=False) if not statistical.empty else "Статистические тесты не рассчитаны."
    performance_md = performance.to_markdown(index=False)
    return f"""# Пайплайн прогнозирования временного ряда

## Цель раздела

Цель раздела — собрать результаты исследования в единый воспроизводимый пайплайн прогнозирования временного ряда электропотребления. Пайплайн должен принимать подготовленный временной ряд, проверять качество входных данных, выбирать модель на основе результатов предыдущих экспериментов, строить прогноз на 24 часа и сохранять результаты тестирования.

## Структура пайплайна

Пайплайн состоит из следующих компонентов:

1. загрузка подготовленного ряда `data/processed/prepared_time_series.csv`;
2. проверка структуры данных, пропусков, дубликатов и регулярности временной сетки;
3. загрузка таблиц качества статистических, ML- и DL-моделей;
4. выбор лучшей модели по основной метрике `sMAPE`;
5. построение финального прогноза на 24 часа;
6. backtesting итогового операционного метода;
7. статистическое тестирование остатков;
8. тестирование производительности;
9. сохранение таблиц, графиков и markdown-отчета.

## Выбор итоговой модели

По результатам всех предыдущих экспериментов лучшей моделью по `sMAPE` стала:

**`{selected_model}`** из группы **`{selected_group}`**.

Ниже приведены лучшие модели из общей таблицы сравнения:

{top_models}

Для операционного пайплайна используется модель `seasonal_window_average`. Она хорошо согласуется с природой ряда электропотребления: прогноз строится как среднее значение для того же часа недели за несколько предыдущих недель. Такой подход учитывает недельную сезонность и устойчивее к случайным всплескам, чем простой перенос одного значения из прошлого.

## Метрики итогового пайплайна

{pipeline_metrics_md}

## Проверка входных данных

{validation_md}

## Статистическое тестирование

{statistical_md}

Интерпретация: если p-value теста Ljung-Box меньше 0.05, в остатках остается автокорреляционная структура. Это не делает пайплайн некорректным, но показывает, что временной ряд содержит зависимости, которые можно дополнительно улучшать более сложными моделями. Парный t-test используется как дополнительная проверка качества ошибок относительно недельного seasonal naive.

## Тестирование производительности

{performance_md}

Производительность оценивается на локальном запуске. Время работы зависит от компьютера, версии Python и установленных библиотек, поэтому эти значения используются как практический ориентир воспроизводимости пайплайна.

## Сохраненные результаты

Таблицы:

- `reports/tables/model_comparison.csv`
- `reports/tables/pipeline_metrics.csv`
- `reports/tables/pipeline_tests.csv`
- `reports/tables/pipeline_statistical_tests.csv`
- `reports/tables/pipeline_performance.csv`

Прогнозы:

- `outputs/forecasts/pipeline_forecast.csv`
- `outputs/forecasts/pipeline_backtest_predictions.csv`

Графики:

- `reports/figures/model_comparison.png`
- `reports/figures/pipeline_backtest.png`
- `reports/figures/pipeline_forecast.png`
- `reports/figures/pipeline_residuals.png`

## Вывод

Итоговый пайплайн объединяет подготовку данных, результаты выбора моделей, прогнозирование, статистическое тестирование и тестирование производительности. Он является воспроизводимой частью проекта: после запуска `python scripts/run_pipeline.py` формируются таблицы, графики, прогноз и отчет по работе пайплайна.

Пайплайн закрывает финальный этап исследования временного ряда и может использоваться как основа для дальнейшей автоматизации прогнозирования.
"""


def update_final_report() -> None:
    report_path = REPORTS_DIR / "REPORT.md"
    text = """# Отчет об исследовании временного ряда электропотребления

## Описание проекта

В работе исследуется временной ряд бытового электропотребления. Целевая переменная — среднее часовое значение `Global_active_power`. Основная задача — построить и сравнить методы прогнозирования на горизонт 24 часа.

## Структура отчета

Подробные разделы исследования находятся в отдельных markdown-файлах:

- `reports/dataset_and_task.md` — описание временного ряда и постановка задачи;
- `reports/data_preparation_and_eda.md` — логика подготовки данных и EDA;
- `reports/eda_results.md` — результаты подготовки данных и EDA;
- `reports/anomaly_detection.md` — анализ аномалий и выбор метода;
- `reports/statistical_models.md` — статистические методы прогнозирования;
- `reports/machine_learning_models.md` — ML-модели;
- `reports/neural_models.md` — DL-модели NeuralForecast;
- `reports/data_driven_models.md` — общее сравнение ML и DL;
- `reports/pipeline.md` — итоговый пайплайн и результаты его тестирования.

## Краткие выводы

1. Исходный минутный ряд преобразован в регулярный часовой ряд без пропусков в целевой переменной.
2. EDA показал наличие суточной и недельной сезонности, поэтому в моделях используются периоды 24 и 168 часов.
3. Для анализа аномалий выбран Seasonal IQR, так как он учитывает типичный профиль по часу недели.
4. В статистическом блоке лучшей моделью по sMAPE стала `seasonal_window_average`.
5. В ML-блоке сравниваются Ridge Regression, Random Forest и HistGradientBoosting.
6. В DL-блоке через NeuralForecast сравниваются MLP, N-BEATS и NHITS.
7. Итоговый пайплайн собирает результаты экспериментов, выбирает модель, строит прогноз и сохраняет результаты тестирования.

## Итоговые артефакты

Основные результаты сохранены в:

- `data/processed/prepared_time_series.csv`;
- `reports/tables/`;
- `reports/figures/`;
- `outputs/forecasts/`;
- `reports/pipeline.md`.

## Запуск проекта

```bash
python scripts/prepare_eda.py
python scripts/statistical_forecast.py
python scripts/analyze_anomalies.py
python scripts/machine_learning_forecast.py
python scripts/neural_forecast.py
python scripts/run_pipeline.py
```
"""
    report_path.write_text(text, encoding="utf-8")


def run_pipeline() -> PipelineResult:
    ensure_dirs()
    tracemalloc.start()
    start_total = time.perf_counter()
    t0 = time.perf_counter(); df = load_prepared_series(); load_seconds = time.perf_counter() - t0
    t0 = time.perf_counter(); validation = validate_series(df); validation_seconds = time.perf_counter() - t0
    t0 = time.perf_counter(); model_metrics = collect_model_metrics(); model_selection_seconds = time.perf_counter() - t0
    selected = model_metrics.iloc[0]
    selected_model = str(selected["model"]); selected_group = str(selected["group"]); selected_smape = float(selected["smape"])
    t0 = time.perf_counter(); backtest_predictions, pipeline_metrics = run_backtest(df, windows=5, horizon=FORECAST_HORIZON); backtest_seconds = time.perf_counter() - t0
    t0 = time.perf_counter(); forecast = seasonal_window_average_forecast(df, horizon=FORECAST_HORIZON); forecast_seconds = time.perf_counter() - t0
    t0 = time.perf_counter(); stat_tests = statistical_tests(backtest_predictions); statistical_seconds = time.perf_counter() - t0
    t0 = time.perf_counter(); save_plots(df, model_metrics, backtest_predictions, forecast); plotting_seconds = time.perf_counter() - t0
    total_seconds = time.perf_counter() - start_total
    _, peak_bytes = tracemalloc.get_traced_memory(); tracemalloc.stop()
    performance = pd.DataFrame([
        {"stage": "load_data", "seconds": load_seconds},
        {"stage": "validate_data", "seconds": validation_seconds},
        {"stage": "model_selection", "seconds": model_selection_seconds},
        {"stage": "backtesting", "seconds": backtest_seconds},
        {"stage": "forecast", "seconds": forecast_seconds},
        {"stage": "statistical_tests", "seconds": statistical_seconds},
        {"stage": "plots", "seconds": plotting_seconds},
        {"stage": "total", "seconds": total_seconds},
        {"stage": "peak_memory_mb", "seconds": peak_bytes / 1024 / 1024},
    ])
    model_metrics.to_csv(TABLES_DIR / "model_comparison.csv", index=False)
    pipeline_metrics.to_csv(TABLES_DIR / "pipeline_metrics.csv", index=False)
    validation.to_csv(TABLES_DIR / "pipeline_tests.csv", index=False)
    stat_tests.to_csv(TABLES_DIR / "pipeline_statistical_tests.csv", index=False)
    performance.to_csv(TABLES_DIR / "pipeline_performance.csv", index=False)
    forecast.to_csv(FORECASTS_DIR / "pipeline_forecast.csv", index=False)
    backtest_predictions.to_csv(FORECASTS_DIR / "pipeline_backtest_predictions.csv", index=False)
    report = make_report(model_metrics, pipeline_metrics, validation, stat_tests, performance, selected_model, selected_group)
    PIPELINE_REPORT_PATH.write_text(report, encoding="utf-8")
    update_final_report()
    return PipelineResult(selected_model, selected_group, selected_smape, FORECASTS_DIR / "pipeline_forecast.csv", PIPELINE_REPORT_PATH)


if __name__ == "__main__":
    result = run_pipeline()
    print("Готово.")
    print(f"Выбранная модель по общей таблице: {result.selected_model} ({result.selected_group})")
    print(f"sMAPE выбранной модели: {result.selected_smape:.4f}")
    print(f"Прогноз: {result.forecast_path}")
    print(f"Отчет: {result.report_path}")
