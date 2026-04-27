"""Базовый пайплайн решения задачи прогнозирования временного ряда.

На текущем этапе пайплайн умеет:
- подготовить UCI-датасет, если файл уже лежит в data/raw/;
- использовать подготовленный файл data/processed/prepared_time_series.csv;
- в крайнем случае запускаться на синтетическом примере для smoke-test.

Дальше он будет расширен аномалиями, статистическими, ML- и DL-моделями.
"""

from __future__ import annotations

import pandas as pd

from src.config import (
    RAW_DATA_PATH,
    SAMPLE_DATA_PATH,
    PROCESSED_DATA_PATH,
    DATE_COL,
    TARGET_COL,
    FREQ,
    SEASON_LENGTH,
    HORIZON,
    TEST_SIZE,
    DEFAULT_SERIES_ID,
)
from src.data_preprocessing import (
    load_time_series,
    make_regular_time_index,
    prepare_household_hourly_dataset,
    train_test_split_ts,
)
from src.statistical_models import naive_forecast, seasonal_naive_forecast
from src.evaluation import evaluate_forecast, metrics_table


def load_or_prepare_series(use_sample_if_missing: bool = True) -> pd.DataFrame:
    """Загружает подготовленный ряд или готовит его из исходных данных."""
    if PROCESSED_DATA_PATH.exists():
        df = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=[DATE_COL])
        return df

    if RAW_DATA_PATH.exists():
        df, _, _ = prepare_household_hourly_dataset(
            raw_path=RAW_DATA_PATH,
            processed_path=PROCESSED_DATA_PATH,
            freq=FREQ,
            unique_id=DEFAULT_SERIES_ID,
            fill_method="time",
        )
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])
        return df

    if use_sample_if_missing and SAMPLE_DATA_PATH.exists():
        df = load_time_series(SAMPLE_DATA_PATH, date_col=DATE_COL, target_col=TARGET_COL)
        df = make_regular_time_index(df, freq=FREQ, date_col=DATE_COL, target_col=TARGET_COL)
        df.insert(0, "unique_id", "sample_series")
        PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(PROCESSED_DATA_PATH, index=False)
        return df

    raise FileNotFoundError(
        "Не найден ни подготовленный файл, ни исходный датасет. "
        "Положите household_power_consumption.txt в data/raw/."
    )


def run_baseline_pipeline() -> pd.DataFrame:
    """Запускает минимальный пайплайн с двумя baseline-моделями."""
    df = load_or_prepare_series(use_sample_if_missing=True)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    effective_test_size = min(TEST_SIZE, max(HORIZON, len(df) // 5))
    train, test = train_test_split_ts(df, test_size=effective_test_size)
    test = test.iloc[:HORIZON].copy()

    y_true = test[TARGET_COL].to_numpy()
    y_train = train[TARGET_COL].to_numpy()

    forecasts = {
        "Naive": naive_forecast(train, len(test), target_col=TARGET_COL),
        "SeasonalNaive": seasonal_naive_forecast(
            train,
            len(test),
            season_length=SEASON_LENGTH,
            target_col=TARGET_COL,
        ),
    }

    rows = []
    for model_name, y_pred in forecasts.items():
        row = {"model": model_name}
        row.update(evaluate_forecast(y_true, y_pred, y_train=y_train, seasonality=SEASON_LENGTH))
        rows.append(row)

    result = metrics_table(rows)
    return result


if __name__ == "__main__":
    metrics = run_baseline_pipeline()
    print(metrics)
