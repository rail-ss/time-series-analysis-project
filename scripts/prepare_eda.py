"""Подготовка датасета UCI и EDA.

Запуск из корня проекта:
    python scripts/prepare_eda.py

Перед запуском положите файл:
    data/raw/household_power_consumption.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    EDA_RESULTS_PATH,
    FIGURES_DIR,
    TABLES_DIR,
    FREQ,
    SEASON_LENGTH,
    WEEKLY_SEASON_LENGTH,
    DEFAULT_SERIES_ID,
)
from src.data_preprocessing import prepare_household_hourly_dataset, train_test_split_ts  # noqa: E402
from src.eda import (  # noqa: E402
    descriptive_stats,
    save_dict_as_csv,
    plot_series,
    plot_last_window,
    plot_rolling_stats,
    seasonal_profiles,
    plot_profile,
    decompose_series,
    stationarity_tests,
    plot_acf_pacf_charts,
    write_eda_markdown_summary,
)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/7] Загрузка и подготовка исходного датасета...")
    df, quality, missing_table = prepare_household_hourly_dataset(
        raw_path=RAW_DATA_PATH,
        processed_path=PROCESSED_DATA_PATH,
        freq=FREQ,
        unique_id=DEFAULT_SERIES_ID,
        fill_method="time",
    )
    df["ds"] = pd.to_datetime(df["ds"])

    print(f"Подготовленный ряд сохранен: {PROCESSED_DATA_PATH}")
    print(f"Размер подготовленного ряда: {df.shape}")

    print("[2/7] Сохранение таблиц качества и описательной статистики...")
    quality_path = TABLES_DIR / "quality_report.csv"
    missing_path = TABLES_DIR / "missing_values_raw.csv"
    stats_path = TABLES_DIR / "descriptive_stats.csv"

    save_dict_as_csv(quality, quality_path)
    missing_table.to_csv(missing_path, index=False)
    descriptive_stats(df, target_col="y").to_csv(stats_path)

    print("[3/7] Построение основных графиков ряда...")
    plot_series(
        df,
        date_col="ds",
        target_col="y",
        title="Подготовленный часовой ряд: Global_active_power",
        save_path=FIGURES_DIR / "full_series.png",
    )
    plot_last_window(
        df,
        date_col="ds",
        target_col="y",
        periods=24 * 60,
        title="Последние 60 дней часового ряда",
        save_path=FIGURES_DIR / "last_60_days.png",
    )
    plot_rolling_stats(
        df,
        date_col="ds",
        target_col="y",
        windows=[24, WEEKLY_SEASON_LENGTH],
        save_path=FIGURES_DIR / "rolling_means.png",
    )

    print("[4/7] Построение сезонных профилей...")
    profiles = seasonal_profiles(df, date_col="ds", target_col="y")
    for name, profile in profiles.items():
        profile.to_csv(TABLES_DIR / f"{name}.csv", index=False)

    plot_profile(
        profiles["hourly_profile"],
        x_col="hour",
        y_col="y",
        title="Средний профиль потребления по часу суток",
        save_path=FIGURES_DIR / "hourly_profile.png",
    )
    plot_profile(
        profiles["weekly_profile"],
        x_col="day_of_week",
        y_col="y",
        title="Средний профиль потребления по дню недели",
        save_path=FIGURES_DIR / "weekly_profile.png",
    )
    plot_profile(
        profiles["monthly_profile"],
        x_col="month",
        y_col="y",
        title="Средний профиль потребления по месяцу",
        save_path=FIGURES_DIR / "monthly_profile.png",
    )

    print("[5/7] Декомпозиция последних 90 дней...")
    decompose_series(
        df,
        date_col="ds",
        target_col="y",
        period=SEASON_LENGTH,
        model="additive",
        max_points=24 * 90,
        save_path=FIGURES_DIR / "decomposition_last_90_days.png",
    )

    print("[6/7] ADF/KPSS и ACF/PACF...")
    stationarity = stationarity_tests(df, target_col="y", max_points=10000)
    pd.DataFrame([stationarity]).to_csv(TABLES_DIR / "stationarity_tests.csv", index=False)
    plot_acf_pacf_charts(
        df,
        target_col="y",
        lags=72,
        max_points=10000,
        save_prefix=FIGURES_DIR / "autocorrelation",
    )

    print("[7/7] Создание markdown-отчета по шагу 2...")
    write_eda_markdown_summary(
        path=EDA_RESULTS_PATH,
        prepared_df=df,
        quality_table_path=str(quality_path.relative_to(PROJECT_ROOT)),
        descriptive_stats_path=str(stats_path.relative_to(PROJECT_ROOT)),
        stationarity=stationarity,
    )

    # Закрываем фигуры, чтобы при пакетном запуске не висела память.
    plt.close("all")

    print("Готово.")
    print(f"Markdown-результаты: {EDA_RESULTS_PATH}")
    print(f"Графики: {FIGURES_DIR}")
    print(f"Таблицы: {TABLES_DIR}")


if __name__ == "__main__":
    main()
