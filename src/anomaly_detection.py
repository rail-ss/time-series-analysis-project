from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.config import (
    ANOMALY_REPORT_PATH,
    FIGURES_DIR,
    PREPARED_DATA_PATH,
    RANDOM_STATE,
    TABLES_DIR,
)


@dataclass
class AnomalyConfig:
    rolling_window: int = 168
    rolling_z_threshold: float = 3.5
    iqr_multiplier: float = 3.0
    isolation_contamination: float = 0.01


def load_series(path: Path = PREPARED_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Не найден подготовленный ряд: {path}. Сначала запустите python scripts/prepare_eda.py"
        )
    df = pd.read_csv(path, parse_dates=["ds"])
    df = df.sort_values("ds").reset_index(drop=True)
    return df[["unique_id", "ds", "y"]]


def detect_rolling_zscore(df: pd.DataFrame, config: AnomalyConfig) -> pd.DataFrame:
    result = df.copy()
    rolling_median = result["y"].rolling(config.rolling_window, min_periods=config.rolling_window // 2).median()
    rolling_std = result["y"].rolling(config.rolling_window, min_periods=config.rolling_window // 2).std()

    score = (result["y"] - rolling_median) / rolling_std.replace(0, np.nan)
    result["rolling_z_score"] = score
    result["rolling_z_anomaly"] = score.abs() > config.rolling_z_threshold
    return result[["ds", "rolling_z_score", "rolling_z_anomaly"]]


def detect_seasonal_iqr(df: pd.DataFrame, config: AnomalyConfig) -> pd.DataFrame:
    result = df.copy()
    result["hour_of_week"] = result["ds"].dt.dayofweek * 24 + result["ds"].dt.hour

    seasonal_profile = result.groupby("hour_of_week")["y"].median()
    result["seasonal_expected"] = result["hour_of_week"].map(seasonal_profile)
    result["seasonal_residual"] = result["y"] - result["seasonal_expected"]

    q1 = result["seasonal_residual"].quantile(0.25)
    q3 = result["seasonal_residual"].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - config.iqr_multiplier * iqr
    upper = q3 + config.iqr_multiplier * iqr

    result["seasonal_iqr_score"] = result["seasonal_residual"]
    result["seasonal_iqr_anomaly"] = (result["seasonal_residual"] < lower) | (result["seasonal_residual"] > upper)
    return result[["ds", "seasonal_expected", "seasonal_iqr_score", "seasonal_iqr_anomaly"]]


def make_isolation_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["lag_1"] = result["y"].shift(1)
    result["lag_24"] = result["y"].shift(24)
    result["lag_168"] = result["y"].shift(168)
    result["rolling_mean_24"] = result["y"].shift(1).rolling(24, min_periods=12).mean()
    result["rolling_std_24"] = result["y"].shift(1).rolling(24, min_periods=12).std()
    result["rolling_mean_168"] = result["y"].shift(1).rolling(168, min_periods=48).mean()
    result["hour"] = result["ds"].dt.hour
    result["dayofweek"] = result["ds"].dt.dayofweek
    result["month"] = result["ds"].dt.month
    return result


def detect_isolation_forest(df: pd.DataFrame, config: AnomalyConfig) -> pd.DataFrame:
    features_df = make_isolation_features(df)
    feature_cols = [
        "y",
        "lag_1",
        "lag_24",
        "lag_168",
        "rolling_mean_24",
        "rolling_std_24",
        "rolling_mean_168",
        "hour",
        "dayofweek",
        "month",
    ]
    valid = features_df.dropna(subset=feature_cols).copy()

    model = IsolationForest(
        n_estimators=200,
        contamination=config.isolation_contamination,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        labels = model.fit_predict(valid[feature_cols])

    valid["isolation_score"] = -model.decision_function(valid[feature_cols])
    valid["isolation_anomaly"] = labels == -1

    result = df[["ds"]].merge(valid[["ds", "isolation_score", "isolation_anomaly"]], on="ds", how="left")
    result["isolation_anomaly"] = result["isolation_anomaly"].fillna(False).astype(bool)
    return result


def combine_anomalies(df: pd.DataFrame, config: AnomalyConfig) -> pd.DataFrame:
    rolling = detect_rolling_zscore(df, config)
    seasonal = detect_seasonal_iqr(df, config)
    isolation = detect_isolation_forest(df, config)

    result = df.copy()
    result = result.merge(rolling, on="ds", how="left")
    result = result.merge(seasonal, on="ds", how="left")
    result = result.merge(isolation, on="ds", how="left")

    flags = ["rolling_z_anomaly", "seasonal_iqr_anomaly", "isolation_anomaly"]
    for col in flags:
        result[col] = result[col].fillna(False).astype(bool)

    result["anomaly_votes"] = result[flags].sum(axis=1)
    result["selected_anomaly"] = result["seasonal_iqr_anomaly"]

    return result


def make_summary(anomalies: pd.DataFrame) -> pd.DataFrame:
    total = len(anomalies)
    rows = []
    for method, col in [
        ("rolling_z_score", "rolling_z_anomaly"),
        ("seasonal_iqr", "seasonal_iqr_anomaly"),
        ("isolation_forest", "isolation_anomaly"),
        ("selected_method", "selected_anomaly"),
    ]:
        count = int(anomalies[col].sum())
        rows.append(
            {
                "method": method,
                "anomalies": count,
                "share_percent": round(count / total * 100, 3),
            }
        )
    return pd.DataFrame(rows)


def plot_anomalies(df: pd.DataFrame, anomalies: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    selected = anomalies[anomalies["selected_anomaly"]].copy()

    plt.figure(figsize=(14, 5))
    plt.plot(anomalies["ds"], anomalies["y"], linewidth=0.8)
    if len(selected) > 0:
        plt.scatter(selected["ds"], selected["y"], s=12)
    plt.title("Аномалии во временном ряду электропотребления")
    plt.xlabel("Дата")
    plt.ylabel("Средняя активная мощность")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "anomalies_full_series.png", dpi=160)
    plt.close()

    last_days = anomalies.tail(24 * 90)
    selected_last = last_days[last_days["selected_anomaly"]]
    plt.figure(figsize=(14, 5))
    plt.plot(last_days["ds"], last_days["y"], linewidth=0.9)
    if len(selected_last) > 0:
        plt.scatter(selected_last["ds"], selected_last["y"], s=18)
    plt.title("Аномалии за последние 90 дней")
    plt.xlabel("Дата")
    plt.ylabel("Средняя активная мощность")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "anomalies_last_90_days.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    counts = [
        anomalies["rolling_z_anomaly"].sum(),
        anomalies["seasonal_iqr_anomaly"].sum(),
        anomalies["isolation_anomaly"].sum(),
    ]
    labels = ["rolling z-score", "seasonal IQR", "Isolation Forest"]
    plt.bar(labels, counts)
    plt.title("Количество найденных аномалий по методам")
    plt.ylabel("Количество наблюдений")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "anomaly_methods_comparison.png", dpi=160)
    plt.close()


def make_report(anomalies: pd.DataFrame, summary: pd.DataFrame, config: AnomalyConfig) -> str:
    summary_md = summary.to_markdown(index=False)

    selected_count = int(anomalies["selected_anomaly"].sum())
    selected_share = selected_count / len(anomalies) * 100

    report = f"""# Анализ аномалий временного ряда

## Цель раздела

Цель раздела — выявить нетипичные наблюдения во временном ряду электропотребления и обосновать выбор метода анализа аномалий. Для временных рядов электропотребления важно не считать обычные вечерние или утренние пики аномалиями, потому что такие пики могут быть частью нормальной суточной сезонности.

## Использованные методы

Были рассмотрены три метода:

| Метод | Параметры | Логика |
|---|---|---|
| Rolling z-score | окно `{config.rolling_window}` часов, порог `{config.rolling_z_threshold}` | Сравнивает значение с локальной медианой и стандартным отклонением. |
| Seasonal IQR | сезонный профиль по часу недели, множитель IQR `{config.iqr_multiplier}` | Сначала учитывает типичный час внутри недели, затем ищет выбросы в остатках. |
| Isolation Forest | contamination `{config.isolation_contamination}` | Многомерный метод, использующий лаги, скользящие признаки и календарные признаки. |

## Результаты сравнения методов

{summary_md}

## Выбор основного метода

В качестве основного метода выбран **Seasonal IQR**.

Причина выбора: ряд электропотребления имеет выраженную суточную и недельную сезонность. Если искать выбросы прямо по исходным значениям, то нормальные пики потребления могут ошибочно попасть в аномалии. Метод Seasonal IQR сначала строит типичный профиль по часу недели, затем анализирует остатки относительно этого профиля. Поэтому он лучше согласуется со структурой данного временного ряда.

## Обоснование параметров

- Сезонный профиль строится по признаку `hour_of_week`, то есть по 168 часам недели. Это соответствует недельному периоду для часовых данных.
- Множитель IQR равен `{config.iqr_multiplier}`. Это более строгий порог, чем классические `1.5 * IQR`, потому что в электропотреблении возможны естественные резкие пики. Более мягкий порог дал бы слишком много ложных срабатываний.
- Isolation Forest использует `contamination={config.isolation_contamination}`. Это означает ожидание небольшой доли аномалий и предотвращает ситуацию, когда модель принудительно помечает слишком много нормальных наблюдений.

## Итог по выбранному методу

Основной метод нашел **{selected_count}** аномальных наблюдений, что составляет **{selected_share:.3f}%** от подготовленного часового ряда.

Эти наблюдения не удаляются автоматически из данных. На следующих этапах они используются как диагностическая информация: можно анализировать, ухудшают ли они качество прогноза, и при необходимости добавить признак аномальности в ML-модели.

## Сохраненные результаты

Таблицы:

- `reports/tables/anomaly_summary.csv`
- `reports/tables/anomalies.csv`

Данные:

- `data/processed/time_series_with_anomalies.csv`

Графики:

- `reports/figures/anomalies_full_series.png`
- `reports/figures/anomalies_last_90_days.png`
- `reports/figures/anomaly_methods_comparison.png`

## Вывод

Для данного временного ряда наиболее обоснованным является метод Seasonal IQR, поскольку он учитывает недельную сезонность и анализирует не исходные значения, а отклонения от типичного сезонного профиля. Rolling z-score полезен как простой локальный метод, а Isolation Forest — как многомерная проверка с лаговыми и календарными признаками.
"""
    return report


def run_anomaly_analysis() -> None:
    print("[1/5] Загрузка подготовленного ряда...")
    df = load_series()
    print(f"Наблюдений: {len(df)}")

    config = AnomalyConfig()

    print("[2/5] Расчет аномалий тремя методами...")
    anomalies = combine_anomalies(df, config)

    print("[3/5] Сохранение таблиц...")
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    summary = make_summary(anomalies)
    summary.to_csv(TABLES_DIR / "anomaly_summary.csv", index=False)
    anomalies.to_csv(TABLES_DIR / "anomalies.csv", index=False)

    processed_path = PREPARED_DATA_PATH.parent / "time_series_with_anomalies.csv"
    anomalies[["unique_id", "ds", "y", "selected_anomaly", "anomaly_votes"]].to_csv(processed_path, index=False)

    print("[4/5] Построение графиков...")
    plot_anomalies(df, anomalies)

    print("[5/5] Создание markdown-отчета...")
    report = make_report(anomalies, summary, config)
    ANOMALY_REPORT_PATH.write_text(report, encoding="utf-8")

    print("Готово.")
    print(f"Markdown-результаты: {ANOMALY_REPORT_PATH}")
    print(f"Таблицы: {TABLES_DIR}")
    print(f"Графики: {FIGURES_DIR}")


if __name__ == "__main__":
    run_anomaly_analysis()
