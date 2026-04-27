from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    BACKTEST_WINDOWS,
    FIGURES_DIR,
    FORECASTS_DIR,
    FORECAST_HORIZON,
    MACHINE_LEARNING_REPORT_PATH,
    PREPARED_DATA_PATH,
    RANDOM_STATE,
    SEASON_LENGTH_DAY,
    TABLES_DIR,
)
from src.evaluation import summarize_metrics


@dataclass
class FeatureConfig:
    lags: tuple[int, ...] = (1, 2, 3, 6, 12, 24, 48, 72, 168)
    rolling_windows: tuple[int, ...] = (24, 168)
    max_train_size: int = 24 * 365 * 2


def load_series(path: Path = PREPARED_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Не найден подготовленный ряд: {path}. Сначала запустите python scripts/prepare_eda.py"
        )
    df = pd.read_csv(path, parse_dates=["ds"])
    df = df.sort_values("ds").reset_index(drop=True)
    return df[["unique_id", "ds", "y"]]


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["hour"] = result["ds"].dt.hour
    result["dayofweek"] = result["ds"].dt.dayofweek
    result["month"] = result["ds"].dt.month
    result["is_weekend"] = result["dayofweek"].isin([5, 6]).astype(int)

    result["hour_sin"] = np.sin(2 * np.pi * result["hour"] / 24)
    result["hour_cos"] = np.cos(2 * np.pi * result["hour"] / 24)
    result["dow_sin"] = np.sin(2 * np.pi * result["dayofweek"] / 7)
    result["dow_cos"] = np.cos(2 * np.pi * result["dayofweek"] / 7)
    return result


def make_features(df: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    result = add_calendar_features(df)
    shifted = result["y"].shift(1)

    for lag in config.lags:
        result[f"lag_{lag}"] = result["y"].shift(lag)

    for window in config.rolling_windows:
        min_periods = max(2, window // 4)
        result[f"rolling_mean_{window}"] = shifted.rolling(window, min_periods=min_periods).mean()
        result[f"rolling_std_{window}"] = shifted.rolling(window, min_periods=min_periods).std()
        result[f"rolling_min_{window}"] = shifted.rolling(window, min_periods=min_periods).min()
        result[f"rolling_max_{window}"] = shifted.rolling(window, min_periods=min_periods).max()

    return result


def feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {"unique_id", "ds", "y"}
    return [col for col in df.columns if col not in excluded]


def make_models() -> dict:
    return {
        "ridge_regression": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "random_forest": RandomForestRegressor(
            n_estimators=120,
            max_depth=12,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=250,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=0.01,
            random_state=RANDOM_STATE,
        ),
    }


def make_future_row(history: pd.DataFrame, ds: pd.Timestamp, config: FeatureConfig) -> pd.DataFrame:
    row = pd.DataFrame({"unique_id": ["household_1"], "ds": [ds], "y": [np.nan]})
    temp = pd.concat([history[["unique_id", "ds", "y"]], row], ignore_index=True)
    features = make_features(temp, config)
    return features.tail(1)


def recursive_predict(model, history: pd.DataFrame, horizon: int, config: FeatureConfig, cols: list[str]) -> pd.DataFrame:
    history = history[["unique_id", "ds", "y"]].copy()
    last_ds = history["ds"].iloc[-1]
    rows = []

    for step in range(1, horizon + 1):
        next_ds = last_ds + pd.Timedelta(hours=step)
        feature_row = make_future_row(history, next_ds, config)
        X_next = feature_row[cols].ffill(axis=1).fillna(0)
        y_pred = float(model.predict(X_next)[0])

        rows.append({"ds": next_ds, "horizon": step, "y_pred": y_pred})
        history = pd.concat(
            [
                history,
                pd.DataFrame({"unique_id": ["household_1"], "ds": [next_ds], "y": [y_pred]}),
            ],
            ignore_index=True,
        )

    return pd.DataFrame(rows)


def fit_predict_window(
    model,
    model_name: str,
    df: pd.DataFrame,
    train_end: int,
    horizon: int,
    window_id: int,
    config: FeatureConfig,
) -> tuple[pd.DataFrame, np.ndarray]:
    train_raw = df.iloc[:train_end].copy()
    test_raw = df.iloc[train_end : train_end + horizon].copy()

    train_for_fit = train_raw.tail(config.max_train_size).copy()
    featured = make_features(train_for_fit, config)
    cols = feature_columns(featured)
    featured = featured.dropna(subset=cols + ["y"]).copy()

    X_train = featured[cols]
    y_train = featured["y"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)

    pred = recursive_predict(model, train_raw, horizon, config, cols)
    pred["unique_id"] = "household_1"
    pred["model"] = model_name
    pred["group"] = "ML"
    pred["window"] = window_id
    pred["y"] = test_raw["y"].values
    pred["ds"] = test_raw["ds"].values
    return pred[["unique_id", "ds", "window", "horizon", "group", "model", "y", "y_pred"]], train_raw["y"].values


def run_backtesting(df: pd.DataFrame, config: FeatureConfig) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    models = make_models()
    h = FORECAST_HORIZON
    n_windows = BACKTEST_WINDOWS
    first_train_end = len(df) - h * n_windows

    predictions = []
    insample_by_window = {}

    for window in range(1, n_windows + 1):
        train_end = first_train_end + (window - 1) * h
        print(f"Окно {window}/{n_windows}: train={train_end}, test={h}, models={len(models)}")
        insample_by_window[window] = df.iloc[:train_end]["y"].values

        for model_name, model in models.items():
            pred, _ = fit_predict_window(
                model=model,
                model_name=model_name,
                df=df,
                train_end=train_end,
                horizon=h,
                window_id=window,
                config=config,
            )
            predictions.append(pred)

    return pd.concat(predictions, ignore_index=True), insample_by_window


def train_final_forecasts(df: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    models = make_models()
    featured = make_features(df.tail(config.max_train_size), config)
    cols = feature_columns(featured)
    featured = featured.dropna(subset=cols + ["y"]).copy()

    forecasts = []
    for model_name, model in models.items():
        print(f"Финальный прогноз: {model_name}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(featured[cols], featured["y"])

        pred = recursive_predict(model, df, FORECAST_HORIZON, config, cols)
        pred["unique_id"] = "household_1"
        pred["model"] = model_name
        pred["group"] = "ML"
        pred = pred[["unique_id", "ds", "horizon", "group", "model", "y_pred"]]
        forecasts.append(pred)

    return pd.concat(forecasts, ignore_index=True)


def make_selection_table(metrics_summary: pd.DataFrame) -> pd.DataFrame:
    comments = {
        "ridge_regression": "Линейная модель с регуляризацией; проверяет, достаточно ли линейной связи между лагами, календарными признаками и будущим значением.",
        "random_forest": "Нелинейная ансамблевая модель, устойчивая к выбросам и хорошо работающая с лаговыми признаками.",
        "hist_gradient_boosting": "Градиентный бустинг по деревьям; обычно силен на табличных признаках временного ряда.",
    }

    rows = []
    for _, row in metrics_summary.iterrows():
        model = row["model"]
        rows.append(
            {
                "rank": int(row["rank"]),
                "group": "ML",
                "model": model,
                "mae": row["mae"],
                "rmse": row["rmse"],
                "smape": row["smape"],
                "mase": row["mase"],
                "comment": comments.get(model, ""),
                "decision": "основной ML-кандидат" if int(row["rank"]) == 1 else "сравнительная модель",
            }
        )
    return pd.DataFrame(rows)


def plot_results(metrics_summary: pd.DataFrame, predictions: pd.DataFrame, forecasts: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    order = metrics_summary.sort_values("smape")
    plt.bar(order["model"], order["smape"])
    plt.title("Сравнение ML-моделей по sMAPE")
    plt.ylabel("sMAPE")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "machine_learning_metrics.png", dpi=160)
    plt.close()

    best_model = metrics_summary.iloc[0]["model"]
    best_predictions = predictions[predictions["model"] == best_model].copy()

    plt.figure(figsize=(14, 5))
    plt.plot(best_predictions["ds"], best_predictions["y"], label="actual", linewidth=1.5)
    plt.plot(best_predictions["ds"], best_predictions["y_pred"], label=best_model, linewidth=1.5)
    plt.title("Backtesting лучшей ML-модели")
    plt.xlabel("Дата")
    plt.ylabel("Средняя активная мощность")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "machine_learning_backtest.png", dpi=160)
    plt.close()

    best_forecast = forecasts[forecasts["model"] == best_model].copy()
    plt.figure(figsize=(12, 5))
    plt.plot(best_forecast["ds"], best_forecast["y_pred"], marker="o", label=best_model)
    plt.title("Финальный прогноз ML-модели на 24 часа")
    plt.xlabel("Дата")
    plt.ylabel("Прогноз")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "machine_learning_forecast.png", dpi=160)
    plt.close()


def make_report(metrics_summary: pd.DataFrame, selection: pd.DataFrame, best_model: str) -> str:
    metrics_md = metrics_summary.to_markdown(index=False)
    selection_md = selection.to_markdown(index=False)
    best_row = metrics_summary[metrics_summary["model"] == best_model].iloc[0]

    return f"""# ML-модели прогнозирования

## Цель раздела

Цель раздела — сравнить классические методы машинного обучения для прогнозирования часового электропотребления. Прогнозирование сведено к задаче табличной регрессии: целевая переменная прогнозируется по лагам, скользящим статистикам и календарным признакам.

## Feature engineering

Для моделей сформированы признаки:

- лаги целевой переменной: `1`, `2`, `3`, `6`, `12`, `24`, `48`, `72`, `168`;
- скользящие статистики по окнам `24` и `168` часов;
- календарные признаки: час суток, день недели, месяц, признак выходного дня;
- циклическое кодирование часа суток и дня недели через `sin` и `cos`.

Такая схема соответствует логике `mlforecast`: временной ряд преобразуется в набор признаков, после чего к нему можно применять регрессионные ML-модели.

## Использованные модели

| Группа | Модель | Параметры |
|---|---|---|
| ML | `ridge_regression` | линейная модель с L2-регуляризацией, `alpha=1.0` |
| ML | `random_forest` | 120 деревьев, ограничение глубины 12 |
| ML | `hist_gradient_boosting` | 250 итераций, learning rate 0.05 |

## Схема проверки качества

Использован backtesting по временным окнам. В каждом окне модели обучаются только на прошлом и прогнозируют следующие 24 часа. Это исключает перемешивание прошлого и будущего.

## Результаты backtesting

{metrics_md}

## Таблица выбора ML-моделей

{selection_md}

## Выбранная ML-модель

По среднему значению `sMAPE` лучшей ML-моделью стала:

**`{best_model}`**

Метрики лучшей ML-модели:

- MAE: **{best_row['mae']:.4f}**
- RMSE: **{best_row['rmse']:.4f}**
- sMAPE: **{best_row['smape']:.4f}**
- MASE: **{best_row['mase']:.4f}**

Выбор сделан по `sMAPE`, но дополнительно учитывались `MAE`, `RMSE` и `MASE`. Это важно, потому что одна относительная метрика может быть нестабильной при низких значениях электропотребления.

## Обоснование параметров

- лаги `24` и `168` отражают суточную и недельную сезонность;
- глубина деревьев ограничена, чтобы снизить переобучение;
- регуляризация в Ridge и HistGradientBoosting используется для повышения устойчивости;
- для всех моделей используется одинаковая схема backtesting, поэтому результаты сопоставимы.

## Сохраненные результаты

Таблицы:

- `reports/tables/machine_learning_metrics_by_window.csv`
- `reports/tables/machine_learning_metrics.csv`
- `reports/tables/machine_learning_model_selection.csv`

Прогнозы:

- `outputs/forecasts/machine_learning_predictions.csv`
- `outputs/forecasts/machine_learning_forecast.csv`

Графики:

- `reports/figures/machine_learning_metrics.png`
- `reports/figures/machine_learning_backtest.png`
- `reports/figures/machine_learning_forecast.png`

## Вывод

ML-модели позволяют использовать больше информации о временном ряде, чем простые статистические методы: лаги, календарные признаки, скользящие статистики и нелинейные зависимости. Результаты этого раздела используются дальше вместе с DL-моделями в итоговом data-driven сравнении.
"""


def run_data_driven_analysis() -> None:
    print("[1/7] Загрузка подготовленного ряда...")
    df = load_series()
    print(f"Наблюдений: {len(df)}")
    print(f"Период: {df['ds'].min()} — {df['ds'].max()}")

    config = FeatureConfig()

    print("[2/7] Backtesting ML-моделей...")
    predictions, insample_by_window = run_backtesting(df, config)

    print("[3/7] Расчет метрик...")
    metrics_by_window, metrics_summary = summarize_metrics(
        predictions,
        insample_by_window=insample_by_window,
        season_length=SEASON_LENGTH_DAY,
    )

    print("[4/7] Финальный прогноз на 24 часа...")
    forecasts = train_final_forecasts(df, config)

    print("[5/7] Таблица выбора ML-моделей...")
    selection = make_selection_table(metrics_summary)
    best_model = metrics_summary.iloc[0]["model"]
    print(f"Лучшая ML-модель по sMAPE: {best_model}")

    print("[6/7] Сохранение таблиц и графиков...")
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FORECASTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_by_window.to_csv(TABLES_DIR / "machine_learning_metrics_by_window.csv", index=False)
    metrics_summary.to_csv(TABLES_DIR / "machine_learning_metrics.csv", index=False)
    selection.to_csv(TABLES_DIR / "machine_learning_model_selection.csv", index=False)
    predictions.to_csv(FORECASTS_DIR / "machine_learning_predictions.csv", index=False)
    forecasts.to_csv(FORECASTS_DIR / "machine_learning_forecast.csv", index=False)

    plot_results(metrics_summary, predictions, forecasts)

    print("[7/7] Создание markdown-отчета...")
    report = make_report(metrics_summary, selection, best_model)
    MACHINE_LEARNING_REPORT_PATH.write_text(report, encoding="utf-8")

    print("Готово.")
    print(f"Markdown-результаты: {MACHINE_LEARNING_REPORT_PATH}")
    print(f"Таблицы: {TABLES_DIR}")
    print(f"Графики: {FIGURES_DIR}")
    print(f"Прогнозы: {FORECASTS_DIR}")


if __name__ == "__main__":
    run_data_driven_analysis()
