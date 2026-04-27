from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    BACKTEST_WINDOWS,
    DATA_DRIVEN_REPORT_PATH,
    FIGURES_DIR,
    FORECASTS_DIR,
    FORECAST_HORIZON,
    FREQUENCY,
    NEURAL_MAX_STEPS,
    NEURAL_MAX_TRAIN_SIZE,
    NEURAL_REPORT_PATH,
    PREPARED_DATA_PATH,
    RANDOM_STATE,
    SEASON_LENGTH_DAY,
    SEASON_LENGTH_WEEK,
    TABLES_DIR,
)
from src.evaluation import summarize_metrics


@dataclass
class NeuralConfig:
    input_size: int = SEASON_LENGTH_WEEK
    max_train_size: int = NEURAL_MAX_TRAIN_SIZE
    max_steps: int = NEURAL_MAX_STEPS


def load_series(path: Path = PREPARED_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Не найден подготовленный ряд: {path}. Сначала запустите python scripts/prepare_eda.py"
        )
    df = pd.read_csv(path, parse_dates=["ds"])
    df = df.sort_values("ds").reset_index(drop=True)
    return df[["unique_id", "ds", "y"]]


def import_neuralforecast():
    try:
        from neuralforecast import NeuralForecast
        from neuralforecast.models import MLP, NBEATS, NHITS
    except Exception as exc:
        raise RuntimeError(
            "Не удалось импортировать neuralforecast. Установите зависимости командой: "
            "python -m pip install -r requirements.txt. "
            "Если установка neuralforecast или torch не проходит на Python 3.13, "
            "создайте окружение Python 3.11 или 3.12 и повторите установку."
        ) from exc
    return NeuralForecast, MLP, NBEATS, NHITS


def create_model(model_class, params: dict):
    """Создает модель и мягко убирает параметры, если версия neuralforecast их не поддерживает."""
    candidates = [params.copy()]
    optional_keys = ["enable_progress_bar", "logger", "alias", "random_seed"]
    stripped = params.copy()
    for key in optional_keys:
        stripped.pop(key, None)
    candidates.append(stripped)

    last_error = None
    for candidate in candidates:
        try:
            return model_class(**candidate)
        except TypeError as exc:
            last_error = exc
    raise last_error


def make_models(config: NeuralConfig):
    _, MLP, NBEATS, NHITS = import_neuralforecast()
    common = {
        "h": FORECAST_HORIZON,
        "input_size": config.input_size,
        "max_steps": config.max_steps,
        "learning_rate": 1e-3,
        "scaler_type": "robust",
        "random_seed": RANDOM_STATE,
        "enable_progress_bar": False,
        "logger": False,
    }

    neural_mlp = create_model(
        MLP,
        {
            **common,
            "alias": "neural_mlp",
            "num_layers": 2,
            "hidden_size": 128,
        },
    )
    nbeats = create_model(
        NBEATS,
        {
            **common,
            "alias": "nbeats",
            "n_blocks": [1, 1, 1],
            "mlp_units": 3 * [[128, 128]],
        },
    )
    nhits = create_model(
        NHITS,
        {
            **common,
            "alias": "nhits",
            "n_blocks": [1, 1, 1],
            "mlp_units": 3 * [[128, 128]],
            "n_pool_kernel_size": [2, 2, 1],
            "n_freq_downsample": [4, 2, 1],
        },
    )
    return [neural_mlp, nbeats, nhits]


def model_name_map(name: str) -> str:
    mapping = {
        "MLP": "neural_mlp",
        "NBEATS": "nbeats",
        "NHITS": "nhits",
        "neural_mlp": "neural_mlp",
        "nbeats": "nbeats",
        "nhits": "nhits",
    }
    return mapping.get(name, name)


def melt_forecasts(df: pd.DataFrame, include_y: bool) -> pd.DataFrame:
    base_cols = ["unique_id", "ds"]
    if "cutoff" in df.columns:
        base_cols.append("cutoff")
    if include_y and "y" in df.columns:
        base_cols.append("y")

    model_cols = [col for col in df.columns if col not in base_cols]
    long = df.melt(id_vars=base_cols, value_vars=model_cols, var_name="raw_model", value_name="y_pred")
    long["model"] = long["raw_model"].map(model_name_map)
    long["group"] = "DL"
    long = long.drop(columns=["raw_model"])
    return long


def run_backtesting(df: pd.DataFrame, config: NeuralConfig):
    NeuralForecast, _, _, _ = import_neuralforecast()
    h = FORECAST_HORIZON
    n_windows = BACKTEST_WINDOWS
    # Берем последний год плюс тестовые окна: это заметно ускоряет DL-обучение на обычном ноутбуке.
    work_size = config.max_train_size + h * n_windows
    work_df = df.tail(work_size).copy()

    models = make_models(config)
    nf = NeuralForecast(models=models, freq=FREQUENCY)

    print(f"NeuralForecast backtesting: train/test rows={len(work_df)}, models={len(models)}, windows={n_windows}")
    cv = nf.cross_validation(df=work_df, n_windows=n_windows, step_size=h, verbose=0)
    cv = cv.reset_index(drop=True)

    predictions = melt_forecasts(cv, include_y=True)
    cutoffs = sorted(pd.to_datetime(predictions["cutoff"].unique()))
    cutoff_to_window = {cutoff: i + 1 for i, cutoff in enumerate(cutoffs)}
    predictions["cutoff"] = pd.to_datetime(predictions["cutoff"])
    predictions["window"] = predictions["cutoff"].map(cutoff_to_window)
    predictions = predictions.sort_values(["model", "window", "ds"]).reset_index(drop=True)
    predictions["horizon"] = predictions.groupby(["model", "window"]).cumcount() + 1
    predictions = predictions[["unique_id", "ds", "window", "horizon", "group", "model", "y", "y_pred", "cutoff"]]

    insample_by_window = {}
    for cutoff, window_id in cutoff_to_window.items():
        insample_by_window[window_id] = work_df[work_df["ds"] <= cutoff]["y"].values

    return predictions, insample_by_window, work_df


def train_final_forecasts(df: pd.DataFrame, config: NeuralConfig) -> pd.DataFrame:
    NeuralForecast, _, _, _ = import_neuralforecast()
    train_df = df.tail(config.max_train_size).copy()
    models = make_models(config)
    nf = NeuralForecast(models=models, freq=FREQUENCY)

    print(f"Финальное обучение NeuralForecast на {len(train_df)} наблюдениях")
    nf.fit(df=train_df)
    wide_forecast = nf.predict().reset_index(drop=True)
    forecasts = melt_forecasts(wide_forecast, include_y=False)
    forecasts = forecasts.sort_values(["model", "ds"]).reset_index(drop=True)
    forecasts["horizon"] = forecasts.groupby("model").cumcount() + 1
    return forecasts[["unique_id", "ds", "horizon", "group", "model", "y_pred"]]


def make_selection_table(metrics_summary: pd.DataFrame) -> pd.DataFrame:
    comments = {
        "neural_mlp": "MLP-модель NeuralForecast: нелинейный прогноз по историческому окну без явной декомпозиции ряда.",
        "nbeats": "N-BEATS: нейросетевая модель с остаточными блоками, способная разлагать ряд на компоненты и учитывать тренд/сезонность.",
        "nhits": "NHITS: иерархическая нейросетевая модель, использующая multi-rate обработку для разных частот ряда.",
    }
    rows = []
    for _, row in metrics_summary.iterrows():
        model = row["model"]
        rows.append(
            {
                "rank": int(row["rank"]),
                "group": "DL",
                "model": model,
                "mae": row["mae"],
                "rmse": row["rmse"],
                "smape": row["smape"],
                "mase": row["mase"],
                "comment": comments.get(model, ""),
                "decision": "основной DL-кандидат" if int(row["rank"]) == 1 else "сравнительная модель",
            }
        )
    return pd.DataFrame(rows)


def plot_neural_results(metrics_summary: pd.DataFrame, predictions: pd.DataFrame, forecasts: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    order = metrics_summary.sort_values("smape")
    plt.bar(order["model"], order["smape"])
    plt.title("Сравнение DL-моделей по sMAPE")
    plt.ylabel("sMAPE")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "neural_metrics.png", dpi=160)
    plt.close()

    best_model = metrics_summary.iloc[0]["model"]
    best_predictions = predictions[predictions["model"] == best_model].copy()
    plt.figure(figsize=(14, 5))
    plt.plot(best_predictions["ds"], best_predictions["y"], label="actual", linewidth=1.5)
    plt.plot(best_predictions["ds"], best_predictions["y_pred"], label=best_model, linewidth=1.5)
    plt.title("Backtesting лучшей DL-модели")
    plt.xlabel("Дата")
    plt.ylabel("Средняя активная мощность")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "neural_backtest.png", dpi=160)
    plt.close()

    best_forecast = forecasts[forecasts["model"] == best_model].copy()
    plt.figure(figsize=(12, 5))
    plt.plot(best_forecast["ds"], best_forecast["y_pred"], marker="o", label=best_model)
    plt.title("Финальный прогноз DL-модели на 24 часа")
    plt.xlabel("Дата")
    plt.ylabel("Прогноз")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "neural_forecast.png", dpi=160)
    plt.close()


def make_neural_report(metrics_summary: pd.DataFrame, selection: pd.DataFrame, best_model: str, config: NeuralConfig) -> str:
    metrics_md = metrics_summary.to_markdown(index=False)
    selection_md = selection.to_markdown(index=False)
    best_row = metrics_summary[metrics_summary["model"] == best_model].iloc[0]

    return f"""# Нейросетевые модели прогнозирования

## Цель раздела

Цель раздела — проверить DL-подходы из `neuralforecast` для прогнозирования часового электропотребления и сравнить несколько разных нейросетевых архитектур.

## Использованные модели

| Группа | Модель | Параметры и логика |
|---|---|---|
| DL | `neural_mlp` | MLP-модель для нелинейного прогноза по историческому окну |
| DL | `nbeats` | N-BEATS, нейросетевая модель с остаточными блоками и декомпозиционной логикой |
| DL | `nhits` | NHITS, иерархическая нейросетевая модель для прогнозирования временных рядов |

## Настройка эксперимента

- Горизонт прогноза: 24 часа.
- Историческое окно входа: {config.input_size} часов.
- Максимальное число шагов обучения: {config.max_steps}.
- Для локального запуска обучение ограничено последним годом наблюдений, чтобы эксперимент был воспроизводимым на обычном компьютере без GPU.
- Качество проверяется через backtesting на последних временных окнах.

## Результаты backtesting

{metrics_md}

## Таблица выбора DL-моделей

{selection_md}

## Выбранная DL-модель

По среднему значению `sMAPE` лучшей DL-моделью стала:

**`{best_model}`**

Метрики лучшей DL-модели:

- MAE: **{best_row['mae']:.4f}**
- RMSE: **{best_row['rmse']:.4f}**
- sMAPE: **{best_row['smape']:.4f}**
- MASE: **{best_row['mase']:.4f}**

## Обоснование параметров

Параметры выбраны как компромисс между качеством и временем локального запуска. Историческое окно `168` часов соответствует недельной сезонности часового ряда. Ограничение числа шагов обучения снижает риск слишком долгого обучения и делает эксперимент воспроизводимым.

## Сохраненные результаты

Таблицы:

- `reports/tables/neural_metrics_by_window.csv`
- `reports/tables/neural_metrics.csv`
- `reports/tables/neural_model_selection.csv`

Прогнозы:

- `outputs/forecasts/neural_predictions.csv`
- `outputs/forecasts/neural_forecast.csv`

Графики:

- `reports/figures/neural_metrics.png`
- `reports/figures/neural_backtest.png`
- `reports/figures/neural_forecast.png`

## Вывод

DL-модели позволяют проверять более сложные нелинейные зависимости временного ряда. В отличие от обычных табличных ML-моделей, модели NeuralForecast напрямую работают с историческим окном временного ряда и обучаются предсказывать будущую траекторию на заданный горизонт.
"""


def combine_ml_and_neural_reports(neural_metrics: pd.DataFrame, neural_selection: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    ml_metrics_path = TABLES_DIR / "machine_learning_metrics.csv"
    ml_selection_path = TABLES_DIR / "machine_learning_model_selection.csv"

    if not ml_metrics_path.exists() or not ml_selection_path.exists():
        raise FileNotFoundError(
            "Не найдены результаты ML-моделей. Сначала запустите: python scripts/machine_learning_forecast.py"
        )

    ml_metrics = pd.read_csv(ml_metrics_path)
    ml_selection = pd.read_csv(ml_selection_path)
    ml_metrics["group"] = "ML"
    neural_metrics = neural_metrics.copy()
    neural_metrics["group"] = "DL"

    combined_metrics = pd.concat([ml_metrics, neural_metrics], ignore_index=True)
    combined_metrics = combined_metrics.sort_values("smape").reset_index(drop=True)
    combined_metrics["rank"] = range(1, len(combined_metrics) + 1)
    combined_metrics = combined_metrics[["rank", "group", "model", "mae", "rmse", "mape", "smape", "mase"]]

    combined_selection = pd.concat([ml_selection, neural_selection], ignore_index=True)
    combined_selection = combined_selection.drop(columns=["rank"], errors="ignore")
    combined_selection = combined_selection.merge(
        combined_metrics[["rank", "group", "model", "mae", "rmse", "smape", "mase"]],
        on=["group", "model"],
        how="right",
        suffixes=("_old", ""),
    )
    for col in ["mae_old", "rmse_old", "smape_old", "mase_old"]:
        if col in combined_selection.columns:
            combined_selection = combined_selection.drop(columns=[col])
    combined_selection["decision"] = "сравнительная модель"
    combined_selection.loc[combined_selection["rank"] == 1, "decision"] = "основной data-driven кандидат"
    combined_selection = combined_selection[["rank", "group", "model", "mae", "rmse", "smape", "mase", "comment", "decision"]]
    best_model = combined_metrics.iloc[0]["model"]
    return combined_metrics, combined_selection, best_model


def plot_combined_results(combined_metrics: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    order = combined_metrics.sort_values("smape")
    labels = [f"{m}\n({g})" for m, g in zip(order["model"], order["group"])]
    plt.bar(labels, order["smape"])
    plt.title("Сравнение ML и DL моделей по sMAPE")
    plt.ylabel("sMAPE")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "data_driven_metrics.png", dpi=160)
    plt.close()


def make_combined_report(combined_metrics: pd.DataFrame, combined_selection: pd.DataFrame, best_model: str) -> str:
    metrics_md = combined_metrics.to_markdown(index=False)
    selection_md = combined_selection.to_markdown(index=False)
    best_row = combined_metrics[combined_metrics["model"] == best_model].iloc[0]

    return f"""# Data-driven модели прогнозирования

## Цель раздела

Цель раздела — сравнить методы машинного обучения и модели глубокого обучения для прогнозирования часового электропотребления. Этот раздел закрывает data-driven часть исследования: feature engineering, ML-модели и DL-модели.

## Feature engineering для ML-моделей

Для ML-моделей сформированы признаки:

- лаги целевой переменной: `1`, `2`, `3`, `6`, `12`, `24`, `48`, `72`, `168`;
- скользящие статистики по окнам `24` и `168` часов;
- календарные признаки: час суток, день недели, месяц, признак выходного дня;
- циклическое кодирование часа суток и дня недели через `sin` и `cos`.

Такая схема соответствует логике `mlforecast`: прогнозирование сводится к задаче табличной регрессии на лаговых, скользящих и календарных признаках.

## Использованные модели

В сравнении участвуют 3 ML-модели и 3 DL-модели.

| Группа | Модель | Параметры и логика |
|---|---|---|
| ML | `ridge_regression` | линейная модель с L2-регуляризацией, `alpha=1.0` |
| ML | `random_forest` | 120 деревьев, ограничение глубины 12 |
| ML | `hist_gradient_boosting` | 250 итераций, learning rate 0.05 |
| DL | `neural_mlp` | MLP-модель NeuralForecast для нелинейного прогноза по историческому окну |
| DL | `nbeats` | N-BEATS, нейросетевая модель с остаточными блоками и декомпозиционной логикой |
| DL | `nhits` | NHITS, иерархическая нейросетевая модель для прогнозирования временных рядов |

## Схема проверки качества

Использован backtesting по временным окнам. В каждом окне модели обучаются только на прошлом и прогнозируют следующие 24 часа. Это исключает перемешивание прошлого и будущего.

Для DL-моделей обучение ограничено последним годом наблюдений, чтобы запуск был реалистичным на обычном компьютере без GPU. Прогнозные окна при этом остаются теми же, что позволяет использовать результаты как практическое сравнение подходов.

## Результаты backtesting

{metrics_md}

## Таблица выбора моделей

{selection_md}

## Выбранная data-driven модель

По среднему значению `sMAPE` лучшей data-driven моделью стала:

**`{best_model}`**

Метрики лучшей модели:

- MAE: **{best_row['mae']:.4f}**
- RMSE: **{best_row['rmse']:.4f}**
- sMAPE: **{best_row['smape']:.4f}**
- MASE: **{best_row['mase']:.4f}**

Выбор сделан по `sMAPE`, но дополнительно учитывались `MAE`, `RMSE` и `MASE`. Это важно, потому что одна относительная метрика может быть нестабильной при низких значениях электропотребления.

## Обоснование параметров

- лаги `24` и `168` отражают суточную и недельную сезонность;
- деревья ограничены по глубине, чтобы снизить переобучение;
- регуляризация используется для повышения устойчивости моделей;
- для NeuralForecast-моделей историческое окно равно `168` часам, то есть одной неделе;
- число шагов обучения DL-моделей ограничено, чтобы эксперимент был воспроизводимым на обычном компьютере.

## Связь с mlforecast и neuralforecast

ML-блок построен по логике `mlforecast`: временной ряд преобразуется в табличные признаки, после чего используются регрессионные модели.

DL-блок реализован через `neuralforecast`: протестированы три разные нейросетевые архитектуры — MLP, N-BEATS и NHITS. Такой набор закрывает требование сравнить несколько методов глубокого обучения для временного ряда.

## Сохраненные результаты

ML-таблицы:

- `reports/tables/machine_learning_metrics_by_window.csv`
- `reports/tables/machine_learning_metrics.csv`
- `reports/tables/machine_learning_model_selection.csv`

DL-таблицы:

- `reports/tables/neural_metrics_by_window.csv`
- `reports/tables/neural_metrics.csv`
- `reports/tables/neural_model_selection.csv`

Итоговые data-driven таблицы:

- `reports/tables/data_driven_metrics.csv`
- `reports/tables/data_driven_model_selection.csv`

Прогнозы:

- `outputs/forecasts/machine_learning_predictions.csv`
- `outputs/forecasts/machine_learning_forecast.csv`
- `outputs/forecasts/neural_predictions.csv`
- `outputs/forecasts/neural_forecast.csv`

Графики:

- `reports/figures/machine_learning_metrics.png`
- `reports/figures/machine_learning_backtest.png`
- `reports/figures/machine_learning_forecast.png`
- `reports/figures/neural_metrics.png`
- `reports/figures/neural_backtest.png`
- `reports/figures/neural_forecast.png`
- `reports/figures/data_driven_metrics.png`

## Вывод

Data-driven модели позволяют использовать больше информации о временном ряде, чем классические статистические методы: лаги, календарные признаки, скользящие статистики и нелинейные зависимости. ML-модели дают интерпретируемый табличный подход, а DL-модели NeuralForecast проверяют более сложные нейросетевые архитектуры. Результаты этого раздела используются в итоговой таблице выбора методов вместе со статистическими моделями.
"""


def run_neural_analysis() -> None:
    print("[1/8] Загрузка подготовленного ряда...")
    df = load_series()
    print(f"Наблюдений: {len(df)}")
    print(f"Период: {df['ds'].min()} — {df['ds'].max()}")

    config = NeuralConfig()

    print("[2/8] Backtesting DL-моделей NeuralForecast...")
    predictions, insample_by_window, _ = run_backtesting(df, config)

    print("[3/8] Расчет метрик DL-моделей...")
    metrics_by_window, metrics_summary = summarize_metrics(
        predictions,
        insample_by_window=insample_by_window,
        season_length=SEASON_LENGTH_DAY,
    )

    print("[4/8] Финальный прогноз DL-моделей на 24 часа...")
    forecasts = train_final_forecasts(df, config)

    print("[5/8] Таблица выбора DL-моделей...")
    selection = make_selection_table(metrics_summary)
    best_model = metrics_summary.iloc[0]["model"]
    print(f"Лучшая DL-модель по sMAPE: {best_model}")

    print("[6/8] Сохранение DL-таблиц и графиков...")
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FORECASTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_by_window.to_csv(TABLES_DIR / "neural_metrics_by_window.csv", index=False)
    metrics_summary.to_csv(TABLES_DIR / "neural_metrics.csv", index=False)
    selection.to_csv(TABLES_DIR / "neural_model_selection.csv", index=False)
    predictions.to_csv(FORECASTS_DIR / "neural_predictions.csv", index=False)
    forecasts.to_csv(FORECASTS_DIR / "neural_forecast.csv", index=False)
    plot_neural_results(metrics_summary, predictions, forecasts)

    print("[7/8] Создание отчета по DL-моделям...")
    neural_report = make_neural_report(metrics_summary, selection, best_model, config)
    NEURAL_REPORT_PATH.write_text(neural_report, encoding="utf-8")

    print("[8/8] Создание общего data-driven отчета...")
    combined_metrics, combined_selection, combined_best_model = combine_ml_and_neural_reports(metrics_summary, selection)
    combined_metrics.to_csv(TABLES_DIR / "data_driven_metrics.csv", index=False)
    combined_selection.to_csv(TABLES_DIR / "data_driven_model_selection.csv", index=False)
    plot_combined_results(combined_metrics)
    combined_report = make_combined_report(combined_metrics, combined_selection, combined_best_model)
    DATA_DRIVEN_REPORT_PATH.write_text(combined_report, encoding="utf-8")

    print("Готово.")
    print(f"DL-отчет: {NEURAL_REPORT_PATH}")
    print(f"Общий data-driven отчет: {DATA_DRIVEN_REPORT_PATH}")
    print(f"Таблицы: {TABLES_DIR}")
    print(f"Графики: {FIGURES_DIR}")
    print(f"Прогнозы: {FORECASTS_DIR}")


if __name__ == "__main__":
    run_neural_analysis()
