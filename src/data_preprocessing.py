"""Загрузка, проверка и подготовка временного ряда.

В проекте основной датасет — UCI Individual Household Electric Power Consumption.
Исходный файл `household_power_consumption.txt` имеет разделитель `;`, отдельные
колонки `Date` и `Time`, а пропуски обозначены символом `?`.

Основной рабочий формат после подготовки:
    unique_id, ds, y
где `ds` — временная метка, `y` — целевая переменная.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

RAW_NA_VALUES = ["?", "", "NA", "NaN", "nan", "null", "None"]
HOUSEHOLD_TARGET = "Global_active_power"


def load_time_series(path: str | Path, date_col: str = "ds", target_col: str = "y") -> pd.DataFrame:
    """Загружает простой CSV с колонками даты и целевой переменной.

    Эта функция оставлена для совместимости с синтетическим примером и smoke-test.
    Для финального датасета UCI используйте `load_household_power_consumption`.
    """
    path = Path(path)
    df = pd.read_csv(path)

    if date_col not in df.columns:
        raise ValueError(f"В данных нет колонки даты: {date_col}")
    if target_col not in df.columns:
        raise ValueError(f"В данных нет целевой колонки: {target_col}")

    df = df[[date_col, target_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.sort_values(date_col).drop_duplicates(subset=[date_col])
    return df


def load_household_power_consumption(path: str | Path) -> pd.DataFrame:
    """Загружает исходный UCI-файл `household_power_consumption.txt`.

    Возвращает DataFrame с колонкой `ds` и всеми числовыми признаками датасета.
    Дату парсим с `dayfirst=True`, потому что исходный формат — dd/mm/yyyy.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Файл {path} не найден. Скачайте household_power_consumption.txt "
            "с UCI и положите его в data/raw/."
        )

    df = pd.read_csv(
        path,
        sep=";",
        na_values=RAW_NA_VALUES,
        low_memory=False,
    )

    required = {"Date", "Time", HOUSEHOLD_TARGET}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"В исходном датасете отсутствуют обязательные колонки: {sorted(missing)}")

    df["ds"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
        dayfirst=True,
    )

    numeric_cols = [col for col in df.columns if col not in {"Date", "Time", "ds"}]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.drop(columns=["Date", "Time"]).sort_values("ds")
    df = df.drop_duplicates(subset=["ds"], keep="first").reset_index(drop=True)
    return df


def basic_quality_report(df: pd.DataFrame, date_col: str = "ds", target_col: str = "y") -> dict:
    """Возвращает базовый отчет о качестве данных."""
    report: dict[str, object] = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
    }

    if date_col in df.columns:
        report.update(
            {
                "date_min": df[date_col].min(),
                "date_max": df[date_col].max(),
                "missing_dates": int(df[date_col].isna().sum()),
                "duplicate_dates": int(df.duplicated(subset=[date_col]).sum()),
            }
        )

        sorted_dates = df[date_col].dropna().sort_values()
        if len(sorted_dates) > 1:
            diffs = sorted_dates.diff().dropna()
            report["most_common_step"] = str(diffs.mode().iloc[0]) if not diffs.empty else None
            report["min_step"] = str(diffs.min()) if not diffs.empty else None
            report["max_step"] = str(diffs.max()) if not diffs.empty else None

    if target_col in df.columns:
        report.update(
            {
                "missing_target": int(df[target_col].isna().sum()),
                "target_min": float(df[target_col].min()) if df[target_col].notna().any() else None,
                "target_max": float(df[target_col].max()) if df[target_col].notna().any() else None,
                "target_mean": float(df[target_col].mean()) if df[target_col].notna().any() else None,
                "target_median": float(df[target_col].median()) if df[target_col].notna().any() else None,
            }
        )

    return report


def missing_values_table(df: pd.DataFrame) -> pd.DataFrame:
    """Таблица пропусков по всем колонкам."""
    out = pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": [int(df[col].isna().sum()) for col in df.columns],
            "missing_share": [float(df[col].isna().mean()) for col in df.columns],
            "dtype": [str(df[col].dtype) for col in df.columns],
        }
    )
    return out.sort_values("missing_share", ascending=False).reset_index(drop=True)


def aggregate_to_hourly(
    df: pd.DataFrame,
    date_col: str = "ds",
    target_col: str = HOUSEHOLD_TARGET,
    agg: Literal["mean", "sum"] = "mean",
) -> pd.DataFrame:
    """Агрегирует минутный ряд до часового.

    Для `Global_active_power` используем среднее за час: это сохраняет единицу kW.
    """
    if date_col not in df.columns:
        raise ValueError(f"Нет колонки времени {date_col}")
    if target_col not in df.columns:
        raise ValueError(f"Нет целевой колонки {target_col}")

    data = df[[date_col, target_col]].copy()
    data = data.dropna(subset=[date_col]).set_index(date_col).sort_index()

    if agg == "mean":
        hourly = data[target_col].resample("h").mean()
    elif agg == "sum":
        hourly = data[target_col].resample("h").sum(min_count=1)
    else:
        raise ValueError("agg должен быть 'mean' или 'sum'")

    return hourly.rename("y").reset_index().rename(columns={date_col: "ds"})


def make_regular_time_index(
    df: pd.DataFrame,
    freq: str,
    date_col: str = "ds",
    target_col: str = "y",
    fill_method: str = "interpolate",
) -> pd.DataFrame:
    """Приводит ряд к регулярной сетке времени и заполняет пропуски.

    fill_method:
    - "interpolate" — линейная интерполяция;
    - "time" — интерполяция с учетом временного индекса;
    - "ffill" — перенос последнего известного значения;
    - "none" — не заполнять пропуски.
    """
    regular = df[[date_col, target_col]].copy()
    regular[date_col] = pd.to_datetime(regular[date_col], errors="coerce")
    regular[target_col] = pd.to_numeric(regular[target_col], errors="coerce")
    regular = regular.dropna(subset=[date_col]).sort_values(date_col).drop_duplicates(subset=[date_col])
    regular = regular.set_index(date_col).asfreq(freq).rename_axis(date_col)

    if fill_method == "interpolate":
        regular[target_col] = regular[target_col].interpolate(method="linear", limit_direction="both")
    elif fill_method == "time":
        regular[target_col] = regular[target_col].interpolate(method="time", limit_direction="both")
    elif fill_method == "ffill":
        regular[target_col] = regular[target_col].ffill().bfill()
    elif fill_method == "none":
        pass
    else:
        raise ValueError("fill_method должен быть одним из: interpolate, time, ffill, none")

    return regular.reset_index()


def to_nixtla_format(
    df: pd.DataFrame,
    date_col: str = "ds",
    target_col: str = "y",
    unique_id: str = "series_1",
) -> pd.DataFrame:
    """Форматирует данные под statsforecast/mlforecast/neuralforecast."""
    out = df[[date_col, target_col]].copy()
    out.insert(0, "unique_id", unique_id)
    out = out.rename(columns={date_col: "ds", target_col: "y"})
    return out[["unique_id", "ds", "y"]]


def prepare_household_hourly_dataset(
    raw_path: str | Path,
    processed_path: str | Path,
    freq: str = "h",
    unique_id: str = "household_1",
    fill_method: str = "time",
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Полная подготовка датасета UCI к рабочему часовому формату.

    Возвращает:
    - подготовленный ряд в формате unique_id, ds, y;
    - словарь с отчетами качества;
    - таблицу пропусков в исходных данных.
    """
    raw = load_household_power_consumption(raw_path)
    raw_quality = basic_quality_report(raw, date_col="ds", target_col=HOUSEHOLD_TARGET)
    missing_table = missing_values_table(raw)

    hourly = aggregate_to_hourly(raw, date_col="ds", target_col=HOUSEHOLD_TARGET, agg="mean")
    hourly_regular = make_regular_time_index(
        hourly,
        freq=freq,
        date_col="ds",
        target_col="y",
        fill_method=fill_method,
    )
    prepared = to_nixtla_format(hourly_regular, date_col="ds", target_col="y", unique_id=unique_id)

    processed_path = Path(processed_path)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(processed_path, index=False)

    quality = {
        "raw": raw_quality,
        "hourly_before_fill": basic_quality_report(hourly, date_col="ds", target_col="y"),
        "prepared": basic_quality_report(prepared, date_col="ds", target_col="y"),
    }
    return prepared, quality, missing_table


def train_test_split_ts(df: pd.DataFrame, test_size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Разбивает временной ряд на train/test без перемешивания."""
    if test_size <= 0:
        raise ValueError("test_size должен быть положительным")
    if test_size >= len(df):
        raise ValueError("test_size должен быть меньше длины ряда")
    return df.iloc[:-test_size].copy(), df.iloc[-test_size:].copy()
