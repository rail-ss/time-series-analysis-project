"""EDA временного ряда: таблицы, графики, сезонность, стационарность, ACF/PACF."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def ensure_parent_dir(save_path: str | Path | None) -> None:
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)


def descriptive_stats(df: pd.DataFrame, target_col: str = "y") -> pd.DataFrame:
    """Расширенная описательная статистика целевого ряда."""
    s = pd.to_numeric(df[target_col], errors="coerce")
    stats = s.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_frame("value")
    extra = pd.DataFrame(
        {
            "value": {
                "missing_count": int(s.isna().sum()),
                "missing_share": float(s.isna().mean()),
                "zero_count": int((s == 0).sum()),
                "zero_share": float((s == 0).mean()),
                "skew": float(s.skew()),
                "kurtosis": float(s.kurtosis()),
            }
        }
    )
    return pd.concat([stats, extra])


def save_dict_as_csv(data: dict, path: str | Path) -> pd.DataFrame:
    """Сохраняет вложенный словарь отчетов в табличном виде."""
    rows = []
    for block, values in data.items():
        if isinstance(values, dict):
            for metric, value in values.items():
                rows.append({"block": block, "metric": metric, "value": value})
        else:
            rows.append({"block": "main", "metric": block, "value": values})
    out = pd.DataFrame(rows)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


def plot_series(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    title: str = "Временной ряд",
    save_path: str | Path | None = None,
):
    ensure_parent_dir(save_path)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df[date_col], df[target_col], linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Дата")
    ax.set_ylabel(target_col)
    ax.grid(True)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_last_window(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    periods: int,
    title: str,
    save_path: str | Path | None = None,
):
    """График последних N наблюдений."""
    data = df.tail(periods).copy()
    return plot_series(data, date_col, target_col, title=title, save_path=save_path)


def plot_rolling_stats(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    windows: list[int] | tuple[int, ...],
    save_path: str | Path | None = None,
):
    """Строит исходный ряд и несколько скользящих средних."""
    ensure_parent_dir(save_path)
    data = df[[date_col, target_col]].copy()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(data[date_col], data[target_col], label="Исходный ряд", linewidth=0.8, alpha=0.5)

    for window in windows:
        data[f"rolling_mean_{window}"] = data[target_col].rolling(window, min_periods=max(2, window // 3)).mean()
        ax.plot(data[date_col], data[f"rolling_mean_{window}"], label=f"Скользящее среднее, window={window}")

    ax.set_title("Скользящие средние")
    ax.set_xlabel("Дата")
    ax.set_ylabel(target_col)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def seasonal_profiles(df: pd.DataFrame, date_col: str = "ds", target_col: str = "y") -> dict[str, pd.DataFrame]:
    """Средние профили по часу суток, дню недели и месяцу."""
    data = df[[date_col, target_col]].copy()
    data[date_col] = pd.to_datetime(data[date_col])
    data["hour"] = data[date_col].dt.hour
    data["day_of_week"] = data[date_col].dt.dayofweek
    data["month"] = data[date_col].dt.month

    return {
        "hourly_profile": data.groupby("hour", as_index=False)[target_col].mean(),
        "weekly_profile": data.groupby("day_of_week", as_index=False)[target_col].mean(),
        "monthly_profile": data.groupby("month", as_index=False)[target_col].mean(),
    }


def plot_profile(profile: pd.DataFrame, x_col: str, y_col: str = "y", title: str = "Профиль", save_path: str | Path | None = None):
    ensure_parent_dir(save_path)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(profile[x_col], profile[y_col], marker="o")
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(f"Среднее {y_col}")
    ax.grid(True)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def decompose_series(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    period: int,
    model: str = "additive",
    max_points: int | None = None,
    save_path: str | Path | None = None,
):
    """STL-like классическая декомпозиция ряда на trend/seasonal/resid.

    Для читаемости графика можно передать `max_points`, например 24 * 90.
    """
    ensure_parent_dir(save_path)
    data = df[[date_col, target_col]].dropna().copy()
    if max_points is not None and len(data) > max_points:
        data = data.tail(max_points)
    series = data.set_index(date_col)[target_col]
    result = seasonal_decompose(series, model=model, period=period, extrapolate_trend="freq")
    fig = result.plot()
    fig.set_size_inches(12, 8)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return result, fig


def stationarity_tests(df: pd.DataFrame, target_col: str = "y", max_points: int | None = 10000) -> dict:
    """ADF и KPSS тесты.

    Интерпретация:
    - ADF: p-value < 0.05 говорит в пользу стационарности.
    - KPSS: p-value > 0.05 говорит в пользу стационарности.

    Для ускорения можно использовать последние `max_points` наблюдений.
    """
    series = pd.to_numeric(df[target_col], errors="coerce").dropna()
    if max_points is not None and len(series) > max_points:
        series = series.tail(max_points)

    adf_stat, adf_pvalue, *_ = adfuller(series, autolag="AIC")

    try:
        kpss_stat, kpss_pvalue, *_ = kpss(series, regression="c", nlags="auto")
    except Exception:
        kpss_stat, kpss_pvalue = None, None

    return {
        "adf_statistic": float(adf_stat),
        "adf_pvalue": float(adf_pvalue),
        "kpss_statistic": None if kpss_stat is None else float(kpss_stat),
        "kpss_pvalue": None if kpss_pvalue is None else float(kpss_pvalue),
        "used_observations": int(len(series)),
    }


def plot_acf_pacf_charts(
    df: pd.DataFrame,
    target_col: str,
    lags: int = 72,
    max_points: int | None = 10000,
    save_prefix: str | Path | None = None,
):
    series = pd.to_numeric(df[target_col], errors="coerce").dropna()
    if max_points is not None and len(series) > max_points:
        series = series.tail(max_points)

    fig_acf = plot_acf(series, lags=lags).figure
    fig_acf.set_size_inches(10, 4)
    fig_acf.tight_layout()

    fig_pacf = plot_pacf(series, lags=lags, method="ywm").figure
    fig_pacf.set_size_inches(10, 4)
    fig_pacf.tight_layout()

    if save_prefix:
        save_prefix = Path(save_prefix)
        save_prefix.parent.mkdir(parents=True, exist_ok=True)
        fig_acf.savefig(save_prefix.with_name(save_prefix.name + "_acf.png"), dpi=150)
        fig_pacf.savefig(save_prefix.with_name(save_prefix.name + "_pacf.png"), dpi=150)

    return fig_acf, fig_pacf


def write_eda_markdown_summary(
    path: str | Path,
    prepared_df: pd.DataFrame,
    quality_table_path: str,
    descriptive_stats_path: str,
    stationarity: dict,
) -> None:
    """Создает краткий markdown-отчет по EDA после локального запуска."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ds_min = prepared_df["ds"].min()
    ds_max = prepared_df["ds"].max()
    rows = len(prepared_df)
    missing_y = int(prepared_df["y"].isna().sum())

    adf_p = stationarity.get("adf_pvalue")
    kpss_p = stationarity.get("kpss_pvalue")

    text = f"""# Результаты подготовки данных и EDA

## Подготовленный ряд

- Количество часовых наблюдений: **{rows}**
- Начало ряда: **{ds_min}**
- Конец ряда: **{ds_max}**
- Пропуски в целевой переменной после подготовки: **{missing_y}**
- Рабочая частота: **1 час**
- Целевая переменная: **y = среднее часовое значение Global_active_power**

## Что сделано

1. Исходные минутные данные загружены из `data/raw/household_power_consumption.txt`.
2. Колонки `Date` и `Time` объединены в единую временную метку `ds`.
3. Символы пропусков `?` преобразованы в `NaN`.
4. Числовые признаки приведены к числовому типу.
5. Минутный ряд агрегирован до часового ряда по среднему значению `Global_active_power`.
6. Ряд приведен к регулярной часовой сетке.
7. Пропуски после агрегации заполнены интерполяцией по времени.
8. Подготовленный ряд сохранен в `data/processed/prepared_time_series.csv`.

## Таблицы

- Отчет о качестве данных: `{quality_table_path}`
- Описательная статистика целевого ряда: `{descriptive_stats_path}`
- Результаты тестов стационарности: `reports/tables/stationarity_tests.csv`

## Проверка стационарности

Для проверки стационарности были использованы тесты ADF и KPSS.

- ADF p-value: **{adf_p}**
- KPSS p-value: **{kpss_p}**

Тест ADF проверяет нулевую гипотезу о наличии единичного корня, то есть о нестационарности ряда. Если p-value ADF меньше 0.05, это является аргументом в пользу стационарности ряда.

Тест KPSS, наоборот, проверяет нулевую гипотезу о стационарности ряда. Если p-value KPSS меньше 0.05, нулевая гипотеза стационарности отвергается.

Если ADF и KPSS дают разные выводы, ряд нельзя считать уверенно стационарным без дополнительной обработки. Для электропотребления такое поведение ожидаемо, поскольку ряд содержит выраженную внутридневную сезонность, возможные недельные эффекты, изменения уровня потребления и отдельные выбросы.

В дальнейших моделях необходимо учитывать сезонность. Для статистических моделей целесообразно использовать сезонный период **24** для часовых данных, а также рассмотреть недельный период **168**. Для ARIMA-подобных моделей может потребоваться сезонное дифференцирование или автоматический подбор параметров.

## Графики

Графики сохранены в папку `reports/figures/`:

- `full_series.png` — общий график подготовленного ряда;
- `last_60_days.png` — последние 60 дней;
- `rolling_means.png` — скользящие средние;
- `hourly_profile.png` — средний профиль по часу суток;
- `weekly_profile.png` — средний профиль по дню недели;
- `monthly_profile.png` — средний профиль по месяцу;
- `decomposition_last_90_days.png` — декомпозиция последних 90 дней;
- `autocorrelation_acf.png` и `autocorrelation_pacf.png` — ACF/PACF.

## Итоговый вывод по EDA

Временной ряд электропотребления приведен к регулярной часовой частоте и пригоден для дальнейшего анализа. Для ряда ожидаема внутридневная сезонность, поэтому в следующих этапах для baseline и статистических моделей необходимо учитывать сезонный период `24`, а для расширенного анализа — недельный период `168`.

Следующий этап исследования — анализ аномалий временного ряда с обоснованным выбором метода и параметров.
"""
    path.write_text(text, encoding="utf-8")
