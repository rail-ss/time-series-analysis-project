# Чек-лист итогового задания

Документ показывает, как материалы проекта закрывают требования итогового задания по дисциплине **«Анализ временных рядов»**.

---

## 1. Репозиторий и структура

| Требование | Статус | Где находится |
|---|---|---|
| Git-репозиторий создан | ✅ выполнено | GitHub-репозиторий проекта |
| Репозиторий доступен преподавателю | ✅ выполнено | публичный репозиторий |
| Есть структура проекта | ✅ выполнено | `data/`, `src/`, `scripts/`, `notebooks/`, `reports/`, `outputs/`, `tests/` |
| Есть данные или ссылка на данные | ✅ выполнено | `README.md`, `data/raw/README_data.md` |
| Есть подготовленная тестовая/рабочая выборка ВР | ✅ выполнено | `data/processed/prepared_time_series.csv` |
| Есть Jupyter Notebook с кодом и комментариями | ✅ выполнено | `notebooks/main_pipeline.ipynb` |
| Есть код пайплайна | ✅ выполнено | `scripts/run_pipeline.py`, `src/` |
| Есть отчет в Markdown | ✅ выполнено | `README.md`, `reports/REPORT.md` |

---

## 2. Задание №1. Подготовка данных и EDA

| Требование | Статус | Где находится |
|---|---|---|
| Выбран датасет временного ряда | ✅ выполнено | `README.md`, `reports/dataset_and_task.md` |
| Описана предметная задача | ✅ выполнено | `README.md`, `reports/dataset_and_task.md` |
| Определены горизонт прогноза, режим, метрики | ✅ выполнено | `README.md`, `reports/dataset_and_task.md` |
| Данные загружены в pandas | ✅ выполнено | `scripts/prepare_eda.py`, `src/` |
| Обработаны временные метки | ✅ выполнено | `scripts/prepare_eda.py`, `src/` |
| Проверены пропуски и типы данных | ✅ выполнено | `reports/tables/quality_report.csv`, `reports/tables/missing_values_raw.csv` |
| Ряд агрегирован до часовой частоты | ✅ выполнено | `data/processed/prepared_time_series.csv` |
| Проведен EDA | ✅ выполнено | `reports/eda_results.md`, `reports/figures/` |
| Выполнена проверка стационарности | ✅ выполнено | `reports/tables/stationarity_tests.csv`, `reports/eda_results.md` |
| Сохранены графики EDA | ✅ выполнено | `reports/figures/full_series.png`, `last_60_days.png`, `hourly_profile.png`, `weekly_profile.png`, `autocorrelation_acf.png`, `autocorrelation_pacf.png` |
| Сформулирован вывод по EDA | ✅ выполнено | `README.md`, `reports/REPORT.md`, `reports/eda_results.md` |

**Краткий результат:** подготовлено 34 589 часовых наблюдений без пропусков в целевой переменной. EDA показал суточную и недельную сезонность.

---

## 3. Задание №2. Статистические методы прогнозирования

| Требование | Статус | Где находится |
|---|---|---|
| Использованы baseline-модели | ✅ выполнено | `reports/statistical_models.md`, `reports/tables/statistical_metrics.csv` |
| Использовано не менее 5 статистических методов | ✅ выполнено | `reports/statistical_models.md`, `reports/tables/statistical_metrics.csv` |
| Есть ручные модели | ✅ выполнено | `manual_ets_additive`, `manual_theta_24`, `manual_arima` |
| Есть автоматические модели | ✅ выполнено | `auto_arima`, `auto_ets`, `auto_theta` |
| Использован `statsforecast` | ✅ выполнено | `reports/statistical_models.md`, код статистических моделей |
| Проведен backtesting | ✅ выполнено | `reports/tables/statistical_metrics_by_window.csv`, `reports/figures/statistical_backtest.png` |
| Выполнен анализ остатков | ✅ выполнено | `reports/tables/residual_diagnostics.csv`, `reports/figures/statistical_residuals.png` |
| Сформированы численные метрики | ✅ выполнено | `reports/tables/statistical_metrics.csv` |
| Сохранены графики | ✅ выполнено | `reports/figures/statistical_metrics.png`, `statistical_forecast.png`, `statistical_residuals_acf.png` |
| Есть вывод по статистическим моделям | ✅ выполнено | `README.md`, `reports/REPORT.md`, `reports/statistical_models.md` |

**Краткий результат:** лучшая статистическая модель — `seasonal_window_average`, `sMAPE = 32.4678`.

---

## 4. Задание №3. Аномалии, ML и DL

### 4.1. Аномалии

| Требование | Статус | Где находится |
|---|---|---|
| Сравнены 3 метода выявления аномалий | ✅ выполнено | `reports/anomaly_detection.md`, `reports/tables/anomaly_summary.csv` |
| Обоснован выбор метода | ✅ выполнено | `reports/anomaly_detection.md` |
| Обоснованы параметры метода | ✅ выполнено | `reports/anomaly_detection.md` |
| Сохранены графики аномалий | ✅ выполнено | `reports/figures/anomalies_full_series.png`, `anomalies_last_90_days.png`, `anomaly_methods_comparison.png` |
| Есть вывод по аномалиям | ✅ выполнено | `README.md`, `reports/REPORT.md`, `reports/anomaly_detection.md` |

**Краткий результат:** выбран `Seasonal IQR`; найдено 363 аномальных наблюдения, 1.049% ряда.

### 4.2. ML-модели

| Требование | Статус | Где находится |
|---|---|---|
| Сравнены 3 ML-модели | ✅ выполнено | `ridge_regression`, `random_forest`, `hist_gradient_boosting` |
| Описан feature engineering | ✅ выполнено | `reports/machine_learning_models.md`, `reports/data_driven_models.md` |
| Сформированы численные метрики | ✅ выполнено | `reports/tables/machine_learning_metrics.csv` |
| Сохранены графики | ✅ выполнено | `reports/figures/machine_learning_metrics.png`, `machine_learning_backtest.png`, `machine_learning_forecast.png` |
| Есть вывод по ML | ✅ выполнено | `README.md`, `reports/REPORT.md`, `reports/machine_learning_models.md` |

**Краткий результат:** лучшая ML-модель — `random_forest`, `sMAPE = 35.8655`.

### 4.3. DL-модели

| Требование | Статус | Где находится |
|---|---|---|
| Сравнены 3 DL-модели | ✅ выполнено | `neural_mlp`, `nbeats`, `nhits` |
| Использован подход NeuralForecast / DL-модели | ✅ выполнено | `reports/neural_models.md` |
| Сформированы численные метрики | ✅ выполнено | `reports/tables/neural_metrics.csv` |
| Сохранены графики | ✅ выполнено | `reports/figures/neural_metrics.png`, `neural_backtest.png`, `neural_forecast.png` |
| Есть вывод по DL | ✅ выполнено | `README.md`, `reports/REPORT.md`, `reports/neural_models.md` |

**Краткий результат:** лучшая DL-модель — `nbeats`, `sMAPE = 35.7025`.

---

## 5. Таблица выбора методов анализа ВР

| Требование | Статус | Где находится |
|---|---|---|
| Есть baseline | ✅ выполнено | `naive`, `seasonal_naive_24`, `seasonal_naive_168` |
| Есть минимум 5 статистических методов | ✅ выполнено | 12 статистических/baseline моделей |
| Есть минимум 3 ML-метода | ✅ выполнено | 3 ML-модели |
| Есть минимум 3 DL-метода | ✅ выполнено | 3 DL-модели |
| Есть комментарии и выбор модели | ✅ выполнено | `README.md`, `reports/REPORT.md`, `reports/statistical_models.md`, `reports/data_driven_models.md` |
| Есть общая таблица результатов | ✅ выполнено | `reports/tables/model_comparison.csv` |
| Есть визуализация общей таблицы | ✅ выполнено | `reports/figures/model_comparison.png` |

**Краткий результат:** по общей таблице лучшей моделью стала `seasonal_window_average`, `sMAPE = 32.4678`.

---

## 6. Задание №4. Итоговый пайплайн

| Требование | Статус | Где находится |
|---|---|---|
| Подготовлен пайплайн решения задачи | ✅ выполнено | `scripts/run_pipeline.py`, `src/` |
| Описана логика пайплайна | ✅ выполнено | `README.md`, `reports/REPORT.md`, `reports/pipeline.md` |
| Проведены проверки входных данных | ✅ выполнено | `reports/tables/pipeline_tests.csv` |
| Проведено статистическое тестирование | ✅ выполнено | `reports/tables/pipeline_statistical_tests.csv` |
| Проведено тестирование производительности | ✅ выполнено | `reports/tables/pipeline_performance.csv` |
| Сохранены прогнозы | ✅ выполнено | `outputs/forecasts/pipeline_forecast.csv` |
| Сохранены графики пайплайна | ✅ выполнено | `reports/figures/pipeline_backtest.png`, `pipeline_forecast.png`, `pipeline_residuals.png` |
| Есть вывод по пайплайну | ✅ выполнено | `README.md`, `reports/REPORT.md`, `reports/pipeline.md` |

**Краткий результат:** пайплайн выполняется примерно за 2.85 секунды, проходит проверки данных и формирует прогноз на 24 часа.

---

## 7. Что было дополнительно улучшено после замечания преподавателя

| Замечание | Что исправлено |
|---|---|
| В README отсутствовали результаты 4 заданий | В `README.md` добавлены отдельные разделы по заданиям №1–№4 |
| Не было выводов по заданиям | В каждый раздел добавлены выводы |
| Не было общего вывода по работе | Добавлен раздел «Общие выводы по работе» |
| Не были видны визуализации моделей | В README и REPORT добавлены ссылки на графики из `reports/figures/` |
| Не были видны численные метрики | В README и REPORT добавлены таблицы метрик по статистическим, ML, DL и итоговым моделям |
| Общий отчет был слишком кратким | `reports/REPORT.md` переписан как полноценный итоговый отчет |

---

## 8. Итоговый статус

Проект закрывает требования итогового задания: есть Git-репозиторий, данные/ссылка на данные, подготовленный временной ряд, Jupyter Notebook, код пайплайна, markdown-отчет, EDA, анализ аномалий, baseline, статистические модели, ML-модели, DL-модели, таблица выбора методов, визуализации, численные метрики и тестирование пайплайна.
