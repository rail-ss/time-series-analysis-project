# Отчет об исследовании временного ряда электропотребления

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
