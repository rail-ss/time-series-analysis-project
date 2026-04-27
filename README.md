# Анализ временного ряда электропотребления

Проект по дисциплине «Анализ временных рядов».

Цель проекта — подготовить временной ряд электропотребления, провести EDA, исследовать аномалии, сравнить статистические, ML- и DL-модели прогнозирования, а затем собрать итоговый пайплайн прогноза на 24 часа.

## Данные

Используется датасет **Individual Household Electric Power Consumption**.

Официальный источник данных:  
https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

Дополнительная ссылка на исходный файл в облачном хранилище:  
https://drive.google.com/file/d/1huxQbuI0wZFlrCssrp3KM6mY3CMDV9ZL/view?usp=sharing

Исходный файл `household_power_consumption.txt` имеет большой размер, поэтому он не добавляется в Git-репозиторий. Для воспроизведения проекта его нужно скачать и положить вручную:

```text
data/raw/household_power_consumption.txt
```

Подготовленный ряд сохраняется сюда:

```text
data/processed/prepared_time_series.csv
```

## Структура проекта

```text
data/
  raw/
  processed/
notebooks/
  main_pipeline.ipynb
outputs/
  forecasts/
reports/
  figures/
  tables/
src/
scripts/
tests/
README.md
requirements.txt
TASK_CHECKLIST.md
```

## Основные разделы отчета

- `reports/dataset_and_task.md` — описание временного ряда и постановка задачи;
- `reports/data_preparation_and_eda.md` — подготовка данных и EDA;
- `reports/eda_results.md` — результаты EDA;
- `reports/anomaly_detection.md` — анализ аномалий;
- `reports/statistical_models.md` — статистические модели;
- `reports/machine_learning_models.md` — ML-модели;
- `reports/neural_models.md` — DL-модели;
- `reports/data_driven_models.md` — общее сравнение ML и DL;
- `reports/pipeline.md` — итоговый пайплайн;
- `reports/REPORT.md` — общий отчет.

## Окружение

Рекомендуется Python 3.11.

```bash
py -3.11 -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

## Запуск по этапам

Подготовка данных и EDA:

```bash
python scripts/prepare_eda.py
```

Статистические модели:

```bash
python scripts/statistical_forecast.py
```

Анализ аномалий:

```bash
python scripts/analyze_anomalies.py
```

ML-модели:

```bash
python scripts/machine_learning_forecast.py
```

DL-модели:

```bash
python scripts/neural_forecast.py
```

Итоговый пайплайн:

```bash
python scripts/run_pipeline.py
```

Полный запуск всех этапов:

```bash
python scripts/run_all.py
```

## Постановка задачи

- Целевая переменная: `y`, среднее часовое значение `Global_active_power`.
- Частота ряда: 1 час.
- Горизонт прогноза: 24 часа.
- Тип задачи: offline-прогнозирование.
- Основные метрики: MAE, RMSE, sMAPE, MASE.

## Итоговая логика

Финальный пайплайн собирает результаты всех предыдущих этапов, формирует общую таблицу сравнения моделей, дополнительно проверяет устойчивость финальных кандидатов на последних временных окнах, строит прогноз на 24 часа и сохраняет результаты статистического тестирования и тестирования производительности.
