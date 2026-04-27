"""Опциональный полный запуск проекта.

Скрипт запускает основные этапы исследования по порядку.
DL-блок может выполняться дольше остальных этапов.
"""

from pathlib import Path
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parents[1]

COMMANDS = [
    ["python", "scripts/prepare_eda.py"],
    ["python", "scripts/statistical_forecast.py"],
    ["python", "scripts/analyze_anomalies.py"],
    ["python", "scripts/machine_learning_forecast.py"],
    ["python", "scripts/neural_forecast.py"],
    ["python", "scripts/run_pipeline.py"],
]


def main() -> None:
    for command in COMMANDS:
        print("\n>>>", " ".join(command))
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
