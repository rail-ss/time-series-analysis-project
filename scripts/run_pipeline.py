"""Запуск итогового пайплайна прогнозирования."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.final_pipeline import run_pipeline  # noqa: E402


if __name__ == "__main__":
    run_pipeline()
