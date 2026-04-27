"""Базовая smoke-проверка файлов проекта."""

from pathlib import Path


def test_core_files_exist():
    root = Path(__file__).resolve().parents[1]
    assert (root / "scripts" / "run_pipeline.py").exists()
    assert (root / "src" / "final_pipeline.py").exists()
    assert (root / "requirements.txt").exists()
