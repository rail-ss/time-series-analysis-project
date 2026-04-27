
"""Минимальный smoke-test пайплайна.

Запуск:
pytest tests/test_pipeline_smoke.py
"""

from src.pipeline import run_baseline_pipeline


def test_baseline_pipeline_runs():
    metrics = run_baseline_pipeline()
    assert not metrics.empty
    assert "MAE" in metrics.columns
    assert "RMSE" in metrics.columns
