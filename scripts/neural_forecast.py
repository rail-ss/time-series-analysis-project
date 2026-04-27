import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.neural_models import run_neural_analysis


if __name__ == "__main__":
    run_neural_analysis()
