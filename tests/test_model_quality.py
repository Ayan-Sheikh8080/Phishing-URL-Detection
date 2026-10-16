import json
from pathlib import Path

MIN_F1 = 0.90


def test_model_meets_minimum_f1():
    metrics = json.loads(Path("reports/metrics.json").read_text())
    assert metrics["best_f1"] >= MIN_F1, (
        f"Best F1 {metrics['best_f1']} is below the minimum {MIN_F1}"
    )