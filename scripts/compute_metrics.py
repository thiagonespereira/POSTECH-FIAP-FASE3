#!/usr/bin/env python3
"""
Calcula Accuracy e Macro-F1 a partir de predictions.json e test_ground_truth.json.

Formato esperado (igual ao evaluation.py do PubMedQA):
- ground_truth: JSON com PMID -> "yes"|"no"|"maybe"
- predictions: JSON com PMID -> "yes"|"no"|"maybe"

Uso:

    python scripts/compute_metrics.py data/test_ground_truth.json outputs/eval/predictions.json
"""
import json
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.models.evaluate_pqal import compute_metrics


def main() -> None:
    if len(sys.argv) != 3:
        print("Uso: python scripts/compute_metrics.py <ground_truth.json> <predictions.json>")
        sys.exit(1)
    gt_path = Path(sys.argv[1])
    pred_path = Path(sys.argv[2])
    with open(gt_path) as f:
        ground_truth = json.load(f)
    with open(pred_path) as f:
        predictions = json.load(f)
    metrics = compute_metrics(ground_truth, predictions)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
