#!/usr/bin/env python3
"""
Avaliação do modelo fine-tunado no test set PubMedQA (Step 4).

Roda inferência com o modelo em outputs/finetune_pqal (ou --model-dir),
gera predictions.json e calcula Accuracy e Macro-F1.

Uso (na raiz do projeto):

    cd POSTECH-FIAP-FASE3
    python scripts/run_evaluate.py

Ou com caminhos explícitos:

    python scripts/run_evaluate.py --model-dir outputs/finetune_pqal --data-dir data --out-dir outputs/eval
"""
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.models.evaluate_pqal import evaluate


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Avaliação do modelo no test set PubMedQA.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=_project_root / "outputs" / "finetune_pqal",
        help="Pasta do modelo fine-tunado (adaptador PEFT + tokenizer)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_project_root / "data",
        help="Pasta com test.jsonl e test_ground_truth.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_project_root / "outputs" / "eval",
        help="Pasta de saída para predictions.json e metrics.json",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Máximo de tokens gerados por exemplo",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Avaliar só os N primeiros exemplos (útil para teste rápido)",
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / "predictions.json"
    metrics_path = out_dir / "metrics.json"

    n = args.max_samples or "todos"
    print(f"Carregando modelo e rodando inferência no test set (amostras: {n})...")
    metrics = evaluate(
        model_dir=args.model_dir,
        data_dir=args.data_dir,
        output_predictions_path=predictions_path,
        output_metrics_path=metrics_path,
        max_new_tokens=args.max_new_tokens,
        max_samples=args.max_samples,
    )
    print(f"Predictions salvas em: {predictions_path}")
    print(f"Métricas salvas em: {metrics_path}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
