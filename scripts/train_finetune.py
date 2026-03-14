#!/usr/bin/env python3
"""
Script de fine-tuning do modelo médico (Step 3).

Execute a partir da raiz do projeto (POSTECH-FIAP-FASE3):

    cd POSTECH-FIAP-FASE3
    python scripts/train_finetune.py

Ou com argumentos:

    python scripts/train_finetune.py --data-dir data --output-dir outputs/finetune_pqal --epochs 3

Requisitos: dados preparados (train.jsonl, dev.jsonl em data/) e GPU com
suficiente VRAM (recomendado: Colab com T4 ou superior, ou máquina local
com 8+ GB). O modelo e a quantização 4-bit estão configurados para caber
em ~15 GB (QLoRA). Veja config/finetune_defaults.py e README.
"""
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config.finetune_defaults import (
    MODEL_NAME,
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    NUM_EPOCHS,
    BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    MAX_SEQ_LENGTH,
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_TARGET_MODULES,
    USE_4BIT,
    SAVE_STRATEGY,
    SAVE_TOTAL_LIMIT,
    LOGGING_STEPS,
    WARMUP_RATIO,
    WEIGHT_DECAY,
)
from src.models.run_finetune import run_finetune


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Fine-tuning do modelo médico (PubMedQA) com PEFT LoRA."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Pasta com train.jsonl e dev.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Pasta para checkpoints e modelo final (adaptador PEFT)",
    )
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="Modelo base Hugging Face")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--max-length", type=int, default=MAX_SEQ_LENGTH)
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Desativa quantização 4-bit (usa mais VRAM)",
    )
    args = parser.parse_args()

    run_finetune(
        args.data_dir,
        args.output_dir,
        model_name=args.model,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_seq_length=args.max_length,
        use_4bit=not args.no_4bit,
        lora_r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        lora_target_modules=LORA_TARGET_MODULES,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        save_strategy=SAVE_STRATEGY,
        save_total_limit=SAVE_TOTAL_LIMIT,
        logging_steps=LOGGING_STEPS,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
    )


if __name__ == "__main__":
    main()
