"""
Pipeline de fine-tuning para o assistente médico (PubMedQA).
Usa Hugging Face Transformers + PEFT (LoRA/QLoRA) + TRL SFTTrainer.
"""
from pathlib import Path
import json
from typing import Any, Optional

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer


def load_jsonl_to_list(path: Path) -> list[dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def build_prompt_completion_dataset(
    train_path: Path, dev_path: Path
) -> tuple[Dataset, Dataset]:
    """
    Carrega train.jsonl e dev.jsonl (formato instruction/input/output)
    e converte para dataset com colunas 'prompt' e 'completion' para SFTTrainer.
    """
    train_records = load_jsonl_to_list(train_path)
    dev_records = load_jsonl_to_list(dev_path)

    def to_prompt_completion(r: dict) -> dict:
        instruction = r.get("instruction", "")
        user_input = r.get("input", "")
        output = r.get("output", "")
        prompt = f"{instruction}\n\n{user_input}\n\n"
        return {"prompt": prompt, "completion": output}

    train_data = [to_prompt_completion(r) for r in train_records]
    dev_data = [to_prompt_completion(r) for r in dev_records]

    return Dataset.from_list(train_data), Dataset.from_list(dev_data)


def run_finetune(
    data_dir: Path,
    output_dir: Path,
    *,
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    train_file: str = "train.jsonl",
    dev_file: str = "dev.jsonl",
    num_epochs: int = 3,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-5,
    max_seq_length: int = 2048,
    use_4bit: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: Optional[list[str]] = None,
    save_strategy: str = "epoch",
    save_total_limit: int = 2,
    logging_steps: int = 10,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
) -> None:
    """
    Executa fine-tuning com SFTTrainer + PEFT LoRA (opcional 4bit).
    Salva o adaptador em output_dir (modelo final = base + adaptador).
    """
    if lora_target_modules is None:
        lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = Path(data_dir) / train_file
    dev_path = Path(data_dir) / dev_file
    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not dev_path.exists():
        raise FileNotFoundError(f"Dev file not found: {dev_path}")

    train_dataset, eval_dataset = build_prompt_completion_dataset(train_path, dev_path)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Quantization config (4bit para caber em GPU pequena)
    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="bfloat16",
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    # LoRA
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # SFT training args (dataset com colunas "prompt" e "completion")
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        save_strategy=save_strategy,
        save_total_limit=save_total_limit,
        bf16=True,
        max_seq_length=max_seq_length,
        dataset_text_field=None,
        dataset_num_proc=1,
        packing=False,
        completion_only_loss=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        max_seq_length=max_seq_length,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    trainer.save_state()
    print(f"Model and tokenizer saved to {output_dir}")


if __name__ == "__main__":
    import argparse
    # Garantir que a raiz do projeto está no path (para importar config)
    _root = Path(__file__).resolve().parent.parent.parent
    if str(_root) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(_root))
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

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--max-length", type=int, default=MAX_SEQ_LENGTH)
    parser.add_argument("--no-4bit", action="store_true", help="Use full precision (more VRAM)")
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
