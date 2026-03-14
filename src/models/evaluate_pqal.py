"""
Avaliação do modelo fine-tunado no test set PubMedQA (Step 4).

- Carrega o modelo (base + adaptador PEFT) e o tokenizer.
- Roda inferência em data/test.jsonl (ou test_set.json).
- Extrai a decisão (yes/no/maybe) da saída gerada e gera predictions.json.
- Calcula Accuracy e Macro-F1 (formato do evaluation.py do PubMedQA).
"""
from pathlib import Path
import json
import re
from typing import Any, Dict, List, Optional

import torch
from sklearn.metrics import accuracy_score, f1_score


# Padrão para extrair "Decision: yes/no/maybe" da resposta do modelo
DECISION_PATTERN = re.compile(
    r"\bDecision\s*:\s*(yes|no|maybe)\b",
    re.IGNORECASE,
)


def parse_decision(generated_text: str) -> str:
    """
    Extrai a decisão (yes, no, maybe) do texto gerado pelo modelo.
    Se não encontrar, retorna 'maybe' como fallback.
    """
    match = DECISION_PATTERN.search(generated_text)
    if match:
        return match.group(1).lower()
    # Fallback: última palavra conhecida no final do texto
    text_lower = generated_text.strip().lower()
    for choice in ("yes", "no", "maybe"):
        if text_lower.endswith(choice) or choice in text_lower[-20:]:
            return choice
    return "maybe"


def load_test_records(test_jsonl_path: Path) -> List[Dict[str, Any]]:
    """Carrega test.jsonl (cada linha: instruction, input, output, final_decision, pmid)."""
    records = []
    with open(test_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def run_inference(
    model,
    tokenizer,
    records: List[Dict[str, Any]],
    *,
    max_new_tokens: int = 256,
    prompt_template: str = "{instruction}\n\n{input}\n\n",
) -> Dict[str, str]:
    """
    Roda inferência para cada registro. record deve ter 'instruction', 'input', 'pmid'.
    Retorna dict pmid -> decisão (yes/no/maybe).
    """
    try:
        from tqdm import tqdm
        iterator = tqdm(records, desc="Inferência", unit="ex")
    except ImportError:
        iterator = records
    predictions = {}
    for rec in iterator:
        pmid = rec.get("pmid")
        if not pmid:
            continue
        prompt = prompt_template.format(
            instruction=rec.get("instruction", ""),
            input=rec.get("input", ""),
        )
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                top_k=0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        # Só os tokens gerados (excluir o prompt)
        gen_ids = out[0][inputs["input_ids"].shape[1] :]
        generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        decision = parse_decision(generated_text)
        predictions[str(pmid)] = decision
    return predictions


def compute_metrics(ground_truth: Dict[str, str], predictions: Dict[str, str]) -> Dict[str, float]:
    """Calcula Accuracy e Macro-F1. Chaves devem coincidir."""
    pmids = list(ground_truth.keys())
    if set(pmids) != set(predictions.keys()):
        raise ValueError(
            "predictions must have exactly the same keys as ground_truth. "
            f"Missing: {set(pmids) - set(predictions.keys())}, "
            f"Extra: {set(predictions.keys()) - set(pmids)}"
        )
    truth = [ground_truth[pmid] for pmid in pmids]
    preds = [predictions[pmid] for pmid in pmids]
    return {
        "accuracy": float(accuracy_score(truth, preds)),
        "macro_f1": float(f1_score(truth, preds, average="macro")),
    }


def evaluate(
    model_dir: Path,
    data_dir: Path,
    output_predictions_path: Optional[Path] = None,
    output_metrics_path: Optional[Path] = None,
    *,
    test_file: str = "test.jsonl",
    ground_truth_file: str = "test_ground_truth.json",
    max_new_tokens: int = 256,
    max_samples: Optional[int] = None,
) -> Dict[str, float]:
    """
    Carrega modelo em model_dir (deve ser pasta com adapter + tokenizer),
    roda inferência no test set, grava predictions e métricas.

    model_dir: pasta do modelo fine-tunado (ex.: outputs/finetune_pqal).
    data_dir: pasta com test.jsonl e test_ground_truth.json.
    output_predictions_path: onde salvar predictions.json (formato PMID -> yes/no/maybe).
    output_metrics_path: onde salvar métricas JSON (opcional).
    Retorna dict com accuracy e macro_f1.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    model_dir = Path(model_dir)
    data_dir = Path(data_dir)
    test_path = data_dir / test_file
    gt_path = data_dir / ground_truth_file

    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")

    # adapter_config.json indica o base_model
    adapter_config_path = model_dir / "adapter_config.json"
    if adapter_config_path.exists():
        with open(adapter_config_path) as f:
            adapter_config = json.load(f)
        base_model_name = adapter_config.get("base_model_name_or_path", "Qwen/Qwen2.5-0.5B-Instruct")
    else:
        base_model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    try:
        model = PeftModel.from_pretrained(base_model, model_dir)
    except TypeError as e:
        if "unexpected keyword argument" in str(e):
            # Adaptador foi salvo com PEFT mais novo; criar config enxuto para PEFT antigo
            import tempfile
            import shutil
            with open(model_dir / "adapter_config.json") as f:
                cfg = json.load(f)
            safe_keys = {"base_model_name_or_path", "peft_type", "r", "lora_alpha", "lora_dropout",
                         "target_modules", "bias", "task_type", "inference_mode", "init_lora_weights"}
            cfg_safe = {k: v for k, v in cfg.items() if k in safe_keys}
            tmp = Path(tempfile.mkdtemp())
            for fname in ("adapter_model.safetensors", "adapter_model.bin"):
                src = model_dir / fname
                if src.exists():
                    shutil.copy(src, tmp / fname)
                    break
            with open(tmp / "adapter_config.json", "w") as f:
                json.dump(cfg_safe, f, indent=2)
            model = PeftModel.from_pretrained(base_model, tmp)
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            raise
    model.eval()

    records = load_test_records(test_path)
    if max_samples is not None:
        records = records[: max_samples]
    n_records = len(records)
    print(f"Test set: {n_records} exemplos. Inferência um por vez (pode demorar vários minutos).")
    predictions = run_inference(model, tokenizer, records, max_new_tokens=max_new_tokens)

    with open(gt_path) as f:
        ground_truth = json.load(f)
    if max_samples is not None:
        pmids_in_pred = set(predictions.keys())
        ground_truth = {k: v for k, v in ground_truth.items() if k in pmids_in_pred}

    metrics = compute_metrics(ground_truth, predictions)

    if output_predictions_path is not None:
        output_predictions_path = Path(output_predictions_path)
        output_predictions_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_predictions_path, "w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=2)

    if output_metrics_path is not None:
        output_metrics_path = Path(output_metrics_path)
        output_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    return metrics
