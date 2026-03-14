"""
Prepara o dataset PubMedQA PQA-L para fine-tuning e entrega.

- Converte ori_pqal.json para formato de instrução (JSONL: instruction/input/output e chat).
- Aplica split estratificado (500 test, 500 CV -> train/dev).
- Gera dataset anonimizado para entrega (IDs sintéticos no lugar de PMIDs).
"""
from pathlib import Path
import json
import random
from typing import Any

from .split import split_stratified, train_dev_split_from_cv

RANDOM_SEED = 0

# Template para formato instrução (Alpaca-style)
INSTRUCTION_TEMPLATE = (
    "Based on the following medical literature abstracts, answer the clinical question. "
    "Provide a concise evidence-based answer and state your decision: yes, no, or maybe."
)
CONTEXT_SEP = "\n\n---\n\n"


def _format_contexts(contexts: list[str]) -> str:
    return CONTEXT_SEP.join(contexts)


def to_instruction_record(pmid: str, item: dict[str, Any]) -> dict[str, Any]:
    """
    Converte um item PubMedQA para formato de instrução (Alpaca-style).
    Campos: instruction, input, output (e opcionalmente text para modelos que usam concat).
    """
    question = item["QUESTION"]
    contexts = item.get("CONTEXTS") or []
    long_answer = item.get("LONG_ANSWER") or ""
    decision = item.get("final_decision", "maybe")

    context_block = _format_contexts(contexts)
    user_input = f"Question: {question}\n\nAbstracts:\n{context_block}"
    # Output: resposta longa + decisão explícita (para o modelo aprender o formato)
    output = f"{long_answer}\n\nDecision: {decision}"

    return {
        "instruction": INSTRUCTION_TEMPLATE,
        "input": user_input,
        "output": output,
        "final_decision": decision,
    }


def to_chat_record(pmid: str, item: dict[str, Any]) -> dict[str, Any]:
    """
    Converte para formato chat (messages) usado por muitos pipelines de fine-tuning.
    """
    rec = to_instruction_record(pmid, item)
    return {
        "messages": [
            {"role": "system", "content": rec["instruction"]},
            {"role": "user", "content": rec["input"]},
            {"role": "assistant", "content": rec["output"]},
        ],
        "final_decision": rec["final_decision"],
    }


def anonymize_record(record: dict[str, Any], synthetic_id: str) -> dict[str, Any]:
    """
    Remove identificadores originais para entrega.
    O record já está em formato instruction ou chat; não contém PMID.
    synthetic_id é usado apenas em metadados se necessário.
    """
    out = {k: v for k, v in record.items() if k != "pmid"}
    out["id"] = synthetic_id
    return out


def load_pqal(data_path: Path) -> dict[str, Any]:
    """Carrega ori_pqal.json."""
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(records: list[dict], path: Path) -> None:
    """Escreve lista de dicts em JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run(
    data_dir: Path,
    *,
    format_type: str = "instruction",
    dev_ratio: float = 0.2,
    seed: int = RANDOM_SEED,
    write_anonymized: bool = True,
) -> None:
    """
    Executa preparação completa: load -> split -> convert -> write.

    data_dir: pasta que contém ori_pqal.json e onde serão escritos os outputs.
    format_type: "instruction" (instruction/input/output) ou "chat" (messages).
    dev_ratio: fração do CV usada como dev (resto train).
    write_anonymized: se True, escreve dataset anonimizado em data/anonymized/.
    """
    random.seed(seed)

    ori_path = data_dir / "ori_pqal.json"
    if not ori_path.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {ori_path}")

    dataset = load_pqal(ori_path)

    # Split: 500 test, 500 CV (igual ao split_dataset.py)
    test_set, cv_set = split_stratified(dataset, 2, seed=seed)
    train_set, dev_set = train_dev_split_from_cv(cv_set, dev_ratio=dev_ratio, seed=seed)

    to_record = to_chat_record if format_type == "chat" else to_instruction_record

    def to_records(name_to_items: dict[str, Any]) -> list[dict]:
        records = []
        for pmid, item in name_to_items.items():
            r = to_record(pmid, item)
            r["pmid"] = pmid  # mantém para avaliação; removido na versão anonimizada
            records.append(r)
        return records

    train_records = to_records(train_set)
    dev_records = to_records(dev_set)
    test_records = to_records(test_set)

    # JSONL para fine-tuning
    write_jsonl(train_records, data_dir / "train.jsonl")
    write_jsonl(dev_records, data_dir / "dev.jsonl")
    print(f"Escritos {len(train_records)} train, {len(dev_records)} dev -> train.jsonl, dev.jsonl")

    # test_set.json (formato original PMID -> item) para evaluation.py
    with open(data_dir / "test_set.json", "w", encoding="utf-8") as f:
        json.dump(test_set, f, indent=2, ensure_ascii=False)
    # test_ground_truth.json (PMID -> final_decision) para evaluation
    test_ground_truth = {pmid: item["final_decision"] for pmid, item in test_set.items()}
    with open(data_dir / "test_ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(test_ground_truth, f, indent=2)
    print(f"Escritos test_set.json e test_ground_truth.json ({len(test_set)} exemplos)")

    # test.jsonl (formato instrução, sem PMID no conteúdo) para inferência
    test_for_inference = []
    for r in test_records:
        r2 = {k: v for k, v in r.items() if k != "pmid"}
        r2["pmid"] = r["pmid"]  # manter pmid separado para métricas
        test_for_inference.append(r2)
    write_jsonl(test_for_inference, data_dir / "test.jsonl")

    # Dataset anonimizado para entrega (sem PMIDs; IDs sintéticos)
    if write_anonymized:
        anon_dir = data_dir / "anonymized"
        anon_dir.mkdir(parents=True, exist_ok=True)
        all_records = train_records + dev_records
        anon_records = []
        for i, r in enumerate(all_records):
            r_copy = {k: v for k, v in r.items() if k != "pmid"}
            r_copy["id"] = f"id_{i:05d}"
            anon_records.append(r_copy)
        write_jsonl(anon_records, anon_dir / "train_dev_anonymized.jsonl")
        # Metadados mínimos para entrega
        with open(anon_dir / "README.txt", "w", encoding="utf-8") as f:
            f.write(
                "Dataset anonimizado para entrega (Tech Challenge Fase 3).\n"
                "IDs de publicação (PMIDs) foram substituídos por IDs sintéticos (id_00000, ...).\n"
                "Formato: uma linha por exemplo em JSONL (instruction/input/output ou messages).\n"
            )
        print(f"Dataset anonimizado: {anon_dir}/train_dev_anonymized.jsonl ({len(anon_records)} exemplos)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepara dataset PubMedQA PQA-L para fine-tuning.")
    # Project root = solution2 (parent of src)
    _project_root = Path(__file__).resolve().parent.parent.parent
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_project_root / "data",
        help="Pasta com ori_pqal.json e saídas",
    )
    parser.add_argument(
        "--format",
        choices=["instruction", "chat"],
        default="instruction",
        help="Formato dos exemplos: instruction (Alpaca) ou chat (messages)",
    )
    parser.add_argument("--dev-ratio", type=float, default=0.2, help="Fração do CV para dev")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--no-anonymized", action="store_true", help="Não gerar dataset anonimizado")
    args = parser.parse_args()

    run(
        args.data_dir,
        format_type=args.format,
        dev_ratio=args.dev_ratio,
        seed=args.seed,
        write_anonymized=not args.no_anonymized,
    )
