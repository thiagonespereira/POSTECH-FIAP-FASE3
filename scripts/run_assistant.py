#!/usr/bin/env python3
"""
Assistente médico com LangChain (Step 5).

Carrega o modelo fine-tunado, monta a chain e responde a uma pergunta clínica.
Opcional: passar contexto (abstracts) ou ler um exemplo de data/test.jsonl.

Uso (na raiz do projeto):

    python scripts/run_assistant.py --pergunta "Can the PHQ-9 assess depression in people with vision loss?"

    python scripts/run_assistant.py   # usa pergunta de exemplo do test set
"""
import sys
import json
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.chains.medical_assistant import create_assistant, ask, build_medical_chain


def _first_example_from_test(data_dir: Path):
    """Lê o primeiro exemplo de data/test.jsonl para testar com input real."""
    test_path = data_dir / "test.jsonl"
    if not test_path.exists():
        return None, None
    with open(test_path, "r", encoding="utf-8") as f:
        first = f.readline()
    if not first.strip():
        return None, None
    rec = json.loads(first)
    # input contém "Question: ... \n\nAbstracts: ..."
    return rec.get("input", ""), rec.get("instruction", "")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Assistente médico (LangChain + modelo fine-tunado).")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=_project_root / "outputs" / "finetune_pqal",
        help="Pasta do modelo fine-tunado",
    )
    parser.add_argument(
        "--pergunta",
        type=str,
        default=None,
        help="Pergunta clínica (se omitido, usa exemplo do test set)",
    )
    parser.add_argument(
        "--contexto",
        type=str,
        default=None,
        help="Contexto opcional (abstracts ou texto). Se omitido e --pergunta for passada, usa texto genérico.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_project_root / "data",
        help="Pasta com test.jsonl (para exemplo padrão)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Máximo de tokens gerados",
    )
    args = parser.parse_args()

    print("Carregando modelo e montando chain...", flush=True)
    chain, instruction = create_assistant(args.model_dir, max_new_tokens=args.max_new_tokens)

    if args.pergunta:
        pergunta = args.pergunta
        contexto = args.contexto or ""
    else:
        # Usar primeiro exemplo do test set
        input_exemplo, inst_exemplo = _first_example_from_test(args.data_dir)
        if input_exemplo is None:
            print("Nenhum exemplo em data/test.jsonl. Use --pergunta 'sua pergunta'.")
            return
        # Extrair Question e Abstracts do input
        if "Question:" in input_exemplo and "Abstracts:" in input_exemplo:
            parts = input_exemplo.split("Abstracts:", 1)
            pergunta = parts[0].replace("Question:", "").strip()
            contexto = parts[1].strip() if len(parts) > 1 else ""
            instruction = inst_exemplo or instruction
        else:
            pergunta = input_exemplo[:200] + "..." if len(input_exemplo) > 200 else input_exemplo
            contexto = ""

    print("Pergunta:", pergunta[:200] + ("..." if len(pergunta) > 200 else ""), flush=True)
    if contexto:
        print("Contexto (abstracts):", len(contexto), "caracteres", flush=True)
    print("\nGerando resposta...\n", flush=True)
    resposta = ask(chain, instruction, pergunta, contexto=contexto or None, add_disclaimer=True)
    print(resposta, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Erro: {e}", flush=True)
        raise
