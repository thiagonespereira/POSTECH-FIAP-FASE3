#!/usr/bin/env python3
"""
Assistente médico via LangGraph (Step 6).

Fluxo: pergunta → classificar_intencao → buscar_contexto → gerar_resposta → validar → log.
Usa o mesmo modelo fine-tunado do Step 5; imprime resposta e histórico ao final.

Uso (na raiz do projeto):

    python scripts/run_graph_assistant.py --pergunta "Can the PHQ-9 assess depression in people with vision loss?"
    python scripts/run_graph_assistant.py --pergunta "Does aspirin reduce fever?" --contexto "Abstract: Aspirin is an antipyretic..."
    python scripts/run_graph_assistant.py   # usa pergunta de exemplo do test set
"""
import sys
import json
from pathlib import Path

# LangGraph usa sintaxe de Python 3.9+ (ex.: dict[str, Any]); falha em 3.8
if sys.version_info < (3, 9):
    print("Erro: O Step 6 (LangGraph) exige Python 3.9 ou superior.", file=sys.stderr)
    print("Versão atual:", sys.version, file=sys.stderr)
    print("Use um venv com Python 3.9+ ou rode apenas o assistente (Step 5): .venv/bin/python scripts/run_assistant.py", file=sys.stderr)
    sys.exit(1)

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.graphs.medical_flow import build_medical_graph


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
    return rec.get("input", ""), rec.get("instruction", "")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Assistente médico via LangGraph (modelo fine-tunado + logging no estado)."
    )
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
        help="Contexto opcional (abstracts). Se omitido, o grafo usa stub vazio.",
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
    parser.add_argument(
        "--draw",
        action="store_true",
        help="Imprimir grafo em ASCII (draw_ascii)",
    )
    args = parser.parse_args()

    print("Carregando modelo e compilando grafo...", flush=True)
    app = build_medical_graph(args.model_dir, max_new_tokens=args.max_new_tokens)

    if args.draw:
        print("\n=== Estrutura do grafo (ASCII) ===\n", flush=True)
        print(app.get_graph().draw_ascii(), flush=True)

    if args.pergunta:
        pergunta = args.pergunta
        contexto = args.contexto or ""
    else:
        input_exemplo, _ = _first_example_from_test(args.data_dir)
        if input_exemplo is None:
            print("Nenhum exemplo em data/test.jsonl. Use --pergunta 'sua pergunta'.")
            return
        if "Question:" in input_exemplo and "Abstracts:" in input_exemplo:
            parts = input_exemplo.split("Abstracts:", 1)
            pergunta = parts[0].replace("Question:", "").strip()
            contexto = parts[1].strip() if len(parts) > 1 else ""
        else:
            pergunta = input_exemplo[:200] + "..." if len(input_exemplo) > 200 else input_exemplo
            contexto = ""

    estado_inicial = {"pergunta": pergunta, "contexto": contexto or ""}
    print("\nPergunta:", pergunta[:200] + ("..." if len(pergunta) > 200 else ""), flush=True)
    if contexto:
        print("Contexto:", len(contexto), "caracteres", flush=True)
    print("\nExecutando grafo...\n", flush=True)

    resultado = app.invoke(estado_inicial)

    print("--- Resposta ---\n", resultado.get("resposta", ""), flush=True)
    print("\n--- Histórico (log) ---", flush=True)
    for i, h in enumerate(resultado.get("historico") or []):
        content = getattr(h, "content", h) if not isinstance(h, dict) else h.get("content", h)
        print(f"  {i+1}. {content}", flush=True)
    if resultado.get("valido") is not None:
        print(f"\nValido: {resultado['valido']}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Erro: {e}", flush=True)
        raise
