"""
Fluxo LangGraph do assistente médico (Step 6).

Grafo: receber pergunta → classificar intenção → buscar prontuário (stub) → gerar resposta → validar → log.
Usa o mesmo modelo fine-tunado e chain do Step 5; histórico (logging) no estado.
"""
from pathlib import Path
from typing import Optional, TypedDict

try:
    from typing import Annotated
except ImportError:
    from typing_extensions import Annotated

from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from src.chains.medical_assistant import ask, create_assistant


# ---------- Estado tipado ----------
class State(TypedDict, total=False):
    """Estado do grafo: pergunta → classificar → buscar contexto → gerar resposta → validar → log."""

    pergunta: str
    contexto: Optional[str]
    intencao: str
    resposta: str
    valido: bool
    historico: Annotated[list, add_messages]


# ---------- Nós do grafo ----------
def _node_classificar_intencao(estado: dict) -> dict:
    """Classifica a intenção: clinica (pergunta clínica) ou faq. Por simplicidade, tudo vai para clinica."""
    pergunta = (estado.get("pergunta") or "").strip()
    intencao = "clinica"  # futuro: regras ou LLM para faq vs clinica
    return {
        "intencao": intencao,
        "historico": [{"role": "system", "content": f"classificar_intencao -> {intencao}"}],
    }


def _node_buscar_contexto(estado: dict) -> dict:
    """Busca contexto (ex.: prontuário). Stub: mantém contexto existente ou vazio."""
    # Futuro: consultar base de abstracts ou prontuário
    contexto = estado.get("contexto") or ""
    return {
        "contexto": contexto,
        "historico": [{"role": "system", "content": "buscar_contexto (stub)"}],
    }


def _node_gerar_resposta(chain, instruction: str, estado: dict) -> dict:
    """Gera resposta usando a chain do assistente (modelo fine-tunado)."""
    pergunta = estado.get("pergunta") or ""
    contexto = estado.get("contexto") or None
    resposta = ask(chain, instruction, pergunta, contexto=contexto, add_disclaimer=True)
    return {
        "resposta": resposta,
        "historico": [{"role": "system", "content": "gerar_resposta"}],
    }


def _node_validar(estado: dict) -> dict:
    """Valida a resposta: não vazia e, se possível, contém 'Decision:'."""
    resposta = (estado.get("resposta") or "").strip()
    valido = bool(resposta)
    if valido and "Decision:" not in resposta:
        valido = True  # aceitar mesmo sem Decision (modelo pode variar)
    return {
        "valido": valido,
        "historico": [{"role": "system", "content": f"validar -> valido={valido}"}],
    }


def _node_log(estado: dict) -> dict:
    """Registra no histórico e finaliza (logging)."""
    return {
        "historico": [
            {
                "role": "system",
                "content": f"log -> pergunta_len={len(estado.get('pergunta') or '')} resposta_len={len(estado.get('resposta') or '')}",
            }
        ]
    }


def build_medical_graph(
    model_dir: Path,
    *,
    max_new_tokens: int = 256,
):
    """
    Constrói o grafo compilado do assistente médico.

    model_dir: pasta do modelo fine-tunado (outputs/finetune_pqal ou equivalente).
    Retorna: app compilado (invoke com {"pergunta": "...", "contexto": opcional}).
    """
    model_dir = Path(model_dir)
    chain, instruction = create_assistant(model_dir, max_new_tokens=max_new_tokens)

    grafo = StateGraph(State)

    grafo.add_node("classificar_intencao", _node_classificar_intencao)
    grafo.add_node("buscar_contexto", _node_buscar_contexto)
    grafo.add_node(
        "gerar_resposta",
        lambda s: _node_gerar_resposta(chain, instruction, s),
    )
    grafo.add_node("validar", _node_validar)
    grafo.add_node("log", _node_log)

    grafo.set_entry_point("classificar_intencao")
    grafo.add_edge("classificar_intencao", "buscar_contexto")
    grafo.add_edge("buscar_contexto", "gerar_resposta")
    grafo.add_edge("gerar_resposta", "validar")
    grafo.add_edge("validar", "log")
    grafo.add_edge("log", END)

    return grafo.compile()
