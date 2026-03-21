"""
Assistente médico com LangChain (Step 5).

Chain: pergunta clínica → (opcional) contexto → prompt → LLM → resposta.
Regras de segurança: disclaimer e citação de fonte quando há contexto.
"""
from pathlib import Path
from typing import Optional

# Disclaimer fixo (explainability / segurança)
DISCLAIMER = (
    "\n\n---\n"
    "**Aviso:** Esta resposta é gerada por um modelo de linguagem fine-tunado em literatura médica (PubMedQA). "
    "Ela não substitui avaliação por um profissional de saúde. Consulte sempre um médico para decisões clínicas."
)


def build_medical_chain(llm, *, max_new_tokens: int = 256):
    """
    Constrói a chain do assistente: (instruction + input) formatados → LLM.

    O formato do prompt segue o usado no treino/avaliação (instruction + input)
    para que o modelo retorne resposta evidence-based e "Decision: yes/no/maybe".

    Não usa ``langchain_core.prompts`` / ``PromptTemplate``: em Colab algumas
    versões do LangChain ainda falham com ``No module named 'langchain.prompts'``.
    """
    instruction = (
        "Based on the following medical literature abstracts, answer the clinical question. "
        "Provide a concise evidence-based answer and state your decision: yes, no, or maybe."
    )

    class _MedicalChain:
        """Compatível com ``chain.invoke({\"instruction\", \"input_text\"})``."""

        __slots__ = ("_llm",)

        def __init__(self, llm_):
            self._llm = llm_

        def invoke(self, inputs: dict) -> str:
            text = f"{inputs['instruction']}\n\n{inputs['input_text']}\n\n"
            return self._llm.invoke(text)

    return _MedicalChain(llm), instruction


def ask(
    chain,
    instruction: str,
    pergunta: str,
    contexto: Optional[str] = None,
    *,
    add_disclaimer: bool = True,
) -> str:
    """
    Envia pergunta clínica à chain e retorna a resposta (com optional disclaimer).

    pergunta: questão clínica do usuário.
    contexto: texto opcional (ex.: abstracts, resultado de Document Loader ou DB).
    """
    if contexto and contexto.strip():
        input_text = f"Question: {pergunta}\n\nAbstracts:\n{contexto.strip()}"
        source_note = " (Com base no contexto fornecido.)"
    else:
        input_text = f"Question: {pergunta}\n\nAbstracts:\n(No additional context provided.)"
        source_note = " (Resposta sem contexto adicional.)"

    out = chain.invoke({"instruction": instruction, "input_text": input_text})
    # HuggingFacePipeline / chain customizada: geralmente str; dict legado com "text"
    if isinstance(out, dict):
        response = out.get("text", out.get("output", str(out)))
    elif hasattr(out, "content"):
        response = out.content
    else:
        response = out
    text = (response.strip() if isinstance(response, str) else str(response).strip())

    if add_disclaimer:
        text += source_note + DISCLAIMER
    return text


def create_assistant(model_dir: Path, *, max_new_tokens: int = 256):
    """
    Cria o assistente: carrega a LLM e monta a chain.

    model_dir: pasta do modelo fine-tunado (outputs/finetune_pqal ou equivalente).
    Retorna: (chain, instruction) para usar com ask().
    """
    from src.models.load_llm_for_langchain import load_llm_langchain

    llm = load_llm_langchain(model_dir, max_new_tokens=max_new_tokens)
    chain, instruction = build_medical_chain(llm, max_new_tokens=max_new_tokens)
    return chain, instruction
