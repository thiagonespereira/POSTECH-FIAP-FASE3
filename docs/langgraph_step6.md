# Fluxos LangGraph (Step 6) – Documentação

**Uso:** referência do que foi implementado no Step 6 (grafo do assistente médico com logging no estado).  
**Contexto:** Plano Fase 3 – “Grafo(s): receber pergunta → classificar intenção → buscar prontuário → gerar resposta → validar → log; integração com o mesmo modelo e com logging.”

---

## 1. O que foi implementado

- **Estado tipado (TypedDict):** `pergunta`, `contexto`, `intencao`, `resposta`, `valido`, `historico` (com agregador `add_messages` para logging).
- **Nós do grafo:**  
  - **classificar_intencao** – classifica a intenção (por simplicidade, tudo como `clinica`; futuro: regras ou LLM para faq vs clinica).  
  - **buscar_contexto** – stub que mantém o contexto existente (futuro: consulta a prontuário/base de abstracts).  
  - **gerar_resposta** – chama a chain do Step 5 (mesmo modelo fine-tunado) para gerar a resposta.  
  - **validar** – verifica se a resposta não está vazia e opcionalmente se contém “Decision:”.  
  - **log** – registra no campo `historico` do estado (logging).
- **Grafo linear:** `classificar_intencao` → `buscar_contexto` → `gerar_resposta` → `validar` → `log` → END.
- **Integração:** mesmo modelo e chain do Step 5 (`create_assistant` + `ask`); dependência `langgraph>=0.2.0` em `requirements.txt`.

---

## 2. Componentes

| Componente | Descrição |
|------------|-----------|
| `src/graphs/medical_flow.py` | Define `State`, nós do grafo e `build_medical_graph(model_dir, max_new_tokens=...)` que retorna o app compilado. |
| `src/graphs/__init__.py` | Exporta `State` e `build_medical_graph`. |
| `scripts/run_graph_assistant.py` | Script de linha de comando: compila o grafo, invoca com `pergunta` (e opcionalmente `contexto`), imprime resposta e histórico. |

---

## 3. Como usar

**Pré-requisitos:** **Python 3.9+** (LangGraph não suporta 3.8); modelo fine-tunado em `outputs/finetune_pqal/` (ou `--model-dir`); dependências instaladas (incluindo `langgraph`).

### Comando básico (exemplo do test set)

```bash
cd POSTECH-FIAP-FASE3
.venv/bin/python scripts/run_graph_assistant.py
```

Usa o primeiro registro de `data/test.jsonl` (pergunta + abstracts) e imprime a resposta e o histórico de nós executados.

### Pergunta e contexto customizados

```bash
.venv/bin/python scripts/run_graph_assistant.py --pergunta "Can the PHQ-9 assess depression in people with vision loss?" --contexto "The PHQ-9 was completed by 103 participants."
.venv/bin/python scripts/run_graph_assistant.py --pergunta "Does aspirin reduce fever?" --contexto "Abstract: Aspirin is an antipyretic."
```

### Ver estrutura do grafo (ASCII)

```bash
.venv/bin/python scripts/run_graph_assistant.py --draw --pergunta "Test?"
```

---

## 4. Fluxo e logging

- **Entrada do grafo:** `{"pergunta": str, "contexto": str (opcional)}`.
- **Saída:** estado final com `resposta`, `valido`, `historico` (lista de entradas de log por nó).
- O histórico é acumulado no estado via `add_messages`; cada nó pode retornar `{"historico": [{"role": "system", "content": "..."}]}` para registro.

Referências de aula: PDFs 030601–030604 (LangGraph) e notebooks `FASE3/0306-aula_*_video_*.ipynb`.
