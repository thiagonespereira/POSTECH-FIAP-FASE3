# Assistente com LangChain (Step 5) – Documentação

**Uso:** referência do que foi implementado no Step 5 e rascunho para o relatório técnico (item “Assistente” do checklist).  
**Contexto:** Plano Fase 3 – “Módulo que carrega a LLM fine-tunada; Chains para pergunta clínica → (opcional) contexto → LLM → resposta; regras de segurança e explainability.”

---

## 1. O que foi implementado

- **Carregamento da LLM fine-tunada para o LangChain:** o modelo (base Qwen2.5-0.5B + adaptador PEFT) é carregado e exposto como LLM compatível com LangChain via pipeline de text-generation e `HuggingFacePipeline`.
- **Chain pergunta → contexto (opcional) → LLM → resposta:** prompt no formato do PubMedQA (instrução + pergunta + abstracts opcionais); a saída é resposta em texto com “Decision: yes/no/maybe” quando aplicável.
- **Segurança e explainability:** disclaimer fixo em toda resposta (“não substitui avaliação por profissional de saúde”); indicação de fonte quando há contexto (“Com base no contexto fornecido”) ou quando não há (“Resposta sem contexto adicional”).

---

## 2. Componentes

| Componente | Descrição |
|------------|-----------|
| `src/models/load_llm_for_langchain.py` | Carrega base + PEFT, monta `pipeline` text-generation e retorna `HuggingFacePipeline` (LangChain). |
| `src/chains/medical_assistant.py` | Define o prompt (instrução + input), monta `LLMChain`, função `ask()` (pergunta + contexto opcional + disclaimer). |
| `scripts/run_assistant.py` | Script de linha de comando: carrega modelo, monta chain e responde a uma pergunta (ou ao primeiro exemplo de `data/test.jsonl`). |
| `scripts/run_assistant_test.sh` | Roda o assistente e grava a saída em `outputs/eval/assistant_test_output.txt`. |

O fluxo segue o padrão das aulas 0305 (LangChain): **PromptTemplate** com variáveis, **LLMChain**, e opção de enriquecer o input com contexto (abstracts ou, no futuro, consulta a DB).

---

## 3. Como usar

**Pré-requisito:** modelo fine-tunado em `outputs/finetune_pqal/` (ou caminho indicado por `--model-dir`). Ambiente com dependências do projeto (incluindo `langchain`, `langchain-community`).

### Comando básico (primeiro exemplo do test set)

```bash
# Na raiz do projeto (POSTECH-FIAP-FASE3)
.venv/bin/python scripts/run_assistant.py
```

Usa o primeiro registro de `data/test.jsonl` (pergunta + abstracts) e imprime a resposta no terminal.

### Pergunta e contexto customizados

```bash
.venv/bin/python scripts/run_assistant.py --pergunta "Can the PHQ-9 assess depression in people with vision loss?" --contexto "The PHQ-9 was completed by 103 participants with low vision. Fit to the Rasch model was demonstrated."
```

### Outros exemplos de teste

```bash
# Sem contexto (modelo responde só com a pergunta)
.venv/bin/python scripts/run_assistant.py --pergunta "Can offering influenza vaccination in office-based settings reduce racial disparities in adult vaccination?"

.venv/bin/python scripts/run_assistant.py --pergunta "Do hospitals provide lower quality care on weekends?"
```

### Gravar saída em arquivo

```bash
bash scripts/run_assistant_test.sh
# Saída em: outputs/eval/assistant_test_output.txt
```

### Parâmetros opcionais

| Argumento | Descrição | Padrão |
|-----------|-----------|--------|
| `--model-dir` | Pasta do modelo fine-tunado | `outputs/finetune_pqal` |
| `--data-dir` | Pasta com `test.jsonl` (para exemplo padrão) | `data` |
| `--max-new-tokens` | Máximo de tokens gerados | 256 |

---

## 4. Segurança e explainability

- **Disclaimer:** em todas as respostas é acrescentado um aviso de que a saída é gerada por um modelo fine-tunado em literatura médica e **não substitui avaliação por um profissional de saúde**; o usuário deve consultar um médico para decisões clínicas.
- **Fonte:** quando há contexto (abstracts ou texto fornecido), a resposta indica “Com base no contexto fornecido”; quando não há, “Resposta sem contexto adicional”.

Esses elementos atendem ao item do plano: “Regras de segurança e explainability (fonte da informação)”.

---

## 5. Próximos passos (plano)

- **Step 6 – LangGraph:** grafo(s) para fluxos como “receber pergunta → classificar intenção → buscar prontuário → gerar resposta → validar → log”, integrado ao mesmo modelo e com logging.
- **Consulta a DB:** hoje o “contexto” é passado manualmente (`--contexto`); pode ser estendido para consulta a base de dados ou Document Loaders (PDF/CSV), conforme aulas 030502.

---

## 6. Referências no repositório

- Plano: `PLANO_DESENVOLVIMENTO_FASE3.md` (Step 5, itens 144–147).
- Aulas LangChain: `docs/referencia_langchain_aulas_0305.md`.
- PDFs: `FASE3/pdfs/030501` a `030506` (Introdução a LangChain, Document Loaders, Prompts, Chains, Integração com LLM, Agents).
