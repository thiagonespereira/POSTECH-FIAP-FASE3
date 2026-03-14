# Plano de Desenvolvimento – Tech Challenge IADT Fase 3

**Projeto:** Assistente virtual médico com LLM fine-tunada e LangChain  
**Dataset:** PubMedQA (PQA-L) — 1000 instâncias em `data/ori_pqal.json`  
**Referência:** Instruções em `FASE3/Tech Challenge IADT - Fase 3.pdf`

---

## 1. Visão geral do desafio

Objetivos do hospital:

- **Assistente virtual médico** treinado com dados próprios (protocolos, FAQs, laudos/receitas).
- **Fluxos de decisão automatizados** com LangChain: verificar exames pendentes, sugerir tratamentos, alertas à equipe.
- **Segurança e rastreabilidade:** limites de atuação, logging e explainability.

---

## 2. Entregáveis obrigatórios (checklist)

### 2.1 Repositório Git

| Item | Descrição | Status |
|------|-----------|--------|
| Pipeline de fine-tuning | Scripts/modules para treinar a LLM | A fazer |
| Integração com LangChain | Pipeline que usa a LLM customizada | A fazer |
| Fluxos LangGraph | Grafos de decisão/automação | A fazer |
| Dataset anonimizado ou sintético | `ori_pqal.json` anonimizado ou exemplo sintético | A fazer |
| README | Instruções completas de uso e execução | A fazer |

### 2.2 Relatório técnico

| Item | Conteúdo | Status |
|------|----------|--------|
| Fine-tuning | Explicação do processo (dados, modelo base, método) | A fazer |
| Assistente | Descrição do assistente médico e capacidades | A fazer |
| Diagrama | Fluxo LangChain/LangGraph (figura ou diagrama) | A fazer |
| Avaliação | Métricas do modelo e análise dos resultados | A fazer |

### 2.3 Vídeo (até 15 min)

| Cena | Conteúdo | Status |
|------|----------|--------|
| 1 | Treinamento e funcionamento da LLM personalizada | A fazer |
| 2 | Execução de um fluxo automatizado (LangGraph) | A fazer |
| 3 | Resposta a perguntas clínicas contextualizadas | A fazer |
| 4 | Logs e validação das respostas | A fazer |

---

## 3. Requisitos técnicos detalhados

### 3.1 Fine-tuning de LLM com dados médicos

- **Modelo base:** LLaMA, Falcon ou outro LLM aberto (compatível com Colab/GPU quando possível).
- **Dados a utilizar:**
  - PubMedQA (`ori_pqal.json`): perguntas clínicas, contextos (abstracts), respostas longas, `final_decision` (yes/no/maybe).
  - Opcional: protocolos, FAQs e modelos de laudos/receitas (sintéticos ou anonimizados).
- **Preparação de dados:**
  - Preprocessamento (tokenização, truncamento, formatação para instrução/resposta).
  - Anonimização (remover/camufar PII se houver; PubMedQA é baseado em literatura, não em pacientes).
  - Curadoria (filtrar exemplos ruins, balancear yes/no/maybe se necessário).
- **Entrega de código:**
  - **Script Python** (`train_finetune.py` ou similar): executável localmente ou em servidor com GPU.
  - **Notebook Colab** (`finetune_medical_llm.ipynb`): mesma pipeline para rodar no Google Colab com GPU, garantindo acesso ao modelo fine-tunado (export/upload para Hugging Face ou disco).

### 3.2 Assistente médico com LangChain

- **Pipeline LangChain:**
  - Integrar a LLM fine-tunada (Hugging Face ou local).
  - Prompts para papel de “assistente médico” e limites (ex.: não prescrever).
- **Base de dados estruturada:**
  - Simular prontuários/registros (ex.: SQLite/JSON com pacientes sintéticos).
  - Consultas via LangChain (tool/agent ou chain que chama DB).
- **Contextualização:**
  - Injetar dados do “paciente” (ex.: exames pendentes, histórico) no contexto da pergunta antes de chamar a LLM.

### 3.3 Segurança e validação

- **Limites de atuação:**
  - Regras claras no sistema (ex.: nunca prescrever sem validação humana; respostas apenas informativas).
  - Prompts e possivelmente pós-processamento para recusar pedidos inadequados.
- **Logging:**
  - Registrar perguntas, respostas, fontes e decisões para auditoria.
- **Explainability:**
  - Indicar fonte da informação (ex.: “Com base no protocolo X” ou “Conforme abstract do artigo Y”).

### 3.4 Organização do código

- Projeto **modular** em Python (ex.: `data/`, `models/`, `chains/`, `tools/`, `config/`).
- **README** com: ambiente (Python, pip/conda), instalação, como rodar fine-tuning, como rodar o assistente e onde está o modelo fine-tunado.

---

## 4. Dataset PubMedQA (PQA-L)

- **Local:** `data/ori_pqal.json` na raiz do repositório (copiado de `FASE3/pubmedqa-master/data/`).
- **Estrutura por PMID:**
  - `QUESTION`: pergunta clínica.
  - `CONTEXTS`: lista de abstracts/trechos.
  - `LABELS`, `MESHES`, `YEAR`: metadados.
  - `final_decision`: `"yes"` | `"no"` | `"maybe"`.
  - `LONG_ANSWER`: resposta longa (ótimo para fine-tuning de geração).
- **Uso:** treino/validação do modelo; avaliação com `evaluation.py` (Accuracy, Macro-F1) usando `test_ground_truth.json` após split (referência: `FASE3/pubmedqa-master/preprocess/split_dataset.py`).

---

## 5. Referências no repositório (FASE3)

- **Fine-tuning e preparação de dados:**
  - `FASE3/0304-fine-tuning-rag-documentos-fiap-main/Aula 01 - Preparando dados de treinamento para fine-tuning/prepare-data.ipynb` — formato de dados para treino.
  - `FASE3/0304-fine-tuning-rag-documentos-fiap-main/Aula 02 - Fine tuning de LLM para documentos/finetuning_summarizer.ipynb` — pipeline de fine-tuning.
- **RAG e integração:**
  - `FASE3/0304-fine-tuning-rag-documentos-fiap-main/Aula 03 - RAG para documentos/rag-application.ipynb` — RAG com documentos.
- **LangChain e LangGraph:**
  - `FASE3/0306-aula_1_video_2.ipynb`, `0306-aula_2_video_*.ipynb` — LangChain, FAISS, LangGraph (StateGraph, nós, condicionais).
- **PubMedQA:**
  - `FASE3/pubmedqa-master/preprocess/split_dataset.py` — split train/dev/test (500 test, 500 CV, 10 folds).
  - `FASE3/pubmedqa-master/evaluation.py` — avaliação (Accuracy, Macro-F1).

---

## 6. Plano de execução (ordem sugerida)

1. **Setup do projeto**  
   - Estrutura de pastas (ex.: `data/`, `src/`, `scripts/`, `notebooks/`).  
   - `requirements.txt` e README mínimo.

2. **Preparação do dataset**  
   - Script para converter `ori_pqal.json` para formato de instrução (ex.: JSONL com `instruction`/`input`/`output` ou estilo chat).  
   - Anonimização (se aplicável) e split (reutilizar lógica de `split_dataset.py`).  
   - Gerar dataset anonimizado ou exemplo sintético para entrega.

3. **Pipeline de fine-tuning**  
   - `train_finetune.py`: carregar modelo base, carregar dados, treino (ex.: Hugging Face Transformers + PEFT/QLoRA).  
   - `finetune_medical_llm.ipynb`: mesma pipeline no Colab (montar Drive, instalar deps, salvar modelo/upload).  
   - Garantir que o modelo fine-tunado fique acessível (Hugging Face ou caminho documentado).

4. **Avaliação do modelo**  
   - Script de inferência no test set PubMedQA.  
   - Uso de `evaluation.py` (ou portar métricas) e registro no relatório.

5. **Assistente com LangChain**  
   - Módulo que carrega a LLM fine-tunada.  
   - Chains para: pergunta clínica → (opcional) consulta a DB → contexto → LLM → resposta.  
   - Regras de segurança e explainability (fonte da informação).

6. **Fluxos LangGraph**  
   - Grafo(s): ex. “receber pergunta → classificar intenção → buscar prontuário → gerar resposta → validar → log”.  
   - Integração com o mesmo modelo e com logging.

7. **Segurança, logging e explainability**  
   - Implementar limites, logging detalhado e citação de fontes em todas as respostas.

8. **Relatório técnico e vídeo**  
   - Redigir relatório (fine-tuning, assistente, diagrama, avaliação).  
   - Gravar vídeo demonstrando os quatro pontos do checklist.

---

## 7. Estrutura de pastas (repositório)

```
POSTECH-FIAP-FASE3/
├── data/
│   └── ori_pqal.json          # já copiado
├── src/                       # código modular
│   ├── data/                  # preparação e split
│   ├── models/                # carregamento da LLM
│   ├── chains/                # LangChain chains
│   ├── graphs/                # LangGraph flows
│   └── utils/                 # logging, segurança, explainability
├── scripts/
│   └── train_finetune.py      # treino via linha de comando
├── notebooks/
│   └── finetune_medical_llm.ipynb   # Colab fine-tuning
├── config/                    # configs e prompts
├── docs/                      # diagramas e relatório
├── requirements.txt
├── README.md
└── PLANO_DESENVOLVIMENTO_FASE3.md   # este documento
```

---

## 8. Resumo

- **Dataset:** PubMedQA em `data/ori_pqal.json` (1000 instâncias).  
- **Fine-tuning:** script `.py` + notebook Colab para uso de GPU e acesso ao modelo fine-tunado.  
- **Entregas:** repositório (código + dataset anonimizado/sintético + README), relatório técnico e vídeo de até 15 minutos.  
- **Referências:** PDF da Fase 3, notebooks e scripts em `FASE3` (fine-tuning, RAG, LangChain, LangGraph, PubMedQA split e avaliação).

Este plano pode ser usado como guia de desenvolvimento e checklist dos entregáveis da Fase 3.
