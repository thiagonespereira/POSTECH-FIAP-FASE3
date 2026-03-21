# POSTECH-FIAP-FASE3 – Assistente médico (Tech Challenge IADT Fase 3)

Assistente virtual médico com LLM fine-tunada em dados clínicos (PubMedQA) e integração LangChain/LangGraph.

## Estrutura do projeto

```
POSTECH-FIAP-FASE3/
├── data/                 # Dataset (ori_pqal.json)
├── src/                  # Código modular
│   ├── data/             # Preparação e split
│   ├── models/           # Carregamento da LLM
│   ├── chains/           # LangChain chains
│   ├── graphs/           # LangGraph flows
│   └── utils/            # Logging, segurança, explainability
├── scripts/              # Scripts de treino (linha de comando)
├── notebooks/            # Notebooks (ex.: Colab fine-tuning)
├── config/               # Configurações e prompts
├── docs/                 # Diagramas e relatório
├── requirements.txt
└── PLANO_DESENVOLVIMENTO_FASE3.md
```

**Mapeamento PDF do desafio → arquivos (.py / .ipynb):** `docs/MAPEAMENTO_ENTREGAS_TECH_CHALLENGE_FASE3.md` (referência: `FASE3/Tech Challenge IADT - Fase 3.pdf`).

## Ambiente

Recomendado: **Python 3.9+** (obrigatório para o Step 6 – LangGraph; o restante do projeto roda em 3.8).

```bash
python3.9 -m venv .venv   # ou python3.10 / python3.11
source .venv/bin/activate   # Linux/macOS
# ou: .venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

Nos exemplos abaixo, os comandos usam o Python do venv: **`.venv/bin/python`** (Linux/macOS). No Windows, ative o venv e use `python`.

**GPU (CUDA):** Para rodar o assistente (Step 5) e o grafo LangGraph (Step 6) com o modelo fine-tunado (adaptador PEFT/QLoRA), é recomendado ambiente com **GPU e drivers CUDA**. Em máquina só CPU, o carregamento do adaptador pode falhar (bitsandbytes/Triton esperam um driver ativo). Preparação de dados, avaliação off-line e scripts que não carregam o modelo PEFT funcionam em CPU.

**Se `pip install -r requirements.txt` falhar ao compilar pyarrow** (erro tipo "cmake failed" ou "CMake 3.25 or higher is required"): o projeto fixa `pyarrow>=14.0.0,<17` para usar wheel pré-compilado. Se ainda tentar compilar, instale CMake 3.25+ ou use um ambiente com Python 3.10+ onde wheels estão mais disponíveis.

## Dataset

- **PubMedQA (PQA-L):** `data/ori_pqal.json` — 1000 instâncias (pergunta, contextos, resposta, decisão yes/no/maybe).

### Preparação (Step 2)

Gera train/dev/test em formato de instrução e dataset anonimizado para entrega:

```bash
cd POSTECH-FIAP-FASE3
.venv/bin/python scripts/run_prepare_data.py
```

Ou com opções:

```bash
.venv/bin/python -m src.data.prepare_pqal --data-dir data --format instruction --dev-ratio 0.2
```

**Saídas em `data/`:**
- `train.jsonl`, `dev.jsonl` — fine-tuning (instruction/input/output + final_decision)
- `test.jsonl`, `test_set.json`, `test_ground_truth.json` — avaliação
- `data/anonymized/train_dev_anonymized.jsonl` — dataset anonimizado (IDs sintéticos) para entrega

### Verificação dos dados preparados

O script `scripts/verify_data.py` valida a consistência dos arquivos gerados pela preparação (contagens, formato, ausência de sobreposição entre train/dev/test e conformidade do dataset anonimizado). Útil para quem for avaliar o projeto conferir se os dados estão corretos após rodar a preparação.

```bash
cd POSTECH-FIAP-FASE3
.venv/bin/python scripts/verify_data.py
```

Se tudo estiver correto, o script termina com **All checks passed.**; caso contrário, lista as falhas e encerra com código 1.

## Fine-tuning (Step 3)

Treino do modelo com **Hugging Face Transformers + PEFT (LoRA/QLoRA) + TRL SFTTrainer**. O modelo base é carregado em 4-bit (QLoRA) para caber em GPU com ~15 GB (ex.: Colab T4).

### Por que o modelo Qwen2.5-0.5B-Instruct?

O modelo padrão é **Qwen/Qwen2.5-0.5B-Instruct**. A escolha está documentada em `config/finetune_defaults.py`; resumo:

- **Memória:** Colab costuma oferecer T4 (~15 GB). Com quantização 4-bit (QLoRA), o 0.5B cabe com folga; modelos maiores (7B+) exigiriam mais VRAM ou ambiente local.
- **Tamanho e tempo:** 0.5B parâmetros permite treino rápido e checkpoints menores, adequado ao escopo do desafio.
- **Instruction-tuned:** A variante Instruct já segue instruções e formato de diálogo, alinhada ao nosso formato (pergunta + contexto → resposta + decisão).
- **Alternativas:** Com mais VRAM (A100 ou máquina com 24 GB+), altere `MODEL_NAME` em `config/finetune_defaults.py` para, por exemplo, `Qwen2.5-1.5B-Instruct` ou `3B-Instruct`.

### Rodar o treino (linha de comando)

Requisitos: dados preparados (`data/train.jsonl`, `data/dev.jsonl`) e GPU com VRAM suficiente (recomendado 8+ GB; 4-bit recomendado para ~15 GB).

```bash
cd POSTECH-FIAP-FASE3
.venv/bin/python scripts/train_finetune.py
```

Opções úteis:

```bash
.venv/bin/python scripts/train_finetune.py --data-dir data --output-dir outputs/finetune_pqal --epochs 3
.venv/bin/python scripts/train_finetune.py --no-4bit   # precisão total (mais VRAM)
```

O **modelo final** (adaptador PEFT + tokenizer) é salvo em `outputs/finetune_pqal/` (ou no `--output-dir` indicado). Para inferência, carregue o modelo base e o adaptador desse diretório (veja exemplos no código em `src/models/`).

### Rodar no Google Colab

**Abrir o notebook:** No Cursor/VS Code, use a extensão **Jupyter** e abra `notebooks/finetune_medical_llm.ipynb` (clique no arquivo ou botão “Open in Notebook Editor”). Se abrir como JSON, clique com o botão direito no arquivo → **Open With** → **Jupyter Notebook**. No Colab: *File → Upload notebook* ou faça clone do repositório e abra o `.ipynb`.

1. Ative a GPU: *Runtime → Change runtime type → Hardware accelerator: GPU*.
2. **Recomendado:** Rode as células em ordem. A célula **"2. Clonar o repositório do GitHub"** clona o repo em `/content/POSTECH-FIAP-FASE3`. Se já houver `data/train.jsonl` e `data/dev.jsonl`, siga para o fine-tuning; senão, use a célula opcional para gerar a partir de `ori_pqal.json` ou faça upload manual.
3. Alternativa: coloque a pasta **POSTECH-FIAP-FASE3** no Google Drive e use a célula "Preparar ambiente" apontando para o Drive.
4. Execute as células: montar Drive (opcional), instalar dependências, **clonar o repo**, preparar ambiente (se necessário), rodar o treino e (opcional) copiar o modelo para o Drive.

O notebook usa a mesma pipeline do script e documenta no topo a escolha do modelo. Ao final, o modelo fine-tunado fica em `outputs/finetune_pqal/` (local ou no Drive, conforme configurado).

## Avaliação do modelo (Step 4)

Depois do fine-tuning, avalie no test set PubMedQA (Accuracy e Macro-F1):

```bash
cd POSTECH-FIAP-FASE3
.venv/bin/python scripts/run_evaluate.py
```

O script carrega o modelo em `outputs/finetune_pqal/` (ou `--model-dir`), roda inferência em `data/test.jsonl`, extrai a decisão (yes/no/maybe) de cada resposta e grava:
- `outputs/eval/predictions.json` — PMID → decisão (formato do evaluation.py do PubMedQA)
- `outputs/eval/metrics.json` — accuracy e macro_f1

Para usar só um arquivo de predições já gerado:

```bash
.venv/bin/python scripts/compute_metrics.py data/test_ground_truth.json outputs/eval/predictions.json
```

Se o modelo estiver em um checkpoint (ex.: `outputs/finetune_pqal/checkpoint-153/`), use `--model-dir outputs/finetune_pqal/checkpoint-153`.

## Assistente com LangChain (Step 5)

O assistente usa o modelo fine-tunado via LangChain: pergunta clínica → (opcional) contexto (abstracts) → LLM → resposta, com disclaimer e indicação de fonte.

```bash
# Primeiro exemplo do test set (pergunta + abstracts)
.venv/bin/python scripts/run_assistant.py

# Pergunta e contexto customizados
.venv/bin/python scripts/run_assistant.py --pergunta "Can the PHQ-9 assess depression in people with vision loss?" --contexto "The PHQ-9 was completed by 103 participants with low vision."
```

Documentação completa: **`docs/assistente_langchain_step5.md`** (componentes, exemplos de comando, segurança e explainability).

## Fluxos LangGraph (Step 6)

**Requisito:** Python 3.9 ou superior (o pacote LangGraph usa tipagem que não existe em 3.8).

O assistente também pode ser executado como um grafo LangGraph: pergunta → classificar intenção → buscar contexto (stub) → gerar resposta (mesmo modelo) → validar → log. O histórico de passos fica no estado (`historico`).

```bash
# Primeiro exemplo do test set
.venv/bin/python scripts/run_graph_assistant.py

# Pergunta e contexto customizados
.venv/bin/python scripts/run_graph_assistant.py --pergunta "Can the PHQ-9 assess depression in people with vision loss?" --contexto "The PHQ-9 was completed by 103 participants."

# Ver estrutura do grafo em ASCII
.venv/bin/python scripts/run_graph_assistant.py --draw --pergunta "Test?"
```

Para `--draw` / `draw_ascii()`, instale também **`grandalf`** (está em `requirements.txt`; sem ele ocorre `ImportError` pedindo `pip install grandalf`).

Documentação: **`docs/langgraph_step6.md`**.

**Rodar no Google Colab (com GPU):** use o notebook **`notebooks/run_graph_assistant_colab.ipynb`**. Abra no Colab, ative a GPU, clone o repositório, instale as dependências e aponte o caminho do modelo (ex.: `outputs/finetune_pqal` no clone ou no Drive).

## Próximos passos

1. ~~Preparar dados (preprocessamento, split).~~
2. ~~Fine-tuning da LLM (script + notebook Colab).~~
3. ~~Avaliação do modelo no test set (inferência + métricas).~~
4. ~~Assistente com LangChain (chain, script, documentação).~~
5. ~~Fluxos LangGraph (Step 6).~~
6. Ver `PLANO_DESENVOLVIMENTO_FASE3.md` para o plano completo e entregáveis.
