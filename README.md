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

## Ambiente

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# ou: .venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Dataset

- **PubMedQA (PQA-L):** `data/ori_pqal.json` — 1000 instâncias (pergunta, contextos, resposta, decisão yes/no/maybe).

### Preparação (Step 2)

Gera train/dev/test em formato de instrução e dataset anonimizado para entrega:

```bash
cd POSTECH-FIAP-FASE3
python scripts/run_prepare_data.py
```

Ou com opções:

```bash
python -m src.data.prepare_pqal --data-dir data --format instruction --dev-ratio 0.2
```

**Saídas em `data/`:**
- `train.jsonl`, `dev.jsonl` — fine-tuning (instruction/input/output + final_decision)
- `test.jsonl`, `test_set.json`, `test_ground_truth.json` — avaliação
- `data/anonymized/train_dev_anonymized.jsonl` — dataset anonimizado (IDs sintéticos) para entrega

### Verificação dos dados preparados

O script `scripts/verify_data.py` valida a consistência dos arquivos gerados pela preparação (contagens, formato, ausência de sobreposição entre train/dev/test e conformidade do dataset anonimizado). Útil para quem for avaliar o projeto conferir se os dados estão corretos após rodar a preparação.

```bash
cd POSTECH-FIAP-FASE3
python scripts/verify_data.py
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
python scripts/train_finetune.py
```

Opções úteis:

```bash
python scripts/train_finetune.py --data-dir data --output-dir outputs/finetune_pqal --epochs 3
python scripts/train_finetune.py --no-4bit   # precisão total (mais VRAM)
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
python scripts/run_evaluate.py
```

O script carrega o modelo em `outputs/finetune_pqal/` (ou `--model-dir`), roda inferência em `data/test.jsonl`, extrai a decisão (yes/no/maybe) de cada resposta e grava:
- `outputs/eval/predictions.json` — PMID → decisão (formato do evaluation.py do PubMedQA)
- `outputs/eval/metrics.json` — accuracy e macro_f1

Para usar só um arquivo de predições já gerado:

```bash
python scripts/compute_metrics.py data/test_ground_truth.json outputs/eval/predictions.json
```

Se o modelo estiver em um checkpoint (ex.: `outputs/finetune_pqal/checkpoint-153/`), use `--model-dir outputs/finetune_pqal/checkpoint-153`.

## Assistente com LangChain (Step 5)

O assistente usa o modelo fine-tunado via LangChain: pergunta clínica → (opcional) contexto (abstracts) → LLM → resposta, com disclaimer e indicação de fonte.

```bash
# Primeiro exemplo do test set (pergunta + abstracts)
python scripts/run_assistant.py

# Pergunta e contexto customizados
python scripts/run_assistant.py --pergunta "Can the PHQ-9 assess depression in people with vision loss?" --contexto "The PHQ-9 was completed by 103 participants with low vision."
```

Documentação completa: **`docs/assistente_langchain_step5.md`** (componentes, exemplos de comando, segurança e explainability).

## Próximos passos

1. ~~Preparar dados (preprocessamento, split).~~
2. ~~Fine-tuning da LLM (script + notebook Colab).~~
3. ~~Avaliação do modelo no test set (inferência + métricas).~~
4. ~~Assistente com LangChain (chain, script, documentação).~~
5. Fluxos LangGraph (Step 6).
6. Ver `PLANO_DESENVOLVIMENTO_FASE3.md` para o plano completo e entregáveis.
