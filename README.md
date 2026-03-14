# Solution2 – Assistente médico (Tech Challenge IADT Fase 3)

Assistente virtual médico com LLM fine-tunada em dados clínicos (PubMedQA) e integração LangChain/LangGraph.

## Estrutura do projeto

```
solution2/
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
cd solution2
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
cd solution2
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
cd solution2
python scripts/train_finetune.py
```

Opções úteis:

```bash
python scripts/train_finetune.py --data-dir data --output-dir outputs/finetune_pqal --epochs 3
python scripts/train_finetune.py --no-4bit   # precisão total (mais VRAM)
```

O **modelo final** (adaptador PEFT + tokenizer) é salvo em `outputs/finetune_pqal/` (ou no `--output-dir` indicado). Para inferência, carregue o modelo base e o adaptador desse diretório (veja exemplos no código em `src/models/`).

### Rodar no Google Colab

1. Ative a GPU: *Runtime → Change runtime type → Hardware accelerator: GPU*.
2. Coloque a pasta **solution2** no Google Drive (com `data/`, `src/`, `config/`, `scripts/` e os arquivos `train.jsonl` e `dev.jsonl` em `data/`).
3. Abra o notebook **`notebooks/finetune_medical_llm.ipynb`** no Colab (upload do arquivo ou clone do repositório).
4. Execute as células: montar Drive, instalar dependências, copiar solution2 para `/content`, rodar o treino, e (opcional) copiar a pasta de saída de volta para o Drive.

O notebook usa a mesma pipeline do script e documenta no topo a escolha do modelo. Ao final, o modelo fine-tunado fica em `outputs/finetune_pqal/` (local ou no Drive, conforme configurado).

## Próximos passos

1. ~~Preparar dados (preprocessamento, split).~~
2. ~~Fine-tuning da LLM (script + notebook Colab).~~
3. Assistente com LangChain e fluxos LangGraph.
4. Ver `PLANO_DESENVOLVIMENTO_FASE3.md` para o plano completo e entregáveis.
