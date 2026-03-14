"""
Defaults para fine-tuning (Step 3).
Ajuste MODEL_NAME, OUTPUT_DIR e hiperparâmetros conforme seu ambiente.
"""

# -----------------------------------------------------------------------------
# Por que Qwen2.5-0.5B-Instruct?
# -----------------------------------------------------------------------------
# Este modelo foi escolhido por motivos práticos do projeto (Tech Challenge
# Fase 3), priorizando que o pipeline rode no Google Colab e seja reproduzível:
#
# 1) Memória de GPU (Colab): No Colab a GPU típica é uma T4 com ~15 GB de VRAM.
#    Modelos maiores (7B+) em precisão total não cabem. O 0.5B é um modelo
#    "tiny" que, mesmo em fp16/bf16, cabe com folga. Com quantização 4-bit
#    (QLoRA), sobra memória para batch, gradientes e otimizador, permitindo
#    treinar sem OOM.
#
# 2) Tamanho e velocidade: 0.5B parâmetros implica treino mais rápido e menor
#    uso de disco ao salvar checkpoints e o adaptador LoRA. Adequado para
#    prototipagem e para demonstrar o pipeline de fine-tuning no prazo do
#    desafio.
#
# 3) Já instruction-tuned: A variante "Instruct" já foi treinada para seguir
#    instruções e usar formato de diálogo (user/assistant). Isso alinha bem
#    com nosso formato de dados (pergunta + contexto -> resposta + decisão)
#    e reduz o esforço de adaptação em relação a um modelo base puro.
#
# 4) Ecossistema: O TRL e a documentação Hugging Face usam frequentemente
#    a família Qwen em exemplos de SFT. Mantemos a mesma família para
#    compatibilidade e referência.
#
# Alternativas: Com mais VRAM (A100, V100 ou máquina local com 24 GB+), pode-se
# trocar para Qwen2.5-1.5B-Instruct ou 3B-Instruct para melhor qualidade.
# Outras opções pequenas: TinyLlama, SmolLM.
# -----------------------------------------------------------------------------

from pathlib import Path

# Modelo base (veja documentação acima)
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# Onde salvar checkpoints e adaptador PEFT
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # solution2
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "finetune_pqal"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"

# Dados
TRAIN_JSONL = "train.jsonl"
DEV_JSONL = "dev.jsonl"

# Treino
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4  # effective batch = 2*4 = 8
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_SEQ_LENGTH = 2048
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 10
SAVE_STRATEGY = "epoch"
SAVE_TOTAL_LIMIT = 2

# PEFT LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]  # Qwen2

# Precisão (4bit = QLoRA, para caber em Colab T4)
USE_4BIT = True
BNB_4BIT_COMPUTE_DTYPE = "bfloat16"
BNB_4BIT_QUANT_TYPE = "nf4"
