#!/bin/bash
# Teste do assistente: roda e grava saída em outputs/eval/assistant_test_output.txt
# Uso: ./scripts/run_assistant_test.sh   ou   bash scripts/run_assistant_test.sh
set -e
cd "$(dirname "$0")/.."
OUT=outputs/eval
mkdir -p "$OUT"
echo "Rodando assistente (primeiro exemplo do test set). Saída em $OUT/assistant_test_output.txt"
.venv/bin/python -u scripts/run_assistant.py 2>&1 | tee "$OUT/assistant_test_output.txt"
echo "Concluído. Ver: $OUT/assistant_test_output.txt"
