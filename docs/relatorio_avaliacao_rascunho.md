# Rascunho – Avaliação do modelo (relatório técnico)

**Uso:** copiar ou adaptar esta seção para o relatório técnico da Fase 3 (item “Avaliação” do checklist).  
**Contexto:** resultados obtidos com o modelo fine-tunado no PubMedQA (PQA-L), avaliado no **test set completo**. Métricas reportadas da execução **local** (melhor Accuracy); resultado equivalente no Colab em `outputs/eval_colab/`.

---

## Avaliação do modelo fine-tunado

### Configuração da avaliação

- **Modelo:** Qwen2.5-0.5B-Instruct com adaptador PEFT (LoRA) treinado no dataset PubMedQA em formato de instrução.
- **Test set:** PubMedQA (PQA-L), formato `data/test.jsonl` com `instruction`, `input` e `pmid`; ground truth em `data/test_ground_truth.json` (decisão final: yes / no / maybe).
- **Pipeline:** O script `scripts/run_evaluate.py` (ou o notebook `notebooks/evaluate_pqal_colab.ipynb`) carrega o modelo (base + adaptador), roda inferência exemplo a exemplo, extrai a decisão do texto gerado (padrão “Decision: yes/no/maybe”) e calcula Accuracy e Macro-F1 no mesmo formato do `evaluation.py` do PubMedQA.
- **Execução:** Avaliação no **test set completo** (~500 exemplos). Resultados principais da execução **local** em `outputs/eval/`; execução no Colab (GPU) em `outputs/eval_colab/` (Accuracy ligeiramente menor, Macro-F1 equivalente).

### Resultados (test set completo – execução local)

| Métrica    | Valor   |
|-----------|---------|
| Accuracy  | 0,628   |
| Macro-F1  | 0,461   |

*Referência Colab (mesmo test set):* Accuracy 0,604; Macro-F1 0,463.

As predições estão em `outputs/eval/predictions.json` (mapeamento PMID → yes/no/maybe). O Macro-F1 (~0,46) indica desempenho moderado e relativamente equilibrado entre as três classes (yes, no, maybe).

### Interpretação breve

- **Accuracy 0,628:** o modelo acertou a decisão em cerca de 63% dos exemplos do test set, demonstrando que o fine-tuning conferiu capacidade de responder no formato yes/no/maybe a perguntas do tipo PubMedQA.
- **Macro-F1 0,461:** reflete a média harmônica do F1 das três classes; valor moderado, com margem para melhoria via balanceamento no treino, mais épocas ou ajuste de decoding. Opcionalmente reportar matriz de confusão e F1 por classe no relatório final.
- **Reprodutibilidade:** para subconjunto rápido use `--max-samples N`; para test set completo use o script local ou o notebook no Colab (recomendado por tempo de inferência com GPU).

### Reprodução

```bash
# Na raiz do projeto, com o ambiente ativo:
.venv/bin/python scripts/run_evaluate.py --max-samples 50   # subconjunto (teste rápido)
.venv/bin/python scripts/run_evaluate.py                    # test set completo
```

Ou usar o notebook `notebooks/evaluate_pqal_colab.ipynb` no Google Colab (configurar `MODEL_DIR`, `DATA_DIR` e `MAX_SAMPLES`; saída em `outputs/eval/` ou copiar para `outputs/eval_colab/`).

---

*Resultados principais: `outputs/eval/metrics.json` e `outputs/eval/predictions.json` (execução local). Alternativa: `outputs/eval_colab/` (Colab).*
