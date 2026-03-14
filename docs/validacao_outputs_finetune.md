# Validação dos outputs do fine-tuning

**Pasta:** `outputs/finetune_pqal/`  
**Data da checagem:** com base nos arquivos presentes no repositório.

---

## 1. Estrutura esperada

Após rodar o Step 4 (fine-tuning), a pasta de saída deve conter:

| Item | Descrição | Status |
|------|-----------|--------|
| **adapter_config.json** | Configuração PEFT/LoRA (base_model, r, alpha, target_modules) | ✅ Presente |
| **adapter_model.safetensors** (ou .bin) | Pesos do adaptador LoRA | ⚠️ Ver nota abaixo |
| **tokenizer.json** | Tokenizer do modelo base | ✅ Presente |
| **tokenizer_config.json** | Config do tokenizer | ✅ Presente |
| **trainer_state.json** | Histórico de treino (loss, steps, epochs) | ✅ Presente |
| **README.md** | Model card (base model, TRL/PEFT versions) | ✅ Presente |
| **checkpoint-XXX/** | Checkpoints intermediários (save_strategy=epoch) | ✅ checkpoint-102, checkpoint-153 |

---

## 2. Conteúdo validado

### adapter_config.json

- **base_model_name_or_path:** `Qwen/Qwen2.5-0.5B-Instruct` ✅
- **peft_type:** `LORA` ✅
- **r:** 16, **lora_alpha:** 32, **lora_dropout:** 0.05 ✅
- **target_modules:** `["k_proj", "o_proj", "q_proj", "v_proj"]` ✅
- **task_type:** `CAUSAL_LM` ✅

Configuração compatível com o definido em `config/finetune_defaults.py`.

### trainer_state.json

- **num_train_epochs:** 3 ✅
- **global_step:** 153 ✅
- **train_loss (final):** ~2.296
- **mean_token_accuracy (final):** ~0.564
- **train_runtime:** ~910 s (~15 min)
- **train_samples_per_second:** ~1.32

**Curva de loss (resumo):**

- Step 10: loss ≈ 2.73, mean_token_accuracy ≈ 0.48  
- Step 50: loss ≈ 2.40, mean_token_accuracy ≈ 0.52  
- Step 100: loss ≈ 2.20, mean_token_accuracy ≈ 0.53  
- Step 153 (fim): loss ≈ 2.30 (média final), mean_token_accuracy ≈ 0.56  

A loss diminui ao longo do treino e a acurácia por token sobe, indicando que o modelo está aprendendo. O treino foi concluído até o fim das 3 épocas.

### README.md (model card)

- **base_model:** Qwen/Qwen2.5-0.5B-Instruct ✅
- **library_name:** peft ✅
- **tags:** lora, sft, transformers, trl ✅
- **Framework versions:** PEFT 0.18.1, TRL 0.29.0, Transformers 5.0.0 ✅

---

## 3. Checkpoints

- **checkpoint-102** e **checkpoint-153** contêm: `adapter_config.json`, `tokenizer*.json`, `trainer_state.json`, `README.md`, `chat_template.jinja`.  
- O checkpoint-153 corresponde ao final do treino (último step). Para inferência, pode-se usar `outputs/finetune_pqal/` (modelo final) ou `outputs/finetune_pqal/checkpoint-153/`.

---

## 4. Uso para inferência

Para carregar o modelo fine-tunado:

1. **Modelo base:** `Qwen/Qwen2.5-0.5B-Instruct`
2. **Adaptador:** pasta `outputs/finetune_pqal/` (ou `outputs/finetune_pqal/checkpoint-153/`)

Exemplo com PEFT:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", device_map="auto")
model = PeftModel.from_pretrained(base, "outputs/finetune_pqal")
tokenizer = AutoTokenizer.from_pretrained("outputs/finetune_pqal")
```

**Nota sobre adapter_model.safetensors:** O `trainer.save_model()` do TRL/PEFT normalmente grava os pesos do adaptador em `adapter_model.safetensors` (ou `adapter_model.bin`) na mesma pasta. Se essa pasta foi copiada do Colab/Drive, confira se esse arquivo está incluído. Sem ele, o carregamento com `PeftModel.from_pretrained(..., "outputs/finetune_pqal")` falhará; nesse caso use uma das pastas de checkpoint (ex.: `checkpoint-153/`) que contenha o arquivo de pesos do adaptador.

---

## 5. Conclusão

- A **estrutura** da pasta e o **conteúdo** de `adapter_config.json`, `trainer_state.json` e `README.md` estão **consistentes** com um fine-tuning LoRA concluído (3 épocas, loss decrescente, acurácia por token subindo).
- Os **resultados de treino** são plausíveis para o tamanho do dataset e do modelo.
- Para **inferência**, usar `outputs/finetune_pqal/` (ou um checkpoint) garantindo que o arquivo de pesos do adaptador (`adapter_model.safetensors` ou equivalente) esteja presente na pasta.
