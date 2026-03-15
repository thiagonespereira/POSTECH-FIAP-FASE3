"""
Carrega o modelo fine-tunado (base + PEFT) e expõe como LLM do LangChain.

Usado pelo assistente médico (Step 5): pipeline text-generation → HuggingFacePipeline.
"""
from pathlib import Path
import json
import sys
import types
from typing import Optional

# Stub triton.ops se ausente (bitsandbytes 0.42 exige triton.ops.matmul_perf_model; triton novo removeu)
def _ensure_triton_ops_stub():
    try:
        from triton.ops.matmul_perf_model import early_config_prune, estimate_matmul_time  # noqa: F401
        return
    except (ImportError, ModuleNotFoundError):
        pass
    triton = sys.modules.get("triton")
    if triton is None:
        triton = types.ModuleType("triton")
        sys.modules["triton"] = triton
    if not hasattr(triton, "ops"):
        triton_ops = types.ModuleType("triton.ops")
        triton.ops = triton_ops
        sys.modules["triton.ops"] = triton_ops
    if "triton.ops.matmul_perf_model" not in sys.modules:
        matmul_perf = types.ModuleType("triton.ops.matmul_perf_model")
        matmul_perf.early_config_prune = lambda *a, **k: None
        matmul_perf.estimate_matmul_time = lambda *a, **k: 0.0
        sys.modules["triton.ops.matmul_perf_model"] = matmul_perf


_ensure_triton_ops_stub()

import torch


def load_llm_langchain(
    model_dir: Path,
    *,
    max_new_tokens: int = 256,
    device_map: str = "auto",
) -> "BaseLLM":
    """
    Carrega o modelo (base + adaptador PEFT) e o tokenizer, monta pipeline
    text-generation e retorna um LLM compatível com LangChain.

    model_dir: pasta com adapter_config.json, adapter_model.safetensors (ou .bin) e tokenizer.
    Retorna: instância de HuggingFacePipeline (langchain_community.llms).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from peft import PeftModel

    model_dir = Path(model_dir)
    adapter_config_path = model_dir / "adapter_config.json"
    if adapter_config_path.exists():
        with open(adapter_config_path) as f:
            adapter_config = json.load(f)
        base_model_name = adapter_config.get("base_model_name_or_path", "Qwen/Qwen2.5-0.5B-Instruct")
    else:
        base_model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    # Tokenizer do modelo base (model_dir pode ter só adapter; tokenizer completo vem da base)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    try:
        model = PeftModel.from_pretrained(base_model, model_dir)
    except TypeError as e:
        if "unexpected keyword argument" in str(e):
            import tempfile
            import shutil
            with open(model_dir / "adapter_config.json") as f:
                cfg = json.load(f)
            safe_keys = {
                "base_model_name_or_path", "peft_type", "r", "lora_alpha", "lora_dropout",
                "target_modules", "bias", "task_type", "inference_mode", "init_lora_weights",
            }
            cfg_safe = {k: v for k, v in cfg.items() if k in safe_keys}
            tmp = Path(tempfile.mkdtemp())
            for fname in ("adapter_model.safetensors", "adapter_model.bin"):
                src = model_dir / fname
                if src.exists():
                    shutil.copy(src, tmp / fname)
                    break
            with open(tmp / "adapter_config.json", "w") as f:
                json.dump(cfg_safe, f, indent=2)
            model = PeftModel.from_pretrained(base_model, tmp)
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            raise
    model.eval()

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )

    from langchain_community.llms import HuggingFacePipeline
    return HuggingFacePipeline(pipeline=pipe)
