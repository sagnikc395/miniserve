"""Hugging Face `generate` baseline: static batching, no paged cache."""

import gc
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from miniserve.benchmarks.prompts import build_prompts
from miniserve.platform import empty_device_cache


def _percentile(values, p):
    return float(np.percentile(values, p)) if values else 0.0


def run_hf_baseline(
    model_name: str,
    device,
    dtype,
    num_prompts: int = 16,
    max_tokens: int = 64,
    batch_size: int = 8,
) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, low_cpu_mem_usage=True
    )
    model.to(device)
    model.eval()

    prompts = build_prompts(num_prompts, tokenizer)
    latencies = []
    prompt_tokens = 0
    generated_tokens = 0
    padded_prompt_tokens = 0

    start = time.perf_counter()
    for offset in range(0, len(prompts), batch_size):
        batch = prompts[offset : offset + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        batch_start = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        batch_end = time.perf_counter()

        input_len = inputs["input_ids"].shape[1]
        generated = output[:, input_len:]
        new_tokens = int((generated != tokenizer.pad_token_id).sum().item())

        prompt_tokens += int(inputs["attention_mask"].sum().item())
        padded_prompt_tokens += int(inputs["input_ids"].numel())
        generated_tokens += new_tokens

        # static batching: every row finishes when the slowest row does
        per_row = (batch_end - batch_start) / len(batch)
        latencies.extend([per_row] * len(batch))

    elapsed = time.perf_counter() - start

    del model
    gc.collect()
    empty_device_cache(device)

    return {
        "name": "hf static batching",
        "requests": len(prompts),
        "wall_s": elapsed,
        "req_per_s": len(prompts) / elapsed if elapsed else 0.0,
        "output_tok_per_s": generated_tokens / elapsed if elapsed else 0.0,
        "total_tok_per_s": (prompt_tokens + generated_tokens) / elapsed
        if elapsed
        else 0.0,
        "p50_latency_s": _percentile(latencies, 50),
        "p90_latency_s": _percentile(latencies, 90),
        "prompt_tokens": prompt_tokens,
        "padded_prompt_tokens": padded_prompt_tokens,
        "generated_tokens": generated_tokens,
        "cached_tokens": 0,
    }
