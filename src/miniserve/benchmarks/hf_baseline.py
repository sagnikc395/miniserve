"""Hugging Face `generate` baseline: static batching, no paged cache."""

import gc
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from miniserve.benchmarks.metrics import per_second, percentile
from miniserve.benchmarks.prompts import build_prompts
from miniserve.platform import empty_device_cache, synchronize


def _stop_ids(tokenizer, model) -> set[int]:
    ids = {tokenizer.eos_token_id}
    eos = getattr(model.generation_config, "eos_token_id", None)
    if eos is not None:
        ids |= set(eos if isinstance(eos, list) else [eos])
    ids.discard(None)
    return ids


def _new_token_counts(generated: torch.Tensor, stop_ids: set[int]) -> list[int]:
    """Tokens produced per row, cut at the first stop token (inclusive).

    `generate` right-pads rows that stopped early, and the pad token is usually
    EOS itself, so counting `!= pad_id` would also drop the real EOS tokens.
    """
    counts = []
    for row in generated.tolist():
        count = len(row)
        for i, token in enumerate(row):
            if token in stop_ids:
                count = i + 1
                break
        counts.append(count)
    return counts


def _generate(model, tokenizer, batch, device, max_new_tokens):
    inputs = tokenizer(batch, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    synchronize(device)
    return inputs, output


def run_hf_baseline(
    model_name: str,
    device,
    dtype,
    num_prompts: int = 16,
    max_tokens: int = 64,
    batch_size: int = 8,
) -> dict:
    batch_size = max(1, batch_size)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError(f"{model_name} has neither a pad nor an eos token")
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, low_cpu_mem_usage=True
    )
    model.to(device)
    model.eval()
    stop_ids = _stop_ids(tokenizer, model)

    prompts = build_prompts(num_prompts, tokenizer)
    latencies = []
    prompt_tokens = 0
    generated_tokens = 0
    padded_prompt_tokens = 0

    # warm up so lazy kernel compilation is not charged to the first batch
    _generate(model, tokenizer, prompts[:1], device, 4)

    start = time.perf_counter()
    for offset in range(0, len(prompts), batch_size):
        batch = prompts[offset : offset + batch_size]

        inputs, output = _generate(model, tokenizer, batch, device, max_tokens)
        batch_end = time.perf_counter()

        input_len = inputs["input_ids"].shape[1]
        generated = output[:, input_len:]

        prompt_tokens += int(inputs["attention_mask"].sum().item())
        padded_prompt_tokens += int(inputs["input_ids"].numel())
        generated_tokens += sum(_new_token_counts(generated, stop_ids))

        # static batching: every row in the batch waits for the slowest row,
        # and all prompts arrived at `start`, so that is each row's latency
        latencies.extend([batch_end - start] * len(batch))

    elapsed = time.perf_counter() - start

    del model
    gc.collect()
    empty_device_cache(device)

    return {
        "name": "hf static batching",
        "requests": len(prompts),
        "unfinished": 0,
        "wall_s": elapsed,
        "req_per_s": per_second(len(prompts), elapsed),
        "output_tok_per_s": per_second(generated_tokens, elapsed),
        "total_tok_per_s": per_second(prompt_tokens + generated_tokens, elapsed),
        "p50_latency_s": percentile(latencies, 50),
        "p90_latency_s": percentile(latencies, 90),
        "p50_ttft_s": None,  # `generate` gives no per-token timing
        "prompt_tokens": prompt_tokens,
        "padded_prompt_tokens": padded_prompt_tokens,
        "generated_tokens": generated_tokens,
        "cached_tokens": 0,
        "preemptions": 0,
    }
