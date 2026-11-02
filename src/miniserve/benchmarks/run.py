"""`main.py bench`: miniserve with/without prefix caching, plus HF baseline."""

import dataclasses
import gc
import time

import numpy as np

from miniserve.benchmarks.prompts import build_prompts
from miniserve.config import EngineConfig
from miniserve.engine.engine import Engine
from miniserve.platform import empty_device_cache, synchronize
from miniserve.sampling import SamplingParams


def _percentile(values, p):
    return float(np.percentile(values, p)) if values else 0.0


def _bench_engine(cfg: EngineConfig, num_prompts: int, max_tokens: int) -> dict:
    engine = Engine(cfg)
    prompts = build_prompts(num_prompts, engine.tokenizer)

    start = time.perf_counter()
    arrivals = {}
    for prompt in prompts:
        rid = engine.add_request(
            prompt, SamplingParams(temperature=0.0, max_tokens=max_tokens)
        )
        arrivals[rid] = time.perf_counter()

    first = {}
    last = {}
    final = {}
    while engine.has_work():
        for out in engine.step():
            now = time.perf_counter()
            first.setdefault(out.request_id, now)
            last[out.request_id] = now
            if out.finished:
                final[out.request_id] = out

    synchronize(cfg.device)
    elapsed = time.perf_counter() - start

    latencies = [last[rid] - arrivals[rid] for rid in arrivals if rid in last]
    ttfts = [first[rid] - arrivals[rid] for rid in arrivals if rid in first]
    generated = sum(len(out.token_ids) for out in final.values())
    prompt_tokens = sum(out.num_prompt_tokens for out in final.values())
    cached = sum(out.num_cached_tokens for out in final.values())

    stats = engine.stats()
    del engine
    gc.collect()
    empty_device_cache(cfg.device)

    return {
        "name": "miniserve prefix cache" if cfg.enable_prefix_caching else "miniserve",
        "requests": len(prompts),
        "wall_s": elapsed,
        "req_per_s": len(prompts) / elapsed if elapsed else 0.0,
        "output_tok_per_s": generated / elapsed if elapsed else 0.0,
        "total_tok_per_s": (prompt_tokens + generated) / elapsed if elapsed else 0.0,
        "p50_latency_s": _percentile(latencies, 50),
        "p90_latency_s": _percentile(latencies, 90),
        "p50_ttft_s": _percentile(ttfts, 50),
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated,
        "cached_tokens": cached,
        "preemptions": stats["preemptions"],
    }


def _print_table(results: list[dict]) -> None:
    header = (
        f"{'run':<24} {'req/s':>8} {'out tok/s':>10} {'tot tok/s':>10} "
        f"{'p50 lat':>9} {'p90 lat':>9} {'cached':>8}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        cached_pct = (
            100.0 * result["cached_tokens"] / result["prompt_tokens"]
            if result["prompt_tokens"]
            else 0.0
        )
        print(
            f"{result['name']:<24} {result['req_per_s']:>8.2f} "
            f"{result['output_tok_per_s']:>10.2f} "
            f"{result['total_tok_per_s']:>10.2f} "
            f"{result['p50_latency_s']:>9.3f} "
            f"{result['p90_latency_s']:>9.3f} "
            f"{cached_pct:>7.1f}%"
        )


def run_benchmark(
    cfg: EngineConfig,
    num_prompts: int = 16,
    max_tokens: int = 64,
    hf_baseline: bool = False,
) -> list[dict]:
    results = []

    print(
        f"[bench] {num_prompts} prompts, max_tokens={max_tokens}, "
        f"model={cfg.model}, device={cfg.device}, backend={cfg.attention_backend}"
    )

    results.append(_bench_engine(cfg, num_prompts, max_tokens))

    no_cache = dataclasses.replace(cfg, enable_prefix_caching=False)
    results.append(_bench_engine(no_cache, num_prompts, max_tokens))

    if hf_baseline:
        try:
            from miniserve.benchmarks.hf_baseline import run_hf_baseline

            results.append(
                run_hf_baseline(
                    cfg.model,
                    cfg.device,
                    cfg.dtype,
                    num_prompts=num_prompts,
                    max_tokens=max_tokens,
                    batch_size=cfg.max_num_seqs,
                )
            )
        except Exception as exc:
            print(f"[bench] HF baseline skipped: {exc}")

    _print_table(results)
    return results
