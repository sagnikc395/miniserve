"""`main.py bench`: miniserve with/without prefix caching, plus HF baseline."""

import dataclasses
import gc
import time

from miniserve.benchmarks.metrics import per_second, percentile
from miniserve.benchmarks.prompts import build_prompts
from miniserve.config import EngineConfig
from miniserve.engine.engine import Engine
from miniserve.platform import empty_device_cache, synchronize
from miniserve.sampling import SamplingParams

# A step produces no output while every scheduled request is still mid-prompt
# (chunked prefill), so empty steps are normal. Many in a row mean the
# scheduler can no longer place work, and `has_work()` would spin forever.
_MAX_EMPTY_STEPS = 1000

# short, deliberately unrelated to the benchmark prompts so it cannot seed the
# prefix cache for the run we are about to measure
_WARMUP_PROMPT = "Say hello."


def _drain(engine: Engine) -> None:
    while engine.has_work():
        engine.step()


def _warmup(engine: Engine) -> None:
    """Run one tiny request so lazy kernel compilation is not timed."""
    engine.add_request(_WARMUP_PROMPT, SamplingParams(temperature=0.0, max_tokens=4))
    _drain(engine)
    synchronize(engine.cfg.device)


def _bench_engine(cfg: EngineConfig, num_prompts: int, max_tokens: int) -> dict:
    engine = Engine(cfg)
    prompts = build_prompts(num_prompts, engine.tokenizer)
    _warmup(engine)

    start = time.perf_counter()
    arrivals = {}
    for prompt in prompts:
        rid = engine.add_request(
            prompt, SamplingParams(temperature=0.0, max_tokens=max_tokens)
        )
        arrivals[rid] = time.perf_counter()

    first = {}
    last = {}
    latest = {}  # most recent Output per request, finished or not
    empty_steps = 0
    while engine.has_work():
        outputs = engine.step()
        if not outputs:
            empty_steps += 1
            if empty_steps > _MAX_EMPTY_STEPS:
                raise RuntimeError(
                    f"engine stalled: {_MAX_EMPTY_STEPS} steps without output "
                    f"({len(arrivals) - len(latest)} requests never ran). "
                    "Try a larger --gpu-memory-utilization or fewer prompts."
                )
            continue

        empty_steps = 0
        now = time.perf_counter()
        for out in outputs:
            first.setdefault(out.request_id, now)
            last[out.request_id] = now
            latest[out.request_id] = out

    synchronize(cfg.device)
    elapsed = time.perf_counter() - start

    latencies = [last[rid] - arrivals[rid] for rid in arrivals if rid in last]
    ttfts = [first[rid] - arrivals[rid] for rid in arrivals if rid in first]
    generated = sum(len(out.token_ids) for out in latest.values())
    prompt_tokens = sum(out.num_prompt_tokens for out in latest.values())
    cached = sum(out.num_cached_tokens for out in latest.values())
    unfinished = len(arrivals) - sum(1 for out in latest.values() if out.finished)

    preemptions = engine.stats()["preemptions"]
    del engine
    gc.collect()
    empty_device_cache(cfg.device)

    return {
        "name": "miniserve prefix cache" if cfg.enable_prefix_caching else "miniserve",
        "requests": len(prompts),
        "unfinished": unfinished,
        "wall_s": elapsed,
        "req_per_s": per_second(len(prompts), elapsed),
        "output_tok_per_s": per_second(generated, elapsed),
        "total_tok_per_s": per_second(prompt_tokens + generated, elapsed),
        "p50_latency_s": percentile(latencies, 50),
        "p90_latency_s": percentile(latencies, 90),
        "p50_ttft_s": percentile(ttfts, 50),
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated,
        "cached_tokens": cached,
        "preemptions": preemptions,
    }


def _cell(value, width: int, fmt: str) -> str:
    if value is None:
        return f"{'-':>{width}}"
    return f"{value:>{width}{fmt}}"


def _print_table(results: list[dict]) -> None:
    header = (
        f"{'run':<24} {'req/s':>8} {'out tok/s':>10} {'tot tok/s':>10} "
        f"{'p50 ttft':>9} {'p50 lat':>9} {'p90 lat':>9} {'cached':>8}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        prompt_tokens = result.get("prompt_tokens", 0)
        cached_pct = (
            100.0 * result["cached_tokens"] / prompt_tokens if prompt_tokens else 0.0
        )
        print(
            f"{result['name']:<24} "
            f"{_cell(result.get('req_per_s'), 8, '.2f')} "
            f"{_cell(result.get('output_tok_per_s'), 10, '.2f')} "
            f"{_cell(result.get('total_tok_per_s'), 10, '.2f')} "
            f"{_cell(result.get('p50_ttft_s'), 9, '.3f')} "
            f"{_cell(result.get('p50_latency_s'), 9, '.3f')} "
            f"{_cell(result.get('p90_latency_s'), 9, '.3f')} "
            f"{cached_pct:>7.1f}%"
        )
        if result.get("unfinished"):
            print(
                f"  ! {result['unfinished']} of {result['requests']} requests did not finish"
            )


def run_benchmark(
    cfg: EngineConfig,
    num_prompts: int = 16,
    max_tokens: int = 64,
    hf_baseline: bool = False,
) -> list[dict]:
    if num_prompts < 1:
        raise ValueError("num_prompts must be at least 1")
    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1")

    results = []

    print(
        f"[bench] {num_prompts} prompts, max_tokens={max_tokens}, "
        f"model={cfg.model}, device={cfg.device}, backend={cfg.attention_backend}"
    )

    # run both arms explicitly: `--no-prefix-caching` must not silently
    # benchmark the same configuration twice
    for enabled in (True, False):
        arm = dataclasses.replace(cfg, enable_prefix_caching=enabled)
        results.append(_bench_engine(arm, num_prompts, max_tokens))

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
