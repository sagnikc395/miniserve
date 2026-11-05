"""Drive concurrent load against the server and record per-request client
timings to JSONL (docs/MONITORING.md §8: client sees serialization + network,
so these numbers sit slightly above the server-side histograms).

Arrivals are Poisson (exponential inter-arrival, seed fixed) not fixed
concurrency, so queueing delay is measured.
"""

import asyncio
import json
import os
import random
import time

import httpx

BASE = "http://localhost:8000"
NUM_REQUESTS = int(os.environ.get("LOAD_REQUESTS", "24"))
MEAN_GAP_S = float(os.environ.get("LOAD_GAP", "0.35"))

PROMPTS = [
    "Explain paged attention in one paragraph.",
    "What is continuous batching in inference servers?",
    "Summarize how a paged KV cache reuses pages across requests.",
    "Why does chunked prefill help latency for long prompts?",
    "Describe prefix caching for a shared system prompt.",
    "What is the difference between TTFT and TPOT?",
]

STREAMING = ["Explain paged attention in one paragraph.", "List three uses of prefix caching."]


async def chat(client: httpx.AsyncClient, prompt: str, stream: bool, max_tokens: int = None):
    rid = f"lr-{random.randrange(16**6):06x}"
    body = {
        "model": "Qwen/Qwen3-0.6B",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens or int(os.environ.get("LOAD_MAX_TOKENS", "96")),
        "temperature": 0.6,
    }
    if stream:
        body["stream"] = True
    arrived = time.perf_counter()
    first_tok = None
    n_chunks = 0
    text = ""
    usage = None
    async with client.stream("POST", "/v1/chat/completions", json=body) as r:
        r.raise_for_status()
        if not stream:
            data = (await r.aread()).decode()
            usage = json.loads(data).get("usage")
        else:
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                chunk = json.loads(payload)
                delta = (chunk["choices"][0].get("delta") or {}).get("content")
                if delta is not None:
                    if first_tok is None:
                        first_tok = time.perf_counter()
                    text += delta
                    n_chunks += 1
                usage = chunk.get("usage")
    done = time.perf_counter()
    return {
        "request_id": rid,
        "stream": stream,
        "arrival": arrived,
        "first_token": first_tok,
        "done": done,
        "ttft": first_tok - arrived if first_tok else None,
        "e2e": done - arrived,
        "num_stream_chunks": n_chunks,
        "sample": text[:120],
        "usage": usage,
    }


async def main():
    rng = random.Random(0)
    results = []
    async with httpx.AsyncClient(base_url=BASE, timeout=120) as client:
        t0 = time.perf_counter()
        tasks = []
        for i in range(NUM_REQUESTS):
            gap = rng.expovariate(1 / MEAN_GAP_S)
            prompt = PROMPTS[i % len(PROMPTS)]
            stream = prompt in STREAMING
            tasks.append(
                _delayed(client, prompt, stream, gap)
            )
        results = await asyncio.gather(*tasks)

    with open("output/load-run.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    ttfts = sorted(r["ttft"] for r in results if r["ttft"] is not None)
    e2es = sorted(r["e2e"] for r in results)
    span = results[-1]["done"] - t0
    tok = sum((r["usage"] or {}).get("completion_tokens", 0) for r in results)
    print(f"requests={len(results)} wall={span:.2f}s out_tok={tok} out_tok/s={tok / span:.1f}")
    if ttfts:
        print(f"client ttft p50/p95: {ttfts[len(ttfts)//2]:.3f}s / {ttfts[int(len(ttfts)*0.95)]:.3f}s")
    print(f"client e2e p50/p95: {e2es[len(e2es)//2]:.3f}s / {e2es[int(len(e2es)*0.95)]:.3f}s")


async def _delayed(client, prompt, stream, delay):
    await asyncio.sleep(delay)
    return await chat(client, prompt, stream)


if __name__ == "__main__":
    asyncio.run(main())
