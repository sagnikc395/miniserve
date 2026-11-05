"""Repro of the broken long-context burst shape: 32 chat completions, gap 0.3s,
long prompts, non-stream, 240s timeout; prints which complete and when."""

import asyncio
import json
import random
import time

import httpx


def long_prompt(i):
    topic = (
        "Consider a serving stack with paged KV cache, continuous batching and "
        "chunked prefill. The scheduler admits requests while KV blocks remain, "
        "chunks prompts into the token budget, and preempts the newest sequence "
        "on pressure. "
    )
    rng = random.Random(1234 + i)
    body = topic * (4600 // len(topic) + 1)
    fill = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz ") for _ in range(32))
    return f"Read the following design document. In your reply, state one risk.\n\n{body[:4600]}\n\nIgnore: {fill}. Now summarize."


async def chat(client, i, t0):
    arrived = time.perf_counter()
    body = {
        "model": "Qwen/Qwen3-0.6B",
        "messages": [{"role": "user", "content": long_prompt(i)}],
        "max_tokens": 128,
    }
    try:
        r = await client.post("/v1/chat/completions", json=body)
        r.raise_for_status()
        done = time.perf_counter()
        u = r.json()["usage"]
        print(f"#{i:02} OK   wait={done - arrived:6.1f}s prompt={u['prompt_tokens']} out={u['completion_tokens']}", flush=True)
    except Exception as e:
        print(f"#{i:02} FAIL after {time.perf_counter() - arrived:6.1f}s {type(e).__name__}", flush=True)


async def main():
    rng = random.Random(0)
    t0 = time.perf_counter()
    async with httpx.AsyncClient(
        base_url="http://localhost:8000", timeout=httpx.Timeout(240)
    ) as client:
        async def delayed(i):
            await asyncio.sleep(rng.expovariate(1 / 0.3))
            return await chat(client, i, t0)

        await asyncio.gather(*[delayed(i) for i in range(32)])
    print(f"total {time.perf_counter() - t0:.1f}s")

asyncio.run(main())
