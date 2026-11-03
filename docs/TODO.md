# TODO

Open work across the repo. Background on the macOS constraints is in [MPS.md](MPS.md),
the design walkthrough is in [FLOW.md](FLOW.md), the full project reference is
[MINISERVE.md](../MINISERVE.md), and the architecture diagrams are in
[README.md](../README.md); this file is the actionable list.

Measurements below were taken on: M4, 24 GB, macOS 26.5, torch 2.14.1, Python 3.13.

## What runs today

| Path | Status |
|---|---|
| `uv run main.py attn-demo` | works, matches naive attention to ~2e-7 |
| `uv run main.py kv-demo` | works |
| `uv run pytest` | 10 passed (all `KVCacheManager`) |
| `uv run main.py generate` | works, synchronous engine, streams to stdout |
| `uv run main.py serve` | works, OpenAI-compatible, SSE streaming |
| `uv run main.py bench` | works, both miniserve arms + optional HF baseline |
| `uv run tui.py` | works, `--spawn` owns the server |
| loading a real checkpoint | works (dense + MoE, fused qkv / gate_up) |

The end-to-end path is complete. What is left is **performance** (Part A),
**test coverage** (Part B), and a short list of loose ends (Part C).

---

# Part A — attention performance

This is where the remaining wins are. The repo is correct and slow.

## 1. ~~The `sdpa` backend is an alias, and unreachable on Mac~~ — done

`paged_sdpa_attention()` now returns a real closure that is distinct from
`torch_ref_attention`, and `EngineConfig.__post_init__` selects `sdpa` on anything
that is not CPU (CPU keeps `torch` because it is easier to debug). The split is the
one the README flowchart calls for:

- uniform query length (decode, or uniform speculative verification) ->
  `_batched_attention`, one SDPA over the whole batch
- mixed prefill/decode -> `torch_ref_attention`, per-sequence, bottom-right causal

`torch_ref_attention` stays as the correctness oracle. It matches the naive unpaged
path to ~2e-7 across prefill, decode, chunked prefill, mixed and GQA steps. Keep
diffing the fast path against it.

## 2. ~~Use `enable_gqa=True` instead of `repeat_interleave`~~ — done

Both paths now pass `enable_gqa=True` and no longer materialize `H/Hkv` physical
copies of the KV heads. The mask is also skipped entirely when it would be all-True:
`torch_ref` skips it for `q_len == 1` or `q_len == kv_len`, and `_batched_attention`
skips it when `q_len == 1` and every context is the same length, which keeps SDPA on
its fused vector kernel.

The measurement that motivated this, decode step, fp16, `H=16, Hkv=2, D=128,
kv_len=2048`:

| path | per call |
|---|---|
| `repeat_interleave` + all-True bool mask | 460.2 us |
| `enable_gqa=True`, no mask | 50.4 us |

`max abs diff: 0.0` — identical numerics, ~9x cheaper.

## 3. The page gather is the real ceiling — **still open, largest win**

`_gather_prefix` in `paged_sdpa.py` does

```python
k_cache[blocks].flatten(0, 1)[:kv_len]
```

which copies a request's whole context out of the pool, every layer, every step.
`block_size=16`, same shape as above:

```
page gather (K only): 366.2 us/call, copies 1.05 MB
=> per decode step, 28 layers x K and V: 20.5 ms, 59 MB copied
```

~20 ms per decode step for a *single* sequence before any attention math — a ceiling
of roughly 50 tok/s on the gather alone, growing linearly with context. Items 1 and 2
removed overheads *around* this; they did not touch it.

Two ways out:

- **The real fix:** attend page-by-page and combine the partials with log-sum-exp, so
  K/V never leaves the pool. That is what a paged kernel is for. [MPS.md](MPS.md)
  section 3 lists the options for writing one on Metal.
- **The cheap escape hatch:** the `kv_layout: "dense_slots"` config field (declared,
  never read — see Part C). Reserve capacity per sequence slot and *slice* instead of
  gathering. Costs memory, keeps the paged bookkeeping, removes the copy.

Largest win available, largest effort.

## 4. Collapse the per-request Python loop — **partially done**

Done: a step where every request has the same query length now goes through
`_batched_attention`, which issues **one** SDPA dispatch for the whole batch instead
of one per request. For pure decode that is 28 dispatches per step instead of
16 x 28 = 448.

Still open, two pieces:

- `_batched_attention` still loops in Python to gather and `F.pad` each request's
  prefix before `torch.stack`. The dispatch loop is gone; the gather loop is not. This
  is the same problem as item 3 and goes away with it.
- **Mixed prefill/decode steps still fall back to `torch_ref_attention`**, one
  dispatch per request per layer. With chunked prefill a mixed step is the common
  case under load, not the exception. A block mask over the packed stream would let
  one dispatch cover prefill chunks and decode tokens together; that is the thing the
  original "ideally vectorized the whole thing" comment was pointing at.

## 5. CUDA graphs are captured but never exercised here

`capture_cuda_graphs` / `graph_sizes` / the `self.graphs` lookup in `run()` are
written and look correct, but MPS exposes no capture/replay API and
`EngineConfig.__post_init__` forces `use_cuda_graphs = False` off CUDA
([MPS.md](MPS.md) section 1). So the code is untested in practice. Either verify it on
a CUDA box or treat it as aspirational — just don't expect a decode fast path from it
on a Mac.

---

# Part B — tests

The thinnest part of the repo, and the reason Part A is risky to attempt.

## 6. Only `KVCacheManager` is covered

`tests/test_kv_cache.py` is 10 tests, all passing, all on the cache manager.
`tests/fakes.py` now has real doubles — `FakeTokenizer`, `FakeEngine`,
`FakeAsyncEngine` (yields a canned token stream), `FakeRequest` — but **nothing uses
them yet**.

In rough order of value:

- **`Scheduler`** — the subtlest code in the repo and completely untested. Worth
  covering: the chunked-prefill budget (a prompt longer than
  `max_num_batched_tokens` is split across steps and nothing else starves);
  preemption (newest victim first, budget returned to the pool, victim requeued at the
  front, `num_computed_tokens` reset); the `victim is req` degenerate case where a
  single request outgrows the whole cache; and that admission is skipped in any step
  that preempted. All of it runs on CPU with no model loaded.
- **The attention backends.** `attn-demo`'s naive-path comparison is already an
  integration test; promote it to a real pytest case so a backend change cannot
  regress silently. Do this **before** attempting Part A items 3 and 4.
- **`sample()`** — top-k and top-p boundaries (the exclusive cumsum means the token
  that crosses the threshold is kept), mixed per-row temperatures in one batch,
  `temperature == 0` rows taking the argmax path, and teacher forcing via `token_ids`.
- **`ModelRunner._prepare`** — slot mapping arithmetic (`blocks[p // bs] * bs + p % bs`)
  against a hand-built block table, `query_start_loc` prefix sums, `logit_rows`
  pointing at the last row of each request, and the `pad_to` dummy rows.
- **The API layer** — `FakeAsyncEngine` exists precisely for this. Worth covering:
  the prompt-too-long 400 with `code="context_length_exceeded"`, `n != 1` -> 400, the
  SSE chunk shape (role on the first delta only, empty deltas skipped, the
  `finish_reason` chunk, the usage chunk under `stream_options.include_usage`, then
  `[DONE]`), and abort-on-disconnect.
- **`Detokenizer`** — the multi-byte UTF-8 hold-back. Feed it ids that split a
  character and assert nothing is emitted until `final=True`.
- **`runner/memory.py`** — pure arithmetic, trivial to test, and the docstring already
  claims testability as its reason for existing.

## 7. No end-to-end golden test

Nothing asserts that greedy generation from a fixed prompt produces a fixed token
sequence. That is the test that makes every Part A change safe, and the test
[SPECULATIVE_DECODING.md](SPECULATIVE_DECODING.md) assumes exists ("greedy output with
speculation enabled is token-for-token identical to greedy output without it").

Use the smallest real checkpoint that works (`HuggingFaceTB/SmolLM2-135M-Instruct`),
`--device cpu --attention-backend torch`, temperature 0. Mark it slow so the default
run stays fast.

---

# Part C — loose ends

## 8. Declared-but-unused config fields

`kv_layout` and `decode_backend` in `config.py` are read by nothing.
`kv_layout` is the escape hatch for Part A item 3, so it is worth keeping as a
placeholder; `decode_backend` (`eager` / `compiled` / `metal`) has no consumer and no
concrete plan — either wire it to `torch.compile` or drop it from the dataclass.

## 9. Accepted-but-ignored API fields

`api/schemas.py` declares `stop` and `seed` on both request models, and neither is
honoured anywhere:

- **`stop`** needs string-level stop sequences in `Engine.step`, which means checking
  the detokenized text, not just token ids — plus truncating the emitted delta at the
  match. Non-trivial because a stop string can straddle a token boundary and a delta
  may already have been streamed.
- **`seed`** is per-request; `EngineConfig.seed` is global and applied once in
  `ModelRunner.__init__`. Honouring it per request means a generator per request in
  `sample()`.

Either implement them or reject them with a 400 rather than silently ignoring them —
silently ignoring `stop` is the one most likely to surprise a client.

`SamplingParams.return_logprobs` and `Request.logprobs` are plumbed all the way
through `Engine.step` and `Output`, but no route exposes them: `/v1/completions`
returns a hardcoded `"logprobs": None`. The engine-side work is done; only the
serialisation is missing.

## 10. `pyproject.toml` has no `[build-system]`

`[project.scripts]` declares a `miniserve` console script, but without a build backend
the package is never installed, so the entry point is inert and everything runs as
`uv run main.py`. `cli.py` already has fallbacks for this (`attn-demo` / `kv-demo`
re-derive the repo root to import `demos/`). Either add a build backend and make the
console script real, or drop `[project.scripts]` so it stops implying something that
does not work.

## 11. No public API surface

`miniserve/__init__.py` and every package `__init__.py` are empty, so callers reach
deep into module paths (`from miniserve.engine.engine import Engine`). Fine for a
reading-oriented repo, but re-exporting `EngineConfig`, `Engine`, `AsyncEngine` and
`SamplingParams` from `miniserve/__init__.py` would make the library usable without
knowing the layout.

## 12. Monitoring stack is documented but not committed

[MONITORING.md](MONITORING.md) describes a `deploy/` tree (docker-compose,
`prometheus.yml`, Grafana provisioning and dashboard JSON) that does not exist in the
repo. The `/metrics` endpoint and all the counters and gauges it documents **are**
implemented; only the deployment files are missing. Either commit `deploy/` or mark
those sections as a recipe rather than a description.

Also still open from that doc: **latency histograms**. The current `/metrics` has
counters and gauges only, so there is no TTFT, no end-to-end latency and no per-step
duration. Section 3 of that doc has the design — record arrival / first-token /
finish timestamps on the `Request`, keep bounded histograms on the engine, export in
Prometheus histogram format.

## 13. Stray empty `metrics/` directory

`metrics/` at the repo root is empty and nothing references it. Probably a leftover
from the monitoring work; delete it or put the `deploy/` stack in it.

## 14. Tensor parallelism stays unimplemented, on purpose

`loader.shard_config` and `load_shard` raise `NotImplementedError` for `tp_size > 1`.
Sharding needs per-key split dimensions that do not exist, and a silently wrong
sharded model is far worse than a failed load. Leave it; just don't pass `tp_size`.

---

# Part D — the next feature

[SPECULATIVE_DECODING.md](SPECULATIVE_DECODING.md) is the design note. The repo is
structurally ready for it — verification is a chunked prefill, which `slot_mapping`,
block tables and the token budget already handle — but five places assume exactly one
new token per sequence per step:

1. `_prepare` returns one `logit_row` per sequence; needs a list per sequence so the
   runner can hand back `[B, k+1, V]`.
2. `Engine.step` appends one token and checks stop conditions once; needs a list per
   sequence, truncated at the first stop.
3. **KV rollback vs prefix caching — the one real correctness hazard.**
   `KVCacheManager.allocate` calls `register_full_blocks` with a total that would
   include draft tokens, so a block filled by tokens that are later rejected would be
   hashed and published as a reusable prefix. Split `allocate()` (reserve pages) from
   `commit(req, n)` (hash and register only accepted tokens) and add `rollback(req, n)`.
   Worth doing **now**, before speculation exists, because it is a quiet bug and the
   split is harmless on its own.
4. `_batched_attention` already takes a uniform `q_len`, so verification at `q = k+1`
   reuses that path — but the bottom-right causal mask branch needs checking for
   `q_len > 1` with unequal context lengths.
5. `sample()` needs to accept `[B, k+1, V]` plus optional draft probabilities and
   implement the acceptance rule (greedy: accept while argmax matches; sampling: accept
   with probability `min(1, p(x)/q(x))`, else resample from normalised `max(0, p - q)`).

Build the n-gram / prompt-lookup proposer first: no model, no extra KV budget, works
with greedy and with sampling.
