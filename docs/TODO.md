# TODO

Open work across the repo. Background on the macOS constraints is in [MPS.md](MPS.md),
the design walkthrough is in [FLOW.md](FLOW.md), and the target architecture is the
mermaid diagrams in [README.md](../README.md); this file is the actionable list.

Measurements below were taken on: M4, 24 GB, macOS 26.5, torch 2.14.1, Python 3.13.

## What runs today

| Path | Status |
|---|---|
| `python main.py attn-demo` | works, matches naive attention to ~2e-7 |
| `python main.py kv-demo` | works |
| `pytest tests` | 10 passed (all `KVCacheManager`) |
| `python main.py serve` / `generate` / `bench` | `ClickException("not implemented yet")` |
| anything that loads a real checkpoint | blocked on section 5 |

`core/` (scheduler, KV cache manager, sequence) and `attention/` are the two layers that
look finished. Everything from `DecoderLayer` upward is partial or empty.

---

# Part A — attention backends

## 1. The `sdpa` backend is an alias, and unreachable on Mac

`paged_sdpa_attention()` in `src/miniserve/attention/paged_sdpa.py` returns
`torch_ref_attention` — the same object the `torch` backend returns. The registry holds
two names for one implementation:

```
torch -> <function torch_ref_attention at 0x10f32eca0>
sdpa  -> <function torch_ref_attention at 0x10f32eca0>   # same address
```

It is also dead code on the only platform currently in use. `EngineConfig.__post_init__`
forces `attention_backend = "torch"` on any non-CUDA device, so on MPS:

```
device mps | backend torch | block_size 16 | dtype float16
```

The `block_size = 256` bump in the `else` branch — the one thing that suggests `sdpa`
means something different from `torch` — only ever fires on CUDA.

The README's engine flowchart already says what the split should be:

- `all q_len == 1` (decode) -> **batched paged SDPA, single gather, padding mask**
- prefill / mixed -> **per-sequence SDPA, bottom-right causal mask**

So `paged_sdpa` should become the batched decode path and `torch_ref` stay the
per-sequence reference, with `config.py` selecting `sdpa` on `mps` instead of forcing
`torch`. Items 2-4 are that work.

`torch_ref_attention` itself is fine as a correctness oracle — it matches the naive
unpaged path to ~2e-7 across prefill, decode, chunked prefill, mixed and GQA steps. Keep
it as the thing the fast path is diffed against.

## 2. Use `enable_gqa=True` instead of `repeat_interleave`

`paged_sdpa.py:54-55` materializes `H/Hkv` physical copies of every KV head before
calling SDPA. Torch 2.14 accepts `enable_gqa=True`, which does the broadcast inside the
kernel. Decode step, fp16, `H=16, Hkv=2, D=128, kv_len=2048`:

| path | per call |
|---|---|
| `repeat_interleave` + all-True bool mask | 460.2 us |
| `enable_gqa=True`, no mask | 50.4 us |

`max abs diff: 0.0` — identical numerics, ~9x cheaper.

Part of that gap is the mask. When `q_len == 1` every key is visible, so the mask is
all-True and only forces SDPA off its fast path. Skip `attn_mask` entirely in the decode
case.

Low effort, no architectural change. Do this first.

## 3. The page gather is the real ceiling

`k_cache[blocks].flatten(0, 1)[:kv_len]` copies a request's whole context out of the pool,
every layer, every step. Same shape as above, `block_size=16`:

```
page gather (K only): 366.2 us/call, copies 1.05 MB
=> per decode step, 28 layers x K and V: 20.5 ms, 59 MB copied
```

That is ~20 ms per decode step for a *single* sequence before any attention math — a
ceiling of roughly 50 tok/s on the gather alone, and it grows linearly with context.

This is what a real paged kernel exists to avoid: attend page-by-page and combine the
partials with log-sum-exp, so the KV never leaves the pool. MPS.md section 3 lists the
options; the `kv_layout: "dense_slots"` config field is the cheaper escape hatch (reserve
capacity per slot, slice instead of gather) if the full kernel is too much for now.

Largest win available, largest effort.

## 4. Collapse the per-request Python loop

One SDPA dispatch per request per layer: 16 seqs x 28 layers = 448 dispatches per step,
and MPSGraph launch overhead is heavier than CUDA's. A decode-only step has `q_len == 1`
everywhere, so it can collapse into a single batched `[B, H, 1, kv_max]` SDPA with a
padding mask. The loop comment ("ideally vectorized the whole thing, by creating a block
mask") already anticipates this.

---

# Part B — the rest of the repo

## 5. `model/` — the model is not assembled (blocks everything)

This is the critical path. Nothing that touches a real checkpoint can work until it is
done.

- **`layers.py:36` — `DecoderLayer` is `pass`.** `Model.__init__` builds
  `ModuleList(DecoderLayer(cfg, i) ...)` and `Model.forward` calls
  `layer(x, residual, cos, sin)` expecting `(x, residual)` back. Nothing implements that.
  It also needs to expose its attention module as `self_attn` or `attn`, because
  `ModelRunner.allocate_kv_cache` reaches in via
  `getattr(layer, "self_attn", None) or getattr(layer, "attn", None)` to hang the KV
  tensor on it.
- **No MLP at all.** No `MLP`, `MoE` or `gate_up_proj` class exists anywhere in `src/`.
- **The loader and the model disagree about weight layout.** `loader.py` resolves
  checkpoint keys against a *fused* model — it looks up `module.qkv_sizes`,
  `module.qkv_proj`, `module.gate_up_sizes`, `module.gate_up_proj`, and
  `layer.mlp.load_expert` / `w13` / `w2` for MoE. Confirmed count of non-loader references
  to each of those: **zero**. Meanwhile `model/attention.py` defines separate `q_proj` /
  `k_proj` / `v_proj`. The README says "fused qkv", so `attention.py` is the side that
  should change — fuse the three projections and publish `qkv_sizes`. Until then
  `load_model` raises on the first checkpoint tensor it tries to place, and the
  `assert not missing` at the end would fire anyway.
- **Tensor parallelism is deliberately unimplemented** — `shard_config` and `load_shard`
  raise `NotImplementedError` for `tp_size > 1`. Fine to leave; just don't pass `tp_size`.

## 6. `runner/model_runner.py` — one real bug, one unwired TODO

- **`model_runner.py:180` is a bug.** `blocks = np.asarray(s.req_block_ids)` — `s` is a
  `ScheduledRequest`, whose fields are `req`, `num_computed`, `num_new`. There is no
  `req_block_ids` attribute anywhere in the repo. This should be `s.req.block_ids`. It is
  an `AttributeError` on the first real `_prepare` call; it has not surfaced yet only
  because nothing can get that far (section 5).
- **`model_runner.py:34` — `# TODO: set attention backends`.** The runner never calls
  `get_attention_fn(cfg.attention_backend)`, and never passes an `attn_fn` down. But
  `Attention.__init__` *requires* one, and `Model` / `DecoderLayer` have no parameter to
  thread it through. Resolve together with section 5: decide whether the fn is passed
  down the constructor chain or read from a context var the way `AttnMetadata` is.
- **`prev_tokens` / `tokens_host` are allocated and never used.** The comment says
  `Engine.step` picks them up — that consumer does not exist yet (section 7).
- **`capture_cuda_graphs` is CUDA-only and unreachable on Mac** (`use_cuda_graphs` is
  forced `False` on non-CUDA, and MPS exposes no capture/replay API — MPS.md section 1).
  Leave it; just don't expect a decode fast path from it here. `graph_sizes` and the
  `self.graphs` lookup in `run()` are correct for CUDA.
- **`run()` drops two `SamplingParams` fields.** It calls
  `sample(logits, [s.req.params for s in batch])` with no `token_ids`, so
  `forced_token_ids` (teacher forcing) and `return_logprobs` are silently ignored, and
  `Request.logprobs` / `Output.logprobs` are never populated. `sample()` itself accepts a
  `token_ids` argument it never reads.

## 7. `engine/` — the engine loop is missing

`engine/engine.py` has `Output` and a finished `Detokenizer`, and nothing else. Per the
README both of these are still to be written:

- **`Engine.step()`** — the synchronous loop: `scheduler.schedule()` ->
  `runner.run(batch)` -> discard samples for mid-prompt chunks
  (`ScheduledRequest.completes_prompt` is already there for this) -> append committed
  tokens -> detokenize -> build `Output`. Note that **stop conditions live here, not in
  the scheduler**: nothing currently checks `params.max_tokens`, EOS, or `ignore_eos`, and
  `Scheduler.finish(req, reason)` has no caller.
- **`AsyncEngine`** — inbox queue, dedicated engine thread, `call_soon_threadsafe`
  hand-off back to the event loop for SSE streaming.

## 8. `api/` — entirely empty (13 files, 0 bytes)

`app.py`, `deps.py`, `errors.py`, `schemas.py`, `settings.py` and every router
(`chat.py`, `completions.py`, `health.py`, `metrics.py`, `models.py`) are zero-length,
as are the two `__init__.py`. `fastapi` is already a declared dependency. Needs:
`create_app()` + lifespan holding the engine on `app.state`, a `get_engine` dependency,
pydantic request/response schemas, chat-template application and tokenization, the
prompt-too-long 400, and SSE streaming with a final usage chunk. Depends on section 7.

## 9. `benchmarks/` — entirely empty

`run.py`, `prompts.py`, `hf_baseline.py` and `__init__.py` are all zero-length.
`main.py bench` is the entry point waiting on them. The HF baseline is what makes the
Part A numbers meaningful end to end.

## 10. `runner/memory.py` — a two-line comment

```
# TODO:
# add support for memory and recently used context windows
```

Decide whether this is real scope or should be deleted. Note `allocate_kv_cache` already
handles device memory budgeting, so the filename is misleading.

## 11. Tests — one module covered out of the whole repo

`tests/unit/test_kv_cache.py` (10 tests, all passing) is the only test file.
`tests/fakes.py` is a two-line comment promising `FakeAsyncEngine` and random paged-cache
builders. Nothing covers the scheduler (preemption and the chunked-prefill budget are the
subtlest code in the repo), `sample()`, the attention backends, `_prepare`, or the API.

The attention demo is effectively an integration test already — worth turning
`attn-demo`'s naive-path comparison into a real pytest case so a backend change cannot
regress silently, especially before touching Part A items 2-4.

## 12. Smaller loose ends

- **`config.py:46`** — with the flash-attn resolution commented out, a CUDA device with
  `attention_backend="auto"` is never rewritten to a real backend name, so
  `get_attention_fn("auto")` raises `KeyError`. Non-CUDA is unaffected (forced to
  `"torch"`). Either restore the auto-resolution or drop `"auto"` from the `Literal`.
- **`config.py`** — `kv_layout` and `decode_backend` are declared and never read by
  anything. `kv_layout` ties into Part A item 3.
- **`pyproject.toml`** — `description = "Add your description here"`, and there is no
  `[project.scripts]` entry, so the CLI only runs as `python main.py`.
- **`src/miniserve/__init__.py`** and most package `__init__.py` files are empty — fine,
  but nothing re-exports a public API, so callers reach deep into module paths.
- **`transformers>=5.21.0`** is a dependency but only `AutoConfig` is used. The tokenizer
  and chat template are needed for section 8.
