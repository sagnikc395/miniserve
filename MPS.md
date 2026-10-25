# Running miniserve on Apple Silicon (MPS)

Notes on why the CUDA-centric parts of the design (FlashAttention, CUDA graphs) don't map directly to macOS, what already works, and what to build instead.

Verified against: M4, 24 GB, macOS 26.5, torch 2.14.1, Python 3.13.

## 1. The premise needs a small correction

Neither limitation is actually caused by unified memory.

- **FlashAttention** is CUDA-specific because it is a CUDA/Triton kernel using `__shared__` memory, `cp.async`, warp shuffles, and tensor-core MMA. Metal has the equivalents (`threadgroup` memory, `simdgroup_matrix`, simd reductions), so the *algorithm* ports fine. Unified memory is irrelevant to the execution model.
- **CUDA graphs** need a driver-level capture/replay API (`cudaGraphCapture*` / `cudaGraphLaunch`). Metal/PyTorch-MPS exposes no such API. Again, unified memory is orthogonal.

What actually bites on macOS is narrower: **no page-table-aware attention kernel**, and **no replayable decode step**.

## 2. What already works in this environment

- `F.scaled_dot_product_attention` on MPS dispatches to `aten::_scaled_dot_product_attention_math_for_mps`. Despite the name, this is a fused Metal implementation:
  - a vector decode kernel for `q_len <= 8`
  - a 32-way split-KV + combine path at ≥1024 keys
  - prefill attention over simdgroup matrices
  - a Metal 4 cooperative-tensor path where available

  It supports causal, bool/additive masks, GQA (`enable_gqa=True`), and non-contiguous K/V with unit last stride. The S×S matrix is never materialized.
- `sdpa_kernel(SDPBackend.FLASH_ATTENTION)` is effectively a no-op on MPS — all variants route to the same MPS op.
- `torch.compile` has a real MPS inductor backend in this build: it emits Metal kernels via `async_compile.metal` and fuses chains like RMSNorm + SiLU + residual. It does **not** capture graphs: `mode="reduce-overhead"` logs `skipping cudagraphs due to multiple devices: device(type='mps')`.
- `torch.mps.compile_shader(msl_source)` works for custom Metal kernels (in-place output buffers, scalar args, `threads=` / `threadgroups=`).

So the gap is: **dense KV attention is solved; paged KV attention and decode replay are not.**

## 3. FlashAttention alternatives

| Option | Effort | Use when | Main cost |
|---|---|---|---|
| Dense KV slots + SDPA | Low | ≤~30 concurrent seqs, ≤4k ctx | reserves capacity per slot |
| Gather pages → SDPA | Low | Prefill, prefix hits, short decode | extra KV copy per decode step |
| Custom Metal paged decode kernel | Medium | Long ctx / high concurrency | kernel code + validation |
| FlexAttention on MPS | Low | Packed prefill, custom masks | still needs dense K/V |
| MLX | High (rewrite) | Max Apple performance | port allocator/scheduler |
| Core ML / MPSGraph | High | Single-stream fast path | static shapes, no paging |

### 3.1 Dense KV slots + SDPA (start here)

Keep the paged allocator, block tables, prefix cache, and eviction logic as *scheduler/memory-management* concepts, but give each active request a dense KV slot:

```text
[num_slots, num_kv_heads, capacity, head_dim]   per layer
```

Decode slices `k[:, :, :seq_len]` and calls SDPA once per layer with a padding mask. This works with non-contiguous slices and masks, and matches the dense result. A prefix-cache hit copies the matched prefix KV into the slot once at admission; after that, decode is zero-copy.

Cost is capacity reserved per slot:

- 1B model: ~32 KiB/token → 128 MiB per 4k slot
- 8B model: ~128 KiB/token → 512 MiB per 4k slot (and the fp16 weights alone are ~16 GiB on a 24 GiB machine)

If your concurrency target fits that budget, this is the simplest correct MPS path.

### 3.2 Gather pages → SDPA for prefill, not decode

Prefill is O(S²) compute, so an O(S) gather/scatter is noise. Decode is O(S) memory traffic, so gathering pages every step roughly doubles KV traffic and defeats the point. Use gather for chunked prefill and prefix-cache hits; keep decode on dense slots or a custom kernel.

### 3.3 Custom paged decode kernel (the real fix)

Decode is `q_len=1`, so it isn't classic FlashAttention — it's a split-KV online-softmax reduction over pages.

- Map one threadgroup per `(seq, kv_head, page-chunk)`.
- Pass `block_table` and `seq_lens`.
- Compute partial max/sum/acc.
- Run a second kernel to combine partials.

PyTorch's own `sdpa_vector_2pass` (`kDecodeBlocks=32`, 256 threads per block, then a 1024-thread combine) is the exact template; add one level of indirection for the page table. Reference files in PyTorch: `aten/src/ATen/native/mps/kernels/Attention.h` and `aten/src/ATen/native/mps/operations/Attention.mm`.

Implementation options:

- `torch.mps.compile_shader` to prototype at runtime.
- Compile a `.metal` file with `xcrun metal` and load it via `torch.mps.load_metallib(path)`.

Keep SDPA for prefill. For the kernel, page sizes of 32/64 coalesce better than 16; 16 is a memory-granularity choice, not a tile choice.

### 3.4 FlexAttention on MPS

It works in this build and there is a `metal_flex_attention_template.py`. Use it for:

- document masks
- sliding windows
- packed multi-request prefill
- chunk boundaries

It still requires dense K/V, so it complements paging rather than replacing it.

### 3.5 Third-party / MLX / Core ML

- `mps-flash-attn` (the package `config.py` probes) is worth benchmarking if it builds against torch 2.14, but it is not required now that SDPA has a Metal path.
- MLX is the strongest Apple-native option: `mx.fast.scaled_dot_product_attention`, `mx.fast.metal_kernel` for a paged decode kernel, and `mx.compile` for graph-level fusion. You'd port the allocator/scheduler.
- Core ML stateful models (`MLState` for KV) compile a graph ahead of time, but static shapes and no paging/continuous batching make them a single-stream fast path, not a serving engine.

## 4. CUDA graph alternatives

A CUDA graph eliminates per-kernel launch overhead, allocator work, and Python overhead in the decode loop. On MPS, in rough order of ROI:

1. **Static shapes + preallocation (80% of the benefit).** Bucket `max_num_seqs` (1/2/4/8/16/32/64) and prefill chunk sizes. Preallocate token IDs, positions, slot mappings, block tables, and logits; write into views instead of `torch.cat`/`torch.empty` inside the loop. This also stops `torch.compile` recompiles.
2. **`torch.compile` the pointwise/reduction chains.** Real Metal fusion in this build, but it is *fusion, not replay*. Compile the decoder block with `fullgraph=True, dynamic=False` and bucketed shapes; keep matmuls and SDPA as aten calls. A toy block went 0.11 ms → 0.07 ms here, so the win is real but modest.
3. **Manual fusion / per-layer megakernel.** Fuse RMSNorm+residual, RoPE+KV store, and SiLU(gate)*up. One dispatch per layer is the practical CUDA-graph analog; one dispatch per step is the extreme.
4. **One command buffer per decode step.** PyTorch submits per op. A custom ObjC++/Metal extension can encode the whole step into one `MTLCommandBuffer` and commit once — that's what llama.cpp's Metal backend does, and it removes most CPU submission gaps.
5. **Fewer sequential steps.** Speculative decoding or n-gram/prompt lookup. On a launch-bound decode loop this often beats micro-optimizing kernels.
6. **Batch more.** Continuous batching plus static buckets amortizes launch overhead; larger batches shift the bottleneck to bandwidth, where it belongs.

Avoid:

- `torch.cuda.CUDAGraph` shims
- `mode="reduce-overhead"` on MPS
- Core ML as a serving engine

## 5. What I'd do in miniserve

- **Phase 1:** Keep `attention_backend="sdpa"`. Add a `kv_layout` switch: `"dense_slots"` for MPS, `"paged"` for CUDA/custom kernel. Dense slots + batched SDPA + preallocated buffers. Benchmark decode tok/s at batch 1/8/32 and ctx 128/1k/4k.
- **Phase 2:** Add a `"metal_paged"` decode backend using `torch.mps.compile_shader`. Validate against the `"torch"` reference path using the `forced_token_ids` / `return_logprobs` already in `SamplingParams`.
- **Phase 3:** Fuse the pointwise chains with `torch.compile` or hand-written Metal; preallocate and bucket. Add a single-command-buffer path only if profiling shows submission gaps.
- **Phase 4 (optional):** MLX port for max Apple perf, or a Core ML single-stream fast path.
- Keep `use_cuda_graphs` gated to CUDA; add `decode_backend: eager | compiled | metal` instead of pretending graphs exist on MPS.

Config shape this implies:

```python
attention_backend: Literal["sdpa", "torch", "metal_paged"]
kv_layout: Literal["dense_slots", "paged"]
decode_backend: Literal["eager", "compiled", "metal"]
use_cuda_graphs: bool  # CUDA only
```

## 6. Unified-memory gotchas that do matter

- `gpu_memory_utilization` isn't enforceable like CUDA VRAM. On a 24 GB M4, `torch.mps.recommended_max_memory()` is **17.8 GiB**, so `0.80 × 24 = 19.2 GiB` overcommits. Use `torch.mps.set_per_process_memory_fraction()` and cap below the recommended max.
- Oversubscription causes swap/compression stalls, not clean OOMs. Cap concurrency rather than relying on an exception.
- Decode is bandwidth-bound; KV dtype/layout and avoiding per-step gathers matter more than FLOPs.
- Kill sync points: `.item()`, `.tolist()`, per-sequence `torch.multinomial`. Sample the whole batch on GPU and read tokens once per step.
- For profiling, `torch.profiler` has no MPS activity; use `torch.mps.profiler.start(...)` with Xcode Instruments/Metal System Trace, plus `METAL_CAPTURE_ENABLED=1` for GPU captures.

## 7. Bottom line

On MPS:

- Replace "FlashAttention" with SDPA's Metal kernels, or one custom paged decode kernel.
- Replace "CUDA graphs" with static shapes + preallocation + Metal fusion / one command buffer.

For a learning-oriented vLLM clone, dense slots + SDPA is the honest first implementation; the paged Metal decode kernel is the piece worth building once long-context concurrency matters.
