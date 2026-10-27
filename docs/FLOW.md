Here is a walkthrough of the original single-file tinyserve, following the life of the process and then the life of a single request. The core idea to hold onto throughout: **every engine step runs one flat, unpadded stream of tokens through the model, where each request contributes however many tokens it currently owes, and all K/V lives in fixed-size pages that requests point to through a page table.** Prefill, chunked prefill, and decode are not separate code paths; they are the same operation with different token counts.

## 1. The three layers

The file is organised as three layers that communicate through narrow interfaces:

1. **HTTP layer** (`build_app`, `AsyncEngine`): FastAPI on the asyncio event loop. Validates requests, tokenizes, streams results. Never touches the GPU.
2. **Engine layer** (`Engine`, `Scheduler`, `KVCacheManager`, `Detokenizer`): a synchronous loop on a dedicated thread. Decides who runs each step and how many tokens, manages page ownership, turns sampled ids into text. Pure bookkeeping, apart from calling the runner.
3. **GPU runtime** (`ModelRunner`, `Model`, attention backends, `sample`): turns the scheduler's decisions into tensors, runs the forward pass, samples.

The engine layer only ever manipulates integers (token ids, block ids, counters). The runtime only ever sees a list of `ScheduledRequest(req, num_computed, num_new)`. That separation is what makes the refactor into packages straightforward.

## 2. Startup

`main()` parses arguments into a `Config`, and `__post_init__` resolves the hardware-dependent choices: if `flash_attn` cannot be imported it falls back to the torch backend, picks the page size (256 for FlashAttention's paged kernel, 16 for the torch reference), and disables CUDA graphs for the torch backend because its Python loop cannot be captured.

`Engine.__init__` then builds everything in this order:

**Tokenizer and stop ids.** Stop ids are the tokenizer's EOS plus whatever `generation_config.json` lists, since chat models often have several end-of-turn tokens.

**Model loading (`load_model`).** Downloads only `*.json`, `*.safetensors`, and tokenizer files. Builds the custom `Model` directly on the GPU with the target dtype set as the default (so parameters are allocated once in bf16, not created in fp32 and converted). Then it opens each safetensors shard and copies tensors into matching parameters, stripping the `model.` prefix that HF checkpoints use. The model precomputes RoPE `cos` and `sin` tables for every absolute position up to `max_model_len`, so at runtime rotary embedding is a table lookup indexed by position.

**Persistent buffers.** The runner allocates fixed GPU buffers for `input_ids`, `positions`, `slot_mapping`, `query_start_loc`, `cu_seqlens_k`, and `block_table`, sized for the largest step. Every step copies into these same buffers rather than allocating new tensors, which is a requirement for CUDA graphs (section 9).

**KV cache sizing (`allocate_kv_cache`).** This is a profiling run:

1. Set `ATTN_META = None` and run a forward over `max_num_batched_tokens` dummy tokens, plus logits and softmax for `max_num_seqs` rows. With no metadata, `Attention.forward` skips the cache entirely and returns `o_proj(q * 0)`, so the run only measures weights plus activations.
2. Read `max_memory_allocated()` (the peak), `mem_get_info()`, and compute `other` as memory held by anything outside PyTorch.
3. `budget = total * gpu_memory_utilization - other - peak`.
4. Bytes per page = `2 (K,V) * layers * block_size * Hkv * D * 2 bytes`.
5. `num_blocks = min(budget // bytes_per_block, max_num_seqs * max_blocks_per_seq + 1)`. The cap avoids allocating more pages than could ever be used; the `+1` is page 0.

Each layer gets one tensor shaped `[2, num_blocks, block_size, Hkv, D]`. Every layer uses the same page ids, so one block table per request serves all layers.

One consequence worth knowing: because the profiling pass skips attention, any temporary memory attention itself allocates (notably the per-request gather in the torch backend) is not included in the peak.

**CUDA graph capture** (section 9), then the `KVCacheManager` with `num_blocks`, then the `Scheduler`.

Finally `serve` wraps the engine in `AsyncEngine`, which immediately starts the engine thread, and hands the FastAPI app to uvicorn.

## 3. A request arrives

For `/v1/chat/completions`:

1. Message content given as a list of parts is flattened to text (image parts are dropped; this is a text-only server).
2. `apply_chat_template(..., add_generation_prompt=True)` renders the conversation into the exact string format the model was trained on, ending with the assistant turn header.
3. `respond()` tokenizes **on the event loop**, not the engine thread, for two reasons: it needs the prompt length to compute the default `max_tokens`, and a prompt that is too long must be rejected here with a 400. If it reached the `assert` in `add_request`, it would raise inside the engine thread and kill it, hanging every later request.
4. `params_from()` builds `SamplingParams`. Missing fields fall back to the model's shipped `generation_config.json` defaults (for Qwen3, temperature 0.6, top-p 0.95, top-k 20), matching vLLM. `max_tokens` defaults to the remaining context window.
5. `AsyncEngine.generate()` creates a per-request `asyncio.Queue`, registers `(event_loop, queue)` in `self.streams` under a fresh request id, and puts `("add", rid, (prompt_ids, params))` into the thread-safe `inbox`. It then awaits its queue and yields each `Output` as it arrives.

If the client disconnects mid-stream, the generator's `finally` block runs with `finished == False` and sends `("abort", rid, None)`, which makes the engine free the request's pages.

## 4. The engine thread loop

`AsyncEngine._loop` runs forever:

1. If the engine has no work, block on `inbox.get()` so the thread sleeps instead of spinning.
2. Drain every message currently in the inbox without blocking: `add` calls `engine.add_request`, `abort` calls `engine.abort`.
3. Call `engine.step()` once.
4. For each `Output`, look up its stream and push it with `loop.call_soon_threadsafe(q.put_nowait, o)`. This is the thread boundary: asyncio queues are not thread-safe, so the engine thread schedules the put onto the event loop instead of calling it directly.

`add_request` creates a `Request` and a `Detokenizer` and appends the request to the scheduler's `waiting` deque.

## 5. The `Request` and its central invariant

The fields that matter:

- `token_ids`: prompt plus everything generated so far.
- `num_computed_tokens`: how many of those tokens already have K/V written in the cache.
- `block_ids`: the page table, logical block i maps to physical page `block_ids[i]`.
- `block_hashes`: one hash per full block, for prefix caching.

The amount of work a request owes is simply `num_tokens - num_computed_tokens`:

- A fresh 1000-token prompt owes 1000.
- After a 512-token chunk, it owes 488.
- After the prompt completes and one token is sampled and appended, `num_tokens` grows by 1, so it owes exactly 1. That is decode.

This is why there is no separate decode path in the scheduler.

## 6. The scheduler: who runs, and how many tokens

`schedule()` has a single token budget, `max_num_batched_tokens`, shared by prefill chunks and decode tokens. This is what bounds the latency impact of long prompts: a 10,000-token prompt can only consume one budget's worth per step, so running decodes are never stalled for more than one step.

**Phase 1, running requests, oldest first.** For each running request, `n = min(owed, budget)`, then `kv.allocate(req, n)` reserves pages. If allocation fails, the scheduler **preempts the newest running request**: frees its pages, resets its `num_computed_tokens` to 0, and pushes it to the front of `waiting`. It retries until the allocation succeeds or the request has preempted itself. Evicting from the newest end guarantees a request is never evicted to make room for one younger than itself, so older requests make progress. A preempted request is recomputed from scratch later, but its full blocks usually come back from the prefix cache, so recomputation is mostly a cache hit.

**Phase 2, waiting requests, FIFO.** Skipped entirely if anything was preempted this step (the cache is already oversubscribed). Otherwise, while there is budget and a free sequence slot:

1. `find_cached_prefix(req)` returns the pages holding the longest cached prefix.
2. `n = min(num_tokens - cached_tokens, budget)`.
3. `allocate(req, n, cached)` attaches the cached pages and reserves new ones. If it fails, admission stops. Note it is `break`, not `continue`: the head of the queue blocks everything behind it, even a short request that would fit. That is strict FIFO with head-of-line blocking, chosen for fairness and simplicity.
4. On success, `num_computed_tokens = cached_tokens`, and the request moves to `running`.

**Phase 3, emit the batch.** For each running request with a plan, create `ScheduledRequest(req, num_computed, num_new)` capturing the *current* `num_computed_tokens`, then advance `num_computed_tokens += num_new` **before the model runs**. The runner needs the old value for positions and slots; the advance is optimistic.

A worked example with budget 2048 and block size 256: A and B are decoding, C is a new 3000-token prompt whose first 256 tokens are a cached system prompt.

| Request | Owed | Cached | Scheduled this step | Next step owes |
|---|---|---|---|---|
| A | 1 | | 1 | 1 (after sampling) |
| B | 1 | | 1 | 1 (after sampling) |
| C | 3000 | 256 | min(2744, 2046) = 2046 | 698 |

C's chunk does not reach the end of its prompt, so the token sampled at its last position this step is meaningless and is discarded (section 10).

## 7. The KV cache manager

This class only manages integers; it never touches GPU memory.

**Paging.** Token i of a request lives at logical block `i // B`, offset `i % B`, physical page `block_ids[i // B]`, and flat slot `page * B + offset`. Pages need not be contiguous, so memory never fragments and a request only holds pages it actually uses. Page 0 is a reserved null page that padding writes into and that is always masked out.

**Reference counts and the free list.** `ref_cnt[b]` counts how many active requests use page b. The free list is an `OrderedDict` used as an LRU queue: allocation pops from the front, freeing appends to the back. Both are O(1), including removing a page from the middle when a cached page is reclaimed.

**Prefix hashing.** Each full block's hash is `hash((parent_hash, tokens_in_block))`. Chaining the parent matters because K/V depends on everything before it: the same four tokens after different prefixes produce different K/V, so they must not match. Only full blocks are hashed; a partially filled tail block is never shared.

**`find_cached_prefix`** walks the request's block hashes left to right and stops at the first miss, because a hit after a miss would have been computed against a different prefix. It caps hits at `(num_tokens - 1) // B` blocks so at least one token always runs. The cache stores K/V, not hidden states or logits, so a request whose entire prompt hit the cache would have nothing to feed the model and no logits to sample its first token from.

**`allocate`** computes:

```
total = num_computed + len(cached) * B + num_new
need  = ceil(total / B) - len(block_ids) - len(cached)
reclaim = cached pages with ref_cnt == 0   (they sit on the free list)
```

and fails without mutating anything if `need + reclaim > num_free`. `need` can be 0 during decode, when the new token fits in the existing tail page. When popping a fresh page off the free list, any stale hash on it is removed from `hash_to_block`; this is eviction, and that old prefix is lost.

**`_register_full_blocks`** then hashes any block that will be full after this step and publishes it. This includes generated tokens: once decode fills a block, it becomes cacheable, which is what makes multi-turn chat cheap (the next turn's prompt starts with the previous turn's full history).

A subtle point that makes same-step sharing correct: registration happens during scheduling, before the forward pass, so a second request admitted in the same step can hit pages whose K/V has not been written yet. This still works because in every layer, the K/V write for all tokens in the step (`index_copy_`) happens before attention in that layer. The second request reads those pages after they are written, at the same absolute positions.

**`free_request`** decrements ref counts in *reverse* block order, so a finished request's tail pages land on the free list before its head pages, and are therefore evicted first. Head pages (system prompts, shared instructions) are the ones most likely to be reused, so they survive longest. Hashes stay in `hash_to_block` until the page is actually reused.

## 8. Packing the batch (`ModelRunner._prepare`)

The scheduler's list becomes one flat stream. The docstring example, with block size 16: A is decoding at position 37 with pages `[7, 3, 12]`, B is a new 5-token prompt with page `[5]`:

```
input_ids       [a37, b0, b1, b2, b3, b4]
positions       [37,  0,  1,  2,  3,  4]
slot_mapping    [197, 80, 81, 82, 83, 84]     12*16 + 37%16 = 197
query_start_loc [0, 1, 6]                     request i owns rows qsl[i]:qsl[i+1]
seq_lens        [38, 5]                       keys visible after this step's write
cu_seqlens_k    [0, 38, 43]
block_table     [[7, 3, 12], [5, 0, 0]]       padded with null page 0
logit_rows      [0, 5]                        last row of each request
```

No padding is needed between requests; `query_start_loc` and `cu_seqlens_k` describe where each request's queries and keys begin. These, plus `block_table`, form `AttnMetadata`, which is published through the global `ATTN_META` for every attention layer to read. Only `logit_rows` go through the LM head, which avoids a vocabulary-sized matmul for every prefill token.

## 9. The forward pass

`Model.forward` embeds tokens, looks up `cos` and `sin` at each token's absolute position, and runs each decoder layer: pre-norm, attention, residual, pre-norm, SwiGLU MLP, residual. Absolute positions are what make chunked prefill and prefix reuse correct: the second chunk of a prompt starts at position 2046, not 0, so its rotary embeddings match what a single full prefill would have produced.

Inside `Attention.forward`:

1. Project to q, k, v (with bias for Qwen2, per-head RMSNorm on q and k for Qwen3), apply rotary.
2. Write the new K/V into the cache in one call: `k_cache.view(-1, Hkv, D).index_copy_(0, slot_mapping, k)`. The view flattens page and offset into a single slot axis, so slot `s` is exactly the flat index from section 7.
3. Attend over each request's full context through its page table.

**FlashAttention backend:** `flash_attn_varlen_func` with `block_table` reads pages directly inside the kernel. With `causal=True` and query length shorter than key length, FlashAttention 2 aligns the causal mask to the bottom right, which is exactly the chunked-prefill and decode semantics: this step's queries are the *last* `q_len` positions of a `kv_len` sequence.

**Torch reference backend:** for each request, gather its live pages, flatten them into a dense `[kv_len, Hkv, D]` prefix, expand KV heads for grouped-query attention with `repeat_interleave`, build the bottom-right causal mask explicitly, and call SDPA. Slow (Python loop, full copy of every context), but easy to verify, and `attn-demo` checks the two backends against each other on scattered pages.

**CUDA graphs for decode.** A small model's decode step is hundreds of tiny kernels, and launching them from Python costs more than running them. At startup, for each bucket size (1, 2, 4, 8, 16, ... up to `max_num_seqs`, largest first so smaller graphs reuse its memory pool), the runner prepares a batch of dummy one-token requests, runs one warm-up forward to trigger lazy allocations, then records a second forward into a `CUDAGraph`. A graph records kernel launches with fixed memory addresses, which is why all inputs live in the persistent buffers. At runtime, if every request in the batch has `num_new == 1`, the runner picks the smallest bucket that fits, copies real data into the buffers, pads the remainder with dummy rows that write into the null page, and replays the whole forward with one call. `max_seq_len` is baked in as `max_model_len` because it must be constant across replays. Any batch containing a prefill chunk runs eagerly, since its token count varies every step.

## 10. Sampling and committing tokens

**`sample`** handles a batch where each row may have different settings. If every row is greedy, it returns the argmax. Otherwise it divides by per-row temperature, sorts each row, drops positions failing top-k (`rank >= k`) or top-p (`cumsum - p > top_p`, which always keeps the most likely token), scatters the filtered logits back to vocabulary order, and samples with the Gumbel-max trick (`argmax(log p + Gumbel noise)` is an exact draw from `p`). Rows with temperature 0 are then replaced by their argmax. The final `.tolist()` also synchronises with the GPU.

**Back in `Engine.step`,** for each scheduled request:

1. If `completes_prompt` is false (a mid-prompt chunk), skip it. The K/V was written, which was the point; the sample is junk.
2. Apply teacher forcing if `forced_token_ids` is set, and record full log-probabilities if requested. These two hooks exist for equivalence testing against HF.
3. Append the token. This is what makes the request owe exactly one token on the next step.
4. Finish with `"stop"` on a stop id (unless `ignore_eos`), or `"length"` at `max_tokens` or `max_model_len`. `scheduler.finish` removes the request from `running` and frees its pages onto the LRU list.
5. Run the detokenizer and emit an `Output` with the full text, the new delta, token ids, finish state, and prompt and cached-token counts.

## 11. Incremental detokenization

Decoding the whole output every step would be quadratic, and decoding single tokens in isolation breaks on tokenizers where a token's text depends on its neighbours (leading spaces, multi-byte characters split across tokens). The detokenizer keeps two offsets, `prefix` and `read`, and each step decodes `ids[prefix:read]` (already emitted) and `ids[prefix:]` (with new tokens). The difference is the delta. If the new text ends in U+FFFD, a multi-byte UTF-8 character is incomplete, so it waits for the next token. After emitting, the window slides forward (`prefix, read = read, len(ids)`), so each step decodes only a few tokens of context.

## 12. Back to the client

Each `Output` travels back through the event loop into the request's queue. For streaming, every output becomes an SSE `data:` event shaped like an OpenAI chunk, the last one carries `usage` (including `prompt_tokens_details.cached_tokens`, which shows prefix cache hits), followed by `data: [DONE]`. For non-streaming, the handler drains the generator and returns the final accumulated text.

## 13. The benchmark mode

`bench` compares three runs on identical prompts (Alpaca and Dolly instructions, sampled evenly across length bins, chat-formatted, all sharing a long system prompt):

1. **HF static batching:** fixed batches in arrival order, left-padded to the longest prompt, and each batch finishes only when its slowest row does. It counts the waste directly: padded prompt slots, and decode slots spent on rows that already finished.
2. **tinyserve without prefix caching:** continuous batching means a finished request leaves immediately and a waiting one takes its slot on the next step, with no padding.
3. **tinyserve with prefix caching:** the shared system prompt is computed once and its pages reused, shown in the `cached %` column.

It reports wall time, requests per second, output and total tokens per second, and p50/p90 completion times. The completion-time percentiles are where continuous batching shows its advantage most clearly, since short requests no longer wait behind long ones.

## 14. Limits of the design

These are deliberate simplifications rather than bugs, but they define where the system breaks down:

- **Strict FIFO admission** causes head-of-line blocking: one large waiting prompt blocks smaller ones behind it.
- **Self-preemption loop:** if a single request cannot fit even with the cache empty, it is preempted and retried forever. A fuller engine would reject it upfront by checking it against total cache capacity.
- **Optimistic state advance:** `num_computed_tokens` and block hashes are updated before the forward runs. If the forward raises, the engine thread logs and continues, leaving those requests (and possibly published hashes) inconsistent.
- **No CPU and GPU overlap:** scheduling and batch packing run while the GPU sits idle. Production engines prepare step N+1 while step N executes.
- **Prefill runs eagerly**, and only plain RoPE is supported (the loader asserts this), so models using scaled RoPE variants are rejected.
- **Prefix hashes use Python's built-in `hash`**, which is fine for an in-process cache but has a non-zero collision probability; vLLM offers a cryptographic hash option for this reason.
- **No stop strings, `n > 1`, or logprobs in the HTTP API**, although the engine computes logprobs internally for testing.

If you want, the next useful step is to turn sections 6 and 7 into the CPU-only unit tests in your `tests/unit/` folder, since the scheduler and KV manager are where most of the subtle invariants live, and they need no model or GPU to test.
