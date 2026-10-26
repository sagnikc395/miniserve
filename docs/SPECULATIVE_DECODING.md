Yes, and the architecture you're building is a good fit for it, as long as you leave a few seams in place now. Speculative decoding (Leviathan et al., "Fast Inference from Transformers via Speculative Decoding", ICML 2023) works by proposing k draft tokens cheaply, then running the target model once over `[last_token, d1 ... dk]` and keeping the longest accepted prefix plus one corrected token. On MPS this pays off well: decode is memory-bandwidth bound, so verifying k+1 tokens costs roughly the same as decoding 1, because the weight reads dominate.

## What already works

The verification pass is structurally a **chunked prefill**: a query of length k+1 attending over an existing paged prefix with a bottom-right causal mask. Your `torch_ref` path, `slot_mapping`, and block tables handle that without changes. The scheduler's token budget also generalises naturally: a speculating sequence just costs k+1 tokens instead of 1.

## What breaks, and the seams to leave now

These are the places where the current code assumes exactly one new token per sequence per step. Designing them generically during the refactor makes speculation an addition rather than a rewrite.

1. **Logits rows.** `_prepare` returns one `logit_row` per sequence (the last one). Make it return a list of row indices per sequence, so the runner can hand back `[B, k+1, V]`.

2. **Committing tokens.** `Engine.step` appends one token and checks stop conditions once. Make it accept a list of tokens per sequence and check EOS and `max_tokens` after each one, truncating at the first stop.

3. **KV rollback.** Rejected draft tokens have already had their K/V written. Rolling back is mostly just decrementing `num_computed_tokens` and truncating `token_ids`, since those slots get overwritten later. The real hazard is prefix caching: `allocate()` calls `_register_full_blocks()` with the total that includes draft tokens, so a block filled by tokens that later get rejected would be hashed and published as a reusable prefix. That is a silent correctness bug. Split it now into `allocate()` (reserve pages) and `commit(seq, num_tokens)` (hash and register full blocks only for accepted tokens), and add `rollback(seq, num_tokens)` that releases trailing pages that became empty.

4. **The batched decode attention path.** `paged_sdpa._decode` assumes `q_len == 1`. Parameterise it on a uniform query length `q` with a causal mask over the last `q` positions, so normal decode is `q=1` and verification is `q=k+1` in the same batched kernel path.

5. **Sampler interface.** Have it accept target logits `[B, k+1, V]` plus optional draft probabilities. Greedy acceptance is "accept while argmax matches the draft". For temperature > 0, use the rejection rule from the paper: accept draft token x with probability min(1, p(x)/q(x)), otherwise resample from the normalised max(0, p - q). This preserves the target model's output distribution exactly.

6. **A `Proposer` interface** in `src/miniserve/spec/`:

   ```python
   class Proposer(Protocol):
       def propose(self, seqs: list[Sequence], k: int) -> list[DraftTokens]: ...
       # DraftTokens: token ids, plus probs if the proposer has them
   ```

## Which proposer to build first

- **N-gram / prompt lookup.** Match the last few tokens against earlier context and propose what followed. No model, no extra memory, works with greedy and (treating the draft as one-hot) with sampling. On unified memory this is the best first step because it costs nothing in KV budget. It helps most on summarisation, code editing and RAG-style prompts that copy from context.
- **Draft model** (for example Qwen3-0.6B drafting for Qwen3-4B, same tokenizer). Its layers need their own KV tensors, but they can reuse the target's block tables directly since token positions are identical. That means one `KVCacheManager`, two sets of cache tensors, and the memory sizer must budget for both.
- **Learned heads** (Medusa, Cai et al., ICML 2024; EAGLE, Li et al., ICML 2024). Higher acceptance rates, but they need trained heads for your specific target model and tree-structured attention masks for verification, which is a bigger change to the attention backend. Leave these for last.

For testing, the strongest check is that greedy output with speculation enabled is token-for-token identical to greedy output without it, and that your existing golden test still passes with speculation turned on. Add a CPU unit test for the prefix-cache case in point 3: a draft that fills a block and then gets rejected must leave `hash_to_block` unchanged.