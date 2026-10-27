import torch
import torch.nn.functional as F


def torch_ref_attention(
    q,  # [T ]
    k_cache,
    v_cache,
    meta,
    scale,
):
    """
    Paged attention.

    q:        [T, H, D]              packed queries for this step
    k/v_cache:[n_pages, bs, Hkv, D]  paged pool
    meta:     query_start_loc, seq_lens, block_table
    scale:    1/sqrt(D)

    For each request: gather its pages into a dense prefix, build a
    bottom-right causal mask, call SDPA. Slow (Python loop, copy of
    every context, host sync) but for understanding
    """
    out = torch.empty_like(q)

    # H = query heads, Hkv = KV heads (GQA if H > Hkv), bs = page size
    H, Hkv, bs = q.shape[1], k_cache.shape[2], k_cache.shape[1]

    # qsl[i]:qsl[i+1] = this request's slice of packed q
    # seq_lens[i]     = full context length after this step (kv_len)
    qsl, seq_lens = meta.query_start_loc_cpu, meta.seq_lens_cpu

    # for each sequence that has been packed
    for i in range(meta.num_seqs):
        # ideally vectorized the whole thing, by creating a block mask
        s, e, kv_len = qsl[i], qsl[i + 1], seq_lens[i]

        # tokens processed for i this step
        q_len = e - s

        # live pages only: ceil(kv_len / bs). drop the padded tail of the row (null page 0)
        blocks = meta.block_table[i, : -(-kv_len // bs)].long()

        # Gather to a dense prefix
        #   k_cache[blocks] : [n_pages, bs, Hkv, D]
        #   flatten(0, 1)   : [n_pages*bs, Hkv, D]   pages stitched on the token axis
        #   [:kv_len]       : drop unused slots in the last page
        k = k_cache[blocks].flatten(0, 1)[:kv_len]
        v = v_cache[blocks].flatten(0, 1)[:kv_len]

        # SDPA wants H KV heads. GQA: each KV head is shared by H/Hkv query heads
        k = k.transpose(0, 1).repeat_interleave(H // Hkv, 0)
        v = v.transpose(0, 1).repeat_interleave(H // Hkv, 0)

        # bottom right causal, this steps queries are teh last q_len positions
        # of a length kv_len sequence not 0 ... q_len - 1
        #   full prefill:  q_len == kv_len -> square causal mask
        #   chunked_prefill: q_len < kv_len -> suffix strip
        #   decode: q_len == 1 -> 1 row, all keys
        # if we have 4 kv positions already cached, and q_len = 2, then
        # our kv_len = 6 (as we will have 6 kv positions after this op)
        # and our q_len = 2 for positions 4 and 5. Effectively, q at position
        # 4 can attend to all the kv before it (positions 0-3) and the new
        # kv at the new position 4. Similarly, the q at position 5 can attend to
        # all the positions before it (0-3 + 4 just made) as well as the kv at
        # the new position 5

        # keys    0  1  2  3  4  5
        # q@4     T  T  T  T  T  x
        # q@5     T  T  T  T  T  T
        q_pos = torch.arange(kv_len - q_len, kv_len, device=q.device)
        mask = torch.arange(kv_len, device=q.device)[None, :] <= q_pos[:, None]

        # use SDPA
        sdpa = F.scaled_dot_product_attention(
            q[s:e].transpose(0, 1), k, v, attn_mask=mask, scale=scale
        )
        out[s:e] = sdpa.transpose(0, 1)

    return out
