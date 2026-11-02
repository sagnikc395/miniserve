import torch
import torch.nn.functional as F

from miniserve.attention.backends import AttentionFn


def _gather_prefix(k_cache, v_cache, block_table_row, kv_len, block_size):
    """Gather one request's live pages into a dense [kv_len, Hkv, D] prefix."""
    blocks = block_table_row[: -(-kv_len // block_size)].long()
    k = k_cache[blocks].flatten(0, 1)[:kv_len]
    v = v_cache[blocks].flatten(0, 1)[:kv_len]
    return k, v


def _batched_attention(q, k_cache, v_cache, meta, scale, q_len):
    """One SDPA over the whole batch.

    Used when every request has the same number of query rows. Decode is the
    important case (`q_len == 1`); chunked verification with a uniform query
    length uses the same path with a bottom-right causal mask.
    """
    H, Hkv = q.shape[1], k_cache.shape[2]
    block_size = k_cache.shape[1]
    device = q.device

    qsl, seq_lens = meta.query_start_loc_cpu, meta.seq_lens_cpu
    batch = meta.num_seqs
    max_kv = max(seq_lens)
    total_q = batch * q_len

    # [B, H, Q, D]
    q_b = q[:total_q].view(batch, q_len, H, -1).transpose(1, 2)

    ks, vs = [], []
    for i in range(batch):
        kv_len = seq_lens[i]
        k, v = _gather_prefix(
            k_cache, v_cache, meta.block_table[i], kv_len, block_size
        )
        if kv_len < max_kv:
            pad = max_kv - kv_len
            k = F.pad(k, (0, 0, 0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, 0, 0, pad))
        # [Hkv, max_kv, D]
        ks.append(k.transpose(0, 1))
        vs.append(v.transpose(0, 1))

    k_b = torch.stack(ks)  # [B, Hkv, max_kv, D]
    v_b = torch.stack(vs)

    # Fast path: a pure decode step with equal context lengths needs no mask at
    # all. SDPA then stays on its fused vector kernel.
    if q_len == 1 and all(kv == max_kv for kv in seq_lens):
        mask = None
    else:
        # bottom-right causal: this step's query j is at absolute position
        # kv_len - q_len + j, so it may read keys 0 .. that position
        key_pos = torch.arange(max_kv, device=device)
        q_pos = (
            torch.tensor(seq_lens, device=device)[:, None]
            - q_len
            + torch.arange(q_len, device=device)[None, :]
        )
        allowed = key_pos[None, None, :] <= q_pos[:, :, None]
        # drop the padding of shorter sequences
        allowed &= key_pos[None, None, :] < torch.tensor(
            seq_lens, device=device
        )[:, None, None]
        mask = allowed[:, None, :, :]  # [B, 1, Q, max_kv]

    out = F.scaled_dot_product_attention(
        q_b, k_b, v_b, attn_mask=mask, scale=scale, enable_gqa=True
    )
    return out.transpose(1, 2).reshape(total_q, H, -1)


def torch_ref_attention(
    q,  # [T, H, D]
    k_cache,
    v_cache,
    meta,
    scale,
):
    """
    Paged attention reference.

    q:        [T, H, D]              packed queries for this step
    k/v_cache:[n_pages, bs, Hkv, D]  paged pool
    meta:     query_start_loc, seq_lens, block_table
    scale:    1/sqrt(D)

    For each request: gather its pages into a dense prefix, build a
    bottom-right causal mask, call SDPA. Slow (Python loop, copy of
    every context, host sync) but easy to verify.
    """
    out = torch.empty_like(q)

    H, Hkv, bs = q.shape[1], k_cache.shape[2], k_cache.shape[1]

    # qsl[i]:qsl[i+1] = this request's slice of packed q
    # seq_lens[i]     = full context length after this step (kv_len)
    qsl, seq_lens = meta.query_start_loc_cpu, meta.seq_lens_cpu

    for i in range(meta.num_seqs):
        s, e, kv_len = qsl[i], qsl[i + 1], seq_lens[i]
        q_len = e - s

        k, v = _gather_prefix(
            k_cache, v_cache, meta.block_table[i], kv_len, bs
        )

        # SDPA wants [H, kv_len, D]
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        # decode: one query row sees every key, so the mask only costs time
        if q_len == 1 or q_len == kv_len:
            mask = None
        else:
            # bottom right causal, this step's queries are the last q_len
            # positions of a length kv_len sequence
            q_pos = torch.arange(kv_len - q_len, kv_len, device=q.device)
            mask = torch.arange(kv_len, device=q.device)[None, :] <= q_pos[:, None]

        sdpa = F.scaled_dot_product_attention(
            q[s:e].transpose(0, 1),
            k,
            v,
            attn_mask=mask,
            is_causal=q_len == kv_len and q_len > 1,
            scale=scale,
            enable_gqa=True,
        )
        out[s:e] = sdpa.transpose(0, 1)

    return out


def paged_sdpa_attention() -> AttentionFn:
    """The SDPA backend's entry in the registry.

    Batched when every request has the same query length (the decode case and
    uniform speculative verification); otherwise the per-sequence reference
    keeps mixed prefill/decode steps correct.
    """

    def _attention(q, k_cache, v_cache, meta, scale):
        qsl = meta.query_start_loc_cpu
        q_lens = [qsl[i + 1] - qsl[i] for i in range(meta.num_seqs)]
        if q_lens and len(set(q_lens)) == 1:
            return _batched_attention(
                q, k_cache, v_cache, meta, scale, q_lens[0]
            )
        return torch_ref_attention(q, k_cache, v_cache, meta, scale)

    return _attention
