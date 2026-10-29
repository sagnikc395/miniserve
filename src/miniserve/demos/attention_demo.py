"""Drive the paged attention reference end to end on a toy pool.

Builds a page pool by hand, writes K/V through a slot mapping the way
`Attention.forward` does, then runs `torch_ref_attention` for a prefill, a
decode, a chunked prefill and a GQA batch, checking each against an
unpaged attention written out with plain matmuls. Run with
`python main.py attn-demo`.
"""

import torch

from miniserve.attention.metadata import AttnMetadata
from miniserve.attention.torch_ref import torch_ref_attention

BLOCK_SIZE = 4  # tokens per page, deliberately tiny so pages are easy to read
NUM_PAGES = 16
D = 8  # head dim


def banner(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


class Pool:
    """A paged K/V pool plus the dense history of each request, for checking.

    The pool mirrors one layer's cache: `[2, num_pages, block_size, Hkv, D]`.
    Page 0 is never handed out; it is the padding value in `block_table`.
    """

    def __init__(self, num_heads: int, num_kv_heads: int):
        self.H, self.Hkv = num_heads, num_kv_heads
        self.k_cache = torch.zeros(NUM_PAGES, BLOCK_SIZE, num_kv_heads, D)
        self.v_cache = torch.zeros(NUM_PAGES, BLOCK_SIZE, num_kv_heads, D)
        self.next_page = 1
        self.block_ids: dict[str, list[int]] = {}
        self.history: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.queries: dict[str, torch.Tensor] = {}  # this step's q, kept for checking

    def slots(self, rid: str, start: int, count: int) -> torch.Tensor:
        """Flat cache addresses for positions [start, start+count), paging as needed."""
        pages = self.block_ids.setdefault(rid, [])
        while len(pages) < -(-(start + count) // BLOCK_SIZE):
            pages.append(self.next_page)
            self.next_page += 1
        return torch.tensor(
            [
                pages[p // BLOCK_SIZE] * BLOCK_SIZE + p % BLOCK_SIZE
                for p in range(start, start + count)
            ]
        )

    def write(
        self, rid: str, start: int, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Scatter new K/V into the pool, exactly as the attention layer does."""
        slot_mapping = self.slots(rid, start, k.shape[0])
        self.k_cache.view(-1, self.Hkv, D).index_copy_(0, slot_mapping, k)
        self.v_cache.view(-1, self.Hkv, D).index_copy_(0, slot_mapping, v)

        # keep an unpaged copy of the same tokens to check the gather against
        if rid in self.history:
            hk, hv = self.history[rid]
            self.history[rid] = (torch.cat([hk, k]), torch.cat([hv, v]))
        else:
            self.history[rid] = (k, v)
        return slot_mapping

    def block_table(self, rids: list[str]) -> torch.Tensor:
        """Page tables as one padded tensor, one row per request."""
        width = max(len(self.block_ids[r]) for r in rids)
        rows = [
            self.block_ids[r] + [0] * (width - len(self.block_ids[r])) for r in rids
        ]
        return torch.tensor(rows, dtype=torch.int32)


def metadata(pool: Pool, steps: list[tuple[str, int, int]]) -> AttnMetadata:
    """Describe one engine step.

    steps: (request_id, q_len, kv_len) per request, in packed order. q_len is
    how many tokens this request runs now, kv_len its whole context after it.
    """
    rids = [rid for rid, _, _ in steps]
    q_lens = [q for _, q, _ in steps]
    kv_lens = [kv for _, _, kv in steps]

    qsl = [0]
    for q in q_lens:
        qsl.append(qsl[-1] + q)
    cu_k = [0]
    for kv in kv_lens:
        cu_k.append(cu_k[-1] + kv)

    # slot_mapping covers only the new tokens, in the same packed order as q
    slot_mapping = torch.cat([pool.slots(rid, kv - q, q) for rid, q, kv in steps])

    return AttnMetadata(
        slot_mapping=slot_mapping,
        query_start_loc=torch.tensor(qsl, dtype=torch.int32),
        cu_seqlens_k=torch.tensor(cu_k, dtype=torch.int32),
        block_table=pool.block_table(rids),
        num_seqs=len(steps),
        max_query_len=max(q_lens),
        max_seq_len=max(kv_lens),
        query_start_loc_cpu=qsl,
        seq_lens_cpu=kv_lens,
    )


def naive_attention(q, k, v, scale: float) -> torch.Tensor:
    """Unpaged bottom-right causal attention, spelled out.

    q: [q_len, H, D], k/v: [kv_len, Hkv, D]. The last q_len positions of a
    kv_len long context are the queries, so query j sees keys 0..kv_len-q_len+j.
    """
    q_len, H = q.shape[0], q.shape[1]
    kv_len, Hkv = k.shape[0], k.shape[1]

    # GQA: query head h reads KV head h // (H // Hkv)
    group = H // Hkv
    k = k.repeat_interleave(group, dim=1)  # [kv_len, H, D]
    v = v.repeat_interleave(group, dim=1)

    scores = torch.einsum("qhd,khd->hqk", q, k) * scale
    q_pos = torch.arange(kv_len - q_len, kv_len)
    allowed = torch.arange(kv_len)[None, :] <= q_pos[:, None]
    scores = scores.masked_fill(~allowed[None], float("-inf"))
    return torch.einsum("hqk,khd->qhd", scores.softmax(-1), v)


def check(name: str, pool: Pool, steps, out: torch.Tensor) -> None:
    """Compare the paged output of one step against the naive path, per request."""
    scale = D**-0.5
    worst = 0.0
    start = 0
    for rid, q_len, kv_len in steps:
        hk, hv = pool.history[rid]
        want = naive_attention(
            q=pool.queries[rid], k=hk[:kv_len], v=hv[:kv_len], scale=scale
        )
        got = out[start : start + q_len]
        worst = max(worst, (got - want).abs().max().item())
        start += q_len
    print(f"  {name}: max abs error vs naive attention = {worst:.2e}")
    assert worst < 1e-5, f"{name} diverged from the naive path"


def run_step(pool: Pool, steps: list[tuple[str, int, int]]) -> torch.Tensor:
    """Write this step's K/V, build metadata, run the backend."""
    pool.queries = {}
    qs = []
    for rid, q_len, kv_len in steps:
        q = torch.randn(q_len, pool.H, D)
        k = torch.randn(q_len, pool.Hkv, D)
        v = torch.randn(q_len, pool.Hkv, D)
        pool.write(rid, kv_len - q_len, k, v)
        pool.queries[rid] = q
        qs.append(q)

    meta = metadata(pool, steps)
    print(
        f"  packed q_lens {[q for _, q, _ in steps]}, kv_lens {[kv for _, _, kv in steps]}"
    )
    print(f"  slot_mapping {meta.slot_mapping.tolist()}")
    print(f"  block_table\n{meta.block_table}")
    return torch_ref_attention(torch.cat(qs), pool.k_cache, pool.v_cache, meta, D**-0.5)


def demo() -> None:
    torch.manual_seed(0)

    banner("1. Full prefill of two requests in one packed step")
    pool = Pool(num_heads=4, num_kv_heads=4)
    steps = [("a", 9, 9), ("b", 5, 5)]
    out = run_step(pool, steps)
    print(f"  output {tuple(out.shape)} = [a's 9 tokens then b's 5, H, D]")
    print("  q_len == kv_len, so the mask is square causal")
    check("prefill", pool, steps, out)

    banner("2. One decode step for both requests")
    steps = [("a", 1, 10), ("b", 1, 6)]
    out = run_step(pool, steps)
    print("  q_len == 1: one query row attending over the whole page table")
    print("  a's 10th token opened a third page; b still fits in two")
    check("decode", pool, steps, out)

    banner("3. Chunked prefill: 'c' runs its prompt 6 tokens at a time")
    for start in (0, 6, 12):
        chunk = min(6, 14 - start)
        steps = [("c", chunk, start + chunk)]
        out = run_step(pool, steps)
        print(f"  chunk at offset {start}: q_len {chunk}, kv_len {start + chunk}")
        check(f"chunk@{start}", pool, steps, out)
    print("  only the last chunk's final row is a real next-token prediction;")
    print("  the scheduler throws the earlier samples away")

    banner("4. Mixed step: 'c' decodes while 'd' prefills")
    steps = [("c", 1, 15), ("d", 7, 7)]
    out = run_step(pool, steps)
    print("  one flat stream, no padding; prefill and decode are the same call")
    check("mixed", pool, steps, out)

    banner("5. GQA: 4 query heads over 2 KV heads")
    pool = Pool(num_heads=4, num_kv_heads=2)
    steps = [("g", 7, 7)]
    out = run_step(pool, steps)
    print(
        f"  pages hold {pool.Hkv} KV heads; each is shared by {pool.H // pool.Hkv} query heads"
    )
    check("gqa prefill", pool, steps, out)

    steps = [("g", 1, 8)]
    out = run_step(pool, steps)
    check("gqa decode", pool, steps, out)

    print("\nAll steps match the naive unpaged path.")


if __name__ == "__main__":
    demo()
