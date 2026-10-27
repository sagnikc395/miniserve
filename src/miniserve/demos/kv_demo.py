"""Walk the KV cache manager by hand: pages, page tables, prefix reuse, eviction.

No model and no tensors here, only the integer bookkeeping the engine layer
does. Run with `python main.py kv-demo`.
"""

from miniserve.core.kv_cache import KVCacheManager
from miniserve.core.sequence import Request
from miniserve.sampling import SamplingParams

BLOCK_SIZE = 4
NUM_BLOCKS = 8


def banner(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def make_request(request_id: str, token_ids: list[int]) -> Request:
    return Request(request_id, token_ids, SamplingParams())


def show_pool(kv: KVCacheManager) -> None:
    # free is the LRU queue: leftmost is the next page handed out
    print(f"  free pages ({kv.num_free}): {list(kv.free)}")
    print(f"  ref counts: {kv.ref_cnt}")


def show_request(req: Request) -> None:
    # block_ids is the page table: logical block i lives in physical page block_ids[i]
    print(
        f"  {req.request_id}: {req.num_tokens} tokens, "
        f"{req.num_computed_tokens} computed, page table {req.block_ids}"
    )


def demo() -> None:
    kv = KVCacheManager(NUM_BLOCKS, BLOCK_SIZE, enable_caching=True)

    banner(f"1. A fresh pool of {NUM_BLOCKS} pages, {BLOCK_SIZE} tokens each")
    print("  page 0 is the null page, reserved as block_table padding")
    show_pool(kv)

    banner("2. Prefill request 'a' over a 9 token prompt")
    a = make_request("a", list(range(100, 109)))
    assert kv.allocate(a, a.num_tokens, [])
    a.num_computed_tokens = a.num_tokens
    show_request(a)
    print("  9 tokens over a page size of 4 needs 3 pages; the last one is half empty")
    print(f"  full blocks registered for reuse: {len(kv.hash_to_block)} of 3")
    show_pool(kv)

    banner("3. Four decode steps")
    for tok in (200, 201, 202, 203):
        a.token_ids.append(tok)
        # each step owes exactly num_tokens - num_computed_tokens == 1 token
        assert kv.allocate(a, 1, [])
        a.num_computed_tokens = a.num_tokens
        print(f"  appended {tok} -> page table {a.block_ids}")
    print("  the 13th token crossed a page boundary, so a 4th page was claimed on demand")
    show_pool(kv)

    banner("4. Request 'b' arrives sharing the first 8 tokens of 'a'")
    b = make_request("b", list(range(100, 108)) + [999])
    hits = kv.find_cached_prefix(b)
    print(f"  prefix cache hits: {hits} (a's pages {a.block_ids[:2]})")
    num_new = b.num_tokens - len(hits) * BLOCK_SIZE
    assert kv.allocate(b, num_new, hits)
    b.num_computed_tokens = len(hits) * BLOCK_SIZE
    show_request(b)
    print(f"  {len(hits) * BLOCK_SIZE} tokens skipped, only {num_new} to compute")
    shared = hits[0]
    print(f"  page {shared} is now owned twice: ref_cnt = {kv.ref_cnt[shared]}")
    show_pool(kv)

    banner("5. 'a' finishes while 'b' still reads its pages")
    kv.free_request(a)
    print(f"  page {shared} survives: ref_cnt = {kv.ref_cnt[shared]}, free = {list(kv.free)}")
    print(f"  'a' tail pages went back to the LRU queue but keep their hashes,")
    print("  so a third copy of the same prompt would still hit them")

    banner("6. The pool runs dry")
    big = make_request("big", list(range(50)))
    print(f"  50 tokens need 13 pages, only {kv.num_free} free")
    print(f"  allocate() returns {kv.allocate(big, big.num_tokens, [])}")
    print("  -> the scheduler preempts a running request and retries")

    banner("7. Reusing an evicted page invalidates its hash")
    kv.free_request(b)
    stale = make_request("stale", list(range(100, 108)))
    kv.update_hashes(stale)
    stale_hash = stale.block_hashes[0]
    print(f"  the shared prefix is still addressable: {stale_hash in kv.hash_to_block}")
    # a prompt exactly as long as the free list drains every page, stale hashes included
    num_free_tokens = kv.num_free * BLOCK_SIZE
    fresh = make_request("fresh", list(range(900, 900 + num_free_tokens)))
    assert kv.allocate(fresh, fresh.num_tokens, [])
    print(
        f"  after a {num_free_tokens} token request claimed every free page: "
        f"{stale_hash in kv.hash_to_block}"
    )
    show_pool(kv)


if __name__ == "__main__":
    demo()
