from miniserve.core.kv_cache import KVCacheManager
from miniserve.core.sequence import Request
from miniserve.sampling import SamplingParams

BLOCK_SIZE = 4


def make_manager(num_blocks=8, enable_caching=True):
    return KVCacheManager(num_blocks, BLOCK_SIZE, enable_caching)


def make_request(request_id, token_ids):
    return Request(request_id, token_ids, SamplingParams())


def test_block_zero_is_reserved():
    kv = make_manager(num_blocks=8)
    assert kv.num_free == 7
    assert 0 not in kv.free


def test_allocate_rounds_up_to_whole_blocks():
    kv = make_manager()
    req = make_request("a", list(range(5)))

    assert kv.allocate(req, 5, [])
    # 5 tokens over a page size of 4 needs 2 pages, not 1
    assert len(req.block_ids) == 2


def test_free_returns_every_block():
    kv = make_manager()
    req = make_request("a", list(range(9)))

    kv.allocate(req, 9, [])
    assert kv.num_free == 7 - 3

    kv.free_request(req)
    assert kv.num_free == 7
    assert req.block_ids == []


def test_allocate_is_incremental_across_steps():
    kv = make_manager()
    req = make_request("a", list(range(4)))

    assert kv.allocate(req, 4, [])
    assert len(req.block_ids) == 1

    # one decode step: a 5th token spills into a second page
    req.num_computed_tokens = 4
    req.token_ids.append(99)
    assert kv.allocate(req, 1, [])
    assert len(req.block_ids) == 2


def test_allocate_refuses_when_pool_is_exhausted():
    kv = make_manager(num_blocks=3)  # blocks 1 and 2 usable
    req = make_request("a", list(range(12)))

    assert not kv.allocate(req, 12, [])


def test_full_blocks_are_registered_for_reuse():
    kv = make_manager()
    req = make_request("a", list(range(8)))

    kv.allocate(req, 8, [])
    # both pages are full, so both are addressable by prefix hash
    assert len(kv.hash_to_block) == 2
    assert kv.block_hash[req.block_ids[0]] == req.block_hashes[0]


def test_identical_prefix_hits_the_cache():
    kv = make_manager()
    first = make_request("a", list(range(8)))
    kv.allocate(first, 8, [])

    second = make_request("b", list(range(8)))
    hits = kv.find_cached_prefix(second)

    # the last block is held back so there is always a token left to compute
    assert hits == first.block_ids[:1]


def test_cache_hit_bumps_the_ref_count():
    kv = make_manager()
    first = make_request("a", list(range(8)))
    kv.allocate(first, 8, [])
    shared = first.block_ids[0]

    second = make_request("b", list(range(8)))
    hits = kv.find_cached_prefix(second)
    kv.allocate(second, 4, hits)

    assert kv.ref_cnt[shared] == 2

    # the block survives the first request going away
    kv.free_request(first)
    assert kv.ref_cnt[shared] == 1
    assert shared not in kv.free


def test_reclaiming_a_cached_block_invalidates_its_hash():
    kv = make_manager(num_blocks=3)
    first = make_request("a", list(range(8)))
    kv.allocate(first, 8, [])
    stale_hash = first.block_hashes[0]
    kv.free_request(first)

    second = make_request("b", list(range(100, 108)))
    kv.allocate(second, 8, [])

    assert stale_hash not in kv.hash_to_block


def test_caching_disabled_skips_hashing():
    kv = make_manager(enable_caching=False)
    req = make_request("a", list(range(8)))

    kv.allocate(req, 8, [])

    assert kv.find_cached_prefix(req) == []
    assert kv.hash_to_block == {}
