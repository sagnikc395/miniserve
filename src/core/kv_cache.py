from typing_extensions import OrderedDict
import math


class KVCacheManager:
    def __init__(self, num_blocks, block_size, enable_caching):
        self.block_size = block_size
        self.enable_caching = enable_caching

        # so block 2 cannot be evicted while other requests are still using it
        self.ref_cnt = [0] * num_blocks

        # if a physical block contains a cached prefix, then what prefix does it represent
        self.block_hash: list[int | None] = [None] * num_blocks

        # our LRU queue , to free up the blocks , list of of our unused blocks
        # maps the block_index and the hash
        self.free: OrderedDict[int, None] = OrderedDict(
            (i, None) for i in range(1, num_blocks)
        )

        # now given this prefix hash, where is its kv data (hash -> physical block) ?
        self.hash_to_block: dict[int, int] = {}

    @property
    def num_free(self):
        # keep track of how many available blocks
        return len(self.free)

    def update_hashes(self, req):
        bs = self.block_size
        while len(req.block_hashes) < (req.num_tokens // bs):
            # only hash a full block
            i = len(req.block_hashes)
            parent = req.block_hashes[-1] if req.block_hashes else None
            req.block_hashes.append(
                hash((parent, tuple(req.token_ids[i * bs : (i + 1) * bs])))
            )

    def find_cached_prefix(self, req):
        # if we are not doing prefix caching, this wont work
        if not self.enable_caching:
            return []

        self.update_hashes(req)

        # [A B C D E F G H I]
        # [A B C D E F G H X]
        # A-H IS IN KV Cache and Hashed , technically reused the previos once then | X and predict the next token
        # use only the maximally matching blocks , so atleast 1 token to compute on
        max_blocks = (req.num_tokens - 1) // self.block_size
        # cache hits
        hits = []
        for h in req.block_hashes[:max_blocks]:
            block_id = self.hash_to_block.get(h)
            if block_id is None:
                break
            hits.append(block_id)

        return hits

    def allocate(self, req, num_new_tokens, cached):
        # can assign our blocks to the page table
        # num_new_tokens = 1 (generally per decoding step)
        total = req.num_computed_tokens + len(cached) * self.block_size + num_new_tokens

        total_needed_blocks = math.ceil(total // self.block_size)

        need = total_needed_blocks - len(req.block_ids) - len(cached)
        reclaim = sum(1 for b in cached if self.ref_cnt[b] == 0)

        if need + reclaim > self.num_free:
            # not enough spots left
            return False

        # claim the cached blocks
        for block_idx in cached:
            # santiy check
            if self.ref_cnt[block_idx] == 0:
                # remove from free list and reclaim
                del self.free[block_idx]

            self.ref_cnt[block_idx] += 1
            req.block_ids.append(block_idx)
