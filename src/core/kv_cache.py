from typing_extensions import OrderedDict


class KVCacheManager:
    def __init__(self,num_blocks,block_size,enable_caching):
        self.block_size = block_size
        self.enable_caching = enable_caching

        # so block 2 cannot be evicted while other requests are still using it
        self.ref_cnt = [0] * num_blocks

        # if a physical block contains a cached prefix, then what prefix does it represent
        self.block_hash: list[int|None] = [None] * num_blocks

        #our LRU queue , to free up the blocks , list of of our unused blocks
        self.free: OrderedDict[int,None] = OrderedDict((i,None) for i in range(1, num_blocks))

        # now given this prefix hash, where is its kv data (hash -> physical block) ?
        self.hash_to_block: dict[int,int] = {}
