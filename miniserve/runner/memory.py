"""Small memory-budget helpers used by :mod:`miniserve.runner.model_runner`.

The KV cache is sized from the peak memory of a profiling forward pass. Keeping
the arithmetic here makes it easy to unit test without allocating a model.
"""


def element_size(dtype) -> int:
    import torch

    return torch.finfo(dtype).bits // 8


def kv_cache_bytes_per_block(
    num_layers: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    dtype,
) -> int:
    """Bytes for one page of every layer: K and V, all KV heads."""
    return 2 * num_layers * block_size * num_kv_heads * head_dim * element_size(dtype)


def kv_cache_budget(
    total_bytes: int, utilization: float, other_bytes: int, peak_bytes: int
) -> int:
    """Bytes left for KV after weights, activations and foreign allocations."""
    return int(total_bytes * utilization) - other_bytes - peak_bytes


def num_blocks(
    budget_bytes: int,
    bytes_per_block: int,
    max_blocks: int,
) -> int:
    """Pages that fit in the budget, capped by what the scheduler can use."""
    return min(budget_bytes // bytes_per_block, max_blocks)
