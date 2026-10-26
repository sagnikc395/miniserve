from collections.abc import Callable

import torch

from miniserve.attention.metadata import AttnMetadata
from miniserve.attention.torch_ref import torch_ref_attention

# Every backend is a pure function over the paged pool, which keeps the registry
# independent of Config and of the nn.Module that calls it.
AttentionFn = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, AttnMetadata, float], torch.Tensor
]


def load_flash_attn() -> AttentionFn:
    # Imported on use, not at module scope: a missing package must not break
    # `import miniserve.config` on a machine that never runs this backend.
    from mps_flash_attn import flash_attention

    return flash_attention


def load_torch_ref() -> AttentionFn:
    return torch_ref_attention


def load_paged_sdpa() -> AttentionFn:
    from miniserve.attention.paged_sdpa import paged_sdpa_attention

    return paged_sdpa_attention


ATTENTION_BACKENDS: dict[str, Callable[[], AttentionFn]] = {
    "flash_attn": load_flash_attn,
    "torch": load_torch_ref,
    "sdpa": load_paged_sdpa,
}


def get_attention_fn(name: str) -> AttentionFn:
    """Resolve a backend name to its callable, importing it on first use."""
    if name not in ATTENTION_BACKENDS:
        known = ", ".join(ATTENTION_BACKENDS)
        raise KeyError(f"unknown attention backend {name!r}; known: {known}")
    return ATTENTION_BACKENDS[name]()


def flash_attn_available() -> bool:
    try:
        load_flash_attn()
    except ImportError:
        return False
    return True
