import warnings
from dataclasses import dataclass
from typing import Literal

# some basic configurations of things
import torch

from miniserve.platform import resolve_device


@dataclass
class EngineConfig:
    model: str
    device: torch.device | str = "auto"
    dtype: torch.dtype | None = None

    max_model_len: int = 4096
    attention_backend: Literal["auto", "sdpa", "torch", "flash_attn"] = "auto"
    kv_layout: Literal["dense_slots", "paged"] = "paged"
    decode_backend: Literal["eager", "compiled", "metal"] = "eager"
    block_size: int = 16

    max_num_seqs: int = 16
    max_num_batched_tokens: int = 1024
    gpu_memory_utilization: float = 0.60

    enable_prefix_caching: bool = True
    use_cuda_graphs: bool = False  # CUDA only
    seed: int = 0

    def __post_init__(self):
        self.device = resolve_device(str(self.device))

        if self.dtype is None:
            if self.device.type == "mps":
                # A good compatibility default on Apple Silicon.
                self.dtype = torch.float16
            elif self.device.type == "cuda":
                self.dtype = torch.bfloat16
            else:
                self.dtype = torch.float32

        if self.attention_backend == "auto":
            # SDPA is the fast path on both CUDA and MPS. The torch reference
            # is only the default on CPU, where it is easier to debug.
            self.attention_backend = "torch" if self.device.type == "cpu" else "sdpa"

        if self.attention_backend == "flash_attn":
            warnings.warn("FlashAttention is not available in this build; using sdpa")
            self.attention_backend = "sdpa"

        if self.device.type != "cuda":
            self.use_cuda_graphs = False
        elif self.attention_backend == "torch":
            # the reference backend has a Python loop per request, so there is
            # nothing stable to capture
            self.use_cuda_graphs = False

        if self.max_num_batched_tokens < self.max_num_seqs:
            raise ValueError(
                "max_num_batched_tokens must be at least max_num_seqs "
                "(every running request needs at least one token per step)"
            )
