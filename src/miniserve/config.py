# import warnings
from dataclasses import dataclass
from typing import Literal

# some basic configurations of things
import torch

# from miniserve.attention.backends import flash_attn_available
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

        if self.device.type != "cuda":
            self.attention_backend = "torch"
            self.use_cuda_graphs = False
        else:
#             if self.attention_backend == "auto":
#                 self.attention_backend = (
#                     "flash_attn" if flash_attn_available() else "torch"
#                 )
#
#             if self.attention_backend == "flash_attn" and not flash_attn_available():
#                 warnings.warn("FlashAttention unavailable; using torch attention")
#                 self.attention_backend = "torch"

            if self.attention_backend == "torch":
                self.use_cuda_graphs = False
            else:
                # the paged kernel reads much larger pages than the reference path
                self.block_size = 256
