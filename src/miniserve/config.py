from dataclasses import dataclass

# some basic configurations of things
import torch
import warnings

try:
    from mps_flash_attn import flash_attention

    FLASH_ATTN_AVAILABLE = True
except Exception as _:
    FLASH_ATTN_AVAILABLE = False

from .platform import resolve_device


@dataclass
class Config:
    model: str
    device: torch.device | str = "auto"
    dtype: torch.dtype | None = None

    max_model_len: int = 4096
    attention_backend: str = "auto"  # "auto" | "flash_attn" | "torch"
    block_size: int = 0

    max_num_seqs: int = 16
    max_num_batched_tokens: int = 1024
    gpu_memory_utilization: float = 0.60

    enable_prefix_caching: bool = True
    use_cuda_graphs: bool = False
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
            if self.attention_backend == "auto":
                self.attention_backend = (
                    "flash_attn" if FLASH_ATTN_AVAILABLE else "torch"
                )

            if self.attention_backend == "flash_attn" and not FLASH_ATTN_AVAILABLE:
                warnings.warn("FlashAttention unavailable; using torch attention")
                self.attention_backend = "torch"

            if self.attention_backend == "torch":
                self.use_cuda_graphs = False

        if self.block_size == 0:
            self.block_size = 256 if self.attention_backend == "flash_attn" else 16


@dataclass
class SamplingParams:
    temperature: float = 1.0  # 0 -> greedy decode
    top_p: float = 1.0
    top_k: int = -1  # -1 means off
    max_tokens: int = 64  # max tokens to generate
    ignore_eos: bool = False  # as we normally stop generating on eos token

    # testing
    forced_token_ids: list[int] | None = None  # teacher forcing to compare log probs
    return_logprobs: bool = False
