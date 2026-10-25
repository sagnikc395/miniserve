from dataclasses import dataclass

# some basic configurations of things
import torch
import warnings

try:
    from mps_flash_attn import flash_attention

    FLASH_ATTN_AVAILABLE = True
except:
    FLASH_ATTN_AVAILABLE = False


@dataclass
class Config:
    model: str
    dtype: torch.dtype = torch.float16  # bfloat16 inference by default
    max_model_len: str = 4096  # max prompt + output we accept
    attention_backend: str = "sdpa"  # other alternatives are "torch" (for reference)
    device: str = "mps"

    # tokens per kv-cache page
    # 0- for backend , flash-atten only supports pages that are multiples of 256, vLLM actually uses its own fork for smaller pages (more efficient if we use shorter prompts on average)
    block_size: int = 16
    max_num_seqs: int = 64  # in a batch, max requests, unified memory limits

    # token per budget per step
    max_num_batched_tokens: int = 2048
    gpu_memory_utilization: float = 0.80
    enable_prefix_caching: bool = True
    use_cuda_graphs: bool = False  # set to True on cuda based devices
    seed: int = 0

    def __post_init__(self):
        if not torch.backends.mps.is_available():
            warnings.warn("MPS device not available, falling back to CPU.")
            self.device = "cpu"
