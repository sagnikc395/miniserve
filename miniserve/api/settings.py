"""Environment-driven defaults for `miniserve serve`."""

import os
from dataclasses import dataclass


@dataclass
class Settings:
    model: str = "Qwen/Qwen3-0.6B"
    host: str = "127.0.0.1"
    port: int = 8000
    device: str = "auto"
    max_model_len: int = 4096
    max_num_seqs: int = 16
    max_num_batched_tokens: int = 1024
    gpu_memory_utilization: float = 0.60
    attention_backend: str = "auto"
    block_size: int = 16
    enable_prefix_caching: bool = True
    use_cuda_graphs: bool = False
    log_level: str = "info"

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ

        def flag(name: str, default: bool) -> bool:
            value = env.get(name)
            if value is None:
                return default
            return value.lower() not in ("0", "false", "no", "off")

        return cls(
            model=env.get("MINISERVE_MODEL", cls.model),
            host=env.get("MINISERVE_HOST", cls.host),
            port=int(env.get("MINISERVE_PORT", cls.port)),
            device=env.get("MINISERVE_DEVICE", cls.device),
            max_model_len=int(env.get("MINISERVE_MAX_MODEL_LEN", cls.max_model_len)),
            max_num_seqs=int(env.get("MINISERVE_MAX_NUM_SEQS", cls.max_num_seqs)),
            max_num_batched_tokens=int(
                env.get(
                    "MINISERVE_MAX_NUM_BATCHED_TOKENS",
                    cls.max_num_batched_tokens,
                )
            ),
            gpu_memory_utilization=float(
                env.get(
                    "MINISERVE_GPU_MEMORY_UTILIZATION",
                    cls.gpu_memory_utilization,
                )
            ),
            attention_backend=env.get(
                "MINISERVE_ATTENTION_BACKEND", cls.attention_backend
            ),
            block_size=int(env.get("MINISERVE_BLOCK_SIZE", cls.block_size)),
            enable_prefix_caching=flag(
                "MINISERVE_PREFIX_CACHING", cls.enable_prefix_caching
            ),
            use_cuda_graphs=flag("MINISERVE_CUDA_GRAPHS", cls.use_cuda_graphs),
            log_level=env.get("MINISERVE_LOG_LEVEL", cls.log_level),
        )
