from torch import nn


class Attention(nn.Module):
    def __init__(self, cfg, layer_idx: int):
        super().__init__()
        self.H = cfg.num_attention_heads
        self.Hkv = cfg.num_key_value_heads
        self.D = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.H

        ##qwen2 tricks as it hardcodes true, but other models expose attention_bias in config
        qkv_bias = getattr(cfg, "attention_bias", False) or cfg.model_type == "qwen2"

        self.q_proj = nn.Linear(cfg.hidden_size, self.H * self.D, bias=qkv_bias)
        self.k_proj = nn.Linear(cfg.hidden_size, self.Hkv * self.D, bias=qkv_bias)
        self.v_proj = nn.Linear(cfg.hidden_size, self.Hkv * self.D, bias=qkv_bias)

        self.o_proj = nn.Linear(self.H * self.D, cfg.hidden_size, bias=False)

        self.qk_norm = cfg.model_type == "qwen3"  # as qwen3 normalizes q and k per head
        if self.qk_norm:
            self.q_norm = RMSNorm(self.D, cfg.rms_norm_eps)
