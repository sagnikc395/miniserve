import torch
from torch import nn

from miniserve.attention.backends import AttentionFn
from miniserve.attention.forward_context import get_attn_metadata
from miniserve.model.layers import RMSNorm


class Attention(nn.Module):
    def __init__(self, cfg, layer_idx: int, attn_fn: AttentionFn):
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
            self.k_norm = RMSNorm(self.D, cfg.rms_norm_eps)

        self.layer_idx = layer_idx
        self.attn_fn = attn_fn
        self.kv_cache: torch.Tensor | None = None

    def forward(self, x, cos, sin):
        T = x.shape[0]  # x is passed in as packed

        q = self.q_proj(x).view(T, self.H, self.D)
        k = self.k_proj(x).view(T, self.Hkv, self.D)
        v = self.v_proj(x).view(T, self.Hkv, self.D)

        meta = get_attn_metadata()

        if meta is None:
            # warmup / shape probing pass: no batch descriptor, no paged pool
            return self.o_proj(q.reshape(T, -1) * 0)

        # write this current step's new K/V into the page pool before attention

        k_cache, v_cache = self.kv_cache[0], self.kv_cache[1]
        n = meta.slot_mapping.shape[0]

        k_cache.view(-1, self.Hkv, self.D).index_copy_(0, meta.slot_mapping, k[:n])
        v_cache.view(-1, self.Hkv, self.D).index_copy_(0, meta.slot_mapping, v[:n])

        # attend over each request's whole context through its block table
        out = self.attn_fn(q[:n], k_cache, v_cache, meta, self.D**-0.5)

        return self.o_proj(out.reshape(n, -1))
