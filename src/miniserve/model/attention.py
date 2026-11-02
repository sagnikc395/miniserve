import torch
from torch import nn

from miniserve.attention.backends import AttentionFn
from miniserve.attention.forward_context import get_attn_metadata
from miniserve.model.layers import RMSNorm, rotary


class Attention(nn.Module):
    def __init__(self, cfg, layer_idx: int, attn_fn: AttentionFn):
        super().__init__()
        self.H = cfg.num_attention_heads
        self.Hkv = getattr(cfg, "num_key_value_heads", None) or self.H
        self.D = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.H

        ## qwen2 tricks as it hardcodes true, but other models expose
        ## attention_bias in config
        model_type = cfg.model_type or ""
        qkv_bias = getattr(cfg, "attention_bias", False) or model_type.startswith(
            "qwen2"
        )

        # q, k and v live in one projection so the forward pass has a single
        # matmul. The loader writes each checkpoint projection into its slice.
        self.qkv_sizes = [self.H * self.D, self.Hkv * self.D, self.Hkv * self.D]
        self.qkv_proj = nn.Linear(
            cfg.hidden_size, sum(self.qkv_sizes), bias=qkv_bias
        )
        if qkv_bias:
            # some checkpoints only carry q/v bias; the unused k slice must not
            # stay at the random init value
            with torch.no_grad():
                self.qkv_proj.bias.zero_()

        self.o_proj = nn.Linear(self.H * self.D, cfg.hidden_size, bias=False)

        self.qk_norm = model_type.startswith("qwen3")
        if self.qk_norm:
            eps = getattr(cfg, "rms_norm_eps", 1e-6)
            self.q_norm = RMSNorm(self.D, eps)
            self.k_norm = RMSNorm(self.D, eps)

        self.layer_idx = layer_idx
        self.attn_fn = attn_fn
        # [2, num_blocks, block_size, Hkv, D]; allocated by ModelRunner
        self.kv_cache: torch.Tensor | None = None

    def forward(self, x, cos, sin):
        T = x.shape[0]  # x is passed in as packed

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.qkv_sizes, dim=-1)

        q = q.view(T, self.H, self.D)
        k = k.view(T, self.Hkv, self.D)
        v = v.view(T, self.Hkv, self.D)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = rotary(q, cos, sin)
        k = rotary(k, cos, sin)

        meta = get_attn_metadata()

        if meta is None:
            # warmup / shape probing pass: no batch descriptor, no paged pool
            return self.o_proj(q.reshape(T, -1) * 0)

        # write this step's new K/V into the page pool before attention
        assert self.kv_cache is not None, (
            f"layer {self.layer_idx}: kv_cache not allocated before paged attention"
        )
        k_cache, v_cache = self.kv_cache[0], self.kv_cache[1]
        # how many mappings we are going to have
        n = meta.slot_mapping.shape[0]

        # flatten page and offset into one slot axis, so slot s is the flat
        # index produced by the scheduler: page * block_size + offset
        k_cache.view(-1, self.Hkv, self.D).index_copy_(0, meta.slot_mapping, k[:n])
        v_cache.view(-1, self.Hkv, self.D).index_copy_(0, meta.slot_mapping, v[:n])

        # attend over each request's whole context through its block table
        out = self.attn_fn(q[:n], k_cache, v_cache, meta, self.D**-0.5)

        # project it out
        return self.o_proj(out.reshape(n, -1))
