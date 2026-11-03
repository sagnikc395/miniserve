import torch
from torch import nn

from miniserve.model.layers import DecoderLayer, RMSNorm, add_rmsnorm


class Model(nn.Module):
    def __init__(self, cfg, max_position: int, attn_fn=None):
        super().__init__()

        if attn_fn is None:
            from miniserve.attention.backends import get_attention_fn

            attn_fn = get_attention_fn("torch")

        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)

        self.layers = nn.ModuleList(
            DecoderLayer(cfg, i, attn_fn) for i in range(cfg.num_hidden_layers)
        )
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.tie = getattr(cfg, "tie_word_embeddings", False)

        if not self.tie:
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        # RoPE table, indexed by absolute position
        rope = (
            getattr(cfg, "rope_parameters", None)
            or getattr(cfg, "rope_scaling", None)
            or {}
        )

        theta = getattr(cfg, "rope_theta", None) or rope.get("rope_theta", 10000.0)
        D = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        # build the table in float64 on the CPU (MPS has no float64 at all),
        # then move the float32 result to wherever the model is being built
        device = torch.empty(0).device
        inv_freq = 1.0 / theta ** (
            torch.arange(0, D, 2, dtype=torch.float64, device="cpu") / D
        )
        freqs = torch.outer(
            torch.arange(max_position, dtype=torch.float64, device="cpu"), inv_freq
        )
        self.register_buffer("cos", freqs.cos().float().to(device), persistent=False)
        self.register_buffer("sin", freqs.sin().float().to(device), persistent=False)

    def forward(self, input_ids, positions):
        cos, sin = self.cos[positions], self.sin[positions]
        x, residual = self.embed_tokens(input_ids), None
        for layer in self.layers:
            x, residual = layer(x, residual, cos, sin)
        return add_rmsnorm(self.norm, x, residual)[0]

    def logits(self, hidden):
        w = self.embed_tokens.weight if self.tie else self.lm_head.weight
        return hidden @ w.t()
