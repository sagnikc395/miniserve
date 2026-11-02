import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        return (
            xf
            * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
            * (self.weight.float())
        ).to(x.dtype)


def rotary(x, cos, sin):
    x1, x2 = x.float().chunk(2, dim=-1)
    cos, sin = cos[:, None], sin[:, None]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1).to(x.dtype)


def add_rmsnorm(norm, x, residual):
    # fused residual add + RMSNorm: the first layer has no residual yet, every
    # later one folds the incoming stream in before normalising.
    # returns (normalised activations, the residual stream to carry forward)
    if residual is None:
        return norm(x), x
    residual = residual + x
    return norm(residual), residual


def _activation(name: str | None):
    return {
        "silu": F.silu,
        "swish": F.silu,
        "gelu": F.gelu,
        "gelu_pytorch_tanh": lambda x: F.gelu(x, approximate="tanh"),
    }.get(name or "silu", F.silu)


class MLP(nn.Module):
    """SwiGLU feed-forward, with gate/up fused into one projection.

    The loader writes checkpoint `gate_proj` and `up_proj` tensors into the two
    row ranges of `gate_up_proj`, which keeps the layer to two matmuls and
    matches the key layout the loader already understands.
    """

    def __init__(self, hidden_size, intermediate_size, act="silu"):
        super().__init__()
        self.intermediate_size = intermediate_size
        self.gate_up_sizes = [intermediate_size, intermediate_size]
        self.gate_up_proj = nn.Linear(
            hidden_size, 2 * intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act = _activation(act)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(self.act(gate) * up)


class MoE(nn.Module):
    """Sparse mixture of experts with the same fused layout as `MLP`.

    This is deliberately the simple path: route each token independently and
    run only the selected experts. It is slow compared to a grouped GEMM, but
    it is correct and keeps the loader's per-expert tensor interface.
    """

    def __init__(self, cfg):
        super().__init__()
        self.hidden_size = cfg.hidden_size
        self.intermediate_size = getattr(
            cfg, "moe_intermediate_size", cfg.intermediate_size
        )
        self.num_experts = cfg.num_experts
        self.top_k = min(
            getattr(cfg, "num_experts_per_tok", 2), self.num_experts
        )
        self.norm_topk_prob = getattr(cfg, "norm_topk_prob", False)
        self.act = _activation(getattr(cfg, "hidden_act", "silu"))

        # [E, 2 * inter, hidden] and [E, hidden, inter]
        self.w13 = nn.Parameter(
            torch.empty(
                self.num_experts, 2 * self.intermediate_size, self.hidden_size
            )
        )
        self.w2 = nn.Parameter(
            torch.empty(
                self.num_experts, self.hidden_size, self.intermediate_size
            )
        )
        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)

        shared = getattr(cfg, "shared_expert_intermediate_size", 0) or 0
        if shared:
            self.shared_expert = MLP(
                self.hidden_size, shared, getattr(cfg, "hidden_act", "silu")
            )
            self.shared_expert_gate = nn.Linear(self.hidden_size, 1, bias=False)
        else:
            self.shared_expert = None
            self.shared_expert_gate = None

    def load_expert(self, expert_idx, proj, fh, key, tp_rank=0, tp_size=1):
        """Copy one checkpoint expert tensor into this layer's stacked weights."""
        from miniserve.model.loader import load_shard

        weight = load_shard(fh, key, tp_rank, tp_size)
        with torch.no_grad():
            if proj == "down_proj":
                self.w2[expert_idx].copy_(weight)
            else:
                start = 0 if proj == "gate_proj" else self.intermediate_size
                self.w13[
                    expert_idx, start : start + self.intermediate_size
                ].copy_(weight)
        return True

    def forward(self, x):
        # [T, E]
        scores = F.softmax(self.gate(x).float(), dim=-1).to(x.dtype)
        topk_w, topk_idx = scores.topk(self.top_k, dim=-1)

        if self.norm_topk_prob:
            topk_w = topk_w / topk_w.sum(-1, keepdim=True).clamp_min(1e-9)

        out = torch.zeros_like(x)
        for expert in range(self.num_experts):
            selected = topk_idx == expert
            if not selected.any():
                continue

            # a token can select an expert at most once, so any() is enough
            rows = selected.any(dim=-1).nonzero(as_tuple=True)[0]
            weights = (topk_w * selected).sum(-1)[rows]

            h = x[rows]
            gate, up = (h @ self.w13[expert].t()).chunk(2, dim=-1)
            expert_out = (self.act(gate) * up) @ self.w2[expert].t()
            out[rows] += expert_out * weights[:, None]

        if self.shared_expert is not None:
            out = out + torch.sigmoid(self.shared_expert_gate(x)) * (
                self.shared_expert(x)
            )
        return out


def build_mlp(cfg):
    if getattr(cfg, "num_experts", 0):
        return MoE(cfg)
    return MLP(
        cfg.hidden_size,
        cfg.intermediate_size,
        getattr(cfg, "hidden_act", "silu"),
    )


class DecoderLayer(nn.Module):
    def __init__(self, cfg, layer_idx, attn_fn):
        super().__init__()
        from miniserve.model.attention import Attention

        self.self_attn = Attention(cfg, layer_idx, attn_fn)
        self.mlp = build_mlp(cfg)
        eps = getattr(cfg, "rms_norm_eps", 1e-6)
        self.input_layernorm = RMSNorm(cfg.hidden_size, eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, eps)

    def forward(self, x, residual, cos, sin):
        h, residual = add_rmsnorm(self.input_layernorm, x, residual)
        x = self.self_attn(h, cos, sin)

        h, residual = add_rmsnorm(self.post_attention_layernorm, x, residual)
        x = self.mlp(h)
        return x, residual
