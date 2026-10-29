from torch import nn
import torch


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


class DecoderLayer(nn.Module):
    pass
