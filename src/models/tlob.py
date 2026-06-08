"""TLOB: a dual-attention transformer for limit-order-book forecasting (Berti & Kasneci, 2025).

The idea that distinguishes TLOB from a vanilla transformer is *dual attention*: each block attends along the
**temporal** axis (how an order book evolves over time) and along the **feature** axis (relationships across
the 40 price/size levels), rather than flattening the two. We embed every scalar of the ``(time, feature)``
grid to a model dimension, add separable temporal and feature positional embeddings, then alternate temporal
and feature self-attention with an MLP mixer. Input/output match DeepLOB exactly so the two are
drop-in comparable: ``(B, 1, 100, 40) -> (B, 3)``.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange

__all__ = ["TLOB", "TLOBBlock"]


class TLOBBlock(nn.Module):
    """One dual-attention block: temporal attention, then feature attention, then an MLP (all pre-norm)."""

    def __init__(self, dim: int, heads: int, mlp_ratio: int = 2, dropout: float = 0.0):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by heads ({heads})")
        self.norm_t = nn.LayerNorm(dim)
        self.attn_t = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_f = nn.LayerNorm(dim)
        self.attn_f = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_m = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F, d)
        b, t, f, _ = x.shape
        # Temporal attention: sequences along T, batched over (B, F).
        xt = rearrange(x, "b t f d -> (b f) t d")
        h = self.norm_t(xt)
        xt = xt + self.attn_t(h, h, h, need_weights=False)[0]
        x = rearrange(xt, "(b f) t d -> b t f d", b=b, f=f)
        # Feature attention: sequences along F, batched over (B, T).
        xf = rearrange(x, "b t f d -> (b t) f d")
        h = self.norm_f(xf)
        xf = xf + self.attn_f(h, h, h, need_weights=False)[0]
        x = rearrange(xf, "(b t) f d -> b t f d", b=b, t=t)
        # MLP mixer.
        return x + self.mlp(self.norm_m(x))


class TLOB(nn.Module):
    """Dual-attention transformer over the (time x feature) order-book grid."""

    def __init__(
        self,
        n_classes: int = 3,
        n_features: int = 40,
        seq_len: int = 100,
        dim: int = 40,
        depth: int = 2,
        heads: int = 4,
        mlp_ratio: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed = nn.Linear(1, dim)
        self.t_pos = nn.Parameter(torch.randn(1, seq_len, 1, dim) * 0.02)
        self.f_pos = nn.Parameter(torch.randn(1, 1, n_features, dim) * 0.02)
        self.blocks = nn.ModuleList(
            [TLOBBlock(dim, heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, T, F)
        x = x.squeeze(1).unsqueeze(-1)        # (B, T, F, 1)
        x = self.embed(x)                     # (B, T, F, d)
        x = x + self.t_pos + self.f_pos       # separable positional embeddings
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = x.mean(dim=(1, 2))                # global pool over time and features -> (B, d)
        return self.head(x)                   # (B, n_classes)


if __name__ == "__main__":
    model = TLOB()
    dummy = torch.randn(8, 1, 100, 40)
    print("output shape:", tuple(model(dummy).shape))  # expect (8, 3)
