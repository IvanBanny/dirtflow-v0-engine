"""Attention and normalization with TRELLIS parameter names."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def zero_module(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class LayerNorm32(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)


class ChunkedLayerNorm32(nn.LayerNorm):
    """Normalizes rows in FP32 with at most 2^24 temporary elements per chunk."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        values = x.reshape(-1, shape[-1])  # [L,C]
        step = max(1, (1 << 24) // shape[-1])
        out = torch.empty_like(values)
        weight = self.weight.float() if self.weight is not None else None
        bias = self.bias.float() if self.bias is not None else None
        for start in range(0, len(values), step):
            part = values[start : start + step].float()
            mean = part.mean(dim=-1)
            variance = part.var(dim=-1, unbiased=False)
            scaled = (part - mean[:, None]) * torch.rsqrt(variance + self.eps)[:, None]
            if weight is not None:
                scaled = scaled * weight
            if bias is not None:
                scaled = scaled + bias
            out[start : start + step] = scaled.to(x.dtype)
        return out.reshape(shape)


class ChannelLayerNorm32(LayerNorm32):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dim = x.dim()
        x = x.permute(0, *range(2, dim), 1).contiguous()
        x = super().forward(x)
        return x.permute(0, dim - 1, *range(1, dim - 1)).contiguous()


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(
        t: torch.Tensor, dim: int, max_period: int = 10000
    ) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class AbsolutePositionEmbedder(nn.Module):
    freqs: torch.Tensor

    def __init__(self, channels: int, in_channels: int = 3) -> None:
        super().__init__()
        self.channels = channels
        self.in_channels = in_channels
        freq_dim = channels // in_channels // 2
        freqs = torch.arange(freq_dim, dtype=torch.float32) / freq_dim
        self.register_buffer("freqs", 1.0 / 10000**freqs, persistent=False)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        out = torch.outer(coords.reshape(-1).float(), self.freqs)
        out = torch.cat([torch.sin(out), torch.cos(out)], dim=-1)
        out = out.reshape(coords.shape[0], -1)
        if out.shape[1] < self.channels:
            pad = torch.zeros(
                coords.shape[0], self.channels - out.shape[1], device=out.device
            )
            out = torch.cat([out, pad], dim=-1)
        return out


class MultiHeadRMSNorm(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (F.normalize(x.float(), dim=-1) * self.gamma * self.scale).to(x.dtype)


class FeedForwardNet(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels, int(channels * mlp_ratio)),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(channels * mlp_ratio), channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        *,
        ctx_channels: int | None = None,
        type: str = "self",
        qkv_bias: bool = True,
        qk_rms_norm: bool = False,
    ) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("num_heads must divide channels")
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.ctx_channels = ctx_channels if ctx_channels is not None else channels
        self._type = type
        self.qk_rms_norm = qk_rms_norm
        if type == "self":
            self.to_qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        else:
            self.to_q = nn.Linear(channels, channels, bias=qkv_bias)
            self.to_kv = nn.Linear(self.ctx_channels, channels * 2, bias=qkv_bias)
        if qk_rms_norm:
            self.q_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
            self.k_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
        self.to_out = nn.Linear(channels, channels)

    def forward(
        self, x: torch.Tensor, context: torch.Tensor | None = None
    ) -> torch.Tensor:
        B, L, _ = x.shape
        if self._type == "self":
            qkv = self.to_qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
            q, k, v = qkv.unbind(dim=2)
        else:
            if context is None:
                raise ValueError("cross attention requires context")
            q = self.to_q(x).reshape(B, L, self.num_heads, self.head_dim)
            kv = self.to_kv(context).reshape(
                B, context.shape[1], 2, self.num_heads, self.head_dim
            )
            k, v = kv.unbind(dim=2)
        if self.qk_rms_norm:
            q = self.q_rms_norm(q)
            k = self.k_rms_norm(k)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        h = F.scaled_dot_product_attention(q, k, v)
        h = h.transpose(1, 2).reshape(B, L, self.channels)
        return self.to_out(h)


class ModulatedTransformerCrossBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        *,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-06)
        self.norm2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-06)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-06)
        self.self_attn = MultiHeadAttention(
            channels, num_heads, type="self", qk_rms_norm=qk_rms_norm
        )
        self.cross_attn = MultiHeadAttention(
            channels,
            num_heads,
            ctx_channels=ctx_channels,
            type="cross",
            qk_rms_norm=qk_rms_norm_cross,
        )
        self.mlp = FeedForwardNet(channels, mlp_ratio=mlp_ratio)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(channels, 6 * channels, bias=True)
        )

    def forward(
        self, x: torch.Tensor, mod: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(mod).chunk(6, dim=1)
        )
        h = self.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        h = self.self_attn(h)
        x = x + h * gate_msa.unsqueeze(1)
        h = self.norm2(x)
        x = x + self.cross_attn(h, context)
        h = self.norm3(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h = self.mlp(h)
        x = x + h * gate_mlp.unsqueeze(1)
        return x
