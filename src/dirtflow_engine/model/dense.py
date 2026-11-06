"""Dense occupancy decoder and structure flow."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from . import coordinates
from .layers import (
    AbsolutePositionEmbedder,
    ChannelLayerNorm32,
    ModulatedTransformerCrossBlock,
    TimestepEmbedder,
    zero_module,
)


def pixel_shuffle_3d(x: torch.Tensor, factor: int) -> torch.Tensor:
    B, C, H, W, D = x.shape
    C_ = C // factor**3
    x = x.reshape(B, C_, factor, factor, factor, H, W, D)
    x = x.permute(0, 1, 5, 2, 6, 3, 7, 4)
    return x.reshape(B, C_, H * factor, W * factor, D * factor)


class ResBlock3d(nn.Module):
    def __init__(self, channels: int, out_channels: int | None = None) -> None:
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.norm1 = ChannelLayerNorm32(channels)
        self.norm2 = ChannelLayerNorm32(self.out_channels)
        self.conv1 = nn.Conv3d(channels, self.out_channels, 3, padding=1)
        self.conv2 = zero_module(
            nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1)
        )
        self.skip_connection = (
            nn.Conv3d(channels, self.out_channels, 1)
            if channels != self.out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip_connection(x)


class DownsampleBlock3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, 2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpsampleBlock3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels * 8, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return pixel_shuffle_3d(self.conv(x), 2)


class SparseStructureDecoder(nn.Module):
    """Native latent [B,8,16,16,16] to world occupancy logits [B,1,64,64,64]."""

    def __init__(
        self,
        out_channels: int = 1,
        latent_channels: int = 8,
        num_res_blocks: int = 2,
        channels: tuple[int, ...] = (512, 128, 32),
        num_res_blocks_middle: int = 2,
        structure_coordinate_system: str = "minecraft_y_up",
    ) -> None:
        super().__init__()
        coordinates.validate_system(structure_coordinate_system)
        self.structure_coordinate_system = structure_coordinate_system
        self.input_layer = nn.Conv3d(latent_channels, channels[0], 3, padding=1)
        self.middle_block = nn.Sequential(
            *[ResBlock3d(channels[0]) for _ in range(num_res_blocks_middle)]
        )
        self.blocks = nn.ModuleList([])
        for i, ch in enumerate(channels):
            self.blocks.extend((ResBlock3d(ch, ch) for _ in range(num_res_blocks)))
            if i < len(channels) - 1:
                self.blocks.append(UpsampleBlock3d(ch, channels[i + 1]))
        self.out_layer = nn.Sequential(
            ChannelLayerNorm32(channels[-1]),
            nn.SiLU(),
            nn.Conv3d(channels[-1], out_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        h = self.middle_block(h)
        for block in self.blocks:
            h = block(h)
        return coordinates.to_world_grid(
            self.out_layer(h), self.structure_coordinate_system
        )


class LatentNorm(nn.Module):
    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self, channels: int = 8) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(channels))
        self.register_buffer("std", torch.ones(channels))

    def _view(self, ref: torch.Tensor, stat: torch.Tensor) -> torch.Tensor:
        return stat.view(1, -1, *[1] * (ref.dim() - 2)).to(ref.dtype)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return (z - self._view(z, self.mean)) / self._view(z, self.std)

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        return z * self._view(z, self.std) + self._view(z, self.mean)


class StructureFlowModel(nn.Module):
    """Latent, context and self-condition [B,32,R,R,R] to velocity [B,8,R,R,R]."""

    pos_emb: torch.Tensor

    def __init__(
        self,
        *,
        resolution: int = 16,
        in_channels: int = 24,
        out_channels: int = 8,
        model_channels: int = 1024,
        cond_channels: int = 768,
        num_blocks: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        qk_rms_norm: bool = True,
        qk_rms_norm_cross: bool = False,
        self_conditioning: bool = False,
    ) -> None:
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.self_conditioning = self_conditioning
        self.t_embedder = TimestepEmbedder(model_channels)
        pos_embedder = AbsolutePositionEmbedder(model_channels)
        coords = torch.stack(
            torch.meshgrid(*[torch.arange(resolution)] * 3, indexing="ij"), dim=-1
        ).reshape(-1, 3)
        self.register_buffer("pos_emb", pos_embedder(coords))
        self.input_layer = nn.Linear(in_channels, model_channels)
        self.blocks = nn.ModuleList(
            [
                ModulatedTransformerCrossBlock(
                    model_channels,
                    cond_channels,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qk_rms_norm=qk_rms_norm,
                    qk_rms_norm_cross=qk_rms_norm_cross,
                )
                for i in range(num_blocks)
            ]
        )
        self.out_layer = nn.Linear(model_channels, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        t_extra: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = x.shape[0]
        h = x.reshape(B, self.in_channels, -1).permute(0, 2, 1)
        h = self.input_layer(h)
        h = h + self.pos_emb[None].to(h.dtype)
        t_emb = self.t_embedder(t)
        if t_extra is not None:
            t_emb = t_emb + t_extra
        for block in self.blocks:
            h = block(h, t_emb, cond)
        h = F.layer_norm(h, h.shape[-1:])
        h = self.out_layer(h)
        r = self.resolution
        return h.permute(0, 2, 1).reshape(B, self.out_channels, r, r, r)
