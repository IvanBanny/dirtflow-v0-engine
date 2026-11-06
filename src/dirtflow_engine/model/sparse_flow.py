"""Sparse block-state flow in the checkpoint native frame."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from . import coordinates
from . import sparse as sparse_grid
from .layers import (
    AbsolutePositionEmbedder,
    ChunkedLayerNorm32,
    FeedForwardNet,
    LayerNorm32,
    MultiHeadRMSNorm,
    TimestepEmbedder,
    zero_module,
)
from .sparse import (
    SparseGrid,
    SubMConv3d,
    VarlenPack,
    cached_pack,
    cached_position_embedding,
    downsample,
    upsample,
    varlen_cross_attention,
    varlen_self_attention,
)


def modulate(
    h: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, idx: torch.Tensor
) -> torch.Tensor:
    return h * (1 + scale[idx]) + shift[idx]


def gate_add(
    x: torch.Tensor, h: torch.Tensor, gate: torch.Tensor, idx: torch.Tensor
) -> torch.Tensor:
    return x + h * gate[idx]


def to_native(
    grid: sparse_grid.SparseGrid, system: str
) -> tuple[sparse_grid.SparseGrid, torch.Tensor | None]:
    coordinates.validate_system(system)
    if system == "minecraft_y_up":
        return (grid, None)
    cache = grid.cache
    key = ("blockstate_native", system, grid.resolution)
    if key not in cache:
        native = torch.stack(
            (
                grid.coords[:, 0],
                grid.coords[:, 1],
                grid.resolution - 1 - grid.coords[:, 3],
                grid.coords[:, 2],
            ),
            dim=1,
        )
        order = sparse_grid.sort_coords(native, grid.resolution)
        cache[key] = (native[order], order, torch.argsort(order))
    native, order, inverse = cache[key]
    return (
        sparse_grid.SparseGrid(
            native,
            grid.feats[order],
            grid.batch_size,
            grid.resolution,
            grid.cache.setdefault("native", {}),
        ),
        inverse,
    )


class SparseMultiHeadAttention(nn.Module):
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
        self,
        feats: torch.Tensor,
        lengths: torch.Tensor | VarlenPack,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        L = feats.shape[0]
        if self._type == "self":
            qkv = self.to_qkv(feats).reshape(L, 3, self.num_heads, self.head_dim)
            q, k, v = qkv.unbind(dim=1)
            if self.qk_rms_norm:
                q, k = (self.q_rms_norm(q), self.k_rms_norm(k))
            h = varlen_self_attention(q, k, v, lengths)
        else:
            if context is None:
                raise ValueError("cross attention requires context")
            B, S = context.shape[:2]
            q = self.to_q(feats).reshape(L, self.num_heads, self.head_dim)
            kv = self.to_kv(context).reshape(B, S, 2, self.num_heads, self.head_dim)
            k, v = kv.unbind(dim=2)
            if self.qk_rms_norm:
                q, k = (self.q_rms_norm(q), self.k_rms_norm(k))
            h = varlen_cross_attention(q, k, v, lengths)
        return self.to_out(h.reshape(L, self.channels))


class ModulatedSparseTransformerCrossBlock(nn.Module):
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
        self.self_attn = SparseMultiHeadAttention(
            channels, num_heads, type="self", qk_rms_norm=qk_rms_norm
        )
        self.cross_attn = SparseMultiHeadAttention(
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
        self,
        x: torch.Tensor,
        mod: torch.Tensor,
        context: torch.Tensor,
        lengths: torch.Tensor | VarlenPack,
        batch_index: torch.Tensor,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(mod).chunk(6, dim=1)
        )
        h = modulate(self.norm1(x), scale_msa, shift_msa, batch_index)
        h = self.self_attn(h, lengths)
        x = gate_add(x, h, gate_msa, batch_index)
        h = self.norm2(x)
        x = x + self.cross_attn(h, lengths, context)
        h = modulate(self.norm3(x), scale_mlp, shift_mlp, batch_index)
        h = self.mlp(h)
        x = gate_add(x, h, gate_mlp, batch_index)
        return x


class SparseResBlock3d(nn.Module):
    def __init__(
        self,
        channels: int,
        emb_channels: int,
        out_channels: int | None = None,
        *,
        resample: str | None = None,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.resample = resample
        self.norm1 = ChunkedLayerNorm32(channels, elementwise_affine=True, eps=1e-06)
        self.norm2 = ChunkedLayerNorm32(
            self.out_channels, elementwise_affine=False, eps=1e-06
        )
        self.conv1 = SubMConv3d(channels, self.out_channels, 3)
        self.conv2 = zero_module(SubMConv3d(self.out_channels, self.out_channels, 3))
        self.emb_layers = nn.Sequential(
            nn.SiLU(), nn.Linear(emb_channels, 2 * self.out_channels, bias=True)
        )
        self.skip_connection = (
            nn.Linear(channels, self.out_channels)
            if channels != self.out_channels
            else nn.Identity()
        )

    def forward(self, grid: SparseGrid, emb: torch.Tensor) -> SparseGrid:
        batch_index = grid.coords[:, 0]
        scale, shift = self.emb_layers(emb).chunk(2, dim=1)
        h = grid.replace_feats(F.silu(self.norm1(grid.feats)))
        h = self.conv1(h)
        feats = modulate(self.norm2(h.feats), scale, shift, batch_index)
        h = self.conv2(h.replace_feats(F.silu(feats)))
        skip = self.skip_connection(grid.feats)
        return grid.replace_feats(h.feats + skip)


class BlockStateFlowModel(nn.Module):
    """Sparse features [L,42] to velocity [L,8] and optional IO features [L,C]."""

    def __init__(
        self,
        *,
        resolution: int = 64,
        in_channels: int = 17,
        out_channels: int = 8,
        model_channels: int = 1024,
        cond_channels: int = 768,
        num_blocks: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_io_res_blocks: int = 2,
        io_block_channels: tuple[int, ...] = (128,),
        qk_rms_norm: bool = True,
        qk_rms_norm_cross: bool = False,
        self_conditioning: bool = False,
        coordinate_system: str = "minecraft_y_up",
    ) -> None:
        super().__init__()
        coordinates.validate_system(coordinate_system)
        self.coordinate_system = coordinate_system
        self.resolution = resolution
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.self_conditioning = self_conditioning
        self.io_channels = io_block_channels[0]
        self.t_embedder = TimestepEmbedder(model_channels)
        self.pos_embedder = AbsolutePositionEmbedder(model_channels)
        self.input_layer = nn.Linear(in_channels, io_block_channels[0])
        self.input_blocks = nn.ModuleList([])
        io_pairs = list(
            zip(io_block_channels, list(io_block_channels[1:]) + [model_channels])
        )
        for chs, next_chs in io_pairs:
            self.input_blocks.extend(
                (
                    SparseResBlock3d(chs, model_channels, out_channels=chs)
                    for _ in range(num_io_res_blocks - 1)
                )
            )
            self.input_blocks.append(
                SparseResBlock3d(
                    chs, model_channels, out_channels=next_chs, resample="down"
                )
            )
        self.blocks = nn.ModuleList(
            [
                ModulatedSparseTransformerCrossBlock(
                    model_channels,
                    cond_channels,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qk_rms_norm=qk_rms_norm,
                    qk_rms_norm_cross=qk_rms_norm_cross,
                )
                for _ in range(num_blocks)
            ]
        )
        self.out_blocks = nn.ModuleList([])
        out_pairs = list(
            zip(
                reversed(io_block_channels),
                [model_channels] + list(reversed(list(io_block_channels[1:]))),
            )
        )
        for chs, prev_chs in out_pairs:
            self.out_blocks.append(
                SparseResBlock3d(
                    prev_chs * 2, model_channels, out_channels=chs, resample="up"
                )
            )
            self.out_blocks.extend(
                (
                    SparseResBlock3d(chs * 2, model_channels, out_channels=chs)
                    for _ in range(num_io_res_blocks - 1)
                )
            )
        self.out_layer = nn.Linear(io_block_channels[0], out_channels)

    def forward(
        self,
        grid: SparseGrid,
        t: torch.Tensor,
        cond: torch.Tensor,
        t_extra: torch.Tensor | None = None,
        return_io_feats: bool = False,
    ) -> SparseGrid | tuple[SparseGrid, torch.Tensor]:
        world_grid = grid
        grid, restore = to_native(grid, self.coordinate_system)
        t_emb = self.t_embedder(t)
        if t_extra is not None:
            t_emb = t_emb + t_extra
        h = grid.replace_feats(self.input_layer(grid.feats))
        skips: list[torch.Tensor] = []
        structure: list[tuple[SparseGrid, torch.Tensor]] = []
        for block in self.input_blocks:
            if block.resample == "down":
                coarse, parent = downsample(h)
                structure.append((h, parent))
                h = coarse
            h = block(h, t_emb)
            skips.append(h.feats)
        pack = cached_pack(h)
        batch_index = h.coords[:, 0]
        h = h.replace_feats(
            h.feats
            + cached_position_embedding(
                h, self.pos_embedder, retain=not self.training
            ).to(h.feats.dtype)
        )
        for block in self.blocks:
            h = h.replace_feats(block(h.feats, t_emb, cond, pack, batch_index))
        for block, skip in zip(self.out_blocks, reversed(skips)):
            h = h.replace_feats(torch.cat([h.feats, skip], dim=1))
            if block.resample == "up":
                fine, parent = structure.pop()
                h = upsample(h, parent, fine)
            h = block(h, t_emb)
        feats = F.layer_norm(h.feats, h.feats.shape[-1:])
        out = h.replace_feats(self.out_layer(feats))
        if restore is not None:
            out = world_grid.replace_feats(out.feats[restore])
            feats = feats[restore]
        if return_io_feats:
            return (out, feats)
        return out
