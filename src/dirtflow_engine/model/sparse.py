"""Sparse geometry and bounded gather convolution on active voxels."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SparseGrid:
    """Sorted coordinates [L,4] (batch,x,y,z), features [L,C].

    Coordinates are immutable for the lifetime of the shared geometry cache.
    replace_feats retains that cache across Euler steps.
    """

    coords: torch.Tensor
    feats: torch.Tensor
    batch_size: int
    resolution: int
    cache: dict = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.coords.ndim != 2 or self.coords.shape[1] != 4:
            raise ValueError(f"coords must be [L, 4], got {tuple(self.coords.shape)}")
        if self.feats.shape[0] != self.coords.shape[0]:
            raise ValueError("coords/feats length mismatch")

    def __len__(self) -> int:
        return self.coords.shape[0]

    @property
    def channels(self) -> int:
        return self.feats.shape[1]

    def lengths(self) -> torch.Tensor:
        return torch.bincount(self.coords[:, 0], minlength=self.batch_size)

    def replace_feats(self, feats: torch.Tensor) -> SparseGrid:
        return SparseGrid(
            self.coords, feats, self.batch_size, self.resolution, self.cache
        )

    def to(self, device: torch.device | str) -> SparseGrid:
        return SparseGrid(
            self.coords.to(device),
            self.feats.to(device),
            self.batch_size,
            self.resolution,
        )


def sort_coords(coords: torch.Tensor, resolution: int) -> torch.Tensor:
    keys = (
        (coords[:, 0] * resolution + coords[:, 1]) * resolution + coords[:, 2]
    ) * resolution + coords[:, 3]
    return torch.argsort(keys)


def from_dense_mask(
    mask: torch.Tensor, feats_dense: torch.Tensor | None = None
) -> SparseGrid:
    if mask.ndim != 4:
        raise ValueError(f"mask must be [B, X, Y, Z], got {tuple(mask.shape)}")
    batch_size, resolution = (mask.shape[0], mask.shape[1])
    coords = mask.nonzero()
    if feats_dense is not None:
        feats = feats_dense[mask]
    else:
        feats = torch.zeros(coords.shape[0], 0, device=mask.device, dtype=torch.float32)
    return SparseGrid(coords, feats, batch_size, resolution)


def to_dense(grid: SparseGrid, fill: float = 0.0) -> torch.Tensor:
    resolution = grid.resolution
    out = torch.full(
        (grid.batch_size, resolution, resolution, resolution, grid.channels),
        fill,
        device=grid.feats.device,
        dtype=grid.feats.dtype,
    )
    b, x, y, z = grid.coords.unbind(dim=1)
    out[b, x, y, z] = grid.feats
    return out


def coord_keys(coords: torch.Tensor, resolution: int) -> torch.Tensor:
    b, x, y, z = coords.unbind(dim=1)
    keys = ((b * resolution + x) * resolution + y) * resolution + z
    inside = (
        (x >= 0)
        & (x < resolution)
        & (y >= 0)
        & (y < resolution)
        & (z >= 0)
        & (z < resolution)
    )
    return torch.where(inside, keys, torch.full_like(keys, -1))


class CoordIndex:
    def __init__(self, coords: torch.Tensor, resolution: int) -> None:
        keys = coord_keys(coords, resolution)
        self.sorted_keys, self.perm = torch.sort(keys)
        self.resolution = resolution
        self.neighbors: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def lookup(self, query_coords: torch.Tensor) -> torch.Tensor:
        if self.sorted_keys.numel() == 0:
            return torch.full(
                (query_coords.shape[0],),
                -1,
                device=query_coords.device,
                dtype=torch.long,
            )
        queries = coord_keys(query_coords, self.resolution)
        pos = torch.searchsorted(self.sorted_keys, queries)
        pos = pos.clamp(max=self.sorted_keys.shape[0] - 1)
        hit = (self.sorted_keys[pos] == queries) & (queries >= 0)
        rows = self.perm[pos]
        return torch.where(hit, rows, torch.full_like(rows, -1))


def _col_chunk(total_rows: int, channels: int) -> int:
    if total_rows * channels <= (1 << 26):
        return channels
    return max(1, (1 << 26) // max(total_rows, 1))


def _offset_group(total_rows: int, c_in: int, c_out: int, k3: int) -> int:
    per_offset = max(total_rows * (2 * c_in + c_out), 1)
    return max(1, min(k3, (1 << 26) // per_offset))


def _convolve(
    feats: torch.Tensor, weight: torch.Tensor, rows: torch.Tensor, hit: torch.Tensor
) -> torch.Tensor:
    weight = weight.to(feats.dtype)
    L, c_in = feats.shape
    k3, c_out = (weight.shape[0], weight.shape[2])
    cols = _col_chunk(L, c_in)
    out = feats.new_zeros(L, c_out)
    if cols == c_in:
        g = _offset_group(L, c_in, c_out, k3)
        for k0 in range(0, k3, g):
            idx = rows[k0 : k0 + g].reshape(-1)
            gathered = feats.index_select(0, idx).view(-1, L, c_in)
            gathered = gathered * hit[k0 : k0 + g].unsqueeze(-1).to(feats.dtype)
            out += torch.bmm(gathered, weight[k0 : k0 + g]).sum(0)
    else:
        for k in range(k3):
            mask = hit[k].unsqueeze(1).to(feats.dtype)
            for c0 in range(0, c_in, cols):
                gathered = feats[:, c0 : c0 + cols].index_select(0, rows[k])
                out += gathered * mask @ weight[k, c0 : c0 + cols]
    return out


class SubMConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        k3 = kernel_size**3
        weight = torch.empty(k3, in_channels, out_channels)
        nn.init.kaiming_uniform_(weight.view(k3 * in_channels, out_channels), a=5**0.5)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, grid: SparseGrid) -> SparseGrid:
        coords, feats = (grid.coords, grid.feats)
        index = grid.cache.get("index")
        if index is None:
            index = grid.cache["index"] = CoordIndex(coords, grid.resolution)
        neighbors = index.neighbors.get(self.kernel_size)
        if neighbors is None:
            r = (
                torch.arange(self.kernel_size, device=coords.device)
                - self.kernel_size // 2
            )
            offsets = torch.cartesian_prod(r, r, r)
            k3, length = (offsets.shape[0], coords.shape[0])
            shifted = coords.unsqueeze(0).expand(k3, -1, -1).clone()
            shifted[:, :, 1:] += offsets.unsqueeze(1)
            rows = index.lookup(shifted.reshape(-1, 4)).view(k3, length)
            hit = rows >= 0
            rows = rows.clamp(min=0)
            neighbors = index.neighbors[self.kernel_size] = (rows, hit)
        rows, hit = neighbors
        out = _convolve(feats, self.weight, rows, hit)
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return grid.replace_feats(out)


def _pool_topology(
    grid: SparseGrid, factor: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cache = grid.cache
    key = ("pool", factor, grid.resolution)
    topology = cache.get(key)
    if topology is not None:
        return topology
    coarse_res = grid.resolution // factor
    parent_coords = grid.coords.clone()
    parent_coords[:, 1:] = torch.div(
        parent_coords[:, 1:], factor, rounding_mode="floor"
    )
    keys = (
        (parent_coords[:, 0] * coarse_res + parent_coords[:, 1]) * coarse_res
        + parent_coords[:, 2]
    ) * coarse_res + parent_coords[:, 3]
    unique_keys, parent_index = torch.unique(keys, sorted=True, return_inverse=True)
    counts = torch.bincount(parent_index, minlength=unique_keys.shape[0]).clamp(min=1)
    z = unique_keys % coarse_res
    rest = torch.div(unique_keys, coarse_res, rounding_mode="floor")
    y = rest % coarse_res
    rest = torch.div(rest, coarse_res, rounding_mode="floor")
    x = rest % coarse_res
    b = torch.div(rest, coarse_res, rounding_mode="floor")
    coarse_coords = torch.stack([b, x, y, z], dim=1)
    topology = cache[key] = (coarse_coords, parent_index, counts)
    return topology


def downsample(grid: SparseGrid, factor: int = 2) -> tuple[SparseGrid, torch.Tensor]:
    coarse_coords, parent_index, counts = _pool_topology(grid, factor)
    sums = grid.feats.new_zeros(coarse_coords.shape[0], grid.channels)
    sums.index_add_(0, parent_index, grid.feats)
    means = sums / counts.unsqueeze(1).to(sums.dtype)
    coarse = SparseGrid(
        coarse_coords,
        means,
        grid.batch_size,
        grid.resolution // factor,
        grid.cache.setdefault(("coarse", factor), {}),
    )
    return (coarse, parent_index)


def upsample(
    coarse: SparseGrid, parent_index: torch.Tensor, fine: SparseGrid
) -> SparseGrid:
    return fine.replace_feats(coarse.feats[parent_index])


@dataclass(frozen=True)
class VarlenPack:
    lengths: torch.Tensor
    batch_size: int
    lmax: int
    lmin: int
    total: int
    flat_index: torch.Tensor
    valid: torch.Tensor
    offsets: torch.Tensor

    @property
    def uniform(self) -> bool:
        return self.total == self.batch_size * self.lmax

    def pad(self, feats: torch.Tensor) -> torch.Tensor:
        if self.uniform:
            return feats.view(self.batch_size, self.lmax, *feats.shape[1:])
        padded = feats.new_zeros(self.batch_size * self.lmax, *feats.shape[1:])
        padded = padded.index_copy(0, self.flat_index, feats)
        return padded.view(self.batch_size, self.lmax, *feats.shape[1:])

    def unpad(self, padded: torch.Tensor) -> torch.Tensor:
        if self.uniform:
            return padded.reshape(self.total, *padded.shape[2:])
        flat = padded.reshape(self.batch_size * self.lmax, *padded.shape[2:])
        return flat.index_select(0, self.flat_index)


def build_varlen_pack(lengths: torch.Tensor) -> VarlenPack:
    batch_size = lengths.shape[0]
    lmin, lmax = (
        (int(value) for value in torch.stack((lengths.min(), lengths.max())).tolist())
        if batch_size
        else (0, 0)
    )
    ends = torch.cumsum(lengths, dim=0)
    offsets = ends - lengths
    batch_index = torch.repeat_interleave(
        torch.arange(batch_size, device=lengths.device), lengths
    )
    positions = (
        torch.arange(batch_index.shape[0], device=lengths.device) - offsets[batch_index]
    )
    flat_index = batch_index * lmax + positions
    valid = torch.arange(lmax, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(
        1
    )
    return VarlenPack(
        lengths=lengths,
        batch_size=batch_size,
        lmax=lmax,
        lmin=lmin,
        total=int(batch_index.shape[0]),
        flat_index=flat_index,
        valid=valid,
        offsets=F.pad(ends, (1, 0)),
    )


def _as_pack(lengths: torch.Tensor | VarlenPack) -> VarlenPack:
    if isinstance(lengths, VarlenPack):
        return lengths
    return build_varlen_pack(lengths)


def varlen_self_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lengths: torch.Tensor | VarlenPack,
) -> torch.Tensor:
    pack = _as_pack(lengths)
    if q.is_cuda and not pack.uniform and pack.total >= 4096:
        qn, kn, vn = [
            torch.nested.nested_tensor_from_jagged(
                t, pack.offsets, min_seqlen=pack.lmin, max_seqlen=pack.lmax
            ).transpose(1, 2)
            for t in (q, k, v)
        ]
        return F.scaled_dot_product_attention(qn, kn, vn).transpose(1, 2).values()
    if pack.uniform:
        q_p, k_p, v_p = (pack.pad(t).transpose(1, 2) for t in (q, k, v))
        out = F.scaled_dot_product_attention(q_p, k_p, v_p)
        return pack.unpad(out.transpose(1, 2))
    q_p, k_p, v_p = (pack.pad(t).transpose(1, 2) for t in (q, k, v))
    attn_mask = pack.valid[:, None, None, :]
    out = F.scaled_dot_product_attention(q_p, k_p, v_p, attn_mask=attn_mask)
    return pack.unpad(out.transpose(1, 2))


def varlen_cross_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lengths: torch.Tensor | VarlenPack,
) -> torch.Tensor:
    pack = _as_pack(lengths)
    q_p = pack.pad(q).transpose(1, 2)
    k_p = k.transpose(1, 2)
    v_p = v.transpose(1, 2)
    out = F.scaled_dot_product_attention(q_p, k_p, v_p)
    return pack.unpad(out.transpose(1, 2))


def cached_lengths(grid: SparseGrid) -> torch.Tensor:
    cache = grid.cache
    key = ("lengths", grid.batch_size)
    lengths = cache.get(key)
    if lengths is None:
        lengths = cache[key] = grid.lengths()
    return lengths


def cached_pack(grid: SparseGrid) -> VarlenPack:
    cache = grid.cache
    key = ("pack", grid.batch_size)
    pack = cache.get(key)
    if pack is None:
        pack = cache[key] = build_varlen_pack(cached_lengths(grid))
    return pack


def cached_position_embedding(
    grid: SparseGrid, embedder: torch.nn.Module, *, retain: bool
) -> torch.Tensor:
    cache = grid.cache
    entry = cache.get("pos_emb")
    if entry is not None and entry[0] is embedder:
        return entry[1]
    embedding = embedder(grid.coords[:, 1:])
    if retain:
        cache["pos_emb"] = (embedder, embedding)
    return embedding
