"""Observed-world features in checkpoint channel order."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import functional as functional

from . import coordinates
from .dense import DownsampleBlock3d, ResBlock3d
from .layers import TimestepEmbedder

TASK_NAMES = (
    "scratch_void",
    "scratch_ground",
    "completion",
    "replacement",
    "repair",
    "material_only",
)
NULL_STYLE_ID = 5
MC_DOMAIN_ID = 0
NUM_DOMAINS = 3


def dense_observation_features(
    before_ids: torch.Tensor,
    editable: torch.Tensor,
    observed: torch.Tensor,
    embedding_table: torch.Tensor,
    *,
    context_keep: torch.Tensor | None = None,
) -> torch.Tensor:
    if before_ids.shape != editable.shape or editable.shape != observed.shape:
        raise ValueError("before, editable and observed shapes must agree")
    if editable.dtype != torch.bool or observed.dtype != torch.bool:
        raise ValueError("editable and observed must be boolean")
    if embedding_table.shape[1] != 8:
        raise ValueError("v1 observation layout requires eight embedding channels")
    safe_ids = torch.where(observed, before_ids, 0)
    embeddings = functional.embedding(safe_ids, embedding_table).movedim(-1, 1)
    known = observed & ~editable
    old = observed & editable
    occupancy = safe_ids != 0
    features = torch.cat(
        [
            (occupancy & known).unsqueeze(1),
            embeddings * known.unsqueeze(1),
            editable.unsqueeze(1),
            observed.unsqueeze(1),
            (occupancy & old).unsqueeze(1),
            embeddings * old.unsqueeze(1),
        ],
        dim=1,
    ).float()
    if context_keep is not None:
        features = features * context_keep.reshape(-1, 1, 1, 1, 1)
        features[:, 9] = editable
    return features


def neighbor_observations(coords: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    if coords.ndim != 2 or coords.shape[1] != 4 or observed.ndim != 4:
        raise ValueError("expected coordinates [L,4] and observed [B,X,Y,Z]")
    directions = coords.new_tensor(
        [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]]
    )
    neighbors = coords[:, None, 1:] + directions
    bounds = coords.new_tensor(observed.shape[1:])
    valid = ((neighbors >= 0) & (neighbors < bounds)).all(dim=-1)
    neighbors = torch.minimum(neighbors.clamp_min(0), bounds - 1).long()
    values = observed[
        coords[:, None, 0].long(),
        neighbors[..., 0],
        neighbors[..., 1],
        neighbors[..., 2],
    ]
    return (values & valid).float()


def sparse_observation_features(
    noisy: torch.Tensor,
    known_embedding: torch.Tensor,
    known: torch.Tensor,
    edit_logits: torch.Tensor,
    logit_valid: torch.Tensor,
    self_condition: torch.Tensor,
    old_edit_embedding: torch.Tensor,
    old_edit_valid: torch.Tensor,
    neighbor_observed: torch.Tensor,
    *,
    context_keep: torch.Tensor | None = None,
) -> torch.Tensor:
    length = noisy.shape[0]
    vectors = (noisy, known_embedding, self_condition, old_edit_embedding)
    if any((value.shape != (length, 8) for value in vectors)):
        raise ValueError("all v1 embedding features must have shape [L,8]")
    if neighbor_observed.shape != (length, 6):
        raise ValueError("neighbor observations must have shape [L,6]")
    known = known.reshape(length, 1).to(noisy.dtype)
    old_edit_valid = old_edit_valid.reshape(length, 1).to(noisy.dtype)
    if bool(((known != 0) & (old_edit_valid != 0)).any()):
        raise ValueError("known and old-edit observations must be disjoint")
    keep = 1 if context_keep is None else context_keep.reshape(length, 1)
    valid = logit_valid.reshape(length, 1).to(noisy.dtype) * (1 - known)
    return torch.cat(
        [
            noisy,
            known_embedding * known * keep,
            known * keep,
            edit_logits.reshape(length, 1) * valid,
            valid,
            self_condition,
            old_edit_embedding * old_edit_valid * keep,
            old_edit_valid * keep,
            neighbor_observed * keep,
        ],
        dim=1,
    )


class BlockStateEmbedding(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 8, *, rms_norm: bool = True) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.rms_norm = rms_norm
        weight = torch.randn(vocab_size, dim)
        self.weight = nn.Parameter(weight)

    def table(self) -> torch.Tensor:
        if not self.rms_norm:
            return self.weight
        values = self.weight.float()
        rms = values.pow(2).mean(dim=-1, keepdim=True).add(1e-08).rsqrt()
        return (values * rms).to(self.weight.dtype)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(ids, self.table())


class StructureContextEncoder(nn.Module):
    """Observed world [B,X,Y,Z] to native context [B,16,16,16,16]."""

    def __init__(
        self,
        *,
        in_channels: int = 20,
        out_channels: int = 16,
        widths: tuple[int, ...] = (32, 64, 128),
        vocab_size: int,
        embedding_dim: int = 8,
        embedding_rms_norm: bool = True,
        structure_coordinate_system: str = "minecraft_y_up",
    ) -> None:
        super().__init__()
        if in_channels != 20:
            raise ValueError("context encoder requires 20 observation channels")
        coordinates.validate_system(structure_coordinate_system)
        self.structure_coordinate_system = structure_coordinate_system
        self.embedding = BlockStateEmbedding(
            vocab_size, embedding_dim, rms_norm=embedding_rms_norm
        )
        self.stem = nn.Conv3d(in_channels, widths[0], 3, padding=1)
        stages: list[nn.Module] = []
        for i, width in enumerate(widths):
            stages.append(ResBlock3d(width))
            if i < len(widths) - 1:
                stages.append(DownsampleBlock3d(width, widths[i + 1]))
        self.stages = nn.Sequential(*stages)
        self.head = nn.Conv3d(widths[-1], out_channels, 3, padding=1)

    def forward(
        self, before: torch.Tensor, edit: torch.Tensor, observed: torch.Tensor
    ) -> torch.Tensor:
        features = dense_observation_features(
            before, edit, observed, self.embedding.table()
        )  # [B,20,X,Y,Z]
        context = self.head(self.stages(self.stem(features)))
        return coordinates.to_structure_grid(context, self.structure_coordinate_system)


class ExtraConditioning(nn.Module):
    def __init__(
        self,
        model_channels: int,
        *,
        num_styles: int = 5,
        num_domains: int = NUM_DOMAINS,
        score_max: float = 10.0,
        num_tasks: int = 0,
    ) -> None:
        super().__init__()
        self.score_max = score_max
        self.style_emb = nn.Embedding(num_styles + 1, model_channels)
        self.score_emb = TimestepEmbedder(model_channels)
        self.fragment_emb = nn.Embedding(2, model_channels)
        self.domain_emb = nn.Embedding(num_domains, model_channels)
        self.task_emb = nn.Embedding(num_tasks, model_channels) if num_tasks else None
        if self.task_emb is not None:
            self.null_score = nn.Parameter(torch.zeros(model_channels))
            nn.init.zeros_(self.task_emb.weight)
        else:
            self.register_parameter("null_score", None)
        nn.init.zeros_(self.style_emb.weight)
        nn.init.zeros_(self.score_emb.mlp[2].get_parameter("weight"))
        nn.init.zeros_(self.score_emb.mlp[2].get_parameter("bias"))
        nn.init.zeros_(self.fragment_emb.weight)
        nn.init.zeros_(self.domain_emb.weight)

    def forward(
        self,
        style_ids: torch.Tensor,
        scores: torch.Tensor,
        fragment: torch.Tensor | None = None,
        domain_ids: torch.Tensor | None = None,
        task_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        style = self.style_emb(style_ids)
        known = (scores >= 0).float().unsqueeze(1)
        score = self.score_emb(scores.clamp(min=0.0) / self.score_max * 1000.0)
        out = style + score * known
        if self.null_score is not None:
            out = out + (1 - known) * self.null_score
        if self.task_emb is not None:
            if task_ids is None:
                raise ValueError("v1 conditioning requires explicit task IDs")
            out = out + self.task_emb(task_ids)
        if fragment is not None:
            out = out + self.fragment_emb(fragment.long())
        if domain_ids is None:
            domain_ids = torch.full_like(style_ids, MC_DOMAIN_ID)
        return out + self.domain_emb(domain_ids)
