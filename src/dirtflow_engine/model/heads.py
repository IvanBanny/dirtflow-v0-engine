"""Block-state classification over the legal output map."""

from __future__ import annotations

import itertools

import torch
from torch import nn


def argmax_over_vocab(
    head: CategoricalHead,
    z0_hat: torch.Tensor,
    embedding_table: torch.Tensor,
    io_feats: torch.Tensor | None = None,
    chunk: int = 256,
    *,
    context_indicator: torch.Tensor | None = None,
    batch_index: torch.Tensor | None = None,
    forbidden_global_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    if chunk < 1:
        raise ValueError("chunk must be positive")
    if z0_hat.shape[0] == 0:
        return torch.empty(0, dtype=torch.long, device=z0_hat.device)
    keys = head.output_keys(embedding_table)
    io_chunks = (
        io_feats.split(chunk) if io_feats is not None else itertools.repeat(None)
    )
    bi_chunks = (
        batch_index.split(chunk) if batch_index is not None else itertools.repeat(None)
    )
    outputs = []
    for part, io, bi in zip(z0_hat.split(chunk), io_chunks, bi_chunks):
        logits = head.full_vocab_logits(
            part,
            embedding_table,
            io_feats=io,
            keys=keys,
            context_indicator=context_indicator,
            batch_index=bi,
        )
        if forbidden_global_ids is not None:
            class_ids = torch.arange(keys.shape[0], device=logits.device)
            global_ids = head.global_ids(class_ids)
            forbidden = torch.isin(global_ids, forbidden_global_ids.to(logits.device))
            if bool(forbidden.all()):
                raise ValueError("decode restrictions exclude every output state")
            logits = logits.masked_fill(forbidden, -torch.inf)
        prediction = logits.argmax(dim=1)
        outputs.append(head.global_ids(prediction))
        del logits
    return torch.cat(outputs)


class CategoricalHead(nn.Module):
    """Denoised embeddings [L,8] and IO features [L,C] to legal-state logits."""

    output_to_global: torch.Tensor
    key_proj: nn.Linear | None

    def __init__(
        self,
        latent_dim: int = 8,
        hidden: int = 256,
        feat_dim: int = 0,
        *,
        output_to_global: torch.Tensor,
    ) -> None:
        super().__init__()
        mapping = output_to_global.to(dtype=torch.long).clone()
        if mapping.ndim != 1 or not mapping.numel():
            raise ValueError("output mapping must be a nonempty vector")
        if (mapping < 0).any() or mapping.unique().numel() != mapping.numel():
            raise ValueError("output mapping requires unique nonnegative IDs")
        self.register_buffer("output_to_global", mapping)
        self.output_vocab = len(mapping)
        self.feat_dim = feat_dim
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim + feat_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.key_proj = nn.Linear(latent_dim, hidden, bias=False)
        self.scale = hidden**-0.5
        self.class_bias = nn.Parameter(torch.zeros(self.output_vocab))
        self.context_gamma = nn.Parameter(torch.zeros(()))

    def output_keys(self, embedding_table: torch.Tensor) -> torch.Tensor:
        if self.key_proj is None:
            raise ValueError("tied head requires a key projection")
        return self.key_proj(embedding_table[self.output_to_global])  # [K,H]

    def global_ids(self, class_ids: torch.Tensor) -> torch.Tensor:
        return self.output_to_global[class_ids]

    def target_classes(self, global_ids: torch.Tensor) -> torch.Tensor:
        """Returns class IDs, or -1 for states absent from the legal output map."""
        sorted_ids, classes = self.output_to_global.sort()
        positions = torch.searchsorted(sorted_ids, global_ids)
        safe = positions.clamp(max=sorted_ids.numel() - 1)
        valid = (positions < sorted_ids.numel()) & (sorted_ids[safe] == global_ids)
        return torch.where(valid, classes[safe], -1)

    def _queries(
        self, z0_hat: torch.Tensor, io_feats: torch.Tensor | None
    ) -> torch.Tensor:
        if self.feat_dim:
            if io_feats is None:
                raise ValueError("head built with feat_dim > 0 needs io_feats")
            z0_hat = torch.cat([z0_hat, io_feats], dim=1)
        return self.mlp(z0_hat)

    def full_vocab_logits(
        self,
        z0_hat: torch.Tensor,
        embedding_table: torch.Tensor,
        io_feats: torch.Tensor | None = None,
        *,
        keys: torch.Tensor | None = None,
        context_indicator: torch.Tensor | None = None,
        batch_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        queries = self._queries(z0_hat, io_feats)
        if keys is None:
            keys = self.output_keys(embedding_table)
        logits = queries @ keys.T * self.scale
        if self.class_bias is not None:
            logits = logits + self.class_bias
        if context_indicator is not None:
            ind = (
                context_indicator
                if batch_index is None
                else context_indicator[batch_index]
            )
            logits = logits + self.context_gamma * ind.to(logits.dtype)
        return logits


class IndependentCategoricalHead(CategoricalHead):
    def __init__(
        self,
        *,
        latent_dim: int = 8,
        hidden: int = 256,
        feat_dim: int = 0,
        output_to_global: torch.Tensor,
        key_dim: int = 128,
    ) -> None:
        super().__init__(
            latent_dim, hidden, feat_dim, output_to_global=output_to_global
        )
        if key_dim != 128:
            raise ValueError("independent keys require 128 channels")
        self.mlp[-1] = torch.nn.Linear(hidden, key_dim)
        self.key_proj = None
        self.keys = torch.nn.Parameter(torch.empty(self.output_vocab, key_dim))
        torch.nn.init.normal_(self.keys, std=1.0)
        self.scale = key_dim ** (-0.5)

    def output_keys(self, embedding_table: torch.Tensor) -> torch.Tensor:
        del embedding_table
        return self.keys
