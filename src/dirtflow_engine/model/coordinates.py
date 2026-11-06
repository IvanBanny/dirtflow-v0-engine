"""Minecraft XYZ and the checkpoint native frame."""

from __future__ import annotations

import torch

SYSTEMS = ("minecraft_y_up", "trellis_z_up")


def validate_system(system: str) -> None:
    if system not in SYSTEMS:
        raise ValueError(f"unsupported structure coordinate system: {system}")


def to_structure_grid(value: torch.Tensor, system: str) -> torch.Tensor:
    validate_system(system)
    return value.transpose(-2, -1).flip(-2) if system == "trellis_z_up" else value


def to_world_grid(value: torch.Tensor, system: str) -> torch.Tensor:
    validate_system(system)
    return value.flip(-2).transpose(-2, -1) if system == "trellis_z_up" else value
