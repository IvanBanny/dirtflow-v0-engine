"""HTTP v2 request validation and placement on the 64-cube model canvas."""

from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from . import voxel

Coord = Annotated[
    list[Annotated[int, Field(strict=True, ge=-(2**31), lt=2**31)]],
    Field(min_length=3, max_length=3),
]
Steps = Annotated[int, Field(strict=True, ge=1, le=200)]


class Region(BaseModel):
    min: Coord
    max: Coord

    @property
    def size(self) -> tuple[int, int, int]:
        x, y, z = (hi - lo + 1 for lo, hi in zip(self.min, self.max))
        return x, y, z

    @model_validator(mode="after")
    def ordered(self) -> "Region":
        if min(self.size) <= 0:
            raise ValueError("region max must be at least min")
        return self


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid")
    steps: Steps | None = None
    stage1_steps: Steps | None = None
    stage2_steps: Steps | None = None
    cfg_scale: float = Field(default=5.0, ge=0, le=20, allow_inf_nan=False, strict=True)
    seed: int | None = Field(default=None, strict=True, ge=0, lt=2**53)
    batch_size: int = Field(default=1, strict=True, ge=1, le=8)
    candidates: int | None = Field(default=None, strict=True, ge=1, le=16)
    preview_every_steps: int = Field(default=3, strict=True, ge=0, le=10000)
    tensor_compression: Literal["none", "zlib"] = "none"
    task_type: Literal["completion", "replacement", "repair", "material_only"] = (
        "replacement"
    )


class Payload(BaseModel):
    encoding: Literal["voxel-tensor-v1", "voxel-tensor-v1+zlib"]
    data: str


class Request(BaseModel):
    protocol_version: Literal[2]
    client_version: str = Field(max_length=128)
    correlation_id: str = Field(min_length=1, max_length=128)
    prompt: str = Field(min_length=1, max_length=4096)
    game_version: Literal["1.21.11"]
    dimension: str = Field(min_length=1, max_length=128)
    region: Region
    context_padding: int = Field(strict=True, ge=0, le=8)
    snapshot: Payload
    options: Options = Field(default_factory=Options)

    def decode(self) -> voxel.Tensor:
        tensor = voxel.unpack(self.snapshot.encoding, self.snapshot.data)
        fitting = [r for r in (3, 5, 8) if max(self.region.size) + 2 * r <= 64]
        if not fitting or self.context_padding != max(fitting):
            raise ValueError(
                "region requires the thickest fitting context shell in {3,5,8}; editable axes are at most 58"
            )
        expected = tuple(n + 2 * self.context_padding for n in self.region.size)
        origin = tuple(n - self.context_padding for n in self.region.min)
        if tensor.size != expected or tensor.origin != origin:
            raise ValueError(f"snapshot must cover {expected} at {origin}")
        return tensor


def canvas(
    request: Request, tensor: voxel.Tensor, schema
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[slice, ...]]:
    offset = tuple((64 - n) // 2 for n in tensor.size)
    capture = tuple(slice(o, o + n) for o, n in zip(offset, tensor.size))
    region = tuple(
        slice(o + request.context_padding, o + request.context_padding + n)
        for o, n in zip(offset, request.region.size)
    )
    values = np.zeros((64, 64, 64), dtype=np.int64)
    mapping = np.array([schema.normalize_input(s) for s in tensor.palette])
    values[capture] = mapping[tensor.array()]
    observed = np.zeros_like(values, dtype=bool)
    observed[capture] = True
    edit = np.zeros_like(observed)
    edit[region] = True
    return values, edit, observed, region
