"""Voxel-tensor-v1: big-endian palette tensors, with x as the fastest axis.

The 32-byte header is >IBBHiiiHHHHI: magic DEEP, version, layout, reserved,
origin XYZ, size XYZ, palette length, body length. UTF-8 palette entries have
u16 byte lengths. Sparse bodies contain a u32 count and (u32 index, u16 ID)
pairs; dense bodies contain (u16 ID, u16 run) pairs. Sparse air is an edit.
"""

import base64
import dataclasses
import math
import struct
import zlib

import numpy as np

AIR = "minecraft:air"
MAX_BYTES = 8 * 1024 * 1024
HEADER = struct.Struct(">IBBHiiiHHHHI")


@dataclasses.dataclass(frozen=True)
class Tensor:
    origin: tuple[int, int, int]
    size: tuple[int, int, int]
    palette: list[str]
    positions: list[int]
    values: list[int]
    dense: bool = False

    def validate(self) -> None:
        if len(self.origin) != 3 or any(not -(2**31) <= v < 2**31 for v in self.origin):
            raise ValueError("origin must contain three i32 coordinates")
        if len(self.size) != 3 or any(not 1 <= v <= 65535 for v in self.size):
            raise ValueError("size must contain three positive u16 dimensions")
        if not self.palette or self.palette[0] != AIR or len(self.palette) > 65535:
            raise ValueError("palette must begin with air and fit in u16")
        if len(self.positions) != len(self.values):
            raise ValueError("tensor positions and values have different lengths")
        if any(not 0 <= value < len(self.palette) for value in self.values):
            raise ValueError("palette ID out of range")
        volume = math.prod(self.size)
        if self.dense:
            if (
                any(not 1 <= n <= 65535 for n in self.positions)
                or sum(self.positions) != volume
            ):
                raise ValueError("dense runs must cover the tensor exactly")
        elif any(not 0 <= n < volume for n in self.positions):
            raise ValueError("sparse index out of range")
        elif len(set(self.positions)) != len(self.positions):
            raise ValueError("duplicate sparse index")

    def array(self) -> np.ndarray:
        """Returns palette IDs [X,Y,Z]; omitted sparse entries are air."""
        self.validate()
        if self.dense:
            flat = np.repeat(self.values, self.positions)
        else:
            flat = np.zeros(math.prod(self.size), dtype=np.int64)
            flat[self.positions] = self.values
        return flat.reshape(self.size[::-1]).transpose(2, 1, 0)


def encode(tensor: Tensor) -> bytes:
    tensor.validate()
    palette = bytearray()
    for state in tensor.palette:
        raw = state.encode("utf-8")
        if len(raw) > 65535:
            raise ValueError("palette state exceeds u16 length")
        palette.extend(struct.pack(">H", len(raw)))
        palette.extend(raw)
    if tensor.dense:
        body = b"".join(
            struct.pack(">HH", value, run)
            for run, value in zip(tensor.positions, tensor.values)
        )
    else:
        body = struct.pack(">I", len(tensor.positions)) + b"".join(
            struct.pack(">IH", pos, value)
            for pos, value in zip(tensor.positions, tensor.values)
        )
    header = HEADER.pack(
        0x44454550,
        1,
        int(tensor.dense),
        0,
        *tensor.origin,
        *tensor.size,
        len(tensor.palette),
        len(body),
    )
    result = header + palette + body
    if len(result) > MAX_BYTES:
        raise ValueError("tensor exceeds 8 MiB")
    return result


def decode(raw: bytes) -> Tensor:
    if not 32 <= len(raw) <= MAX_BYTES:
        raise ValueError("tensor length must be between 32 bytes and 8 MiB")
    magic, version, layout, reserved, *fields = HEADER.unpack_from(raw)
    if (magic, version, reserved) != (0x44454550, 1, 0) or layout not in (0, 1):
        raise ValueError("unsupported voxel header")
    origin, size = tuple(fields[:3]), tuple(fields[3:6])
    palette_size, body_length = fields[6:]
    end = len(raw) - body_length
    offset, palette = 32, []
    for _ in range(palette_size):
        if offset + 2 > end:
            raise ValueError("truncated palette")
        length = struct.unpack_from(">H", raw, offset)[0]
        offset += 2
        if offset + length > end:
            raise ValueError("truncated palette state")
        palette.append(raw[offset : offset + length].decode("utf-8"))
        offset += length
    if offset != end:
        raise ValueError("incorrect body length")
    body = raw[offset:]
    if layout:
        if len(body) % 4:
            raise ValueError("truncated dense run")
        pairs = list(struct.iter_unpack(">HH", body))
        values = [value for value, _ in pairs]
        positions = [run for _, run in pairs]
    else:
        if len(body) < 4 or (len(body) - 4) % 6:
            raise ValueError("truncated sparse body")
        count = struct.unpack_from(">I", body)[0]
        if count != (len(body) - 4) // 6:
            raise ValueError("incorrect sparse count")
        pairs = list(struct.iter_unpack(">IH", body[4:]))
        positions = [pos for pos, _ in pairs]
        values = [value for _, value in pairs]
    result = Tensor(origin, size, palette, positions, values, bool(layout))
    result.validate()
    return result


def pack(tensor: Tensor, compression: str = "none") -> dict[str, str]:
    if compression not in {"none", "zlib"}:
        raise ValueError("unknown tensor compression")
    raw = encode(tensor)
    compressed = zlib.compress(raw, 1) if compression == "zlib" else raw
    use_zlib = len(compressed) < len(raw)
    return {
        "encoding": "voxel-tensor-v1+zlib" if use_zlib else "voxel-tensor-v1",
        "data": base64.b64encode(compressed if use_zlib else raw).decode("ascii"),
    }


def unpack(encoding: str, data: str) -> Tensor:
    if len(data) > 4 * ((MAX_BYTES + 2) // 3):
        raise ValueError("encoded tensor exceeds 8 MiB")
    raw = base64.b64decode(data, validate=True)
    if encoding == "voxel-tensor-v1+zlib":
        inflater = zlib.decompressobj()
        raw = inflater.decompress(raw, MAX_BYTES + 1)
        if not inflater.eof or inflater.unused_data or len(raw) > MAX_BYTES:
            raise ValueError("invalid or oversized zlib tensor")
    elif encoding != "voxel-tensor-v1":
        raise ValueError("unknown tensor encoding")
    return decode(raw)


def from_array(
    values: np.ndarray,
    origin: tuple[int, int, int],
    states: list[str],
    before: np.ndarray | None = None,
) -> Tensor:
    """Encodes a full preview, or a sparse diff when before is supplied."""
    flat = values.transpose(2, 1, 0).ravel()
    positions = (
        np.arange(flat.size)
        if before is None
        else np.flatnonzero(flat != before.transpose(2, 1, 0).ravel())
    )
    ids = np.unique(np.r_[0, flat[positions]])
    palette = [states[int(i)] for i in ids]
    local = np.searchsorted(ids, flat[positions])
    if before is not None:
        return Tensor(origin, values.shape, palette, positions.tolist(), local.tolist())
    starts = np.r_[0, np.flatnonzero(local[1:] != local[:-1]) + 1]
    lengths = np.diff(np.r_[starts, len(local)])
    runs, run_ids = [], []
    for start, length in zip(starts, lengths):
        for offset in range(0, int(length), 65535):
            runs.append(min(int(length) - offset, 65535))
            run_ids.append(int(local[start]))
    return Tensor(origin, values.shape, palette, runs, run_ids, True)
