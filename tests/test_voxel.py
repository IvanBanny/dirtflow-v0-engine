import base64
import struct
import zlib

import numpy as np
import pytest

from dirtflow_engine import voxel

# Hand-encoded v1 sparse tensor: origin (-1,2,3), size (2,1,1), stone then air
GOLDEN = bytes.fromhex(
    "4445455001000000ffffffff0000000200000003000200010001000200000010"
    "000d6d696e6563726166743a616972000f6d696e6563726166743a73746f6e65"
    "00000002000000000001000000010000"
)


def test_sparse_golden_and_explicit_air():
    tensor = voxel.decode(GOLDEN)
    assert tensor.origin == (-1, 2, 3)
    assert tensor.positions == [0, 1]
    assert tensor.values == [1, 0]
    assert voxel.encode(tensor) == GOLDEN
    np.testing.assert_array_equal(tensor.array().ravel(), [1, 0])


@pytest.mark.parametrize("compression", ["none", "zlib"])
def test_rectangular_order_and_carving(compression):
    values = np.arange(24, dtype=np.int64).reshape(2, 3, 4) % 3
    states = [voxel.AIR, "minecraft:stone", "minecraft:oak_planks"]
    tensor = voxel.from_array(values, (-3, 4, 5), states)
    decoded = voxel.unpack(**voxel.pack(tensor, compression))
    np.testing.assert_array_equal(decoded.array(), values)
    before = values.copy()
    before[0, 1, 2] = 1
    values[0, 1, 2] = 0
    diff = voxel.from_array(values, (0, 0, 0), states, before)
    assert diff.positions == [(2 * 3 + 1) * 2]
    assert diff.values == [0]


def test_dense_long_runs():
    values = np.ones((58, 58, 58), dtype=np.int64)
    tensor = voxel.from_array(values, (0, 0, 0), [voxel.AIR, "minecraft:stone"])
    assert tensor.positions == [65535, 65535, 64042]
    np.testing.assert_array_equal(voxel.decode(voxel.encode(tensor)).array(), values)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda b: b[:20],
        lambda b: b + b"x",
        lambda b: b[:4] + b"\x02" + b[5:],
        lambda b: b[:6] + b"\x00\x01" + b[8:],
        lambda b: b[:20] + b"\x00\x00" + b[22:],
        lambda b: b[:-2] + b"\xff\xff",
        lambda b: b[:32] + b"\xff\xff" + b[34:],
    ],
)
def test_malformed_binary(mutate):
    with pytest.raises(ValueError):
        voxel.decode(mutate(GOLDEN))


@pytest.mark.parametrize(
    "raw",
    [
        zlib.compress(GOLDEN)[:-1],
        zlib.compress(GOLDEN) + b"junk",
        zlib.compress(b"x" * (voxel.MAX_BYTES + 1)),
    ],
)
def test_invalid_compression(raw):
    with pytest.raises(ValueError):
        voxel.unpack("voxel-tensor-v1+zlib", base64.b64encode(raw).decode())


def test_duplicate_sparse_and_zero_run():
    with pytest.raises(ValueError, match="duplicate"):
        voxel.encode(voxel.Tensor((0, 0, 0), (1, 1, 1), [voxel.AIR], [0, 0], [0, 0]))
    raw = voxel.encode(voxel.Tensor((0, 0, 0), (1, 1, 1), [voxel.AIR], [1], [0], True))
    with pytest.raises(ValueError):
        voxel.decode(raw[:-2] + struct.pack(">H", 0))
