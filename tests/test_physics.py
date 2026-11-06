import types

import numpy as np

from dirtflow_engine.physics import Physics


def test_projection_stays_in_edit_region():
    states = [
        "minecraft:air",
        "dirtflow:unknown",
        "minecraft:stone",
        "minecraft:sand",
        "minecraft:torch",
        "minecraft:water[level=1]",
        "minecraft:oak_door[half=upper]",
    ]
    physics = Physics(types.SimpleNamespace(states=states))
    grid = np.zeros((5, 5, 5), dtype=np.int64)
    grid[:, 0, :] = 2
    grid[1, 3, 1] = 3
    grid[2, 3, 2] = 4
    grid[3, 3, 3] = 5
    grid[1, 2, 3] = 6
    grid[0, 4, 0] = 4
    edit = np.zeros_like(grid, dtype=bool)
    edit[1:4, 1:4, 1:4] = True
    out = physics.project(grid, edit)
    np.testing.assert_array_equal(out[~edit], grid[~edit])
    assert out[1, 1, 1] == 3 and out[1, 3, 1] == 0
    assert out[2, 3, 2] == out[3, 3, 3] == out[1, 2, 3] == 0
    assert grid[1, 3, 1] == 3
