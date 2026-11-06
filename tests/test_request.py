import numpy as np
import pytest

from dirtflow_engine import request
from dirtflow_engine.schema import Schema


def test_request_canvas(request_data):
    class States:
        def normalize_input(self, state):
            return 0

    req = request.Request.model_validate(request_data)
    before, edit, observed, region = request.canvas(req, req.decode(), States())
    assert before.dtype == np.int64
    assert edit.sum() == 24
    assert observed.sum() == 18 * 19 * 20
    assert before[region].shape == (2, 3, 4)
    assert np.all(observed[edit])


@pytest.mark.parametrize(
    "field,value",
    [
        ("seed", -1),
        ("seed", 2**53),
        ("steps", True),
        ("stage1_steps", 0),
        ("batch_size", 9),
        ("candidates", 17),
        ("cfg_scale", float("nan")),
        ("cfg_scale", float("inf")),
        ("preview_every_steps", -1),
        ("task_type", "unknown"),
    ],
)
def test_bad_options(request_data, field, value):
    request_data["options"][field] = value
    with pytest.raises(ValueError):
        request.Request.model_validate(request_data)


@pytest.mark.parametrize(
    "field,value",
    [
        ("protocol_version", 1),
        ("game_version", "1.20"),
        ("context_padding", 5),
        ("prompt", ""),
    ],
)
def test_bad_request(request_data, field, value):
    request_data[field] = value
    with pytest.raises(ValueError):
        request.Request.model_validate(request_data).decode()


def schema_fixture():
    states = [
        "minecraft:air",
        "dirtflow:unknown",
        "minecraft:cave_air",
        "minecraft:stone",
        "minecraft:oak_log[axis=y]",
        "minecraft:oak_log[axis=x]",
    ]
    entries = [
        {
            "state": state,
            "is_air": "air" in state,
            "default": state != "minecraft:oak_log[axis=x]",
        }
        for state in states
        if state != "dirtflow:unknown"
    ]
    payload = {
        "format": "dirtflow-game-state-schema",
        "version": 1,
        "minecraft_version": "1.21.11",
        "states": entries,
    }
    return states, payload, np.array([0, 5, 4, 3], dtype=np.int64)


def test_game_defaults_and_unknown_states():
    schema = Schema(*schema_fixture())
    assert schema.normalize_input("cave_air") == 0
    assert schema.normalize_input("oak_log") == 4
    assert schema.normalize_input("oak_log[axis=x]") == 5
    for state in (
        "mod:stone",
        "oak_log[axis=q]",
        "oak_log[axis=x,axis=y]",
        "oak_log[axis=x",
    ):
        assert schema.normalize_input(state) == 1


def test_mapping_disagreement():
    states, payload, mapping = schema_fixture()
    with pytest.raises(ValueError, match="mapping"):
        Schema(states, payload, mapping[::-1])


@pytest.mark.parametrize(
    "axis,padding", [(1, 8), (48, 8), (49, 5), (54, 5), (55, 3), (58, 3)]
)
def test_trained_padding_boundaries(request_data, axis, padding):
    from dirtflow_engine import voxel

    request_data["region"]["max"] = [axis - 1, 2, 3]
    request_data["context_padding"] = padding
    tensor = voxel.Tensor(
        (-padding,) * 3,
        (axis + 2 * padding, 3 + 2 * padding, 4 + 2 * padding),
        [voxel.AIR],
        [],
        [],
    )
    request_data["snapshot"] = voxel.pack(tensor)
    assert request.Request.model_validate(request_data).decode().size == tensor.size
