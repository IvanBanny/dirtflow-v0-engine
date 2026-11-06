import hashlib
import json

import pytest
import torch

from dirtflow_engine import bundle, voxel


def pytest_sessionstart(session):
    torch.set_num_threads(2)


@pytest.fixture
def model_dir(tmp_path):
    (tmp_path / "text").mkdir()
    for name in (
        "weights.pt",
        "registry.json",
        "schema.json",
        "mapping.npz",
        "text/config.json",
    ):
        (tmp_path / name).write_bytes(b"fixture")
    files = {
        p.relative_to(tmp_path).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in tmp_path.rglob("*")
        if p.is_file()
    }
    manifest = {
        "format": "dirtflow-engine-bundle",
        "version": 1,
        "weight_set": "ema",
        "modules": dict.fromkeys(bundle.STAGE1 + bundle.STAGE2, "weights.pt"),
        "assets": {
            "registry": "registry.json",
            "schema": "schema.json",
            "mapping": "mapping.npz",
            "text_encoder": "text",
        },
        "files": files,
    }
    (tmp_path / "bundle.json").write_text(json.dumps(manifest))
    return tmp_path


@pytest.fixture
def request_data():
    snapshot = voxel.Tensor((-8, -8, -8), (18, 19, 20), [voxel.AIR], [], [])
    return {
        "protocol_version": 2,
        "client_version": "test",
        "correlation_id": "test",
        "prompt": "a stone cube",
        "game_version": "1.21.11",
        "dimension": "minecraft:overworld",
        "region": {"min": [0, 0, 0], "max": [1, 2, 3]},
        "context_padding": 8,
        "snapshot": voxel.pack(snapshot),
        "options": {"preview_every_steps": 0},
    }
