import os
import pathlib
import subprocess

import numpy as np
import pytest

from dirtflow_engine import voxel


@pytest.mark.integration
def test_actual_mod_codec(tmp_path):
    classpath = os.environ.get("DIRTFLOW_MOD_CLASSPATH")
    if not classpath:
        pytest.skip("set DIRTFLOW_MOD_CLASSPATH to the mod runtime classpath")
    source = pathlib.Path(__file__).with_name("CodecInterop.java")
    subprocess.run(
        [
            "javac",
            "--release",
            "21",
            "-cp",
            classpath,
            "-d",
            str(tmp_path),
            str(source),
        ],
        check=True,
        cwd=tmp_path,
    )
    values = np.arange(58**3).reshape(58, 58, 58) % 2
    tensor = voxel.from_array(values, (-7, 11, 25), [voxel.AIR, "minecraft:stone"])
    original = voxel.encode(tensor)
    (tmp_path / "input.vxt").write_bytes(original)
    subprocess.run(
        [
            "java",
            "-cp",
            classpath + os.pathsep + str(tmp_path),
            "CodecInterop",
            str(tmp_path / "input.vxt"),
            str(tmp_path / "output.vxt"),
            str(tmp_path / "carve.vxt"),
        ],
        check=True,
        cwd=tmp_path,
    )
    assert (tmp_path / "output.vxt").read_bytes() == original
    carve = voxel.decode((tmp_path / "carve.vxt").read_bytes())
    assert carve.origin == (-1, 2, 3)
    assert carve.positions == [0, 1] and carve.values == [1, 0]
