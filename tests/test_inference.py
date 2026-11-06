import os

import numpy as np
import pytest
import torch

from dirtflow_engine.sampling import Cancelled, Sampler, Settings


@pytest.fixture(scope="module")
def sampler():
    path = os.environ.get("DIRTFLOW_TEST_BUNDLE")
    if not path:
        pytest.skip("set DIRTFLOW_TEST_BUNDLE to a local inference bundle")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    return Sampler(path)


@pytest.mark.integration
def test_real_generation_previews_and_material_only(sampler):
    before = np.zeros((64, 64, 64), dtype=np.int64)
    edit = np.zeros_like(before, dtype=bool)
    edit[29:35, 30:34, 29:35] = True
    observed = np.zeros_like(edit)
    observed[21:43, 22:42, 21:43] = True
    settings = Settings(3, 3, physics=False, preview_every=0)
    result = sampler.generate(
        before, edit, observed, "a small stone tower", [42], settings=settings
    )
    frames = []
    with_previews = sampler.generate(
        before,
        edit,
        observed,
        "a small stone tower",
        [42],
        settings=Settings(3, 3, physics=False, preview_every=1),
        preview=lambda stage, step, total, final, values: frames.append(
            (stage, step, final)
        ),
    )
    np.testing.assert_array_equal(result, with_previews)
    assert len(frames) == 6
    assert {f[0] for f in frames} == {"structure", "blockstates"}
    assert np.all(result[:, ~edit] == before[~edit])
    assert np.isin(result, sampler.schema.output_to_global).all()
    before[30:34, 30:32, 30:34] = sampler.schema.normalize_input("stone")
    material = sampler.generate(
        before,
        edit,
        observed,
        "oak planks",
        [42],
        task="material_only",
        settings=settings,
    )
    np.testing.assert_array_equal(material[0] != 0, before != 0)


@pytest.mark.integration
def test_cooperative_cancellation(sampler):
    before = np.zeros((64, 64, 64), dtype=np.int64)
    edit = np.zeros_like(before, dtype=bool)
    edit[30:34, 30:34, 30:34] = True
    calls = 0

    def canceled():
        nonlocal calls
        calls += 1
        return calls > 2

    with pytest.raises(Cancelled):
        sampler.generate(
            before,
            edit,
            np.ones_like(edit),
            "stone",
            [42],
            settings=Settings(3, 3),
            canceled=canceled,
        )
