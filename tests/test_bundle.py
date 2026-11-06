import pytest

from dirtflow_engine import bundle


def test_hashes_and_unhashed_assets(model_dir):
    assert bundle.validate_bundle(model_dir)["format"] == "dirtflow-engine-bundle"
    (model_dir / "weights.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        bundle.validate_bundle(model_dir)


@pytest.mark.parametrize(
    "path", ["../secret", "/tmp/secret", "C:/secret", "..\\secret", ""]
)
def test_path_escape(model_dir, path):
    with pytest.raises(ValueError):
        bundle.local_path(model_dir, path)


def test_symlink_escape(model_dir, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside") / "weights"
    outside.write_bytes(b"fixture")
    (model_dir / "link").symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        bundle.local_path(model_dir, "link")


def test_inference_policy():
    with pytest.raises(ValueError):
        bundle.inference_policy(
            {"inference_policy": {"editable_confidence": "zero_normalized_editable"}}
        )
