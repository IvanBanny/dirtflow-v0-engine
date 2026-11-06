import json
import pathlib
import re
import shutil
import subprocess
import sys
from unittest import mock

import pytest
import torch

from dirtflow_engine import bundle, cli, download


@pytest.fixture
def cache(tmp_path_factory, monkeypatch):
    root = tmp_path_factory.mktemp("hub")
    monkeypatch.setattr(download.constants, "HF_HOME", str(root))
    monkeypatch.setattr(download.constants, "HF_HUB_OFFLINE", False)
    return root / "dirtflow-engine" / download.MODEL_REVISION


def test_download_and_offline_reuse(cache, model_dir, monkeypatch):
    def fetch(repo_id, *, revision, local_dir, allow_patterns):
        assert repo_id == "IvanBanny/dirtflow-v0"
        assert re.fullmatch(r"[0-9a-f]{40}", revision)
        assert allow_patterns == ["bundle.json", "stage1.pt", "stage2.pt", "assets/**"]
        shutil.copytree(model_dir, local_dir)

    with mock.patch.object(
        download.huggingface_hub, "snapshot_download", side_effect=fetch
    ) as hub:
        assert download.fetch_model() == cache
        bundle.validate_bundle(cache)
        monkeypatch.setattr(download.constants, "HF_HUB_OFFLINE", True)
        assert download.fetch_model() == cache
        hub.assert_called_once()


def test_partial_download_resumes(cache, model_dir):
    cache.mkdir(parents=True)
    shutil.copyfile(model_dir / "bundle.json", cache / "bundle.json")
    (cache / "weights.pt.incomplete").write_bytes(b"fix")

    def resume(*args, **kwargs):
        assert (cache / "weights.pt.incomplete").read_bytes() == b"fix"
        shutil.copytree(model_dir, cache, dirs_exist_ok=True)

    with mock.patch.object(
        download.huggingface_hub, "snapshot_download", side_effect=resume
    ):
        bundle.validate_bundle(download.fetch_model())


@pytest.mark.parametrize("partial", [False, True])
def test_offline_missing_model(cache, model_dir, monkeypatch, partial):
    if partial:
        cache.mkdir(parents=True)
        shutil.copyfile(model_dir / "bundle.json", cache / "bundle.json")
    monkeypatch.setattr(download.constants, "HF_HUB_OFFLINE", True)
    with mock.patch.object(download.huggingface_hub, "snapshot_download") as hub:
        with pytest.raises(FileNotFoundError, match="not fully cached"):
            download.fetch_model()
        hub.assert_not_called()


def test_download_failure_keeps_partial_files(cache):
    def fail(*args, **kwargs):
        cache.mkdir(parents=True)
        (cache / "weights.pt.incomplete").write_bytes(b"fix")
        raise ConnectionError("download interrupted")

    with mock.patch.object(
        download.huggingface_hub, "snapshot_download", side_effect=fail
    ):
        with pytest.raises(ConnectionError, match="download interrupted"):
            download.fetch_model()
    assert (cache / "weights.pt.incomplete").read_bytes() == b"fix"


def test_download_must_finish(cache):
    with mock.patch.object(
        download.huggingface_hub, "snapshot_download", return_value=str(cache)
    ):
        with pytest.raises(FileNotFoundError, match="download is incomplete"):
            download.fetch_model()


@pytest.mark.parametrize("contents", ["{", '{"format":"wrong","version":1}'])
def test_invalid_cache_manifest(cache, contents):
    cache.mkdir(parents=True)
    (cache / "bundle.json").write_text(contents)
    with pytest.raises(ValueError, match="model manifest"):
        download.fetch_model()


def test_cache_rejects_path_escape(cache):
    cache.mkdir(parents=True)
    (cache / "bundle.json").write_text(
        json.dumps(
            {"format": bundle.FORMAT, "version": 1, "files": {"../weights.pt": "hash"}}
        )
    )
    with pytest.raises(ValueError, match="invalid bundle path"):
        download.fetch_model()


@pytest.mark.parametrize("local", [False, True])
def test_download_only_verifies_bundle(cache, model_dir, monkeypatch, capsys, local):
    shutil.copytree(model_dir, cache)
    monkeypatch.setattr(download.constants, "HF_HUB_OFFLINE", True)
    args = ["dirtflow-engine", "--download-only"]
    if local:
        args += ["--model", str(model_dir)]
    monkeypatch.setattr(sys, "argv", args)
    with mock.patch.object(download.huggingface_hub, "snapshot_download") as hub:
        cli.main()
        assert pathlib.Path(capsys.readouterr().out.strip()) == (
            model_dir if local else cache
        )
        hub.assert_not_called()
        ((model_dir if local else cache) / "weights.pt").write_bytes(b"corrupt")
        with pytest.raises(ValueError, match="changed bundle file: weights.pt"):
            cli.main()


def test_download_only_needs_no_torch(model_dir):
    code = """
import sys
sys.modules["torch"] = None
from dirtflow_engine.cli import main
main()
"""
    result = subprocess.run(
        [sys.executable, "-c", code, "--download-only", "--model", str(model_dir)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == str(model_dir)


def test_missing_cuda_fails_before_downloading(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(sys, "argv", ["dirtflow-engine"])
    with mock.patch.object(download.huggingface_hub, "snapshot_download") as hub:
        with pytest.raises(SystemExit) as error:
            cli.main()
        assert error.value.code == 2
        hub.assert_not_called()
