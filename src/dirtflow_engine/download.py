"""The published model release and its local cache."""

import json
import logging
import pathlib

import huggingface_hub
from huggingface_hub import constants

from . import bundle

MODEL_REPO = "IvanBanny/dirtflow-v0"
MODEL_REVISION = "7c133d62aa1d4c10f0eb7590c157286055140472"


def _complete(root: pathlib.Path) -> bool:
    manifest = root / "bundle.json"
    if not manifest.is_file():
        return False
    try:
        data = json.loads(manifest.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid model manifest: {manifest}") from error
    if data.get("format") != bundle.FORMAT or data.get("version") != bundle.VERSION:
        raise ValueError(f"unsupported model manifest: {manifest}")
    files = data["files"]
    return bool(files) and all(
        bundle.local_path(root, name).is_file() for name in files
    )


def fetch_model() -> pathlib.Path:
    """Returns the pinned release, downloading missing files when online.

    Complete caches need no network access. The model loader verifies every
    file's hash before loading weights.
    """
    root = (
        pathlib.Path(constants.HF_HOME).expanduser()
        / "dirtflow-engine"
        / MODEL_REVISION
    )
    if _complete(root):
        return root
    if constants.HF_HUB_OFFLINE:
        raise FileNotFoundError(
            f"model is not fully cached at {root}; run dirtflow-engine --download-only "
            "with network access, or pass --model /path/to/bundle"
        )
    logging.info("Downloading %s@%s (9.3 GB) to %s", MODEL_REPO, MODEL_REVISION, root)
    # Keep files together; Hub snapshot symlinks leave the bundle root
    huggingface_hub.snapshot_download(
        MODEL_REPO,
        revision=MODEL_REVISION,
        local_dir=root,
        allow_patterns=["bundle.json", "stage1.pt", "stage2.pt", "assets/**"],
    )
    if not _complete(root):
        raise FileNotFoundError(f"model download is incomplete: {root}")
    return root
