"""Strict portable inference bundles; all assets are local and hash verified."""

from __future__ import annotations

import copy
import hashlib
import json
import pathlib
import struct

FORMAT = "dirtflow-engine-bundle"
VERSION = 1
STAGE1 = (
    "context_encoder",
    "structure_flow",
    "structure_extra",
    "latent_norm",
    "ss_encoder",
    "ss_decoder",
)
STAGE2 = ("blockstate_flow", "blockstate_extra", "embedding", "categorical_head")


def inference_policy(manifest: dict) -> dict:
    """Returns the weight-storage and confidence policy carried by the bundle."""
    original = {
        "editable_confidence": "original",
        "parameter_storage": "profile",
        "position_frequencies": "profile",
    }
    corrected = {
        "editable_confidence": "zero_normalized_editable",
        "parameter_storage": "float32",
        "position_frequencies": "native_float32",
    }
    policy = manifest.get("inference_policy", original)
    if policy not in (original, corrected):
        raise ValueError("unsupported or incomplete inference policy")
    if policy == corrected:
        model = manifest.get("model", {})
        if (
            model.get("feature_schema") != "v1"
            or model.get("blockstate_flow", {}).get("in_channels") != 42
        ):
            raise ValueError("corrected confidence requires the v1 42-column model")
    return policy.copy()


def sha256(path: pathlib.Path) -> str:
    """Hashes a file without reading its complete payload into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def local_path(root: pathlib.Path, relative: str) -> pathlib.Path:
    """Rejects absolute paths, traversal and symlink escapes."""
    if not isinstance(relative, str) or "\\" in relative or ":" in relative:
        raise ValueError("invalid bundle path")
    part = pathlib.PurePosixPath(relative)
    if part.is_absolute() or ".." in part.parts or not part.parts:
        raise ValueError(f"invalid bundle path: {relative!r}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"asset escapes bundle: {relative!r}")
    return path


def validate_bundle(root: str | pathlib.Path) -> dict:
    """Validates the manifest and every referenced byte before model loading."""
    root = pathlib.Path(root).resolve()
    manifest = json.loads((root / "bundle.json").read_text())
    if manifest.get("format") != FORMAT or manifest.get("version") != VERSION:
        raise ValueError("unsupported inference bundle")
    inference_policy(manifest)
    policy = manifest.get("weight_set")
    if policy not in {"ema", "ema+live-stage2-head", "stage1-ema+stage2-live"}:
        raise ValueError("bundle requires an explicit supported weight policy")
    expected_weights = dict.fromkeys(STAGE1 + STAGE2, "ema")
    if policy == "ema+live-stage2-head":
        expected_weights["categorical_head"] = "live"
    elif policy == "stage1-ema+stage2-live":
        expected_weights.update(dict.fromkeys(STAGE2, "live"))
    module_weights = manifest.get("module_weight_sets")
    if policy != "ema" or module_weights is not None:
        if module_weights != expected_weights:
            raise ValueError("module weight sets disagree with the declared policy")
    if set(manifest["modules"]) != set(STAGE1 + STAGE2):
        raise ValueError("bundle must explicitly own every inference module")
    assets = manifest["assets"]
    for key in ("registry", "schema", "mapping", "text_encoder"):
        if key not in assets:
            raise ValueError(f"missing asset: {key}")
    files = manifest["files"]
    for relative, expected in files.items():
        path = local_path(root, relative)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"missing or changed bundle file: {relative}")
    for relative in manifest["modules"].values():
        if relative not in files:
            raise ValueError(f"unhashed model payload: {relative}")
    for key, relative in assets.items():
        path = local_path(root, relative)
        if key != "text_encoder" and relative not in files:
            raise ValueError(f"unhashed asset: {key}")
        if key == "text_encoder":
            expected = {
                p.relative_to(root).as_posix() for p in path.rglob("*") if p.is_file()
            }
            if not expected or not expected <= files.keys():
                raise ValueError("text encoder is absent or has unhashed files")
    validate_asset_lineage(root, manifest)
    return manifest


def validate_asset_lineage(root: pathlib.Path, manifest: dict) -> None:
    """Checks copied assets against the identities recorded during training."""
    shared = manifest.get("lineage", {}).get("shared_assets", {})
    assets, files = manifest["assets"], manifest["files"]
    for key, field in (
        ("registry", "lineage.input_registry_sha256"),
        ("schema", "lineage.legal_state_schema_sha256"),
        ("mapping", "lineage.input_output_maps_sha256"),
        ("clip_manifest", "lineage.clip_tokenizer_weights_sha256"),
    ):
        if field in shared:
            if key not in assets or files.get(assets[key]) != shared[field]:
                raise ValueError(f"bundle asset differs from trained identity: {key}")
    if "clip_manifest" not in assets:
        return
    record = json.loads(local_path(root, assets["clip_manifest"]).read_text())
    expected = record.get("files", {})
    if not {"config.json", "vocab.json", "merges.txt"} <= expected.keys():
        raise ValueError("CLIP receipt lacks tokenizer assets")
    if not {"model.safetensors", "pytorch_model.bin"} & expected.keys():
        raise ValueError("CLIP receipt lacks model weights")
    if "clip_derivation" in assets:
        validate_derivation(root, manifest, record)
        return
    for relative, digest in expected.items():
        source = local_path(root / assets["text_encoder"], relative)
        if files.get(source.relative_to(root).as_posix()) != digest:
            raise ValueError(f"CLIP asset differs from trained receipt: {relative}")


TOKENIZER_FILES = frozenset(
    {
        "vocab.json",
        "merges.txt",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
    }
)
POSITION_IDS = "text_model.embeddings.position_ids"
CLIP_FORMAT = "dirtflow-clip-text-derivation"


def tensor_inventory(path: pathlib.Path, prefix: str | None = None) -> dict:
    """Hashes raw safetensors tensor bytes, shapes and dtypes without Torch."""
    size = path.stat().st_size
    with path.open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        if not 0 < length <= min(16 * 1024 * 1024, size - 8):
            raise ValueError("invalid safetensors header size")
        header = json.loads(stream.read(length))
        inventory = {}
        for name, entry in sorted(header.items()):
            if name == "__metadata__" or (prefix and (not name.startswith(prefix))):
                continue
            start, end = entry["data_offsets"]
            if not 0 <= start <= end <= size - 8 - length:
                raise ValueError("invalid safetensors tensor bounds")
            stream.seek(8 + length + start)
            digest = hashlib.sha256()
            remaining = end - start
            while remaining:
                data = stream.read(min(remaining, 1024 * 1024))
                if not data:
                    raise ValueError("truncated safetensors tensor")
                digest.update(data)
                remaining -= len(data)
            inventory[name] = {
                "shape": entry["shape"],
                "dtype": entry["dtype"],
                "sha256": digest.hexdigest(),
            }
    return inventory


def validate_derivation(root: pathlib.Path, manifest: dict, original: dict) -> None:
    """Checks a self-contained extraction receipt against trained source hashes."""
    assets, files = (manifest["assets"], manifest["files"])
    receipt = json.loads(local_path(root, assets["clip_derivation"]).read_text())
    if (
        receipt.get("format") != CLIP_FORMAT
        or receipt.get("version") != 1
        or receipt.get("algorithm") != "select-clip-text-tensors-v1"
        or (receipt.get("source_manifest_sha256") != files[assets["clip_manifest"]])
    ):
        raise ValueError("invalid CLIP text derivation identity")
    source = original["files"]
    if (
        receipt.get("source_weights")
        not in {"model.safetensors", "pytorch_model.bin"} & source.keys()
    ):
        raise ValueError("CLIP derivation lacks original weight identity")
    if files.get(assets.get("clip_source_config")) != source["config.json"]:
        raise ValueError("CLIP source configuration differs from trained identity")
    configuration = json.loads(
        local_path(root, assets["clip_source_config"]).read_text()
    )
    expected_config = copy.deepcopy(configuration.get("text_config", configuration))
    expected_config["architectures"] = ["CLIPTextModel"]
    target = local_path(root, assets["text_encoder"])
    if json.loads((target / "config.json").read_text()) != expected_config:
        raise ValueError("CLIP text configuration differs from source")
    expected_names = {
        "config.json",
        "model.safetensors",
    } | TOKENIZER_FILES & source.keys()
    if set(receipt.get("files", {})) != expected_names:
        raise ValueError("CLIP text derivation has extra or missing files")
    if {p.name for p in target.iterdir()} != expected_names:
        raise ValueError("CLIP text directory has extra or missing files")
    for name, digest in receipt["files"].items():
        if (
            files.get((pathlib.PurePosixPath(assets["text_encoder"]) / name).as_posix())
            != digest
        ):
            raise ValueError(f"CLIP derived file identity changed: {name}")
        if name in TOKENIZER_FILES and digest != source[name]:
            raise ValueError(f"CLIP tokenizer differs from trained identity: {name}")
    inventory = tensor_inventory(target / "model.safetensors")
    if (
        not inventory
        or inventory != receipt.get("text_tensors")
        or any(
            (not k.startswith("text_model.") or k == POSITION_IDS for k in inventory)
        )
    ):
        raise ValueError("CLIP text tensors differ from extraction receipt")
