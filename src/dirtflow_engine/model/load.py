"""Local bundle loading; all learned tensors are required."""

import pathlib
from typing import Any, cast

import torch
import transformers
from torch import nn

from .. import bundle
from ..schema import Schema
from .dense import LatentNorm, SparseStructureDecoder, StructureFlowModel
from .features import BlockStateEmbedding, ExtraConditioning, StructureContextEncoder
from .heads import CategoricalHead, IndependentCategoricalHead
from .sparse_flow import BlockStateFlowModel


class Models(nn.Module):
    def __init__(self, config: dict, schema: Schema) -> None:
        super().__init__()
        if config.get("feature_schema") != "v1":
            raise ValueError("this engine requires v1 observation features")
        emb, vae = config["embedding"], config["ss_vae"]
        structure, blocks = config["structure_flow"], config["blockstate_flow"]
        if (
            structure["resolution"] != 16
            or blocks["resolution"] != 64
            or not structure.get("self_conditioning")
            or not blocks.get("self_conditioning")
            or vae["latent_channels"] != 8
            or len(vae["channels"]) != 3
            or emb["dim"] != 8
            or structure["in_channels"] != 32
            or blocks["in_channels"] != 42
            or blocks.get("patch_size", 2) != 2
        ):
            raise ValueError("unsupported model feature dimensions")
        self.latent_norm = LatentNorm(vae["latent_channels"])
        self.ss_decoder = SparseStructureDecoder(
            latent_channels=vae["latent_channels"],
            channels=tuple(reversed(vae["channels"])),
            num_res_blocks=vae["num_res_blocks"],
            num_res_blocks_middle=vae["num_res_blocks_middle"],
            structure_coordinate_system=config["structure_coordinate_system"],
        )
        self.context_encoder = StructureContextEncoder(
            **config["context_encoder"],
            vocab_size=len(schema.states),
            embedding_dim=emb["dim"],
            embedding_rms_norm=emb["rms_norm"],
            structure_coordinate_system=config["structure_coordinate_system"],
        )
        shared = (
            "resolution",
            "in_channels",
            "out_channels",
            "model_channels",
            "cond_channels",
            "num_blocks",
            "num_heads",
            "mlp_ratio",
            "qk_rms_norm",
            "qk_rms_norm_cross",
            "self_conditioning",
        )
        self.structure_flow = StructureFlowModel(
            **{k: structure[k] for k in shared if k in structure}
        )
        widths = blocks["io_block_channels"]
        self.blockstate_flow = BlockStateFlowModel(
            **{k: blocks[k] for k in shared if k in blocks},
            io_block_channels=(widths,) if isinstance(widths, int) else tuple(widths),
            num_io_res_blocks=blocks["num_io_res_blocks"],
            coordinate_system=config["blockstate_coordinate_system"],
        )
        self.embedding = BlockStateEmbedding(
            len(schema.states), emb["dim"], rms_norm=emb["rms_norm"]
        )
        head = config["categorical_head"]
        head_type = {
            "categorical": CategoricalHead,
            "categorical_independent": IndependentCategoricalHead,
        }.get(head["kind"])
        if head_type is None:
            raise ValueError("unsupported categorical head")
        self.categorical_head = head_type(
            latent_dim=emb["dim"],
            hidden=head["hidden"],
            feat_dim=self.blockstate_flow.io_channels if head["io_features"] else 0,
            output_to_global=torch.from_numpy(schema.output_to_global.copy()),
        )
        conditioning = config["conditioning"]
        if conditioning["num_styles"] != 5 or conditioning["score_range"] != [
            0.0,
            10.0,
        ]:
            raise ValueError("unsupported caption conditioning")
        self.structure_extra = ExtraConditioning(
            structure["model_channels"], num_tasks=6
        )
        self.blockstate_extra = ExtraConditioning(blocks["model_channels"], num_tasks=6)


class TextEncoder:
    def __init__(self, path: pathlib.Path, device: str, attention_mask: bool) -> None:
        self.tokenizer = transformers.CLIPTokenizer.from_pretrained(
            path, local_files_only=True
        )
        self.model = (
            cast(Any, transformers.CLIPTextModel)
            .from_pretrained(path, local_files_only=True)
            .to(device)
            .eval()
            .requires_grad_(False)
        )
        self.device = device
        self.attention_mask = attention_mask

    @torch.no_grad()
    def encode(self, prompts: list[str]) -> torch.Tensor:
        """Returns CLIP token features [B,77,768], with checkpoint padding semantics."""
        tokens = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)
        args = {"input_ids": tokens["input_ids"]}
        if self.attention_mask:
            args["attention_mask"] = tokens["attention_mask"]
        return self.model(**args).last_hidden_state


def load(path: str, device: str = "cuda") -> tuple[Models, TextEncoder, Schema, dict]:
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if device == "cuda" and (
        not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("CUDA inference requires an NVIDIA GPU with BF16 support")
    root = pathlib.Path(path).resolve()
    manifest = bundle.validate_bundle(root)
    schema = Schema.load(root, manifest["assets"])
    with torch.random.fork_rng(devices=[]):
        models = Models(manifest["model"], schema)
    modules = dict(models.named_children())
    for relative in sorted(set(manifest["modules"].values())):
        payload = torch.load(
            root / relative, map_location="cpu", weights_only=True, mmap=True
        )
        expected = {
            name for name, file in manifest["modules"].items() if file == relative
        }
        if set(payload["modules"]) != expected:
            raise ValueError("weight payload ownership differs from manifest")
        for name, state in payload["modules"].items():
            if name != "ss_encoder":
                modules[name].load_state_dict(state, strict=True, assign=True)
    if not torch.equal(
        models.categorical_head.output_to_global.cpu(),
        torch.from_numpy(schema.output_to_global),
    ):
        raise ValueError("loaded output map differs from schema")
    models.float().eval().requires_grad_(False)
    for name in ("latent_norm", "structure_extra", "blockstate_extra"):
        getattr(models, name).to(device)
    text = TextEncoder(
        root / manifest["assets"]["text_encoder"],
        device,
        manifest["model"]["text_encoder"]["use_attention_mask"],
    )
    return models, text, schema, manifest
