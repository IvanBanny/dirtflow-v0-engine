"""Two-stage Euler sampling in the checkpoint's latent spaces."""

import contextlib
import dataclasses
from collections.abc import Callable

import numpy as np
import torch
from torch.nn import functional as F

from . import bundle
from .model import features, heads, sparse
from .model.load import load
from .physics import Physics


class Cancelled(Exception):
    """Sampling stopped at a cooperative cancellation point."""


@dataclasses.dataclass(frozen=True)
class Settings:
    stage1_steps: int = 30
    stage2_steps: int = 30
    cfg_scale: float = 5.0
    physics: bool = True
    batched_guidance: bool = True
    preview_every: int = 3

    def __post_init__(self) -> None:
        if any(
            type(n) is not int or not 1 <= n <= 200
            for n in (self.stage1_steps, self.stage2_steps)
        ):
            raise ValueError("stage steps must be integers in 1..200")
        if not np.isfinite(self.cfg_scale) or not 0 <= self.cfg_scale <= 20:
            raise ValueError("CFG scale must be finite and in 0..20")
        if type(self.preview_every) is not int or not 0 <= self.preview_every <= 10000:
            raise ValueError("preview cadence must be in 0..10000")


def clean_estimate(x: torch.Tensor, velocity: torch.Tensor, t: float) -> torch.Tensor:
    time = x.new_full((x.shape[0], *([1] * (x.ndim - 1))), t)
    return (1 - 1e-5) * x - (1e-5 + (1 - 1e-5) * time) * velocity


def support(
    logits: torch.Tensor,
    edit: torch.Tensor,
    observed: torch.Tensor,
    before: torch.Tensor,
    task: str,
) -> torch.Tensor:
    """Keeps edit proposals and at most 8192 nearest observed context cells."""
    if not torch.isfinite(logits).all():
        raise ValueError("nonfinite occupancy logits")
    selected = (before != 0) & edit if task == "material_only" else (logits >= 0) & edit
    if task == "repair":
        selected |= (before != 0) & edit
    known = observed & ~edit
    retained = torch.zeros_like(known, memory_format=torch.contiguous_format)
    covered, remaining = edit.clone(), 8192
    for _ in range(max(logits.shape)):
        if not remaining or not (known & ~covered).any():
            break
        expanded = F.max_pool3d(covered[None, None].float(), 3, 1, 1)[0, 0].bool()
        layer = (expanded & ~covered & known).reshape(-1).nonzero().flatten()
        chosen = layer[:remaining]
        retained.view(-1)[chosen] = True
        remaining -= chosen.numel()
        covered = expanded
    return selected | retained


def guided_grid(grid: sparse.SparseGrid, feats: torch.Tensor) -> sparse.SparseGrid:
    if "guidance" not in grid.cache:
        coords = grid.coords.repeat(2, 1)
        coords[len(grid) :, 0] += grid.batch_size
        grid.cache["guidance"] = (coords, {})
    coords, cache = grid.cache["guidance"]
    return sparse.SparseGrid(
        coords, torch.cat((feats, feats)), 2 * grid.batch_size, grid.resolution, cache
    )


class Sampler:
    """Owns the models; each call owns its RNG and sparse geometry."""

    def __init__(self, path: str, device: str = "cuda", offload: bool = True) -> None:
        self.models, self.text, self.schema, self.manifest = load(path, device)
        self.device, self.offload = device, offload
        self.corrected = (
            bundle.inference_policy(self.manifest)["editable_confidence"]
            == "zero_normalized_editable"
        )
        self.physics = Physics(self.schema)
        if not offload:
            self.models.to(device)

    def residency(self, stage: int) -> None:
        if not self.offload:
            return
        groups = (
            ("context_encoder", "structure_flow", "ss_decoder"),
            ("blockstate_flow", "embedding", "categorical_head"),
        )
        for name in groups[2 - stage]:
            getattr(self.models, name).to("cpu")
        for name in groups[stage - 1]:
            getattr(self.models, name).to(self.device)

    @torch.no_grad()
    def generate(
        self,
        before: np.ndarray,
        edit: np.ndarray,
        observed: np.ndarray,
        prompt: str,
        seeds: list[int],
        task: str = "replacement",
        settings: Settings = Settings(),
        progress: Callable[[float], None] | None = None,
        canceled: Callable[[], bool] | None = None,
        preview: Callable[[str, int, int, bool, torch.Tensor], None] | None = None,
    ) -> np.ndarray:
        """Returns IDs [B,64,64,64], preserving every cell outside edit.

        Inputs use Minecraft XYZ. before is int64; edit and observed are bool.
        Each candidate has its own RNG. preview receives borrowed device tensors.
        """
        if (
            before.shape != (64, 64, 64)
            or edit.shape != before.shape
            or observed.shape != before.shape
        ):
            raise ValueError("generation requires matching 64-cube arrays")
        if before.dtype != np.int64 or edit.dtype != bool or observed.dtype != bool:
            raise ValueError("expected int64 states and boolean masks")
        if not edit.any() or np.any(edit & ~observed):
            raise ValueError("edit must be nonempty and fully observed")
        if np.any(before < 0) or np.any(before >= len(self.schema.states)):
            raise ValueError("input ID outside registry")
        if task not in features.TASK_NAMES or not 1 <= len(seeds) <= 8:
            raise ValueError("unsupported task or candidate batch")
        if any(type(seed) is not int or not 0 <= seed < 2**53 for seed in seeds):
            raise ValueError("seeds must be integers in 0..2^53-1")
        if task == "replacement" and not np.any(before[edit]):
            task = (
                "scratch_ground" if np.any(before[observed & ~edit]) else "scratch_void"
            )
        batch = len(seeds)
        rngs = [torch.Generator(self.device).manual_seed(seed) for seed in seeds]
        canvas, mask, seen = [
            torch.from_numpy(np.ascontiguousarray(x))
            .to(self.device)[None]
            .expand(batch, -1, -1, -1)
            .contiguous()
            for x in (before, edit, observed)
        ]
        m = self.models

        def report(fraction: float) -> None:
            if canceled is not None and canceled():
                raise Cancelled
            if progress is not None:
                progress(fraction)

        def publish(stage, step, total, values, final=False):
            if (
                preview is not None
                and settings.preview_every
                and (final or step < total and step % settings.preview_every == 0)
            ):
                preview(stage, step, total, final, values)

        report(0)
        # Encode separately to preserve CLIP's single-prompt numerical path
        cond = [self.text.encode([text]).expand(batch, -1, -1) for text in ("", prompt)]
        styles = torch.full((batch,), 5, device=self.device, dtype=torch.long)
        scores = torch.full((batch,), 9.0, device=self.device)
        fragments = torch.zeros(batch, device=self.device, dtype=torch.bool)
        domain = torch.zeros_like(styles)
        tasks = torch.full_like(styles, features.TASK_NAMES.index(task))
        extras = [
            module(styles, scores, fragments, domain, tasks)
            for module in (m.structure_extra, m.blockstate_extra)
        ]
        autocast = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if self.device == "cuda"
            else contextlib.nullcontext()
        )
        with autocast:
            if task == "material_only":
                logits = (canvas != 0).float() * 2 - 1
            else:
                self.residency(1)
                context = m.context_encoder(canvas, mask, seen)
                z = torch.cat(
                    [
                        torch.randn(1, 8, 16, 16, 16, device=self.device, generator=rng)
                        for rng in rngs
                    ]
                )
                previous = torch.zeros_like(z)
                times = torch.linspace(1, 0, settings.stage1_steps + 1)
                for i in range(settings.stage1_steps):
                    report(0.45 * i / settings.stage1_steps)
                    t, next_t = float(times[i]), float(times[i + 1])
                    x = torch.cat((z, context, previous), dim=1)
                    if settings.batched_guidance:
                        v = m.structure_flow(
                            torch.cat((x, x)),
                            torch.full((2 * batch,), t * 1000, device=self.device),
                            torch.cat(cond),
                            torch.cat((extras[0], extras[0])),
                        )
                        uncond, full = v.chunk(2)
                    else:
                        t_vec = torch.full((batch,), t * 1000, device=self.device)
                        uncond, full = [
                            m.structure_flow(x, t_vec, c, extras[0]) for c in cond
                        ]
                    v = uncond + settings.cfg_scale * (full - uncond)
                    previous = clean_estimate(z, v, t)
                    if (
                        preview is not None
                        and settings.preview_every
                        and i + 1 < settings.stage1_steps
                        and (i + 1) % settings.preview_every == 0
                    ):
                        estimate = m.ss_decoder(m.latent_norm.inverse(previous))[
                            :, 0
                        ].float()
                        publish(
                            "structure",
                            i + 1,
                            settings.stage1_steps,
                            torch.where(mask, estimate >= 0, (canvas != 0) & seen),
                        )
                    z = z - (t - next_t) * v
                logits = m.ss_decoder(m.latent_norm.inverse(z))[:, 0].float()
                publish(
                    "structure",
                    settings.stage1_steps,
                    settings.stage1_steps,
                    torch.where(mask, logits >= 0, (canvas != 0) & seen),
                    True,
                )
            report(0.45)
            self.residency(2)
            active = torch.stack(
                [
                    support(logits[i], mask[i], seen[i], canvas[i], task)
                    for i in range(batch)
                ]
            )
            out = canvas.clone().masked_fill(mask, 0)
            if active.any():
                grid = sparse.from_dense_mask(active)
                b, x, y, z = grid.coords.unbind(1)
                known = ~mask[b, x, y, z]
                ids = canvas[b, x, y, z] * known
                table = m.embedding.table()
                clean = table[ids]
                known_feats = clean * known[:, None]
                old = table[canvas[b, x, y, z]]
                old_valid = mask[b, x, y, z] & seen[b, x, y, z]
                neighbors = features.neighbor_observations(grid.coords, seen)
                confidence = torch.zeros(len(grid), device=self.device)
                if not self.corrected:
                    for i in range(batch):
                        select = ~known & (b == i)
                        values = logits[b[select], x[select], y[select], z[select]]
                        if values.numel():
                            confidence[select] = (
                                (values - values.mean())
                                / values.var(unbiased=False).clamp_min(1e-6).sqrt()
                            ).clamp(-4, 4)
                indicator = table.new_zeros(batch, m.categorical_head.output_vocab)
                for i in range(batch):
                    classes = m.categorical_head.target_classes(
                        canvas[i][seen[i]].unique()
                    )
                    indicator[i, classes[classes >= 0]] = 1
                air_class = m.categorical_head.target_classes(
                    torch.tensor([0], device=self.device)
                )
                indicator[:, air_class] = 1
                lengths = grid.lengths().tolist()
                has_known = [bool(part.any()) for part in known.split(lengths)]

                def noise(repaint=False):
                    result = torch.zeros_like(clean)
                    for i, part in enumerate(result.split(lengths)):
                        if not repaint or has_known[i]:
                            part.normal_(generator=rngs[i])
                    return result

                latent = noise()
                previous = torch.zeros_like(latent)
                io = None

                def decode(value):
                    prediction = heads.argmax_over_vocab(
                        m.categorical_head,
                        value,
                        table,
                        io,
                        chunk=256,
                        context_indicator=indicator,
                        batch_index=b,
                        forbidden_global_ids=torch.tensor([0], device=self.device)
                        if task == "material_only"
                        else None,
                    )
                    result = canvas.clone().masked_fill(mask, 0)
                    result[b, x, y, z] = torch.where(known, ids, prediction)
                    return result

                times = torch.linspace(1, 0, settings.stage2_steps + 1)
                for i in range(settings.stage2_steps):
                    report(0.45 + 0.5 * i / settings.stage2_steps)
                    t, next_t = float(times[i]), float(times[i + 1])
                    feats = features.sparse_observation_features(
                        latent,
                        known_feats,
                        known,
                        confidence[:, None],
                        (~known)[:, None].to(clean.dtype),
                        previous,
                        old,
                        old_valid,
                        neighbors,
                    )
                    if settings.batched_guidance:
                        result, all_io = m.blockstate_flow(
                            guided_grid(grid, feats),
                            torch.full((2 * batch,), t * 1000, device=self.device),
                            torch.cat(cond),
                            torch.cat((extras[1], extras[1])),
                            return_io_feats=True,
                        )
                        uncond, full = result.feats.chunk(2)
                        io = all_io[len(grid) :]
                    else:
                        outputs = []
                        for c in cond:
                            result, io = m.blockstate_flow(
                                grid.replace_feats(feats),
                                torch.full((batch,), t * 1000, device=self.device),
                                c,
                                extras[1],
                                return_io_feats=True,
                            )
                            outputs.append(result.feats)
                        uncond, full = outputs
                    velocity = uncond + settings.cfg_scale * (full - uncond)
                    previous = clean_estimate(latent, velocity, t)
                    if (
                        preview is not None
                        and settings.preview_every
                        and i + 1 < settings.stage2_steps
                        and (i + 1) % settings.preview_every == 0
                    ):
                        publish(
                            "blockstates",
                            i + 1,
                            settings.stage2_steps,
                            decode(previous),
                        )
                    latent = latent - (t - next_t) * velocity
                    if any(has_known):
                        t_known = clean.new_full((len(clean), 1), next_t)
                        renoised = (1 - t_known) * clean + (
                            1e-5 + (1 - 1e-5) * t_known
                        ) * noise(True)
                        latent[known] = renoised[known].to(latent.dtype)
                out = decode(latent)
            publish(
                "blockstates", settings.stage2_steps, settings.stage2_steps, out, True
            )
        report(0.95)
        results = out.cpu().numpy()
        for i in range(batch):
            if settings.physics and task != "material_only":
                results[i] = self.physics.project(results[i], edit)
            results[i][~edit] = before[~edit]
            if task == "material_only" and np.any(
                ((results[i] != 0) != (before != 0)) & edit
            ):
                raise ValueError("material-only generation changed occupancy")
            changed = edit & (results[i] != before)
            if not np.isin(results[i][changed], self.schema.output_to_global).all():
                raise ValueError("generation emitted an illegal output state")
        report(1)
        return results
