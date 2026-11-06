"""Request sampling, candidate selection, and voxel results."""

import logging
import secrets
import time

import numpy as np

from . import request, sampling, voxel
from .jobs import Job, Mailbox


def choose(
    candidates: list[np.ndarray], before: np.ndarray, edit: np.ndarray, physics
) -> int:
    baseline = physics.violations(before)
    violations = [max(0, physics.violations(c) - baseline) for c in candidates]
    occupied = [(c != 0)[edit] for c in candidates]
    scores = []
    for i, a in enumerate(occupied):
        overlaps = []
        for j, b in enumerate(occupied):
            if i != j:
                union = int((a | b).sum())
                overlaps.append(float((a & b).sum()) / union if union else 1.0)
        scores.append(float(np.mean(overlaps)) if overlaps else 1.0)
    return min(range(len(candidates)), key=lambda i: (-scores[i], violations[i], i))


class Engine:
    def __init__(
        self,
        sampler: sampling.Sampler,
        physics: bool = True,
        batched_guidance: bool = True,
    ) -> None:
        self.sampler = sampler
        self.physics, self.batched_guidance = physics, batched_guidance

    def generate(self, job: Job, mailbox: Mailbox) -> dict:
        req, options = job.request, job.request.options
        logging.info("Request %s prompt: %r", job.id, req.prompt)
        if job.tensor is None:
            raise ValueError("job snapshot has been released")
        origin = (req.region.min[0], req.region.min[1], req.region.min[2])
        before, edit, observed, region = request.canvas(
            req, job.tensor, self.sampler.schema
        )
        settings = sampling.Settings(
            options.stage1_steps or options.steps or 30,
            options.stage2_steps or options.steps or 30,
            options.cfg_scale,
            self.physics,
            self.batched_guidance,
            options.preview_every_steps,
        )
        count = options.candidates or options.batch_size
        base_seed = options.seed if options.seed is not None else secrets.randbits(52)
        candidates: list[np.ndarray] = []
        sequence = 0
        started = time.monotonic()
        task: str = options.task_type
        if task == "replacement" and not np.any(before[edit]):
            task = (
                "scratch_ground" if np.any(before[observed & ~edit]) else "scratch_void"
            )
        for first in range(0, count, options.batch_size):
            n = min(options.batch_size, count - first)
            seeds = [(base_seed + first + i) % 2**53 for i in range(n)]

            def preview(stage, step, total, final, values):
                nonlocal sequence
                for i in range(n):
                    if job.canceled.is_set():
                        raise sampling.Cancelled
                    crop = values[(i, *region)].detach().cpu().numpy()
                    states = (
                        ["minecraft:air", "minecraft:stone"]
                        if crop.dtype == bool
                        else self.sampler.schema.states
                    )
                    tensor = voxel.from_array(
                        crop.astype(np.int64, copy=False), origin, states
                    )
                    payload = voxel.pack(tensor, options.tensor_compression)
                    if len(payload["data"]) > 4 * ((512 * 1024 + 2) // 3):
                        continue
                    sequence += 1
                    mailbox.put(
                        "preview",
                        frame_sequence=sequence,
                        candidate_index=first + i,
                        candidate_count=count,
                        seed=seeds[i],
                        stage=stage,
                        completed_step=step,
                        total_steps=total,
                        stage_end=final,
                        provisional=True,
                        full_replacement=True,
                        dimension=req.dimension,
                        origin=req.region.min,
                        size=req.region.size,
                        snapshot=payload,
                    )

            output = self.sampler.generate(
                before,
                edit,
                observed,
                req.prompt,
                seeds,
                task,
                settings,
                progress=lambda f: mailbox.put(
                    "progress",
                    fraction=(first + n * f) / count,
                    phase="GENERATING",
                    stage1_skipped=task == "material_only",
                ),
                canceled=job.canceled.is_set,
                preview=preview if options.preview_every_steps else None,
            )
            candidates.extend(output)
        if job.canceled.is_set():
            raise sampling.Cancelled
        winner = (
            choose(candidates, before, edit, self.sampler.physics) if count > 1 else 0
        )
        diff = voxel.from_array(
            candidates[winner][region],
            origin,
            self.sampler.schema.states,
            before[region],
        )
        return {
            "diff": voxel.pack(diff, options.tensor_compression),
            "candidate_index": winner,
            "candidate_count": count,
            "seed": (base_seed + winner) % 2**53,
            "effective_task_type": task,
            "stage1_steps": settings.stage1_steps,
            "stage2_steps": settings.stage2_steps,
            "cfg_scale": settings.cfg_scale,
            "batch_size": min(count, options.batch_size),
            "apply_physics": self.physics and task != "material_only",
            "batched_guidance": self.batched_guidance,
            "weight_set": self.sampler.manifest["weight_set"],
            "inference_policy": self.sampler.manifest.get("inference_policy"),
            "metrics": {"generation_seconds": time.monotonic() - started},
        }
