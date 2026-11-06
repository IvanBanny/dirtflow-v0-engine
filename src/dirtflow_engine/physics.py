"""Editable-region projection for gravity, attachment, paired blocks and fluids.

These rules approximate Minecraft placement. They do not simulate game ticks.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .schema import Schema
from .schema import parse_state as parse_block_state

_GRAVITY_NAMES = (
    "minecraft:sand",
    "minecraft:red_sand",
    "minecraft:gravel",
    "minecraft:suspicious_sand",
    "minecraft:suspicious_gravel",
    "minecraft:anvil",
    "minecraft:dragon_egg",
)
_GRAVITY_SUFFIXES = ("_concrete_powder", "_anvil")
_FLOOR_SUFFIXES = (
    "_sapling",
    "_flower",
    "_tulip",
    "_orchid",
    "_mushroom",
    "_fungus",
    "_roots",
    "_sprouts",
    "_carpet",
    "_pressure_plate",
    "_rail",
)
_FLOOR_NAMES = (
    "minecraft:torch",
    "minecraft:redstone_wire",
    "minecraft:repeater",
    "minecraft:comparator",
    "minecraft:rail",
    "minecraft:short_grass",
    "minecraft:tall_grass",
    "minecraft:fern",
    "minecraft:dandelion",
    "minecraft:poppy",
    "minecraft:wheat",
    "minecraft:snow",
    "minecraft:sugar_cane",
    "minecraft:grass",
)
_WALL_PREFIXES = ("minecraft:wall_torch", "minecraft:ladder")
_WALL_SUFFIXES = ("_wall_sign", "_button", "_wall_torch", "_wall_banner")
_WALL_NAMES = ("minecraft:lever", "minecraft:vine", "minecraft:tripwire_hook")
_NON_SOLID_SUFFIXES = (
    _FLOOR_SUFFIXES
    + _WALL_SUFFIXES
    + ("_sign", "_banner", "_door", "_pane", "_fence", "_wall", "_torch", "_bars")
)
_NON_SOLID_NAMES = (
    _FLOOR_NAMES
    + _WALL_NAMES
    + (
        "minecraft:air",
        "minecraft:water",
        "minecraft:lava",
        "minecraft:fire",
        "minecraft:cobweb",
    )
)


@dataclass(frozen=True)
class BlockFlags:
    solid: bool = False
    gravity: bool = False
    needs_floor: bool = False
    needs_ceiling: bool = False
    wall_facing: str | None = None
    fluid_level: int | None = None


@lru_cache(maxsize=65536)
def classify(state: str) -> BlockFlags:
    name, props = parse_block_state(state)
    if name == "minecraft:air":
        return BlockFlags()
    if name in ("minecraft:water", "minecraft:lava"):
        return BlockFlags(fluid_level=int(props.get("level", "0")))
    gravity = name in _GRAVITY_NAMES or name.endswith(_GRAVITY_SUFFIXES)
    wall = (
        name.startswith(_WALL_PREFIXES)
        or name.endswith(_WALL_SUFFIXES)
        or name in _WALL_NAMES
    )
    if wall and props.get("face") == "floor":
        return BlockFlags(needs_floor=True)
    if wall and props.get("face") == "ceiling":
        return BlockFlags(needs_ceiling=True)
    if wall:
        return BlockFlags(wall_facing=props.get("facing", "north"))
    needs_floor = name in _FLOOR_NAMES or name.endswith(_FLOOR_SUFFIXES)
    if needs_floor:
        return BlockFlags(needs_floor=True)
    solid = not (name in _NON_SOLID_NAMES or name.endswith(_NON_SOLID_SUFFIXES))
    return BlockFlags(solid=solid or gravity, gravity=gravity)


class FlagTable:
    def __init__(self, registry: Schema) -> None:
        n = len(registry.states)
        self.solid = np.zeros(n, dtype=bool)
        self.gravity = np.zeros(n, dtype=bool)
        self.needs_floor = np.zeros(n, dtype=bool)
        self.needs_ceiling = np.zeros(n, dtype=bool)
        self.wall_facing = np.full(n, "", dtype=object)
        self.flowing_fluid = np.zeros(n, dtype=bool)
        self.fluid = np.zeros(n, dtype=bool)
        for i, state in enumerate(registry.states):
            flags = classify(state)
            self.solid[i] = flags.solid
            self.gravity[i] = flags.gravity
            self.needs_floor[i] = flags.needs_floor
            self.needs_ceiling[i] = flags.needs_ceiling
            self.wall_facing[i] = flags.wall_facing or ""
            self.fluid[i] = flags.fluid_level is not None
            self.flowing_fluid[i] = (
                flags.fluid_level is not None and flags.fluid_level > 0
            )


def gravity_apply(
    grid: np.ndarray, edit_mask: np.ndarray, flags: FlagTable
) -> tuple[np.ndarray, int]:
    out = grid.copy()
    moved = 0
    falling = flags.gravity[out] & edit_mask
    columns = np.argwhere(falling.any(axis=1))
    for x, z in columns:
        ys = np.nonzero(falling[x, :, z])[0]
        for y in ys:
            target = y
            while (
                target > 0
                and edit_mask[x, target - 1, z]
                and (out[x, target - 1, z] == 0)
            ):
                target -= 1
            if target != y:
                out[x, target, z] = out[x, y, z]
                out[x, y, z] = 0
                moved += 1
    return (out, moved)


def gravity_violations(grid: np.ndarray, flags: FlagTable) -> int:
    below_air = np.zeros(grid.shape, dtype=bool)
    below_air[:, 1:, :] = grid[:, :-1, :] == 0
    return int((flags.gravity[grid] & below_air).sum())


_FACING_BACK = {
    "north": (0, 0, 1),
    "south": (0, 0, -1),
    "east": (-1, 0, 0),
    "west": (1, 0, 0),
}


def _supported_below(grid: np.ndarray, flags: FlagTable) -> np.ndarray:
    support = np.zeros(grid.shape, dtype=bool)
    support[:, 1:, :] = flags.solid[grid[:, :-1, :]]
    return support


def _supported_above(grid: np.ndarray, flags: FlagTable) -> np.ndarray:
    support = np.zeros(grid.shape, dtype=bool)
    support[:, :-1, :] = flags.solid[grid[:, 1:, :]]
    return support


def _supported_behind(grid: np.ndarray, flags: FlagTable) -> np.ndarray:
    supported = np.zeros(grid.shape, dtype=bool)
    facings = flags.wall_facing[grid]
    solid = flags.solid[grid]
    for facing, (dx, dy, dz) in _FACING_BACK.items():
        here = facings == facing
        if not here.any():
            continue
        neighbor_solid = np.zeros(grid.shape, dtype=bool)
        if dx == 1:
            neighbor_solid[:-1, :, :] = solid[1:, :, :]
        elif dx == -1:
            neighbor_solid[1:, :, :] = solid[:-1, :, :]
        elif dz == 1:
            neighbor_solid[:, :, :-1] = solid[:, :, 1:]
        elif dz == -1:
            neighbor_solid[:, :, 1:] = solid[:, :, :-1]
        supported |= here & neighbor_solid
    return supported


def attachment_apply(
    grid: np.ndarray, edit_mask: np.ndarray, flags: FlagTable
) -> tuple[np.ndarray, int]:
    out = grid.copy()
    removed = 0
    for _ in range(3):
        floor_bad = flags.needs_floor[out] & ~_supported_below(out, flags) & edit_mask
        wall_ids = flags.wall_facing[out] != ""
        wall_bad = wall_ids & ~_supported_behind(out, flags) & edit_mask
        ceiling_bad = (
            flags.needs_ceiling[out] & ~_supported_above(out, flags) & edit_mask
        )
        bad = floor_bad | wall_bad | ceiling_bad
        if not bad.any():
            break
        out[bad] = 0
        removed += int(bad.sum())
    return (out, removed)


def attachment_violation_mask(grid: np.ndarray, flags: FlagTable) -> np.ndarray:
    floor_bad = flags.needs_floor[grid] & ~_supported_below(grid, flags)
    wall_bad = (flags.wall_facing[grid] != "") & ~_supported_behind(grid, flags)
    ceiling_bad = flags.needs_ceiling[grid] & ~_supported_above(grid, flags)
    return floor_bad | wall_bad | ceiling_bad


def attachment_violations(grid: np.ndarray, flags: FlagTable) -> int:
    return int(attachment_violation_mask(grid, flags).sum())


def _half_masks(
    grid: np.ndarray, registry: Schema
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    uppers = np.zeros(len(registry.states), dtype=bool)
    lowers = np.zeros(len(registry.states), dtype=bool)
    names = np.zeros(len(registry.states), dtype=np.int64)
    name_codes: dict[str, int] = {}
    for i, state in enumerate(registry.states):
        name, props = parse_block_state(state)
        half = props.get("half")
        if half in ("upper", "lower") and (
            name.endswith("_door")
            or "plant" in name
            or name.endswith("_rose_bush")
            or name.endswith("lilac")
            or name.endswith("peony")
            or name.endswith("sunflower")
            or name.endswith("tall_grass")
            or name.endswith("large_fern")
        ):
            uppers[i] = half == "upper"
            lowers[i] = half == "lower"
            names[i] = name_codes.setdefault(name, len(name_codes) + 1)
    return (uppers[grid], lowers[grid], names[grid])


def orientation_apply(
    grid: np.ndarray, edit_mask: np.ndarray, registry: Schema
) -> tuple[np.ndarray, int]:
    bad = _orphaned(grid, registry) & edit_mask
    out = grid.copy()
    out[bad] = 0
    return out, int(bad.sum())


def _orphaned(grid: np.ndarray, registry: Schema) -> np.ndarray:
    upper, lower, names = _half_masks(grid, registry)
    pair_below = np.zeros(grid.shape, dtype=bool)
    pair_below[:, 1:, :] = lower[:, :-1, :] & (names[:, 1:, :] == names[:, :-1, :])
    pair_above = np.zeros(grid.shape, dtype=bool)
    pair_above[:, :-1, :] = upper[:, 1:, :] & (names[:, :-1, :] == names[:, 1:, :])
    return (upper & ~pair_below) | (lower & ~pair_above)


def _has_fluid_neighbor(grid: np.ndarray, flags: FlagTable) -> np.ndarray:
    fluid = flags.fluid[grid]
    near = np.zeros(grid.shape, dtype=bool)
    near[1:] |= fluid[:-1]
    near[:-1] |= fluid[1:]
    near[:, 1:] |= fluid[:, :-1]
    near[:, :-1] |= fluid[:, 1:]
    near[:, :, 1:] |= fluid[:, :, :-1]
    near[:, :, :-1] |= fluid[:, :, 1:]
    return near


def fluid_apply(
    grid: np.ndarray, edit_mask: np.ndarray, flags: FlagTable
) -> tuple[np.ndarray, int]:
    out = grid.copy()
    bad = flags.flowing_fluid[out] & ~_has_fluid_neighbor(out, flags) & edit_mask
    out[bad] = 0
    return (out, int(bad.sum()))


def fluid_violations(grid: np.ndarray, flags: FlagTable) -> int:
    return int((flags.flowing_fluid[grid] & ~_has_fluid_neighbor(grid, flags)).sum())


class Physics:
    def __init__(self, schema: Schema) -> None:
        self.schema = schema
        self.flags = FlagTable(schema)

    def project(self, values: np.ndarray, edit: np.ndarray) -> np.ndarray:
        values, _ = gravity_apply(values, edit, self.flags)
        values, _ = attachment_apply(values, edit, self.flags)
        values, _ = orientation_apply(values, edit, self.schema)
        values, _ = fluid_apply(values, edit, self.flags)
        return values

    def violations(self, values: np.ndarray) -> int:
        return (
            gravity_violations(values, self.flags)
            + attachment_violations(values, self.flags)
            + int(_orphaned(values, self.schema).sum())
            + fluid_violations(values, self.flags)
        )
