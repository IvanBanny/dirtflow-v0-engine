"""Game-exported states and the checkpoint's legal class map."""

import gzip
import json
import pathlib

import numpy as np


def parse_state(state: str) -> tuple[str, dict[str, str]]:
    name, sep, tail = state.strip().partition("[")
    if ":" not in name:
        name = "minecraft:" + name
    properties = {}
    if sep:
        if not tail.endswith("]") or "[" in tail or "]" in tail[:-1]:
            raise ValueError("malformed block-state properties")
        for pair in tail[:-1].split(","):
            key, equals, value = pair.partition("=")
            if not equals or not key or not value or key in properties or "=" in value:
                raise ValueError("invalid or duplicate block-state property")
            properties[key] = value
    elif "]" in name:
        raise ValueError("malformed block state")
    return name, properties


def format_state(name: str, properties: dict[str, str]) -> str:
    return name + (
        "[" + ",".join(f"{k}={properties[k]}" for k in sorted(properties)) + "]"
        if properties
        else ""
    )


class Schema:
    def __init__(self, states: list[str], payload: dict, mapping: np.ndarray) -> None:
        if states[:2] != ["minecraft:air", "dirtflow:unknown"] or len(
            set(states)
        ) != len(states):
            raise ValueError("invalid input registry")
        if (
            payload.get("format") != "dirtflow-game-state-schema"
            or payload.get("version") != 1
            or payload.get("minecraft_version") != "1.21.11"
        ):
            raise ValueError("expected Minecraft 1.21.11 game schema v1")
        self.states = states
        self.index = {s: i for i, s in enumerate(states)}
        self.entries: dict[str, dict] = {}
        self.defaults: dict[str, dict[str, str]] = {}
        for entry in payload["states"]:
            state = entry["state"]
            name, props = parse_state(state)
            if (
                state in self.entries
                or format_state(name, props) != state
                or type(entry["is_air"]) is not bool
            ):
                raise ValueError("invalid game schema entry")
            self.entries[state] = entry
            if entry.get("default"):
                if name in self.defaults:
                    raise ValueError("duplicate game default")
                self.defaults[name] = props
        if {parse_state(s)[0] for s in self.entries} != self.defaults.keys():
            raise ValueError("missing game default")
        expected = ["minecraft:air"] + sorted(
            s for s, e in self.entries.items() if not e["is_air"]
        )
        try:
            ids = np.array([self.index[s] for s in expected], dtype=np.int64)
        except KeyError as error:
            raise ValueError("game state absent from input registry") from error
        if mapping.dtype != np.int64 or not np.array_equal(ids, mapping):
            raise ValueError("class mapping disagrees with game schema")
        self.output_to_global = mapping

    def normalize_input(self, state: str) -> int:
        """Completes game defaults; air aliases become zero and unknown states one."""
        try:
            name, props = parse_state(state)
            defaults = self.defaults.get(name)
            if defaults is None or not props.keys() <= defaults.keys():
                return 1
            canonical = format_state(name, defaults | props)
            entry = self.entries.get(canonical)
            if entry is None:
                return 1
            return 0 if entry["is_air"] else self.index[canonical]
        except ValueError:
            return 1

    @classmethod
    def load(cls, root: pathlib.Path, assets: dict) -> "Schema":
        registry = json.loads((root / assets["registry"]).read_text())
        if (
            registry.get("format") != "dirtflow-blockstate-vocab"
            or registry.get("version") != 1
        ):
            raise ValueError("unsupported input registry")
        path = root / assets["schema"]
        raw = (
            gzip.decompress(path.read_bytes())
            if path.suffix == ".gz"
            else path.read_bytes()
        )
        with np.load(root / assets["mapping"], allow_pickle=False) as saved:
            mapping = saved["output_to_global"].copy()
        return cls(registry["states"], json.loads(raw), mapping)
