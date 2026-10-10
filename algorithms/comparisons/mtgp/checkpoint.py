"""Atomic generation checkpoints, including exact fitness cache and RNG state."""

import json
import math
import os
from pathlib import Path


SCHEMA_VERSION = 1


def write_checkpoint(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def read_checkpoint(path, identity, settings):
    state = json.loads(Path(path).read_text(encoding="utf-8"))
    if state.get("schema_version") != SCHEMA_VERSION or state.get("identity") != identity:
        raise ValueError("MTGP checkpoint protocol/input/source/runtime identity mismatch")
    if state.get("settings") != settings:
        raise ValueError("MTGP checkpoint training settings mismatch")
    generation = state["next_generation"]
    if not isinstance(generation, int) or not 0 <= generation <= settings["generations"]:
        raise ValueError("invalid MTGP checkpoint generation")
    if len(state["population"]) != settings["population_size"] or len(state["history"]) != generation:
        raise ValueError("invalid MTGP checkpoint population/history")
    for row in state["fitness_cache"]:
        fitness = row["fitness"]
        if fitness is not None and (len(fitness) != 4 or any(not math.isfinite(value) or value < 0 for value in fitness)):
            raise ValueError("invalid cached MTGP fitness")
    return state
