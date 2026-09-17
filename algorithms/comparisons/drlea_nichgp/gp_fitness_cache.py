"""Versioned persistent cache for exact GP training fitness values."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import torch

from algorithms.llm_safe_hrl import scenario_registry
from project_paths import PROJECT_ROOT

from .checkpointing import (
    SOURCE_HASH_SCHEME_VERSION,
    file_sha256,
    python_tree_identity as _python_tree_identity,
    read_json,
    source_hash,
    write_json,
)
from .gp_features import GP_TERMINALS, GP_TERMINAL_VERSION


CACHE_SCHEMA_VERSION = 1


def _resolve_project_file(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _file_identity(value: str | Path) -> dict:
    path = _resolve_project_file(value)
    return {
        "path": Path(value).as_posix(),
        "sha256": file_sha256(path) if path.is_file() else None,
    }


def routing_online_hash(routing_agent) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(
        routing_agent.online.state_dict().items()
    ):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def evaluation_identity(config, routing_agent, device: str) -> dict:
    deadline_paths = {
        scenario: _file_identity(path)
        for scenario, path in sorted(config.deadline_cache_paths.items())
    }
    deadline_paths.setdefault(
        config.scenario,
        _file_identity(config.deadline_cache_path),
    )
    payload = {
        "config": config.to_dict(),
        "gp_terminal_version": GP_TERMINAL_VERSION,
        "gp_terminals": list(GP_TERMINALS),
        "ra_online_sha256": routing_online_hash(routing_agent),
        "dax_files": [_file_identity(path) for path in config.dax_paths],
        "deadline_cache_files": deadline_paths,
        "drlea_source_hash": source_hash(),
        "drlea_source_hash_scheme": SOURCE_HASH_SCHEME_VERSION,
        "shared_source_files": {
            Path(scenario_registry.__file__)
            .relative_to(PROJECT_ROOT)
            .as_posix(): file_sha256(Path(scenario_registry.__file__)),
            **_python_tree_identity(
                PROJECT_ROOT / "algorithms" / "llm_safe_hrl" / "base"
            ),
            **_python_tree_identity(PROJECT_ROOT / "common"),
        },
        "evaluation_device": str(device),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "sha256": hashlib.sha256(canonical).hexdigest(),
        "payload": payload,
    }


class GPFitnessCache:
    def __init__(self, path: str | Path, identity: dict):
        self.path = Path(path)
        self.identity = identity
        self.entries: dict[tuple[str, ...], tuple[float, ...]] = {}
        self.status = "missing"
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            payload = read_json(self.path)
            if (
                int(payload.get("schema_version", -1))
                != CACHE_SCHEMA_VERSION
                or payload.get("evaluation_fingerprint")
                != self.identity["sha256"]
            ):
                self.status = "stale"
                return
            for row in payload.get("entries", []):
                tokens = tuple(str(token) for token in row["tokens"])
                fitness = tuple(
                    float(value) for value in row["comparison_key"]
                )
                if len(fitness) != 4 or not all(
                    np.isfinite(fitness)
                ):
                    raise ValueError("invalid cached GP fitness")
                self.entries[tokens] = fitness
            self.status = "loaded"
        except (OSError, KeyError, TypeError, ValueError):
            self.entries.clear()
            self.status = "corrupt"

    def save(self, entries) -> Path:
        rows = [
            {
                "tokens": list(tokens),
                "comparison_key": list(fitness),
            }
            for tokens, fitness in sorted(entries.items())
            if len(fitness) == 4
            and all(np.isfinite(float(value)) for value in fitness)
        ]
        write_json(
            self.path,
            {
                "schema_version": CACHE_SCHEMA_VERSION,
                "evaluation_fingerprint": self.identity["sha256"],
                "evaluation_identity": self.identity["payload"],
                "entries": rows,
            },
        )
        self.entries = {
            tuple(row["tokens"]): tuple(row["comparison_key"])
            for row in rows
        }
        self.status = "saved"
        return self.path
