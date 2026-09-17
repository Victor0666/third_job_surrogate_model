"""Stable short-path artifacts, hashes, JSON and CSV output."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

from project_paths import PROJECT_ROOT

from . import METHOD_ID
from .config import ComparisonConfig


def prepare_output(config: ComparisonConfig) -> Path:
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config.to_dict())
    write_json(
        output / "source.json",
        {
            "method_id": METHOD_ID,
        "artifact_protocol_status": (
            "legacy" if config.protocol == "legacy" else "formal"
        ),
            "source_hash": source_hash(),
            "source_hash_scheme": SOURCE_HASH_SCHEME_VERSION,
            "source_files": source_fingerprint(),
        },
    )
    return output


def portable_path(path: str | Path) -> str:
    """Return a project-relative POSIX path for persisted metadata."""

    value = Path(path)
    resolved = value.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return value.name


def write_json(path: str | Path, payload) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(
            payload,
            stream,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return path


def read_json(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_csv(
    path: str | Path,
    rows: Iterable[dict],
    *,
    fields: list[str] | None = None,
) -> Path:
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted(
            {
                key
                for row in rows
                for key, value in row.items()
                if not isinstance(value, (dict, list, tuple))
            }
        )
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fields})
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return path


def file_sha256(path: str | Path) -> str:
    """Hash a file without loading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


#: Bump whenever the set of fingerprinted files changes. Recorded alongside
#: every ``source_hash`` so an artifact written under an older scheme is
#: recognizable as "not comparable" rather than being reported as a mismatch.
SOURCE_HASH_SCHEME_VERSION = 2

#: Directories whose ``*.py`` files can change a simulation result. The
#: adapter layer lives in this package, but the environment, the fuzzy
#: machinery and the scenario tables it drives do not — a stage-1 checkpoint
#: trained against a different ``base/`` is not interchangeable with this one,
#: and scheme 1 (this package only) could not see that.
_FINGERPRINT_TREES = (
    ("algorithms", "comparisons", "drlea_nichgp"),
    ("algorithms", "llm_safe_hrl", "base"),
    ("common",),
)

#: Individually fingerprinted files outside those trees.
_FINGERPRINT_FILES = (
    ("algorithms", "llm_safe_hrl", "scenario_registry.py"),
)


def python_tree_identity(root: str | Path) -> dict[str, str]:
    """Map project-relative POSIX path -> sha256 for every ``*.py`` under root.

    Non-recursive on purpose: the fingerprinted trees are flat module
    directories, and descending into ``__pycache__`` would make the digest
    depend on the interpreter rather than on the source.
    """

    root = Path(root)
    if not root.is_dir():
        return {}
    return {
        path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix(): (
            file_sha256(path)
        )
        for path in sorted(root.glob("*.py"))
    }


def source_fingerprint() -> dict[str, str]:
    """Per-file digests of everything that can change a stage result."""

    root = PROJECT_ROOT.resolve()
    identity: dict[str, str] = {}
    for parts in _FINGERPRINT_TREES:
        identity.update(python_tree_identity(root.joinpath(*parts)))
    for parts in _FINGERPRINT_FILES:
        path = root.joinpath(*parts)
        if path.is_file():
            identity[path.relative_to(root).as_posix()] = file_sha256(path)
    return identity


def source_hash() -> str:
    digest = hashlib.sha256()
    digest.update(f"scheme={SOURCE_HASH_SCHEME_VERSION}".encode("utf-8"))
    for relative_path, file_digest in sorted(source_fingerprint().items()):
        digest.update(relative_path.encode("utf-8"))
        digest.update(file_digest.encode("utf-8"))
    return digest.hexdigest()


def compare_recorded_source_hash(manifest: dict | None) -> tuple[str, str]:
    """Classify a manifest's recorded ``source_hash`` against current code.

    Returns ``(status, message)`` where status is one of ``"match"``,
    ``"mismatch"``, ``"legacy_scheme"`` or ``"absent"``. Deliberately returns
    a verdict instead of raising: the stage-1 RA that is already trained
    recorded a scheme-1 hash, and failing hard on it would strand a
    checkpoint that cost days of compute.
    """

    if not isinstance(manifest, dict) or "source_hash" not in manifest:
        return ("absent", "no source_hash recorded")
    recorded_scheme = manifest.get("source_hash_scheme")
    if recorded_scheme is None:
        return (
            "legacy_scheme",
            "recorded under fingerprint scheme 1 (this package only); "
            f"not comparable with scheme {SOURCE_HASH_SCHEME_VERSION}",
        )
    if int(recorded_scheme) != SOURCE_HASH_SCHEME_VERSION:
        return (
            "legacy_scheme",
            f"recorded under fingerprint scheme {int(recorded_scheme)}; "
            f"not comparable with scheme {SOURCE_HASH_SCHEME_VERSION}",
        )
    recorded = str(manifest["source_hash"])
    current = source_hash()
    if recorded == current:
        return ("match", current)
    return (
        "mismatch",
        f"recorded {recorded[:16]}... but current code hashes "
        f"{current[:16]}...",
    )


def warn_if_source_changed(manifest_path: str | Path, *, label: str) -> str:
    """Print a warning when an artifact was produced by different code.

    Warn-only by design; see :func:`compare_recorded_source_hash`.
    """

    path = Path(manifest_path)
    if not path.is_file():
        print(f"[source guard] {label}: {path} not found, skipping check")
        return "absent"
    try:
        manifest = read_json(path)
    except (OSError, json.JSONDecodeError):
        print(f"[source guard] {label}: {path} is unreadable, skipping check")
        return "absent"
    status, message = compare_recorded_source_hash(manifest)
    if status == "match":
        return status
    if status == "mismatch":
        print(
            f"[source guard] WARNING {label}: {message}. The artifact was "
            "produced by different code; results may not be comparable."
        )
    else:
        print(f"[source guard] {label}: {message}")
    return status


def experiment_manifest(
    config: ComparisonConfig,
    *,
    stage: str,
    elapsed_seconds: float,
    failures: int = 0,
    invalid_individuals: int = 0,
    extra: dict | None = None,
) -> dict:
    return {
        "method_id": METHOD_ID,
        "artifact_protocol_status": (
            "legacy" if config.protocol == "legacy" else "formal"
        ),
        "schema_version": config.schema_version,
        "scenario": config.scenario,
        "ddl": config.ddl,
        "algorithm_seed": config.algorithm_seed,
        "optimizer_seed": int(config.algorithm_seed),
        "deadline_cache_paths": dict(config.deadline_cache_paths),
        "stage": str(stage),
        "source_hash": source_hash(),
        "source_hash_scheme": SOURCE_HASH_SCHEME_VERSION,
        "train_seeds": list(config.train_seeds),
        "validation_seeds": list(config.validation_seeds),
        "test_seeds": list(config.test_seeds),
        "protocol": config.protocol,
        "source_scenario": config.source_scenario,
        "resource_scale": config.resource_scale,
        "training_scenarios": list(config.training_scenarios),
        "test_scenarios": list(config.test_scenarios),
        "validation_interval": int(config.validation_interval),
        "elapsed_seconds": float(elapsed_seconds),
        "failure_count": int(failures),
        "invalid_individual_count": int(invalid_individuals),
        **dict(extra or {}),
    }
