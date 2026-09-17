"""Auditable parameter-level evaluation cache with configuration isolation."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .parameter_schema import canonical_json_sha256


#: Bump when :data:`_SIMULATOR_TREES` or :data:`_SIMULATOR_FILES` changes.
SIMULATOR_FINGERPRINT_SCHEME = 1

# .../algorithms/llm_safe_hrl/LLM/rule_optimization/evaluation_cache.py
_PROJECT_ROOT = Path(__file__).resolve().parents[4]

#: Directories whose ``*.py`` files can change a simulation result. A cache
#: entry is a recorded simulator output, so reusing one produced by different
#: environment code is silently wrong in a way no key field can detect —
#: ``evaluation_config_hash`` and ``resource_config_hash`` only cover the
#: *configuration*, never the code that consumes it.
_SIMULATOR_TREES = (
    ("algorithms", "llm_safe_hrl", "base"),
    ("algorithms", "llm_safe_hrl", "LLM", "problems", "cews_task_constructive"),
    ("common",),
)

_SIMULATOR_FILES = (
    ("algorithms", "llm_safe_hrl", "scenario_registry.py"),
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def simulator_fingerprint() -> str:
    """Digest of every source file that can change a cached evaluation.

    Cached for the process lifetime: the fingerprint must be constant within
    one run, and re-hashing on every cache construction would be wasteful.

    Candidate rules generated during a run are written under the run's
    ``generated`` directory, never into the fingerprinted trees, so this stays
    stable across generations.
    """

    identity: dict[str, str] = {}
    for parts in _SIMULATOR_TREES:
        root = _PROJECT_ROOT.joinpath(*parts)
        if not root.is_dir():
            continue
        for path in sorted(root.glob("*.py")):
            key = path.relative_to(_PROJECT_ROOT).as_posix()
            identity[key] = _file_sha256(path)
    for parts in _SIMULATOR_FILES:
        path = _PROJECT_ROOT.joinpath(*parts)
        if path.is_file():
            identity[path.relative_to(_PROJECT_ROOT).as_posix()] = (
                _file_sha256(path)
            )
    if not identity:
        # An empty fingerprint would compare equal across unrelated codebases,
        # which is exactly the failure this guard exists to prevent.
        raise RuntimeError(
            "simulator fingerprint found no source files under "
            f"{_PROJECT_ROOT}; the project layout moved"
        )
    return canonical_json_sha256(
        {
            "scheme": SIMULATOR_FINGERPRINT_SCHEME,
            "files": identity,
        }
    )


def aggregate_seed_evaluations(
    results: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
) -> dict[str, Any]:
    """Aggregate cached one-seed evaluator outputs using the existing mean semantics."""
    if not results or len(results) != len(seeds):
        raise ValueError("one successful evaluation result is required for every seed")
    common_keys = set(results[0])
    for result in results[1:]:
        common_keys.intersection_update(result)
    numeric_keys = [
        key
        for key in common_keys
        if all(
            not isinstance(result[key], bool)
            and isinstance(result[key], (int, float))
            and math.isfinite(float(result[key]))
            for result in results
        )
    ]
    aggregate = {
        key: float(sum(float(result[key]) for result in results) / len(results))
        for key in numeric_keys
    }
    objectives = [float(result["objective"]) for result in results]
    objective_mean = float(sum(objectives) / len(objectives))
    objective_variance = float(
        sum((value - objective_mean) ** 2 for value in objectives) / len(objectives)
    )
    aggregate["objective_std_across_seeds"] = math.sqrt(objective_variance)
    aggregate["objective_max_across_seeds"] = max(objectives)
    aggregate["objective_cv_across_seeds"] = (
        aggregate["objective_std_across_seeds"]
        / max(abs(objective_mean), 1e-12)
    )
    aggregate["constraint_feasible"] = all(
        bool(result.get("constraint_feasible", False)) for result in results
    )
    aggregate["feasible_seed_rate"] = float(
        sum(bool(result.get("constraint_feasible", False)) for result in results)
        / len(results)
    )
    aggregate["max_deadline_violation_rate_across_seeds"] = max(
        float(
            result.get(
                "max_deadline_violation_rate_across_seeds",
                result.get("deadline_violation_rate", float("inf")),
            )
        )
        for result in results
    )
    aggregate["max_fuzzy_lateness"] = max(
        float(result.get("max_fuzzy_lateness", 0.0)) for result in results
    )
    aggregate["seeds"] = [int(seed) for seed in seeds]
    aggregate["evaluation_seed_count"] = len(results)
    aggregate["completed_seed_count"] = len(results)
    aggregate["all_evaluation_seeds_completed"] = True
    aggregate["per_seed_metrics"] = []
    for seed, result in zip(seeds, results):
        rows = result.get("per_seed_metrics", [])
        if isinstance(rows, list) and rows:
            row = dict(rows[0])
        else:
            row = {
                "seed": int(seed),
                "scenario_id": str(result.get("scenario_id", "unknown")),
                "constraint_feasible": bool(result.get("constraint_feasible", False)),
                "deadline_violation_rate": float(
                    result.get("deadline_violation_rate", float("inf"))
                ),
                "total_lateness": float(result.get("total_lateness", float("inf"))),
                "fuzzy_total_energy_score": float(result.get("objective", float("inf"))),
                "objective": float(result.get("objective", float("inf"))),
            }
        row["seed"] = int(seed)
        aggregate["per_seed_metrics"].append(row)
    passthrough = (
        "scenario_id",
        "structure_hash",
        "parameter_schema_hash",
        "best_parameter_hash",
        "optimizer_config_hash",
        "optimizer_seed",
        "parameter_diagnostics_hash",
        "training_seeds",
        "validation_seeds",
    )
    for key in passthrough:
        if key in results[0]:
            aggregate[key] = results[0][key]
    return aggregate


@dataclass(frozen=True)
class EvaluationCacheKey:
    """Fields that make one simulation result safe to reuse."""

    structure_hash: str
    parameter_vector: tuple[float, ...]
    seed: int
    scenario_id: str
    evaluation_config_hash: str
    resource_config_hash: str

    @classmethod
    def create(
        cls,
        *,
        structure_hash: str,
        parameter_vector: Sequence[float],
        seed: int,
        scenario_id: str,
        evaluation_config_hash: str,
        resource_config_hash: str,
        precision: int = 12,
    ) -> "EvaluationCacheKey":
        vector = tuple(round(float(value), int(precision)) for value in parameter_vector)
        if not all(math.isfinite(value) for value in vector):
            raise ValueError("cache parameter vector must contain finite values")
        return cls(
            structure_hash=str(structure_hash),
            parameter_vector=vector,
            seed=int(seed),
            scenario_id=str(scenario_id),
            evaluation_config_hash=str(evaluation_config_hash),
            resource_config_hash=str(resource_config_hash),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "structure_hash": self.structure_hash,
            "parameter_vector": list(self.parameter_vector),
            "seed": self.seed,
            "scenario_id": self.scenario_id,
            "evaluation_config_hash": self.evaluation_config_hash,
            "resource_config_hash": self.resource_config_hash,
        }

    @property
    def digest(self) -> str:
        return canonical_json_sha256(self.as_dict())


class EvaluationCache:
    """Small JSON-backed cache; failed evaluations are never stored.

    Durability uses a snapshot plus an append-only journal instead of
    rewriting the whole snapshot on every store. A full rewrite costs
    ``O(len(entries))``, so doing it per ``put`` makes a run quadratic in the
    number of cached evaluations; on a formal SeEvo run the snapshot reaches
    tens of megabytes and the serialization alone dominates the store.

    On-disk contract is unchanged: ``<path>`` always holds a complete,
    self-consistent ``schema_version == 1`` document. The journal beside it
    (``<path>.journal``) holds only entries added since the last compaction,
    one JSON object per line, and is replayed on load. A run interrupted
    mid-append loses at most the final partial line, which is skipped.

    Both files carry the :func:`simulator_fingerprint` they were written
    under, and a mismatch is **fail-closed**: the files are renamed aside and
    the run starts from an empty cache. Nothing is lost that cannot be
    recomputed, whereas reusing an entry produced by different environment
    code would silently corrupt every downstream fitness comparison.
    """

    #: Minimum number of journaled entries before a compaction is considered.
    _MIN_COMPACTION_ENTRIES = 64
    #: Compact once the journal reaches this fraction of the snapshot size,
    #: which keeps the amortized per-store cost constant.
    _COMPACTION_RATIO = 8

    def __init__(
        self,
        *,
        enabled: bool = True,
        path: str | Path | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.path = Path(path).resolve() if path else None
        self._journal_path = (
            self.path.with_name(self.path.name + ".journal")
            if self.path is not None
            else None
        )
        self._entries: dict[str, dict[str, Any]] = {}
        self._journaled = 0
        self.hits = 0
        self.misses = 0
        self.simulator_fingerprint = simulator_fingerprint()
        #: Set when a fingerprint mismatch forced the on-disk cache aside.
        self.discarded_reason: str | None = None
        if self.enabled and self.path is not None:
            if self.path.is_file():
                self._load()
            if self._journal_path is not None and self._journal_path.is_file():
                self._replay_journal()

    def _fingerprint_header(self) -> dict[str, Any]:
        return {
            "simulator_fingerprint": self.simulator_fingerprint,
            "simulator_fingerprint_scheme": SIMULATOR_FINGERPRINT_SCHEME,
        }

    def _set_aside(self, path: Path, reason: str) -> None:
        """Rename a cache file out of the way instead of deleting it.

        Keeping the file makes the discard auditable after the fact; deleting
        it would leave no evidence that a reusable cache was rejected.
        """
        if not path.is_file():
            return
        target = path.with_name(path.name + ".stale")
        index = 1
        while target.exists():
            target = path.with_name(f"{path.name}.stale.{index}")
            index += 1
        path.replace(target)
        self.discarded_reason = reason
        print(
            f"[evaluation cache] discarding {path.name}: {reason}; "
            f"moved to {target.name}"
        )

    def _discard_on_disk(self, reason: str) -> None:
        """Drop snapshot and journal together — the journal is only valid
        against the snapshot it was compacted from."""
        self._entries = {}
        self._journaled = 0
        self._set_aside(self.path, reason)
        if self._journal_path is not None:
            self._set_aside(self._journal_path, reason)

    def _fingerprint_mismatch(self, recorded: Any) -> str | None:
        """Return a human-readable reason, or ``None`` when reuse is safe."""
        if not isinstance(recorded, str):
            return (
                "written before simulator fingerprints were recorded, so the "
                "environment code behind it is unknown"
            )
        if recorded != self.simulator_fingerprint:
            return (
                f"simulator fingerprint {recorded[:16]}... does not match "
                f"current code {self.simulator_fingerprint[:16]}..."
            )
        return None

    def _load(self) -> None:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid evaluation cache: {self.path}") from exc
        if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
            raise ValueError(f"unsupported evaluation cache schema: {self.path}")
        reason = self._fingerprint_mismatch(payload.get("simulator_fingerprint"))
        if reason is not None:
            self._discard_on_disk(reason)
            return
        entries = payload.get("entries", {})
        if not isinstance(entries, Mapping):
            raise ValueError(f"evaluation cache entries must be a mapping: {self.path}")
        self._entries = {
            str(key): dict(value)
            for key, value in entries.items()
            if isinstance(value, Mapping) and value.get("status") == "ok"
        }

    def _replay_journal(self) -> None:
        """Fold journaled entries back in; a torn final line is dropped."""
        try:
            text = self._journal_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(
                f"invalid evaluation cache journal: {self._journal_path}"
            ) from exc
        replayed = 0
        header_seen = False
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # Only a crash-truncated tail may be unparsable. Anything
                # earlier would mean the file was rewritten out of band, and
                # dropping one entry then merely costs a re-evaluation.
                continue
            if not isinstance(record, Mapping):
                continue
            if "header" in record:
                # The header is the journal's own fingerprint. A journal can
                # outlive its snapshot (nothing is compacted until the
                # threshold), so it has to be checkable on its own.
                header = record["header"]
                recorded = (
                    header.get("simulator_fingerprint")
                    if isinstance(header, Mapping)
                    else None
                )
                reason = self._fingerprint_mismatch(recorded)
                if reason is not None:
                    self._discard_on_disk(reason)
                    return
                header_seen = True
                continue
            digest = record.get("digest")
            entry = record.get("entry")
            if not isinstance(digest, str) or not isinstance(entry, Mapping):
                continue
            if entry.get("status") != "ok":
                continue
            self._entries[digest] = dict(entry)
            replayed += 1
        if not header_seen and replayed:
            self._discard_on_disk(
                "journal has no fingerprint header, so the environment code "
                "behind it is unknown"
            )
            return
        self._journaled = replayed

    def _append_journal(self, records: Sequence[tuple[str, dict[str, Any]]]) -> None:
        """Append new entries, then compact once the journal grows large."""
        if self._journal_path is None or not records:
            return
        self._journal_path.parent.mkdir(parents=True, exist_ok=True)
        prefix = ""
        if not self._journal_path.is_file():
            prefix = (
                json.dumps(
                    {"header": self._fingerprint_header()},
                    ensure_ascii=True,
                    allow_nan=False,
                    sort_keys=True,
                )
                + "\n"
            )
        lines = "".join(
            json.dumps(
                {"digest": digest, "entry": entry},
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
            )
            + "\n"
            for digest, entry in records
        )
        with self._journal_path.open("a", encoding="utf-8") as handle:
            handle.write(prefix + lines)
            handle.flush()
        self._journaled += len(records)
        threshold = max(
            self._MIN_COMPACTION_ENTRIES,
            len(self._entries) // self._COMPACTION_RATIO,
        )
        if self._journaled >= threshold:
            self._persist()

    def _persist(self) -> None:
        """Write a complete snapshot and drop the now-redundant journal."""
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            **self._fingerprint_header(),
            "entries": self._entries,
        }
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=True, allow_nan=False, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(self.path)
        # Order matters: the snapshot must be in place before the journal is
        # discarded, otherwise a crash in between would lose entries.
        if self._journal_path is not None:
            self._journal_path.unlink(missing_ok=True)
        self._journaled = 0

    def flush(self) -> None:
        """Force a snapshot so ``<path>`` alone is complete.

        Call this at the end of a run, or before any consumer that reads the
        snapshot file directly rather than through this class.
        """
        if not self.enabled or self.path is None:
            return
        if self._journaled:
            self._persist()

    def get(self, key: EvaluationCacheKey) -> dict[str, Any] | None:
        if not self.enabled:
            self.misses += 1
            return None
        entry = self._entries.get(key.digest)
        if entry is None or entry.get("key") != key.as_dict():
            self.misses += 1
            return None
        result = entry.get("result")
        if not isinstance(result, Mapping):
            self.misses += 1
            return None
        self.hits += 1
        return dict(result)

    def put(
        self,
        key: EvaluationCacheKey,
        result: Mapping[str, Any],
        *,
        successful: bool = True,
    ) -> None:
        if not self.enabled or not successful:
            return
        if not isinstance(result, Mapping):
            raise TypeError("cached evaluation result must be a mapping")
        entry = {
            "status": "ok",
            "key": key.as_dict(),
            "result": dict(result),
        }
        self._entries[key.digest] = entry
        self._append_journal(((key.digest, entry),))

    def put_many(
        self,
        entries: Sequence[tuple[EvaluationCacheKey, Mapping[str, Any]]],
    ) -> None:
        """Store successful results with one atomic persistence operation."""
        if not self.enabled or not entries:
            return
        records: list[tuple[str, dict[str, Any]]] = []
        for key, result in entries:
            if not isinstance(result, Mapping):
                raise TypeError("cached evaluation result must be a mapping")
            entry = {
                "status": "ok",
                "key": key.as_dict(),
                "result": dict(result),
            }
            self._entries[key.digest] = entry
            records.append((key.digest, entry))
        self._append_journal(records)

    @property
    def requests(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return float(self.hits / self.requests) if self.requests else 0.0

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "entries": len(self._entries),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
            "journaled_entries": self._journaled,
            "simulator_fingerprint": self.simulator_fingerprint,
            "simulator_fingerprint_scheme": SIMULATOR_FINGERPRINT_SCHEME,
            "discarded_reason": self.discarded_reason,
        }
