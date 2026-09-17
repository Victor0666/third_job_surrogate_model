"""Exact diagnostic pruning by replaying frozen rules on complete baseline traces."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Iterable

import numpy as np

from base.hrl_env import validate_task_priority_scores
from counterfactual_feedback.critical_state_replay import load_frozen_priority_rule
from counterfactual_feedback.trace_recorder import FEATURE_NAMES


TRACE_SCHEMA_VERSION = "diagnostic_decision_trace_v1"


class _TraceMismatch(ValueError):
    pass


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=256)
def _cached_trace_sha256(path: str, size: int, modified_ns: int) -> str:
    del size, modified_ns
    return _file_sha256(path)


@dataclass(frozen=True)
class ReplayVerdict:
    scenario_id: str
    seed: int
    identical: bool
    divergence_count_along_baseline: int
    first_divergence_index: int | None
    margin_stats: dict[str, float] = field(default_factory=dict)
    status: str = "ok"
    baseline_frozen_rule_hash: str = ""
    candidate_frozen_rule_hash: str = ""
    trace_sha256: str = ""
    replay_seconds: float = 0.0


class DecisionTraceCapture:
    """Collect one complete evaluator trajectory without object arrays."""

    def __init__(
        self,
        path: str | Path,
        *,
        baseline_frozen_rule_hash: str,
        evaluation_config_sha256: str,
        scenario_id: str,
        seed: int,
        max_mb: int = 128,
    ) -> None:
        self.path = Path(path)
        self.baseline_frozen_rule_hash = str(baseline_frozen_rule_hash)
        self.evaluation_config_sha256 = str(evaluation_config_sha256)
        self.scenario_id = str(scenario_id)
        self.seed = int(seed)
        self.max_bytes = int(max_mb) * 1024 * 1024
        self.ready_ids: list[np.ndarray] = []
        self.features = {name: [] for name in FEATURE_NAMES}
        self.selected_indices: list[int] = []
        self._estimated_bytes = 0

    def record(self, ready_ids: Iterable[int], selection_details: dict[str, Any]) -> None:
        ids = np.asarray(list(ready_ids), dtype=np.int64)
        if ids.ndim != 1 or not len(ids):
            raise ValueError("decision trace requires a non-empty ready set")
        selected_task_id = int(selection_details["selected_task_id"])
        matches = np.flatnonzero(ids == selected_task_id)
        if len(matches) != 1:
            raise ValueError("selected task must occur exactly once in the ready set")
        feature_rows = {}
        for name in FEATURE_NAMES:
            values = np.asarray(selection_details["features"][name], dtype=np.float64)
            if values.shape != ids.shape or not np.isfinite(values).all():
                raise ValueError(f"invalid decision-trace feature: {name}")
            feature_rows[name] = values.copy()
        projected = self._estimated_bytes + ids.nbytes + sum(
            values.nbytes for values in feature_rows.values()
        ) + np.dtype(np.int32).itemsize
        if projected > self.max_bytes:
            raise RuntimeError("decision trace exceeded max_trace_mb_per_structure")
        self._estimated_bytes = projected
        self.ready_ids.append(ids.copy())
        for name, values in feature_rows.items():
            self.features[name].append(values)
        self.selected_indices.append(int(matches[0]))

    def finalize(self, *, expected_workflows: int, completed_workflows: int) -> Path:
        if int(completed_workflows) != int(expected_workflows):
            raise RuntimeError("cannot finalize an incomplete decision trace")
        lengths = np.asarray([len(row) for row in self.ready_ids], dtype=np.int64)
        offsets = np.concatenate((np.asarray([0], dtype=np.int64), np.cumsum(lengths)))
        ready_flat = (
            np.concatenate(self.ready_ids).astype(np.int64, copy=False)
            if self.ready_ids else np.asarray([], dtype=np.int64)
        )
        payload: dict[str, Any] = {
            "schema_version": np.asarray(TRACE_SCHEMA_VERSION),
            "feature_names": np.asarray(FEATURE_NAMES, dtype="U32"),
            "baseline_frozen_rule_hash": np.asarray(self.baseline_frozen_rule_hash),
            "evaluation_config_sha256": np.asarray(self.evaluation_config_sha256),
            "scenario_id": np.asarray(self.scenario_id),
            "seed": np.asarray(self.seed, dtype=np.int64),
            "decision_count": np.asarray(len(self.selected_indices), dtype=np.int64),
            "expected_workflows": np.asarray(expected_workflows, dtype=np.int64),
            "completed_workflows": np.asarray(completed_workflows, dtype=np.int64),
            "complete": np.asarray(True),
            "ready_ids_flat": ready_flat,
            "ready_offsets": offsets,
            "selected_indices": np.asarray(self.selected_indices, dtype=np.int32),
        }
        for name in FEATURE_NAMES:
            payload[f"feature_{name}_flat"] = (
                np.concatenate(self.features[name]).astype(np.float64, copy=False)
                if self.features[name] else np.asarray([], dtype=np.float64)
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=self.path.parent,
            prefix=self.path.name + ".",
            suffix=".tmp",
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                np.savez(stream, **payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, self.path)
        except BaseException:
            try:
                os.close(descriptor)
            except OSError:
                pass
            Path(temporary_name).unlink(missing_ok=True)
            raise
        return self.path


def replay_frozen_rule(
    trace_paths: Iterable[str | Path],
    candidate_rule_path: str | Path,
    *,
    expected_config_hashes: dict[tuple[str, int], str] | None = None,
) -> dict[tuple[str, int], ReplayVerdict]:
    """Return one fail-open exact verdict per trace; load the rule only once."""
    paths = [Path(path) for path in trace_paths]
    started = time.perf_counter()
    candidate_hash = _file_sha256(candidate_rule_path)
    try:
        priority_rule = load_frozen_priority_rule(candidate_rule_path)
    except Exception:
        elapsed = time.perf_counter() - started
        return {
            (path.stem, -1): ReplayVerdict(
                path.stem, -1, False, 0, None,
                status="rule_error",
                candidate_frozen_rule_hash=candidate_hash,
                replay_seconds=elapsed,
            )
            for path in paths
        }

    verdicts: dict[tuple[str, int], ReplayVerdict] = {}
    for path in paths:
        trace_started = time.perf_counter()
        scenario_id = path.stem
        seed = -1
        baseline_hash = ""
        try:
            with np.load(path, allow_pickle=False) as trace:
                scenario_id = str(trace["scenario_id"].item())
                seed = int(trace["seed"].item())
                baseline_hash = str(trace["baseline_frozen_rule_hash"].item())
                if str(trace["schema_version"].item()) != TRACE_SCHEMA_VERSION:
                    raise _TraceMismatch("trace schema mismatch")
                if not bool(trace["complete"].item()):
                    raise _TraceMismatch("incomplete trace")
                if tuple(str(value) for value in trace["feature_names"].tolist()) != FEATURE_NAMES:
                    raise _TraceMismatch("trace feature order mismatch")
                expected_hash = (expected_config_hashes or {}).get(
                    (scenario_id, seed)
                )
                if (
                    expected_hash is not None
                    and str(trace["evaluation_config_sha256"].item())
                    != expected_hash
                ):
                    raise _TraceMismatch("trace evaluation config mismatch")
                decision_count = int(trace["decision_count"].item())
                offsets = np.asarray(trace["ready_offsets"], dtype=np.int64)
                selected = np.asarray(trace["selected_indices"], dtype=np.int64)
                ready_flat = np.asarray(trace["ready_ids_flat"], dtype=np.int64)
                if (
                    len(offsets) != decision_count + 1
                    or len(selected) != decision_count
                    or offsets[0] != 0
                    or offsets[-1] != len(ready_flat)
                    or np.any(np.diff(offsets) <= 0)
                ):
                    raise _TraceMismatch("invalid trace offsets")
                feature_arrays = {
                    name: np.asarray(trace[f"feature_{name}_flat"], dtype=np.float64)
                    for name in FEATURE_NAMES
                }
                if any(
                    len(values) != len(ready_flat) or not np.isfinite(values).all()
                    for values in feature_arrays.values()
                ):
                    raise _TraceMismatch("invalid trace feature array")
                divergence_count = 0
                first_divergence = None
                margins = []
                for decision_index in range(decision_count):
                    start, stop = int(offsets[decision_index]), int(offsets[decision_index + 1])
                    with np.errstate(all="ignore"):
                        scores = priority_rule(
                            *[feature_arrays[name][start:stop] for name in FEATURE_NAMES]
                        )
                    scores = validate_task_priority_scores(scores, stop - start)
                    replay_selected = int(np.argmin(scores))
                    if replay_selected != int(selected[decision_index]):
                        divergence_count += 1
                        if first_divergence is None:
                            first_divergence = decision_index
                        ordered = np.sort(scores)
                        margins.append(
                            float(ordered[1] - ordered[0]) if len(ordered) > 1 else 0.0
                        )
                verdict = ReplayVerdict(
                    scenario_id=scenario_id,
                    seed=seed,
                    identical=divergence_count == 0,
                    divergence_count_along_baseline=divergence_count,
                    first_divergence_index=first_divergence,
                    margin_stats={
                        "minimum": min(margins) if margins else 0.0,
                        "mean": float(np.mean(margins)) if margins else 0.0,
                        "maximum": max(margins) if margins else 0.0,
                    },
                    baseline_frozen_rule_hash=baseline_hash,
                    candidate_frozen_rule_hash=candidate_hash,
                    trace_sha256=_cached_trace_sha256(
                        str(path.resolve()),
                        path.stat().st_size,
                        path.stat().st_mtime_ns,
                    ),
                    replay_seconds=time.perf_counter() - trace_started,
                )
        except Exception as exc:
            status = (
                "trace_mismatch"
                if isinstance(exc, (KeyError, OSError, _TraceMismatch))
                else "rule_error"
            )
            verdict = ReplayVerdict(
                scenario_id=scenario_id,
                seed=seed,
                identical=False,
                divergence_count_along_baseline=0,
                first_divergence_index=None,
                status=status,
                baseline_frozen_rule_hash=baseline_hash,
                candidate_frozen_rule_hash=candidate_hash,
                replay_seconds=time.perf_counter() - trace_started,
            )
        verdicts[(scenario_id, seed)] = verdict
    return verdicts
