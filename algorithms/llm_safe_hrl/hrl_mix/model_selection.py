# -*- coding: utf-8 -*-
"""Feasibility-first validation metrics and checkpoint audit helpers.

The comparison code is deliberately independent from the training runner and
neural-network implementation so it can be unit-tested without PyTorch.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from scenario_registry import validate_protocol_identity
except ModuleNotFoundError:  # Package-style imports used by some test runners.
    from algorithms.llm_safe_hrl.scenario_registry import (
        validate_protocol_identity,
    )

from base.safe_replay import (
    SAFE_REPLAY_BUFFER_SCHEMA_VERSION,
    SAFE_REPLAY_TRANSITION_SCHEMA_VERSION,
)


MODEL_SELECTION_SCHEMA_VERSION = 1
BEST_CHECKPOINT_MANIFEST_SCHEMA_VERSION = 1
CONFIG_SNAPSHOT_SCHEMA_VERSION = 1

_PROTOCOL_IDENTITY_FIELDS = (
    "protocol",
    "source_scenario",
    "training_scenarios",
    "test_scenarios",
    "llm_train_seeds",
    "llm_validation_seeds",
    "safe_hrl_train_seeds",
    "safe_hrl_validation_seeds",
    "final_test_seeds",
)

MODEL_COMPARISON_FIELDS = (
    "deadline_violation_rate",
    "max_fuzzy_lateness",
    "mean_fuzzy_lateness",
    "fuzzy_energy_score",
)


def _finite_non_negative(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _rate(value: Any, name: str) -> float:
    result = _finite_non_negative(value, name)
    if result > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


@dataclass(frozen=True)
class FeasibilityFirstModelMetrics:
    """Validation metrics used by the lexicographic model comparator."""

    deadline_violation_rate: float
    max_fuzzy_lateness: float
    mean_fuzzy_lateness: float
    fuzzy_energy_score: float
    all_seed_feasible: bool
    feasible_seed_rate: float
    worst_seed_violation: float
    worst_seed_lateness: float
    validation_seed_count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "deadline_violation_rate",
            _rate(
                self.deadline_violation_rate,
                "deadline_violation_rate",
            ),
        )
        object.__setattr__(
            self,
            "max_fuzzy_lateness",
            _finite_non_negative(
                self.max_fuzzy_lateness,
                "max_fuzzy_lateness",
            ),
        )
        object.__setattr__(
            self,
            "mean_fuzzy_lateness",
            _finite_non_negative(
                self.mean_fuzzy_lateness,
                "mean_fuzzy_lateness",
            ),
        )
        object.__setattr__(
            self,
            "fuzzy_energy_score",
            _finite_non_negative(
                self.fuzzy_energy_score,
                "fuzzy_energy_score",
            ),
        )
        object.__setattr__(
            self,
            "feasible_seed_rate",
            _rate(
                self.feasible_seed_rate,
                "feasible_seed_rate",
            ),
        )
        object.__setattr__(
            self,
            "worst_seed_violation",
            _rate(
                self.worst_seed_violation,
                "worst_seed_violation",
            ),
        )
        object.__setattr__(
            self,
            "worst_seed_lateness",
            _finite_non_negative(
                self.worst_seed_lateness,
                "worst_seed_lateness",
            ),
        )
        seed_count = int(self.validation_seed_count)
        if seed_count <= 0:
            raise ValueError(
                "validation_seed_count must be positive"
            )
        object.__setattr__(
            self,
            "validation_seed_count",
            seed_count,
        )

    @property
    def comparison_key(self) -> tuple[float, float, float, float]:
        """Exact feasibility-first key requested by the experiment protocol."""
        return (
            self.deadline_violation_rate,
            self.max_fuzzy_lateness,
            self.mean_fuzzy_lateness,
            self.fuzzy_energy_score,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_selection_schema_version": (
                MODEL_SELECTION_SCHEMA_VERSION
            ),
            **asdict(self),
            "comparison_key_fields": list(
                MODEL_COMPARISON_FIELDS
            ),
            "comparison_key": list(self.comparison_key),
        }

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
    ) -> "FeasibilityFirstModelMetrics":
        if not isinstance(values, Mapping):
            raise ValueError(
                "model selection metrics must be a mapping"
            )
        return cls(
            deadline_violation_rate=values[
                "deadline_violation_rate"
            ],
            max_fuzzy_lateness=values[
                "max_fuzzy_lateness"
            ],
            mean_fuzzy_lateness=values[
                "mean_fuzzy_lateness"
            ],
            fuzzy_energy_score=values["fuzzy_energy_score"],
            all_seed_feasible=bool(
                values["all_seed_feasible"]
            ),
            feasible_seed_rate=values["feasible_seed_rate"],
            worst_seed_violation=values[
                "worst_seed_violation"
            ],
            worst_seed_lateness=values[
                "worst_seed_lateness"
            ],
            validation_seed_count=values[
                "validation_seed_count"
            ],
        )


def is_better_model(
    candidate: FeasibilityFirstModelMetrics,
    incumbent: FeasibilityFirstModelMetrics | None,
) -> bool:
    """Return whether candidate strictly improves the lexicographic key."""
    if incumbent is None:
        return True
    return candidate.comparison_key < incumbent.comparison_key


def aggregate_seed_feasibility_metrics(
    seed_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate completed-workflow safety metrics across validation seeds.

    ``worst_seed_violation`` is explicitly the largest per-seed violation
    *rate*. ``worst_seed_lateness`` is the largest per-seed maximum fuzzy
    lateness. Mean fuzzy lateness includes zero-lateness completed workflows.
    """
    if not seed_records:
        raise ValueError(
            "feasibility aggregation requires validation seeds"
        )

    normalized = []
    for record in seed_records:
        seed = int(record["seed"])
        violation_count = int(record["deadline_violation_count"])
        completed_count = int(record["completed_workflow_count"])
        if violation_count < 0 or completed_count < 0:
            raise ValueError(
                "workflow safety counts must be non-negative"
            )
        if violation_count > completed_count:
            raise ValueError(
                "deadline violations cannot exceed completed workflows"
            )
        lateness_sum = _finite_non_negative(
            record["fuzzy_lateness_sum"],
            "fuzzy_lateness_sum",
        )
        max_lateness = _finite_non_negative(
            record["max_fuzzy_lateness"],
            "max_fuzzy_lateness",
        )
        fuzzy_energy_score = _finite_non_negative(
            record["fuzzy_energy_score"],
            "fuzzy_energy_score",
        )
        evaluation_completed = bool(
            record.get("evaluation_completed", True)
        )
        violation_rate = float(
            violation_count / max(completed_count, 1)
        )
        feasible = bool(
            evaluation_completed
            and completed_count > 0
            and violation_count == 0
            and max_lateness == 0.0
        )
        normalized.append(
            {
                **dict(record),
                "seed": seed,
                "deadline_violation_count": violation_count,
                "completed_workflow_count": completed_count,
                "deadline_violation_rate": violation_rate,
                "fuzzy_lateness_sum": lateness_sum,
                "max_fuzzy_lateness": max_lateness,
                "mean_fuzzy_lateness": float(
                    lateness_sum / max(completed_count, 1)
                ),
                "fuzzy_energy_score": fuzzy_energy_score,
                "evaluation_completed": evaluation_completed,
                "seed_feasible": feasible,
            }
        )

    total_violations = int(
        sum(
            item["deadline_violation_count"]
            for item in normalized
        )
    )
    total_completed = int(
        sum(
            item["completed_workflow_count"]
            for item in normalized
        )
    )
    total_lateness = float(
        sum(item["fuzzy_lateness_sum"] for item in normalized)
    )
    feasible_count = int(
        sum(bool(item["seed_feasible"]) for item in normalized)
    )
    completed_seed_count = int(
        sum(
            bool(item["evaluation_completed"])
            for item in normalized
        )
    )
    return {
        "deadline_violation_rate": float(
            total_violations / max(total_completed, 1)
        ),
        "max_fuzzy_lateness": float(
            max(item["max_fuzzy_lateness"] for item in normalized)
        ),
        "mean_fuzzy_lateness": float(
            total_lateness / max(total_completed, 1)
        ),
        "fuzzy_energy_score": float(
            sum(
                item["fuzzy_energy_score"]
                for item in normalized
            )
            / len(normalized)
        ),
        "all_seed_feasible": bool(
            feasible_count == len(normalized)
        ),
        "all_seed_evaluation_completed": bool(
            completed_seed_count == len(normalized)
        ),
        "completed_evaluation_seed_rate": float(
            completed_seed_count / len(normalized)
        ),
        "feasible_seed_rate": float(
            feasible_count / len(normalized)
        ),
        "worst_seed_violation": float(
            max(
                item["deadline_violation_rate"]
                for item in normalized
            )
        ),
        "worst_seed_lateness": float(
            max(item["max_fuzzy_lateness"] for item in normalized)
        ),
        "validation_seed_count": int(len(normalized)),
        "per_seed_safety_metrics": normalized,
    }


def build_replay_metadata(agent: Any) -> dict[str, Any]:
    """Build replay audit metadata without embedding replay transitions."""
    safe_enabled = bool(
        getattr(agent, "safe_rl_enabled", False)
    )
    buffer = getattr(agent, "buffer", ())
    category_counts: Counter[str] = Counter()
    if safe_enabled:
        for transition in buffer:
            category = getattr(transition, "risk_category", None)
            if category is not None:
                category_counts[str(category)] += 1
    return {
        "metadata_schema_version": 1,
        "safe_rl_enabled": safe_enabled,
        "replay_buffer_schema_version": (
            SAFE_REPLAY_BUFFER_SCHEMA_VERSION
            if safe_enabled
            else "legacy_tuple_v1"
        ),
        "transition_schema_version": (
            SAFE_REPLAY_TRANSITION_SCHEMA_VERSION
            if safe_enabled
            else "legacy_tuple_v1"
        ),
        "transition_count": int(len(buffer)),
        "capacity": int(getattr(agent, "buffer_size", 0)),
        "input_dim": int(getattr(agent, "input_dim", 0)),
        "action_dim": int(getattr(agent, "output_dim", 0)),
        "use_per": bool(getattr(agent, "use_per", False)),
        "priority_count": int(
            len(getattr(agent, "priorities", ()))
        ),
        "risk_category_counts": dict(
            sorted(category_counts.items())
        ),
        "replay_transitions_embedded": False,
    }


def build_config_snapshot(config: Any) -> dict[str, Any]:
    """Create a canonical, hash-bound JSON configuration snapshot."""
    if is_dataclass(config):
        values = asdict(config)
    elif isinstance(config, Mapping):
        values = dict(config)
    else:
        raise ValueError(
            "config snapshot source must be a dataclass or mapping"
        )
    canonical = json.dumps(
        values,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "config_snapshot_schema_version": (
            CONFIG_SNAPSHOT_SCHEMA_VERSION
        ),
        "sha256": hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest(),
        "config": values,
    }


def protocol_identity_from_config_snapshot(
    config_snapshot: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Extract a complete protocol identity from a config snapshot."""
    if not isinstance(config_snapshot, Mapping):
        raise ValueError("config_snapshot must be a mapping")
    config = config_snapshot.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("config_snapshot config must be a mapping")
    nested = config.get("experiment_protocol")
    present = [field for field in _PROTOCOL_IDENTITY_FIELDS if field in config]
    # A canonical nested identity takes precedence over legacy flat fields.
    # TrainConfig snapshots intentionally keep runtime fields such as
    # protocol/source_scenario/training_scenarios at the top level, so those
    # partial overlaps are not themselves a second protocol identity.
    if nested is not None:
        return validate_protocol_identity(
            nested,
            nested,
            artifact_name="config snapshot protocol",
        )
    if nested is None and str(config.get("protocol", "")).strip().lower() == "legacy":
        return None
    if nested is None and "protocol" not in config:
        return None
    if len(present) != len(_PROTOCOL_IDENTITY_FIELDS):
        raise ValueError("config snapshot has incomplete protocol identity")
    identity = {field: config[field] for field in _PROTOCOL_IDENTITY_FIELDS}
    return validate_protocol_identity(
        identity,
        identity,
        artifact_name="config snapshot protocol",
    )


def _resolve_checkpoint_protocol_identity(
    config_snapshot: Mapping[str, Any],
    experiment_protocol,
    *,
    artifact_name: str,
) -> dict[str, Any] | None:
    snapshot_identity = protocol_identity_from_config_snapshot(config_snapshot)
    explicit_identity = None
    if experiment_protocol is not None:
        explicit_identity = validate_protocol_identity(
            experiment_protocol,
            experiment_protocol,
            artifact_name=artifact_name,
        )
    if snapshot_identity is not None and explicit_identity is not None:
        validate_protocol_identity(
            explicit_identity,
            snapshot_identity,
            artifact_name="config snapshot protocol",
        )
    return explicit_identity or snapshot_identity


def build_heuristic_library_version(
    env: Any,
    *,
    manifest_path: str | os.PathLike[str] | None,
) -> dict[str, Any]:
    """Read version metadata only; this helper never imports rule code."""
    manager_mode = str(
        getattr(
            env,
            "manager_mode",
            "legacy_rule_weight_mode",
        )
    )
    result = {
        "metadata_schema_version": 1,
        "manager_mode": manager_mode,
        "manager_action_schema_version": getattr(
            env,
            "manager_heuristic_schema_version",
            None,
        ),
        "manifest_path": None,
        "manifest_sha256": None,
        "manifest_schema_version": None,
        "manifest_id": None,
        "manifest_version": (
            "legacy_v1"
            if manager_mode == "legacy_rule_weight_mode"
            else None
        ),
        "manifest_revision": None,
        "experiment_protocol": None,
        "admitted_heuristic_ids": [],
    }
    if manager_mode == "legacy_rule_weight_mode":
        return result
    if not manifest_path:
        raise ValueError(
            "heuristic selection checkpoint requires a manifest path"
        )
    source = Path(manifest_path).resolve()
    raw = source.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(
            "heuristic library manifest must be a mapping"
        )
    admitted_ids = [
        str(getattr(item, "heuristic_id"))
        for item in getattr(env, "manager_heuristics", ())
        if bool(getattr(item, "admitted", True))
    ]
    result.update(
        {
            "manifest_path": str(source),
            "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "manifest_schema_version": payload.get(
                "schema_version"
            ),
            "manifest_id": payload.get("manifest_id"),
            "manifest_version": payload.get(
                "manifest_version"
            ),
            "manifest_revision": payload.get(
                "manifest_revision"
            ),
            "experiment_protocol": (
                None
                if payload.get("experiment_protocol") is None
                else validate_protocol_identity(
                    payload["experiment_protocol"],
                    payload["experiment_protocol"],
                    artifact_name="heuristic library manifest",
                )
            ),
            "admitted_heuristic_ids": admitted_ids,
        }
    )
    return result


def save_best_checkpoint_bundle(
    directory: str | os.PathLike[str],
    *,
    agents: Mapping[str, Any],
    lagrange_controller: Any,
    model_metrics: FeasibilityFirstModelMetrics,
    curriculum_state: Mapping[str, Any],
    replay_metadata: Mapping[str, Any],
    heuristic_library_version: Mapping[str, Any],
    config_snapshot: Mapping[str, Any],
    experiment_protocol=None,
) -> str:
    """Atomically bind three Agent files to one feasibility-first manifest."""
    target_dir = Path(directory).resolve()
    expected_layers = {"manager", "host", "vm"}
    if set(agents) != expected_layers:
        raise ValueError(
            "best checkpoint bundle requires manager/host/vm agents"
        )
    if set(replay_metadata) != expected_layers:
        raise ValueError(
            "best checkpoint replay metadata layer mismatch"
        )
    if not all(
        bool(getattr(agent, "safe_rl_enabled", False))
        for agent in agents.values()
    ):
        raise ValueError(
            "feasibility-first bundle requires safe_rl agents"
        )

    lagrange_state = (
        lagrange_controller.state_dict()
        if lagrange_controller is not None
        else None
    )
    protocol_identity = _resolve_checkpoint_protocol_identity(
        config_snapshot,
        experiment_protocol,
        artifact_name="best checkpoint protocol",
    )
    library_protocol = heuristic_library_version.get(
        "experiment_protocol"
    )
    if protocol_identity is not None and library_protocol is not None:
        validate_protocol_identity(
            protocol_identity,
            library_protocol,
            artifact_name="checkpoint heuristic library",
        )
    config = config_snapshot["config"]
    optimizer_seed = int(config["optimizer_seed"])
    deadline_cache_paths = dict(config.get("deadline_cache_paths", {}))

    target_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_files = {}
    for layer in sorted(expected_layers):
        filename = (
            "best_manager.pth"
            if layer == "manager"
            else f"best_{layer}.pth"
        )
        agents[layer].save(
            str(target_dir / filename),
            lagrange_controller_state=lagrange_state,
        )
        checkpoint_files[layer] = filename

    payload = {
        "checkpoint_manifest_schema_version": (
            BEST_CHECKPOINT_MANIFEST_SCHEMA_VERSION
        ),
        "selection_policy": "feasibility_first_lexicographic",
        "optimizer_seed": optimizer_seed,
        "deadline_cache_paths": deadline_cache_paths,
        "model_selection_metrics": model_metrics.to_dict(),
        "agent_checkpoints": checkpoint_files,
        "agent_checkpoint_contents": {
            layer: {
                "q_r": {
                    "online_key": "online",
                    "target_key": "target",
                    "optimizer_key": "optim",
                },
                "q_c": {
                    "online_key": "q_c_online",
                    "target_key": "q_c_target",
                    "optimizer_key": "q_c_optim",
                },
                "lagrange_multiplier_key": (
                    "lagrange_multiplier"
                ),
            }
            for layer in sorted(expected_layers)
        },
        "lagrange_controller_state": lagrange_state,
        "curriculum_state": dict(curriculum_state),
        "replay_metadata": dict(replay_metadata),
        "heuristic_library_version": dict(
            heuristic_library_version
        ),
        "config_snapshot": dict(config_snapshot),
    }
    if protocol_identity is not None:
        payload["experiment_protocol"] = protocol_identity
    manifest_path = target_dir / "best_checkpoint_manifest.json"
    temporary_path = target_dir / "best_checkpoint_manifest.json.tmp"
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        handle.write("\n")
    os.replace(temporary_path, manifest_path)
    return str(manifest_path)


def read_best_checkpoint_manifest(
    path: str | os.PathLike[str],
    *,
    expected_protocol_identity=None,
) -> dict[str, Any]:
    """Read a best bundle and optionally reject cross-protocol loading."""
    source = Path(path).resolve()
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("best checkpoint manifest must be a mapping")
    payload = dict(payload)
    if (
        int(payload.get("checkpoint_manifest_schema_version", -1))
        != BEST_CHECKPOINT_MANIFEST_SCHEMA_VERSION
    ):
        raise ValueError("unsupported best checkpoint manifest schema")
    checkpoints = payload.get("agent_checkpoints")
    if not isinstance(checkpoints, Mapping) or set(checkpoints) != {
        "manager",
        "host",
        "vm",
    }:
        raise ValueError("best checkpoint manifest layer set mismatch")
    actual_identity = payload.get("experiment_protocol")
    if expected_protocol_identity is not None:
        if not isinstance(actual_identity, Mapping):
            raise ValueError(
                "best checkpoint manifest is missing experiment_protocol"
            )
        payload["experiment_protocol"] = validate_protocol_identity(
            expected_protocol_identity,
            actual_identity,
            artifact_name="best checkpoint manifest",
        )
    elif payload.get("experiment_protocol") is not None:
        identity = payload["experiment_protocol"]
        payload["experiment_protocol"] = validate_protocol_identity(
            identity,
            identity,
            artifact_name="best checkpoint manifest",
        )
    snapshot = payload.get("config_snapshot")
    if not isinstance(snapshot, Mapping):
        raise ValueError("best checkpoint config_snapshot must be a mapping")
    snapshot_identity = protocol_identity_from_config_snapshot(snapshot)
    normalized_identity = payload.get("experiment_protocol")
    if normalized_identity is not None and snapshot_identity is not None:
        validate_protocol_identity(
            normalized_identity,
            snapshot_identity,
            artifact_name="best checkpoint config snapshot",
        )
    library = payload.get("heuristic_library_version")
    if not isinstance(library, Mapping):
        raise ValueError(
            "best checkpoint heuristic_library_version must be a mapping"
        )
    library_identity = library.get("experiment_protocol")
    if normalized_identity is not None and library_identity is not None:
        validate_protocol_identity(
            normalized_identity,
            library_identity,
            artifact_name="best checkpoint heuristic library",
        )
    payload["resolved_agent_checkpoints"] = {
        layer: str((source.parent / filename).resolve())
        for layer, filename in checkpoints.items()
    }
    return payload


__all__ = [
    "BEST_CHECKPOINT_MANIFEST_SCHEMA_VERSION",
    "CONFIG_SNAPSHOT_SCHEMA_VERSION",
    "FeasibilityFirstModelMetrics",
    "MODEL_COMPARISON_FIELDS",
    "MODEL_SELECTION_SCHEMA_VERSION",
    "aggregate_seed_feasibility_metrics",
    "build_config_snapshot",
    "build_heuristic_library_version",
    "build_replay_metadata",
    "is_better_model",
    "protocol_identity_from_config_snapshot",
    "read_best_checkpoint_manifest",
    "save_best_checkpoint_bundle",
]
