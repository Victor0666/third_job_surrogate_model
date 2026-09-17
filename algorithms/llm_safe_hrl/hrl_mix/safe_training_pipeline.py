# -*- coding: utf-8 -*-
"""Versioned orchestration for staged safe-HRL training.

This module deliberately contains no scheduling policy, reward, safety-cost,
shield, or neural-network logic.  It validates a five-stage training plan,
keeps train/validation/final-test seeds disjoint, advances online curriculum
stages from validation metrics only, and checkpoints the orchestration state.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from scenario_registry import validate_protocol_identity
except ModuleNotFoundError:  # Package-style imports used by some test runners.
    from algorithms.llm_safe_hrl.scenario_registry import (
        validate_protocol_identity,
    )

from hrl_mix.model_selection import protocol_identity_from_config_snapshot


SAFE_TRAINING_PLAN_SCHEMA_VERSION = 1
SAFE_TRAINING_STATE_SCHEMA_VERSION = 1
SAFE_TRAINING_CHECKPOINT_SCHEMA_VERSION = 2

PREPARATION_STAGE_TYPES = (
    "demonstration_generation",
    "offline_pretraining",
)
ONLINE_STAGE_TYPES = (
    "shield_online_training",
    "curriculum_training",
    "cross_seed_robust_training",
)
REQUIRED_STAGE_TYPES = (
    "demonstration_generation",
    "offline_pretraining",
    "shield_online_training",
    "curriculum_training",
    "cross_seed_robust_training",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _finite_float(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return number


def _positive_int(value: Any, name: str) -> int:
    number = int(value)
    if number <= 0:
        raise ValueError(f"{name} must be positive")
    return number


def _resolve_path(base_dir: Path, value: Any) -> str | None:
    if value in (None, ""):
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())


@dataclass(frozen=True)
class StrictTrainingSeedSplit:
    """Disjoint seeds for fitting, curriculum validation, and final test."""

    training: tuple[int, ...]
    validation: tuple[int, ...]
    final_test: tuple[int, ...]

    def __post_init__(self) -> None:
        groups = {
            "training": tuple(int(v) for v in self.training),
            "validation": tuple(int(v) for v in self.validation),
            "final_test": tuple(int(v) for v in self.final_test),
        }
        for name, values in groups.items():
            if not values:
                raise ValueError(f"{name} seeds must not be empty")
            if len(values) != len(set(values)):
                raise ValueError(
                    f"{name} seeds contain duplicates"
                )
            if any(value < 0 for value in values):
                raise ValueError(
                    f"{name} seeds must be non-negative"
                )
        names = tuple(groups)
        for index, left_name in enumerate(names):
            for right_name in names[index + 1 :]:
                overlap = set(groups[left_name]).intersection(
                    groups[right_name]
                )
                if overlap:
                    raise ValueError(
                        "safe training seed leakage between "
                        f"{left_name} and {right_name}: "
                        f"{sorted(overlap)}"
                    )

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "StrictTrainingSeedSplit":
        payload = _require_mapping(payload, "seed_split")
        return cls(
            training=tuple(payload.get("training", ())),
            validation=tuple(payload.get("validation", ())),
            final_test=tuple(payload.get("final_test", ())),
        )

    def to_dict(self) -> dict[str, list[int]]:
        return {
            "training": list(self.training),
            "validation": list(self.validation),
            "final_test": list(self.final_test),
        }


@dataclass(frozen=True)
class CurriculumProfile:
    """Environment profile that preserves Host/VM topology dimensions."""

    ddl_level: str
    deadline_alpha_small: float
    deadline_alpha_large: float
    deadline_alpha_small_prob: float
    arrival_level: str
    arrival_lambda: float
    uncertainty_level: str
    fuzzy_delta1: float
    fuzzy_delta2: float
    resource_scale: str
    resource_capacity_scale: float
    resource_topology: str
    workflow_scale: str
    workflows_per_episode: int

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        stage_id: str,
    ) -> "CurriculumProfile":
        payload = _require_mapping(
            payload,
            f"{stage_id}.curriculum",
        )
        deadline = _require_mapping(
            payload.get("deadline"),
            f"{stage_id}.curriculum.deadline",
        )
        arrival = _require_mapping(
            payload.get("arrival"),
            f"{stage_id}.curriculum.arrival",
        )
        uncertainty = _require_mapping(
            payload.get("uncertainty"),
            f"{stage_id}.curriculum.uncertainty",
        )
        resource = _require_mapping(
            payload.get("resource"),
            f"{stage_id}.curriculum.resource",
        )
        workflow = _require_mapping(
            payload.get("workflow"),
            f"{stage_id}.curriculum.workflow",
        )
        ddl_level = str(deadline.get("level", "")).strip().lower()
        arrival_level = str(
            arrival.get("level", "")
        ).strip().lower()
        uncertainty_level = str(
            uncertainty.get("level", "")
        ).strip().lower()
        expected_levels = {
            "DDL": (ddl_level, {"loose", "medium", "tight"}),
            "arrival": (
                arrival_level,
                {"low", "medium", "high"},
            ),
            "uncertainty": (
                uncertainty_level,
                {"low", "medium", "high"},
            ),
        }
        for name, (value, allowed) in expected_levels.items():
            if value not in allowed:
                raise ValueError(
                    f"{stage_id} {name} level must be one of "
                    f"{sorted(allowed)}"
                )
        topology = str(
            resource.get("topology", "")
        ).strip().lower()
        if topology != "inherit":
            raise ValueError(
                f"{stage_id} resource.topology must be 'inherit'. "
                "Changing Host/VM counts changes D3QN action dimensions "
                "and requires a separate training plan/checkpoint."
            )
        alpha_small = _finite_float(
            deadline.get("alpha_small"),
            f"{stage_id}.deadline.alpha_small",
            minimum=0.0,
        )
        alpha_large = _finite_float(
            deadline.get("alpha_large"),
            f"{stage_id}.deadline.alpha_large",
            minimum=0.0,
        )
        if alpha_small > alpha_large:
            raise ValueError(
                f"{stage_id} deadline alpha_small must not exceed "
                "alpha_large"
            )
        delta1 = _finite_float(
            uncertainty.get("fuzzy_delta1"),
            f"{stage_id}.uncertainty.fuzzy_delta1",
            minimum=0.0,
        )
        delta2 = _finite_float(
            uncertainty.get("fuzzy_delta2"),
            f"{stage_id}.uncertainty.fuzzy_delta2",
            minimum=0.0,
        )
        if not delta1 <= 1.0 <= delta2:
            raise ValueError(
                f"{stage_id} fuzzy uncertainty must satisfy "
                "delta1 <= 1 <= delta2"
            )
        resource_scale = str(
            resource.get("scale", "")
        ).strip()
        workflow_scale = str(
            workflow.get("scale", "")
        ).strip()
        if not resource_scale or not workflow_scale:
            raise ValueError(
                f"{stage_id} resource/workflow scale labels must be "
                "explicit"
            )
        return cls(
            ddl_level=ddl_level,
            deadline_alpha_small=alpha_small,
            deadline_alpha_large=alpha_large,
            deadline_alpha_small_prob=_finite_float(
                deadline.get("alpha_small_probability"),
                f"{stage_id}.deadline.alpha_small_probability",
                minimum=0.0,
                maximum=1.0,
            ),
            arrival_level=arrival_level,
            arrival_lambda=_finite_float(
                arrival.get("poisson_lambda"),
                f"{stage_id}.arrival.poisson_lambda",
                minimum=1e-12,
            ),
            uncertainty_level=uncertainty_level,
            fuzzy_delta1=delta1,
            fuzzy_delta2=delta2,
            resource_scale=resource_scale,
            resource_capacity_scale=_finite_float(
                resource.get("capacity_scale"),
                f"{stage_id}.resource.capacity_scale",
                minimum=1e-12,
            ),
            resource_topology=topology,
            workflow_scale=workflow_scale,
            workflows_per_episode=_positive_int(
                workflow.get("workflows_per_episode"),
                f"{stage_id}.workflow.workflows_per_episode",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "deadline": {
                "level": self.ddl_level,
                "alpha_small": self.deadline_alpha_small,
                "alpha_large": self.deadline_alpha_large,
                "alpha_small_probability": (
                    self.deadline_alpha_small_prob
                ),
            },
            "arrival": {
                "level": self.arrival_level,
                "poisson_lambda": self.arrival_lambda,
            },
            "uncertainty": {
                "level": self.uncertainty_level,
                "fuzzy_delta1": self.fuzzy_delta1,
                "fuzzy_delta2": self.fuzzy_delta2,
            },
            "resource": {
                "scale": self.resource_scale,
                "capacity_scale": self.resource_capacity_scale,
                "topology": self.resource_topology,
            },
            "workflow": {
                "scale": self.workflow_scale,
                "workflows_per_episode": (
                    self.workflows_per_episode
                ),
            },
        }


@dataclass(frozen=True)
class StageTransition:
    mode: str
    fixed_episodes: int | None = None
    minimum_episodes: int = 0
    consecutive_evaluations: int = 1
    thresholds: Mapping[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        stage_id: str,
    ) -> "StageTransition":
        payload = _require_mapping(
            payload,
            f"{stage_id}.transition",
        )
        mode = str(payload.get("mode", "")).strip()
        if mode == "fixed_episodes":
            return cls(
                mode=mode,
                fixed_episodes=_positive_int(
                    payload.get("episodes"),
                    f"{stage_id}.transition.episodes",
                ),
            )
        if mode != "validation_threshold":
            raise ValueError(
                f"{stage_id} transition mode must be "
                "fixed_episodes or validation_threshold"
            )
        thresholds_payload = _require_mapping(
            payload.get("thresholds"),
            f"{stage_id}.transition.thresholds",
        )
        allowed = {
            "max_fuzzy_energy_score",
            "max_safety_cost",
            "max_violation_rate",
            "max_shield_intervention_rate",
            "max_fallback_rate",
            "max_q_c_prediction_error",
        }
        unknown = set(thresholds_payload).difference(allowed)
        if unknown:
            raise ValueError(
                f"{stage_id} has unsupported validation thresholds: "
                f"{sorted(unknown)}"
            )
        if not thresholds_payload:
            raise ValueError(
                f"{stage_id} validation thresholds must not be empty"
            )
        thresholds = {
            key: _finite_float(
                value,
                f"{stage_id}.transition.thresholds.{key}",
                minimum=0.0,
            )
            for key, value in thresholds_payload.items()
        }
        return cls(
            mode=mode,
            minimum_episodes=_positive_int(
                payload.get("minimum_episodes", 1),
                f"{stage_id}.transition.minimum_episodes",
            ),
            consecutive_evaluations=_positive_int(
                payload.get("consecutive_evaluations"),
                f"{stage_id}.transition.consecutive_evaluations",
            ),
            thresholds=thresholds,
        )

    def to_dict(self) -> dict[str, Any]:
        if self.mode == "fixed_episodes":
            return {
                "mode": self.mode,
                "episodes": self.fixed_episodes,
            }
        return {
            "mode": self.mode,
            "minimum_episodes": self.minimum_episodes,
            "consecutive_evaluations": (
                self.consecutive_evaluations
            ),
            "thresholds": dict(self.thresholds),
        }


@dataclass(frozen=True)
class OnlineTrainingStage:
    stage_id: str
    stage_type: str
    training_seed_mode: str
    curriculum: CurriculumProfile
    transition: StageTransition

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "OnlineTrainingStage":
        payload = _require_mapping(payload, "online stage")
        stage_id = str(payload.get("stage_id", "")).strip()
        stage_type = str(payload.get("stage_type", "")).strip()
        if not stage_id:
            raise ValueError("online stage_id must not be empty")
        if stage_type not in ONLINE_STAGE_TYPES:
            raise ValueError(
                f"{stage_id} has unsupported online stage_type "
                f"{stage_type!r}"
            )
        training_seed_mode = str(
            payload.get("training_seed_mode", "")
        ).strip()
        if training_seed_mode not in {
            "primary",
            "round_robin",
        }:
            raise ValueError(
                f"{stage_id} training_seed_mode must be primary or "
                "round_robin"
            )
        if (
            stage_type == "cross_seed_robust_training"
            and training_seed_mode != "round_robin"
        ):
            raise ValueError(
                f"{stage_id} cross-seed training must use "
                "training_seed_mode=round_robin"
            )
        return cls(
            stage_id=stage_id,
            stage_type=stage_type,
            training_seed_mode=training_seed_mode,
            curriculum=CurriculumProfile.from_dict(
                payload.get("curriculum"),
                stage_id=stage_id,
            ),
            transition=StageTransition.from_dict(
                payload.get("transition"),
                stage_id=stage_id,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "stage_type": self.stage_type,
            "training_seed_mode": self.training_seed_mode,
            "curriculum": self.curriculum.to_dict(),
            "transition": self.transition.to_dict(),
        }


@dataclass(frozen=True)
class SafeTrainingPlan:
    pipeline_id: str
    source_path: str
    plan_hash: str
    seed_split: StrictTrainingSeedSplit
    preparation_stages: tuple[Mapping[str, Any], ...]
    online_stages: tuple[OnlineTrainingStage, ...]
    checkpoint_interval_episodes: int
    metrics_path: str

    @property
    def offline_pretraining(self) -> Mapping[str, Any]:
        for stage in self.preparation_stages:
            if stage["stage_type"] == "offline_pretraining":
                return stage["offline_pretraining"]
        raise RuntimeError("offline_pretraining stage is missing")

    @property
    def demonstration_manifest_path(self) -> str:
        for stage in self.preparation_stages:
            if stage["stage_type"] == "demonstration_generation":
                return str(stage["artifact_manifest_path"])
        raise RuntimeError("demonstration_generation stage is missing")


def load_safe_training_plan(path: str | os.PathLike[str]) -> SafeTrainingPlan:
    source = Path(path).resolve()
    with source.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    raw = _require_mapping(raw, "safe training plan")
    if (
        int(raw.get("schema_version", -1))
        != SAFE_TRAINING_PLAN_SCHEMA_VERSION
    ):
        raise ValueError("unsupported safe training plan schema version")
    pipeline_id = str(raw.get("pipeline_id", "")).strip()
    if not pipeline_id:
        raise ValueError("safe training pipeline_id must not be empty")
    base_dir = source.parent

    preparation_raw = raw.get("preparation_stages")
    if not isinstance(preparation_raw, list):
        raise ValueError("preparation_stages must be a list")
    preparation: list[dict[str, Any]] = []
    preparation_types: list[str] = []
    for entry in preparation_raw:
        entry = dict(_require_mapping(entry, "preparation stage"))
        stage_id = str(entry.get("stage_id", "")).strip()
        stage_type = str(entry.get("stage_type", "")).strip()
        if not stage_id:
            raise ValueError("preparation stage_id must not be empty")
        if stage_type not in PREPARATION_STAGE_TYPES:
            raise ValueError(
                f"{stage_id} has unsupported preparation stage type"
            )
        preparation_types.append(stage_type)
        normalized = {
            "stage_id": stage_id,
            "stage_type": stage_type,
        }
        if stage_type == "demonstration_generation":
            manifest_path = _resolve_path(
                base_dir,
                entry.get("artifact_manifest_path"),
            )
            if not manifest_path:
                raise ValueError(
                    f"{stage_id} requires artifact_manifest_path"
                )
            normalized.update(
                {
                    "execution": str(
                        entry.get("execution", "external")
                    ),
                    "artifact_manifest_path": manifest_path,
                    "require_completed_artifact": bool(
                        entry.get(
                            "require_completed_artifact",
                            True,
                        )
                    ),
                }
            )
        else:
            offline = dict(
                _require_mapping(
                    entry.get("offline_pretraining"),
                    f"{stage_id}.offline_pretraining",
                )
            )
            dataset_path = _resolve_path(
                base_dir,
                offline.get("dataset_manifest_path"),
            )
            if not dataset_path:
                raise ValueError(
                    f"{stage_id} requires dataset_manifest_path"
                )
            offline["dataset_manifest_path"] = dataset_path
            offline["epochs"] = _positive_int(
                offline.get("epochs"),
                f"{stage_id}.offline_pretraining.epochs",
            )
            offline["batch_size"] = _positive_int(
                offline.get("batch_size"),
                f"{stage_id}.offline_pretraining.batch_size",
            )
            normalized["offline_pretraining"] = offline
        preparation.append(normalized)

    online_raw = raw.get("online_stages")
    if not isinstance(online_raw, list) or not online_raw:
        raise ValueError("online_stages must be a non-empty list")
    online = tuple(
        OnlineTrainingStage.from_dict(entry)
        for entry in online_raw
    )
    if any(stage.transition.mode != "fixed_episodes" for stage in online):
        raise ValueError(
            "every online curriculum stage must use fixed_episodes; "
            "validation cannot control stage duration"
        )
    online_episode_budget = sum(
        int(stage.transition.fixed_episodes) for stage in online
    )
    if online_episode_budget != 600:
        raise ValueError(
            "Safe-HRL online curriculum stages must total exactly 600 "
            f"episodes, got {online_episode_budget}"
        )
    all_ids = [
        stage["stage_id"] for stage in preparation
    ] + [stage.stage_id for stage in online]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("safe training stage_id values must be unique")
    all_types = preparation_types + [
        stage.stage_type for stage in online
    ]
    missing_types = [
        stage_type
        for stage_type in REQUIRED_STAGE_TYPES
        if stage_type not in all_types
    ]
    if missing_types:
        raise ValueError(
            "safe training plan is missing required stage types: "
            f"{missing_types}"
        )
    type_rank = {
        stage_type: index
        for index, stage_type in enumerate(REQUIRED_STAGE_TYPES)
    }
    type_sequence = [type_rank[value] for value in all_types]
    if type_sequence != sorted(type_sequence):
        raise ValueError(
            "safe training stages must follow demonstration, offline, "
            "shield-online, curriculum, cross-seed order"
        )

    demonstration_path = next(
        stage["artifact_manifest_path"]
        for stage in preparation
        if stage["stage_type"] == "demonstration_generation"
    )
    offline_path = next(
        stage["offline_pretraining"]["dataset_manifest_path"]
        for stage in preparation
        if stage["stage_type"] == "offline_pretraining"
    )
    if Path(demonstration_path) != Path(offline_path):
        raise ValueError(
            "stage-1 demonstration manifest and stage-2 offline "
            "dataset manifest must match"
        )

    metrics_path = _resolve_path(
        base_dir,
        raw.get("metrics_path", "safe_training_stage_metrics.jsonl"),
    )
    normalized_for_hash = {
        "schema_version": SAFE_TRAINING_PLAN_SCHEMA_VERSION,
        "pipeline_id": pipeline_id,
        "seed_split": StrictTrainingSeedSplit.from_dict(
            raw.get("seed_split")
        ).to_dict(),
        "preparation_stages": preparation,
        "online_stages": [
            stage.to_dict() for stage in online
        ],
        "checkpoint_interval_episodes": _positive_int(
            raw.get("checkpoint_interval_episodes", 1),
            "checkpoint_interval_episodes",
        ),
        "metrics_path": metrics_path,
    }
    return SafeTrainingPlan(
        pipeline_id=pipeline_id,
        source_path=str(source),
        # Hash the versioned source document itself so the identifier is
        # stable when the same relative-path plan is moved with the project.
        plan_hash=_sha256_json(raw),
        seed_split=StrictTrainingSeedSplit.from_dict(
            raw.get("seed_split")
        ),
        preparation_stages=tuple(preparation),
        online_stages=online,
        checkpoint_interval_episodes=(
            normalized_for_hash["checkpoint_interval_episodes"]
        ),
        metrics_path=str(metrics_path),
    )


def validate_preparation_artifacts(plan: SafeTrainingPlan) -> None:
    """Fail closed when stage-1 data required by stage 2 is absent."""
    for stage in plan.preparation_stages:
        if (
            stage["stage_type"] == "demonstration_generation"
            and stage["require_completed_artifact"]
        ):
            manifest = Path(stage["artifact_manifest_path"])
            if not manifest.is_file():
                raise FileNotFoundError(
                    "stage-1 safe demonstration artifact is missing: "
                    f"{manifest}"
                )


def apply_curriculum_to_env_kwargs(
    base_env_kwargs: Mapping[str, Any],
    profile: CurriculumProfile,
    *,
    training_seed: int,
) -> dict[str, Any]:
    """Return constructor kwargs for one profile without changing topology.

    Resource *capacity* is varied by scaling existing PC/BW tier values.
    Host and VM counts are copied unchanged, because changing those counts
    changes action dimensions and therefore requires a separate plan.
    """
    result = dict(base_env_kwargs)
    result.update(
        {
            "random_seed": int(training_seed),
            "arrival_lambda": profile.arrival_lambda,
            "workflows_per_episode": (
                profile.workflows_per_episode
            ),
            "deadline_alpha_small": (
                profile.deadline_alpha_small
            ),
            "deadline_alpha_large": (
                profile.deadline_alpha_large
            ),
            "deadline_alpha_small_prob": (
                profile.deadline_alpha_small_prob
            ),
            "fuzzy_delta1": profile.fuzzy_delta1,
            "fuzzy_delta2": profile.fuzzy_delta2,
        }
    )
    for key in (
        "cloud_pc_tiers",
        "edge_pc_tiers",
        "cloud_bw_tiers",
        "edge_bw_tiers",
    ):
        if key not in base_env_kwargs:
            raise ValueError(
                f"base environment kwargs are missing {key}"
            )
        result[key] = tuple(
            float(value) * profile.resource_capacity_scale
            for value in base_env_kwargs[key]
        )
    for key in (
        "num_cloud_hosts",
        "num_edge_hosts",
        "cloud_vms_per_host",
        "edge_vms_per_host",
    ):
        if result.get(key) != base_env_kwargs.get(key):
            raise ValueError(
                "curriculum cannot change Host/VM topology inside one "
                "D3QN training plan"
            )
    return result


@dataclass(frozen=True)
class StageMetrics:
    fuzzy_energy_score: float
    safety_cost: float
    violation_rate: float
    shield_intervention_rate: float
    fallback_rate: float
    lagrange_multiplier: float
    q_c_prediction_error: float
    q_c_prediction_error_sample_count: int = 0

    def __post_init__(self) -> None:
        nonnegative = (
            "fuzzy_energy_score",
            "safety_cost",
            "lagrange_multiplier",
            "q_c_prediction_error",
        )
        for name in nonnegative:
            _finite_float(
                getattr(self, name),
                name,
                minimum=0.0,
            )
        for name in (
            "violation_rate",
            "shield_intervention_rate",
            "fallback_rate",
        ):
            _finite_float(
                getattr(self, name),
                name,
                minimum=0.0,
                maximum=1.0,
            )
        if int(self.q_c_prediction_error_sample_count) < 0:
            raise ValueError(
                "q_c_prediction_error_sample_count must be non-negative"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "fuzzy_energy_score": float(
                self.fuzzy_energy_score
            ),
            "safety_cost": float(self.safety_cost),
            "violation_rate": float(self.violation_rate),
            "shield_intervention_rate": float(
                self.shield_intervention_rate
            ),
            "fallback_rate": float(self.fallback_rate),
            "lambda": float(self.lagrange_multiplier),
            "q_c_prediction_error": float(
                self.q_c_prediction_error
            ),
            "q_c_prediction_error_sample_count": int(
                self.q_c_prediction_error_sample_count
            ),
        }


def q_c_prediction_error_from_agents(
    agents: Iterable[Any],
) -> tuple[float, int]:
    """Mean latest Q_c Bellman Huber error across updated layers."""
    values: list[float] = []
    for agent in agents:
        info = getattr(agent, "last_update_info", None)
        if not isinstance(info, Mapping):
            continue
        value = info.get("safety_loss")
        if value is None:
            continue
        number = float(value)
        if math.isfinite(number) and number >= 0.0:
            values.append(number)
    if not values:
        return 0.0, 0
    return float(sum(values) / len(values)), len(values)


def _zero_metric_sums() -> dict[str, float]:
    return {
        "fuzzy_energy_score": 0.0,
        "safety_cost": 0.0,
        "violation_rate": 0.0,
        "shield_intervention_rate": 0.0,
        "fallback_rate": 0.0,
        "lambda": 0.0,
        "q_c_prediction_error": 0.0,
    }


class SafeTrainingController:
    """Advance online stages using validation metrics only."""

    def __init__(self, plan: SafeTrainingPlan):
        self.plan = plan
        self.current_stage_index = 0
        self.stage_episode_count = 0
        self.total_episode_count = 0
        self.consecutive_validation_passes = 0
        self.completed = False
        self.stage_metric_count = 0
        self.stage_metric_sums = _zero_metric_sums()
        self.completed_stage_summaries: list[dict[str, Any]] = []

    @property
    def current_stage(self) -> OnlineTrainingStage:
        return self.plan.online_stages[self.current_stage_index]

    def training_seed_for_next_episode(self) -> int:
        seeds = self.plan.seed_split.training
        if self.current_stage.training_seed_mode == "primary":
            return int(seeds[0])
        return int(
            seeds[self.stage_episode_count % len(seeds)]
        )

    def _threshold_passes(
        self,
        metrics: StageMetrics,
        transition: StageTransition,
    ) -> bool:
        values = metrics.to_dict()
        key_map = {
            "max_fuzzy_energy_score": "fuzzy_energy_score",
            "max_safety_cost": "safety_cost",
            "max_violation_rate": "violation_rate",
            "max_shield_intervention_rate": (
                "shield_intervention_rate"
            ),
            "max_fallback_rate": "fallback_rate",
            "max_q_c_prediction_error": (
                "q_c_prediction_error"
            ),
        }
        return all(
            float(values[key_map[key]]) <= float(limit)
            for key, limit in transition.thresholds.items()
        )

    def _stage_summary(self) -> dict[str, Any]:
        denominator = max(self.stage_metric_count, 1)
        return {
            "stage_id": self.current_stage.stage_id,
            "stage_type": self.current_stage.stage_type,
            "episode_count": int(self.stage_episode_count),
            "validation_metric_count": int(
                self.stage_metric_count
            ),
            "mean_metrics": {
                key: float(value / denominator)
                for key, value in self.stage_metric_sums.items()
            },
        }

    def record_episode(self) -> dict[str, Any]:
        """Advance a fixed stage after one training episode."""
        if self.completed:
            raise RuntimeError("safe training pipeline is complete")
        stage_before = self.current_stage
        training_seed_used = self.training_seed_for_next_episode()
        self.stage_episode_count += 1
        self.total_episode_count += 1
        transition = stage_before.transition
        if transition.mode != "fixed_episodes":
            raise ValueError(
                "curriculum stage duration must use fixed_episodes; "
                "validation metrics cannot control stage duration"
            )
        should_advance = self.stage_episode_count >= int(
            transition.fixed_episodes
        )
        summary = None
        if should_advance:
            summary = self._stage_summary()
            self.completed_stage_summaries.append(summary)
            if self.current_stage_index + 1 >= len(self.plan.online_stages):
                self.completed = True
            else:
                self.current_stage_index += 1
                self.stage_episode_count = 0
                self.consecutive_validation_passes = 0
                self.stage_metric_count = 0
                self.stage_metric_sums = _zero_metric_sums()
        return {
            "stage_id": stage_before.stage_id,
            "stage_type": stage_before.stage_type,
            "training_seed_mode": stage_before.training_seed_mode,
            "training_seed_used": int(training_seed_used),
            "transitioned": bool(should_advance),
            "pipeline_completed": bool(self.completed),
            "transition_reason": (
                "fixed_episode_limit"
                if should_advance
                else "fixed_episode_progress"
            ),
            "next_stage_id": (
                None if self.completed else self.current_stage.stage_id
            ),
            "completed_stage_summary": summary,
        }

    def observe_validation(
        self,
        metrics: StageMetrics,
        *,
        source: str = "curriculum_validation",
        count_episode: bool = False,
    ) -> dict[str, Any]:
        """Record stage diagnostics without changing stage duration."""
        if source not in {"validation", "curriculum_validation"}:
            raise ValueError(
                "stage diagnostics accept curriculum validation only; "
                "formal validation and final test are read-only"
            )
        if count_episode:
            raise ValueError(
                "validation cannot count as a training episode or control "
                "curriculum duration"
            )
        if self.completed:
            raise RuntimeError("safe training pipeline is complete")
        stage = self.current_stage
        self.stage_metric_count += 1
        for key, value in metrics.to_dict().items():
            if key in self.stage_metric_sums:
                self.stage_metric_sums[key] += float(value)
        threshold_pass = (
            self._threshold_passes(metrics, stage.transition)
            if stage.transition.thresholds
            else None
        )
        return {
            "stage_id": stage.stage_id,
            "stage_type": stage.stage_type,
            "training_seed_mode": stage.training_seed_mode,
            "training_seed_used": int(self.training_seed_for_next_episode()),
            "transitioned": False,
            "pipeline_completed": False,
            "transition_reason": "curriculum_validation_observation_only",
            "curriculum_check_pass": threshold_pass,
            "next_stage_id": stage.stage_id,
            "completed_stage_summary": None,
        }
    def state_dict(self) -> dict[str, Any]:
        return {
            "state_schema_version": (
                SAFE_TRAINING_STATE_SCHEMA_VERSION
            ),
            "pipeline_id": self.plan.pipeline_id,
            "plan_hash": self.plan.plan_hash,
            "current_stage_index": int(
                self.current_stage_index
            ),
            "current_stage_id": self.current_stage.stage_id,
            "stage_episode_count": int(
                self.stage_episode_count
            ),
            "total_episode_count": int(
                self.total_episode_count
            ),
            "consecutive_validation_passes": int(
                self.consecutive_validation_passes
            ),
            "completed": bool(self.completed),
            "stage_metric_count": int(self.stage_metric_count),
            "stage_metric_sums": dict(self.stage_metric_sums),
            "completed_stage_summaries": list(
                self.completed_stage_summaries
            ),
        }

    def load_state_dict(
        self,
        state: Mapping[str, Any],
    ) -> None:
        state = _require_mapping(
            state,
            "safe training controller state",
        )
        if (
            int(state.get("state_schema_version", -1))
            != SAFE_TRAINING_STATE_SCHEMA_VERSION
        ):
            raise ValueError(
                "unsupported safe training state schema version"
            )
        if state.get("pipeline_id") != self.plan.pipeline_id:
            raise ValueError(
                "safe training checkpoint pipeline_id mismatch"
            )
        if state.get("plan_hash") != self.plan.plan_hash:
            raise ValueError(
                "safe training checkpoint plan hash mismatch"
            )
        index = int(state.get("current_stage_index", -1))
        if not 0 <= index < len(self.plan.online_stages):
            raise ValueError(
                "safe training checkpoint stage index is invalid"
            )
        if (
            state.get("current_stage_id")
            != self.plan.online_stages[index].stage_id
        ):
            raise ValueError(
                "safe training checkpoint stage ID mismatch"
            )
        counters = {
            "stage_episode_count": int(
                state.get("stage_episode_count", -1)
            ),
            "total_episode_count": int(
                state.get("total_episode_count", -1)
            ),
            "consecutive_validation_passes": int(
                state.get("consecutive_validation_passes", -1)
            ),
            "stage_metric_count": int(
                state.get("stage_metric_count", -1)
            ),
        }
        if any(value < 0 for value in counters.values()):
            raise ValueError(
                "safe training checkpoint counters must be "
                "non-negative"
            )
        sums = _require_mapping(
            state.get("stage_metric_sums"),
            "stage_metric_sums",
        )
        if set(sums) != set(_zero_metric_sums()):
            raise ValueError(
                "safe training checkpoint metric schema mismatch"
            )
        validated_sums = {
            key: _finite_float(
                value,
                f"stage_metric_sums.{key}",
                minimum=0.0,
            )
            for key, value in sums.items()
        }
        summaries = state.get("completed_stage_summaries", [])
        if not isinstance(summaries, list):
            raise ValueError(
                "completed_stage_summaries must be a list"
            )
        self.current_stage_index = index
        self.stage_episode_count = counters[
            "stage_episode_count"
        ]
        self.total_episode_count = counters[
            "total_episode_count"
        ]
        self.consecutive_validation_passes = counters[
            "consecutive_validation_passes"
        ]
        self.stage_metric_count = counters[
            "stage_metric_count"
        ]
        self.completed = bool(state.get("completed", False))
        self.stage_metric_sums = validated_sums
        self.completed_stage_summaries = list(summaries)


class SafeStageMetricsLogger:
    """Append versioned per-stage validation records as JSON Lines."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        *,
        controller: SafeTrainingController,
        metrics: StageMetrics,
        transition_event: Mapping[str, Any],
        global_step: int,
        episode: int,
    ) -> None:
        record = {
            "schema_version": 1,
            "pipeline_id": controller.plan.pipeline_id,
            "plan_hash": controller.plan.plan_hash,
            "stage_id": transition_event["stage_id"],
            "stage_type": transition_event["stage_type"],
            "global_step": int(global_step),
            "episode": int(episode),
            "metric_source": "validation",
            "validation_seeds": list(
                controller.plan.seed_split.validation
            ),
            "final_test_consumed": False,
            "metrics": metrics.to_dict(),
            "transition": dict(transition_event),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(_canonical_json(record))
            handle.write("\n")


def save_pipeline_checkpoint(
    directory: str | os.PathLike[str],
    *,
    controller: SafeTrainingController,
    agents: Mapping[str, Any],
    lagrange_controller: Any,
    global_step: int,
    next_episode: int,
    best_model_metrics: Mapping[str, Any] | None,
    replay_metadata: Mapping[str, Any],
    heuristic_library_version: Mapping[str, Any],
    config_snapshot: Mapping[str, Any],
    experiment_protocol=None,
) -> str:
    """Save complete safe-HRL state plus a versioned orchestration manifest."""
    target_dir = Path(directory).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    expected_layers = {"manager", "host", "vm"}
    if set(agents) != expected_layers:
        raise ValueError(
            "pipeline checkpoint requires manager/host/vm agents"
        )
    if not all(
        bool(getattr(agent, "safe_rl_enabled", False))
        for agent in agents.values()
    ):
        raise ValueError(
            "pipeline checkpoint requires safe_rl agents with Q_c"
        )
    if set(replay_metadata) != expected_layers:
        raise ValueError(
            "pipeline checkpoint replay metadata layer mismatch"
        )
    if not isinstance(heuristic_library_version, Mapping):
        raise ValueError(
            "pipeline checkpoint requires heuristic library version"
        )
    if not isinstance(config_snapshot, Mapping):
        raise ValueError(
            "pipeline checkpoint requires config snapshot"
        )
    lagrange_state = (
        lagrange_controller.state_dict()
        if lagrange_controller is not None
        else None
    )
    checkpoint_files: dict[str, str] = {}
    for layer in sorted(expected_layers):
        filename = f"pipeline_{layer}.pth"
        path = target_dir / filename
        agents[layer].save(
            str(path),
            lagrange_controller_state=lagrange_state,
        )
        checkpoint_files[layer] = filename

    if best_model_metrics is not None:
        best_model_metrics = dict(
            _require_mapping(
                best_model_metrics,
                "best_model_metrics",
            )
        )
        for field_name in (
            "deadline_violation_rate",
            "max_fuzzy_lateness",
            "mean_fuzzy_lateness",
            "fuzzy_energy_score",
        ):
            if field_name not in best_model_metrics:
                raise ValueError(
                    "pipeline checkpoint best_model_metrics "
                    f"is missing {field_name}"
                )
    best_energy_json = (
        None
        if best_model_metrics is None
        else float(best_model_metrics["fuzzy_energy_score"])
    )
    snapshot_protocol = protocol_identity_from_config_snapshot(
        config_snapshot
    )
    explicit_protocol = None
    if experiment_protocol is not None:
        explicit_protocol = validate_protocol_identity(
            experiment_protocol,
            experiment_protocol,
            artifact_name="safe training checkpoint protocol",
        )
    if snapshot_protocol is not None and explicit_protocol is not None:
        validate_protocol_identity(
            explicit_protocol,
            snapshot_protocol,
            artifact_name="safe training config snapshot",
        )
    protocol_identity = explicit_protocol or snapshot_protocol
    library_protocol = heuristic_library_version.get(
        "experiment_protocol"
    )
    if protocol_identity is not None and library_protocol is not None:
        validate_protocol_identity(
            protocol_identity,
            library_protocol,
            artifact_name="safe training heuristic library",
        )
    payload = {
        "checkpoint_schema_version": (
            SAFE_TRAINING_CHECKPOINT_SCHEMA_VERSION
        ),
        "pipeline_id": controller.plan.pipeline_id,
        "plan_hash": controller.plan.plan_hash,
        "controller_state": controller.state_dict(),
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
        "global_step": int(global_step),
        "next_episode": int(next_episode),
        "best_model_metrics": best_model_metrics,
        # Deprecated convenience alias retained for audit readers.
        "best_validation_energy": best_energy_json,
        "curriculum_stage": {
            "stage_index": int(controller.current_stage_index),
            "stage_id": controller.current_stage.stage_id,
            "stage_type": controller.current_stage.stage_type,
        },
        "replay_metadata": dict(replay_metadata),
        "heuristic_library_version": dict(
            heuristic_library_version
        ),
        "config_snapshot": dict(config_snapshot),
    }
    if protocol_identity is not None:
        payload["experiment_protocol"] = protocol_identity
    manifest_path = target_dir / "safe_training_checkpoint.json"
    temp_path = target_dir / "safe_training_checkpoint.json.tmp"
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        handle.write("\n")
    os.replace(temp_path, manifest_path)
    return str(manifest_path)


def read_pipeline_checkpoint(
    path: str | os.PathLike[str],
    *,
    controller: SafeTrainingController,
    expected_protocol_identity=None,
) -> dict[str, Any]:
    """Read and validate orchestration state before environments are built."""
    source = Path(path).resolve()
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload = dict(
        _require_mapping(payload, "safe training checkpoint")
    )
    if (
        int(payload.get("checkpoint_schema_version", -1))
        != SAFE_TRAINING_CHECKPOINT_SCHEMA_VERSION
    ):
        raise ValueError(
            "unsupported safe training checkpoint schema version"
        )
    if payload.get("pipeline_id") != controller.plan.pipeline_id:
        raise ValueError(
            "safe training checkpoint pipeline_id mismatch"
        )
    if payload.get("plan_hash") != controller.plan.plan_hash:
        raise ValueError(
            "safe training checkpoint plan hash mismatch"
        )
    actual_protocol = payload.get("experiment_protocol")
    if expected_protocol_identity is not None:
        if not isinstance(actual_protocol, Mapping):
            raise ValueError(
                "safe training checkpoint is missing experiment_protocol"
            )
        payload["experiment_protocol"] = validate_protocol_identity(
            expected_protocol_identity,
            actual_protocol,
            artifact_name="safe training checkpoint",
        )
    elif actual_protocol is not None:
        payload["experiment_protocol"] = validate_protocol_identity(
            actual_protocol,
            actual_protocol,
            artifact_name="safe training checkpoint",
        )
    best_model_metrics = payload.get("best_model_metrics")
    if best_model_metrics is not None:
        best_model_metrics = _require_mapping(
            best_model_metrics,
            "best_model_metrics",
        )
        for field_name in (
            "deadline_violation_rate",
            "max_fuzzy_lateness",
            "mean_fuzzy_lateness",
            "fuzzy_energy_score",
        ):
            if field_name not in best_model_metrics:
                raise ValueError(
                    "safe training checkpoint best_model_metrics "
                    f"is missing {field_name}"
                )
    replay_metadata = _require_mapping(
        payload.get("replay_metadata"),
        "replay_metadata",
    )
    if set(replay_metadata) != {"manager", "host", "vm"}:
        raise ValueError(
            "safe training checkpoint replay metadata layer mismatch"
        )
    heuristic_library = _require_mapping(
        payload.get("heuristic_library_version"),
        "heuristic_library_version",
    )
    config_snapshot = _require_mapping(
        payload.get("config_snapshot"),
        "config_snapshot",
    )
    snapshot_protocol = protocol_identity_from_config_snapshot(
        config_snapshot
    )
    normalized_protocol = payload.get("experiment_protocol")
    if normalized_protocol is not None and snapshot_protocol is not None:
        validate_protocol_identity(
            normalized_protocol,
            snapshot_protocol,
            artifact_name="safe training config snapshot",
        )
    library_protocol = heuristic_library.get("experiment_protocol")
    if normalized_protocol is not None and library_protocol is not None:
        validate_protocol_identity(
            normalized_protocol,
            library_protocol,
            artifact_name="safe training heuristic library",
        )
    _require_mapping(
        payload.get("curriculum_stage"),
        "curriculum_stage",
    )
    controller.load_state_dict(payload.get("controller_state"))
    checkpoints = _require_mapping(
        payload.get("agent_checkpoints"),
        "agent_checkpoints",
    )
    if set(checkpoints) != {"manager", "host", "vm"}:
        raise ValueError(
            "safe training checkpoint layer set mismatch"
        )
    payload["resolved_agent_checkpoints"] = {
        layer: str((source.parent / filename).resolve())
        for layer, filename in checkpoints.items()
    }
    if int(payload.get("global_step", -1)) < 0:
        raise ValueError("checkpoint global_step must be non-negative")
    if int(payload.get("next_episode", -1)) < 0:
        raise ValueError("checkpoint next_episode must be non-negative")
    if (
        int(payload["next_episode"])
        != controller.total_episode_count
    ):
        raise ValueError(
            "checkpoint next_episode does not match the curriculum "
            "controller episode count"
        )
    return payload


def restore_pipeline_agents(
    payload: Mapping[str, Any],
    *,
    agents: Mapping[str, Any],
    lagrange_controller: Any,
) -> None:
    """Restore dimension-checked Agent and shared-lambda states."""
    checkpoints = _require_mapping(
        payload.get("resolved_agent_checkpoints"),
        "resolved_agent_checkpoints",
    )
    if set(agents) != {"manager", "host", "vm"}:
        raise ValueError(
            "pipeline restore requires manager/host/vm agents"
        )
    for layer in ("manager", "host", "vm"):
        agents[layer].load(str(checkpoints[layer]))
    lagrange_state = payload.get("lagrange_controller_state")
    if lagrange_controller is not None and lagrange_state is not None:
        lagrange_controller.load_state_dict(
            lagrange_state,
            strict=True,
        )
        for agent in agents.values():
            agent.lagrange_multiplier = float(
                lagrange_controller.current_lambda
            )


__all__ = [
    "CurriculumProfile",
    "OnlineTrainingStage",
    "SAFE_TRAINING_CHECKPOINT_SCHEMA_VERSION",
    "SAFE_TRAINING_PLAN_SCHEMA_VERSION",
    "SAFE_TRAINING_STATE_SCHEMA_VERSION",
    "SafeStageMetricsLogger",
    "SafeTrainingController",
    "SafeTrainingPlan",
    "StageMetrics",
    "StrictTrainingSeedSplit",
    "apply_curriculum_to_env_kwargs",
    "load_safe_training_plan",
    "q_c_prediction_error_from_agents",
    "read_pipeline_checkpoint",
    "restore_pipeline_agents",
    "save_pipeline_checkpoint",
    "validate_preparation_artifacts",
]
