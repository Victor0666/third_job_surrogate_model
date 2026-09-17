"""安全 HRL 示范轨迹的数据集、严格 seed 划分与版本校验。

每条轨迹直接复用 :class:`SafeReplayTransition`，因此离线数据与在线
replay 的动作、reward、cost、mask 和安全诊断字段保持同一协议。episode
是否属于 safe demonstration 只由最终评价指标和显式安全标准计算，不能由
调用方直接标记。
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Mapping, Sequence

from base.safe_replay import (
    SAFE_REPLAY_TRANSITION_SCHEMA_VERSION,
    SafeReplayTransition,
)


SAFE_DEMONSTRATION_DATASET_SCHEMA_VERSION = 1
SAFE_DEMONSTRATION_EPISODE_SCHEMA_VERSION = 1
SAFE_DEMONSTRATION_GENERATOR_POLICY = (
    "safe_heuristic_fixed_vm_deterministic_fallback_v1"
)
DEMONSTRATION_LAYERS = ("manager", "host", "vm")
TRAINING_SPLITS = frozenset({"train", "validation"})
ALL_SPLITS = frozenset({"train", "validation", "final_test"})

_SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


def _canonical_json_sha256(value) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _seed_tuple(values: Sequence[int], name: str) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if not result:
        raise ValueError(f"{name} must not be empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


@dataclass(frozen=True)
class StrictSeedSplit:
    """工作流 seed 与模糊资源 seed 的严格 train/val/test 划分。"""

    train_workflow_seeds: tuple[int, ...]
    validation_workflow_seeds: tuple[int, ...]
    final_test_workflow_seeds: tuple[int, ...]
    train_resource_seeds: tuple[int, ...]
    validation_resource_seeds: tuple[int, ...]
    final_test_resource_seeds: tuple[int, ...]
    split_version: str = "strict_seed_split_v1"

    def __post_init__(self) -> None:
        fields = (
            "train_workflow_seeds",
            "validation_workflow_seeds",
            "final_test_workflow_seeds",
            "train_resource_seeds",
            "validation_resource_seeds",
            "final_test_resource_seeds",
        )
        for name in fields:
            object.__setattr__(
                self,
                name,
                _seed_tuple(getattr(self, name), name),
            )
        if not str(self.split_version).strip():
            raise ValueError("split_version must not be empty")

        # 同一个整数 seed 不得以工作流或资源的任一身份跨 split 复用。
        split_values = {
            "train": set(self.train_workflow_seeds)
            | set(self.train_resource_seeds),
            "validation": set(self.validation_workflow_seeds)
            | set(self.validation_resource_seeds),
            "final_test": set(self.final_test_workflow_seeds)
            | set(self.final_test_resource_seeds),
        }
        for left, right in (
            ("train", "validation"),
            ("train", "final_test"),
            ("validation", "final_test"),
        ):
            overlap = split_values[left] & split_values[right]
            if overlap:
                raise ValueError(
                    f"seed leakage between {left} and {right}: "
                    f"{sorted(overlap)}"
                )

    def split_for(
        self,
        workflow_seed: int,
        resource_seed: int,
    ) -> str:
        workflow_seed = int(workflow_seed)
        resource_seed = int(resource_seed)
        matches = []
        for split in ALL_SPLITS:
            workflow_values = getattr(
                self,
                f"{split}_workflow_seeds",
            )
            resource_values = getattr(
                self,
                f"{split}_resource_seeds",
            )
            if (
                workflow_seed in workflow_values
                and resource_seed in resource_values
            ):
                matches.append(split)
        if len(matches) != 1:
            raise ValueError(
                "workflow/resource seeds must belong to the same "
                "single split"
            )
        return matches[0]

    def to_dict(self) -> dict:
        return {
            "split_version": str(self.split_version),
            "train_workflow_seeds": list(
                self.train_workflow_seeds
            ),
            "validation_workflow_seeds": list(
                self.validation_workflow_seeds
            ),
            "final_test_workflow_seeds": list(
                self.final_test_workflow_seeds
            ),
            "train_resource_seeds": list(
                self.train_resource_seeds
            ),
            "validation_resource_seeds": list(
                self.validation_resource_seeds
            ),
            "final_test_resource_seeds": list(
                self.final_test_resource_seeds
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> "StrictSeedSplit":
        if not isinstance(payload, Mapping):
            raise ValueError("seed split must be a mapping")
        return cls(
            train_workflow_seeds=tuple(
                payload["train_workflow_seeds"]
            ),
            validation_workflow_seeds=tuple(
                payload["validation_workflow_seeds"]
            ),
            final_test_workflow_seeds=tuple(
                payload["final_test_workflow_seeds"]
            ),
            train_resource_seeds=tuple(
                payload["train_resource_seeds"]
            ),
            validation_resource_seeds=tuple(
                payload["validation_resource_seeds"]
            ),
            final_test_resource_seeds=tuple(
                payload["final_test_resource_seeds"]
            ),
            split_version=str(payload["split_version"]),
        )


@dataclass(frozen=True)
class DemonstrationSafetyStandard:
    """episode 获得 safe demonstration 标签所需的最终安全标准。"""

    deadline_violation_rate_max: float = 0.0
    max_fuzzy_lateness_max: float = 0.0
    require_constraint_feasible: bool = True
    require_all_workflows_completed: bool = True
    standard_version: str = "zero_fuzzy_ddl_violation_v1"

    def __post_init__(self) -> None:
        violation = _finite(
            self.deadline_violation_rate_max,
            "deadline_violation_rate_max",
        )
        lateness = _finite(
            self.max_fuzzy_lateness_max,
            "max_fuzzy_lateness_max",
        )
        if violation != 0.0 or lateness != 0.0:
            raise ValueError(
                "safe demonstrations require zero final fuzzy-DDL "
                "violation and zero maximum fuzzy lateness"
            )
        if not str(self.standard_version).strip():
            raise ValueError("standard_version must not be empty")

    def evaluate(self, metrics: Mapping) -> tuple[bool, list[str]]:
        reasons = []
        violation_rate = _finite(
            metrics.get("deadline_violation_rate"),
            "deadline_violation_rate",
        )
        maximum_lateness = _finite(
            metrics.get("max_fuzzy_lateness"),
            "max_fuzzy_lateness",
        )
        if (
            violation_rate
            > self.deadline_violation_rate_max + 1e-12
        ):
            reasons.append("nonzero_deadline_violation_rate")
        if maximum_lateness > self.max_fuzzy_lateness_max + 1e-9:
            reasons.append("nonzero_max_fuzzy_lateness")
        if (
            self.require_constraint_feasible
            and not bool(metrics.get("constraint_feasible", False))
        ):
            reasons.append("episode_constraint_not_feasible")
        if (
            self.require_all_workflows_completed
            and not bool(
                metrics.get("all_workflows_completed", False)
            )
        ):
            reasons.append("episode_not_fully_completed")
        return not reasons, reasons

    def to_dict(self) -> dict:
        return {
            "standard_version": str(self.standard_version),
            "deadline_violation_rate_max": float(
                self.deadline_violation_rate_max
            ),
            "max_fuzzy_lateness_max": float(
                self.max_fuzzy_lateness_max
            ),
            "require_constraint_feasible": bool(
                self.require_constraint_feasible
            ),
            "require_all_workflows_completed": bool(
                self.require_all_workflows_completed
            ),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping,
    ) -> "DemonstrationSafetyStandard":
        if not isinstance(payload, Mapping):
            raise ValueError("safety standard must be a mapping")
        return cls(
            deadline_violation_rate_max=payload[
                "deadline_violation_rate_max"
            ],
            max_fuzzy_lateness_max=payload[
                "max_fuzzy_lateness_max"
            ],
            require_constraint_feasible=payload[
                "require_constraint_feasible"
            ],
            require_all_workflows_completed=payload[
                "require_all_workflows_completed"
            ],
            standard_version=str(payload["standard_version"]),
        )


@dataclass(frozen=True)
class SafeDemonstrationEpisode:
    """一个三层示范 episode 及其可复算安全标签。"""

    episode_id: str
    split: str
    generator_policy: str
    heuristic_id: str
    heuristic_source: str
    heuristic_version: str
    workflow_seed: int
    resource_seed: int
    ddl_setting: Mapping
    fuzzy_parameters: Mapping
    observation_schema_versions: Mapping[str, str]
    episode_metrics: Mapping
    trajectories: Mapping[str, Sequence[SafeReplayTransition]]
    safety_standard: DemonstrationSafetyStandard

    def __post_init__(self) -> None:
        identifier = str(self.episode_id).strip()
        if not identifier or _SAFE_ID.fullmatch(identifier) is None:
            raise ValueError(
                "episode_id must contain only letters, digits, '.', "
                "'_' or '-'"
            )
        split = str(self.split).strip().lower()
        if split not in ALL_SPLITS:
            raise ValueError(f"unsupported demonstration split: {split}")
        if not str(self.generator_policy).strip():
            raise ValueError("generator_policy must not be empty")
        if (
            str(self.generator_policy)
            != SAFE_DEMONSTRATION_GENERATOR_POLICY
        ):
            raise ValueError(
                "unsupported safe demonstration generator policy"
            )
        if not str(self.heuristic_id).strip():
            raise ValueError("heuristic_id must not be empty")
        if not isinstance(self.ddl_setting, Mapping):
            raise ValueError("ddl_setting must be a mapping")
        if not isinstance(self.fuzzy_parameters, Mapping):
            raise ValueError("fuzzy_parameters must be a mapping")
        observation_schemas = dict(
            self.observation_schema_versions
        )
        if set(observation_schemas) != set(DEMONSTRATION_LAYERS):
            raise ValueError(
                "observation_schema_versions must contain manager, "
                "host and vm"
            )
        if any(
            not str(observation_schemas[layer]).strip()
            for layer in DEMONSTRATION_LAYERS
        ):
            raise ValueError(
                "observation schema versions must not be empty"
            )
        eta = _finite(
            self.fuzzy_parameters.get("deadline_eta"),
            "fuzzy_parameters.deadline_eta",
        )
        energy_lambda = _finite(
            self.fuzzy_parameters.get(
                "energy_uncertainty_weight"
            ),
            "fuzzy_parameters.energy_uncertainty_weight",
        )
        if not math.isclose(eta, 0.95, abs_tol=1e-12):
            raise ValueError(
                "demonstrations must use fuzzy deadline eta=0.95"
            )
        if not math.isclose(
            energy_lambda,
            1.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "demonstrations must use fuzzy energy lambda_E=1.0"
            )
        metrics = dict(self.episode_metrics)
        for name in (
            "deadline_violation_rate",
            "max_fuzzy_lateness",
            "fuzzy_energy_mean",
            "fuzzy_energy_std",
            "fuzzy_energy_score",
        ):
            _finite(metrics.get(name), f"episode_metrics.{name}")

        trajectories = dict(self.trajectories)
        if set(trajectories) != set(DEMONSTRATION_LAYERS):
            raise ValueError(
                "trajectories must contain manager, host and vm layers"
            )
        copied = {}
        for layer in DEMONSTRATION_LAYERS:
            rows = tuple(trajectories[layer])
            if not rows:
                raise ValueError(
                    f"{layer} demonstration trajectory is empty"
                )
            if any(
                not isinstance(row, SafeReplayTransition)
                for row in rows
            ):
                raise ValueError(
                    f"{layer} trajectory contains a non-safe replay row"
                )
            copied[layer] = rows

        object.__setattr__(self, "episode_id", identifier)
        object.__setattr__(self, "split", split)
        object.__setattr__(self, "workflow_seed", int(self.workflow_seed))
        object.__setattr__(self, "resource_seed", int(self.resource_seed))
        object.__setattr__(self, "ddl_setting", dict(self.ddl_setting))
        object.__setattr__(
            self,
            "fuzzy_parameters",
            dict(self.fuzzy_parameters),
        )
        object.__setattr__(
            self,
            "observation_schema_versions",
            {
                layer: str(observation_schemas[layer])
                for layer in DEMONSTRATION_LAYERS
            },
        )
        object.__setattr__(self, "episode_metrics", metrics)
        object.__setattr__(self, "trajectories", copied)

    @property
    def safety_evaluation(self) -> tuple[bool, list[str]]:
        return self.safety_standard.evaluate(self.episode_metrics)

    @property
    def safe_demonstration(self) -> bool:
        return bool(self.safety_evaluation[0])

    @property
    def safety_rejection_reasons(self) -> list[str]:
        return list(self.safety_evaluation[1])

    @property
    def trajectory_count(self) -> dict:
        by_layer = {
            layer: len(self.trajectories[layer])
            for layer in DEMONSTRATION_LAYERS
        }
        return {
            **by_layer,
            "total": int(sum(by_layer.values())),
        }

    @property
    def layer_dimensions(self) -> dict:
        return {
            layer: {
                "input_dim": int(
                    self.trajectories[layer][0].state.size
                ),
                "action_dim": int(
                    self.trajectories[layer][
                        0
                    ].final_action_mask.size
                ),
            }
            for layer in DEMONSTRATION_LAYERS
        }

    def to_dict(self) -> dict:
        return {
            "episode_schema_version": (
                SAFE_DEMONSTRATION_EPISODE_SCHEMA_VERSION
            ),
            "safe_replay_transition_schema_version": (
                SAFE_REPLAY_TRANSITION_SCHEMA_VERSION
            ),
            "episode_id": self.episode_id,
            "split": self.split,
            "generator_policy": str(self.generator_policy),
            "heuristic_id": str(self.heuristic_id),
            "heuristic_source": str(self.heuristic_source),
            "heuristic_version": str(self.heuristic_version),
            "workflow_seed": int(self.workflow_seed),
            "resource_seed": int(self.resource_seed),
            "ddl_setting": dict(self.ddl_setting),
            "fuzzy_parameters": dict(self.fuzzy_parameters),
            "observation_schema_versions": dict(
                self.observation_schema_versions
            ),
            "feasibility": {
                "safe_demonstration": self.safe_demonstration,
                "constraint_feasible": bool(
                    self.episode_metrics[
                        "constraint_feasible"
                    ]
                ),
                "all_workflows_completed": bool(
                    self.episode_metrics[
                        "all_workflows_completed"
                    ]
                ),
                "deadline_violation_rate": float(
                    self.episode_metrics[
                        "deadline_violation_rate"
                    ]
                ),
                "max_fuzzy_lateness": float(
                    self.episode_metrics[
                        "max_fuzzy_lateness"
                    ]
                ),
                "rejection_reasons": (
                    self.safety_rejection_reasons
                ),
            },
            "fuzzy_energy_score": float(
                self.episode_metrics["fuzzy_energy_score"]
            ),
            "episode_metrics": dict(self.episode_metrics),
            "trajectory_count": self.trajectory_count,
            "layer_dimensions": self.layer_dimensions,
            "trajectories": {
                layer: [
                    transition.to_dict()
                    for transition in self.trajectories[layer]
                ]
                for layer in DEMONSTRATION_LAYERS
            },
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping,
        *,
        safety_standard: DemonstrationSafetyStandard,
    ) -> "SafeDemonstrationEpisode":
        if (
            payload.get("episode_schema_version")
            != SAFE_DEMONSTRATION_EPISODE_SCHEMA_VERSION
        ):
            raise ValueError(
                "safe demonstration episode schema mismatch"
            )
        if (
            payload.get(
                "safe_replay_transition_schema_version"
            )
            != SAFE_REPLAY_TRANSITION_SCHEMA_VERSION
        ):
            raise ValueError(
                "safe demonstration replay schema mismatch"
            )
        episode = cls(
            episode_id=payload["episode_id"],
            split=payload["split"],
            generator_policy=payload["generator_policy"],
            heuristic_id=payload["heuristic_id"],
            heuristic_source=payload["heuristic_source"],
            heuristic_version=payload["heuristic_version"],
            workflow_seed=payload["workflow_seed"],
            resource_seed=payload["resource_seed"],
            ddl_setting=payload["ddl_setting"],
            fuzzy_parameters=payload["fuzzy_parameters"],
            observation_schema_versions=payload[
                "observation_schema_versions"
            ],
            episode_metrics=payload["episode_metrics"],
            trajectories={
                layer: [
                    SafeReplayTransition.from_dict(row)
                    for row in payload["trajectories"][layer]
                ]
                for layer in DEMONSTRATION_LAYERS
            },
            safety_standard=safety_standard,
        )
        feasibility = payload.get("feasibility", {})
        if (
            bool(feasibility.get("safe_demonstration", False))
            != episode.safe_demonstration
        ):
            raise ValueError(
                "stored safe demonstration label does not match "
                "episode metrics"
            )
        if payload.get("trajectory_count") != episode.trajectory_count:
            raise ValueError(
                "stored demonstration trajectory count mismatch"
            )
        if payload.get("layer_dimensions") != episode.layer_dimensions:
            raise ValueError(
                "stored demonstration layer dimensions mismatch"
            )
        return episode


def _manifest_episode_record(
    episode: SafeDemonstrationEpisode,
    *,
    episode_file: str,
    episode_file_sha256: str,
) -> dict:
    """生成包含用户要求字段的轻量数据集索引记录。"""
    return {
        "episode_id": episode.episode_id,
        "episode_file": str(episode_file),
        "episode_file_sha256": str(episode_file_sha256),
        "split": episode.split,
        "generator_policy": str(episode.generator_policy),
        "heuristic_id": str(episode.heuristic_id),
        "heuristic_source": str(episode.heuristic_source),
        "heuristic_version": str(episode.heuristic_version),
        "resource_seed": int(episode.resource_seed),
        "workflow_seed": int(episode.workflow_seed),
        "ddl_setting": dict(episode.ddl_setting),
        "fuzzy_parameters": dict(episode.fuzzy_parameters),
        "observation_schema_versions": dict(
            episode.observation_schema_versions
        ),
        "feasibility": {
            "safe_demonstration": episode.safe_demonstration,
            "constraint_feasible": bool(
                episode.episode_metrics["constraint_feasible"]
            ),
            "all_workflows_completed": bool(
                episode.episode_metrics[
                    "all_workflows_completed"
                ]
            ),
            "deadline_violation_rate": float(
                episode.episode_metrics[
                    "deadline_violation_rate"
                ]
            ),
            "max_fuzzy_lateness": float(
                episode.episode_metrics["max_fuzzy_lateness"]
            ),
            "rejection_reasons": (
                episode.safety_rejection_reasons
            ),
        },
        "fuzzy_energy_score": float(
            episode.episode_metrics["fuzzy_energy_score"]
        ),
        "trajectory_count": episode.trajectory_count,
        "layer_dimensions": episode.layer_dimensions,
    }


def append_demonstration_episode(
    manifest_path: str | Path,
    episode: SafeDemonstrationEpisode,
    *,
    seed_split: StrictSeedSplit,
    dataset_id: str = "safe_hrl_demonstrations",
    dataset_version: str = "2026-07-28.stage13.v1",
) -> dict:
    """原子写入 episode 并追加 manifest；不覆盖已有 ID。"""
    path = Path(manifest_path).resolve()
    expected_split = seed_split.split_for(
        episode.workflow_seed,
        episode.resource_seed,
    )
    if expected_split != episode.split:
        raise ValueError(
            "episode split does not match strict seed split"
        )
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if (
            manifest.get("schema_version")
            != SAFE_DEMONSTRATION_DATASET_SCHEMA_VERSION
        ):
            raise ValueError(
                "safe demonstration dataset schema mismatch"
            )
        if (
            manifest.get("generator_policy")
            != SAFE_DEMONSTRATION_GENERATOR_POLICY
        ):
            raise ValueError(
                "safe demonstration generator policy mismatch"
            )
        stored_split = StrictSeedSplit.from_dict(
            manifest["seed_split"]
        )
        if stored_split.to_dict() != seed_split.to_dict():
            raise ValueError(
                "demonstration seed split cannot change in place"
            )
        stored_standard = (
            DemonstrationSafetyStandard.from_dict(
                manifest["safety_standard"]
            )
        )
        if (
            stored_standard.to_dict()
            != episode.safety_standard.to_dict()
        ):
            raise ValueError(
                "demonstration safety standard cannot change in place"
            )
    else:
        if not str(dataset_id).strip():
            raise ValueError("dataset_id must not be empty")
        if not str(dataset_version).strip():
            raise ValueError("dataset_version must not be empty")
        manifest = {
            "schema_version": (
                SAFE_DEMONSTRATION_DATASET_SCHEMA_VERSION
            ),
            "dataset_id": str(dataset_id),
            "dataset_version": str(dataset_version),
            "manifest_revision": 0,
            "generator_policy": (
                SAFE_DEMONSTRATION_GENERATOR_POLICY
            ),
            "safe_replay_transition_schema_version": (
                SAFE_REPLAY_TRANSITION_SCHEMA_VERSION
            ),
            "seed_split": seed_split.to_dict(),
            "safety_standard": (
                episode.safety_standard.to_dict()
            ),
            "episodes": [],
        }
    entries = manifest.get("episodes")
    if not isinstance(entries, list):
        raise ValueError(
            "demonstration manifest episodes must be a list"
        )
    if any(
        row.get("episode_id") == episode.episode_id
        for row in entries
    ):
        raise ValueError(
            f"duplicate demonstration episode_id: "
            f"{episode.episode_id}"
        )
    layer_dimensions = manifest.get("layer_dimensions")
    if layer_dimensions is None:
        manifest["layer_dimensions"] = episode.layer_dimensions
    elif layer_dimensions != episode.layer_dimensions:
        raise ValueError(
            "demonstration layer dimensions cannot change in one "
            "dataset"
        )
    observation_schemas = manifest.get(
        "observation_schema_versions"
    )
    if observation_schemas is None:
        manifest["observation_schema_versions"] = dict(
            episode.observation_schema_versions
        )
    elif observation_schemas != dict(
        episode.observation_schema_versions
    ):
        raise ValueError(
            "demonstration observation schemas cannot change in one "
            "dataset"
        )

    episodes_root = path.parent / "episodes"
    episodes_root.mkdir(parents=True, exist_ok=True)
    episode_path = episodes_root / f"{episode.episode_id}.json"
    if episode_path.exists():
        raise FileExistsError(episode_path)
    episode_payload = episode.to_dict()
    episode_temporary = episode_path.with_suffix(".json.tmp")
    episode_temporary.write_text(
        json.dumps(
            episode_payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    episode_temporary.replace(episode_path)
    relative_file = episode_path.relative_to(path.parent).as_posix()
    entries.append(
        _manifest_episode_record(
            episode,
            episode_file=relative_file,
            episode_file_sha256=_file_sha256(episode_path),
        )
    )
    manifest["manifest_revision"] = int(
        manifest.get("manifest_revision", 0)
    ) + 1
    manifest["episode_count"] = len(entries)
    manifest["safe_episode_count"] = sum(
        bool(
            row.get("feasibility", {}).get(
                "safe_demonstration",
                False,
            )
        )
        for row in entries
    )
    manifest["trajectory_count"] = {
        layer: int(
            sum(
                row["trajectory_count"][layer]
                for row in entries
            )
        )
        for layer in DEMONSTRATION_LAYERS
    }
    manifest["trajectory_count"]["total"] = int(
        sum(
            manifest["trajectory_count"][layer]
            for layer in DEMONSTRATION_LAYERS
        )
    )
    manifest["manifest_content_sha256"] = _canonical_json_sha256(
        {
            key: value
            for key, value in manifest.items()
            if key != "manifest_content_sha256"
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return manifest


def load_demonstration_split(
    manifest_path: str | Path,
    split: str,
    *,
    require_safe: bool = True,
) -> dict:
    """严格加载 train/validation；final_test 永不作为预训练数据返回。"""
    selected_split = str(split).strip().lower()
    if selected_split not in TRAINING_SPLITS:
        raise ValueError(
            "offline pretraining may load only train or validation "
            "demonstrations"
        )
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version")
        != SAFE_DEMONSTRATION_DATASET_SCHEMA_VERSION
    ):
        raise ValueError(
            "safe demonstration dataset schema mismatch"
        )
    if (
        manifest.get("generator_policy")
        != SAFE_DEMONSTRATION_GENERATOR_POLICY
    ):
        raise ValueError(
            "safe demonstration generator policy mismatch"
        )
    expected_manifest_hash = manifest.get(
        "manifest_content_sha256"
    )
    actual_manifest_hash = _canonical_json_sha256(
        {
            key: value
            for key, value in manifest.items()
            if key != "manifest_content_sha256"
        }
    )
    if expected_manifest_hash != actual_manifest_hash:
        raise ValueError(
            "safe demonstration manifest hash mismatch"
        )
    seed_split = StrictSeedSplit.from_dict(
        manifest["seed_split"]
    )
    standard = DemonstrationSafetyStandard.from_dict(
        manifest["safety_standard"]
    )
    episodes = []
    transitions = {
        layer: [] for layer in DEMONSTRATION_LAYERS
    }
    for record in manifest.get("episodes", []):
        if str(record.get("split")) != selected_split:
            continue
        episode_path = (
            path.parent / str(record["episode_file"])
        ).resolve()
        try:
            episode_path.relative_to(path.parent)
        except ValueError as exc:
            raise ValueError(
                "demonstration episode path escapes dataset root"
            ) from exc
        if _file_sha256(episode_path) != str(
            record["episode_file_sha256"]
        ):
            raise ValueError(
                "demonstration episode file hash mismatch"
            )
        payload = json.loads(
            episode_path.read_text(encoding="utf-8")
        )
        episode = SafeDemonstrationEpisode.from_dict(
            payload,
            safety_standard=standard,
        )
        expected_record_values = {
            "generator_policy": episode.generator_policy,
            "heuristic_id": episode.heuristic_id,
            "heuristic_source": episode.heuristic_source,
            "heuristic_version": episode.heuristic_version,
            "workflow_seed": episode.workflow_seed,
            "resource_seed": episode.resource_seed,
            "ddl_setting": episode.ddl_setting,
            "fuzzy_parameters": episode.fuzzy_parameters,
            "observation_schema_versions": (
                episode.observation_schema_versions
            ),
            "fuzzy_energy_score": float(
                episode.episode_metrics["fuzzy_energy_score"]
            ),
            "trajectory_count": episode.trajectory_count,
            "layer_dimensions": episode.layer_dimensions,
        }
        for field, expected in expected_record_values.items():
            if record.get(field) != expected:
                raise ValueError(
                    "demonstration manifest/episode mismatch: "
                    f"{field}"
                )
        if (
            seed_split.split_for(
                episode.workflow_seed,
                episode.resource_seed,
            )
            != selected_split
        ):
            raise ValueError(
                "demonstration episode violates strict seed split"
            )
        if (
            bool(
                record.get("feasibility", {}).get(
                    "safe_demonstration",
                    False,
                )
            )
            != episode.safe_demonstration
        ):
            raise ValueError(
                "manifest safe demonstration label mismatch"
            )
        if require_safe and not episode.safe_demonstration:
            continue
        episodes.append(episode)
        for layer in DEMONSTRATION_LAYERS:
            transitions[layer].extend(
                episode.trajectories[layer]
            )
    if not episodes:
        raise ValueError(
            f"no eligible {selected_split} demonstration episodes"
        )
    return {
        "manifest": manifest,
        "seed_split": seed_split,
        "safety_standard": standard,
        "episodes": tuple(episodes),
        "transitions": {
            layer: tuple(transitions[layer])
            for layer in DEMONSTRATION_LAYERS
        },
    }


__all__ = [
    "ALL_SPLITS",
    "DEMONSTRATION_LAYERS",
    "SAFE_DEMONSTRATION_DATASET_SCHEMA_VERSION",
    "SAFE_DEMONSTRATION_EPISODE_SCHEMA_VERSION",
    "SAFE_DEMONSTRATION_GENERATOR_POLICY",
    "SafeDemonstrationEpisode",
    "DemonstrationSafetyStandard",
    "StrictSeedSplit",
    "append_demonstration_episode",
    "load_demonstration_split",
]
