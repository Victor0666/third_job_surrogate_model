"""Centralized and auditable protocol for fuzzy comparison algorithms."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import LLM_SAFE_HRL_ROOT  # noqa: F401  Ensures legacy imports resolve.
from hrl_mix.train_config import (
    TrainConfig,
    build_train_config,
    normalize_ddl,
    parse_deadline_cache_overrides,
)
from algorithms.llm_safe_hrl.scenario_registry import (
    FINAL_TEST_SEEDS,
    ExperimentProtocolContext,
)


PROTOCOL_SCHEMA_VERSION = 1
DDL_SMALL_PROBABILITY = {
    "Tight": 0.8,
    "Medium": 0.5,
    "Loose": 0.2,
}


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _unique_non_negative(values: Sequence[int], name: str) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if not result or any(value < 0 for value in result):
        raise ValueError(f"{name} must contain non-negative seeds")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


@dataclass(frozen=True)
class FuzzyComparisonProtocol:
    """Frozen inputs shared by IRWS, MARL and PD3QN.

    Model selection is never performed on ``test_seeds``. The three methods
    receive identical workflows, arrivals, resources, fuzzy parameters and
    DDL generation for a given seed.
    """

    scenario: str
    ddl_setting: str
    train_seeds: tuple[int, ...]
    validation_seeds: tuple[int, ...]
    test_seeds: tuple[int, ...]
    deadline_cache_path: str | None = None
    deadline_cache_paths: Mapping[str, str] = field(default_factory=dict)
    protocol_mode: str = "legacy"
    source_scenario: str | None = None
    resource_scale: str | None = None
    training_scenarios: tuple[str, ...] = ()
    test_scenarios: tuple[str, ...] = ()
    workflows_per_episode: int = 50
    fuzzy_delta1: float = 0.75
    fuzzy_delta2: float = 1.2
    fuzzy_energy_uncertainty_weight: float = 1.0
    fuzzy_deadline_eta: float = 0.95
    schema_version: int = PROTOCOL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        scenario = str(self.scenario).strip().upper()
        if len(scenario) != 2 or any(value not in "SML" for value in scenario):
            raise ValueError("scenario must be one of SS, SM, SL, MS, MM, ML, LS, LM, LL")
        object.__setattr__(self, "scenario", scenario)
        cache_paths = parse_deadline_cache_overrides(self.deadline_cache_paths)
        if self.deadline_cache_path is not None:
            cache_paths.setdefault(scenario, str(self.deadline_cache_path))
        object.__setattr__(self, "deadline_cache_paths", cache_paths)
        mode = str(self.protocol_mode).strip().lower()
        if mode not in {"legacy", "single", "multi"}:
            raise ValueError("protocol_mode must be legacy, single, or multi")
        object.__setattr__(self, "protocol_mode", mode)
        training = tuple(self.training_scenarios) or (scenario,)
        testing = tuple(self.test_scenarios) or (scenario,)
        for name, values in (("training_scenarios", training), ("test_scenarios", testing)):
            normalized = tuple(str(value).strip().upper() for value in values)
            if any(value not in {
                "SS", "MS", "LS", "SM", "MM", "LM", "SL", "ML", "LL"
            } for value in normalized):
                raise ValueError(f"invalid {name}")
            object.__setattr__(self, name, normalized)
        source = self.source_scenario or scenario
        if mode == "single" and training != (source,):
            raise ValueError("Single baseline training must use only source scenario")
        object.__setattr__(self, "source_scenario", source)
        object.__setattr__(self, "resource_scale", self.resource_scale or scenario[1])
        ddl_setting = normalize_ddl(self.ddl_setting)
        object.__setattr__(self, "ddl_setting", ddl_setting)
        for name in ("train_seeds", "validation_seeds", "test_seeds"):
            object.__setattr__(
                self,
                name,
                _unique_non_negative(getattr(self, name), name),
            )
        reserved = set(FINAL_TEST_SEEDS)
        leaked = reserved.intersection(self.train_seeds).union(
            reserved.intersection(self.validation_seeds)
        )
        if leaked:
            raise ValueError(
                "paper final-test seeds 201-230 are reserved for the "
                f"frozen final-evaluation runner: {sorted(leaked)}"
            )
        all_roles = {
            "train": set(self.train_seeds),
            "validation": set(self.validation_seeds),
            "test": set(self.test_seeds),
        }
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
            overlap = all_roles[left].intersection(all_roles[right])
            if overlap:
                raise ValueError(f"{left}/{right} seed leakage: {sorted(overlap)}")
        if int(self.workflows_per_episode) <= 0:
            raise ValueError("workflows_per_episode must be positive")
        for name in (
            "fuzzy_delta1",
            "fuzzy_delta2",
            "fuzzy_energy_uncertainty_weight",
            "fuzzy_deadline_eta",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if not 0.0 <= float(self.fuzzy_deadline_eta) <= 1.0:
            raise ValueError("fuzzy_deadline_eta must be in [0, 1]")

    def for_scenario(self, scenario: str) -> "FuzzyComparisonProtocol":
        """Return a read-only evaluation view with identical frozen seed roles."""
        target = str(scenario).strip().upper()
        return replace(
            self,
            scenario=target,
            deadline_cache_path=self.deadline_cache_paths.get(target),
        )

    @property
    def ddl_small_probability(self) -> float:
        return float(DDL_SMALL_PROBABILITY[self.ddl_setting])

    @property
    def protocol_hash(self) -> str:
        return _sha256(
            {
                "protocol": asdict(self),
                "environment": self.environment_manifest(),
            }
        )

    def seeds_for_split(self, split: str) -> tuple[int, ...]:
        key = str(split).strip().lower()
        mapping = {
            "train": self.train_seeds,
            "training": self.train_seeds,
            "validation": self.validation_seeds,
            "test": self.test_seeds,
            "final_test": self.test_seeds,
        }
        if key not in mapping:
            raise ValueError("split must be train, validation or final_test")
        return mapping[key]

    def assert_not_test_seed(self, seed: int, purpose: str) -> None:
        if int(seed) in set(FINAL_TEST_SEEDS) or int(seed) in set(self.test_seeds):
            raise ValueError(
                f"final-test seed {int(seed)} cannot be used for {purpose}"
            )

    def train_config(
        self,
        seed: int,
        *,
        max_episodes: int = 1,
        require_deadline_cache: bool = True,
    ) -> TrainConfig:
        config = build_train_config(
            scenario=self.scenario,
            ddl=self.ddl_setting,
            max_episodes=max_episodes,
            safe_rl_enabled=False,
            require_deadline_cache=bool(require_deadline_cache),
            deadline_cache_override=self.deadline_cache_path,
        )
        return replace(
            config,
            random_seed=int(seed),
            workflows_per_episode=int(self.workflows_per_episode),
            deadline_alpha_small_prob=self.ddl_small_probability,
            eval_seeds=tuple(self.validation_seeds),
        )

    def environment_kwargs(
        self,
        seed: int,
        *,
        require_deadline_cache: bool = True,
    ) -> dict[str, Any]:
        config = self.train_config(
            seed,
            require_deadline_cache=require_deadline_cache,
        )
        return {
            "dax_paths": list(config.dax_list),
            "horizon": float(config.horizon),
            "arrival_lambda": float(config.arrival_lambda),
            "random_seed": int(seed),
            "max_ready_tasks": config.max_ready_tasks,
            "normalize": bool(config.normalize_obs),
            "num_cloud_hosts": int(config.num_cloud_hosts),
            "num_edge_hosts": int(config.num_edge_hosts),
            "cloud_vms_per_host": tuple(config.cloud_vms_per_host),
            "edge_vms_per_host": tuple(config.edge_vms_per_host),
            "cloud_pc_tiers": tuple(config.cloud_pc_tiers),
            "edge_pc_tiers": tuple(config.edge_pc_tiers),
            "cloud_bw_tiers": tuple(config.cloud_bw_tiers),
            "edge_bw_tiers": tuple(config.edge_bw_tiers),
            "workflows_per_episode": int(config.workflows_per_episode),
            "fuzzy_enabled": True,
            "fuzzy_delta1": float(self.fuzzy_delta1),
            "fuzzy_delta2": float(self.fuzzy_delta2),
            "fuzzy_energy_uncertainty_weight": float(
                self.fuzzy_energy_uncertainty_weight
            ),
            "fuzzy_deadline_eta": float(self.fuzzy_deadline_eta),
            "fuzzy_resource_seed": int(seed),
            "fuzzy_use_deadline_constraint": True,
            "safe_rl_enabled": False,
            "safe_rl_shield_enabled": False,
            "safe_rl_state_enabled": False,
            "deadline_mode": "cache_fcfs",
            "deadline_cache_path": str(config.deadline_cache_path),
            "deadline_cache_strict": True,
            "deadline_alpha_small": float(config.deadline_alpha_small),
            "deadline_alpha_large": float(config.deadline_alpha_large),
            "deadline_alpha_small_prob": self.ddl_small_probability,
            "manager_alpha_delay": float(config.manager_alpha_delay),
            "manager_delay_mode": str(config.manager_delay_mode),
            "scenario_code": str(config.scenario),
            "task_code": str(config.task_code),
            "resource_code": str(config.res_code),
            "workflow_families": tuple(config.workflow_families),
        }

    def environment_manifest(self) -> dict[str, Any]:
        config = self.train_config(self.train_seeds[0])
        payload = {
            "workflow_files": [
                str(Path(value).resolve()) for value in config.dax_list
            ],
            "workflow_families": list(config.workflow_families),
            "deadline_cache_path": str(Path(config.deadline_cache_path).resolve()),
            "deadline_alpha_small": float(config.deadline_alpha_small),
            "deadline_alpha_large": float(config.deadline_alpha_large),
            "deadline_alpha_small_probability": self.ddl_small_probability,
            "arrival_lambda": float(config.arrival_lambda),
            "horizon": float(config.horizon),
            "resource_topology": {
                "num_cloud_hosts": int(config.num_cloud_hosts),
                "num_edge_hosts": int(config.num_edge_hosts),
                "cloud_vms_per_host": list(config.cloud_vms_per_host),
                "edge_vms_per_host": list(config.edge_vms_per_host),
                "cloud_pc_tiers": list(config.cloud_pc_tiers),
                "edge_pc_tiers": list(config.edge_pc_tiers),
                "cloud_bw_tiers": list(config.cloud_bw_tiers),
                "edge_bw_tiers": list(config.edge_bw_tiers),
            },
            "fuzzy_resource_seed_mode": "episode_seed",
            "safe_rl_enabled": False,
        }
        return {**payload, "environment_hash": _sha256(payload)}

    def to_manifest(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "environment": self.environment_manifest(),
            "protocol_hash": self.protocol_hash,
        }


def load_protocol_config(path: str | Path) -> Mapping[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("comparison config must be a JSON object")
    return payload


def protocol_from_config(
    payload: Mapping[str, Any],
    *,
    scenario: str,
    ddl: str,
    workflows_per_episode: int | None = None,
    experiment_context: ExperimentProtocolContext | None = None,
    deadline_cache_path: str | Path | None = None,
    deadline_cache_paths: Mapping[str, str] | None = None,
) -> FuzzyComparisonProtocol:
    seeds = payload.get("seeds", {})
    fuzzy = payload.get("fuzzy", {})
    workload = payload.get("workload", {})
    return FuzzyComparisonProtocol(
        scenario=scenario,
        ddl_setting=ddl,
        train_seeds=tuple(seeds.get("training", (1, 2, 3, 4, 5))),
        validation_seeds=tuple(seeds.get("validation", (101, 102, 103))),
        test_seeds=tuple(seeds.get("final_test", range(201, 231))),
        deadline_cache_path=(
            str(deadline_cache_path)
            if deadline_cache_path is not None
            else None
        ),
        deadline_cache_paths=dict(deadline_cache_paths or {}),
        protocol_mode=(
            experiment_context.protocol if experiment_context is not None else "legacy"
        ),
        source_scenario=(
            experiment_context.source_scenario if experiment_context is not None else scenario
        ),
        resource_scale=(
            experiment_context.resource_scale if experiment_context is not None else scenario[1]
        ),
        training_scenarios=(
            experiment_context.training_scenarios if experiment_context is not None else (scenario,)
        ),
        test_scenarios=(
            experiment_context.test_scenarios if experiment_context is not None else (scenario,)
        ),
        workflows_per_episode=int(
            workflows_per_episode
            if workflows_per_episode is not None
            else workload.get("workflows_per_episode", 50)
        ),
        fuzzy_delta1=float(fuzzy.get("delta1", 0.75)),
        fuzzy_delta2=float(fuzzy.get("delta2", 1.2)),
        fuzzy_energy_uncertainty_weight=float(fuzzy.get("energy_lambda", 1.0)),
        fuzzy_deadline_eta=float(fuzzy.get("deadline_eta", 0.95)),
    )


__all__ = [
    "DDL_SMALL_PROBABILITY",
    "FuzzyComparisonProtocol",
    "PROTOCOL_SCHEMA_VERSION",
    "load_protocol_config",
    "protocol_from_config",
]
