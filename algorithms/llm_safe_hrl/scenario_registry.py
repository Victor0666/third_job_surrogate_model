"""Canonical scenario, resource, and experiment-protocol definitions."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import random
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from project_paths import PROJECT_ROOT


_SIZE_NAMES = {"S": "small", "M": "med", "L": "large"}
WORKLOAD_CATEGORY_REGISTRY = MappingProxyType({
    "30": ("CyberShake_30.xml", "Epigenomics_24.xml", "Ligo_30.xml", "Montage_25.xml", "Sipht_29.xml"),
    "50": ("CyberShake_50.xml", "Epigenomics_47.xml", "Ligo_50.xml", "Montage_50.xml", "Sipht_58.xml"),
    "100": ("CyberShake_100.xml", "Epigenomics_100.xml", "Ligo_100.xml", "Montage_100.xml", "Sipht_97.xml"),
})
TASK_SCALE_CATEGORY_MIX = MappingProxyType({
    "S": MappingProxyType({"30": 1.0}),
    "M": MappingProxyType({"30": 0.3, "50": 0.7}),
    "L": MappingProxyType({"30": 0.2, "50": 0.3, "100": 0.5}),
})
TASK_DAX_FILES = MappingProxyType({
    code: tuple(
        dax
        for category in mix
        for dax in WORKLOAD_CATEGORY_REGISTRY[category]
    )
    for code, mix in TASK_SCALE_CATEGORY_MIX.items()
})
DEFAULT_PC_TIERS = (1.0, 2.0, 4.0, 6.0, 8.0)
DEFAULT_BW_TIERS = (1000.0, 2000.0, 4000.0, 6000.0, 8000.0)
LLM_OFFLINE_TRAIN_SEEDS = (1, 2, 3)
LLM_OFFLINE_VALIDATION_SEEDS = (4, 5)
SAFE_HRL_TRAIN_SEEDS = (1, 2, 3, 4, 5)
SAFE_HRL_VALIDATION_SEEDS = (101, 102, 103)
FINAL_TEST_SEEDS = tuple(range(201, 231))
_IDENTITY_FIELDS = (
    "protocol", "source_scenario", "training_scenarios", "test_scenarios",
    "llm_train_seeds", "llm_validation_seeds",
    "safe_hrl_train_seeds", "safe_hrl_validation_seeds",
    "final_test_seeds",
)


def _normalize_scenario(value: str) -> str:
    scenario = str(value).strip().upper()
    if len(scenario) != 2 or any(code not in _SIZE_NAMES for code in scenario):
        raise ValueError("scenario must be one of SS, MS, LS, SM, MM, LM, SL, ML, LL")
    return scenario


def _normalize_scenarios(values: Sequence[str], label: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{label} must be a sequence of scenario IDs")
    scenarios = tuple(_normalize_scenario(value) for value in values)
    if not scenarios or len(set(scenarios)) != len(scenarios):
        raise ValueError(f"{label} must be non-empty and contain no duplicates")
    return scenarios


def _normalize_seeds(values: Sequence[int], label: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{label} must be a sequence of integer seeds")
    seeds = tuple(int(value) for value in values)
    if not seeds or any(seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError(f"{label} must be non-empty, non-negative, and unique")
    return seeds


def assert_no_final_test_seed(values: Sequence[int], component: str) -> tuple[int, ...]:
    """Reject paper final-test seeds outside the frozen evaluation runner."""
    seeds = tuple(int(value) for value in values)
    leaked = sorted(set(seeds).intersection(FINAL_TEST_SEEDS))
    if leaked:
        raise ValueError(
            f"{component} cannot access paper final-test seeds {leaked}; "
            "only the frozen final-evaluation runner may use 201-230"
        )
    return seeds


def workload_category_counts(task_code: str, count: int) -> dict[str, int]:
    """Allocate one episode by largest remainder using the explicit mix."""
    code = str(task_code).strip().upper()
    if code not in TASK_SCALE_CATEGORY_MIX:
        raise ValueError("task_code must be S, M, or L")
    total = int(count)
    if total < 0:
        raise ValueError("workload count must be non-negative")
    mix = TASK_SCALE_CATEGORY_MIX[code]
    raw = {category: total * probability for category, probability in mix.items()}
    result = {category: int(value) for category, value in raw.items()}
    remaining = total - sum(result.values())
    order = sorted(
        mix,
        key=lambda category: (-(raw[category] - result[category]), category),
    )
    for category in order[:remaining]:
        result[category] += 1
    return result


def deterministic_workload_sequence(
    task_code: str,
    environment_seed: int,
    count: int,
) -> tuple[str, ...]:
    """Build one exact-quota episode from the explicit workload registry.

    The category counts use a deterministic largest-remainder allocation. The
    environment seed alone chooses concrete DAX files within each category and
    shuffles the episode; algorithm/optimizer seeds are deliberately absent.
    Category membership is never inferred from actual DAG node count.
    """
    code = str(task_code).strip().upper()
    total = int(count)
    if code not in TASK_SCALE_CATEGORY_MIX:
        raise ValueError("task_code must be S, M, or L")
    if total < 0:
        raise ValueError("workload count must be non-negative")
    counts = workload_category_counts(code, total)
    rng = np.random.RandomState(int(environment_seed))
    episode: list[str] = []
    for category, category_count in counts.items():
        choices = WORKLOAD_CATEGORY_REGISTRY[category]
        episode.extend(
            choices[int(index)]
            for index in rng.randint(len(choices), size=category_count)
        )
    if len(episode) != total:
        raise RuntimeError("workload category allocation produced wrong size")
    if episode and len(counts) > 1:
        order = rng.permutation(len(episode))
        episode = [episode[int(index)] for index in order]
    return tuple(episode)

@dataclass(frozen=True)
class ResourceScaleSpec:
    """Canonical Host/VM scale with an explicit, preserved Host type layout."""

    resource_code: str
    resource_size: str
    vms_per_host: tuple[int, ...]
    host_types: tuple[str, ...]
    cloud_pc_tiers: tuple[float, ...] = DEFAULT_PC_TIERS
    edge_pc_tiers: tuple[float, ...] = DEFAULT_PC_TIERS
    cloud_bw_tiers: tuple[float, ...] = DEFAULT_BW_TIERS
    edge_bw_tiers: tuple[float, ...] = DEFAULT_BW_TIERS

    def __post_init__(self) -> None:
        if self.resource_code not in _SIZE_NAMES:
            raise ValueError("resource_code must be S, M, or L")
        if self.resource_size != _SIZE_NAMES[self.resource_code]:
            raise ValueError("resource_size does not match resource_code")
        if not self.vms_per_host or any(int(count) <= 0 for count in self.vms_per_host):
            raise ValueError("vms_per_host must contain positive VM counts")
        if len(self.host_types) != len(self.vms_per_host):
            raise ValueError("host_types must define every Host in vms_per_host")
        if any(host_type not in {"cloud", "edge"} for host_type in self.host_types):
            raise ValueError("host_types entries must be 'cloud' or 'edge'")

    @property
    def total_hosts(self) -> int:
        return len(self.vms_per_host)

    @property
    def total_vms(self) -> int:
        return sum(self.vms_per_host)

    @property
    def num_cloud_hosts(self) -> int:
        return self.host_types.count("cloud")

    @property
    def num_edge_hosts(self) -> int:
        return self.host_types.count("edge")

    @property
    def cloud_vms_per_host(self) -> tuple[int, ...]:
        return tuple(
            count for count, host_type in zip(self.vms_per_host, self.host_types)
            if host_type == "cloud"
        )

    @property
    def edge_vms_per_host(self) -> tuple[int, ...]:
        return tuple(
            count for count, host_type in zip(self.vms_per_host, self.host_types)
            if host_type == "edge"
        )

    def resource_mapping(self) -> dict[str, Any]:
        """Return only fields accepted by the existing cloud/edge constructor."""
        return {
            "num_cloud_hosts": self.num_cloud_hosts,
            "num_edge_hosts": self.num_edge_hosts,
            "cloud_vms_per_host": list(self.cloud_vms_per_host),
            "edge_vms_per_host": list(self.edge_vms_per_host),
            "cloud_pc_tiers": list(self.cloud_pc_tiers),
            "edge_pc_tiers": list(self.edge_pc_tiers),
            "cloud_bw_tiers": list(self.cloud_bw_tiers),
            "edge_bw_tiers": list(self.edge_bw_tiers),
        }

    def scale_mapping(self) -> dict[str, Any]:
        """Return auditable scale metadata without changing constructor inputs."""
        return {
            "resource_code": self.resource_code,
            "resource_size": self.resource_size,
            "num_hosts": self.total_hosts,
            "total_vms": self.total_vms,
            "vms_per_host": list(self.vms_per_host),
            "host_types": list(self.host_types),
        }


RESOURCE_SCALE_REGISTRY: Mapping[str, ResourceScaleSpec] = MappingProxyType({
    "S": ResourceScaleSpec(
        "S", "small", (9, 8, 8), ("cloud", "cloud", "edge")
    ),
    "M": ResourceScaleSpec(
        "M", "med", (9, 9, 8, 8, 8, 8),
        ("cloud", "cloud", "cloud", "edge", "edge", "edge"),
    ),
    "L": ResourceScaleSpec(
        "L", "large", (9, 9, 9, 8, 8, 8, 8, 8, 8),
        ("cloud", "cloud", "cloud", "cloud", "cloud", "edge", "edge", "edge", "edge"),
    ),
})


@dataclass(frozen=True)
class ScenarioSpec:
    """Immutable task, resource, and deadline-cache inputs for one scenario."""

    scenario_id: str
    task_code: str
    resource_code: str
    task_size: str
    resource_size: str
    dax_files: tuple[str, ...]

    @property
    def resource_scale(self) -> ResourceScaleSpec:
        return RESOURCE_SCALE_REGISTRY[self.resource_code]

    @property
    def num_cloud_hosts(self) -> int:
        return self.resource_scale.num_cloud_hosts

    @property
    def num_edge_hosts(self) -> int:
        return self.resource_scale.num_edge_hosts

    @property
    def cloud_vms_per_host(self) -> tuple[int, ...]:
        return self.resource_scale.cloud_vms_per_host

    @property
    def edge_vms_per_host(self) -> tuple[int, ...]:
        return self.resource_scale.edge_vms_per_host

    @property
    def cloud_pc_tiers(self) -> tuple[float, ...]:
        return self.resource_scale.cloud_pc_tiers

    @property
    def edge_pc_tiers(self) -> tuple[float, ...]:
        return self.resource_scale.edge_pc_tiers

    @property
    def cloud_bw_tiers(self) -> tuple[float, ...]:
        return self.resource_scale.cloud_bw_tiers

    @property
    def edge_bw_tiers(self) -> tuple[float, ...]:
        return self.resource_scale.edge_bw_tiers

    @property
    def total_hosts(self) -> int:
        return self.resource_scale.total_hosts

    @property
    def total_vms(self) -> int:
        return self.resource_scale.total_vms

    @property
    def vms_per_host(self) -> tuple[int, ...]:
        return self.resource_scale.vms_per_host

    @property
    def host_types(self) -> tuple[str, ...]:
        return self.resource_scale.host_types

    @property
    def deadline_cache_relative_path(self) -> str:
        return (
            "data/deadlines/fcfs/exact_mix_v1/"
            f"fcfs_{self.task_size}Task_{self.resource_size}Res_"
            "exactmix_formal38.json"
        )

    def deadline_cache_path(self, project_root: str | Path = PROJECT_ROOT) -> Path:
        return Path(project_root).resolve() / self.deadline_cache_relative_path

    def resource_mapping(self) -> dict[str, Any]:
        return self.resource_scale.resource_mapping()


def _build_registry() -> Mapping[str, ScenarioSpec]:
    registry: dict[str, ScenarioSpec] = {}
    for resource_code in ("S", "M", "L"):
        for task_code in ("S", "M", "L"):
            scenario_id = task_code + resource_code
            resource = RESOURCE_SCALE_REGISTRY[resource_code]
            registry[scenario_id] = ScenarioSpec(
                scenario_id=scenario_id,
                task_code=task_code,
                resource_code=resource_code,
                task_size=_SIZE_NAMES[task_code],
                resource_size=_SIZE_NAMES[resource_code],
                dax_files=TASK_DAX_FILES[task_code],
            )
    return MappingProxyType(registry)


SCENARIO_REGISTRY: Mapping[str, ScenarioSpec] = _build_registry()


@dataclass(frozen=True)
class ExperimentProtocolContext:
    """Resolved Single or Multi protocol identity and artifact namespace."""

    protocol: str
    source_scenario: str | None
    resource_scale: str
    training_scenarios: tuple[str, ...]
    test_scenarios: tuple[str, ...]
    llm_train_seeds: tuple[int, ...]
    llm_validation_seeds: tuple[int, ...]
    safe_hrl_train_seeds: tuple[int, ...]
    safe_hrl_validation_seeds: tuple[int, ...]
    final_test_seeds: tuple[int, ...]

    @property
    def train_seeds(self) -> tuple[int, ...]:
        """Backward-compatible alias for LLM offline training seeds."""
        return self.llm_train_seeds

    @property
    def validation_seeds(self) -> tuple[int, ...]:
        """Backward-compatible alias for LLM offline validation seeds."""
        return self.llm_validation_seeds

    @property
    def artifact_group(self) -> str:
        return str(self.source_scenario) if self.protocol == "single" else self.resource_scale

    @property
    def artifact_output_root(self) -> Path:
        parent = "main_single" if self.protocol == "single" else "enhancement_multi"
        return PROJECT_ROOT / "out" / parent / self.artifact_group

    @property
    def checkpoint_root(self) -> Path:
        parent = "main_single" if self.protocol == "single" else "enhancement_multi"
        return PROJECT_ROOT / "checkpoints" / parent / self.artifact_group

    @property
    def library_filename(self) -> str:
        if self.protocol == "single":
            return f"safe_heuristic_library_single_{self.source_scenario}.json"
        return f"safe_heuristic_library_multi_{self.resource_scale}.json"

    @property
    def library_path(self) -> Path:
        return self.artifact_output_root / self.library_filename

    def identity(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "source_scenario": self.source_scenario,
            "training_scenarios": list(self.training_scenarios),
            "test_scenarios": list(self.test_scenarios),
            "llm_train_seeds": list(self.llm_train_seeds),
            "llm_validation_seeds": list(self.llm_validation_seeds),
            "safe_hrl_train_seeds": list(self.safe_hrl_train_seeds),
            "safe_hrl_validation_seeds": list(self.safe_hrl_validation_seeds),
            "final_test_seeds": list(self.final_test_seeds),
        }


def resolve_experiment_protocol(
    protocol: str = "single",
    *,
    source_scenario: str | None = "SS",
    resource_scale: str | None = None,
    train_seeds: Sequence[int] = LLM_OFFLINE_TRAIN_SEEDS,
    validation_seeds: Sequence[int] = LLM_OFFLINE_VALIDATION_SEEDS,
    safe_hrl_train_seeds: Sequence[int] = SAFE_HRL_TRAIN_SEEDS,
    safe_hrl_validation_seeds: Sequence[int] = SAFE_HRL_VALIDATION_SEEDS,
    final_test_seeds: Sequence[int] = FINAL_TEST_SEEDS,
) -> ExperimentProtocolContext:
    """Resolve Single or Multi without changing algorithm implementations."""
    mode = str(protocol).strip().lower()
    train = _normalize_seeds(train_seeds, "train_seeds")
    validation = _normalize_seeds(validation_seeds, "validation_seeds")
    safe_train = _normalize_seeds(safe_hrl_train_seeds, "safe_hrl_train_seeds")
    safe_validation = _normalize_seeds(
        safe_hrl_validation_seeds, "safe_hrl_validation_seeds"
    )
    final_test = _normalize_seeds(final_test_seeds, "final_test_seeds")
    if set(train).intersection(validation):
        raise ValueError("train_seeds and validation_seeds must be disjoint")
    if train != LLM_OFFLINE_TRAIN_SEEDS or validation != LLM_OFFLINE_VALIDATION_SEEDS:
        raise ValueError(
            "formal LLM offline seeds are fixed to train=[1,2,3] and "
            "validation=[4,5]"
        )
    forbidden_llm = set(SAFE_HRL_VALIDATION_SEEDS).union(FINAL_TEST_SEEDS)
    if forbidden_llm.intersection(train) or forbidden_llm.intersection(validation):
        raise ValueError("LLM offline seeds must not use 101-103 or 201-230")
    safe_groups = (set(safe_train), set(safe_validation), set(final_test))
    if (
        safe_groups[0].intersection(safe_groups[1])
        or safe_groups[0].intersection(safe_groups[2])
        or safe_groups[1].intersection(safe_groups[2])
    ):
        raise ValueError(
            "Safe-HRL train, validation, and final-test seeds must be disjoint"
        )
    if (
        safe_train != SAFE_HRL_TRAIN_SEEDS
        or safe_validation != SAFE_HRL_VALIDATION_SEEDS
        or final_test != FINAL_TEST_SEEDS
    ):
        raise ValueError(
            "formal Safe-HRL seeds are fixed to train=[1,2,3,4,5], "
            "validation=[101,102,103], and final_test=[201,...,230]"
        )
    if mode == "single":
        source = _normalize_scenario(source_scenario or "")
        if source not in {"SS", "SM", "SL"}:
            raise ValueError("single source_scenario must be SS, SM, or SL")
        supplied_scale = str(resource_scale or "").strip().upper()
        if supplied_scale and supplied_scale != source[1]:
            raise ValueError("single resource_scale conflicts with source_scenario")
        training = (source,)
        testing = tuple(task_code + source[1] for task_code in ("S", "M", "L"))
        context = ExperimentProtocolContext(
            mode, source, source[1], training, testing, train, validation,
            safe_train, safe_validation, final_test,
        )
        validate_component_scenarios(context, "single", training)
        return context
    if mode == "multi":
        scale = str(resource_scale or "").strip().upper()
        if scale not in _SIZE_NAMES:
            raise ValueError("multi resource_scale must be S, M, or L")
        if source_scenario not in (None, "", "SS"):
            raise ValueError("multi protocol does not accept source_scenario")
        scenarios = tuple(task_code + scale for task_code in ("S", "M", "L"))
        return ExperimentProtocolContext(
            mode, None, scale, scenarios, scenarios, train, validation,
            safe_train, safe_validation, final_test,
        )
    raise ValueError("protocol must be 'single' or 'multi'")


def apply_scenario_to_problem_config(
    config: Mapping[str, Any],
    scenario: str,
    *,
    project_root: str | Path = PROJECT_ROOT,
    require_files: bool = False,
) -> dict[str, Any]:
    """Deep-copy config and atomically switch DAX, resources, and cache."""
    if not isinstance(config, Mapping):
        raise ValueError("problem config must be a mapping")
    spec = SCENARIO_REGISTRY[_normalize_scenario(scenario)]
    result = deepcopy(dict(config))
    dataset = dict(result.get("dataset") or {})
    dataset.update({
        "scenario": spec.scenario_id,
        "dax_files": list(spec.dax_files),
        "deadline_cache_path": spec.deadline_cache_relative_path,
    })
    result["dataset"] = dataset
    result["resources"] = spec.resource_mapping()
    if require_files:
        root = Path(project_root).resolve()
        missing = [
            str(root / "data" / "dax" / name)
            for name in spec.dax_files
            if not (root / "data" / "dax" / name).is_file()
        ]
        if missing:
            raise FileNotFoundError(f"missing scenario DAX files: {missing}")
        cache = spec.deadline_cache_path(root)
        if not cache.is_file():
            raise FileNotFoundError(f"missing scenario deadline cache: {cache}")
    return result


def validate_component_scenarios(
    context: ExperimentProtocolContext,
    component_name: str,
    scenarios: Sequence[str],
) -> tuple[str, ...]:
    """Fail closed unless a training component uses the protocol domain."""
    if not isinstance(context, ExperimentProtocolContext):
        raise ValueError("context must be an ExperimentProtocolContext")
    normalized = _normalize_scenarios(scenarios, f"{component_name or 'component'} scenarios")
    expected = tuple(context.training_scenarios)
    if context.protocol == "single" and (
        len(expected) != 1 or expected != (context.source_scenario,)
    ):
        raise ValueError("invalid Single protocol training scenario invariant")
    if normalized != expected:
        raise ValueError(
            f"{component_name or 'component'} scenarios {list(normalized)} "
            f"do not match protocol training scenarios {list(expected)}"
        )
    return normalized


def _identity_mapping(value: ExperimentProtocolContext | Mapping[str, Any]) -> dict:
    if isinstance(value, ExperimentProtocolContext):
        return value.identity()
    identity_method = getattr(value, "identity", None)
    if callable(identity_method):
        value = identity_method()
    if not isinstance(value, Mapping):
        raise ValueError("protocol identity must be a context or mapping")
    missing = [field for field in _IDENTITY_FIELDS if field not in value]
    if missing:
        raise ValueError("protocol identity is missing: " + ", ".join(missing))
    source_raw = value["source_scenario"]
    return {
        "protocol": str(value["protocol"]).strip().lower(),
        "source_scenario": None if source_raw in (None, "") else _normalize_scenario(source_raw),
        "training_scenarios": list(_normalize_scenarios(value["training_scenarios"], "training_scenarios")),
        "test_scenarios": list(_normalize_scenarios(value["test_scenarios"], "test_scenarios")),
        "llm_train_seeds": list(_normalize_seeds(value["llm_train_seeds"], "llm_train_seeds")),
        "llm_validation_seeds": list(_normalize_seeds(value["llm_validation_seeds"], "llm_validation_seeds")),
        "safe_hrl_train_seeds": list(_normalize_seeds(value["safe_hrl_train_seeds"], "safe_hrl_train_seeds")),
        "safe_hrl_validation_seeds": list(_normalize_seeds(value["safe_hrl_validation_seeds"], "safe_hrl_validation_seeds")),
        "final_test_seeds": list(_normalize_seeds(value["final_test_seeds"], "final_test_seeds")),
    }


def validate_protocol_identity(
    expected: ExperimentProtocolContext | Mapping[str, Any],
    actual: Mapping[str, Any],
    *,
    artifact_name: str = "artifact",
) -> dict[str, Any]:
    """Validate an artifact identity and reject cross-protocol reuse."""
    expected_identity = _identity_mapping(expected)
    actual_identity = _identity_mapping(actual)
    mismatches = [
        field for field in _IDENTITY_FIELDS
        if actual_identity[field] != expected_identity[field]
    ]
    if mismatches:
        raise ValueError(
            f"{artifact_name} protocol identity mismatch: " + ", ".join(mismatches)
        )
    return actual_identity


__all__ = [
    "DEFAULT_BW_TIERS", "DEFAULT_PC_TIERS", "ExperimentProtocolContext",
    "LLM_OFFLINE_TRAIN_SEEDS", "LLM_OFFLINE_VALIDATION_SEEDS",
    "SAFE_HRL_TRAIN_SEEDS", "SAFE_HRL_VALIDATION_SEEDS", "FINAL_TEST_SEEDS",
    "RESOURCE_SCALE_REGISTRY", "ResourceScaleSpec", "SCENARIO_REGISTRY",
    "ScenarioSpec", "TASK_DAX_FILES", "TASK_SCALE_CATEGORY_MIX",
    "WORKLOAD_CATEGORY_REGISTRY", "assert_no_final_test_seed",
    "deterministic_workload_sequence", "workload_category_counts",
    "apply_scenario_to_problem_config", "resolve_experiment_protocol",
    "validate_component_scenarios", "validate_protocol_identity",
]
