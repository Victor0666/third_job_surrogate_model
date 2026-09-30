"""Configuration and scenario construction for ``drlea_nichgp``."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from algorithms.llm_safe_hrl.scenario_registry import (
    FINAL_TEST_SEEDS,
    ExperimentProtocolContext,
    SCENARIO_REGISTRY,
    resolve_experiment_protocol,
)
from project_paths import PROJECT_ROOT
from algorithms.llm_safe_hrl.hrl_mix.train_config import (
    parse_deadline_cache_overrides,
)

from . import METHOD_ID, SCHEMA_VERSION


DDL_NAMES = {
    "T": "Tight",
    "TIGHT": "Tight",
    "M": "Medium",
    "MEDIUM": "Medium",
    "L": "Loose",
    "LOOSE": "Loose",
}
DDL_SMALL_PROBABILITY = {
    "Tight": 0.8,
    "Medium": 0.5,
    "Loose": 0.2,
}
REWARD_MODES = (
    "completion_only",
    "deadline_energy",
    "deadline_energy_slack",
)


def ensure_disjoint_seeds(
    train_seeds,
    validation_seeds,
    test_seeds,
) -> None:
    groups = [
        set(map(int, train_seeds)),
        set(map(int, validation_seeds)),
        set(map(int, test_seeds)),
    ]
    reserved_leak = (groups[0] | groups[1]) & set(FINAL_TEST_SEEDS)
    if reserved_leak:
        raise ValueError(
            "paper final-test seeds 201-230 are reserved for frozen "
            f"evaluation: {sorted(reserved_leak)}"
        )
    if (
        groups[0] & groups[1]
        or groups[0] & groups[2]
        or groups[1] & groups[2]
    ):
        raise ValueError(
            "train, validation and test seeds must be strictly disjoint"
        )


@dataclass(frozen=True)
class AgentConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    hidden_dims: tuple[int, int] = (128, 128)
    replay_capacity: int = 50_000
    batch_size: int = 64


@dataclass(frozen=True)
class GPConfig:
    population_size: int = 80
    generations: int = 20
    archive_size: int = 4
    tournament_size: int = 4
    elite_size: int = 4
    crossover_rate: float = 0.8
    mutation_rate: float = 0.15
    max_depth: int = 6
    decision_ready_threshold: int = 6
    target_situations: int = 20
    clearing_radius: float = 0.0
    clearing_capacity: int = 1


@dataclass(frozen=True)
class RewardConfig:
    mode: str = "deadline_energy"
    alpha_deadline: float = 1.0
    beta_energy: float = 0.05
    slack_weight: float = 0.1
    energy_normalizer: float = 1_000.0
    time_normalizer: float = 300.0

    def __post_init__(self) -> None:
        if self.mode not in REWARD_MODES:
            raise ValueError(f"Unsupported reward mode: {self.mode}")


@dataclass(frozen=True)
class NormalizationConfig:
    workload_mi: float = 150_000.0
    data_bits: float = 1e10
    time_seconds: float = 300.0
    rank_seconds: float = 1_000.0
    remaining_work_mi: float = 3_000_000.0
    energy_joules: float = 1_000.0
    ready_count: float = 32.0


@dataclass(frozen=True)
class ComparisonConfig:
    method_id: str
    schema_version: int
    scenario: str
    ddl: str
    algorithm_seed: int
    train_seeds: tuple[int, ...]
    validation_seeds: tuple[int, ...]
    test_seeds: tuple[int, ...]
    protocol: str
    source_scenario: str | None
    resource_scale: str
    training_scenarios: tuple[str, ...]
    test_scenarios: tuple[str, ...]
    workflows_per_episode: int
    ra_episodes: int
    sa_episodes: int
    validation_interval: int
    allow_busy_vm_queueing: bool
    arrival_lambda: float
    horizon: float
    fuzzy_delta1: float
    fuzzy_delta2: float
    fuzzy_deadline_eta: float
    fuzzy_energy_lambda: float
    deadline_alpha_small: float
    deadline_alpha_large: float
    deadline_alpha_small_prob: float
    dax_paths: tuple[str, ...]
    deadline_cache_path: str
    deadline_cache_paths: Mapping[str, str]
    num_cloud_hosts: int
    num_edge_hosts: int
    cloud_vms_per_host: tuple[int, ...]
    edge_vms_per_host: tuple[int, ...]
    cloud_pc_tiers: tuple[float, ...]
    edge_pc_tiers: tuple[float, ...]
    cloud_bw_tiers: tuple[float, ...]
    edge_bw_tiers: tuple[float, ...]
    routing: AgentConfig = field(default_factory=AgentConfig)
    sequencing: AgentConfig = field(default_factory=AgentConfig)
    gp: GPConfig = field(default_factory=GPConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    normalization: NormalizationConfig = field(
        default_factory=NormalizationConfig
    )

    @property
    def output_dir(self) -> Path:
        short_ddl = self.ddl[0].upper()
        if self.protocol == "legacy":
            namespace = Path(f"{self.scenario}_{short_ddl}_a{self.algorithm_seed}")
        else:
            parent = "main_single" if self.protocol == "single" else "enhancement_multi"
            group = self.source_scenario if self.protocol == "single" else self.resource_scale
            namespace = Path(parent) / str(group) / f"{short_ddl}_a{self.algorithm_seed}"
        return PROJECT_ROOT / "out" / "comparisons" / METHOD_ID / namespace

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_TUPLE_CONFIG_FIELDS = {
    "train_seeds",
    "validation_seeds",
    "test_seeds",
    "training_scenarios",
    "test_scenarios",
    "dax_paths",
    "cloud_vms_per_host",
    "edge_vms_per_host",
    "cloud_pc_tiers",
    "edge_pc_tiers",
    "cloud_bw_tiers",
    "edge_bw_tiers",
}


def _project_data_relative_path(
    value: str | Path,
    *,
    field_name: str,
) -> str:
    """Persist project data inputs without a machine-specific root."""

    text = str(value).strip().replace(chr(92), "/")
    while text.startswith("./"):
        text = text[2:]
    if text.startswith("data/"):
        relative = text
    elif "/data/" in text:
        relative = "data/" + text.split("/data/", 1)[1]
    else:
        raise ValueError(
            f"{field_name} must be located under the project data directory: "
            f"{value}"
        )
    parts = PurePosixPath(relative).parts
    if (
        not parts
        or parts[0] != "data"
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise ValueError(f"invalid project data path for {field_name}: {value}")
    return PurePosixPath(*parts).as_posix()


def config_from_dict(payload: Mapping[str, Any]) -> ComparisonConfig:
    """Restore the exact configuration persisted beside an RA checkpoint."""

    if not isinstance(payload, Mapping):
        raise ValueError("DRL-EA config payload must be a mapping")
    expected = {item.name for item in fields(ComparisonConfig)}
    missing = sorted(expected.difference(payload))
    unknown = sorted(set(payload).difference(expected))
    if missing or unknown:
        raise ValueError(
            "incompatible DRL-EA config fields: "
            f"missing={missing}, unknown={unknown}"
        )
    values = dict(payload)
    for name in _TUPLE_CONFIG_FIELDS:
        values[name] = tuple(values[name])
    values["deadline_cache_path"] = _project_data_relative_path(
        values["deadline_cache_path"],
        field_name="deadline_cache_path",
    )
    values["deadline_cache_paths"] = {
        str(scenario): _project_data_relative_path(
            path,
            field_name=f"deadline_cache_paths[{scenario}]",
        )
        for scenario, path in dict(
            values["deadline_cache_paths"]
        ).items()
    }
    routing = dict(values["routing"])
    routing["hidden_dims"] = tuple(routing["hidden_dims"])
    sequencing = dict(values["sequencing"])
    sequencing["hidden_dims"] = tuple(sequencing["hidden_dims"])
    values["routing"] = AgentConfig(**routing)
    values["sequencing"] = AgentConfig(**sequencing)
    values["gp"] = GPConfig(**dict(values["gp"]))
    values["reward"] = RewardConfig(**dict(values["reward"]))
    values["normalization"] = NormalizationConfig(
        **dict(values["normalization"])
    )
    config = ComparisonConfig(**values)
    if config.method_id != METHOD_ID or config.schema_version != SCHEMA_VERSION:
        raise ValueError("incompatible DRL-EA config identity")
    ensure_disjoint_seeds(
        config.train_seeds,
        config.validation_seeds,
        config.test_seeds,
    )
    return config


def load_config(path: str | Path) -> ComparisonConfig:
    """Load an exact config written beside a trained RA checkpoint."""

    with Path(path).open("r", encoding="utf-8") as stream:
        return config_from_dict(json.load(stream))


PROTOCOL_ARTIFACT_FIELDS = (
    "protocol",
    "source_scenario",
    "resource_scale",
    "training_scenarios",
    "test_scenarios",
    "train_seeds",
    "validation_seeds",
    "test_seeds",
    "ddl",
    "algorithm_seed",
)


def protocol_artifact_identity(config: ComparisonConfig) -> dict[str, Any]:
    """Return the immutable training/evaluation identity for DRL-EA artifacts."""
    return {
        "protocol": str(config.protocol),
        "source_scenario": config.source_scenario,
        "resource_scale": str(config.resource_scale),
        "training_scenarios": list(config.training_scenarios),
        "test_scenarios": list(config.test_scenarios),
        "train_seeds": list(config.train_seeds),
        "validation_seeds": list(config.validation_seeds),
        "test_seeds": list(config.test_seeds),
        "ddl": str(config.ddl),
        "algorithm_seed": int(config.algorithm_seed),
    }


def validate_protocol_artifact_identity(expected, actual, *, artifact_name: str) -> dict:
    """Reject legacy, cross-protocol, cross-scenario, or cross-seed artifacts."""
    if not isinstance(actual, dict):
        raise ValueError(f"{artifact_name} is missing protocol identity")
    missing = [name for name in PROTOCOL_ARTIFACT_FIELDS if name not in actual]
    if missing:
        raise ValueError(f"{artifact_name} protocol identity is missing {missing}")
    mismatches = [
        name for name in PROTOCOL_ARTIFACT_FIELDS
        if actual[name] != expected[name]
    ]
    if mismatches:
        raise ValueError(
            f"{artifact_name} protocol identity mismatch: {', '.join(mismatches)}"
        )
    return {name: actual[name] for name in PROTOCOL_ARTIFACT_FIELDS}

def normalize_scenario(value: str) -> str:
    value = str(value).strip().upper()
    if value not in SCENARIO_REGISTRY:
        raise ValueError("scenario must be one of SS through LL")
    return value


def normalize_ddl(value: str) -> str:
    key = str(value).strip().upper()
    if key not in DDL_NAMES:
        raise ValueError("ddl must be T/M/L or Tight/Medium/Loose")
    return DDL_NAMES[key]


def build_config(
    scenario: str = "SS",
    ddl: str = "T",
    algorithm_seed: int = 0,
    *,
    workflows_per_episode: int = 50,
    ra_episodes: int = 300,
    sa_episodes: int = 300,
    reward_mode: str = "deadline_energy",
    smoke: bool = False,
    deadline_cache_path: str | None = None,
    deadline_cache_paths: Mapping[str, str] | None = None,
    protocol: str = "single",
    source_scenario: str | None = None,
    resource_scale: str | None = None,
) -> ComparisonConfig:
    """Build a comparison config from the current project's scenario tables."""

    protocol_context = None
    if str(protocol).strip().lower() != "legacy":
        protocol_context = resolve_experiment_protocol(
            protocol,
            source_scenario=(source_scenario or scenario if protocol == "single" else None),
            resource_scale=resource_scale,
            train_seeds=(1, 2, 3),
            validation_seeds=(4, 5),
        )
        scenario = protocol_context.training_scenarios[0]
    scenario = normalize_scenario(scenario)
    cache_overrides = {
        str(cache_scenario): _project_data_relative_path(
            path,
            field_name=f"deadline_cache_paths[{cache_scenario}]",
        )
        for cache_scenario, path in parse_deadline_cache_overrides(
            deadline_cache_paths
        ).items()
    }
    if deadline_cache_path is not None:
        cache_overrides[scenario] = _project_data_relative_path(
            deadline_cache_path,
            field_name="deadline_cache_path",
        )
    ddl_name = normalize_ddl(ddl)
    spec = SCENARIO_REGISTRY[scenario]
    dax_paths = tuple(
        (Path("data") / "dax" / name).as_posix()
        for name in spec.dax_files
    )
    missing = [
        path
        for path in dax_paths
        if not (PROJECT_ROOT / path).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing current-project DAX files: {missing}")
    cache_path = Path(
        cache_overrides[scenario]
        if scenario in cache_overrides
        else Path("data") / "deadlines" / "fcfs" / "diagnostic"
        / f"fcfs_fixed_{scenario}_exactmix_formal38.json"
    )
    if not (PROJECT_ROOT / cache_path).is_file():
        raise FileNotFoundError(
            f"Missing current-project deadline cache: {cache_path}"
        )

    if smoke:
        workflows_per_episode = 3
        ra_episodes = 2
        sa_episodes = 2
        routing = AgentConfig(
            hidden_dims=(32, 32),
            replay_capacity=512,
            batch_size=4,
            epsilon_decay=0.8,
        )
        sequencing = AgentConfig(
            hidden_dims=(32, 32),
            replay_capacity=512,
            batch_size=4,
            epsilon_decay=0.8,
        )
        gp_config = GPConfig(
            population_size=8,
            generations=2,
            elite_size=2,
            max_depth=4,
            decision_ready_threshold=2,
            target_situations=4,
        )
    else:
        routing = AgentConfig()
        sequencing = AgentConfig()
        gp_config = GPConfig()

    algorithm_seed = int(algorithm_seed)
    config = ComparisonConfig(
        method_id=METHOD_ID,
        schema_version=SCHEMA_VERSION,
        scenario=scenario,
        ddl=ddl_name,
        algorithm_seed=algorithm_seed,
        train_seeds=(1, 2, 3, 4, 5),
        validation_seeds=(101, 102, 103),
        test_seeds=tuple(range(201, 231)),
        protocol=(protocol_context.protocol if protocol_context is not None else "legacy"),
        source_scenario=(
            protocol_context.source_scenario if protocol_context is not None else scenario
        ),
        resource_scale=(
            protocol_context.resource_scale if protocol_context is not None else scenario[1]
        ),
        training_scenarios=(
            protocol_context.training_scenarios if protocol_context is not None else (scenario,)
        ),
        test_scenarios=(
            protocol_context.test_scenarios if protocol_context is not None else (scenario,)
        ),
        workflows_per_episode=int(workflows_per_episode),
        ra_episodes=int(ra_episodes),
        sa_episodes=int(sa_episodes),
        validation_interval=25,
        allow_busy_vm_queueing=False,
        arrival_lambda=0.03,
        horizon=1e9,
        fuzzy_delta1=0.75,
        fuzzy_delta2=1.2,
        fuzzy_deadline_eta=0.95,
        fuzzy_energy_lambda=1.0,
        deadline_alpha_small=2.0,
        deadline_alpha_large=3.0,
        deadline_alpha_small_prob=DDL_SMALL_PROBABILITY[ddl_name],
        dax_paths=dax_paths,
        deadline_cache_path=cache_path.as_posix(),
        deadline_cache_paths=cache_overrides,
        num_cloud_hosts=int(spec.num_cloud_hosts),
        num_edge_hosts=int(spec.num_edge_hosts),
        cloud_vms_per_host=tuple(spec.cloud_vms_per_host),
        edge_vms_per_host=tuple(spec.edge_vms_per_host),
        cloud_pc_tiers=tuple(spec.cloud_pc_tiers),
        edge_pc_tiers=tuple(spec.edge_pc_tiers),
        cloud_bw_tiers=tuple(spec.cloud_bw_tiers),
        edge_bw_tiers=tuple(spec.edge_bw_tiers),
        routing=routing,
        sequencing=sequencing,
        gp=gp_config,
        reward=RewardConfig(mode=reward_mode),
    )
    ensure_disjoint_seeds(
        config.train_seeds,
        config.validation_seeds,
        config.test_seeds,
    )
    return config


def config_for_scenario(config: ComparisonConfig, scenario: str) -> ComparisonConfig:
    """Switch only environment inputs while retaining frozen algorithm state."""
    target = build_config(
        scenario,
        config.ddl,
        config.algorithm_seed,
        workflows_per_episode=config.workflows_per_episode,
        ra_episodes=config.ra_episodes,
        sa_episodes=config.sa_episodes,
        reward_mode=config.reward.mode,
        deadline_cache_paths=config.deadline_cache_paths,
        protocol="legacy",
    )
    environment_fields = (
        "scenario", "dax_paths", "deadline_cache_path", "num_cloud_hosts",
        "num_edge_hosts", "cloud_vms_per_host", "edge_vms_per_host",
        "cloud_pc_tiers", "edge_pc_tiers", "cloud_bw_tiers", "edge_bw_tiers",
    )
    return replace(
        config,
        training_scenarios=(target.scenario,),
        **{name: getattr(target, name) for name in environment_fields},
    )
