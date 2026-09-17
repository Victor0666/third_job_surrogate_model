# -*- coding: utf-8 -*-
"""
训练配置模块。

整体调用关系：
1. train.py 解析命令行参数后调用 train_runner.train()。
2. train_runner.py 在训练开始时调用 build_train_config()。
3. build_train_config() 根据 scenario、ddl、episodes 生成 TrainConfig。
4. train_runner.py 之后只读取 TrainConfig，不再在主训练循环里散落大量超参数。

文件职责：
- 统一管理场景命名：例如 SS、SM、LL。
- 统一管理 deadline 名称：例如 T/Tight、M/Medium、L/Loose。
- 统一管理 DAX 输入文件、资源规模、deadline cache 路径。
- 统一管理 VM、Host、Manager 三类 D3QNAgent 的超参数。
- 统一生成 checkpoint、log 等输出路径。

设计说明：
- TrainConfig 和 AgentConfig 使用 frozen=True，表示配置生成后不应在训练中被随意修改。
- 这样可以避免训练主循环里到处写局部变量，也方便复现实验参数。
"""
from __future__ import annotations

import json
import os
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from algorithms.llm_safe_hrl.paths import LLM_ROOT, PROJECT_ROOT
from algorithms.llm_safe_hrl.scenario_registry import (
    DEFAULT_BW_TIERS,
    DEFAULT_PC_TIERS,
    ExperimentProtocolContext,
    RESOURCE_SCALE_REGISTRY,
    SCENARIO_REGISTRY,
    TASK_DAX_FILES as SHARED_TASK_DAX_FILES,
    resolve_experiment_protocol,
    validate_protocol_identity,
)
from algorithms.llm_safe_hrl.run_context import (
    resolve_deadline_setting,
)
from base.heuristic_admission import (
    workflow_families_from_dax_files,
)
from output_naming import (
    legacy_hrl_run_id,
    pipeline_run_id,
    safe_hrl_run_id,
    training_output_paths,
)

# 保留 ROOT_DIR 名称供现有测试和外部脚本读取，但其语义固定为项目根目录，
# 不再依赖 hrl_mix 在文件系统中的嵌套深度。
ROOT_DIR = PROJECT_ROOT

# 场景代码中每一位的含义：
# 第一位表示任务规模，第二位表示资源规模。
# S/M/L 分别对应 small/med/large，用于拼接输入文件名和输出目录名。
SIZE_FULL = {
    "S": "small",
    "M": "med",
    "L": "large",
}

# deadline 输入允许短写和完整写法，最终统一归一化成完整英文名称。
# 例如用户传入 T 或 Tight，都会得到 Tight。
DDL_FULL = {
    alias: resolve_deadline_setting(alias).name
    for alias in ("T", "M", "L", "TIGHT", "MEDIUM", "LOOSE")
}

# 不同任务规模对应的 DAX 工作流 XML 文件集合。
# build_train_config() 会根据 scenario 第一位选择其中一组。
TASK_DAX_FILES = {
    code: list(dax_files) for code, dax_files in SHARED_TASK_DAX_FILES.items()
}

# 不同资源规模分别配置云主机和边缘主机，且保持原有总 host/VM 规模不变。
# 每个元组依次为：云主机数、边缘主机数、云主机 VM 模板、边缘主机 VM 模板。
RESOURCE_CONFIG = {
    code: (
        spec.num_cloud_hosts,
        spec.num_edge_hosts,
        spec.cloud_vms_per_host,
        spec.edge_vms_per_host,
    )
    for code, spec in RESOURCE_SCALE_REGISTRY.items()
}

# 云端和边缘端使用独立字段。当前取值一致，以保持既有实验的资源档位；
# 后续可单独修改任意一端而不会影响另一端。
CLOUD_PC_TIERS = DEFAULT_PC_TIERS
EDGE_PC_TIERS = DEFAULT_PC_TIERS
CLOUD_BW_TIERS = DEFAULT_BW_TIERS
EDGE_BW_TIERS = DEFAULT_BW_TIERS


@dataclass(frozen=True)
class AgentConfig:
    """单个 D3QNAgent 的超参数配置。"""

    lr: float
    gamma: float
    batch_size: int
    buffer_size: int
    eps_start: float
    eps_end: float
    eps_decay_steps: int
    target_update_tau: float
    grad_clip: float
    hidden_dims: tuple[int, ...]


@dataclass(frozen=True)
class SafetyShieldConfig:
    """阶段 4 模糊 DDL safety shield 配置，默认关闭。"""

    enabled: bool = False
    fallback_controller: str = "fixed_vm_rule"


@dataclass(frozen=True)
class SafetyStateConfig:
    """阶段 6 三层安全 observation 配置，默认关闭。"""

    enabled: bool = False
    high_uncertainty_threshold: float = 0.2
    recent_record_window: int = 100


@dataclass(frozen=True)
class LagrangianSafetyConfig:
    """阶段 8 episode/EMA 动态拉格朗日配置，默认关闭。"""

    enabled: bool = False
    lambda_init: float = 1.0
    lambda_lr: float = 0.01
    lambda_min: float = 0.0
    lambda_max: float = 100.0
    cost_budget: float = 0.0
    update_interval: int = 1
    cost_ema_factor: float = 0.9
    # 单位是已经完成并观测的 episode 数，不是 transition/global step。
    warmup_steps: int = 5


@dataclass(frozen=True)
class SafetyReplayConfig:
    """阶段 10 版本化安全 replay 与可选组合 PER 配置。"""

    transition_schema_version: int = 1
    near_boundary_margin: float = 1.0
    combined_per_priority: bool = False
    performance_td_weight: float = 1.0
    safety_td_weight: float = 1.0

    def __post_init__(self) -> None:
        if int(self.transition_schema_version) != 1:
            raise ValueError(
                "unsupported safe replay transition schema version"
            )
        if (
            not math.isfinite(self.near_boundary_margin)
            or self.near_boundary_margin < 0.0
        ):
            raise ValueError(
                "near_boundary_margin must be finite and non-negative"
            )
        for name, value in (
            ("performance_td_weight", self.performance_td_weight),
            ("safety_td_weight", self.safety_td_weight),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"{name} must be finite and non-negative"
                )
        if (
            self.combined_per_priority
            and self.performance_td_weight
            + self.safety_td_weight
            <= 0.0
        ):
            raise ValueError(
                "combined PER requires a positive TD weight"
            )


@dataclass(frozen=True)
class SafeManagerHeuristicConfig:
    """阶段 11 Manager 启发式选择配置，默认保留旧权重模式。"""

    mode: str = "legacy_rule_weight_mode"
    library_manifest_path: str | None = None
    recent_window: int = 20
    llm_only: bool = False

    def __post_init__(self) -> None:
        if self.mode not in {
            "legacy_rule_weight_mode",
            "heuristic_selection_mode",
        }:
            raise ValueError(
                "safe Manager mode must be legacy_rule_weight_mode "
                "or heuristic_selection_mode"
            )
        if int(self.recent_window) <= 0:
            raise ValueError(
                "safe Manager heuristic recent_window must be positive"
            )
        if (
            self.mode == "heuristic_selection_mode"
            and not self.library_manifest_path
        ):
            raise ValueError(
                "heuristic_selection_mode requires "
                "library_manifest_path"
            )
        if self.llm_only and self.mode != "heuristic_selection_mode":
            # legacy_rule_weight_mode 下 Manager 输出的是五条固定规则的权重增量，
            # 根本没有“只选 LLM 规则”这个语义，静默忽略会让实验记录失真。
            raise ValueError(
                "llm_only requires heuristic_selection_mode"
            )


@dataclass(frozen=True)
class OfflinePretrainingConfig:
    """阶段 13 安全示范离线预训练配置；默认关闭且不引入 PyTorch。"""

    enabled: bool = False
    dataset_manifest_path: str | None = None
    epochs: int = 5
    batch_size: int = 128
    performance_learning_rate: float = 1e-4
    safety_learning_rate: float = 1e-4
    train_q_r: bool = True
    train_q_c: bool = True
    behavior_cloning_enabled: bool = False
    behavior_cloning_weight: float = 0.1
    sync_targets_after_pretraining: bool = True
    random_seed: int = 17

    def __post_init__(self) -> None:
        if self.enabled and not self.dataset_manifest_path:
            raise ValueError(
                "enabled offline pretraining requires a dataset "
                "manifest"
            )
        if int(self.epochs) <= 0:
            raise ValueError(
                "offline pretraining epochs must be positive"
            )
        if int(self.batch_size) <= 0:
            raise ValueError(
                "offline pretraining batch_size must be positive"
            )
        for name, value in (
            (
                "performance_learning_rate",
                self.performance_learning_rate,
            ),
            (
                "safety_learning_rate",
                self.safety_learning_rate,
            ),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(
                    f"{name} must be finite and positive"
                )
        if (
            not math.isfinite(
                float(self.behavior_cloning_weight)
            )
            or float(self.behavior_cloning_weight) < 0.0
        ):
            raise ValueError(
                "behavior_cloning_weight must be finite and "
                "non-negative"
            )
        if self.enabled and not (
            self.train_q_r
            or self.train_q_c
            or self.behavior_cloning_enabled
        ):
            raise ValueError(
                "offline pretraining must enable Q_r, Q_c or "
                "behavior cloning"
            )


@dataclass(frozen=True)
class SafeTrainingPipelineConfig:
    """阶段 14 五阶段安全训练编排配置；仅显式提供计划文件时启用。"""

    enabled: bool = False
    plan_path: str | None = None
    resume_checkpoint_path: str | None = None

    def __post_init__(self) -> None:
        if self.enabled and not self.plan_path:
            raise ValueError(
                "enabled safe training pipeline requires plan_path"
            )
        if self.resume_checkpoint_path and not self.enabled:
            raise ValueError(
                "safe training resume requires the pipeline to be enabled"
            )


@dataclass(frozen=True)
class FeasibilityFirstModelSelectionConfig:
    """阶段 15 可行性优先 best-checkpoint 选择配置。"""

    enabled: bool = True
    # 默认正式验证要求每一个 validation seed 都满足零违反。
    require_all_validation_seeds_feasible: bool = True
    comparison_key: tuple[str, ...] = (
        "deadline_violation_rate",
        "max_fuzzy_lateness",
        "mean_fuzzy_lateness",
        "fuzzy_energy_score",
    )

    def __post_init__(self) -> None:
        expected = (
            "deadline_violation_rate",
            "max_fuzzy_lateness",
            "mean_fuzzy_lateness",
            "fuzzy_energy_score",
        )
        if tuple(self.comparison_key) != expected:
            raise ValueError(
                "feasibility-first model comparison key is frozen as "
                f"{expected}"
            )


@dataclass(frozen=True)
class SafeMetricsConfig:
    """阶段 16 安全 HRL 统一评价指标配置。"""

    enabled: bool = True
    schema_version: int = 1
    output_subdir: str = "safe_metrics"
    convergence_window: int = 5

    def __post_init__(self) -> None:
        if int(self.schema_version) != 1:
            raise ValueError(
                "unsupported safe metrics schema version"
            )
        if not str(self.output_subdir).strip():
            raise ValueError(
                "safe metrics output_subdir must be non-empty"
            )
        output_path = Path(str(self.output_subdir))
        if output_path.is_absolute() or ".." in output_path.parts:
            raise ValueError(
                "safe metrics output_subdir must be a relative "
                "child path"
            )
        if int(self.convergence_window) <= 0:
            raise ValueError(
                "safe metrics convergence_window must be positive"
            )


@dataclass(frozen=True)
class SafeRLConfig:
    """安全强化学习分阶段配置；默认不启用安全训练语义。"""

    enabled: bool = False
    safety_discount: float = 0.95
    safety_learning_rate: float = 3e-4
    safety_loss_weight: float = 1.0
    initial_lagrange_multiplier: float = 1.0
    fuzzy_energy_uncertainty_weight: float = 1.0
    fuzzy_deadline_eta: float = 0.95
    process_risk_aggregation: str = "mean"
    shield: SafetyShieldConfig = field(
        default_factory=SafetyShieldConfig
    )
    state: SafetyStateConfig = field(
        default_factory=SafetyStateConfig
    )
    lagrangian: LagrangianSafetyConfig = field(
        default_factory=LagrangianSafetyConfig
    )
    replay: SafetyReplayConfig = field(
        default_factory=SafetyReplayConfig
    )
    manager_heuristics: SafeManagerHeuristicConfig = field(
        default_factory=SafeManagerHeuristicConfig
    )
    offline_pretraining: OfflinePretrainingConfig = field(
        default_factory=OfflinePretrainingConfig
    )
    training_pipeline: SafeTrainingPipelineConfig = field(
        default_factory=SafeTrainingPipelineConfig
    )
    model_selection: FeasibilityFirstModelSelectionConfig = field(
        default_factory=FeasibilityFirstModelSelectionConfig
    )
    metrics: SafeMetricsConfig = field(
        default_factory=SafeMetricsConfig
    )


@dataclass(frozen=True)
class TrainConfig:
    """一次训练运行所需的完整配置。"""

    # Experiment identity. ``legacy`` is retained only for old Python callers
    # that do not opt into the isolated Single/Multi protocol.
    experiment_protocol: dict[str, object] | None
    protocol: str
    source_scenario: str | None
    resource_scale: str
    training_scenarios: tuple[str, ...]
    test_scenarios: tuple[str, ...]
    train_seeds: tuple[int, ...]
    validation_seeds: tuple[int, ...]
    final_test_seeds: tuple[int, ...]

    # 场景和 deadline 基本信息。
    scenario: str
    ddl_name: str
    task_code: str
    res_code: str
    task_size: str
    res_size: str

    # 输入数据和环境规模。
    dax_list: list[str]
    workflow_families: tuple[str, ...]
    num_cloud_hosts: int
    num_edge_hosts: int
    cloud_vms_per_host: tuple[int, ...]
    edge_vms_per_host: tuple[int, ...]
    cloud_pc_tiers: tuple[float, ...]
    edge_pc_tiers: tuple[float, ...]
    cloud_bw_tiers: tuple[float, ...]
    edge_bw_tiers: tuple[float, ...]
    horizon: float
    arrival_lambda: float
    random_seed: int
    optimizer_seed: int
    max_ready_tasks: str
    normalize_obs: bool
    workflows_per_episode: int

    # reward 和归一化相关参数。
    energy_reward_scale: float
    task_baseline_norm: float
    energy_norm_per_mi_ref: float
    alpha_delay_host: float
    alpha_delay_vm: float
    manager_alpha_delay: float
    manager_delay_mode: str

    # workflow deadline 混合采样参数。
    deadline_alpha_small: float
    deadline_alpha_large: float
    deadline_alpha_small_prob: float

    # 训练过程控制参数。
    max_episodes: int
    validation_interval: int
    curriculum_enabled: bool
    save_interval: int
    skip_manager_update_if_zero_assign: bool
    eval_seeds: tuple[int, ...]
    warmup_frac: float
    hard_max_steps: int

    # 本次运行的输出和输入路径。
    run_name: str
    save_dir: str
    log_path: str
    deadline_cache_path: str
    deadline_cache_paths: dict[str, str]

    # 安全强化学习分阶段配置。enabled=False 时保持原 HRL reward/replay 语义。
    safe_rl: SafeRLConfig

    # 三层智能体的独立配置。
    vm_agent: AgentConfig
    host_agent: AgentConfig
    manager_agent: AgentConfig


def normalize_scenario(scenario: str) -> str:
    """将场景代码规范化为大写，并校验格式是否合法。"""
    value = str(scenario).strip().upper()
    if len(value) != 2 or value[0] not in SIZE_FULL or value[1] not in SIZE_FULL:
        raise ValueError("scenario must be one of SS, SM, SL, MS, MM, ML, LS, LM, LL")
    return value


def parse_deadline_cache_overrides(
    values,
    *,
    default_scenario: str | None = None,
) -> dict[str, str]:
    """Parse ``SCENARIO=PATH`` entries, preserving one plain source path."""
    if values is None:
        return {}
    if isinstance(values, Mapping):
        entries = [f"{key}={value}" for key, value in values.items()]
    elif isinstance(values, (str, Path)):
        entries = [values]
    else:
        entries = list(values)
    result: dict[str, str] = {}
    for entry in entries:
        text = str(entry).strip()
        if "=" in text:
            scenario, path = text.split("=", 1)
            scenario = normalize_scenario(scenario)
        elif default_scenario is not None and len(entries) == 1:
            scenario = normalize_scenario(default_scenario)
            path = text
        else:
            raise ValueError(
                "deadline cache overrides must use SCENARIO=PATH"
            )
        if not path.strip():
            raise ValueError("deadline cache path must not be empty")
        result[scenario] = path.strip()
    return result


def validate_single_deadline_cache_paths(
    protocol: str,
    deadline_cache_paths,
    *,
    source_scenario: str | None = None,
    required_scenarios=None,
) -> dict[str, str]:
    """Require and resolve every cache used by a formal Single protocol."""
    paths = parse_deadline_cache_overrides(deadline_cache_paths)
    if str(protocol).strip().lower() != "single":
        return paths
    required = tuple(
        required_scenarios
        or resolve_experiment_protocol(
            "single",
            source_scenario=source_scenario,
        ).test_scenarios
    )
    missing = [scenario for scenario in required if scenario not in paths]
    if missing:
        raise ValueError(
            "formal Single requires deadline cache mappings for: "
            + ", ".join(missing)
        )
    resolved = {}
    for scenario, value in paths.items():
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = ROOT_DIR / path
        resolved[scenario] = str(path.resolve())
    return resolved


def normalize_ddl(ddl: str) -> str:
    """将 deadline 参数统一转换成 Tight/Medium/Loose。"""
    return resolve_deadline_setting(ddl).name


def environment_scenario_values(
    scenario: str,
    *,
    project_root: str | Path = ROOT_DIR,
) -> dict:
    """Return canonical environment inputs for one registered scenario.

    The returned mapping is a fresh value object suitable for rebuilding an
    episode environment. It contains no policy, optimizer, or mutable runtime
    state.
    """
    scenario_id = normalize_scenario(scenario)
    spec = SCENARIO_REGISTRY[scenario_id]
    root = Path(project_root).resolve()
    dax_list = [
        str(root / "data" / "dax" / name)
        for name in spec.dax_files
    ]
    return {
        "scenario": scenario_id,
        "task_code": spec.task_code,
        "resource_code": spec.resource_code,
        "task_size": spec.task_size,
        "resource_size": spec.resource_size,
        "dax_list": dax_list,
        "workflow_families": tuple(
            workflow_families_from_dax_files(dax_list)
        ),
        "num_cloud_hosts": spec.num_cloud_hosts,
        "num_edge_hosts": spec.num_edge_hosts,
        "cloud_vms_per_host": spec.cloud_vms_per_host,
        "edge_vms_per_host": spec.edge_vms_per_host,
        "cloud_pc_tiers": spec.cloud_pc_tiers,
        "edge_pc_tiers": spec.edge_pc_tiers,
        "cloud_bw_tiers": spec.cloud_bw_tiers,
        "edge_bw_tiers": spec.edge_bw_tiers,
        "deadline_cache_path": str(spec.deadline_cache_path(root)),
    }


def resolve_manager_heuristic_manifest(
    resource_code: str,
    override: str | None = None,
    *,
    protocol_context: ExperimentProtocolContext | None = None,
    llm_run_manifest: str | None = None,
    ddl: str | None = None,
) -> str:
    """Resolve an explicit library or one completed, matching LLM run."""
    code = str(resource_code).strip().upper()
    if code not in SIZE_FULL:
        raise ValueError("resource_code must be S, M, or L")
    if override and llm_run_manifest:
        raise ValueError(
            "manager_heuristic_manifest and llm_run_manifest are mutually exclusive"
        )
    if llm_run_manifest:
        if protocol_context is None or ddl is None:
            raise ValueError(
                "llm_run_manifest requires protocol_context and ddl"
            )
        run_manifest_path = Path(llm_run_manifest).expanduser().resolve()
        if not run_manifest_path.is_file():
            raise FileNotFoundError(
                f"LLM run manifest not found: {run_manifest_path}"
            )
        run_manifest = json.loads(
            run_manifest_path.read_text(encoding="utf-8")
        )
        if str(run_manifest.get("status", "")).upper() != "COMPLETED":
            raise ValueError(
                "LLM run manifest must have status=COMPLETED"
            )
        validate_protocol_identity(
            protocol_context,
            run_manifest.get("experiment_protocol", {}),
            artifact_name="LLM run manifest",
        )
        expected_deadline = resolve_deadline_setting(ddl).identity()
        if run_manifest.get("deadline_setting") != expected_deadline:
            raise ValueError(
                "LLM run manifest deadline setting does not match Safe-HRL ddl"
            )
        library_value = str(run_manifest.get("heuristic_library", "")).strip()
        if not library_value:
            raise ValueError(
                "LLM run manifest does not declare heuristic_library"
            )
        path = Path(library_value).expanduser()
        if not path.is_absolute():
            path = run_manifest_path.parent / path
        path = path.resolve()
    elif override:
        path = Path(override).resolve()
    elif protocol_context is not None:
        raise FileNotFoundError(
            "formal Safe-HRL heuristic Manager requires --llm-run-manifest "
            "or --manager-heuristic-manifest; automatic latest-run selection "
            "is intentionally disabled"
        )
    else:
        path = (
            LLM_ROOT
            / "problems"
            / "cews_task_constructive"
            / f"safe_heuristic_library_res{code}.json"
        ).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"safe Manager heuristic manifest not found: {path}"
        )
    return str(path)


def build_train_config(
    scenario: str | None = None,
    ddl: str = "T",
    max_episodes: int | None = None,
    safe_rl_enabled: bool = False,
    safe_rl_shield_enabled: bool = False,
    safe_rl_state_enabled: bool = False,
    safe_rl_dynamic_lambda_enabled: bool = False,
    safe_rl_heuristic_manager_enabled: bool = False,
    manager_heuristic_llm_only: bool = False,
    manager_heuristic_manifest: str | None = None,
    llm_run_manifest: str | None = None,
    safe_rl_offline_pretrain_manifest: str | None = None,
    safe_rl_offline_pretrain_epochs: int = 5,
    safe_rl_offline_pretrain_behavior_cloning: bool = False,
    safe_rl_offline_pretrain_q_r: bool = True,
    safe_rl_offline_pretrain_q_c: bool = True,
    safe_rl_training_pipeline_plan: str | None = None,
    safe_rl_training_resume_checkpoint: str | None = None,
    safe_rl_curriculum_enabled: bool = True,
    optimizer_seed: int = 0,
    protocol: str | None = None,
    source_scenario: str | None = None,
    resource_scale: str | None = None,
    require_deadline_cache: bool = True,
    deadline_cache_override: str | Path | None = None,
    deadline_cache_paths: Mapping[str, str] | None = None,
) -> TrainConfig:
    """根据命令行参数构造完整训练配置。

    参数：
    - scenario：两位场景代码，第一位为任务规模，第二位为资源规模。
    - ddl：deadline 条件，可传 T/M/L 或 Tight/Medium/Loose。
    - max_episodes：可选训练轮数；为空时使用默认 600。

    返回：
    - TrainConfig：训练主循环所需的全部配置。
    """
    if safe_rl_shield_enabled and not safe_rl_enabled:
        raise ValueError(
            "safe_rl_shield_enabled=True requires "
            "safe_rl_enabled=True"
        )
    if safe_rl_state_enabled and not safe_rl_enabled:
        raise ValueError(
            "safe_rl_state_enabled=True requires "
            "safe_rl_enabled=True"
        )
    if safe_rl_dynamic_lambda_enabled and not safe_rl_enabled:
        raise ValueError(
            "safe_rl_dynamic_lambda_enabled=True requires "
            "safe_rl_enabled=True"
        )
    if safe_rl_heuristic_manager_enabled:
        if not safe_rl_enabled:
            raise ValueError(
                "safe_rl_heuristic_manager_enabled=True requires "
                "safe_rl_enabled=True"
            )
        if not safe_rl_shield_enabled:
            raise ValueError(
                "safe_rl_heuristic_manager_enabled=True requires "
                "safe_rl_shield_enabled=True"
            )
        if not safe_rl_state_enabled:
            raise ValueError(
                "safe_rl_heuristic_manager_enabled=True requires "
                "safe_rl_state_enabled=True"
            )
    if (
        manager_heuristic_llm_only
        and not safe_rl_heuristic_manager_enabled
    ):
        raise ValueError(
            "manager_heuristic_llm_only=True requires "
            "safe_rl_heuristic_manager_enabled=True"
        )
    if llm_run_manifest and not safe_rl_heuristic_manager_enabled:
        raise ValueError(
            "llm_run_manifest requires safe_rl_heuristic_manager_enabled=True"
        )
    if manager_heuristic_manifest and not safe_rl_heuristic_manager_enabled:
        raise ValueError(
            "manager_heuristic_manifest requires "
            "safe_rl_heuristic_manager_enabled=True"
        )
    pipeline_plan = None
    pipeline_offline = None
    if safe_rl_training_pipeline_plan:
        from hrl_mix.safe_training_pipeline import (
            load_safe_training_plan,
        )

        pipeline_plan = load_safe_training_plan(
            safe_rl_training_pipeline_plan
        )
        pipeline_offline = dict(
            pipeline_plan.offline_pretraining
        )
        pipeline_manifest = str(
            pipeline_offline["dataset_manifest_path"]
        )
        if (
            safe_rl_offline_pretrain_manifest
            and Path(safe_rl_offline_pretrain_manifest).resolve()
            != Path(pipeline_manifest).resolve()
        ):
            raise ValueError(
                "offline pretraining manifest conflicts with the "
                "stage-2 manifest in the safe training plan"
            )
        safe_rl_offline_pretrain_manifest = pipeline_manifest
        for name, enabled in (
            ("safe_rl_enabled", safe_rl_enabled),
            ("safe_rl_shield_enabled", safe_rl_shield_enabled),
            ("safe_rl_state_enabled", safe_rl_state_enabled),
            (
                "safe_rl_dynamic_lambda_enabled",
                safe_rl_dynamic_lambda_enabled,
            ),
            (
                "safe_rl_heuristic_manager_enabled",
                safe_rl_heuristic_manager_enabled,
            ),
        ):
            if not enabled:
                raise ValueError(
                    "safe training pipeline requires "
                    f"{name}=True"
                )
    if (
        safe_rl_training_resume_checkpoint
        and not safe_rl_training_pipeline_plan
    ):
        raise ValueError(
            "safe training resume requires a pipeline plan"
        )
    offline_pretraining_enabled = bool(
        safe_rl_offline_pretrain_manifest
    )
    if offline_pretraining_enabled:
        if not safe_rl_enabled:
            raise ValueError(
                "offline safe pretraining requires "
                "safe_rl_enabled=True"
            )
        if not safe_rl_shield_enabled:
            raise ValueError(
                "offline safe pretraining requires "
                "safe_rl_shield_enabled=True"
            )
        if not safe_rl_state_enabled:
            raise ValueError(
                "offline safe pretraining requires "
                "safe_rl_state_enabled=True"
            )
        if not safe_rl_heuristic_manager_enabled:
            raise ValueError(
                "offline safe pretraining requires "
                "safe_rl_heuristic_manager_enabled=True"
            )
    protocol_context = None
    if protocol is None:
        if source_scenario is not None or resource_scale is not None:
            raise ValueError(
                "source_scenario/resource_scale require an explicit protocol"
            )
        scenario = normalize_scenario(scenario or "SS")
        protocol_name = "legacy"
        protocol_source = scenario
        protocol_resource_scale = scenario[1]
        training_scenarios = (scenario,)
        test_scenarios = (scenario,)
        protocol_train_seeds = (1,)
        protocol_validation_seeds = (1,)
        protocol_final_test_seeds = tuple(range(201, 231))
    else:
        mode = str(protocol).strip().lower()
        if mode == "single":
            if (
                scenario is not None
                and source_scenario is not None
                and normalize_scenario(scenario)
                != normalize_scenario(source_scenario)
            ):
                raise ValueError(
                    "legacy scenario conflicts with single source_scenario"
                )
            source = source_scenario or scenario or "SS"
            protocol_context = resolve_experiment_protocol(
                "single",
                source_scenario=source,
                resource_scale=resource_scale,
            )
        elif mode == "multi":
            if scenario is not None:
                raise ValueError(
                    "multi protocol does not accept the legacy scenario argument"
                )
            protocol_context = resolve_experiment_protocol(
                "multi",
                source_scenario=None,
                resource_scale=resource_scale,
            )
        else:
            raise ValueError("protocol must be 'single' or 'multi'")
        protocol_name = protocol_context.protocol
        protocol_source = protocol_context.source_scenario
        protocol_resource_scale = protocol_context.resource_scale
        training_scenarios = protocol_context.training_scenarios
        test_scenarios = protocol_context.test_scenarios
        protocol_train_seeds = protocol_context.safe_hrl_train_seeds
        protocol_validation_seeds = protocol_context.safe_hrl_validation_seeds
        protocol_final_test_seeds = protocol_context.final_test_seeds
        scenario = (
            protocol_context.source_scenario
            if protocol_context.protocol == "single"
            else protocol_context.training_scenarios[0]
        )

    deadline_setting = resolve_deadline_setting(ddl)
    ddl_name = deadline_setting.name
    environment_values = environment_scenario_values(scenario)
    normalized_cache_paths = {}
    for cache_scenario, cache_value in parse_deadline_cache_overrides(
        deadline_cache_paths
    ).items():
        cache_path = Path(cache_value).expanduser()
        if not cache_path.is_absolute():
            cache_path = ROOT_DIR / cache_path
        normalized_cache_paths[cache_scenario] = str(
            cache_path.resolve()
        )
    if deadline_cache_override is not None:
        override_path = Path(deadline_cache_override).expanduser()
        if not override_path.is_absolute():
            override_path = ROOT_DIR / override_path
        override_path = override_path.resolve()
        normalized_cache_paths[scenario] = str(override_path)

    # An explicit scenario mapping is authoritative. Previously only the
    # legacy single-path override replaced the registry value here, so callers
    # that supplied the newer deadline_cache_paths mapping still had the old
    # exact_mix_v1 cache checked and recorded as the active source cache.
    selected_cache_path = normalized_cache_paths.get(scenario)
    if selected_cache_path is not None:
        environment_values["deadline_cache_path"] = selected_cache_path

    if require_deadline_cache:
        missing_explicit_caches = {
            cache_scenario: cache_path
            for cache_scenario, cache_path in normalized_cache_paths.items()
            if not Path(cache_path).is_file()
        }
        if missing_explicit_caches:
            details = ", ".join(
                f"{cache_scenario}={cache_path}"
                for cache_scenario, cache_path in sorted(
                    missing_explicit_caches.items()
                )
            )
            raise FileNotFoundError(
                "explicit deadline cache file(s) not found: " + details
            )

    task_code = environment_values["task_code"]
    res_code = environment_values["resource_code"]
    task_size = environment_values["task_size"]
    res_size = environment_values["resource_size"]
    num_cloud_hosts = environment_values["num_cloud_hosts"]
    num_edge_hosts = environment_values["num_edge_hosts"]
    cloud_vms_per_host = environment_values["cloud_vms_per_host"]
    edge_vms_per_host = environment_values["edge_vms_per_host"]
    dax_list = environment_values["dax_list"]
    workflow_families = environment_values["workflow_families"]
    resolved_manager_manifest = None
    if safe_rl_heuristic_manager_enabled:
        resolved_manager_manifest = Path(
            resolve_manager_heuristic_manifest(
                res_code,
                manager_heuristic_manifest,
                protocol_context=protocol_context,
                llm_run_manifest=llm_run_manifest,
                ddl=ddl_name,
            )
        )

    run_name = legacy_hrl_run_id(
        scenario,
        ddl_name,
        seed=1,
    )
    if safe_rl_enabled and pipeline_plan is None:
        # 原阶段后缀全部展开后，在当前 Windows 工作区会让完整路径超过
        # Win32 的常见路径限制，导致 safe_rl 在创建输出目录时直接失败。
        # legacy run_name 完全不变；只有显式 safe_rl 使用紧凑、可追踪且
        # 带配置摘要的独立目录，仍不会覆盖旧实验。
        safe_name_payload = {
            "scenario": scenario,
            "ddl": ddl_name,
            "shield": bool(safe_rl_shield_enabled),
            "state": bool(safe_rl_state_enabled),
            "dynamic_lambda": bool(
                safe_rl_dynamic_lambda_enabled
            ),
            "heuristic_manager": bool(
                safe_rl_heuristic_manager_enabled
            ),
            "manager_heuristic_manifest": (
                str(resolved_manager_manifest)
                if resolved_manager_manifest is not None
                else None
            ),
            "offline_pretraining": bool(
                offline_pretraining_enabled
            ),
            "offline_manifest": (
                str(
                    Path(
                        safe_rl_offline_pretrain_manifest
                    ).resolve()
                )
                if safe_rl_offline_pretrain_manifest
                else None
            ),
            "offline_epochs": int(
                safe_rl_offline_pretrain_epochs
            ),
            "offline_behavior_cloning": bool(
                safe_rl_offline_pretrain_behavior_cloning
            ),
            "offline_q_r": bool(
                safe_rl_offline_pretrain_q_r
            ),
            "offline_q_c": bool(
                safe_rl_offline_pretrain_q_c
            ),
            "curriculum_enabled": bool(safe_rl_curriculum_enabled),
            "optimizer_seed": int(optimizer_seed),
        }
        if manager_heuristic_llm_only:
            # 只在开启时才写入这一项：run_name 是整个 payload 的摘要，无条件新增
            # 键会改变所有既有 safe_rl 运行的目录名。开/关两种配置仍然互不相同，
            # 因为“缺少该键”与“该键为 True”的摘要必然不同。
            safe_name_payload["llm_only_heuristics"] = True
        run_name = safe_hrl_run_id(
            scenario,
            ddl_name,
            safe_name_payload,
        )
    if pipeline_plan is not None:
        run_name = pipeline_run_id(
            scenario,
            ddl_name,
            pipeline_plan.plan_hash + (
                ":curriculum" if safe_rl_curriculum_enabled
                else ":without_curriculum"
            ) + f":optimizer_seed={int(optimizer_seed)}",
        )
    if protocol_context is None:
        output_paths = training_output_paths(ROOT_DIR, run_name)
        save_dir = output_paths.checkpoint_dir
        log_path = output_paths.log_path
    else:
        save_dir = protocol_context.checkpoint_root / run_name
        log_path = (
            protocol_context.artifact_output_root
            / run_name
            / "train.csv"
        )
    deadline_cache_path = Path(
        environment_values["deadline_cache_path"]
    )
    normalized_cache_paths.setdefault(
        scenario, str(deadline_cache_path.resolve())
    )
    if require_deadline_cache and not deadline_cache_path.exists():
        raise FileNotFoundError(
            f"deadline cache 不存在：{deadline_cache_path}\n"
            f"请先生成对应的 FCFS deadline cache。"
        )

    # 训练开始前确保输出目录存在，避免保存模型或写日志时失败。
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_path.parent, exist_ok=True)

    return TrainConfig(
        experiment_protocol=(
            protocol_context.identity()
            if protocol_context is not None
            else None
        ),
        protocol=protocol_name,
        source_scenario=protocol_source,
        resource_scale=protocol_resource_scale,
        training_scenarios=training_scenarios,
        test_scenarios=test_scenarios,
        train_seeds=protocol_train_seeds,
        validation_seeds=protocol_validation_seeds,
        final_test_seeds=protocol_final_test_seeds,
        scenario=scenario,
        ddl_name=ddl_name,
        task_code=task_code,
        res_code=res_code,
        task_size=task_size,
        res_size=res_size,
        dax_list=dax_list,
        workflow_families=workflow_families,
        num_cloud_hosts=num_cloud_hosts,
        num_edge_hosts=num_edge_hosts,
        cloud_vms_per_host=cloud_vms_per_host,
        edge_vms_per_host=edge_vms_per_host,
        cloud_pc_tiers=environment_values['cloud_pc_tiers'],
        edge_pc_tiers=environment_values['edge_pc_tiers'],
        cloud_bw_tiers=environment_values['cloud_bw_tiers'],
        edge_bw_tiers=environment_values['edge_bw_tiers'],
        horizon=1e9,
        arrival_lambda=0.03,
        random_seed=protocol_train_seeds[0],
        optimizer_seed=int(optimizer_seed),
        max_ready_tasks="auto",
        normalize_obs=True,
        workflows_per_episode=50,
        energy_reward_scale=1e-3,
        task_baseline_norm=300.0,
        energy_norm_per_mi_ref=20.0,
        alpha_delay_host=0.75,
        alpha_delay_vm=0.75,
        manager_alpha_delay=0.75,
        manager_delay_mode="tardiness",
        deadline_alpha_small=deadline_setting.alpha_small,
        deadline_alpha_large=deadline_setting.alpha_large,
        deadline_alpha_small_prob=(
            deadline_setting.alpha_small_probability
        ),
        max_episodes=int(max_episodes) if max_episodes is not None else 600,
        validation_interval=25,
        curriculum_enabled=bool(safe_rl_curriculum_enabled),
        save_interval=2000,
        skip_manager_update_if_zero_assign=True,
        eval_seeds=protocol_validation_seeds,
        warmup_frac=0.03,
        hard_max_steps=10**9,
        run_name=run_name,
        save_dir=str(save_dir),
        log_path=str(log_path),
        deadline_cache_path=str(deadline_cache_path),
        deadline_cache_paths=normalized_cache_paths,
        safe_rl=SafeRLConfig(
            enabled=bool(safe_rl_enabled),
            safety_discount=0.95,
            safety_learning_rate=3e-4,
            safety_loss_weight=1.0,
            initial_lagrange_multiplier=1.0,
            fuzzy_energy_uncertainty_weight=1.0,
            fuzzy_deadline_eta=0.95,
            process_risk_aggregation="mean",
            shield=SafetyShieldConfig(
                enabled=bool(safe_rl_shield_enabled),
                fallback_controller="fixed_vm_rule",
            ),
            state=SafetyStateConfig(
                enabled=bool(safe_rl_state_enabled),
                high_uncertainty_threshold=0.2,
                recent_record_window=100,
            ),
            lagrangian=LagrangianSafetyConfig(
                enabled=bool(safe_rl_dynamic_lambda_enabled),
                lambda_init=1.0,
                lambda_lr=0.01,
                lambda_min=0.0,
                lambda_max=100.0,
                cost_budget=0.0,
                update_interval=1,
                cost_ema_factor=0.9,
                warmup_steps=5,
            ),
            replay=SafetyReplayConfig(
                transition_schema_version=1,
                near_boundary_margin=1.0,
                combined_per_priority=False,
                performance_td_weight=1.0,
                safety_td_weight=1.0,
            ),
            manager_heuristics=SafeManagerHeuristicConfig(
                mode=(
                    "heuristic_selection_mode"
                    if safe_rl_heuristic_manager_enabled
                    else "legacy_rule_weight_mode"
                ),
                library_manifest_path=(
                    str(resolved_manager_manifest)
                    if safe_rl_heuristic_manager_enabled
                    else None
                ),
                recent_window=20,
                llm_only=bool(manager_heuristic_llm_only),
            ),
            offline_pretraining=OfflinePretrainingConfig(
                enabled=offline_pretraining_enabled,
                dataset_manifest_path=(
                    str(safe_rl_offline_pretrain_manifest)
                    if offline_pretraining_enabled
                    else None
                ),
                epochs=int(
                    pipeline_offline.get(
                        "epochs",
                        safe_rl_offline_pretrain_epochs,
                    )
                    if pipeline_offline is not None
                    else safe_rl_offline_pretrain_epochs
                ),
                batch_size=int(
                    pipeline_offline.get("batch_size", 128)
                    if pipeline_offline is not None
                    else 128
                ),
                performance_learning_rate=float(
                    pipeline_offline.get(
                        "performance_learning_rate",
                        1e-4,
                    )
                    if pipeline_offline is not None
                    else 1e-4
                ),
                safety_learning_rate=float(
                    pipeline_offline.get(
                        "safety_learning_rate",
                        1e-4,
                    )
                    if pipeline_offline is not None
                    else 1e-4
                ),
                train_q_r=bool(
                    pipeline_offline.get(
                        "train_q_r",
                        safe_rl_offline_pretrain_q_r,
                    )
                    if pipeline_offline is not None
                    else safe_rl_offline_pretrain_q_r
                ),
                train_q_c=bool(
                    pipeline_offline.get(
                        "train_q_c",
                        safe_rl_offline_pretrain_q_c,
                    )
                    if pipeline_offline is not None
                    else safe_rl_offline_pretrain_q_c
                ),
                behavior_cloning_enabled=bool(
                    pipeline_offline.get(
                        "behavior_cloning_enabled",
                        safe_rl_offline_pretrain_behavior_cloning,
                    )
                    if pipeline_offline is not None
                    else safe_rl_offline_pretrain_behavior_cloning
                ),
                behavior_cloning_weight=float(
                    pipeline_offline.get(
                        "behavior_cloning_weight",
                        0.1,
                    )
                    if pipeline_offline is not None
                    else 0.1
                ),
                sync_targets_after_pretraining=bool(
                    pipeline_offline.get(
                        "sync_targets_after_pretraining",
                        True,
                    )
                    if pipeline_offline is not None
                    else True
                ),
                random_seed=int(
                    pipeline_offline.get("random_seed", 17)
                    if pipeline_offline is not None
                    else 17
                ),
            ),
            training_pipeline=SafeTrainingPipelineConfig(
                enabled=pipeline_plan is not None,
                plan_path=(
                    pipeline_plan.source_path
                    if pipeline_plan is not None
                    else None
                ),
                resume_checkpoint_path=(
                    str(
                        Path(
                            safe_rl_training_resume_checkpoint
                        ).resolve()
                    )
                    if safe_rl_training_resume_checkpoint
                    else None
                ),
            ),
            model_selection=FeasibilityFirstModelSelectionConfig(
                enabled=True,
                require_all_validation_seeds_feasible=True,
            ),
            metrics=SafeMetricsConfig(
                enabled=True,
                schema_version=1,
                output_subdir="safe_metrics",
                convergence_window=5,
            ),
        ),
        vm_agent=AgentConfig(
            lr=3e-4,
            gamma=0.95,
            batch_size=256,
            buffer_size=120000,
            eps_start=1.0,
            eps_end=0.05,
            eps_decay_steps=80000,
            target_update_tau=0.005,
            grad_clip=10.0,
            hidden_dims=(1024, 1024, 512, 512, 256),
        ),
        host_agent=AgentConfig(
            lr=3e-4,
            gamma=0.95,
            batch_size=256,
            buffer_size=120000,
            eps_start=1.0,
            eps_end=0.05,
            eps_decay_steps=80000,
            target_update_tau=0.005,
            grad_clip=10.0,
            hidden_dims=(1024, 1024, 512, 512, 256),
        ),
        manager_agent=AgentConfig(
            lr=3e-4,
            gamma=0.95,
            batch_size=256,
            buffer_size=60000,
            eps_start=1.0,
            eps_end=0.05,
            eps_decay_steps=40000,
            target_update_tau=0.01,
            grad_clip=10.0,
            hidden_dims=(512, 512, 256, 128),
        ),
    )
