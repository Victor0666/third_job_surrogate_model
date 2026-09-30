# seevo.py
import logging
import subprocess
import numpy as np
import os
import sys
import json
import math
import hashlib
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from time import time
from omegaconf import OmegaConf

os.environ['MKL_THREADING_LAYER'] = 'INTEL'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
# Configure Intel MKL runtime to avoid threading-library conflicts during evaluation.
# 配置 Intel MKL 运行时，避免评估过程中出现线程库冲突。

from utils.utils import *
import random
from rule_optimization import (
    CMAESOptimizer,
    EvaluationCache,
    EvaluationCacheKey,
    OptimizerConfig,
    RuleValidationError,
    accumulate_cross_generation_diagnostics,
    aggregate_seed_evaluations,
    canonical_json_sha256 as rule_json_sha256,
    constraint_priority_key,
    freeze_rule_source,
    generate_parameter_diagnostics,
    parse_rule_candidate,
)
from rule_optimization.evaluation_cache import simulator_fingerprint
from surrogate import (
    ReplayVerdict,
    SurrogateConfig,
    SurrogateContext,
    SurrogateManager,
    replay_frozen_rule,
)
from counterfactual_feedback import (
    ARCHIVE_VERSION,
    CandidateTaskSnapshot,
    CriticalStateArchive,
    CriticalStateReplayCache,
    CriticalStateReplayConfig,
    CriticalStateReplayer,
    CounterfactualComparison,
    CounterfactualConfig,
    CounterfactualOutcome,
    DecisionTrace,
    DiagnosticReport,
    FeedbackAggregator,
    aggregate_replay_results,
    build_critical_state_record,
    canonical_hash as counterfactual_hash,
    compact_feedback_for_prompt,
    compact_replay_feedback,
    load_frozen_priority_rule,
    load_jsonl,
    reject_test_seeds,
    strict_replay_gate_triggered,
)
from algorithms.llm_safe_hrl.scenario_registry import (
    resolve_experiment_protocol,
    validate_component_scenarios,
    validate_protocol_identity,
)
from protocol_config import apply_seevo_scenario_config


RESULT_JSON_PREFIX = "RESULT_JSON="
LLM_EVOLUTION_FORBIDDEN_SEEDS = frozenset({101, 102, 103}).union(range(201, 231))


def _config_mapping(value, label: str) -> dict:
    """Return a resolved mapping from an OmegaConf or plain mapping value."""
    if value is None:
        raise ValueError(f"{label} is required")
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return dict(value)


def _resolve_seevo_protocol_context(cfg):
    """Rebuild and verify the immutable protocol boundary used by SeEvo."""
    identity = _config_mapping(
        getattr(cfg, "experiment_protocol", None),
        "cfg.experiment_protocol",
    )
    training = list(identity.get("training_scenarios", []))
    if not training:
        raise ValueError("experiment protocol has no training scenarios")
    context = resolve_experiment_protocol(
        identity.get("protocol", ""),
        source_scenario=identity.get("source_scenario"),
        resource_scale=str(training[0])[1:2],
        train_seeds=identity.get("llm_train_seeds", []),
        validation_seeds=identity.get("llm_validation_seeds", []),
    )
    validate_protocol_identity(
        context,
        identity,
        artifact_name="SeEvo configuration",
    )
    problem = _config_mapping(cfg.problem, "cfg.problem")
    validate_protocol_identity(
        context,
        _config_mapping(
            problem.get("experiment_protocol"),
            "cfg.problem.experiment_protocol",
        ),
        artifact_name="SeEvo problem configuration",
    )
    return context


def parse_result_json(stdout_text: str) -> dict:
    """从评价 stdout 末尾向前解析最后一条 ``RESULT_JSON`` 记录。

    评价环境、数据加载器和候选代码都可能输出普通日志，因此不能再依赖
    “倒数第二行是目标值”这类脆弱位置约定。选择最后一条匹配记录还允许评价器
    在调试时输出中间结果，而最终记录始终具有最高优先级。

    返回完整指标字典，但会额外保证 objective 可转为有限 float。缺失记录、
    非法 JSON、非对象载荷、缺失 objective 或非有限 objective 都抛出 ValueError，
    由 evaluate_population 统一把对应候选标记为无效。
    """
    result_line = next(
        (
            line for line in reversed(stdout_text.splitlines())
            if line.startswith(RESULT_JSON_PREFIX)
        ),
        None,
    )
    if result_line is None:
        raise ValueError("Evaluator stdout does not contain a RESULT_JSON line.")
    # 只去掉协议前缀，后面的内容必须是完整 JSON 对象。
    try:
        metrics = json.loads(result_line[len(RESULT_JSON_PREFIX):])
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid RESULT_JSON payload: {exc}") from exc
    if not isinstance(metrics, dict):
        raise ValueError("RESULT_JSON payload must be a JSON object.")
    try:
        objective = float(metrics["objective"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("RESULT_JSON must contain a numeric objective.") from exc
    if not math.isfinite(objective):
        raise ValueError("RESULT_JSON objective must be finite.")
    # 写回统一 float 类型，后续最小化/最大化逻辑无需处理字符串或 NumPy 标量。
    metrics["objective"] = objective
    return metrics


def individual_comparison_key(individual: dict) -> tuple[float, ...]:
    """返回可行性优先的通用个体比较键，元组越小表示个体越优。

    评价器的 ``objective``/``individual['obj']`` 始终保存原始优化目标。对于
    CEWS，它就是总能耗，绝不混入 DDL penalty。约束通过 metrics 中的独立字段
    处理：

    1. 可行个体：``(0, 0, 0, obj)``，因此只按目标值排序；
    2. 不可行个体：``(1, 违反率, 总延期, obj)``；
    3. 执行失败或非有限目标：``(2, inf, inf, inf)``。

    没有提供约束字段的其他问题默认视为可行，因而退化为原来的单目标排序。
    最大化问题的 obj 已在 evaluate_population 中取负，仍可直接使用该键。
    """
    try:
        objective = float(individual.get("obj", float("inf")))
    except (TypeError, ValueError):
        objective = float("inf")
    if individual.get("exec_success") is False or not math.isfinite(objective):
        return (2.0, float("inf"), float("inf"), float("inf"))

    metrics = individual.get("metrics") or {}
    def finite_or_infinity(value):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return float("inf")
        return number if math.isfinite(number) else float("inf")

    robustness = finite_or_infinity(
        metrics.get(
            "objective_std_across_seeds",
            metrics.get("objective_cv_across_seeds", 0.0),
        )
    )
    complexity = individual.get("complexity") or {}
    complexity_key = (
        int(complexity.get("branch_count", 0)),
        int(complexity.get("interaction_count", 0)),
        int(complexity.get("ast_node_count", 0)),
    )
    tolerance = max(0.0, float(individual.get("performance_tolerance", 0.0)))
    objective_bucket = (
        float(round(objective / tolerance)) if tolerance > 0.0 else objective
    )
    if bool(metrics.get("constraint_feasible", True)):
        return (
            0.0,
            0.0,
            0.0,
            objective_bucket,
            robustness,
            *complexity_key,
            objective,
        )

    primary_violation = finite_or_infinity(
        metrics.get(
            "constraint_violation",
            metrics.get("deadline_violation_rate", float("inf")),
        )
    )
    secondary_violation = finite_or_infinity(
        metrics.get(
            "constraint_secondary_violation",
            metrics.get("total_lateness", float("inf")),
        )
    )
    return (
        1.0,
        primary_violation,
        secondary_violation,
        objective_bucket,
        robustness,
        *complexity_key,
        objective,
    )


# def individual_performance_summary(individual: dict) -> str:
#     """为反思 Prompt 生成人类可读的目标与约束状态摘要。"""
#     metrics = individual.get("metrics") or {}
#     if "constraint_feasible" not in metrics:
#         return f"objective={float(individual['obj']):.4f}"
#     return (
#         f"energy={float(individual['obj']):.4f} J, "
#         f"DDL feasible={bool(metrics['constraint_feasible'])}, "
#         f"violation_rate={float(metrics.get('deadline_violation_rate', 0.0)):.6f}, "
#         f"total_lateness={float(metrics.get('total_lateness', 0.0)):.4f} s"
#     )
def individual_performance_summary(individual: dict) -> str:
    """为反思 Prompt 生成人类可读的模糊能耗与约束状态摘要。"""
    metrics = individual.get("metrics") or {}
    objective = float(individual["obj"])

    if "constraint_feasible" not in metrics:
        return f"objective={objective:.4f}"

    fuzzy_score = float(
        metrics.get("fuzzy_total_energy_score", objective)
    )
    fuzzy_mean = float(
        metrics.get("fuzzy_total_energy_mean", fuzzy_score)
    )
    fuzzy_std = float(
        metrics.get("fuzzy_total_energy_std", 0.0)
    )
    modal_energy = float(
        metrics.get("total_energy", fuzzy_score)
    )

    parts = [
        f"fuzzy_energy_score={fuzzy_score:.4f} J",
        f"fuzzy_energy_mean={fuzzy_mean:.4f} J",
        f"fuzzy_energy_std={fuzzy_std:.4f} J",
        f"modal_energy={modal_energy:.4f} J",
        f"DDL_feasible={bool(metrics['constraint_feasible'])}",
        (
            "violation_rate="
            f"{float(metrics.get('deadline_violation_rate', 0.0)):.6f}"
        ),
        (
            "total_lateness="
            f"{float(metrics.get('total_lateness', 0.0)):.4f} s"
        ),
    ]

    # 以下字段只用于诊断，不参与 objective。
    if "objective_std_across_seeds" in metrics:
        parts.append(
            "cross_seed_objective_std="
            f"{float(metrics['objective_std_across_seeds']):.4f} J"
        )

    if "objective_max_across_seeds" in metrics:
        parts.append(
            "worst_seed_objective="
            f"{float(metrics['objective_max_across_seeds']):.4f} J"
        )

    if "feasible_seed_rate" in metrics:
        parts.append(
            "feasible_seed_rate="
            f"{float(metrics['feasible_seed_rate']):.4f}"
        )

    return ", ".join(parts)


def rule_source_for_evolution(individual: dict) -> str:
    """Return the discrete parameterized structure when one is available."""
    source = individual.get("parameterized_rule_source")
    if source and individual.get("parameter_schema"):
        return str(source)
    return filter_code(individual.get("code"))


def parameter_feedback_summary(individual: dict) -> str:
    """Serialize three distinct, compact evidence roles for reflector prompts."""
    diagnostics = individual.get("parameter_diagnostics")
    if not diagnostics:
        return json.dumps(
            {
                "structure_hash": individual.get("structure_hash"),
                "parameter_optimization": "not_applicable_legacy_rule",
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    feedback_diagnostics = json.loads(
        json.dumps(diagnostics, ensure_ascii=True, allow_nan=False)
    )
    correlation = feedback_diagnostics.get("correlation_analysis", {})
    if isinstance(correlation, dict):
        evidence = correlation.get("evidence", {})
        if isinstance(evidence, dict):
            evidence.pop("matrix", None)
    counterfactual = individual.get("counterfactual_feedback")
    max_feedback_chars = int(
        individual.get("counterfactual_max_feedback_chars", 12000)
    )
    compact_counterfactual = (
        compact_feedback_for_prompt(counterfactual, max_feedback_chars)
        if isinstance(counterfactual, dict)
        else {
            "status": "not_analyzed_for_this_structure",
            "reason": "only configured frozen-rule elites receive offline traces",
        }
    )
    replay = individual.get("critical_state_replay_summary")
    replay_budget = int(individual.get("critical_state_max_feedback_chars", 10000))
    compact_replay = (
        compact_replay_feedback(replay, replay_budget)
        if isinstance(replay, dict)
        else {
            "status": "archive_empty_or_structure_not_replayed",
            "reason": "replay uses only historical training-seed critical states",
        }
    )
    parameter_focus = {
        "cross_generation_evidence": feedback_diagnostics.get("cross_generation_evidence", {}),
        "fragility_analysis": feedback_diagnostics.get("fragility_analysis", {}),
    }
    joint_actions = []
    for source, rows in (
        ("current_counterfactual", compact_counterfactual.get("high_confidence_structural_actions", [])),
        ("historical_replay", compact_replay.get("high_confidence_structural_actions", [])),
    ):
        for row in rows[:4] if isinstance(rows, list) else []:
            joint_actions.append({"source": source, **dict(row)})
    payload = {
        "structure_hash": individual.get("structure_hash"),
        "parameter_schema_hash": individual.get("parameter_schema_hash"),
        "best_parameter_hash": individual.get("best_parameter_hash"),
        "best_parameters": individual.get("best_parameters", {}),
        "parameter_diagnostics": feedback_diagnostics,
        "counterfactual_mechanism_feedback": compact_counterfactual,
        "critical_state_replay_feedback": compact_replay,
        "joint_evolution_focus": {
            "A_parameter_structure_problem": parameter_focus,
            "B_current_decision_mechanism_problem": compact_counterfactual.get(
                "repeated_failure_patterns", []
            )[:3],
            "C_persistent_historical_state_patterns": compact_replay.get(
                "hard_state_failure_patterns", []
            )[:3],
            "D_resolved_state_regressions": compact_replay.get(
                "resolved_state_regressions", []
            )[:3],
            "E_high_confidence_structural_actions": joint_actions[:6],
            "F_evidence_limitations": (
                compact_counterfactual.get("limitations", [])[:2]
                + compact_replay.get("limitations", [])[:2]
            ),
        },
        "evidence_roles": {
            "parameter_diagnostics": "parameter-space evidence exposing structural limitations",
            "counterfactual_mechanism_feedback": "decision-level evidence explaining why task choices fail",
            "critical_state_replay_feedback": "historical stress-state evidence showing whether a frozen rule repeats prior errors",
        },
    }
    return json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
    )

class SeEvo:
    """Self-Evolution (SeEvo) Algorithm.
    自进化（SeEvo）算法。
    
    An LLM-based evolutionary algorithm that combines population inter-evolution,
    individual self-evolution, and reflection mechanisms to evolve heuristics
    for optimization problems.
    这是一种基于大语言模型的进化算法，结合种群间进化、个体自进化和反思机制，
    用于为优化问题演化启发式规则。
    """

    # 一代里每个个体各跑一次 CMA-ES，而单次 CMA-ES 批只有
    # population_size × scenarios × stage_seeds 个上下文（quick 阶段仅 6 个），
    # 远填不满 max_parallel_evaluations 个槽位。因此：个体级并发用一个临时的
    # structure-prepare 池，评价子进程全部投递到共享的 _evaluation_executor，
    # 由它统一限流。两个池必须相互独立，否则等待批结果的个体线程会占满评价
    # 线程而死锁。
    _evaluation_executor = None
    # 锁定义在类上而不是 __init__ 里：测试用 object.__new__(SeEvo) 构造替身、
    # 不走 __init__，这些属性仍必须可用。一次运行只有一个 SeEvo 实例，
    # 类级共享不会造成额外串行。
    _evaluation_lock = threading.Lock()
    # 结构哈希在同一代内唯一，所以受这把锁保护的字典键天然不冲突；
    # 加锁只是为了让计数与 setdefault/append 序列在多线程下仍然确定。
    _shared_state_lock = threading.Lock()

    def __init__(self, cfg, root_dir, case_num) -> None:
        """Initialize SeEvo algorithm.
        初始化 SeEvo 算法。

        作用：
            初始化 SeEvo 对象，保存配置、路径、算例编号、进化状态和反思状态，
            并完成提示词与初始种群初始化。
        输入：
            cfg: 包含算法、模型、问题和运行模式等参数的配置对象。
            root_dir: 项目根目录。
            case_num: 用于评估的算例编号列表。
        输出：
            None。该函数通过设置实例属性完成初始化。
        """
        self.cfg = cfg
        self.root_dir = root_dir
        self.case_num = case_num
        self.mode = cfg.mode  # train or test / 训练模式或测试模式
        
        # Algorithm parameters
        # 算法参数
        self.mutation_rate = cfg.mutation_rate
        self.iteration = 0
        self.function_evals = 0
        
        # Population tracking
        # 种群跟踪信息
        self.elitist = None
        self.best_obj_overall = None
        # 与 best_obj_overall 分开保存：obj 是纯目标，comparison key 才包含约束。
        self.best_comparison_key_overall = None
        self.best_code_overall = None
        self.best_code_path_overall = None
        self.generation_fallback_individual = None
        
        # Reflection mechanism
        # 反思机制
        self.long_term_reflection_str = ""

        raw_optimizer_config = getattr(cfg, "parameter_optimization", None)
        if raw_optimizer_config is not None and OmegaConf.is_config(
            raw_optimizer_config
        ):
            raw_optimizer_config = OmegaConf.to_container(
                raw_optimizer_config,
                resolve=True,
            )
        self.parameter_optimizer_config = OptimizerConfig.from_mapping(
            raw_optimizer_config
        )
        raw_surrogate_config = getattr(cfg, "surrogate", None)
        if raw_surrogate_config is not None and OmegaConf.is_config(
            raw_surrogate_config
        ):
            raw_surrogate_config = OmegaConf.to_container(
                raw_surrogate_config,
                resolve=True,
            )
        self.surrogate_config = SurrogateConfig.from_mapping(raw_surrogate_config)
        cache_path = self.parameter_optimizer_config.cache_path
        if cache_path and not os.path.isabs(cache_path):
            cache_path = os.path.abspath(cache_path)
        self.parameter_evaluation_cache = EvaluationCache(
            enabled=(
                self.parameter_optimizer_config.enabled
                and self.parameter_optimizer_config.cache_enabled
                and self.mode == "train"
            ),
            path=cache_path,
        )
        self.parameter_warm_starts = {}
        self.parameter_diagnostic_history = {}
        self.parameter_evaluation_count = 0
        self._evaluation_executor = None
        raw_counterfactual_config = getattr(
            cfg,
            "counterfactual_feedback",
            None,
        )
        if raw_counterfactual_config is not None and OmegaConf.is_config(
            raw_counterfactual_config
        ):
            raw_counterfactual_config = OmegaConf.to_container(
                raw_counterfactual_config,
                resolve=True,
            )
        self.counterfactual_config = (
            CounterfactualConfig.from_mapping(raw_counterfactual_config)
            if raw_counterfactual_config is not None
            else CounterfactualConfig(enabled=False)
        )
        self.experiment_protocol_context = _resolve_seevo_protocol_context(cfg)
        self.experiment_protocol_identity = (
            self.experiment_protocol_context.identity()
        )
        validate_component_scenarios(
            self.experiment_protocol_context,
            "CMA-ES",
            self.parameter_optimizer_config.scenario_ids,
        )
        validate_component_scenarios(
            self.experiment_protocol_context,
            "Counterfactual",
            self.counterfactual_config.scenario_ids,
        )
        raw_replay_config = getattr(cfg, "critical_state_replay", None)
        if raw_replay_config is not None and OmegaConf.is_config(raw_replay_config):
            raw_replay_config = OmegaConf.to_container(raw_replay_config, resolve=True)
        self.critical_state_replay_config = (
            CriticalStateReplayConfig.from_mapping(raw_replay_config)
            if raw_replay_config is not None
            else CriticalStateReplayConfig(enabled=False)
        )
        self.critical_state_archive = None
        
        # Performance tracking for intelligent evolution guidance (only in train mode)
        # 用于智能进化引导的性能跟踪（仅训练模式）
        if self.mode == "train":
            self.improvement_history = []  # Track improvement trends / 跟踪改进趋势
            self.best_obj_history = []      # Track best objective over iterations / 跟踪各迭代最优目标值
        
        # Initialize prompts and population
        # 初始化提示词和种群
        self.init_prompt()
        self.surrogate_manager = self._build_surrogate_manager()
        self.init_population()


    def init_prompt(self) -> None:
        """Initialize prompt templates and problem-specific settings.
        初始化提示词模板和问题相关设置。

        作用：
            读取问题配置、问题专用提示词、通用提示词模板和外部知识，
            并生成后续 LLM 交叉、变异、反思时使用的 prompt。
        输入：
            无显式输入；使用 self.cfg 和 self.root_dir 中的配置与路径信息。
        输出：
            None。该函数将提示词模板、问题描述和路径信息保存到实例属性中。
        """
        # Problem configuration
        # 读取问题配置
        self.problem = self.cfg.problem.problem_name
        self.problem_desc = self.cfg.problem.description
        self.problem_size = self.cfg.problem.problem_size
        self.func_name = self.cfg.problem.func_name
        self.obj_type = self.cfg.problem.obj_type  # "min" or "max" / 最小化或最大化目标
        self.problem_type = self.cfg.problem.problem_type
        
        # Log problem information
        # 记录问题信息
        logging.info("Problem: " + self.problem)
        logging.info("Problem description: " + self.problem_desc)
        logging.info("Function name: " + self.func_name)
        
        # Set up file paths
        # 设置文件路径
        self.prompt_dir = f"{self.root_dir}/prompts"
        self.problem_dir = os.path.join(self.root_dir, "problems", self.problem)
        # CEWS 每个个体都会在此目录写入独立候选模块。exist_ok=True 允许恢复
        # 训练或多次执行同一实验，不会删除既有候选和用户文件。
        protocol_paths = _config_mapping(
            getattr(self.cfg, "protocol_paths", None),
            "cfg.protocol_paths",
        )
        self.protocol_output_root = os.path.abspath(
            str(protocol_paths.get("output_root", ""))
        )
        if not str(protocol_paths.get("output_root", "")).strip():
            raise ValueError("cfg.protocol_paths.output_root is required")
        self.generated_dir = os.path.join(
            self.protocol_output_root,
            "generated",
        )
        os.makedirs(self.generated_dir, exist_ok=True)
        self.runtime_output_root = os.path.abspath(
            str(
                protocol_paths.get(
                    "runtime_output_root",
                    self.protocol_output_root,
                )
            )
        )
        self.response_dir = os.path.join(
            self.runtime_output_root,
            "responses",
        )
        self.stdout_dir = os.path.join(
            self.runtime_output_root,
            "candidate_stdout",
        )
        self.reflection_dir = os.path.join(
            self.runtime_output_root,
            "reflections",
        )
        for artifact_dir in (
            self.response_dir,
            self.stdout_dir,
            self.reflection_dir,
        ):
            os.makedirs(artifact_dir, exist_ok=True)

        # Hydra 命令行 override 已经合并进 self.cfg.problem。评价子进程不能继续
        # 读取仓库中的原始 YAML，否则 workflows_per_instance 等运行参数会被忽略。
        # 当前 Hydra run 目录彼此隔离，因此每次运行保存一个只读有效配置快照。
        self.effective_problem_config_path = os.path.join(
            self.protocol_output_root,
            f"effective_{self.problem}_config.yaml",
        )
        OmegaConf.save(
            config=self.cfg.problem,
            f=self.effective_problem_config_path,
            resolve=True,
        )
        
        # Load problem-specific prompt components
        # 加载问题相关的提示词组件
        prompt_path_suffix = "_black_box" if self.problem_type == "black_box" else "" # 如果是黑盒问题，变换加载方式
        problem_prompt_path = f'{self.prompt_dir}/{self.problem}{prompt_path_suffix}' #设置加载路径

        self.seed_func = file_to_string(f'{problem_prompt_path}/seed_func.txt') # 读取初始启发式函数 seed_function
        parameterized_reference_path = (
            f'{problem_prompt_path}/parameterized_seed_func.txt'
        )
        self.generation_reference_func = (
            file_to_string(parameterized_reference_path)
            if os.path.exists(parameterized_reference_path)
            else self.seed_func
        )
        self.func_signature = file_to_string(f'{problem_prompt_path}/func_signature.txt') # 告诉大模型：需要生成的函数应该长什么样
        self.func_desc = file_to_string(f'{problem_prompt_path}/func_desc.txt') # 读取函数描述文件
        
        # Load external knowledge if available
        # 如果存在外部知识文件，则加载外部知识
        external_knowledge_path = f'{problem_prompt_path}/external_knowledge.txt'  # 读取外部知识内容
        if os.path.exists(external_knowledge_path):
            self.external_knowledge = file_to_string(external_knowledge_path)
            self.long_term_reflection_str = self.external_knowledge
        else:
            self.external_knowledge = ""
        
        
        # Load common prompt templates
        # 加载通用提示词模板
        common_prompt_dir = f'{self.prompt_dir}/common'
        self.system_generator_prompt = file_to_string(f'{common_prompt_dir}/system_generator.txt')  # 读取系统生成器 prompt
        self.system_reflector_prompt = file_to_string(f'{common_prompt_dir}/system_reflector.txt')  # 读取系统反思器 prompt
        
        # Short-term reflection prompt
        # 短期反思提示词
        if self.problem_type != "black_box":
            self.user_reflector_st_prompt = file_to_string(f'{common_prompt_dir}/user_reflector_st.txt')
        else:
            self.user_reflector_st_prompt = file_to_string(f'{common_prompt_dir}/user_reflector_st_black_box.txt')
        
        # Long-term reflection and evolution prompts
        # 长期反思和进化相关提示词
        self.user_reflector_lt_prompt = file_to_string(f'{common_prompt_dir}/user_reflector_lt.txt')
        self.user_reflector_ise_prompt = file_to_string(f'{common_prompt_dir}/user_reflector_ise.txt')
        self.crossover_prompt = file_to_string(f'{common_prompt_dir}/crossover.txt')
        self.individual_self_evolution_prompt = file_to_string(f'{common_prompt_dir}/Individual_self_evolution.txt')
        self.mutation_prompt = file_to_string(f'{common_prompt_dir}/mutation.txt')
        self.candidate_repair_prompt = file_to_string(
            f'{common_prompt_dir}/candidate_repair.txt'
        )
        # Format user prompts with problem-specific information
        # 使用问题相关信息格式化用户提示词
        self.user_generator_prompt = file_to_string(f'{common_prompt_dir}/user_generator.txt').format(
            func_name=self.func_name,
            problem_desc=self.problem_desc,
            func_desc=self.func_desc,
        )
        self.seed_prompt = file_to_string(f'{common_prompt_dir}/seed.txt').format(
            seed_func=self.generation_reference_func,
            func_name=self.func_name,
        )

        # Flags to control prompt logging (print only once for the first iteration)
        # 控制提示词日志输出的标志（仅在第一次迭代时打印一次）
        self.print_crossover_prompt = True
        self.print_mutate_prompt = True
        self.print_short_term_reflection_prompt = True
        self.print_long_term_reflection_prompt = True
        self.print_individual_self_evolution_prompt = True
        self.print_individual_self_evolution_reflection_prompt = True


    def _runtime_artifact_path(self, category: str, file_name: str) -> str:
        """Return an absolute path inside this execution's Hydra directory."""
        directory_attributes = {
            "response": "response_dir",
            "stdout": "stdout_dir",
            "reflection": "reflection_dir",
        }
        if category not in directory_attributes:
            raise ValueError(f"unknown runtime artifact category: {category}")
        directory = getattr(
            self,
            directory_attributes[category],
            os.getcwd(),
        )
        os.makedirs(directory, exist_ok=True)
        name = os.path.basename(str(file_name))
        if name != str(file_name) or name in {"", ".", ".."}:
            raise ValueError("runtime artifact file_name must be a plain filename")
        return os.path.abspath(os.path.join(directory, name))



    def init_population(self) -> None:
        """Initialize population with seed function and LLM-generated individuals (train mode)
        or load trained rules from data directory (test mode).
        使用种子函数和 LLM 生成的个体初始化种群（训练模式），
        或从数据目录加载已训练规则（测试模式）。

        作用：
            根据运行模式初始化种群。训练模式下先评估种子函数，再调用 LLM 生成初始种群；
            测试模式下从 data_dir 加载已训练规则并评估。
        输入：
            无显式输入；依赖 self.mode、self.cfg、self.seed_func、self.case_num 等实例属性。
        输出：
            None。该函数会更新 self.population、self.seed_ind、self.elitist
            以及全局最优解相关属性。
        """
        
        if self.mode == "test":
            # Test mode: Load trained rules from data_dir
            # 测试模式：从 data_dir 加载已训练规则
            logging.info(f"Test mode: Loading trained rules from {self.cfg.data_dir}...")
            data_dir = os.path.join(self.root_dir, self.cfg.data_dir)
            
            if not os.path.exists(data_dir):
                raise RuntimeError(f"Data directory not found: {data_dir}")
            # 上面为确定规则目录，并确定目录是否存在

            # 创建临时种群和编号
            population = []
            response_id = 0
            
            # Load all .txt files from subdirectories
            # 加载data_dir 子目录中的所有 .txt 文件 每个.txt文件视为一条候选启发式规则
            for folder in sorted(os.listdir(data_dir)):
                folder_path = os.path.join(data_dir, folder)
                if os.path.isdir(folder_path):
                    for file in os.listdir(folder_path):
                        if file.endswith('.txt'):
                            file_path = os.path.join(folder_path, file)
                            # file_to_string(file_path)把整个文本文件读取为字符串
                            # extract_code_from_generator() 从文本中提取 Python 代码
                            code = extract_code_from_generator(file_to_string(file_path)) 
                            
                            # Convert each saved rule file into the same individual structure used in training.
                            # 将每个已保存规则文件转换为与训练阶段一致的个体结构。 将读取到的规则封装为统一个体
                            individual = {
                                "stdout_filepath": self._runtime_artifact_path(
                                    "stdout",
                                    f"problem_iter{self.iteration}_stdout{response_id}.txt",
                                ), # 运行评价程序时保存输出和报错的文件
                                "code_path": os.path.join(
                                    self.generated_dir,
                                    f"candidate_iter{self.iteration}_ind{response_id}.py",
                                ), # 候选代码的逻辑路径或追踪标识
                                "code": code, # 真正要评价的 Python 启发式函数
                                "response_id": response_id, # 个体编号
                            }
                            population.append(individual) # 将其加入测试种群
                            response_id += 1 
            # 检查是否读取到了规则
            if len(population) == 0:
                raise RuntimeError(f"No trained rules found in {data_dir}")
            
            logging.info(f"Loaded {len(population)} trained rules")
            
            # Evaluate loaded population
            # 评估加载得到的种群
            self.population = self.evaluate_population(population, self.case_num) # 放入调度环境中执行
            self.update_iter() # 更新测试种群的最优解
            
        else:
            # Train mode: Normal initialization with seed and LLM generation
            # 训练模式：使用种子函数和 LLM 生成结果进行正常初始化
            logging.info("Train mode: Evaluating seed function...")
            code = extract_code_from_generator(self.seed_func)
            if code is not None:
                # 只替换完整目标函数名，避免普通变量、注释或其他标识符中的
                # 字符串 "v1" 被误改。CEWS 的 v1 seed 由此变为可评价的 v2。
                code = code.replace(
                    f"{self.func_name}_v1",
                    f"{self.func_name}_v2"
                )
            logging.info("Seed function code: \n" + code)
            
            # The seed individual provides a baseline that later generated heuristics must outperform.
            # 构建种子个体， 种子个体作为基线，后续生成的启发式规则需要在其基础上改进。
            # 初始时self.iteration = 0， 所以文件名通常是 problem_iter0_stdout0.txt  problem_iter0_code0.py
            seed_ind = {
                "stdout_filepath": self._runtime_artifact_path(
                    "stdout",
                    f"problem_iter{self.iteration}_stdout0.txt",
                ),
                "code_path": os.path.join(
                    self.generated_dir,
                    f"candidate_iter{self.iteration}_ind0.py",
                ),
                "code": code,
                "response_id": 0,
            }
            self.seed_ind = seed_ind # 保存到对象属性 
            self.population = self.evaluate_population([self.seed_ind], self.case_num) # 评价种子个体

            # Validate seed function
            # 验证种子函数是否能够正常执行， 如果种子函数出现问题，程序停止，不再调用LLM
            if not self.seed_ind["exec_success"]:
                raise RuntimeError(
                    "Seed function is invalid. Please check the stdout file: "
                    + str(self.seed_ind["stdout_filepath"])
                )

            self.update_iter() # 更新测试种群的最优解
            
            # Generate initial population using LLM
            # 使用 LLM 生成初始种群
            system = self.system_generator_prompt #取得系统提示词 prompts/common/system_generator.txt
            # 拼接用户提示词 
            user = self.user_generator_prompt + "\n" + self.seed_prompt + "\n" + self.long_term_reflection_str 
            # 构造聊天消息格式 OpenAI 风格的 Chat Completion 消息结构
            messages = [
                {
                    "role": "system", "content": system
                },
                {
                    "role": "user", "content": user
                }
            ]
            logging.info("Initial Population Prompt: \nSystem Prompt: \n" + system + "\nUser Prompt: \n" + user)

            # Increase temperature for diverse initial population
            # 提高温度参数以增强初始种群多样性
            # 批量调用 LLM
            responses = multi_chat_completion(
                [messages], # 一组提示词
                self.cfg.init_pop_size, # 需要生成多少条响应
                self.cfg.model,  # 使用哪个大语言模型
                self.cfg.temperature + 0.3 # 实际生成温度
            )
            # 将LLM响应转换为个体
            population = self._responses_to_validated_individuals(
                responses,
                [messages],
            )

            # Evaluate generated population
            # 评价 LLM 生成的初始种群
            population = self.evaluate_population(population, self.case_num)

            if not any(individual.get("exec_success") for individual in population):
                logging.error(
                    "Every initial LLM candidate failed validation or execution; "
                    "activating the validated parameterized reference fallback."
                )
                fallback = self._validated_reference_fallback()
                if fallback is not None:
                    population.append(fallback)

            # Update iteration and population
            # 更新迭代计数和当前种群 用 LLM 种群替换当前种群
            self.population = population
            self.update_iter() # 更新测试种群的最优解

    # def post_thought(self, code, algorithm):

    #     prompt_content = self.get_prompt_refine(code, algorithm)
        
    #     return prompt_content

    # def get_prompt_refine(self, code, algorithm):

    #     prompt_content = "Dynamic deadline-aware cloud-edge workflow scheduling with constructive priority heuristics." + "\n"
    #     prompt_content += "The following describes the heuristic algorithm for the problem and the code with function name '" + self.func_name + "'.\n"
    #     prompt_content += "\nAlgorithm Design:\n" + algorithm
    #     if code == None:
    #         prompt_content += "\n\nCode:\n" + self.best_code_overall # 这帧数据是准备放弃的，没有意义我选择最好的个体，该数据无法正常运行肯定会报错
    #     else:            
    #         prompt_content += "\n\nCode:\n" + code

    #     prompt_content += "\n\nPlease summarize the algorithm's idea in **no more than 30 words** without any code or excessive detail. Be brief and clear."

    #     return prompt_content

    def _optimizer_config(self) -> OptimizerConfig:
        return getattr(
            self,
            "parameter_optimizer_config",
            OptimizerConfig(),
        )

    def _build_surrogate_manager(self) -> SurrogateManager:
        """Bind learned artifacts to the complete offline evaluation context."""
        problem = self._problem_config_dict()
        resources = problem.get("resources", {})
        protocol = getattr(self, "experiment_protocol_identity", {})
        training_scenarios = list(protocol.get("training_scenarios", []))
        context = SurrogateContext(
            schema_version=self.surrogate_config.schema_version,
            simulator_fingerprint=simulator_fingerprint(),
            evaluation_config_hash=rule_json_sha256(problem),
            resource_config_hash=rule_json_sha256(resources),
            protocol_identity=rule_json_sha256(protocol),
            scenario=",".join(str(item) for item in training_scenarios),
            domain=str(getattr(self, "problem", "")),
            ddl=str(problem.get("ddl", problem.get("deadline", ""))),
        )
        return SurrogateManager(
            self.surrogate_config,
            context,
            artifact_root=self.generated_dir,
        )

    def _parse_rule_candidate(self, individual: dict):
        """Attach RuleCandidate metadata without changing legacy rule behavior."""
        if individual.get("rule_candidate_parsed"):
            return individual.get("rule_candidate_object")
        source = individual.get("code")
        if source is None:
            raise RuleValidationError("candidate response contains no Python rule")
        config = self._optimizer_config()
        candidate = parse_rule_candidate(
            source,
            max_parameters=config.max_parameters,
            max_branches=config.max_branches,
            max_ast_depth=config.max_ast_depth,
            max_interactions=config.max_interactions,
        )
        individual["rule_candidate_parsed"] = True
        individual["rule_candidate_object"] = candidate
        individual["structure_id"] = candidate.structure_id
        individual["structure_hash"] = candidate.structure_hash
        individual["parameterized_rule_source"] = source
        individual["parameter_schema"] = (
            candidate.parameter_schema.as_dict()
            if candidate.parameter_schema is not None
            else None
        )
        individual["complexity"] = dict(candidate.complexity)
        individual["performance_tolerance"] = config.performance_tolerance
        return candidate

    def _problem_config_dict(self) -> dict:
        value = getattr(self.cfg, "problem", {})
        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
        return dict(value)

    def _optimization_seed_sets(self) -> tuple[list[int], list[int], list[int]]:
        config = self._problem_config_dict()
        dataset = config.get("dataset", {})
        train = (
            [int(seed) for seed in self.case_num]
            if self.case_num
            else [int(seed) for seed in dataset.get("train_seeds", [])]
        )
        validation = [
            int(seed) for seed in dataset.get("validation_seeds", [])
        ]
        final_test = [int(seed) for seed in dataset.get("test_seeds", [])]
        forbidden = LLM_EVOLUTION_FORBIDDEN_SEEDS.intersection(
            {*train, *validation}
        )
        if forbidden:
            raise ValueError(
                "LLM rule generation/CMA-ES cannot use reserved comparison "
                f"validation or final-test seeds: {sorted(forbidden)}"
            )
        return train, validation, final_test

    def _optimization_scenario_ids(self) -> list[str]:
        problem_config = self._problem_config_dict()
        configured = list(self._optimizer_config().scenario_ids)
        scenarios = configured or [
            str(problem_config.get("dataset", {}).get("scenario", "unknown"))
        ]
        return [str(item).strip().upper() for item in scenarios]

    def _shared_evaluation_executor(self):
        """Return the one pool that bounds every evaluation subprocess.

        All CMA-ES batches — including those of structures being optimized
        concurrently — submit here, so ``max_parallel_evaluations`` stays a
        hard, process-wide ceiling no matter how many structures are in
        flight. The pool is created on first use and reused for the whole run
        because thread startup is irrelevant next to a multi-second
        simulation, but repeatedly building and tearing down a pool per batch
        would serialize batches at their own barriers.
        """
        with self._evaluation_lock:
            if self._evaluation_executor is None:
                workers = max(
                    1,
                    int(self._optimizer_config().max_parallel_evaluations),
                )
                self._evaluation_executor = ThreadPoolExecutor(
                    max_workers=workers,
                    thread_name_prefix="parameter-eval",
                )
            return self._evaluation_executor

    def _diagnostic_trace_semaphore(self, config):
        """Bound aggregate trace storage across concurrently optimized structures."""
        with self._shared_state_lock:
            semaphore = getattr(self, "_trace_semaphore", None)
            if semaphore is None:
                slots = max(
                    1,
                    int(config.max_trace_mb_total)
                    // int(config.max_trace_mb_per_structure),
                )
                semaphore = threading.BoundedSemaphore(slots)
                self._trace_semaphore = semaphore
            return semaphore

    def _shutdown_evaluation_executor(self) -> None:
        """Release evaluation threads once the run is finished."""
        with self._evaluation_lock:
            executor = self._evaluation_executor
            self._evaluation_executor = None
        if executor is not None:
            executor.shutdown(wait=True)

    def _run_parameter_evaluation_context(
        self,
        command: list[str],
        scenario_id: str,
        seed: int,
    ) -> dict:
        """Run one isolated scenario/seed simulation without shared mutations."""
        child_env = os.environ.copy()
        child_env["PYTHONIOENCODING"] = "utf-8"
        child_env["PYTHONUTF8"] = "1"
        # Each subprocess handles small NumPy arrays. One BLAS thread per
        # process prevents nested oversubscription when several contexts run.
        child_env["OMP_NUM_THREADS"] = "1"
        child_env["MKL_NUM_THREADS"] = "1"
        child_env["OPENBLAS_NUM_THREADS"] = "1"
        child_env["NUMEXPR_NUM_THREADS"] = "1"
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=child_env,
            timeout=self.cfg.timeout,
            check=False,
        )
        output = (completed.stdout or "") + "\n" + (completed.stderr or "")
        traceback_message = filter_traceback(output)
        if completed.returncode != 0 or traceback_message:
            raise RuntimeError(
                "parameter evaluation failed for scenario/seed "
                f"{scenario_id}/{seed}: "
                f"{traceback_message or output[-1000:]}"
            )
        return parse_result_json(output)

    def _evaluate_parameter_map(
        self,
        candidate,
        parameters: dict[str, float],
        stage: str,
        seeds: list[int],
    ) -> dict:
        """Backward-compatible single-vector parameter evaluation."""
        return self._evaluate_parameter_maps(
            candidate,
            [parameters],
            stage,
            seeds,
        )[0]

    def _frozen_parameter_artifact(self, candidate, parameters):
        """Freeze and persist one parameter vector for evaluation or replay."""
        schema = candidate.parameter_schema
        values = [parameters[name] for name in schema.names]
        frozen_source = freeze_rule_source(
            candidate.parameterized_rule_source,
            schema,
            values,
        )
        source_hash = hashlib.sha256(frozen_source.encode("utf-8")).hexdigest()
        search_dir = os.path.join(
            self.generated_dir,
            "parameter_search",
            candidate.structure_hash[:16],
        )
        os.makedirs(search_dir, exist_ok=True)
        candidate_path = os.path.abspath(
            os.path.join(search_dir, f"params_{source_hash[:20]}.py")
        )
        if not os.path.isfile(candidate_path):
            with open(candidate_path, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(frozen_source)
        return values, frozen_source, source_hash, candidate_path

    def _build_diagnostic_replay_gate(self, candidate, trace_dir):
        scenario_ids = self._optimization_scenario_ids()
        problem_config = self._problem_config_dict()
        config_hashes = {
            scenario_id: rule_json_sha256(
                apply_seevo_scenario_config(
                    problem_config,
                    scenario_id,
                    problem_config.get("deadline_cache_paths", {}),
                    require_files=True,
                )
            )
            for scenario_id in scenario_ids
        }

        def replay(parameter_map, contexts):
            _values, _source, candidate_hash, candidate_path = (
                self._frozen_parameter_artifact(candidate, parameter_map)
            )
            paths = [
                os.path.join(trace_dir, f"{scenario_id}_seed{int(seed)}.npz")
                for scenario_id, seed in contexts
            ]
            loaded = replay_frozen_rule(
                paths,
                candidate_path,
                expected_config_hashes={
                    (str(scenario_id), int(seed)): config_hashes[str(scenario_id)]
                    for scenario_id, seed in contexts
                },
            )
            rule_error = next(
                (
                    verdict
                    for verdict in loaded.values()
                    if verdict.status == "rule_error"
                ),
                None,
            )
            verdicts = {}
            for context, path in zip(contexts, paths):
                verdict = loaded.get((str(context[0]), int(context[1])))
                if verdict is None:
                    verdict = ReplayVerdict(
                        scenario_id=str(context[0]),
                        seed=int(context[1]),
                        identical=False,
                        divergence_count_along_baseline=0,
                        first_divergence_index=None,
                        status=(
                            "rule_error"
                            if rule_error is not None
                            else "trace_mismatch"
                        ),
                        candidate_frozen_rule_hash=candidate_hash,
                    )
                verdicts[(str(context[0]), int(context[1]))] = verdict
            return verdicts

        return replay

    def _evaluate_parameter_maps(
        self,
        candidate,
        parameter_maps,
        stage: str,
        seeds: list[int],
        *,
        capture_trace_dir: str | None = None,
        bypass_cache_read: bool = False,
    ) -> list[dict]:
        """Evaluate a parameter batch through one bounded global context pool."""
        del stage  # Stage affects the common seed set, not scheduling semantics.
        parameter_maps = [dict(parameters) for parameters in parameter_maps]
        if not parameter_maps:
            return []
        if capture_trace_dir is not None and len(parameter_maps) != 1:
            raise ValueError("decision trace capture requires exactly one parameter vector")
        schema = candidate.parameter_schema
        problem_config = self._problem_config_dict()
        cache = getattr(self, "parameter_evaluation_cache", None)
        config = self._optimizer_config()
        scenario_ids = self._optimization_scenario_ids()
        trace_context_count = max(1, len(scenario_ids) * len(seeds))
        trace_max_mb = max(
            1,
            int(config.diagnostic_replay_gate.max_trace_mb_per_structure)
            // trace_context_count,
        )
        contexts = []
        context_groups = []
        for parameter_index, parameters in enumerate(parameter_maps):
            values, _frozen_source, _source_hash, candidate_path = (
                self._frozen_parameter_artifact(candidate, parameters)
            )
            group = []
            for scenario_id in scenario_ids:
                scenario_config = apply_seevo_scenario_config(
                    problem_config,
                    scenario_id,
                    problem_config.get("deadline_cache_paths", {}),
                    require_files=True,
                )
                evaluation_config_hash = rule_json_sha256(scenario_config)
                resource_config_hash = rule_json_sha256(
                    {
                        "resources": scenario_config.get("resources", {}),
                        "fuzzy": scenario_config.get("fuzzy", {}),
                    }
                )
                for seed in seeds:
                    cache_key = EvaluationCacheKey.create(
                        structure_hash=candidate.structure_hash,
                        parameter_vector=values,
                        seed=int(seed),
                        scenario_id=scenario_id,
                        evaluation_config_hash=evaluation_config_hash,
                        resource_config_hash=resource_config_hash,
                        precision=config.cache_precision,
                    )
                    if cache is None or bypass_cache_read:
                        cached = None
                    else:
                        # 缓存键含 structure_hash，同代结构互不共享条目，
                        # 因此加锁只影响命中统计与日志写入的原子性，不改命中结果。
                        with self._shared_state_lock:
                            cached = cache.get(cache_key)
                    group.append(len(contexts))
                    command_kwargs = {
                        "dataset_mode": "train",
                        "scenario_id": scenario_id,
                    }
                    if capture_trace_dir is not None:
                        command_kwargs.update(
                            {
                                "capture_trace_dir": capture_trace_dir,
                                "decision_trace_max_mb": (
                                    trace_max_mb
                                ),
                            }
                        )
                    contexts.append(
                        {
                            "parameter_index": parameter_index,
                            "scenario_id": scenario_id,
                            "seed": int(seed),
                            "cache_key": cache_key,
                            "cached": cached,
                            "command": (
                                None
                                if cached is not None
                                else self._evaluation_command(
                                    candidate_path,
                                    [int(seed)],
                                    **command_kwargs,
                                )
                            ),
                        }
                    )
            context_groups.append(group)

        pending = []
        result_source_indexes = {}
        pending_by_digest = {}
        for index, context in enumerate(contexts):
            if context["cached"] is not None:
                continue
            digest = context["cache_key"].digest
            source_index = pending_by_digest.get(digest)
            if source_index is None:
                source_index = index
                pending_by_digest[digest] = index
                pending.append((index, context))
            result_source_indexes[index] = source_index
        workers = min(
            max(1, int(config.max_parallel_evaluations)),
            max(1, len(pending)),
        )
        cached_count = sum(
            context["cached"] is not None for context in contexts
        )
        duplicate_count = len(contexts) - cached_count - len(pending)
        logging.info(
            "Parameter batch structure=%s vectors=%d contexts=%d cached=%d "
            "deduplicated=%d pending=%d workers=%d",
            candidate.structure_hash,
            len(parameter_maps),
            len(contexts),
            cached_count,
            duplicate_count,
            len(pending),
            workers if pending else 0,
        )
        completed_results = {}
        if workers == 1:
            for index, context in pending:
                completed_results[index] = self._run_parameter_evaluation_context(
                    context["command"],
                    context["scenario_id"],
                    context["seed"],
                )
        elif pending:
            # 共享池而不是每批新建：同代其他结构的批次也投在这里，
            # 于是 max_parallel_evaluations 是全进程上限，而不再是每批上限。
            executor = self._shared_evaluation_executor()
            future_to_index = {
                executor.submit(
                    self._run_parameter_evaluation_context,
                    context["command"],
                    context["scenario_id"],
                    context["seed"],
                ): index
                for index, context in pending
            }
            failure = None
            for future in as_completed(future_to_index):
                try:
                    completed_results[future_to_index[future]] = future.result()
                except BaseException as exc:  # noqa: BLE001 - 见下方说明
                    # 共享池不能像 with 语句那样在异常时关掉整个池——别的结构
                    # 还在用它。先把本批剩余结果收完再抛出第一个异常，
                    # 语义与原来 with 块内 future.result() 直接抛出一致。
                    if failure is None:
                        failure = exc
            if failure is not None:
                raise failure

        cache_entries = []
        aggregates = []
        cached_result_by_index = {}
        for group in context_groups:
            seed_results = []
            evaluation_seeds = []
            for index in group:
                context = contexts[index]
                result = context["cached"]
                if result is None:
                    source_index = result_source_indexes[index]
                    result = completed_results[source_index]
                    if source_index not in cached_result_by_index:
                        cache_entries.append((context["cache_key"], result))
                        cached_result_by_index[source_index] = result
                seed_results.append(result)
                evaluation_seeds.append(context["seed"])
            aggregate = aggregate_seed_evaluations(
                seed_results,
                evaluation_seeds,
            )
            aggregate["seeds"] = [int(seed) for seed in seeds]
            aggregate["scenario_ids"] = scenario_ids
            aggregate["evaluation_context_count"] = len(seed_results)
            aggregates.append(aggregate)
        with self._shared_state_lock:
            if cache is not None:
                cache.put_many(cache_entries)
            self.parameter_evaluation_count = int(
                getattr(self, "parameter_evaluation_count", 0)
            ) + len(pending)
        return aggregates

    def _write_parameter_artifacts(
        self,
        individual: dict,
        optimization_result: dict,
        diagnostics: dict,
    ) -> tuple[str, str]:
        artifact_dir = os.path.join(
            self.generated_dir,
            "optimization",
            str(individual["structure_hash"])[:16],
        )
        os.makedirs(artifact_dir, exist_ok=True)
        stem = (
            f"iter{self.iteration}_ind"
            f"{int(individual.get('response_id', 0))}"
        )
        history_path = os.path.abspath(
            os.path.join(artifact_dir, stem + "_search.json")
        )
        diagnostics_path = os.path.abspath(
            os.path.join(artifact_dir, stem + "_diagnostics.json")
        )
        with open(history_path, "w", encoding="utf-8") as handle:
            json.dump(
                optimization_result,
                handle,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
            )
        with open(diagnostics_path, "w", encoding="utf-8") as handle:
            json.dump(
                diagnostics,
                handle,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
            )
        return history_path, diagnostics_path

    def _warm_start_for(self, structure_hash: str):
        """Read this structure's warm start under the shared-state lock.

        Warm starts are written by the *previous* generation, and structure
        hashes are unique within a generation, so a concurrently prepared
        sibling can never write the key being read here. The lock only keeps
        the dict itself consistent.
        """
        with self._shared_state_lock:
            warm_starts = getattr(self, "parameter_warm_starts", {})
            value = warm_starts.get(structure_hash)
            return dict(value) if isinstance(value, dict) else value

    def _record_cross_generation_diagnostics(
        self,
        structure_hash: str,
        diagnostics: dict,
    ) -> dict:
        # 同代结构哈希互不重复，因此各自只碰自己那条记录；锁只保证
        # 字典本身在多结构并发准备时的完整性。
        with self._shared_state_lock:
            history_by_structure = getattr(
                self,
                "parameter_diagnostic_history",
                None,
            )
            if history_by_structure is None:
                history_by_structure = {}
                self.parameter_diagnostic_history = history_by_structure
            records = history_by_structure.setdefault(str(structure_hash), [])
            records.append(
                {
                    "iteration": int(self.iteration),
                    "diagnostics": json.loads(
                        json.dumps(diagnostics, ensure_ascii=True, allow_nan=False)
                    ),
                }
            )
            del records[:-20]
            records = list(records)
        result = dict(diagnostics)
        result["cross_generation_evidence"] = (
            accumulate_cross_generation_diagnostics(records)
        )
        return result

    def _prepare_individual_for_evaluation(self, individual: dict) -> dict:
        """Optimize a parameterized structure once, then expose only frozen code."""
        if individual.get("rule_prepared"):
            return individual
        if getattr(self, "problem", None) != "cews_task_constructive":
            individual["rule_prepared"] = True
            return individual
        candidate = self._parse_rule_candidate(individual)
        schema = candidate.parameter_schema
        if schema is None:
            candidate.frozen_rule_source = candidate.parameterized_rule_source
            candidate.final_code_hash = hashlib.sha256(
                candidate.frozen_rule_source.encode("utf-8")
            ).hexdigest()
            individual["rule_candidate"] = candidate.as_dict()
            individual["parameter_optimization_skipped"] = "legacy_rule_without_schema"
            individual["rule_prepared"] = True
            return individual

        config = self._optimizer_config()
        optimizer_config_hash = rule_json_sha256(config.as_dict())
        train_seeds, validation_seeds, final_test_seeds = (
            self._optimization_seed_sets()
        )
        if not train_seeds:
            raise RuleValidationError("parameter optimization has no training seeds")

        optimization = None
        if config.enabled and self.mode == "train":
            optimizer = CMAESOptimizer(config)
            replay_config = config.diagnostic_replay_gate
            trace_dir = os.path.join(
                self.generated_dir,
                "parameter_search",
                candidate.structure_hash[:16],
                "diagnostic_traces",
            )
            replay_gate = None
            trace_capture_evaluator = None
            trace_semaphore = None
            try:
                if replay_config.enabled:
                    trace_semaphore = self._diagnostic_trace_semaphore(
                        replay_config
                    )
                    trace_semaphore.acquire()
                    shutil.rmtree(trace_dir, ignore_errors=True)
                    os.makedirs(trace_dir, exist_ok=True)
                    replay_gate = self._build_diagnostic_replay_gate(
                        candidate,
                        trace_dir,
                    )
                    trace_capture_evaluator = (
                        lambda params, stage, seeds:
                        self._evaluate_parameter_maps(
                            candidate,
                            [params],
                            stage,
                            list(seeds),
                            capture_trace_dir=trace_dir,
                            bypass_cache_read=True,
                        )[0]
                    )
                optimization = optimizer.optimize(
                    schema,
                    lambda params, stage, seeds: self._evaluate_parameter_map(
                        candidate,
                        params,
                        stage,
                        list(seeds),
                    ),
                    batch_evaluator=(
                        lambda parameter_maps, stage, seeds:
                        self._evaluate_parameter_maps(
                            candidate,
                            parameter_maps,
                            stage,
                            list(seeds),
                        )
                    ),
                    train_seeds=train_seeds,
                    validation_seeds=validation_seeds,
                    final_test_seeds=final_test_seeds,
                    warm_start=self._warm_start_for(candidate.structure_hash),
                    surrogate_manager=getattr(self, "surrogate_manager", None),
                    surrogate_candidate=candidate,
                    surrogate_iteration=int(getattr(self, "iteration", 0)),
                    replay_gate=replay_gate,
                    trace_capture_evaluator=trace_capture_evaluator,
                    diagnostic_scenario_ids=self._optimization_scenario_ids(),
                )
            finally:
                if replay_config.enabled and replay_config.delete_traces_after_use:
                    shutil.rmtree(trace_dir, ignore_errors=True)
                if trace_semaphore is not None:
                    trace_semaphore.release()
            best_parameters = optimization.best_parameters
            diagnostics = generate_parameter_diagnostics(
                schema,
                optimization,
                boundary_epsilon=config.boundary_epsilon,
                sensitivity_epsilon=config.sensitivity_epsilon,
                correlation_threshold=config.correlation_threshold,
            )
            diagnostics = self._record_cross_generation_diagnostics(
                candidate.structure_hash,
                diagnostics,
            )
            optimization_payload = optimization.as_dict()
            optimization_payload["scenario_ids"] = (
                self._optimization_scenario_ids()
            )
            with self._shared_state_lock:
                self.parameter_warm_starts[candidate.structure_hash] = dict(
                    best_parameters
                )
            quick_metrics = individual.get("surrogate_initial_quick_metrics")
            quick_parameters = individual.get("surrogate_initial_parameters")
            manager = getattr(self, "surrogate_manager", None)
            if (
                manager is not None
                and getattr(manager, "enabled", False)
                and isinstance(quick_metrics, dict)
                and isinstance(quick_parameters, dict)
            ):
                try:
                    manager.record_structure_exact(
                        candidate,
                        quick_parameters,
                        quick_metrics,
                        optimization.best_metrics,
                        {
                            "iteration": int(getattr(self, "iteration", 0)),
                            "structure_hash": candidate.structure_hash,
                            "quick_comparison_key": list(
                                constraint_priority_key(quick_metrics)
                            ),
                            "final_comparison_key": list(
                                constraint_priority_key(optimization.best_metrics)
                            ),
                        },
                    )
                except Exception as exc:
                    logging.warning(
                        "Structure surrogate label write failed without affecting exact result: %s: %s",
                        type(exc).__name__,
                        exc,
                    )
        else:
            best_parameters = schema.values_dict(schema.initial_values)
            diagnostics = {
                "schema_version": "parameter_diagnostics_v1",
                "status": "not_run",
                "reason": (
                    "optimizer_disabled"
                    if not config.enabled
                    else "non_training_mode"
                ),
                "boundary_analysis": [],
                "inactivity_analysis": [],
                "correlation_analysis": {
                    "evidence": {"insufficient_samples": True},
                    "confidence": "low",
                    "suggested_structural_action": [
                        "run_offline_parameter_optimization_before_admission"
                    ],
                },
                "scenario_sensitivity": [],
                "fragility_analysis": {
                    "evidence": {"rule_fragile": None},
                    "confidence": "low",
                    "suggested_structural_action": [
                        "run_offline_parameter_optimization_before_admission"
                    ],
                },
            }
            optimization_payload = {
                "status": "not_run",
                "best_parameters": best_parameters,
                "history": [],
                "stage_seeds": {},
            }

        parameter_hash = rule_json_sha256(best_parameters)
        diagnostics_hash = rule_json_sha256(diagnostics)
        metadata = {
            "structure_hash": candidate.structure_hash,
            "parameter_schema_hash": schema.schema_hash,
            "best_parameter_hash": parameter_hash,
            "best_parameters": best_parameters,
            "optimizer_config_hash": optimizer_config_hash,
            "parameter_diagnostics_hash": diagnostics_hash,
            "optimizer_seed": config.optimizer_seed,
            "training_seeds": train_seeds,
            "validation_seeds": validation_seeds,
        }
        frozen_source = freeze_rule_source(
            candidate.parameterized_rule_source,
            schema,
            best_parameters,
            metadata=metadata,
        )
        frozen_hash = hashlib.sha256(frozen_source.encode("utf-8")).hexdigest()
        candidate.best_parameters = dict(best_parameters)
        candidate.frozen_rule_source = frozen_source
        candidate.evaluation_result = (
            dict(optimization.best_metrics) if optimization is not None else {}
        )
        candidate.parameter_diagnostics = diagnostics
        candidate.optimizer_config = config.as_dict()
        candidate.optimizer_seed = config.optimizer_seed
        candidate.final_code_hash = frozen_hash
        candidate.parameter_hash = parameter_hash

        history_path, diagnostics_path = self._write_parameter_artifacts(
            individual,
            optimization_payload,
            diagnostics,
        )
        individual.update(
            {
                "code": frozen_source,
                "frozen_rule_source": frozen_source,
                "best_parameters": dict(best_parameters),
                "best_parameter_hash": parameter_hash,
                "parameter_schema_hash": schema.schema_hash,
                "optimizer_config_hash": optimizer_config_hash,
                "parameter_diagnostics": diagnostics,
                "parameter_diagnostics_hash": diagnostics_hash,
                "frozen_rule_hash": frozen_hash,
                "parameter_optimization": optimization_payload,
                "parameter_search_history_path": history_path,
                "parameter_diagnostics_path": diagnostics_path,
                "rule_candidate": candidate.as_dict(),
                "rule_prepared": True,
            }
        )
        cache_stats = getattr(
            self,
            "parameter_evaluation_cache",
            EvaluationCache(enabled=False),
        ).stats()
        logging.info(
            "Parameter optimization structure=%s dim=%d generations=%s "
            "evaluations=%s stage_seeds=%s scenarios=%s best_ddl=%s best_energy=%s cache_hit_rate=%.3f "
            "stop=%s best_parameters=%s diagnostics=%s surrogate=%s",
            candidate.structure_hash,
            len(schema.parameters),
            optimization_payload.get("generations", 0),
            optimization_payload.get("evaluations", 0),
            optimization_payload.get("stage_seeds", {}),
            optimization_payload.get("scenario_ids", []),
            (
                optimization_payload.get("best_metrics", {}).get(
                    "max_deadline_violation_rate_across_seeds"
                )
            ),
            (
                optimization_payload.get("best_metrics", {}).get(
                    "fuzzy_total_energy_score"
                )
            ),
            cache_stats["hit_rate"],
            optimization_payload.get("stop_reason", "not_run"),
            best_parameters,
            diagnostics.get("summary", diagnostics.get("status")),
            optimization_payload.get("surrogate_stats", {}),
        )
        return individual

    
    def response_to_individual(self, response: str, response_id: int, file_name: str = None) -> dict:
        """Convert LLM response to an individual dictionary.
        将 LLM 响应转换为个体字典。

        作用：
            保存 LLM 原始响应，从响应中提取 Python 代码，并封装成统一的个体结构。
        输入：
            response: LLM 生成的文本响应，通常包含候选启发式代码。
            response_id: 当前响应或个体的编号。
            file_name: 可选的自定义文件名；不传入时按当前迭代自动生成。
        输出：
            dict。返回包含 stdout_filepath、code_path、code、response_id 的个体字典。
        """
        # Save response to file
        # 将响应保存到文件
        default_file_name = file_name is None
        if default_file_name:
            file_name = self._runtime_artifact_path(
                "response",
                f"problem_iter{self.iteration}_response{response_id}.txt",
            )
        else:
            file_name = os.path.abspath(file_name + ".txt")
            
        with open(file_name, 'w', encoding="utf-8") as file:
            file.write(response + '\n')

        # Extract code from response
        # 从响应中提取代码
        code = extract_code_from_generator(response)

        # Determine stdout filepath
        # 确定标准输出文件路径
        if default_file_name:
            std_out_filepath = self._runtime_artifact_path(
                "stdout",
                f"problem_iter{self.iteration}_stdout{response_id}.txt",
            )
        else:
            std_out_filepath = file_name[:-4] + "_stdout.txt"
        
        # Keep all files for one individual traceable through response_id.
        # 通过 response_id 保持单个个体的响应、代码和输出文件可追踪。
        individual = {
            "stdout_filepath": std_out_filepath,
            "code_path": os.path.join(
                getattr(self, "generated_dir", os.getcwd()),
                f"candidate_iter{self.iteration}_ind{response_id}.py",
            ),
            "code": code,
            "response_id": response_id,
            "response_filepath": os.path.abspath(file_name),
        }
        if (
            code is not None
            and getattr(self, "problem", None) == "cews_task_constructive"
        ):
            try:
                self._parse_rule_candidate(individual)
            except RuleValidationError as exc:
                individual["candidate_validation_error"] = str(exc)
                logging.warning(
                    "Candidate iteration=%s response_id=%s rejected before "
                    "simulation: %s (response=%s)",
                    self.iteration,
                    response_id,
                    exc,
                    individual["response_filepath"],
                )
        return individual

    def _candidate_generation_setting(self, name: str, default):
        """Read an optional candidate-generation setting without breaking old configs."""
        config = getattr(self.cfg, "candidate_generation", None)
        if config is None:
            return default
        if isinstance(config, dict):
            return config.get(name, default)
        return getattr(config, name, default)

    @staticmethod
    def _candidate_failure_reason(individual: dict) -> str | None:
        if individual.get("code") is None:
            return "response does not contain one complete ```python ... ``` block"
        return individual.get("candidate_validation_error")

    def _responses_to_validated_individuals(
        self,
        responses: list[str],
        messages_list: list[list[dict]],
    ) -> list[dict]:
        """Parse LLM responses and perform bounded, error-directed repair.

        Repair remains outside CMA-ES and simulation. Each retry receives the
        exact static-validation error and the rejected response, so it corrects
        the contract rather than silently changing parameter semantics.
        """
        if not responses:
            return []
        if len(messages_list) == 1:
            base_messages = [messages_list[0] for _ in responses]
        elif len(messages_list) == len(responses):
            base_messages = list(messages_list)
        else:
            raise ValueError(
                "messages_list must contain one shared prompt or one prompt per response"
            )

        max_retries = max(
            0,
            int(self._candidate_generation_setting("validation_retries", 0)),
        )
        repair_temperature = float(
            self._candidate_generation_setting("repair_temperature", 0.0)
        )
        def repair_one(response_id, response, original_messages):
            individual = self.response_to_individual(response, response_id)
            validation_history = []
            failure_reason = self._candidate_failure_reason(individual)

            for attempt in range(1, max_retries + 1):
                if failure_reason is None:
                    break
                validation_history.append(failure_reason)
                repair_instruction = self.candidate_repair_prompt.format(
                    validation_error=failure_reason
                )
                repair_messages = list(original_messages) + [
                    {"role": "assistant", "content": response},
                    {"role": "user", "content": repair_instruction},
                ]
                logging.info(
                    "Repairing candidate iteration=%s response_id=%s "
                    "attempt=%s/%s error=%s",
                    self.iteration,
                    response_id,
                    attempt,
                    max_retries,
                    failure_reason,
                )
                repaired = multi_chat_completion(
                    [repair_messages],
                    1,
                    self.cfg.model,
                    repair_temperature,
                )[0]
                repair_file = (
                    f"problem_iter{self.iteration}_response{response_id}"
                    f"_repair{attempt}"
                )
                response = repaired
                individual = self.response_to_individual(
                    repaired,
                    response_id,
                    repair_file,
                )
                failure_reason = self._candidate_failure_reason(individual)

            individual["candidate_generation_attempts"] = 1 + len(
                validation_history
            )
            individual["candidate_validation_history"] = validation_history
            if failure_reason is None and validation_history:
                logging.info(
                    "Candidate iteration=%s response_id=%s passed static "
                    "validation after %s repair attempt(s).",
                    self.iteration,
                    response_id,
                    len(validation_history),
                )
            elif failure_reason is not None:
                logging.error(
                    "Candidate iteration=%s response_id=%s exhausted %s repair "
                    "attempt(s): %s. No simulation stdout will be produced.",
                    self.iteration,
                    response_id,
                    max_retries,
                    failure_reason,
                )
            return response_id, individual

        workers = min(
            len(responses),
            max(
                1,
                int(
                    self._candidate_generation_setting(
                        "candidate_repair_workers",
                        20,
                    )
                ),
            ),
        )
        items = list(enumerate(zip(responses, base_messages)))
        if workers == 1:
            repaired = [
                repair_one(response_id, response, original_messages)
                for response_id, (response, original_messages) in items
            ]
        else:
            with ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="candidate-repair",
            ) as executor:
                repaired = list(
                    executor.map(
                        lambda item: repair_one(item[0], *item[1]),
                        items,
                    )
                )
        return [individual for _, individual in sorted(repaired)]

    def _validated_reference_fallback(self) -> dict | None:
        """Evaluate the checked parameterized reference only after batch failure."""
        enabled = bool(
            self._candidate_generation_setting(
                "use_validated_reference_fallback",
                False,
            )
        )
        if not enabled:
            return None
        cached = getattr(self, "generation_fallback_individual", None)
        if cached is not None and cached.get("exec_success"):
            return cached

        source = extract_code_from_generator(self.generation_reference_func)
        if source is None:
            raise RuntimeError(
                "Validated reference fallback contains no Python code block."
            )
        fallback = self.response_to_individual(
            "```python\n" + source + "\n```",
            -1,
            f"problem_iter{self.iteration}_validated_reference_fallback",
        )
        failure_reason = self._candidate_failure_reason(fallback)
        if failure_reason is not None:
            raise RuntimeError(
                "Configured parameterized reference fallback is invalid: "
                + failure_reason
            )
        fallback = self.evaluate_population([fallback], self.case_num)[0]
        if not fallback.get("exec_success"):
            raise RuntimeError(
                "Validated reference fallback failed scheduling evaluation; "
                f"inspect {fallback.get('stdout_filepath')}: "
                f"{fallback.get('traceback_msg', 'unknown error')}"
            )
        fallback["is_generation_reference_fallback"] = True
        self.generation_fallback_individual = fallback
        logging.info(
            "Validated reference fallback accepted structure=%s objective=%s",
            fallback.get("structure_hash"),
            fallback.get("obj"),
        )
        return fallback

    def _recover_population_from_validated_anchors(self) -> bool:
        """Recover an invalid generation using already evaluated offline anchors."""
        anchors = []
        fallback = self._validated_reference_fallback()
        candidates = [
            getattr(self, "elitist", None),
            getattr(self, "seed_ind", None),
            fallback,
        ]
        seen_sources = set()
        for candidate in candidates:
            if not candidate or not candidate.get("exec_success"):
                continue
            source_hash = hashlib.sha256(
                candidate.get("code", "").encode("utf-8")
            ).hexdigest()
            if source_hash in seen_sources:
                continue
            seen_sources.add(source_hash)
            anchors.append(candidate)
        if len(anchors) < 2:
            return False
        self.population = anchors
        logging.warning(
            "Recovered an all-invalid generated population with %s previously "
            "evaluated anchors; LLM/CMA evolution remains offline.",
            len(anchors),
        )
        return True

    def mark_invalid_individual(self, individual: dict, traceback_msg: str) -> dict:
        """
        Mark an individual as invalid.
        将个体标记为无效。

        作用：
            在代码无效、运行失败或解析目标值失败时，统一标记个体状态。
        输入：
            individual: 需要标记的个体字典。
            traceback_msg: 失败原因或 traceback 信息。
        输出：
            dict。返回更新后的个体字典，其中 exec_success=False，obj=inf。
        """
        # 无效个体的目标值设为正无穷。SeEvo 已把最大化目标转换为取负后的
        # 最小化比较，因此无论原问题方向如何，正无穷都代表最差个体。
        # metrics 清空可防止调用方误读上一次评价残留的指标。
        individual["exec_success"] = False
        individual["obj"] = float("inf")
        individual["metrics"] = {}
        individual["traceback_msg"] = traceback_msg
        return individual


    def _prepare_individuals_concurrently(
        self,
        population: list[dict],
        response_ids: list[int],
    ) -> None:
        """Run every structure's CMA-ES stage, overlapping them when allowed.

        Each structure is independent: cache keys, warm starts, diagnostic
        records and on-disk artifacts are all namespaced by ``structure_hash``,
        which is unique within a generation because duplicates were already
        rejected. Overlapping them therefore changes only scheduling, never a
        single evaluated value.

        Failures are recorded per structure with the same message the serial
        version produced, so one bad candidate cannot abort the generation.
        """
        if not response_ids:
            return
        concurrency = min(
            len(response_ids),
            max(1, int(self._optimizer_config().max_parallel_evaluations)),
        )
        if concurrency == 1:
            for response_id in response_ids:
                self._prepare_one_individual(population, response_id)
            return
        # 这个池只承载"等待自己那批评价结果"的个体线程，与
        # _shared_evaluation_executor 必须是两个池，否则等待方会占满执行方。
        with ThreadPoolExecutor(
            max_workers=concurrency,
            thread_name_prefix="structure-prepare",
        ) as executor:
            futures = [
                executor.submit(
                    self._prepare_one_individual,
                    population,
                    response_id,
                )
                for response_id in response_ids
            ]
            for future in futures:
                # _prepare_one_individual 自己吞掉候选级异常；这里 result()
                # 只会因为真正的内部缺陷而抛出，那种情况就该让整轮失败。
                future.result()

    def _apply_structure_surrogate_gate(
        self,
        population: list[dict],
        response_ids: list[int],
    ) -> list[int]:
        """Run the exact initial quick evaluation, then optionally screen CMA runs.

        Any feature, model, dataset, or evaluator failure returns the original
        response list.  Thus screening can only remove work after a complete,
        successful gate decision.
        """
        manager = getattr(self, "surrogate_manager", None)
        if manager is None or not getattr(manager, "enabled", False) or not response_ids:
            return response_ids
        parameterized = []
        candidates = []
        parameter_maps = []
        try:
            train_seeds, _, final_test_seeds = self._optimization_seed_sets()
            quick_count = int(
                dict(self._optimizer_config().stage_seed_counts).get("quick", 1)
            )
            quick_seeds = [int(seed) for seed in train_seeds[:quick_count]]
            if not quick_seeds or set(quick_seeds).intersection(final_test_seeds):
                raise ValueError("structure surrogate quick seeds are not train-only")
            for response_id in response_ids:
                candidate = population[response_id].get("rule_candidate_object")
                schema = getattr(candidate, "parameter_schema", None)
                if candidate is None or schema is None:
                    continue
                parameterized.append(response_id)
                candidates.append(candidate)
                parameter_maps.append(schema.values_dict(schema.initial_values))
            if not parameterized:
                return response_ids

            def evaluate_one(item):
                response_id, candidate, parameters = item
                metrics = self._evaluate_parameter_map(
                    candidate,
                    parameters,
                    "quick",
                    quick_seeds,
                )
                return response_id, dict(metrics)

            items = list(zip(parameterized, candidates, parameter_maps))
            workers = min(
                len(items),
                max(1, int(self._optimizer_config().max_parallel_evaluations)),
            )
            if workers == 1:
                evaluated = [evaluate_one(item) for item in items]
            else:
                with ThreadPoolExecutor(
                    max_workers=workers,
                    thread_name_prefix="structure-surrogate-quick",
                ) as executor:
                    evaluated = list(executor.map(evaluate_one, items))
            metrics_by_id = {response_id: metrics for response_id, metrics in evaluated}
            quick_metrics = [metrics_by_id[response_id] for response_id in parameterized]
            for response_id, parameters, metrics in zip(
                parameterized,
                parameter_maps,
                quick_metrics,
            ):
                population[response_id]["surrogate_initial_parameters"] = dict(parameters)
                population[response_id]["surrogate_initial_quick_metrics"] = dict(metrics)
            decision = manager.select_structures(
                candidates,
                parameter_maps,
                quick_metrics,
                iteration=int(getattr(self, "iteration", 0)),
            )
            selected_local = set(int(index) for index in decision["selected_indices"])
            selected_ids = {
                parameterized[index]
                for index in selected_local
                if 0 <= index < len(parameterized)
            }
            # Legacy rules are not representable by the parameter-aware model.
            selected_ids.update(set(response_ids) - set(parameterized))
            for local_index, response_id in enumerate(parameterized):
                population[response_id]["surrogate_decision"] = {
                    "gate_used": bool(decision.get("gate_used", False)),
                    "selected_for_full_cma": response_id in selected_ids,
                    "reasons": list(decision.get("reasons", {}).get(local_index, [])),
                    "failure_reason": str(decision.get("failure_reason", "")),
                }
                if response_id not in selected_ids:
                    population[response_id] = self.mark_invalid_individual(
                        population[response_id],
                        "Screened out by the offline structure surrogate.",
                    )
                    population[response_id]["surrogate_screened_out"] = True
            return [response_id for response_id in response_ids if response_id in selected_ids]
        except Exception as exc:
            logging.warning(
                "Structure surrogate failed open to exact CMA: %s: %s",
                type(exc).__name__,
                exc,
            )
            for response_id in response_ids:
                population[response_id]["surrogate_decision"] = {
                    "gate_used": False,
                    "selected_for_full_cma": True,
                    "reasons": ["fail_open"],
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                }
            return response_ids

    def _prepare_one_individual(
        self,
        population: list[dict],
        response_id: int,
    ) -> None:
        """Prepare one structure, marking it invalid on any candidate failure."""
        try:
            population[response_id] = self._prepare_individual_for_evaluation(
                population[response_id]
            )
        except Exception as exc:
            population[response_id] = self.mark_invalid_individual(
                population[response_id],
                f"Rule preparation failed: {type(exc).__name__}: {exc}",
            )

    def evaluate_population(self, population: list[dict], case_num: list) -> list[dict]:
        """Evaluate population by running code in parallel and computing objective values.
        通过并行运行代码并计算目标值来评估种群。

        作用：
            将种群中每个个体的代码写入评估文件并启动评估进程，随后读取输出、
            判断执行是否成功并解析目标值。
        输入：
            population: 待评估的个体字典列表，每个个体应包含 code 等字段。 一个个体字典列表
            case_num: 用于评估的算例编号列表。表示让每条启发式规则在这些调度实例上运行
        输出：
            list[dict]。返回更新后的种群，每个个体包含 obj、exec_success
            和 traceback_msg 等评估信息。
        """
        # 下标与 population 严格对齐；None 表示该个体在启动前或启动时已失败。
        inner_runs = [None] * len(population)  # 每个个体对应的评价子进程
        seen_structure_hashes = set()

        # 第一步（必须串行）：合法性与"同代重复结构"判定。重复判定依赖遍历
        # 顺序——先出现的保留、后出现的作废——并发化会改变哪个个体被作废。
        prepare_ids = []
        for response_id in range(len(population)):
            if population[response_id].get("candidate_validation_error"):
                population[response_id] = self.mark_invalid_individual(
                    population[response_id],
                    "Candidate validation failed: "
                    + population[response_id]["candidate_validation_error"],
                )
                continue
            # Skip if response contains no valid code
            # 如果响应中没有有效代码，则跳过该个体  LLM 的响应不一定始终包含合法 Python 代码
            if population[response_id]["code"] is None:
                population[response_id] = self.mark_invalid_individual(
                    population[response_id], "Invalid response!"
                )
                continue
            try:
                parsed_candidate = self._parse_rule_candidate(
                    population[response_id]
                )
                structure_hash = parsed_candidate.structure_hash
                if structure_hash in seen_structure_hashes:
                    population[response_id]["duplicate_structure_hash"] = (
                        structure_hash
                    )
                    population[response_id] = self.mark_invalid_individual(
                        population[response_id],
                        "Duplicate structure_hash in the same population: "
                        + structure_hash,
                    )
                    continue
                seen_structure_hashes.add(structure_hash)
            except Exception as exc:
                population[response_id] = self.mark_invalid_individual(
                    population[response_id],
                    f"Rule preparation failed: {type(exc).__name__}: {exc}",
                )
                continue
            prepare_ids.append(response_id)

        # 第二步（并发）：每个结构各跑一次 CMA-ES。单个批次只有
        # population_size × scenarios × stage_seeds 个上下文，填不满评价池；
        # 让多个结构同时准备，才能把共享池的槽位吃满。结构之间不共享任何
        # 结果——缓存键、暖启动、诊断记录、产物路径都以 structure_hash 区分，
        # 而同代结构哈希唯一，所以并发不改变任何个体的评价结果。
        prepare_ids = self._apply_structure_surrogate_gate(population, prepare_ids)
        self._prepare_individuals_concurrently(population, prepare_ids)

        # 第三步（串行）：按原下标顺序启动最终评价子进程，保持日志与
        # inner_runs 的顺序语义不变。
        for response_id in prepare_ids:
            if not population[response_id].get("rule_prepared"):
                # 准备阶段已在 _prepare_individuals_concurrently 里标记为无效。
                continue

            # 用于追踪当前正在启动哪个个体
            logging.info(f"Iteration {self.iteration}: Running Code {response_id}")

            try:
                # 每个个体由独立 Python 子进程评价，避免候选全局状态相互污染。
                process = self._run_code(population[response_id], response_id, case_num)
                inner_runs[response_id] = process
            except Exception as e:
                # 启动失败
                logging.info(f"Error for response_id {response_id}: {e}")
                population[response_id] = self.mark_invalid_individual(
                    population[response_id], str(e)
                )

        # Collect results and update population with objective values
        # 收集评价结果，并用目标值更新种群
        for response_id, inner_run in enumerate(inner_runs):
            if inner_run is None: # 启动失败
                continue
                
            # Wait for code execution to finish
            # 等待代码执行完成
            try:
                # 超时时间来自 cfg.timeout；候选若包含慢操作或死循环会被强制终止。
                inner_run.communicate(timeout=self.cfg.timeout)
            except subprocess.TimeoutExpired as e:
                logging.info(f"Timeout for response_id {response_id}: {e}")
                population[response_id] = self.mark_invalid_individual(
                    population[response_id], str(e)
                )
                inner_run.kill() # 强制终止子进程，并跳出后续输出解析
                continue
            
            # 读取个体的输出文件
            individual = population[response_id]
            stdout_filepath = individual["stdout_filepath"]
            
            # Read execution output
            # 读取执行输出
            # with open(stdout_filepath, 'r') as f:
            # 使用默认编码读取输出文件
            #     stdout_str = f.read()
            with open(stdout_filepath, "r", encoding="utf-8", errors="replace") as f:
                stdout_str = f.read()
            
            # Check for execution errors
            # 先检查 traceback。若 Python 执行已经失败，就不再尝试把日志误解析
            # 为有效 RESULT_JSON；没有 traceback 时仍需通过严格结果协议验证。
            traceback_msg = filter_traceback(stdout_str)
            
            # Parse objective value if no errors
            # 如果没有错误，则解析目标值
            if traceback_msg == '':
                try:
                    metrics = parse_result_json(stdout_str)
                    obj_value = float(metrics["objective"])
                    # SeEvo 的选择算子统一按较小 obj 更优。最小化问题保存原值；
                    # 最大化问题保存负值，但 metrics 中保留评价器输出的原 objective。
                    individual["obj"] = obj_value if self.obj_type == "min" else -obj_value
                    # 完整指标作为个体元数据保存；个体的可进化基因仍只有 code 字符串。
                    individual["metrics"] = metrics
                    individual["exec_success"] = True
                    individual["traceback_msg"] = ""
                    if isinstance(individual.get("rule_candidate"), dict):
                        individual["rule_candidate"]["evaluation_result"] = dict(
                            metrics
                        )
                except Exception as exc:
                    # 设置为无效
                    population[response_id] = self.mark_invalid_individual(
                        population[response_id], f"Invalid RESULT_JSON: {exc}"
                    )
            else:
                # Mark individual as invalid if execution failed
                # 如果执行失败，则将个体标记为无效
                population[response_id] = self.mark_invalid_individual(
                    population[response_id], traceback_msg
                )

            # 记录目标值日志
            logging.info(
                f"Iteration {self.iteration}, response_id {response_id}: "
                f"Objective value: {individual['obj']}"
            )
        counterfactual_config = getattr(
            self,
            "counterfactual_config",
            CounterfactualConfig(enabled=False),
        )
        if (
            getattr(self, "mode", None) == "train"
            and self.problem == "cews_task_constructive"
            and counterfactual_config.enabled
            and counterfactual_config.run_after_parameter_optimization
        ):
            analyzed = self._attach_counterfactual_feedback(population)
            replay_config = getattr(
                self,
                "critical_state_replay_config",
                CriticalStateReplayConfig(enabled=False),
            )
            if replay_config.enabled and replay_config.run_after_counterfactual_feedback:
                self._attach_critical_state_replay(analyzed)
        return population

    @staticmethod
    def _trace_from_dict(payload: dict) -> DecisionTrace:
        value = dict(payload)
        value["candidate_tasks"] = [
            CandidateTaskSnapshot(**row)
            for row in value.get("candidate_tasks", [])
        ]
        return DecisionTrace(**value)

    @staticmethod
    def _comparison_from_dict(payload: dict) -> CounterfactualComparison:
        value = dict(payload)
        value["selected_outcome"] = CounterfactualOutcome(
            **value["selected_outcome"]
        )
        value["alternative_outcome"] = CounterfactualOutcome(
            **value["alternative_outcome"]
        )
        return CounterfactualComparison(**value)

    def _counterfactual_analysis_seeds(self) -> list[int]:
        dataset = self.cfg.problem.dataset
        train_seeds = [int(seed) for seed in list(dataset.train_seeds)]
        validation_seeds = [
            int(seed)
            for seed in list(getattr(dataset, "validation_seeds", []))
        ]
        test_seeds = [int(seed) for seed in list(dataset.test_seeds)]
        seeds = (
            train_seeds[: self.counterfactual_config.train_seed_count]
            + validation_seeds[: self.counterfactual_config.validation_seed_count]
        )
        if not seeds:
            raise ValueError(
                "counterfactual feedback has no configured train/validation seeds"
            )
        if len(seeds) != len(set(seeds)):
            raise ValueError(
                "counterfactual train and validation seed selections overlap"
            )
        reject_test_seeds(seeds, test_seeds)
        return seeds

    def _select_counterfactual_structures(
        self,
        population: list[dict],
    ) -> list[dict]:
        eligible = [
            individual
            for individual in population
            if individual.get("exec_success")
            and individual.get("parameter_schema")
            and individual.get("frozen_rule_hash")
            and individual.get("code_path")
        ]
        overall_elites = sorted(
            eligible,
            key=individual_comparison_key,
        )[:1]
        feasible = sorted(
            [
                individual
                for individual in eligible
                if bool((individual.get("metrics") or {}).get("constraint_feasible", False))
            ],
            key=individual_comparison_key,
        )[: self.counterfactual_config.include_top_feasible]
        infeasible = sorted(
            [
                individual
                for individual in eligible
                if not bool((individual.get("metrics") or {}).get("constraint_feasible", False))
            ],
            key=lambda individual: (
                float((individual.get("metrics") or {}).get("objective", float("inf"))),
                individual_comparison_key(individual),
            ),
        )[: self.counterfactual_config.include_infeasible_low_energy]
        selected = []
        seen = set()
        for individual in [*overall_elites, *feasible, *infeasible]:
            structure_hash = str(individual["structure_hash"])
            if structure_hash in seen:
                continue
            selected.append(individual)
            seen.add(structure_hash)
            if len(selected) >= self.counterfactual_config.max_structures_per_generation:
                break
        return selected

    def _run_one_counterfactual_job(
        self,
        scenario_id: str,
        seed: int,
        command: list[str],
    ) -> dict:
        """Replay one frozen rule on one scenario/seed and return its manifest."""
        child_env = os.environ.copy()
        child_env["PYTHONIOENCODING"] = "utf-8"
        child_env["PYTHONUTF8"] = "1"
        # 与参数评价子进程同样限制 BLAS 线程，避免并发回放时嵌套超订。
        child_env["OMP_NUM_THREADS"] = "1"
        child_env["MKL_NUM_THREADS"] = "1"
        child_env["OPENBLAS_NUM_THREADS"] = "1"
        child_env["NUMEXPR_NUM_THREADS"] = "1"
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=child_env,
            timeout=self.cfg.timeout,
            check=False,
        )
        output = (completed.stdout or "") + "\n" + (completed.stderr or "")
        traceback_msg = filter_traceback(output)
        if completed.returncode != 0 or traceback_msg:
            raise RuntimeError(
                "counterfactual frozen-rule evaluation failed for "
                f"{scenario_id}/{seed}: "
                + (traceback_msg or output[-2000:])
            )
        result = parse_result_json(output)
        run_manifests = result.get("counterfactual_manifests", [])
        if len(run_manifests) != 1:
            raise RuntimeError(
                "counterfactual evaluator did not return exactly one run manifest"
            )
        manifest = dict(run_manifests[0])
        if bool(manifest.get("used_test_seed", True)):
            raise RuntimeError("counterfactual manifest is not test-seed isolated")
        return manifest

    def _run_counterfactual_jobs(self, jobs: list[tuple]) -> list[dict]:
        """Run replay jobs through the shared pool, returning manifests in order."""
        if not jobs:
            return []
        workers = min(
            len(jobs),
            max(1, int(self._optimizer_config().max_parallel_evaluations)),
        )
        if workers == 1:
            return [
                self._run_one_counterfactual_job(scenario_id, seed, command)
                for scenario_id, seed, command in jobs
            ]
        executor = self._shared_evaluation_executor()
        futures = [
            executor.submit(
                self._run_one_counterfactual_job,
                scenario_id,
                seed,
                command,
            )
            for scenario_id, seed, command in jobs
        ]
        manifests = []
        failure = None
        for future in futures:
            try:
                manifests.append(future.result())
            except BaseException as exc:  # noqa: BLE001
                # 共享池不能整体关闭，先等所有任务收敛再抛出第一个异常，
                # 与原串行版"首个失败即中止本结构分析"的可观察行为一致。
                if failure is None:
                    failure = exc
        if failure is not None:
            raise failure
        return manifests

    def _run_counterfactual_analysis(self, individual: dict) -> None:
        seeds = self._counterfactual_analysis_seeds()
        scenarios = list(self.counterfactual_config.scenario_ids)
        output_root = os.path.abspath(
            os.path.join(self.generated_dir, "counterfactual_feedback")
        )
        os.makedirs(output_root, exist_ok=True)
        manifests = []
        traces = []
        comparisons = []
        reports = []
        jobs = []
        for scenario_id in scenarios:
            for seed in seeds:
                metadata = {
                    "run_id": (
                        f"iter{self.iteration}:"
                        f"{str(individual['structure_hash'])[:16]}"
                    ),
                    "structure_hash": individual["structure_hash"],
                    "frozen_rule_hash": individual["frozen_rule_hash"],
                    "parameter_hash": individual.get("best_parameter_hash", ""),
                    "scenario_id": scenario_id,
                    "seed": seed,
                    "archive_ready_features": bool(
                        getattr(
                            self,
                            "critical_state_replay_config",
                            CriticalStateReplayConfig(enabled=False),
                        ).enabled
                    ),
                }
                command = self._evaluation_command(
                    os.path.abspath(individual["code_path"]),
                    [seed],
                    dataset_mode="train",
                    scenario_id=scenario_id,
                )
                command.extend(
                    [
                        "--counterfactual-config-json",
                        json.dumps(
                            self.counterfactual_config.to_dict(),
                            ensure_ascii=True,
                            allow_nan=False,
                            sort_keys=True,
                        ),
                        "--counterfactual-metadata-json",
                        json.dumps(
                            metadata,
                            ensure_ascii=True,
                            allow_nan=False,
                            sort_keys=True,
                        ),
                        "--counterfactual-output-dir",
                        output_root,
                    ]
                )
                jobs.append((scenario_id, seed, command))

        # 每个 (scenario, seed) 都是一次完整的冻结规则回放，产物文件按
        # structure_hash/scenario/seed 命名，互不读写对方的路径，因此可以并发。
        # 结果按 jobs 的原顺序取回，聚合输入序列逐条不变。
        # 唯一的例外是 counterfactual 估计缓存：它的路径只由 output_root 决定，
        # 不含上述三个字段，于是所有并发子进程会抢同一个 cache.json.tmp。
        # cfg 里已把 cache.enabled 关掉；再打开之前必须先把它按 job 分片。
        run_manifests = self._run_counterfactual_jobs(jobs)

        for manifest in run_manifests:
            manifests.append(manifest)
            traces.extend(
                self._trace_from_dict(row)
                for row in load_jsonl(manifest["trace_path"])
            )
            comparisons.extend(
                self._comparison_from_dict(row)
                for row in load_jsonl(manifest["comparison_path"])
            )
            with open(
                manifest["diagnostics_path"],
                "r",
                encoding="utf-8",
            ) as handle:
                reports.extend(
                    DiagnosticReport(**row) for row in json.load(handle)
                )

        used_test_seed = any(
            bool(manifest.get("used_test_seed", True)) for manifest in manifests
        )
        if used_test_seed:
            raise RuntimeError("counterfactual feedback used a final test seed")
        feedback = FeedbackAggregator(self.counterfactual_config).aggregate(
            reports,
            traces,
            comparisons,
            structure_hash=str(individual["structure_hash"]),
            frozen_rule_hash=str(individual["frozen_rule_hash"]),
        ).to_dict()
        summary_path = os.path.join(
            output_root,
            "summaries",
            str(individual["structure_hash"])[:16]
            + f"_iter{self.iteration}_feedback.json",
        )
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(
                feedback,
                handle,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                indent=2,
            )
        combined_manifest = {
            "manifest_version": "counterfactual_structure_manifest_v1",
            "structure_hash": individual["structure_hash"],
            "frozen_rule_hash": individual["frozen_rule_hash"],
            "parameter_hash": individual.get("best_parameter_hash", ""),
            "analyzed_seeds": seeds,
            "analyzed_scenarios": scenarios,
            "run_manifests": [row["manifest_path"] for row in manifests],
            "trace_hashes": [row["trace_hash"] for row in manifests],
            "diagnostics_hash": feedback["diagnostics_hash"],
            "counterfactual_feedback_hash": counterfactual_hash(feedback),
            "counterfactual_config_hash": self.counterfactual_config.config_hash,
            "used_test_seed": used_test_seed,
            "summary_path": summary_path,
        }
        combined_manifest["manifest_hash"] = counterfactual_hash(combined_manifest)
        manifest_path = os.path.join(
            output_root,
            "manifests",
            str(individual["structure_hash"])[:16]
            + f"_iter{self.iteration}_manifest.json",
        )
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(
                combined_manifest,
                handle,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                indent=2,
            )
        feedback_hash = counterfactual_hash(feedback)
        individual.update(
            {
                "counterfactual_trace_manifest": manifest_path,
                "counterfactual_feedback": feedback,
                "counterfactual_feedback_hash": feedback_hash,
                "counterfactual_config_hash": self.counterfactual_config.config_hash,
                "analyzed_scenarios": scenarios,
                "analyzed_seeds": seeds,
                "counterfactual_used_test_seed": used_test_seed,
                "representative_case_ids": [
                    row["decision_id"]
                    for row in feedback.get("representative_cases", [])
                ],
                "feedback_confidence": (
                    "high"
                    if feedback.get("high_confidence_structural_actions")
                    else (
                        "medium"
                        if feedback.get("medium_confidence_actions")
                        else "low"
                    )
                ),
                "counterfactual_max_feedback_chars": (
                    self.counterfactual_config.max_feedback_chars
                ),
            }
        )
        if isinstance(individual.get("rule_candidate"), dict):
            individual["rule_candidate"].update(
                {
                    "counterfactual_trace_manifest": manifest_path,
                    "counterfactual_feedback": feedback,
                    "counterfactual_feedback_hash": feedback_hash,
                    "counterfactual_config_hash": (
                        self.counterfactual_config.config_hash
                    ),
                    "analyzed_scenarios": scenarios,
                    "analyzed_seeds": seeds,
                    "counterfactual_used_test_seed": used_test_seed,
                }
            )
        logging.info(
            "Counterfactual feedback structure=%s runs=%d critical=%d comparisons=%d confidence=%s",
            str(individual["structure_hash"])[:16],
            len(manifests),
            int(feedback["analyzed_decisions"]),
            int(feedback["compared_alternatives"]),
            individual["feedback_confidence"],
        )

    def _attach_counterfactual_feedback(self, population: list[dict]) -> list[dict]:
        """Analyze only post-CMA frozen elites, never ordinary CMA vectors."""
        selected = self._select_counterfactual_structures(population)
        if len(selected) <= 1:
            for individual in selected:
                self._run_counterfactual_analysis(individual)
            return selected
        # 被选中的结构之间不共享可变状态：产物路径、汇总文件、写回的 individual
        # 字段全部按 structure_hash 区分，聚合也只吃自己那批 manifest。
        # 让它们并发，才能把每结构 scenarios × seeds 的回放一起填进共享池。
        # 注意这层并发会与内层的 scenario×seed 并发相乘，同时在跑的子进程数是
        # 两层的乘积——任何按 output_root（而非 structure_hash）定址的写入都会
        # 被它们同时命中，counterfactual 估计缓存正是因此关掉的。
        with ThreadPoolExecutor(
            max_workers=len(selected),
            thread_name_prefix="counterfactual",
        ) as executor:
            futures = [
                executor.submit(self._run_counterfactual_analysis, individual)
                for individual in selected
            ]
            for future in futures:
                future.result()
        return selected

    def _critical_state_archive_instance(self) -> CriticalStateArchive:
        """Lazily restore one experiment-local archive; missing old state is valid."""
        existing = getattr(self, "critical_state_archive", None)
        if existing is not None:
            return existing
        config = self.critical_state_replay_config
        archive_path = config.archive_path
        if not os.path.isabs(archive_path):
            archive_path = os.path.join(self.generated_dir, archive_path)
        train_seeds, validation_seeds, test_seeds = self._optimization_seed_sets()
        archive = CriticalStateArchive.load_or_create(
            config,
            path=archive_path,
            train_seeds=train_seeds,
            validation_seeds=validation_seeds,
            test_seeds=test_seeds,
        )
        self.critical_state_archive = archive
        return archive

    def _load_individual_counterfactual_evidence(
        self,
        individual: dict,
    ) -> tuple[list[DecisionTrace], list[CounterfactualComparison], list[DiagnosticReport]]:
        manifest_path = individual.get("counterfactual_trace_manifest")
        if not manifest_path or not os.path.isfile(manifest_path):
            return [], [], []
        with open(manifest_path, "r", encoding="utf-8") as handle:
            combined = json.load(handle)
        if bool(combined.get("used_test_seed", True)):
            raise RuntimeError("critical-state ingestion received a test-seed manifest")
        traces = []
        comparisons = []
        reports = []
        for run_manifest_path in combined.get("run_manifests", []):
            with open(run_manifest_path, "r", encoding="utf-8") as handle:
                run_manifest = json.load(handle)
            if bool(run_manifest.get("used_test_seed", True)):
                raise RuntimeError("critical-state ingestion received a test-seed run")
            traces.extend(
                self._trace_from_dict(row)
                for row in load_jsonl(run_manifest["trace_path"])
                if bool(row.get("is_critical", False))
            )
            comparisons.extend(
                self._comparison_from_dict(row)
                for row in load_jsonl(run_manifest["comparison_path"])
            )
            with open(run_manifest["diagnostics_path"], "r", encoding="utf-8") as handle:
                reports.extend(DiagnosticReport(**row) for row in json.load(handle))
        return traces, comparisons, reports

    def _attach_critical_state_replay(self, individuals: list[dict] | None) -> None:
        """Replay history first, then ingest this generation's trace evidence."""
        individuals = [
            row for row in (individuals or [])
            if row.get("exec_success")
            and row.get("parameter_schema")
            and row.get("frozen_rule_hash")
            and row.get("code_path")
        ]
        if not individuals:
            return
        config = self.critical_state_replay_config
        archive = self._critical_state_archive_instance()
        sampled_records = archive.sample_for_replay()
        historical_records = [
            type(record).from_dict(record.to_dict()) for record in sampled_records
        ]
        source_archive_hash = archive.archive_hash
        replay_root = archive.path.parent.parent
        cache_path = config.cache_path
        if not os.path.isabs(cache_path):
            cache_path = os.path.join(self.generated_dir, cache_path)
        replay_cache = CriticalStateReplayCache(
            config.cache_enabled,
            cache_path,
        )
        pending = []
        for individual in individuals:
            results = []
            if sampled_records:
                priority_function = load_frozen_priority_rule(individual["code_path"])
                replayer = CriticalStateReplayer(config, cache=replay_cache)
                results = [
                    replayer.replay(
                        record,
                        priority_function,
                        structure_hash=str(individual["structure_hash"]),
                        frozen_rule_hash=str(individual["frozen_rule_hash"]),
                        generation=int(self.iteration),
                        archive_hash=source_archive_hash,
                    )
                    for record in sampled_records
                ]
            pending.append((individual, results))

        for individual, results in pending:
            for result in results:
                archive.update_after_replay(
                    result,
                    replay_structure_hash=str(individual["structure_hash"]),
                )

        admission_counts = {"added": 0, "merged_exact_duplicate": 0, "rejected": 0}
        for individual in individuals:
            traces, comparisons, reports = self._load_individual_counterfactual_evidence(
                individual
            )
            for trace in traces:
                try:
                    record = build_critical_state_record(
                        trace,
                        comparisons,
                        reports,
                        generation=int(self.iteration),
                        config=config,
                    )
                except ValueError as exc:
                    logging.info(
                        "Critical state rejected decision=%s reason=%s",
                        trace.decision_id,
                        exc,
                    )
                    admission_counts["rejected"] += 1
                    continue
                if record is None:
                    admission_counts["rejected"] += 1
                    continue
                _stored, reason = archive.add_or_merge(record)
                admission_counts[reason if reason in admission_counts else "rejected"] += 1

        archive.compact(int(self.iteration))
        archive_manifest = archive.save(
            statistics_root=replay_root / "statistics",
        )
        final_archive_hash = archive.archive_hash

        for individual, results in pending:
            summary = aggregate_replay_results(
                results,
                historical_records,
                structure_hash=str(individual["structure_hash"]),
                frozen_rule_hash=str(individual["frozen_rule_hash"]),
                generation=int(self.iteration),
                archive_hash=source_archive_hash,
                config=config,
            ).to_dict()
            stem = (
                str(individual["structure_hash"])[:16]
                + f"_generation_{int(self.iteration)}"
            )
            replay_path = replay_root / "replays" / f"{stem}.jsonl"
            summary_path = replay_root / "summaries" / f"{stem}_summary.json"
            replay_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with replay_path.open("w", encoding="utf-8", newline="\n") as handle:
                for result in results:
                    handle.write(
                        json.dumps(
                            result.to_dict(),
                            ensure_ascii=True,
                            allow_nan=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
            summary_path.write_text(
                json.dumps(
                    summary,
                    ensure_ascii=True,
                    allow_nan=False,
                    sort_keys=True,
                    indent=2,
                ),
                encoding="utf-8",
            )
            failures = [
                row.state_id for row in results
                if row.replay_status == "VERIFIED_FAILURE"
            ]
            successes = [
                row.state_id for row in results
                if row.replay_status == "VERIFIED_SUCCESS"
            ]
            strict_failed = strict_replay_gate_triggered(
                results,
                historical_records,
                config,
            )
            individual.update(
                {
                    "critical_state_replay_summary": summary,
                    "critical_state_replay_hash": summary["summary_hash"],
                    "critical_state_config_hash": config.config_hash,
                    "critical_state_archive_hash": final_archive_hash,
                    "critical_state_archive_manifest": archive_manifest["manifest_path"],
                    "critical_state_replay_path": str(replay_path),
                    "critical_state_replay_summary_path": str(summary_path),
                    "replayed_state_ids": [row.state_id for row in results],
                    "replay_failure_state_ids": failures,
                    "replay_success_state_ids": successes,
                    "replay_confidence": (
                        "high" if summary["high_confidence_structural_actions"]
                        else "medium" if summary["medium_confidence_actions"]
                        else "low"
                    ),
                    "replay_generation": int(self.iteration),
                    "critical_state_max_feedback_chars": config.max_feedback_chars,
                    "critical_state_used_test_seed": False,
                    "critical_state_archive_version": ARCHIVE_VERSION,
                    "strict_replay_gate_triggered": strict_failed,
                }
            )
            if isinstance(individual.get("rule_candidate"), dict):
                individual["rule_candidate"].update(
                    {
                        "critical_state_replay_summary": summary,
                        "critical_state_replay_hash": summary["summary_hash"],
                        "critical_state_archive_hash": final_archive_hash,
                        "replayed_state_ids": [row.state_id for row in results],
                    }
                )
            if strict_failed:
                individual["exec_success"] = False
                individual["obj"] = float("inf")
                individual["traceback_msg"] = (
                    "Explicit strict critical-state replay gate rejected repeated HARD DDL failure"
                )
        logging.info(
            "Critical-state replay generation=%d archive=%d clusters=%d replayed=%d "
            "added=%d merged=%d rejected=%d cache_hit_rate=%.3f",
            int(self.iteration),
            len(archive.records),
            len(archive.clusters),
            len(sampled_records),
            admission_counts["added"],
            admission_counts["merged_exact_duplicate"],
            admission_counts["rejected"],
            replay_cache.hits / max(replay_cache.hits + replay_cache.misses, 1),
        )

    def _evaluation_command(
        self,
        candidate_path: str,
        case_num: list,
        *,
        dataset_mode: str | None = None,
        scenario_id: str | None = None,
        config_path: str | None = None,
        capture_trace_dir: str | None = None,
        decision_trace_max_mb: int = 128,
    ) -> list[str]:
        """Build the isolated CEWS evaluator command used by all rule evaluations."""
        eval_file_path = os.path.join(self.problem_dir, "eval.py")
        config_file_path = config_path or getattr(
            self,
            "effective_problem_config_path",
            os.path.join(
                self.root_dir,
                "cfg",
                "problem",
                f"{self.problem}.yaml",
            ),
        )
        selected_mode = dataset_mode or (
            "test" if self.mode == "test" else "train"
        )
        command = [
            sys.executable,
            "-u",
            eval_file_path,
            "--candidate",
            candidate_path,
            "--config",
            config_file_path,
            "--dataset-mode",
            selected_mode,
        ]
        if case_num:
            command.extend(["--cases", *[str(case) for case in case_num]])
        if scenario_id is not None:
            command.extend(["--scenario", str(scenario_id)])
        if capture_trace_dir is not None:
            command.extend(
                [
                    "--capture-decision-trace",
                    os.path.abspath(capture_trace_dir),
                    "--decision-trace-max-mb",
                    str(int(decision_trace_max_mb)),
                ]
            )
        return command

    def _finalize_best_rule_admission(self) -> dict | None:
        """Re-evaluate and register the best frozen rule through existing admission."""
        config = self._optimizer_config()
        if not config.auto_admission_enabled:
            return None
        individual = self.elitist
        if not individual or not individual.get("parameter_schema"):
            return None

        source_path = os.path.abspath(individual["code_path"])
        admission_config_path = config.admission_config_path
        manifest_path = config.admission_manifest_path
        if not os.path.isabs(admission_config_path):
            admission_config_path = os.path.join(
                self.root_dir,
                admission_config_path,
            )
        if not os.path.isabs(manifest_path):
            manifest_path = os.path.join(self.root_dir, manifest_path)
        admission_config_path = os.path.abspath(admission_config_path)
        manifest_path = os.path.abspath(manifest_path)
        if not os.path.isfile(admission_config_path):
            raise FileNotFoundError(admission_config_path)
        admission_cfg = OmegaConf.to_container(
            OmegaConf.load(admission_config_path),
            resolve=True,
        )
        required_seeds = [
            int(seed)
            for seed in admission_cfg.get("admission", {}).get(
                "required_evaluation_seeds",
                admission_cfg.get("dataset", {}).get("train_seeds", []),
            )
        ]
        if not required_seeds:
            raise ValueError("final admission has no configured evaluation seeds")
        final_test_seeds = {
            int(seed)
            for seed in admission_cfg.get("dataset", {}).get("test_seeds", [])
        }
        if final_test_seeds.intersection(required_seeds):
            raise ValueError("final admission must not use final test seeds")

        child_env = os.environ.copy()
        child_env["PYTHONIOENCODING"] = "utf-8"
        child_env["PYTHONUTF8"] = "1"
        admission_scenarios = [
            str(value).strip().upper()
            for value in admission_cfg.get(
                "admission_evaluation_scenarios",
                [admission_cfg.get("dataset", {}).get("scenario", "SS")],
            )
        ]
        validate_component_scenarios(
            self.experiment_protocol_context,
            "Rule Admission",
            admission_scenarios,
        )
        context_results = []
        context_seeds = []
        for scenario_id in admission_scenarios:
            for seed in required_seeds:
                command = self._evaluation_command(
                    source_path,
                    [seed],
                    dataset_mode="train",
                    scenario_id=scenario_id,
                    config_path=admission_config_path,
                )
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=child_env,
                    timeout=self.cfg.timeout,
                    check=False,
                )
                output = (completed.stdout or "") + "\n" + (completed.stderr or "")
                if completed.returncode != 0 or filter_traceback(output):
                    raise RuntimeError(
                        "final frozen-rule admission evaluation failed for "
                        f"{scenario_id}/{seed}: "
                        + (filter_traceback(output) or output[-2000:])
                    )
                context_results.append(parse_result_json(output))
                context_seeds.append(seed)
        result = aggregate_seed_evaluations(context_results, context_seeds)
        result["seeds"] = required_seeds
        result["evaluation_seed_count"] = len(required_seeds)
        result["completed_seed_count"] = len(required_seeds)
        result["scenario_ids"] = admission_scenarios
        result["experiment_protocol"] = dict(
            self.experiment_protocol_identity
        )
        result["evaluation_context_count"] = len(context_results)
        result["evaluation_config_sha256"] = rule_json_sha256(admission_cfg)
        result["scenario_id"] = str(
            admission_cfg.get("dataset", {}).get("scenario", "SS")
        )
        per_seed_metrics = []
        for seed in required_seeds:
            scenario_rows = []
            for context_result in context_results:
                rows = context_result.get("per_seed_metrics", [])
                if rows and int(rows[0].get("seed", -1)) == seed:
                    scenario_rows.append(dict(rows[0]))
            if len(scenario_rows) != len(admission_scenarios):
                raise RuntimeError(
                    f"incomplete multi-scenario admission rows for seed {seed}"
                )
            per_seed_metrics.append(
                {
                    "seed": seed,
                    "scenario_id": "multi:" + ",".join(admission_scenarios),
                    "completed_workflows": min(
                        int(row.get("completed_workflows", -1))
                        for row in scenario_rows
                    ),
                    "constraint_feasible": all(
                        bool(row.get("constraint_feasible", False))
                        for row in scenario_rows
                    ),
                    "deadline_violation_rate": max(
                        float(row.get("deadline_violation_rate", float("inf")))
                        for row in scenario_rows
                    ),
                    "total_lateness": float(
                        sum(float(row.get("total_lateness", 0.0)) for row in scenario_rows)
                    ),
                    "max_fuzzy_lateness": max(
                        float(row.get("max_fuzzy_lateness", float("inf")))
                        for row in scenario_rows
                    ),
                    "fuzzy_total_energy_score": float(
                        sum(
                            float(
                                row.get(
                                    "fuzzy_total_energy_score",
                                    row.get("objective", float("inf")),
                                )
                            )
                            for row in scenario_rows
                        )
                        / len(scenario_rows)
                    ),
                    "objective": float(
                        sum(float(row.get("objective", float("inf"))) for row in scenario_rows)
                        / len(scenario_rows)
                    ),
                    "scenario_metrics": scenario_rows,
                }
            )
        result["per_seed_metrics"] = per_seed_metrics
        for field_name in (
            "candidate_source_file",
            "candidate_sha256",
            "interface_valid",
            "function_name",
            "evaluator_protocol_version",
            "structure_hash",
            "parameter_schema_hash",
            "best_parameter_hash",
            "best_parameters",
            "optimizer_config_hash",
            "optimizer_seed",
            "frozen_rule_hash",
            "parameter_diagnostics_hash",
            "training_seeds",
            "validation_seeds",
        ):
            if field_name in context_results[0]:
                result[field_name] = context_results[0][field_name]
        if individual.get("counterfactual_feedback_hash"):
            result.update(
                {
                    "counterfactual_feedback_hash": individual[
                        "counterfactual_feedback_hash"
                    ],
                    "counterfactual_config_hash": individual[
                        "counterfactual_config_hash"
                    ],
                    "counterfactual_analyzed_seeds": list(
                        individual.get("analyzed_seeds", [])
                    ),
                    "counterfactual_analyzed_scenarios": list(
                        individual.get("analyzed_scenarios", [])
                    ),
                    "counterfactual_trace_manifest": individual.get(
                        "counterfactual_trace_manifest"
                    ),
                    "counterfactual_used_test_seed": bool(
                        individual.get("counterfactual_used_test_seed", True)
                    ),
                }
            )
        if individual.get("critical_state_replay_hash"):
            replay_summary = individual.get("critical_state_replay_summary") or {}
            result.update(
                {
                    "critical_state_archive_hash": individual[
                        "critical_state_archive_hash"
                    ],
                    "critical_state_replay_hash": individual[
                        "critical_state_replay_hash"
                    ],
                    "critical_state_config_hash": individual[
                        "critical_state_config_hash"
                    ],
                    "critical_state_replayed_state_count": int(
                        replay_summary.get("replayed_state_count", 0)
                    ),
                    "critical_state_verified_failure_count": int(
                        replay_summary.get("verified_failure_count", 0)
                    ),
                    "critical_state_used_test_seed": bool(
                        individual.get("critical_state_used_test_seed", True)
                    ),
                    "critical_state_archive_version": individual.get(
                        "critical_state_archive_version", ""
                    ),
                }
            )

        report_dir = os.path.join(os.path.dirname(manifest_path), "admission_reports")
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(
            report_dir,
            os.path.splitext(os.path.basename(source_path))[0] + "_admission.jsonl",
        )
        with open(report_path, "w", encoding="utf-8") as handle:
            handle.write(
                RESULT_JSON_PREFIX
                + json.dumps(
                    result,
                    ensure_ascii=True,
                    allow_nan=False,
                    sort_keys=True,
                )
                + "\n"
            )

        safe_hrl_root = os.path.abspath(os.path.join(self.root_dir, os.pardir))
        if safe_hrl_root not in sys.path:
            sys.path.insert(0, safe_hrl_root)
        from base.heuristic_admission import (  # pylint: disable=import-outside-toplevel
            append_admission_record,
            build_admission_record,
        )

        heuristic_id = (
            "seevo_cma_"
            + str(individual["structure_hash"])[:12]
            + "_"
            + str(individual["frozen_rule_hash"])[:12]
        )
        record = build_admission_record(
            heuristic_id=heuristic_id,
            display_name="SeEvo CMA frozen rule " + heuristic_id,
            source_path=source_path,
            evaluation_report_path=report_path,
            evaluation_config=admission_cfg,
            manifest_path=manifest_path,
            seevo_iteration=int(individual["candidate_iteration"]),
            seevo_individual=int(individual["candidate_individual"]),
            experiment_protocol=self.experiment_protocol_identity,
        )
        existing = None
        if os.path.isfile(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            validate_protocol_identity(
                self.experiment_protocol_context,
                manifest.get("experiment_protocol", {}),
                artifact_name="safe heuristic library",
            )
            existing = next(
                (
                    entry
                    for entry in manifest.get("llm_rules", [])
                    if entry.get("heuristic_id") == heuristic_id
                    and entry.get("source_hash") == record.get("source_hash")
                ),
                None,
            )
        if existing is None:
            append_admission_record(
                manifest_path,
                record,
                expected_protocol_identity=self.experiment_protocol_identity,
            )
        else:
            record = existing
        individual["final_admission"] = dict(record)
        individual["final_admission_report_path"] = report_path
        individual["final_admission_manifest_path"] = manifest_path
        logging.info(
            "Final frozen-rule admission structure=%s admitted=%s report=%s",
            individual.get("structure_hash"),
            record.get("admitted", False),
            report_path,
        )
        if config.admission_required and not bool(record.get("admitted", False)):
            raise RuntimeError(
                "best frozen rule failed final admission: "
                + ";".join(record.get("rejection_reasons", []))
            )
        return dict(record)

    def _run_code(self, individual: dict, response_id: int, case_num: list) -> subprocess.Popen:
        """Write code to file and execute evaluation script.
        将代码写入文件并执行评估脚本。

        作用：
            将每个候选代码写入独立模块，并把模块路径传给 eval.py 启动评估。
        输入：
            individual: 包含待执行代码和输出文件路径的个体字典。
            response_id: 当前个体编号，用于日志和输出文件追踪。
            case_num: 需要评估的算例编号列表。
        输出：
            subprocess.Popen。返回正在运行的评估子进程句柄。
        """
        logging.debug(f"Iteration {self.iteration}: Processing Code Run {response_id}")
        
        # 文件名同时包含迭代号和稳定个体号：并发子进程只读取各自文件，不会覆盖
        # 或导入共享 gpt.py。优先使用 individual.response_id，可在种群重排后继续
        # 保持候选身份；缺失时退回当前列表下标 response_id。
        individual_id = int(individual.get("response_id", response_id))
        candidate_path = os.path.abspath(os.path.join(
            self.generated_dir,
            f"candidate_iter{self.iteration}_ind{individual_id}.py",
        ))
        # UTF-8 支持 LLM 生成的中文注释。flush + fsync 确保启动子进程之前文件内容
        # 已对其他进程可见，避免高速并发下读取到未完全写入的候选。
        with open(
            candidate_path,
            'w',
            encoding='utf-8',
            newline='\n',
        ) as file:
            source = individual["code"]
            file.write(source if source.endswith("\n") else source + "\n")
            file.flush()  # Ensure buffer is written to disk / 确保缓冲区写入磁盘
            os.fsync(file.fileno())  # Force OS to write buffer to disk / 强制操作系统落盘
        individual["code_path"] = candidate_path
        individual["candidate_iteration"] = int(self.iteration)
        individual["candidate_individual"] = int(individual_id)
        if individual.get("parameter_schema"):
            logging.info(
                "Frozen rule structure=%s path=%s hash=%s",
                individual.get("structure_hash"),
                candidate_path,
                individual.get("frozen_rule_hash"),
            )

        command = self._evaluation_command(candidate_path, case_num)
        
        # stdout 与 stderr 合并进该个体独有日志；既保留 traceback，也保证最终
        # RESULT_JSON 能被 evaluate_population 从同一文本中读取。
        # 子进程直接继承 Windows 文件句柄时不会使用父进程 TextIOWrapper 的
        # encoding，默认可能按 GBK 写中文路径。显式设置 Python I/O 编码，保证
        # eval.py 的 stdout/stderr 与父进程读取端统一为 UTF-8。
        child_env = os.environ.copy()
        child_env["PYTHONIOENCODING"] = "utf-8"
        child_env["PYTHONUTF8"] = "1"
        with open(individual["stdout_filepath"], 'w', encoding='utf-8') as f:
            process = subprocess.Popen(
                command,
                stdout=f, 
                stderr=f,
                env=child_env,
            )

        # Wait until evaluation starts
        # 等待评估进程开始输出
        block_until_running(
            individual["stdout_filepath"], 
            log_status=True, 
            iter_num=self.iteration, 
            response_id=response_id
        )
        return process
    
    def update_iter(self) -> None:
        """Update statistics and advance iteration counter.
        更新统计信息并推进迭代计数器。

        作用：
            根据当前种群目标值更新全局最优解、精英个体、改进历史和迭代编号。
        输入：
            无显式输入；使用 self.population 和当前训练/测试状态。
        输出：
            None。该函数更新 self.best_obj_overall、self.elitist、
            self.improvement_history、self.best_obj_history 和 self.iteration。
        """
        population = self.population
        # 不能直接对纯能耗 obj 取最小值，否则低能耗但违反 DDL 的规则可能击败
        # 能耗略高的可行规则。比较键先处理可行性，再处理约束违反，最后才是 obj。
        best_sample_idx = min(
            range(len(population)),
            key=lambda index: individual_comparison_key(population[index]),
        )
        best_individual = population[best_sample_idx]
        best_obj = float(best_individual["obj"])
        best_key = individual_comparison_key(best_individual)
        
        # Track improvement (train mode only)
        # 跟踪改进情况（仅训练模式）
        if self.mode == "train":
            previous_best = self.best_obj_overall
            previous_key = getattr(self, "best_comparison_key_overall", None)
            
            # Update best overall solution
            # 更新全局最优解
            if previous_key is None or best_key < previous_key:
                # 只有在约束层级相同的情况下，能耗差才有同一物理意义。
                # 若本次改进来自约束可行性/违反程度，记为 0，避免把秒、比例和焦耳相加。
                improvement = (
                    0
                    if previous_best is None or previous_key[:3] != best_key[:3]
                    else max(0.0, previous_best - best_obj)
                )
                self.improvement_history.append(improvement)
                self.best_obj_overall = best_obj
                self.best_comparison_key_overall = best_key
                self.best_code_overall = best_individual["code"]
                self.best_code_path_overall = best_individual["code_path"]
                logging.info(f"Iteration {self.iteration}: Improvement = {improvement:.4f}")
            else:
                self.improvement_history.append(0)
            
            # Track best objective history
            # 记录最优目标值历史
            self.best_obj_history.append(best_obj)
        else:
            # Test mode: just update best solution without tracking history
            # 测试模式：只更新最优解，不记录历史趋势
            previous_key = getattr(self, "best_comparison_key_overall", None)
            if previous_key is None or best_key < previous_key:
                self.best_obj_overall = best_obj
                self.best_comparison_key_overall = best_key
                self.best_code_overall = best_individual["code"]
                self.best_code_path_overall = best_individual["code_path"]
        
        # Update elitist individual
        # 更新精英个体
        if (
            self.elitist is None
            or best_key < individual_comparison_key(self.elitist)
        ):
            self.elitist = best_individual
            logging.info(f"Iteration {self.iteration}: New Elitist: {self.elitist['obj']}")
        
        # Log iteration summary
        # 记录迭代摘要
        logging.info(f"Iteration {self.iteration} finished...")
        logging.info(
            f"Best obj: {self.best_obj_overall}, "
            f"Constraint key: {self.best_comparison_key_overall}, "
            f"Best Code Path: {self.best_code_path_overall}"
        )
        logging.info(f"Function Evals: {self.function_evals}")
        self.iteration += 1
        
    def rank_select(self, population: list[dict]) -> list[dict]:
        """Rank-based selection with probability proportional to rank.
        基于排名的选择方法，选择概率与排名相关。

        作用：
            先过滤有效个体，再按目标值排名计算选择概率，生成用于交叉的父代配对。
        输入：
            population: 候选个体列表。
        输出：
            list[dict] | None。成功时返回长度为 2 * pop_size 的父代列表；
            若有效个体不足或多次采样失败，则返回 None。
        """
        # Filter valid individuals
        # 过滤有效个体
        if self.problem_type == "black_box":
            # In black-box mode, only individuals better than the seed are considered useful.
            # 在黑盒模式下，仅保留优于种子个体的候选解。
            population = [
                ind for ind in population 
                if ind["exec_success"]
                and individual_comparison_key(ind)
                < individual_comparison_key(self.seed_ind)
            ]
        else:
            population = [ind for ind in population if ind["exec_success"]]
            
        if len(population) < 2:
            return None
            
        # 按可行性优先键排序；无约束问题自动退化为按 obj 排序。
        population = sorted(population, key=individual_comparison_key)
        ranks = list(range(len(population)))
        probs = [1 / (rank + 1 + len(population)) for rank in ranks]
        probs = [prob / sum(probs) for prob in probs]  # Normalize / 归一化
        
        # Select parents
        # 选择父代个体
        selected_population = []
        trial = 0
        while len(selected_population) < 2 * self.cfg.pop_size:
            trial += 1
            parents = np.random.choice(population, size=2, replace=False, p=probs)
            # 比较键不同才有明确的优劣关系，可供反思器学习。
            if individual_comparison_key(parents[0]) != individual_comparison_key(parents[1]):
                selected_population.extend(parents)
            if trial > 1000:
                return None
        return selected_population
    
    
    def random_select(self, population: list[dict]) -> list[dict]:
        """Random selection with equal probability for all individuals.
        对所有个体进行等概率随机选择。

        作用：
            过滤有效个体后，等概率随机抽取目标值不同的父代配对。
        输入：
            population: 候选个体列表。
        输出：
            list[dict] | None。成功时返回长度为 2 * pop_size 的父代列表；
            若有效个体不足或多次采样失败，则返回 None。
        """
        # Filter valid individuals
        # 过滤有效个体
        if self.problem_type == "black_box":
            # In black-box mode, discard valid but non-improving individuals.
            # 在黑盒模式下，丢弃虽然可运行但没有优于种子解的个体。
            population = [
                ind for ind in population 
                if ind["exec_success"]
                and individual_comparison_key(ind)
                < individual_comparison_key(self.seed_ind)
            ]
        else:
            population = [ind for ind in population if ind["exec_success"]]
            
        if len(population) < 2:
            return None
            
        # Randomly select parent pairs
        # 随机选择父代配对
        selected_population = []
        trial = 0
        while len(selected_population) < 2 * self.cfg.pop_size:
            trial += 1
            parents = np.random.choice(population, size=2, replace=False)
            # 约束状态或纯目标至少一项不同，才形成有意义的父代对。
            if individual_comparison_key(parents[0]) != individual_comparison_key(parents[1]):
                selected_population.extend(parents)
            if trial > 1000:
                return None
        return selected_population

    def population_inter_envoltion_reflection_prompt(self, ind1: dict, ind2: dict) -> tuple[list[dict], str, str]:
        """Generate short-term reflection prompt by comparing two individuals.
        通过比较两个个体生成短期反思提示词。

        作用：
            比较两个父代个体的目标值，识别较优/较差代码，并构造用于短期反思的 LLM prompt。
        输入：
            ind1: 第一个父代个体字典。
            ind2: 第二个父代个体字典。
        输出：
            tuple[list[dict], str, str]。返回 LLM 消息列表、较差个体代码片段和较优个体代码片段。
        """
        ind1_key = individual_comparison_key(ind1)
        ind2_key = individual_comparison_key(ind2)
        if ind1_key == ind2_key:
            raise ValueError(
                "Two individuals to crossover have the same objective and "
                f"constraint status: {ind1_key}"
            )
        
        # Determine which individual is better
        # 判断哪个个体更优
        if ind1_key < ind2_key:
            better_ind, worse_ind = ind1, ind2
        else:
            better_ind, worse_ind = ind2, ind1

        # Strip signatures/imports so the prompt focuses on heuristic logic.
        # 去除函数签名和导入语句，使提示词聚焦于启发式逻辑本身。
        worse_code = rule_source_for_evolution(worse_ind)
        better_code = rule_source_for_evolution(better_ind)
        
        # 约束优化不能把违反率、延期秒数和焦耳合成一个“改进百分比”。
        # 直接向反思器展示两者的能耗和约束状态，并声明采用可行性优先比较。
        performance_context = (
            f"\n\n**Performance Comparison:**\n"
            f"- Worse code: {individual_performance_summary(worse_ind)}\n"
            f"- Better code: {individual_performance_summary(better_ind)}\n"
            "- Ranking is feasibility-first: satisfy all DDL constraints before "
            "minimizing the pure energy objective. Among infeasible rules, reduce "
            "violation rate and total lateness before comparing energy.\n"
            "- Worse parameter, current counterfactual, and historical replay evidence (JSON):\n"
            f"{parameter_feedback_summary(worse_ind)}\n"
            "- Better parameter, current counterfactual, and historical replay evidence (JSON):\n"
            f"{parameter_feedback_summary(better_ind)}\n"
            "Use parameter-space, current decision-mechanism, and persistent "
            "historical stress-state evidence jointly to "
            "propose a bounded structural change; do not merely edit optimized "
            "numeric values. Correlation is not causation, local counterfactuals "
            "and feature replay are not full simulation or strict causal proof, "
            "and strong changes require repeated "
            "cross-decision, cross-generation, cross-seed, or cross-scenario evidence.\n"
            "A replay failure is diagnostic and does not replace full scheduling evaluation; "
            "every changed structure must re-enter CMA-ES, full simulation, and replay.\n"
        )
        
        # Create reflection prompt
        # 创建反思提示词
        system = self.system_reflector_prompt
        user = self.user_reflector_st_prompt.format(
            func_name=self.func_name,
            func_desc=self.func_desc,
            problem_desc=self.problem_desc,
            worse_code=worse_code,
            better_code=better_code
        ) + performance_context
        message = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        
        # Log prompt for the first iteration only
        # 仅在首次迭代记录该提示词
        if self.print_short_term_reflection_prompt:
            logging.info("Short-term Reflection Prompt: \nSystem Prompt: \n" + system + "\nUser Prompt: \n" + user)
            self.print_short_term_reflection_prompt = False
        return message, worse_code, better_code
    
    def individual_self_evolution_reflection_prompt(self, ind: dict, ind1: dict, ind2: dict, older_prompt: str) -> tuple[list[dict], str, str]:
        """Generate reflection prompt for individual self-evolution.
        为个体自进化生成反思提示词。

        作用：
            将交叉生成的新个体与较优父代比较，结合上一阶段反思内容，
            构造个体自进化反思 prompt。
        输入：
            ind: 当前待自进化的新个体。
            ind1: 第一个父代个体。
            ind2: 第二个父代个体。
            older_prompt: 种群间进化阶段产生的上一轮反思内容。
        输出：
            tuple[list[dict], str, str]。返回 LLM 消息列表、较优父代代码片段和当前新代码片段。
        """
        # Determine which parent is better
        # 判断哪个父代更优
        better_ind = (
            ind1
            if individual_comparison_key(ind1) < individual_comparison_key(ind2)
            else ind2
        )

        # Compare current individual with better parent
        # 将当前个体与较优父代进行比较
        better_key = individual_comparison_key(better_ind)
        current_key = individual_comparison_key(ind)
        if better_key < current_key:
            result = "worse"
        elif better_key == current_key:
            result = "same"
        else:
            result = "better"

        # Provide compact code snippets to the reflector for targeted diagnosis.
        # 向反思器提供精简代码片段，便于进行有针对性的诊断。
        better_code = rule_source_for_evolution(better_ind)
        new_code = rule_source_for_evolution(ind)

        performance_context = (
            "\n\n**Detailed Performance Comparison:**\n"
            f"- Better parent: "
            f"{individual_performance_summary(better_ind)}\n"
            f"- Current offspring: "
            f"{individual_performance_summary(ind)}\n"
            "- Better parent parameter/current-decision/historical-replay evidence (JSON):\n"
            f"{parameter_feedback_summary(better_ind)}\n"
            "- Offspring parameter/current-decision/historical-replay evidence (JSON):\n"
            f"{parameter_feedback_summary(ind)}\n"
            "Prioritize a structural modification (feature, interaction, "
            "normalization, gate, or nonlinear/piecewise form), not a numeric-only "
            "change. Re-evaluate every changed structure and keep complexity bounded.\n"
            "Persistent HARD-state failures and resolved-state regressions support "
            "feature/interaction/normalization/gate/piecewise changes, never manual "
            "continuous weights; the new structure must run CMA-ES again.\n"
            "- Ranking is feasibility-first. A DDL-feasible rule always "
            "outranks an infeasible rule.\n"
            "- For two feasible rules, reduce fuzzy energy mean and fuzzy "
            "energy standard deviation without violating DDL constraints.\n"
            "- Cross-seed statistics are diagnostic only and are not part "
            "of the optimization objective.\n"
        )

        # Create reflection prompt
        # 创建反思提示词
        system = self.system_reflector_prompt
        user = self.user_reflector_ise_prompt.format(
            func_name=self.func_name,
            func_desc=self.func_desc,
            problem_desc=self.problem_desc,
            result=result,
            first_version_code=better_code,
            second_version_code=new_code,
            hint=older_prompt
        ) + performance_context
        message = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        
        # Log prompt for the first iteration only
        # 仅在首次迭代记录该提示词
        if self.print_individual_self_evolution_reflection_prompt:
            logging.info("Individual Self Evolution Reflection Prompt: \nSystem Prompt: \n" + system + "\nUser Prompt: \n" + user)
            self.print_individual_self_evolution_reflection_prompt = False
        return message, better_code, new_code  

    def population_inter_envoltion_reflection(self, population: list[dict]) -> tuple[list[str], list[str], list[str]]:
        """Perform short-term reflection for population inter-evolution.
        为种群间进化执行短期反思。

        作用：
            对已选择的父代两两配对，为每一对生成短期反思 prompt，并批量调用 LLM 获取反思结果。
        输入：
            population: 已选择的父代个体列表，长度应为偶数。
        输出：
            tuple[list[str], list[str], list[str]]。返回反思文本列表、较差代码列表和较优代码列表。
        """
        messages_lst = []
        worse_code_lst = []
        better_code_lst = []
        
        # Generate reflection prompts for each parent pair
        # 为每一对父代生成反思提示词
        for i in range(0, len(population), 2):
            parent_1 = population[i]
            parent_2 = population[i + 1]
            
            # Generate short-term reflection
            # 生成短期反思
            messages, worse_code, better_code = self.population_inter_envoltion_reflection_prompt(
                parent_1, parent_2
            )
            messages_lst.append(messages)
            worse_code_lst.append(worse_code)
            better_code_lst.append(better_code)
        
        # Generate LLM responses asynchronously
        # 异步生成 LLM 响应
        response_lst = multi_chat_completion(
            messages_lst, 1, self.cfg.model, self.cfg.temperature
        )
        return response_lst, worse_code_lst, better_code_lst
    
    def individual_self_evolution_reflection(self, population: list[dict], population_inter_envoltion_reflection_tuple: list[str], selected_population: list[dict]) -> tuple[list[str], list[str], list[str]]:
        """Perform reflection for individual self-evolution.
        为个体自进化执行反思。

        作用：
            将种群间进化产生的新个体与原父代配对比较，批量生成个体自进化反思。
        输入：
            population: 种群间进化后得到并已评估的新种群。
            population_inter_envoltion_reflection_tuple: 种群间进化阶段生成的反思响应列表。
            selected_population: 原始选中的父代种群。
        输出：
            tuple[list[str], list[str], list[str]]。返回个体自进化反思文本列表、
            较优父代代码列表和新个体代码列表。
        """
        messages_lst = []
        older_code_lst = []
        now_code_list = []
        
        # Generate reflection prompts for each individual
        # 为每个个体生成反思提示词
        for i in range(0, len(selected_population), 2):
            new_individual = population[i // 2]
            parent_1 = selected_population[i]
            parent_2 = selected_population[i + 1]
            older_prompt = population_inter_envoltion_reflection_tuple[i // 2]
            
            # Generate self-evolution reflection
            # 生成个体自进化反思
            messages, better_code, new_code = self.individual_self_evolution_reflection_prompt(
                new_individual, parent_1, parent_2, older_prompt
            )
            messages_lst.append(messages)
            older_code_lst.append(better_code)
            now_code_list.append(new_code)
        
        # Generate LLM responses asynchronously
        # 异步生成 LLM 响应
        response_lst = multi_chat_completion(
            messages_lst, 1, self.cfg.model, self.cfg.temperature
        )
        return response_lst, older_code_lst, now_code_list

    def _generate_evolution_trend_summary(self) -> str:
        """Generate a summary of evolution trends based on performance history.
        根据性能历史生成进化趋势摘要。

        作用：
            根据历史改进幅度和最优目标值变化，判断进化处于改进、停滞、
            波动或收敛状态，并生成给长期反思使用的策略提示。
        输入：
            无显式输入；使用 self.improvement_history、self.best_obj_history、
            self.best_obj_overall 和 self.iteration。
        输出：
            str。返回描述进化趋势、性能模式和策略建议的文本；历史不足时返回空字符串。
        """
        if len(self.improvement_history) < 2:
            return ""
        
        # Calculate statistics
        # 计算统计信息
        total_improvement = sum(self.improvement_history)
        recent_improvements = self.improvement_history[-3:]  # Last 3 iterations / 最近 3 次迭代
        avg_improvement = np.mean([x for x in self.improvement_history if x > 0]) if any(x > 0 for x in self.improvement_history) else 0
        
        # Determine trend
        # 判断进化趋势
        if sum(recent_improvements) > 0:
            trend = "improving"
        elif all(x == 0 for x in recent_improvements):
            trend = "stagnant"
        else:
            trend = "fluctuating"
        
        # Calculate convergence
        # 计算收敛状态
        if len(self.best_obj_history) >= 3:
            recent_variance = np.var(self.best_obj_history[-3:])
            convergence_status = "converging" if recent_variance < 0.01 else "still exploring"
        else:
            convergence_status = "early stage"
        
        summary = (
            f"\n\n**Evolution Progress Summary:**\n"
            f"- Current iteration: {self.iteration}\n"
            f"- Total improvement achieved: {total_improvement:.4f}\n"
            f"- Current best objective: {self.best_obj_overall:.4f}\n"
            f"- Current best constraint status: {individual_performance_summary(self.elitist)}\n"
            f"- Average improvement per successful iteration: {avg_improvement:.4f}\n"
            f"- Evolution trend: {trend}\n"
            f"- Convergence status: {convergence_status}\n"
        )
        
        # Add strategic guidance based on trend
        # 根据趋势添加策略性引导
        if trend == "stagnant":
            summary += "\n**Guidance**: Consider more radical variations or underused deadline, DAG-criticality, and fuzzy-risk features.\n"
        elif trend == "improving":
            summary += "\n**Guidance**: The current direction is promising. Refine successful patterns while maintaining diversity.\n"
        elif convergence_status == "converging":
            summary += "\n**Guidance**: Solutions are converging. Focus on fine-tuning and robust adaptations.\n"
        
        return summary
    
    def long_term_reflection(self, short_term_reflections: list[str]) -> None:
        """Perform long-term reflection by aggregating short-term insights.
        通过聚合短期洞察执行长期反思。

        作用：
            聚合短期反思和进化趋势摘要，调用 LLM 生成长期反思，并保存短期/长期反思文件。
        输入：
            short_term_reflections: 当前迭代产生的短期反思文本列表。
        输出：
            None。该函数会更新 self.long_term_reflection_str，并写入反思结果文件。
        """
        # Generate evolution trend summary
        # 生成进化趋势摘要
        trend_summary = self._generate_evolution_trend_summary()
        
        # Create long-term reflection prompt with evolution context
        # 创建包含进化上下文的长期反思提示词
        system = self.system_reflector_prompt
        diagnostic_individuals = sorted(
            [
                individual
                for individual in getattr(self, "population", [])
                if individual.get("exec_success")
            ],
            key=individual_comparison_key,
        )[:3]
        diagnostic_feedback = [
            json.loads(parameter_feedback_summary(individual))
            for individual in diagnostic_individuals
        ]
        combined_reflection = (
            "\n".join(short_term_reflections)
            + "\n\nStructured parameter, current counterfactual, and historical critical-state replay evidence from current elites (JSON):\n"
            + json.dumps(
                diagnostic_feedback,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
            )
            + "\nParameter diagnostics expose parameter-space limitations; "
            "counterfactual diagnostics explain concrete task-choice failures; "
            "critical-state replay detects persistent historical failures and resolved-state regressions. "
            "Treat correlation, local counterfactuals, and feature replay as non-causal evidence. "
            "Make a strong structural change only when evidence persists across "
            "decisions, generations, seeds, or scenarios; re-run CMA-ES and full "
            "evaluation plus replay after every change and do not grow complexity without need."
        )
        user = self.user_reflector_lt_prompt.format(
            problem_desc=self.problem_desc,
            prior_reflection=self.long_term_reflection_str,
            new_reflection=combined_reflection,
        ) + trend_summary
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        
        # Log prompt for the first iteration only
        # 仅在首次迭代记录该提示词
        if self.print_long_term_reflection_prompt:
            logging.info("Long-term Reflection Prompt: \nSystem Prompt: \n" + system + "\nUser Prompt: \n" + user)
            self.print_long_term_reflection_prompt = False
        
        # Generate long-term reflection
        # 生成长期反思内容
        self.long_term_reflection_str = multi_chat_completion(
            [messages], 1, self.cfg.model, self.cfg.temperature
        )[0]
        
        # Save reflections to files
        # 将反思结果保存到文件
        short_term_file = self._runtime_artifact_path(
            "reflection",
            f"problem_iter{self.iteration}_short_term_reflections.txt",
        )
        with open(short_term_file, 'w', encoding="utf-8") as file:
            file.write("\n".join(short_term_reflections) + '\n')
        
        long_term_file = self._runtime_artifact_path(
            "reflection",
            f"problem_iter{self.iteration}_long_term_reflection.txt",
        )
        with open(long_term_file, 'w', encoding="utf-8") as file:
            file.write(self.long_term_reflection_str + '\n')


    def population_inter_envoltion(self, population_inter_envoltion_reflection_tuple: tuple[list[str], list[str], list[str]]) -> list[dict]:
        """Perform population inter-evolution (crossover) based on reflections.
        基于反思执行种群间进化（交叉）。

        作用：
            基于短期反思、较优代码和较差代码构造交叉 prompt，调用 LLM 生成新的候选个体。
        输入：
            population_inter_envoltion_reflection_tuple: 包含反思内容、较差代码列表、
            较优代码列表的元组。
        输出：
            list[dict]。返回由交叉生成的新个体列表，长度应等于 pop_size。
        """
        reflection_content_lst, worse_code_lst, better_code_lst = population_inter_envoltion_reflection_tuple
        messages_lst = []
        
        # Generate crossover prompts for each pair
        # 为每对个体生成交叉提示词
        for reflection, worse_code, better_code in zip(reflection_content_lst, worse_code_lst, better_code_lst):
            system = self.system_generator_prompt
            func_signature0 = self.func_signature.format(version=0)
            func_signature1 = self.func_signature.format(version=1)
            # Crossover prompt asks the LLM to combine useful patterns from better and worse code.
            # 交叉提示词要求 LLM 结合较优和较差代码中的有效模式生成新规则。
            user = self.crossover_prompt.format(
                user_generator=self.user_generator_prompt,
                func_signature0=func_signature0,
                func_signature1=func_signature1,
                worse_code=worse_code,
                better_code=better_code,
                reflection=reflection,
                func_name=self.func_name,
            )
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            messages_lst.append(messages)
            
            # Log prompt for the first iteration only
            # 仅在首次迭代记录该提示词
            if self.print_crossover_prompt:
                logging.info("Crossover Prompt: \nSystem Prompt: \n" + system + "\nUser Prompt: \n" + user)
                self.print_crossover_prompt = False
        
        # Generate new individuals asynchronously
        # 异步生成新个体
        response_lst = multi_chat_completion(
            messages_lst, 1, self.cfg.model, self.cfg.temperature
        )
        crossed_population = self._responses_to_validated_individuals(
            response_lst,
            messages_lst,
        )

        assert len(crossed_population) == self.cfg.pop_size, \
            f"Expected {self.cfg.pop_size} individuals, got {len(crossed_population)}"
        return crossed_population


    def individual_self_evolution(self, individual_self_evolution_reflection_tuple: tuple[list[str], list[str], list[str]]) -> list[dict]:
        """Perform individual self-evolution based on reflections.
        基于反思执行个体自进化。

        作用：
            基于个体自进化反思、较优父代代码和当前新代码，调用 LLM 进一步改进候选个体。
        输入：
            individual_self_evolution_reflection_tuple: 包含反思内容、较优父代代码列表、
            新个体代码列表的元组。
        输出：
            list[dict]。返回自进化后的个体列表，长度应等于 pop_size。
        """
        reflection_content_lst, better_code_lst, new_code_lst = individual_self_evolution_reflection_tuple
        messages_lst = []
        
        # Generate self-evolution prompts for each individual
        # 为每个个体生成自进化提示词
        for reflection, older_code, new_code in zip(reflection_content_lst, better_code_lst, new_code_lst):
            system = self.system_generator_prompt
            func_signature0 = self.func_signature.format(version=0)
            func_signature1 = self.func_signature.format(version=1)
            # Self-evolution prompt asks the LLM to refine a generated individual using reflection feedback.
            # 自进化提示词要求 LLM 基于反思反馈改进已生成个体。
            user = self.individual_self_evolution_prompt.format(
                user_generator=self.user_generator_prompt,
                func_signature0=func_signature0,
                func_signature1=func_signature1,
                older_code=older_code,
                new_code=new_code,
                reflection=reflection,
                func_name=self.func_name,
            )
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            messages_lst.append(messages)
            
            # Log prompt for the first iteration only
            # 仅在首次迭代记录该提示词
            if self.print_individual_self_evolution_prompt:
                logging.info("Individual Self Evolution Prompt: \nSystem Prompt: \n" + system + "\nUser Prompt: \n" + user)
                self.print_individual_self_evolution_prompt = False
        
        # Generate evolved individuals asynchronously
        # 异步生成自进化后的个体
        response_lst = multi_chat_completion(
            messages_lst, 1, self.cfg.model, self.cfg.temperature
        )
        evolved_population = self._responses_to_validated_individuals(
            response_lst,
            messages_lst,
        )

        assert len(evolved_population) == self.cfg.pop_size, \
            f"Expected {self.cfg.pop_size} individuals, got {len(evolved_population)}"
        return evolved_population


    def mutate(self) -> list[dict]:
        """Elitist-based mutation.
        基于精英个体的变异。
        
        Mutates the best individual to generate new individuals based on
        long-term reflection and external knowledge.
        基于长期反思和外部知识，对最优个体进行变异以生成新个体。

        作用：
            以当前精英个体为基础，结合长期反思和外部知识构造变异 prompt，
            调用 LLM 生成若干变异个体。
        输入：
            无显式输入；使用 self.elitist、self.long_term_reflection_str、
            self.external_knowledge、self.mutation_rate 和 self.cfg.pop_size。
        输出：
            list[dict]。返回变异生成的个体列表，数量为 int(pop_size * mutation_rate)。
        """
        system = self.system_generator_prompt
        func_signature1 = self.func_signature.format(version=1)
        # Mutation uses the elitist as an anchor while injecting long-term guidance for exploration.
        # 变异以精英个体为基础，同时注入长期反思指导以进行探索。
        elite_evidence = parameter_feedback_summary(self.elitist)
        user = self.mutation_prompt.format(
            user_generator=self.user_generator_prompt,
            reflection=(
                self.long_term_reflection_str
                + self.external_knowledge
                + "\nCurrent elite parameter/current-decision/historical-replay evidence (JSON):\n"
                + elite_evidence
                + "\nUse persistent HARD failures or resolved-state regressions only for bounded "
                "feature, interaction, normalization, gate, nonlinear, or piecewise structural "
                "changes. Do not hand-edit continuous weights; every changed structure must "
                "re-enter CMA-ES, full scheduling evaluation, and critical-state replay."
            ),
            func_signature1=func_signature1,
            elitist_code=rule_source_for_evolution(self.elitist),
            func_name=self.func_name,
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        
        # Log prompt for the first iteration only
        # 仅在首次迭代记录该提示词
        if self.print_mutate_prompt:
            logging.info("Mutation Prompt: \nSystem Prompt: \n" + system + "\nUser Prompt: \n" + user)
            self.print_mutate_prompt = False
        
        # Generate mutated individuals
        # 生成变异个体
        num_mutations = int(self.cfg.pop_size * self.mutation_rate)
        responses = multi_chat_completion(
            [messages], num_mutations, self.cfg.model, self.cfg.temperature
        )
        population = self._responses_to_validated_individuals(
            responses,
            [messages],
        )
        return population


    def _run_single_iteration(self) -> None:
        """Execute one iteration of the evolutionary process.
        执行一次进化流程迭代。

        作用：
            执行一次完整进化流程，包括父代选择、种群间反思与交叉、个体自进化、
            训练模式下的长期反思和精英变异，并更新评估次数。
        输入：
            无显式输入；使用 self.population、self.elitist、self.mode、
            self.case_num 和各类 prompt/配置属性。
        输出：
            None。该函数会更新 self.population、self.elitist、最优解记录、
            迭代编号和 self.function_evals。
        """
        # Check if all individuals are invalid
        # 检查是否所有个体都无效
        if all(not individual["exec_success"] for individual in self.population):
            if not self._recover_population_from_validated_anchors():
                errors = [
                    individual.get("candidate_validation_error")
                    or individual.get("traceback_msg")
                    or "unknown candidate failure"
                    for individual in self.population
                ]
                raise RuntimeError(
                    "All generated individuals are invalid and fewer than two "
                    "validated anchors are available. Candidate errors: "
                    + " | ".join(str(error) for error in errors)
                )
        
        # Selection: add elitist to population if not already present
        # 选择：如果精英个体不在当前种群中，则加入候选池
        if self.elitist is None or self.elitist in self.population:
            population_to_select = self.population
        else:
            population_to_select = [self.elitist] + self.population
        
        # Parent pairs are selected from the current candidate pool for crossover.
        # 从当前候选池中选择父代配对，用于后续交叉。
        selected_population = self.random_select(population_to_select)
        if selected_population is None:
            raise RuntimeError("Selection failed. Please check the population.")
        
        # Population inter-evolution: reflection + crossover
        # 种群间进化：反思 + 交叉
        population_inter_envoltion_reflection_tuple = self.population_inter_envoltion_reflection(
            selected_population
        )
        population_inter_envoltion_population = self.population_inter_envoltion(
            population_inter_envoltion_reflection_tuple
        )
        self.population = self.evaluate_population(
            population_inter_envoltion_population, self.case_num
        )
        self.update_iter()

        # Individual self-evolution further refines crossover offspring using reflection feedback.
        # 个体自进化会基于反思反馈进一步改进交叉后代。
        individual_self_evolution_reflection_tuple = self.individual_self_evolution_reflection(
            self.population, 
            population_inter_envoltion_reflection_tuple[0], 
            selected_population
        )
        individual_self_evolution_population = self.individual_self_evolution(
            individual_self_evolution_reflection_tuple
        )
        self.population = self.evaluate_population(
            individual_self_evolution_population, self.case_num
        )
        self.update_iter()
        # Individual self-evolution: reflection + improvement (train mode only)
        # 个体自进化：反思 + 改进（仅训练模式）
        if self.mode == "train":
            # Long-term reflection: aggregate short-term insights
            # 长期反思：聚合短期洞察
            self.long_term_reflection(population_inter_envoltion_reflection_tuple[0])
            
            # Mutation: mutate elitist individual
            # 变异：对精英个体进行变异
            mutated_population = self.mutate()
            evaluated_mutated_population = self.evaluate_population(
                mutated_population, self.case_num
            )
            self.population.extend(evaluated_mutated_population)
            
            # Update iteration
            # 更新迭代状态
            self.update_iter()
        
        self.function_evals += 1


    def evolve(self) -> tuple[str, str]:
        """Main evolutionary loop.
        主进化循环。

        作用：
            作为算法主入口。训练模式下循环执行进化直到达到最大函数评估次数；
            测试模式下对已加载规则执行一次迭代。
        输入：
            无显式输入；使用 self.mode 和 self.cfg.max_fe 控制运行流程。
        输出：
            tuple[str, str]。返回全局最优代码 self.best_code_overall
            和对应路径 self.best_code_path_overall。
        """
        try:
            if self.mode == "test":
                # Test mode: run single iteration
                # 测试模式：运行单次迭代
                logging.info("Test mode: Running single iteration...")
                self._run_single_iteration()
            else:
                # Train mode: normal multi-iteration loop
                # 训练模式：正常执行多轮迭代
                while self.function_evals < self.cfg.max_fe:
                    self._run_single_iteration()
                self._finalize_best_rule_admission()
        finally:
            self._shutdown_evaluation_executor()
            # 评估缓存平时只追加日志，退出前压实一次，让快照文件自身完整。
            # 放在 finally 里是因为中断的长跑同样需要可直接续跑的缓存。
            cache = getattr(self, "parameter_evaluation_cache", None)
            if cache is not None:
                cache.flush()

        return self.best_code_overall, self.best_code_path_overall
