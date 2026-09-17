"""在现有云边环境中评价一个 ready-task 优先级规则。

评价链路严格分成两层：

1. 候选函数只接收八组 ready-task 特征并返回优先级分数；分数越小越优先。
2. 环境选出任务后，再用固定、确定性的 deadline/energy 规则选择可行 VM。

这样 SeEvo 只进化“选哪个任务”，不会把 Host/VM 选择编码进 LLM 个体，也不会
    变成强化学习。fuzzy 模式的优化目标严格等于风险调整模糊总能耗，modal 模式
    严格等于旧总能耗；DDL 违反率和延期量作为约束字段返回，绝不与能耗加权求和。
    脚本最后只通过 ``RESULT_JSON=...`` 行向 SeEvo 返回结构化结果。
普通日志可以出现在它之前，SeEvo 会从 stdout 末尾向前寻找最后一条结果记录。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import uuid

import numpy as np


# eval.py 可能由项目根目录直接运行，也可能由 SeEvo 在 Hydra 改变工作目录后
# 作为子进程运行。因此先基于本文件定位 LLM 与项目根目录，再导入统一路径常量，
# 不依赖调用者的 current working directory。
LLM_ROOT = Path(__file__).resolve().parents[2]
_BOOTSTRAP_PROJECT_ROOT = LLM_ROOT.parents[2]
for import_root in (str(_BOOTSTRAP_PROJECT_ROOT), str(LLM_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from algorithms.llm_safe_hrl.paths import PROJECT_ROOT
from algorithms.llm_safe_hrl.LLM.protocol_config import (
    apply_seevo_scenario_config,
    parse_deadline_cache_overrides,
)
from base.hrl_env import HrlFcfsCacheEnv, NoFeasibleVMError
from base.heuristic_admission import (
    CEWS_EVALUATOR_PROTOCOL_VERSION,
    canonical_json_sha256,
    file_sha256,
)
from common.resource_opt import TriangularFuzzyNumber
from rule_optimization import (
    extract_rule_metadata,
    validate_frozen_rule_source,
)
from counterfactual_feedback import (
    CounterfactualConfig,
    CounterfactualRunSession,
    reject_test_seeds,
)
from surrogate.replay_gate import DecisionTraceCapture


DEFAULT_CONFIG_PATH = LLM_ROOT / "cfg" / "problem" / "cews_task_constructive.yaml"
LLM_EVOLUTION_FORBIDDEN_SEEDS = frozenset({101, 102, 103}).union(range(201, 231))


def _apply_eval_scenario_config(
    config: dict,
    scenario: str,
    deadline_cache_paths,
    *,
    require_files: bool = False,
) -> dict:
    """Apply scenario inputs while leaving cache fallback to this evaluator."""
    cache_paths = parse_deadline_cache_overrides(deadline_cache_paths)
    result = apply_seevo_scenario_config(
        config,
        scenario,
        cache_paths,
        require_files=(
            require_files
            and str(scenario).strip().upper() in cache_paths
        ),
    )
    if str(scenario).strip().upper() not in cache_paths:
        result["dataset"].pop("deadline_cache_path", None)
    return result


def load_problem_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    """直接读取问题 YAML，使评价子进程不依赖 Hydra 的运行时状态。

    SeEvo 主进程使用 Hydra 组合配置；子进程只需要问题本身的资源、数据集、
    纯能耗目标声明和 DDL 约束。使用独立 YAML 可避免并发评价共享 Hydra 状态。
    """
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to run the CEWS evaluator; install "
            "algorithms/llm_safe_hrl/LLM/requirements.txt."
        ) from exc
    with Path(config_path).resolve().open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Problem config must contain a YAML mapping: {config_path}")
    dataset = config.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("Problem config must contain a dataset mapping")
    scenario = dataset.get("scenario", "SS")
    cache_paths = config.get("deadline_cache_paths", {})
    if not cache_paths and dataset.get("deadline_cache_path"):
        cache_paths = {scenario: dataset["deadline_cache_path"]}
    return _apply_eval_scenario_config(
        config, scenario, cache_paths, require_files=False
    )


def load_priority_function(candidate_path: str | Path, function_name: str = "get_task_priority_v2"):
    """从候选文件的独立路径动态加载指定任务优先级函数。

    模块名同时包含规范路径摘要和随机 UUID：路径摘要便于辨识来源，UUID 防止
    Python 的 ``sys.modules`` 缓存让两个并发评价错误地复用同一模块对象。
    候选文件本身也由 SeEvo 按迭代号和个体号隔离，因此不存在共享 gpt.py 的
    覆盖竞态。
    """
    candidate_path = Path(candidate_path).resolve()
    if not candidate_path.is_file():
        raise FileNotFoundError(f"Candidate module not found: {candidate_path}")
    source = candidate_path.read_text(encoding="utf-8")
    # Legacy complete rules have no RULE_METADATA and keep their historical path.
    # New optimized artifacts are statically rechecked before Python imports them.
    if extract_rule_metadata(source):
        validate_frozen_rule_source(source, require_metadata=True)
    # 对绝对路径求摘要；不读取或改写候选代码内容。
    digest = hashlib.sha256(str(candidate_path).encode("utf-8")).hexdigest()[:12]
    module_name = f"cews_candidate_{digest}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, candidate_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create import specification for {candidate_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    priority_function = getattr(module, function_name, None)
    if not callable(priority_function):
        raise AttributeError(
            f"Candidate {candidate_path} does not define callable {function_name}."
        )
    return priority_function


def _scenario_names(scenario: str):
    """把 SS/SM/.../LL 场景码映射为现有数据目录使用的规模名称。"""
    size_name = {"S": "small", "M": "med", "L": "large"}
    value = str(scenario).strip().upper()
    if len(value) != 2 or any(code not in size_name for code in value):
        raise ValueError(f"Invalid scenario {scenario!r}; expected SS through LL.")
    return size_name[value[0]], size_name[value[1]]


def build_environment(config: dict, seed: int):
    """根据 YAML 与随机种子创建现有 ``HrlFcfsCacheEnv``。

    这里复用项目已有的 Workflow/Task/Host/VM、模糊处理能力、带宽、能耗模型、
    DAX 加载器和 FCFS deadline cache，不复制任何环境实体类。相同配置和 seed
    会构造相同的到达序列与资源实例，保证候选之间可以公平比较。
    """
    dataset = config["dataset"]
    resources = config["resources"]
    fuzzy = config.get("fuzzy", {})
    resource_seed_mode = str(
        fuzzy.get("resource_seed_mode", "episode_seed")
    )
    if resource_seed_mode != "episode_seed":
        raise ValueError(
            "cews_task_constructive currently supports only "
            "fuzzy.resource_seed_mode='episode_seed'."
        )
    # 资源只在环境构造时模糊化一次。同一个评价 seed 和 offset 对所有候选产生
    # 完全相同的 pc/bw 三角数，调度过程中绝不重新采样。
    fuzzy_resource_seed = int(seed) + int(
        fuzzy.get("resource_seed_offset", 0)
    )
    scenario_code = str(dataset.get("scenario", "SS")).strip().upper()
    task_size, resource_size = _scenario_names(scenario_code)
    # YAML 中只保存 DAX 文件名；实际路径相对项目 data/dax 解析。
    dax_paths = [PROJECT_ROOT / "data" / "dax" / name for name in dataset["dax_files"]]
    missing = [str(path) for path in dax_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing DAX input(s): {missing}")

    # 若用户未指定缓存路径，就按现有项目命名规则选择 FCFS deadline cache。
    deadline_cache = dataset.get("deadline_cache_path")
    if deadline_cache is None:
        deadline_cache = (
            PROJECT_ROOT
            / "data"
            / "deadlines"
            / "fcfs"
            / "diagnostic"
            / f"fcfs_fixed_{scenario_code}_exactmix_formal38.json"
        )
    else:
        deadline_cache = Path(deadline_cache)
        if not deadline_cache.is_absolute():
            deadline_cache = PROJECT_ROOT / deadline_cache

    # max_ready_tasks="auto" 允许 ready 集大小随 DAG 状态动态变化；这正是候选
    # 函数使用长度 N 数组而不能假设固定任务数量的原因。
    return HrlFcfsCacheEnv(
        dax_paths=[str(path) for path in dax_paths],
        horizon=float(dataset.get("horizon", 1e9)),
        arrival_lambda=float(dataset.get("arrival_lambda", 0.03)),
        random_seed=int(seed),
        max_ready_tasks="auto",
        normalize=True,
        workflows_per_episode=int(dataset.get("workflows_per_instance", config["problem_size"])),
        num_cloud_hosts=int(resources["num_cloud_hosts"]),
        num_edge_hosts=int(resources["num_edge_hosts"]),
        cloud_vms_per_host=tuple(resources["cloud_vms_per_host"]),
        edge_vms_per_host=tuple(resources["edge_vms_per_host"]),
        cloud_pc_tiers=tuple(resources["cloud_pc_tiers"]),
        edge_pc_tiers=tuple(resources["edge_pc_tiers"]),
        cloud_bw_tiers=tuple(resources["cloud_bw_tiers"]),
        edge_bw_tiers=tuple(resources["edge_bw_tiers"]),
        deadline_mode=str(dataset.get("deadline_mode", "cache_fcfs")),
        deadline_cache_path=str(deadline_cache),
        deadline_cache_strict=True,
        deadline_alpha_small=float(dataset.get("deadline_alpha_small", 2.0)),
        deadline_alpha_large=float(dataset.get("deadline_alpha_large", 3.0)),
        deadline_alpha_small_prob=float(dataset.get("deadline_alpha_small_prob", 0.8)),
        fuzzy_enabled=bool(fuzzy.get("enabled", False)),
        fuzzy_delta1=float(fuzzy.get("delta1", 0.75)),
        fuzzy_delta2=float(fuzzy.get("delta2", 1.2)),
        fuzzy_energy_uncertainty_weight=float(
            fuzzy.get("energy_uncertainty_weight", 1.0)
        ),
        fuzzy_deadline_eta=float(fuzzy.get("deadline_eta", 0.95)),
        fuzzy_resource_seed=fuzzy_resource_seed,
        fuzzy_use_deadline_constraint=bool(
            fuzzy.get("use_fuzzy_deadline_constraint", True)
        ),
        scenario_code=scenario_code,
        task_code=scenario_code[0],
        resource_code=scenario_code[1],
    )


def _schedule_metrics(environment) -> dict:
    """从已结束环境提取原始、未加权的调度指标。

    - workflow completion time = 完成绝对时刻 - 到达时刻；
    - makespan = 最后完成时刻 - 首个工作流到达时刻；
    - fuzzy 模式按三场景完成时刻和 eta 风险测度计算约束；
    - 同时保留 modal lateness/timeout 供历史结果诊断；
    - edge/cloud ratio 按实际任务分配历史统计，而不是按 VM 数量估计。
    """
    workflow_count = len(environment.workflows)
    finish_times = [
        float(environment.wf_finish_time[workflow_id])
        for workflow_id in range(workflow_count)
        if workflow_id in environment.wf_finish_time
    ]
    completion_times = [
        float(environment.wf_finish_time[workflow_id])
        - float(environment.workflows[workflow_id].arrival_time)
        for workflow_id in range(workflow_count)
        if workflow_id in environment.wf_finish_time
    ]
    # 对同一任务顺序和 VM 映射读取三条完整时间线。fuzzy 关闭时影子时间线不会
    # 参与评价，optimistic/pessimistic 退化为 modal，保持历史指标不变。
    fuzzy_enabled = bool(getattr(environment, "fuzzy_enabled", False))
    use_fuzzy_deadline = bool(
        fuzzy_enabled
        and getattr(environment, "fuzzy_use_deadline_constraint", True)
    )
    fuzzy_lateness = []
    fuzzy_violations = []
    modal_lateness = []
    modal_violations = []
    workflow_finish_triplets = []
    for workflow_id in range(workflow_count):
        task_ids = [
            task_id
            for task_id, (owner_workflow_id, _local_id)
            in enumerate(environment.task_meta)
            if int(owner_workflow_id) == int(workflow_id)
        ]
        workflow = environment.workflows[workflow_id]
        deadline = float(workflow.deadline)
        modal_finish = max(
            (
                float(environment.task_end_time[task_id])
                for task_id in task_ids
            ),
            default=float(workflow.arrival_time),
        )
        if fuzzy_enabled:
            optimistic_finish = max(
                (
                    float(
                        environment.shadow_task_end_time[
                            "optimistic"
                        ][task_id]
                    )
                    for task_id in task_ids
                ),
                default=float(workflow.arrival_time),
            )
            pessimistic_finish = max(
                (
                    float(
                        environment.shadow_task_end_time[
                            "pessimistic"
                        ][task_id]
                    )
                    for task_id in task_ids
                ),
                default=float(workflow.arrival_time),
            )
            tolerance = 1e-8
            if (
                optimistic_finish > modal_finish + tolerance
                or modal_finish > pessimistic_finish + tolerance
            ):
                raise ValueError(
                    "Workflow fuzzy finish order violated for "
                    f"workflow_id={workflow_id}: "
                    f"optimistic={optimistic_finish}, "
                    f"modal={modal_finish}, "
                    f"pessimistic={pessimistic_finish}"
                )
            if optimistic_finish > modal_finish:
                optimistic_finish = modal_finish
            if pessimistic_finish < modal_finish:
                pessimistic_finish = modal_finish
            # 环境方法负责 eta 合法性；这里用 TFN 表达工作流三点完成时刻。
            finish_tfn = TriangularFuzzyNumber(
                optimistic_finish,
                modal_finish,
                pessimistic_finish,
            )
            finish_risk = environment.fuzzy_deadline_measure(finish_tfn)
        else:
            optimistic_finish = modal_finish
            pessimistic_finish = modal_finish
            finish_risk = modal_finish

        fuzzy_late = max(0.0, float(finish_risk - deadline))
        modal_late = max(0.0, float(modal_finish - deadline))
        fuzzy_lateness.append(fuzzy_late)
        fuzzy_violations.append(bool(finish_risk > deadline))
        modal_lateness.append(modal_late)
        modal_violations.append(bool(modal_finish > deadline))
        workflow_finish_triplets.append({
            "workflow_id": int(workflow_id),
            "workflow_finish_lower": float(optimistic_finish),
            "workflow_finish_modal": float(modal_finish),
            "workflow_finish_upper": float(pessimistic_finish),
            "fuzzy_finish_risk": float(finish_risk),
            "fuzzy_lateness": float(fuzzy_late),
        })

    active_lateness = fuzzy_lateness if use_fuzzy_deadline else modal_lateness
    active_violations = (
        fuzzy_violations if use_fuzzy_deadline else modal_violations
    )
    # assignment_history 由真实 assign_task 路径写入，可准确区分 cloud/edge。
    assignments = list(environment.assignment_history)
    edge_count = sum(item["server_type"] == "edge" for item in assignments)
    cloud_count = sum(item["server_type"] == "cloud" for item in assignments)
    assignment_count = len(assignments)

    first_arrival = min(
        (float(workflow.arrival_time) for workflow in environment.workflows),
        default=0.0,
    )
    makespan = max(finish_times, default=first_arrival) - first_arrival
    result = {
        "total_energy": float(environment.total_energy),
        "makespan": float(makespan),
        "average_workflow_completion_time": float(np.mean(completion_times)) if completion_times else 0.0,
        "total_lateness": float(sum(active_lateness)),
        "average_lateness": float(np.mean(active_lateness)) if active_lateness else 0.0,
        "max_lateness": float(max(active_lateness, default=0.0)),
        "max_fuzzy_lateness": float(
            max(fuzzy_lateness, default=0.0)
        ),
        "max_modal_lateness": float(
            max(modal_lateness, default=0.0)
        ),
        "deadline_violation_rate": float(np.mean(active_violations)) if active_violations else 0.0,
        "modal_total_lateness": float(sum(modal_lateness)),
        "modal_average_lateness": float(np.mean(modal_lateness)) if modal_lateness else 0.0,
        "modal_deadline_violation_rate": float(np.mean(modal_violations)) if modal_violations else 0.0,
        "edge_task_ratio": float(edge_count / assignment_count) if assignment_count else 0.0,
        "cloud_task_ratio": float(cloud_count / assignment_count) if assignment_count else 0.0,
        "completed_workflows": int(environment.completed_workflows),
        "assigned_tasks": int(assignment_count),
        "workflow_fuzzy_finishes": workflow_finish_triplets,
    }
    result.update(environment.get_fuzzy_energy_summary())
    return result


def _add_objective_and_constraints(metrics: dict, config: dict) -> dict:
    """写入纯能耗目标和独立的 DDL 约束状态。

    modal 模式下 ``objective`` 严格等于 ``total_energy``；fuzzy 模式下严格等于
    ``fuzzy_total_energy_score``。两者都不包含任何延期或违反率惩罚项。
    ``constraint_feasible`` 表示违反率是否不超过 YAML 阈值。对不可行个体，
    ``constraint_violation`` 与 ``constraint_secondary_violation`` 分别提供违反率和
    总延期，供 SeEvo 做可行性优先的字典序比较，而不是伪造加权目标。

    强化学习环境原有的 reward 加和不经过此函数，因此完全不受影响。
    """
    objective_config = config.get("objective", {})
    metric_name = str(objective_config.get("metric", "total_energy"))
    if metric_name != "total_energy":
        raise ValueError(
            "cews_task_constructive only supports pure-energy objective metric "
            f"'total_energy'; got {metric_name!r}."
        )
    constraints = config.get("constraints", {})
    fuzzy_enabled = bool(config.get("fuzzy", {}).get("enabled", False))
    max_violation_rate = float(
        constraints.get("deadline_violation_rate_max", 0.0)
    )
    violation_rate = float(metrics["deadline_violation_rate"])
    result = dict(metrics)
    if fuzzy_enabled:
        result["objective"] = float(metrics["fuzzy_total_energy_score"])
        result["energy"] = float(metrics["fuzzy_total_energy_score"])
        result["modal_energy"] = float(metrics["total_energy"])
    else:
        result["objective"] = float(metrics["total_energy"])
        result["energy"] = float(metrics["total_energy"])
    result["constraint_feasible"] = bool(
        violation_rate <= max_violation_rate
    )
    result["constraint_violation"] = max(
        0.0, violation_rate - max_violation_rate
    )
    result["constraint_secondary_violation"] = float(metrics["total_lateness"])
    return result


def run_instance(
    priority_function,
    config: dict,
    seed: int,
    environment_factory=None,
    counterfactual_session=None,
    decision_trace_sink=None,
) -> dict:
    """运行一个 seed 对应的完整离散事件调度实例。

    循环不训练策略：ready task 与空闲 VM 同时存在时，调用一次 LLM 规则选任务，
    再由环境按 Host -> VM 的固定规则分配；ready 集非空但所有 VM 都在忙时，等待
    最近一次资源事件而不是把任务排到忙 VM 之后；ready 集为空时推进到下一个决策点。

    只从空闲 VM 中选择，与 deadline 缓存生成器、各比较算法以及 Safe-HRL 运行时
    的口径一致；允许 busy-VM 排队会让任务堆积到少数最省电的机器上，产生本评价器
    独有的调度结果。``environment_factory`` 仅用于测试注入轻量环境。
    """
    environment = (
        environment_factory(config, seed)
        if environment_factory is not None
        else build_environment(config, seed)
    )
    environment.reset()
    safety = config.get("safety", {})
    max_decisions = int(safety.get("max_decisions", 1_000_000))
    max_stalled_steps = int(safety.get("max_stalled_steps", 10))
    decisions = 0
    stalled_steps = 0

    def progress_snapshot():
        """推进事件前后的最小状态快照，用于检测环境停滞。"""
        return (
            float(environment.current_time),
            int(environment.completed_workflows),
            len(environment.event_heap),
            int(environment.next_arrival_idx),
        )

    # done_flag 只由环境在所有工作流完成或既定终止状态到达时设置。
    while not bool(environment.done_flag):
        ready_tasks = environment.get_ready_tasks()
        # 空闲集必须在调用规则之前求出：无法分配的步上不应消耗一次规则调用，
        # 也不应向反事实轨迹注入没有对应 assign_task 的幽灵决策。
        # get_feasible_vms 只校验任务、不依赖任务，因而任一 ready task 都可作
        # 探针，得到的空闲集对本步所有 ready task 相同。
        idle_vm_ids = (
            environment.idle_feasible_vm_ids(ready_tasks[0])
            if ready_tasks
            else []
        )
        if ready_tasks and idle_vm_ids:
            # 第一阶段：候选规则只对 ready task 排序，不接触 Host 和 VM。
            if counterfactual_session is None and decision_trace_sink is None:
                selected_task = environment.select_task_with_priority_rule(
                    ready_tasks, priority_function
                )
            else:
                selected_task, selection_details = (
                    environment.select_task_with_priority_rule(
                        ready_tasks,
                        priority_function,
                        return_details=True,
                    )
                )
                if counterfactual_session is not None:
                    counterfactual_session.observe_decision(
                        environment,
                        ready_tasks,
                        selection_details,
                        decisions,
                    )
                if decision_trace_sink is not None:
                    decision_trace_sink.record(
                        [environment._task_id(task) for task in ready_tasks],
                        selection_details,
                    )
            # 第二阶段：环境只在“可行且空闲”的 VM 中执行固定、可重复的选择
            # 规则；先在含空闲 VM 的 Host 之间选择，再在该 Host 内选 VM。
            _selected_host, selected_vm, _selection_details = (
                environment.select_host_then_vm_deterministic(
                    selected_task,
                    candidate_vm_ids=idle_vm_ids,
                )
            )
            # 真正的状态修改只发生在 assign_task；此前全部估计函数均为只读。
            environment.assign_task(selected_task, selected_vm)
            decisions += 1
            stalled_steps = 0
        elif ready_tasks:
            # 有 ready task 但没有任何空闲 VM：必须等待最近一次资源事件，而不是
            # 把任务排到某台正在忙的 VM 之后。这里不能用 advance_to_next_event，
            # 它只推进到“下一个决策点”，而 ready 集已非空时它可能立即返回、不
            # 推进时间，从而误触发下面的停滞保护。
            before = progress_snapshot()
            environment.advance_to_next_resource_event()
            stalled_steps = (
                stalled_steps + 1 if progress_snapshot() == before else 0
            )
        else:
            # 没有 ready task 不代表调度结束：可能有正在运行的任务，或下一个
            # 工作流尚未到达。
            before = progress_snapshot()
            environment.advance_to_next_event()
            stalled_steps = (
                stalled_steps + 1 if progress_snapshot() == before else 0
            )

        # 两个安全条件只防止错误候选/异常环境无限占用评价进程；它们不改变
        # 正常实例的目标值和调度选择。
        if decisions > max_decisions:
            raise RuntimeError(f"Safety termination: exceeded {max_decisions} assignments.")
        if stalled_steps >= max_stalled_steps:
            raise RuntimeError(
                f"Safety termination: environment made no progress for {stalled_steps} steps."
            )

    # 即使环境设置了 done_flag，也再次检查完成数量，防止部分完成被误当作有效解。
    expected = int(config["dataset"].get("workflows_per_instance", config["problem_size"]))
    if int(environment.completed_workflows) != expected:
        raise RuntimeError(
            "Evaluation terminated before all workflows completed: "
            f"{environment.completed_workflows}/{expected}."
        )
    if decision_trace_sink is not None:
        decision_trace_sink.finalize(
            expected_workflows=expected,
            completed_workflows=int(environment.completed_workflows),
        )
    return _add_objective_and_constraints(_schedule_metrics(environment), config)


def evaluate_candidate(
    candidate_path: str | Path,
    config: dict,
    seeds,
    function_name: str = "get_task_priority_v2",
    counterfactual_options: dict | None = None,
    decision_trace_dir: str | Path | None = None,
    decision_trace_max_mb: int = 128,
) -> dict:
    """在全部指定 seed 上评价同一隔离候选，并对数值指标取算术均值。

    ``energy`` 和 ``violation_rate`` 是 RESULT_JSON 协议要求的便捷别名；完整的
    ``total_energy``、``deadline_violation_rate`` 等指标仍全部保留在结果中。
    只有全部 seed 都满足 DDL 时，聚合结果才标记为 constraint_feasible。
    """
    resolved_candidate = Path(candidate_path).resolve()
    priority_function = load_priority_function(
        resolved_candidate,
        function_name,
    )
    seed_values = [int(seed) for seed in seeds]
    final_test_seeds = [
        int(seed) for seed in config.get("dataset", {}).get("test_seeds", [])
    ]
    if counterfactual_options is not None:
        reject_test_seeds(seed_values, final_test_seeds)
    per_seed = []
    counterfactual_manifests = []
    for seed in seed_values:
        session = None
        trace_sink = None
        if decision_trace_dir is not None:
            scenario_id = str(config.get("dataset", {}).get("scenario", "unknown"))
            trace_sink = DecisionTraceCapture(
                Path(decision_trace_dir) / f"{scenario_id}_seed{int(seed)}.npz",
                baseline_frozen_rule_hash=file_sha256(resolved_candidate),
                evaluation_config_sha256=canonical_json_sha256(config),
                scenario_id=scenario_id,
                seed=int(seed),
                max_mb=int(decision_trace_max_mb),
            )
        if counterfactual_options is not None:
            cf_config = counterfactual_options["config"]
            if not isinstance(cf_config, CounterfactualConfig):
                cf_config = CounterfactualConfig.from_mapping(cf_config)
            metadata = dict(counterfactual_options["metadata"])
            metadata["seed"] = int(seed)
            configured_scenario = str(
                config.get("dataset", {}).get("scenario", "unknown")
            )
            supplied_scenario = metadata.get("scenario_id")
            if (
                supplied_scenario is not None
                and str(supplied_scenario) != configured_scenario
            ):
                raise ValueError(
                    "counterfactual metadata scenario does not match evaluation config"
                )
            metadata["scenario_id"] = configured_scenario
            metadata["run_id"] = (
                str(metadata.get("run_id", "counterfactual"))
                + f":{metadata['scenario_id']}:{int(seed)}"
            )
            resource_config_hash = canonical_json_sha256(
                {
                    "resources": config.get("resources", {}),
                    "fuzzy": config.get("fuzzy", {}),
                    "dataset_scenario": config.get("dataset", {}).get("scenario"),
                }
            )
            planning_session = CounterfactualRunSession(
                cf_config,
                metadata,
                counterfactual_options["output_dir"],
                resource_config_hash,
                final_test_seeds=final_test_seeds,
                planning_only=True,
            )
            # The first deterministic pass records only compact traces.  The
            # second pass analyzes the most critical decisions under the exact
            # same frozen rule, seed, scenario, and resource configuration.
            run_instance(
                priority_function,
                config,
                int(seed),
                counterfactual_session=planning_session,
            )
            session = CounterfactualRunSession(
                cf_config,
                metadata,
                counterfactual_options["output_dir"],
                resource_config_hash,
                final_test_seeds=final_test_seeds,
                analysis_decision_indices=(
                    planning_session.selected_critical_decision_indices()
                ),
            )
        metrics = run_instance(
            priority_function,
            config,
            int(seed),
            counterfactual_session=session,
            decision_trace_sink=trace_sink,
        )
        per_seed.append(metrics)
        if session is not None:
            counterfactual_manifests.append(session.finalize())
    numeric_keys = [
        "objective",
        "total_energy",
        "fuzzy_total_energy_lower",
        "fuzzy_total_energy_modal",
        "fuzzy_total_energy_upper",
        "fuzzy_total_energy_mean",
        "fuzzy_total_energy_std",
        "fuzzy_total_energy_score",
        "energy_optimistic",
        "energy_modal",
        "energy_pessimistic",
        "makespan",
        "average_workflow_completion_time",
        "total_lateness",
        "average_lateness",
        "max_lateness",
        "max_modal_lateness",
        "deadline_violation_rate",
        "modal_deadline_violation_rate",
        "modal_total_lateness",
        "modal_average_lateness",
        "edge_task_ratio",
        "cloud_task_ratio",
        "constraint_violation",
        "constraint_secondary_violation",
        "completed_workflows",
        "assigned_tasks",
    ]
    # 每个 seed 的工作流数相同，因此按 seed 等权平均，避免样本数隐式加权。
    result = {
        key: float(np.mean([item[key] for item in per_seed]))
        for key in numeric_keys
    }

    objectives = np.asarray(
        [float(item["objective"]) for item in per_seed],
        dtype=float,
    )

    within_seed_fuzzy_stds = np.asarray(
        [
            float(item.get("fuzzy_total_energy_std", 0.0))
            for item in per_seed
        ],
        dtype=float,
    )

    seed_feasible_flags = np.asarray(
        [
            bool(item["constraint_feasible"])
            for item in per_seed
        ],
        dtype=float,
    )

    # 以下均为诊断字段，不进入 objective。
    result["objective_std_across_seeds"] = float(
        np.std(objectives)
    )

    result["objective_max_across_seeds"] = float(
        np.max(objectives)
    )

    objective_mean = float(np.mean(objectives))
    objective_std = float(np.std(objectives))
    result["objective_cv_across_seeds"] = float(
        objective_std / max(abs(objective_mean), 1e-12)
    )

    result["fuzzy_energy_std_mean_across_seeds"] = float(
        np.mean(within_seed_fuzzy_stds)
    )

    result["feasible_seed_rate"] = float(
        np.mean(seed_feasible_flags)
    )

    result["max_deadline_violation_rate_across_seeds"] = float(
        np.max(
            [
                float(item["deadline_violation_rate"])
                for item in per_seed
            ]
        )
    )
    result["max_fuzzy_lateness"] = float(
        np.max(
            [
                float(item["max_fuzzy_lateness"])
                for item in per_seed
            ]
        )
    )

    fuzzy_enabled = bool(config.get("fuzzy", {}).get("enabled", False))
    if fuzzy_enabled:
        result["energy"] = result["fuzzy_total_energy_score"]
        result["modal_energy"] = result["total_energy"]
    else:
        result["energy"] = result["total_energy"]
    result["violation_rate"] = result["deadline_violation_rate"]
    result["constraint_feasible"] = all(
        bool(item["constraint_feasible"]) for item in per_seed
    )
    result["seeds"] = seed_values
    result["evaluation_seed_count"] = len(per_seed)
    result["completed_seed_count"] = len(per_seed)
    result["all_evaluation_seeds_completed"] = True
    result["interface_valid"] = True
    result["function_name"] = str(function_name)
    result["evaluator_protocol_version"] = (
        CEWS_EVALUATOR_PROTOCOL_VERSION
    )
    result["candidate_source_file"] = resolved_candidate.name
    result["candidate_sha256"] = file_sha256(
        resolved_candidate
    )
    result["evaluation_config_sha256"] = (
        canonical_json_sha256(config)
    )
    scenario_id = str(config.get("dataset", {}).get("scenario", "unknown"))
    result["scenario_id"] = scenario_id
    rule_metadata = extract_rule_metadata(
        resolved_candidate.read_text(encoding="utf-8")
    )
    if rule_metadata:
        result.update(rule_metadata)
        result["frozen_rule_hash"] = result["candidate_sha256"]
    result["per_seed_metrics"] = [
        {
            "seed": int(seed),
            "scenario_id": scenario_id,
            "completed_workflows": int(
                item["completed_workflows"]
            ),
            "constraint_feasible": bool(
                item["constraint_feasible"]
            ),
            "deadline_violation_rate": float(
                item["deadline_violation_rate"]
            ),
            "total_lateness": float(
                item["total_lateness"]
            ),
            "max_fuzzy_lateness": float(
                item["max_fuzzy_lateness"]
            ),
            "fuzzy_total_energy_mean": float(
                item["fuzzy_total_energy_mean"]
            ),
            "fuzzy_total_energy_std": float(
                item["fuzzy_total_energy_std"]
            ),
            "fuzzy_total_energy_score": float(
                item["fuzzy_total_energy_score"]
            ),
            "objective": float(item["objective"]),
        }
        for seed, item in zip(seeds, per_seed)
    ]
    if counterfactual_manifests:
        result["counterfactual_manifests"] = counterfactual_manifests
        result["counterfactual_feedback_executed"] = True

    diagnostic_keys = [
    "objective_std_across_seeds",
    "objective_max_across_seeds",
    "objective_cv_across_seeds",
    "fuzzy_energy_std_mean_across_seeds",
    "feasible_seed_rate",
    "max_deadline_violation_rate_across_seeds",
    "max_fuzzy_lateness",
    ]
    for key in [*numeric_keys, *diagnostic_keys]:
        if not math.isfinite(float(result[key])):
            raise ValueError(
                f"Evaluator produced a non-finite numeric field: {key}."
            )
    return result


def _resolve_seeds(config: dict, dataset_mode: str, cases) -> list[int]:
    """解析评价种子：命令行 --cases 优先，否则读取 YAML 的 train/test 划分。"""
    if cases:
        seeds = [int(case) for case in cases]
    else:
        key = "test_seeds" if dataset_mode == "test" else "train_seeds"
        seeds = [int(seed) for seed in config["dataset"].get(key, [])]
    if not seeds:
        raise ValueError(f"No seeds configured for dataset mode {dataset_mode!r}.")
    if dataset_mode != "test":
        forbidden = LLM_EVOLUTION_FORBIDDEN_SEEDS.intersection(seeds)
        if forbidden:
            raise ValueError(
                "LLM rule generation/CMA-ES cannot use reserved comparison "
                f"validation or final-test seeds: {sorted(forbidden)}"
            )
    return seeds


def parse_args(argv=None):
    """定义独立评价进程的命令行接口。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, help="Path to one isolated candidate .py module.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Problem YAML path.")
    parser.add_argument("--dataset-mode", choices=("train", "test"), default="train")
    parser.add_argument("--cases", nargs="*", type=int, default=None, help="Optional seed override.")
    parser.add_argument(
        "--scenario",
        default=None,
        help="Optional offline parameter-search scenario override (SS through LL).",
    )
    parser.add_argument("--function-name", default="get_task_priority_v2")
    parser.add_argument(
        "--counterfactual-config-json",
        default=None,
        help="Offline frozen-rule counterfactual configuration JSON.",
    )
    parser.add_argument(
        "--counterfactual-metadata-json",
        default=None,
        help="Frozen-rule hashes and run identity for trace artifacts.",
    )
    parser.add_argument(
        "--counterfactual-output-dir",
        default=None,
        help="Artifact root for offline train/validation analysis.",
    )
    parser.add_argument(
        "--capture-decision-trace",
        default=None,
        help="Directory for complete diagnostic baseline traces.",
    )
    parser.add_argument(
        "--decision-trace-max-mb",
        type=int,
        default=128,
        help="Hard size budget for one decision trace.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """加载配置和候选，执行评价，并输出唯一机器可读结果记录。"""
    args = parse_args(argv)
    config = load_problem_config(args.config)
    if args.scenario is not None:
        scenario = str(args.scenario).strip().upper()
        _scenario_names(scenario)
        config = _apply_eval_scenario_config(
            config,
            scenario,
            config.get("deadline_cache_paths", {}),
            require_files=True,
        )
    seeds = _resolve_seeds(config, args.dataset_mode, args.cases)
    counterfactual_options = None
    counterfactual_values = (
        args.counterfactual_config_json,
        args.counterfactual_metadata_json,
        args.counterfactual_output_dir,
    )
    if any(value is not None for value in counterfactual_values):
        if not all(value is not None for value in counterfactual_values):
            raise ValueError(
                "counterfactual config, metadata, and output directory must be supplied together"
            )
        if args.dataset_mode != "train":
            raise ValueError("counterfactual feedback is forbidden in test mode")
        reject_test_seeds(
            seeds,
            config.get("dataset", {}).get("test_seeds", []),
        )
        cf_config = CounterfactualConfig.from_mapping(
            json.loads(args.counterfactual_config_json)
        )
        if not cf_config.enabled or not cf_config.run_after_parameter_optimization:
            raise ValueError(
                "counterfactual CLI requested while offline feedback is disabled"
            )
        counterfactual_options = {
            "config": cf_config,
            "metadata": json.loads(args.counterfactual_metadata_json),
            "output_dir": args.counterfactual_output_dir,
        }
    print(
        f"[cews-eval] candidate={Path(args.candidate).resolve()} seeds={seeds}",
        flush=True,
    )
    result = evaluate_candidate(
        args.candidate,
        config,
        seeds,
        args.function_name,
        counterfactual_options=counterfactual_options,
        decision_trace_dir=args.capture_decision_trace,
        decision_trace_max_mb=args.decision_trace_max_mb,
    )
    # allow_nan=False 保证 JSON 严格合法；若存在 NaN/Infinity 会在此明确失败，
    # 而不是让 SeEvo 接受一个不可比较的目标值。
    print("RESULT_JSON=" + json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except NoFeasibleVMError:
        # 保留原异常类型与 traceback，让 SeEvo 将该候选明确标记为无效。
        raise
