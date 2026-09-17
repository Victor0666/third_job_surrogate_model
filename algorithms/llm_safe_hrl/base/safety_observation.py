"""安全状态扩展的稳定 schema 与归一化约定。"""

from __future__ import annotations

from copy import deepcopy


SAFE_OBSERVATION_SCHEMA_VERSION = "safe_observation_v1"


def _feature(name, low, high, normalization, description):
    return {
        "name": str(name),
        "low": float(low),
        "high": float(high),
        "normalization": str(normalization),
        "description": str(description),
    }


MANAGER_SAFETY_FEATURE_SCHEMA = (
    _feature(
        "minimum_fuzzy_safety_margin_norm",
        -1.0,
        1.0,
        "clip(min workflow margin / workflow DDL budget, -1, 1)",
        "全局最小动态模糊安全裕量。",
    ),
    _feature(
        "mean_fuzzy_safety_margin_norm",
        -1.0,
        1.0,
        "mean of per-workflow normalized fuzzy safety margins",
        "未完成工作流的平均动态模糊安全裕量。",
    ),
    _feature(
        "risk_workflow_ratio",
        0.0,
        1.0,
        "risk workflow count / unfinished workflow count",
        "位于安全边界或预测越界的工作流比例。",
    ),
    _feature(
        "predicted_ddl_violation_rate",
        0.0,
        1.0,
        "predicted violation count / unfinished workflow count",
        "严格负安全裕量工作流的预测违反率。",
    ),
    _feature(
        "high_uncertainty_task_ratio",
        0.0,
        1.0,
        "high-uncertainty ready task count / ready task count",
        "最佳可行 VM 相对模糊时长跨度超过配置阈值的 ready-task 比例。",
    ),
    _feature(
        "cloud_congestion",
        0.0,
        1.0,
        "busy cloud VM count / cloud VM count",
        "云端 VM 当前拥塞程度。",
    ),
    _feature(
        "edge_congestion",
        0.0,
        1.0,
        "busy edge VM count / edge VM count",
        "边缘端 VM 当前拥塞程度。",
    ),
    _feature(
        "safe_host_ratio",
        0.0,
        1.0,
        "mean safe-host fraction over current ready tasks",
        "当前 ready tasks 的硬合法 Host 中含安全 VM 的平均比例。",
    ),
    _feature(
        "mean_safe_vm_ratio",
        0.0,
        1.0,
        "mean safe/selectable VM ratio over legal task-host pairs",
        "当前 ready tasks 在合法 Host 内的安全 VM 平均比例。",
    ),
    _feature(
        "recent_shield_intervention_rate",
        0.0,
        1.0,
        "shield-intervened records / configured recent record window",
        "最近 Host/VM 决策记录中的 shield 干预率。",
    ),
    _feature(
        "recent_fallback_trigger_rate",
        0.0,
        1.0,
        "fallback-triggered records / configured recent record window",
        "最近 Host/VM 决策记录中的回退控制器触发率。",
    ),
)


HOST_SAFETY_FEATURE_SCHEMA = (
    _feature(
        "safe_vm_count_norm",
        0.0,
        1.0,
        "safe selectable VM count / maximum VM slots per Host",
        "该 Host 下硬合法且预测安全的 VM 数量归一化值。",
    ),
    _feature(
        "safe_vm_ratio",
        0.0,
        1.0,
        "safe selectable VM count / selectable VM count",
        "该 Host 当前可选 VM 中的安全比例。",
    ),
    _feature(
        "host_queue_risk",
        0.0,
        1.0,
        "clip(mean positive VM availability delay / horizon, 0, 1)",
        "该 Host 的平均 VM 排队风险。",
    ),
    _feature(
        "minimum_fuzzy_safety_margin_after_host_norm",
        -1.0,
        1.0,
        "clip(min selectable-VM task margin / workflow DDL budget, -1, 1)",
        "选择该 Host 后内部当前可选 VM 的最小模糊安全裕量。",
    ),
    _feature(
        "cross_platform_communication_risk",
        0.0,
        1.0,
        "cross cloud-edge parent bits / total parent input bits",
        "当前任务父数据跨 cloud/edge 平台传输的比例。",
    ),
    _feature(
        "future_critical_resource_occupation_risk",
        0.0,
        1.0,
        "projected Host busy ratio * remaining critical-path pressure",
        "当前分配占用 Host 后对剩余关键路径资源的压力。",
    ),
)


VM_SAFETY_FEATURE_SCHEMA = (
    _feature(
        "optimistic_finish_time_norm",
        0.0,
        1.0,
        "clip((optimistic finish - now) / workflow DDL budget, 0, 1)",
        "当前任务在该 VM 上的乐观完成时间距离。",
    ),
    _feature(
        "modal_finish_time_norm",
        0.0,
        1.0,
        "clip((modal finish - now) / workflow DDL budget, 0, 1)",
        "当前任务在该 VM 上的模态完成时间距离。",
    ),
    _feature(
        "pessimistic_finish_time_norm",
        0.0,
        1.0,
        "clip((pessimistic finish - now) / workflow DDL budget, 0, 1)",
        "当前任务在该 VM 上的悲观完成时间距离。",
    ),
    _feature(
        "risk_finish_time_norm",
        0.0,
        1.0,
        "clip((risk finish - now) / workflow DDL budget, 0, 1)",
        "eta 风险完成时间距离。",
    ),
    _feature(
        "task_safe_deadline_norm",
        -1.0,
        1.0,
        "clip((task safe deadline - now) / workflow DDL budget, -1, 1)",
        "动态任务安全完成边界相对当前时刻的剩余预算。",
    ),
    _feature(
        "fuzzy_safety_margin_horizon_norm",
        -1.0,
        1.0,
        "clip(fuzzy safety margin / horizon, -1, 1)",
        "按仿真 horizon 归一化的任务级模糊安全裕量。",
    ),
    _feature(
        "normalized_fuzzy_safety_margin",
        -1.0,
        1.0,
        "clip(fuzzy safety margin / workflow DDL budget, -1, 1)",
        "按工作流确定性 DDL 总预算归一化的安全裕量。",
    ),
    _feature(
        "fuzzy_marginal_energy_norm",
        0.0,
        1.0,
        "energy / (energy + task MI * energy reference)",
        "风险调整模糊边际能耗的有界单调归一化值。",
    ),
    _feature(
        "queue_time_norm",
        0.0,
        1.0,
        "clip(modal VM queue time / horizon, 0, 1)",
        "该 VM 当前模态排队时间。",
    ),
    _feature(
        "computation_uncertainty",
        0.0,
        1.0,
        "clip((pc upper - pc lower) / pc modal, 0, 1)",
        "VM 模糊处理能力相对跨度。",
    ),
    _feature(
        "bandwidth_uncertainty",
        0.0,
        1.0,
        "clip((bw upper - bw lower) / bw modal, 0, 1)",
        "VM 模糊带宽相对跨度。",
    ),
    _feature(
        "remaining_critical_path_risk_time_norm",
        0.0,
        1.0,
        "clip(remaining critical-path risk time / workflow DDL budget, 0, 1)",
        "当前任务完成后剩余关键路径的 eta 风险时间。",
    ),
)


def get_safety_feature_schema(layer: str) -> list[dict]:
    """返回指定层的安全扩展字段副本，调用方不能修改模块常量。"""
    key = str(layer).strip().lower()
    schemas = {
        "manager": MANAGER_SAFETY_FEATURE_SCHEMA,
        "host": HOST_SAFETY_FEATURE_SCHEMA,
        "vm": VM_SAFETY_FEATURE_SCHEMA,
    }
    if key not in schemas:
        raise ValueError("layer must be 'manager', 'host', or 'vm'")
    return deepcopy(list(schemas[key]))


__all__ = [
    "SAFE_OBSERVATION_SCHEMA_VERSION",
    "MANAGER_SAFETY_FEATURE_SCHEMA",
    "HOST_SAFETY_FEATURE_SCHEMA",
    "VM_SAFETY_FEATURE_SCHEMA",
    "get_safety_feature_schema",
]
