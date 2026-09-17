"""Normalized RA and fixed-width SA observations."""

from __future__ import annotations

import numpy as np

from .config import NormalizationConfig
from .env_adapter import CEWSEnvAdapter


RA_TASK_FEATURES = (
    "workload",
    "input_size",
    "output_size",
    "fuzzy_slack",
    "upward_rank",
    "remaining_work",
    "ready_wait_time",
    "uncertainty",
)
RA_VM_FEATURES = (
    "legal",
    "available_wait",
    "modal_comm",
    "modal_exec",
    "modal_finish",
    "risk_finish",
    "incremental_fuzzy_energy",
    "host_utilization",
    "host_type",
    "fuzzy_uncertainty",
)
SA_FEATURES = (
    "ready_count",
    "min_exec_mean",
    "min_exec_std",
    "min_exec_max",
    "min_exec_min",
    "workflow_completion_mean",
    "workflow_completion_std",
    "workflow_completion_max",
    "workflow_completion_min",
    "fuzzy_slack_mean",
    "fuzzy_slack_std",
    "predicted_fuzzy_lateness_mean",
    "predicted_fuzzy_lateness_std",
    "predicted_fuzzy_lateness_max",
    "predicted_fuzzy_lateness_min",
    "at_risk_workflow_ratio",
)


def _positive(value: float, scale: float) -> float:
    value = max(0.0, float(value))
    return float(np.clip(value / max(float(scale), 1e-12), 0.0, 1.0))


def _signed(value: float, scale: float) -> float:
    value = float(value)
    return float(value / (abs(value) + max(float(scale), 1e-12)))


def _summary(values: list[float]) -> tuple[float, float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    array = np.asarray(values, dtype=np.float64)
    return (
        float(np.mean(array)),
        float(np.std(array)),
        float(np.max(array)),
        float(np.min(array)),
    )


def routing_state(
    adapter: CEWSEnvAdapter,
    task_id: int,
    normalization: NormalizationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    env = adapter.env
    task_id = int(task_id)
    legal_mask = adapter.legal_vm_mask(task_id)
    legal_ids = set(adapter.legal_vm_ids(task_id))
    uncertainty_values = [
        adapter.uncertainty(task_id, vm_id)
        for vm_id in adapter.vm_ids
    ]
    task_uncertainty = min(uncertainty_values, default=0.0)
    task_features = [
        _positive(env.task_mi[task_id], normalization.workload_mi),
        _positive(env.task_in_bits[task_id], normalization.data_bits),
        _positive(env.task_out_bits[task_id], normalization.data_bits),
        _signed(
            adapter.task_fuzzy_slack(task_id),
            normalization.time_seconds,
        ),
        _positive(
            env.calculate_upward_rank(task_id),
            normalization.rank_seconds,
        ),
        _positive(
            env.calculate_remaining_work(task_id),
            normalization.remaining_work_mi,
        ),
        _positive(
            adapter.current_time - env.task_ready_time[task_id],
            normalization.time_seconds,
        ),
        _positive(task_uncertainty, normalization.time_seconds),
    ]
    vm_features = []
    for index, vm_id in enumerate(adapter.vm_ids):
        prediction = adapter.task_vm_prediction(task_id, vm_id)
        vm = env.vms[int(vm_id)]
        host = env.hosts[int(vm.host_id)]
        available_wait = max(
            0.0,
            float(env.vm_available_at[index]) - adapter.current_time,
        )
        vm_features.extend(
            [
                1.0 if vm_id in legal_ids else 0.0,
                _positive(available_wait, normalization.time_seconds),
                _positive(
                    prediction["current_task_communication_time_modal"],
                    normalization.time_seconds,
                ),
                _positive(
                    prediction["current_task_execution_time_modal"],
                    normalization.time_seconds,
                ),
                _positive(
                    prediction["modal_finish"] - adapter.current_time,
                    normalization.time_seconds,
                ),
                _positive(
                    prediction["risk_finish"] - adapter.current_time,
                    normalization.time_seconds,
                ),
                _positive(
                    prediction["incremental_fuzzy_energy"],
                    normalization.energy_joules,
                ),
                adapter.host_utilization(int(vm.host_id)),
                1.0 if str(host.server_type).lower() == "cloud" else 0.0,
                _positive(
                    adapter.uncertainty(task_id, vm_id),
                    normalization.time_seconds,
                ),
            ]
        )
    state = np.asarray(task_features + vm_features, dtype=np.float32)
    expected_dim = 8 + 10 * adapter.num_vms
    if state.shape != (expected_dim,) or not np.all(np.isfinite(state)):
        raise ValueError("invalid RA observation")
    return state, legal_mask


def sequencing_state(
    adapter: CEWSEnvAdapter,
    normalization: NormalizationConfig,
    ready_tasks: list[int] | None = None,
) -> np.ndarray:
    ready = adapter.ready_tasks() if ready_tasks is None else list(ready_tasks)
    if not ready:
        return np.zeros(16, dtype=np.float32)
    min_exec = []
    workflow_ids = sorted(
        {adapter.task_workflow_id(task_id) for task_id in ready}
    )
    for task_id in ready:
        feasible = adapter.feasible_vms(task_id)
        values = [
            adapter.env.estimate_exec_time(task_id, vm_id)
            for vm_id in feasible
        ]
        min_exec.append(
            _positive(min(values), normalization.time_seconds)
            if values
            else 0.0
        )
    completion = [
        adapter.workflow_completion_ratio(workflow_id)
        for workflow_id in workflow_ids
    ]
    margins = []
    lateness = []
    for workflow_id in workflow_ids:
        prediction = adapter.workflow_finish_prediction(workflow_id)
        margins.append(
            _signed(prediction["margin"], normalization.time_seconds)
        )
        lateness.append(
            _positive(prediction["lateness"], normalization.time_seconds)
        )
    min_exec_stats = _summary(min_exec)
    completion_stats = _summary(completion)
    margin_array = np.asarray(margins, dtype=np.float64)
    lateness_stats = _summary(lateness)
    state = np.asarray(
        [
            _positive(len(ready), normalization.ready_count),
            *min_exec_stats,
            *completion_stats,
            float(np.mean(margin_array)) if margins else 0.0,
            float(np.std(margin_array)) if margins else 0.0,
            *lateness_stats,
            float(
                np.mean(np.asarray(lateness, dtype=np.float64) > 0.0)
            )
            if lateness
            else 0.0,
        ],
        dtype=np.float32,
    )
    if state.shape != (16,) or not np.all(np.isfinite(state)):
        raise ValueError("invalid SA observation")
    return state
