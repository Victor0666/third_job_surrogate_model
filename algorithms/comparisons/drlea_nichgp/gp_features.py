"""Versioned 14-terminal global-ready-task GP features."""

from __future__ import annotations

import numpy as np

from .config import NormalizationConfig
from .env_adapter import CEWSEnvAdapter


GP_TERMINAL_VERSION = "drlea_gp14_v1"
GP_TERMINALS = (
    "READY_COUNT",
    "READY_WORK",
    "MIN_INPUT_COMM",
    "MIN_OUTPUT_COMM",
    "MIN_COMP",
    "MIN_ENERGY",
    "WF_AGE",
    "TASK_WAIT",
    "TIME_TO_DEADLINE",
    "FUZZY_SLACK",
    "REMAIN_TASKS",
    "REMAIN_WORK",
    "TOTAL_WORK",
    "UNCERTAINTY",
)


def _positive(value: float, scale: float) -> float:
    return float(
        np.clip(
            max(0.0, float(value)) / max(float(scale), 1e-12),
            0.0,
            1.0,
        )
    )


def _signed(value: float, scale: float) -> float:
    value = float(value)
    return float(value / (abs(value) + max(float(scale), 1e-12)))


def task_gp_terminals(
    adapter: CEWSEnvAdapter,
    task_id: int,
    normalization: NormalizationConfig,
    ready_tasks: list[int] | None = None,
) -> np.ndarray:
    env = adapter.env
    task_id = int(task_id)
    ready = adapter.ready_tasks() if ready_tasks is None else list(ready_tasks)
    workflow_id = adapter.task_workflow_id(task_id)
    workflow = adapter.workflow(workflow_id)
    feasible = adapter.feasible_vms(task_id)
    if not feasible:
        raise ValueError("GP task has no feasible VM")
    input_comm = []
    output_comm = []
    computation = []
    energy = []
    uncertainty = []
    for vm_id in feasible:
        components = adapter.modal_components(task_id, vm_id)
        input_comm.append(components["input_communication_time"])
        output_comm.append(components["output_communication_time"])
        computation.append(components["execution_time"])
        energy.append(
            adapter.task_vm_prediction(task_id, vm_id)[
                "incremental_fuzzy_energy"
            ]
        )
        uncertainty.append(adapter.uncertainty(task_id, vm_id))
    workflow_task_ids = adapter.workflow_task_ids(workflow_id)
    remaining_task_ids = [
        value
        for value in workflow_task_ids
        if env.task_state[value] != "Finished"
    ]
    values = np.asarray(
        [
            _positive(len(ready), normalization.ready_count),
            _positive(
                sum(env.task_mi[value] for value in ready),
                normalization.remaining_work_mi,
            ),
            _positive(min(input_comm), normalization.time_seconds),
            _positive(min(output_comm), normalization.time_seconds),
            _positive(min(computation), normalization.time_seconds),
            _positive(min(energy), normalization.energy_joules),
            _positive(
                adapter.current_time - float(workflow.arrival_time),
                normalization.time_seconds,
            ),
            _positive(
                adapter.current_time - env.task_ready_time[task_id],
                normalization.time_seconds,
            ),
            _signed(
                float(workflow.deadline) - adapter.current_time,
                normalization.time_seconds,
            ),
            _signed(
                adapter.task_fuzzy_slack(task_id),
                normalization.time_seconds,
            ),
            _positive(
                len(remaining_task_ids),
                max(len(workflow_task_ids), 1),
            ),
            _positive(
                sum(env.task_mi[value] for value in remaining_task_ids),
                normalization.remaining_work_mi,
            ),
            _positive(
                sum(env.task_mi[value] for value in workflow_task_ids),
                normalization.remaining_work_mi,
            ),
            _positive(min(uncertainty), normalization.time_seconds),
        ],
        dtype=np.float64,
    )
    if values.shape != (14,) or not np.all(np.isfinite(values)):
        raise ValueError("GP terminals must be 14 finite values")
    return values


def choose_task_with_rule(
    adapter: CEWSEnvAdapter,
    program,
    ready_tasks: list[int] | None = None,
) -> tuple[int, list[float]]:
    ready = adapter.ready_tasks() if ready_tasks is None else list(ready_tasks)
    if not ready:
        raise ValueError("GP rule requires a non-empty ready set")
    scores = [
        float(
            program(
                task_gp_terminals(
                    adapter,
                    task_id,
                    adapter.config.normalization,
                    ready,
                )
            )
        )
        for task_id in ready
    ]
    if not np.all(np.isfinite(np.asarray(scores, dtype=np.float64))):
        raise ValueError("GP rule produced NaN or Inf")
    index = min(
        range(len(ready)),
        key=lambda value: (
            scores[value],
            adapter.fcfs_key(ready[value]),
        ),
    )
    return int(ready[index]), scores
