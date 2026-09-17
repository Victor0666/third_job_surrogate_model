"""Generate versioned three-layer demonstrations from the real HRL environment.

The Manager action is a fixed admitted heuristic index for each phase.  The
heuristic orders ready tasks only.  Resource actions are selected with the
environment's existing deterministic VM rule from the safe/legal VM set; when
that set is empty, the environment's deterministic safety fallback is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from base.manager_heuristics import HEURISTIC_SELECTION_MODE
from base.safe_demonstration import (
    DEMONSTRATION_LAYERS,
    SAFE_DEMONSTRATION_GENERATOR_POLICY,
    DemonstrationSafetyStandard,
    SafeDemonstrationEpisode,
)
from base.safe_replay import SafeReplayTransition
from hrl_mix.train_utils import (
    finalize_action_audit,
    performance_reward_components,
    safe_replay_metadata,
)


@dataclass(frozen=True)
class DemonstrationGenerationOptions:
    """One real-environment demonstration generation request."""

    heuristic_id: str
    workflow_seed: int
    resource_seed: int
    split: str
    episode_id: str | None = None
    near_boundary_margin: float = 0.0
    max_manager_phases: int = 100000

    def __post_init__(self) -> None:
        if not str(self.heuristic_id).strip():
            raise ValueError("heuristic_id must not be empty")
        if str(self.split).lower() not in {
            "train",
            "validation",
            "final_test",
        }:
            raise ValueError("unsupported demonstration split")
        if (
            not np.isfinite(float(self.near_boundary_margin))
            or float(self.near_boundary_margin) < 0.0
        ):
            raise ValueError(
                "near_boundary_margin must be finite and non-negative"
            )
        if int(self.max_manager_phases) <= 0:
            raise ValueError("max_manager_phases must be positive")


def _copy_mask(state: Mapping, name: str) -> np.ndarray:
    value = state.get(name, state.get("mask"))
    if value is None:
        raise KeyError(f"layer state is missing {name}")
    return (np.asarray(value, dtype=np.float32).reshape(-1) > 0.5).astype(
        np.float32
    )


def _demonstration_selection(action: int, *, fallback: bool) -> dict:
    action = int(action)
    if fallback:
        return {
            "action": action,
            "proposed_action": None,
            "executed_action": action,
            "selected_by_agent": False,
            "selection_type": "fallback_action",
            "policy_selection_type": "fallback_action",
            "action_source": "fallback_action",
            "action_modified": False,
            "random_exploration": False,
            "safe_rl_enabled": True,
            "deterministic": True,
            "epsilon": 0.0,
            "valid_action_count": 0,
        }
    return {
        "action": action,
        "proposed_action": action,
        "executed_action": action,
        "selected_by_agent": True,
        "selection_type": "demonstration_fixed_vm_rule",
        "policy_selection_type": "demonstration_fixed_vm_rule",
        "action_source": "demonstration_fixed_vm_rule",
        "action_modified": False,
        "random_exploration": False,
        "safe_rl_enabled": True,
        "deterministic": True,
        "epsilon": 0.0,
    }


def _pending_transition(
    layer_state: Mapping,
    audit: Mapping,
    safety_info: Mapping,
    *,
    reward: float,
    cost: float,
    reward_components: Mapping,
    shield_decision: Mapping | None,
) -> dict:
    metadata = safe_replay_metadata(
        dict(layer_state),
        dict(audit),
        dict(safety_info),
        shield_decision=(
            dict(shield_decision)
            if shield_decision is not None
            else None
        ),
    )
    return {
        "state": np.asarray(
            layer_state["obs"], dtype=np.float32
        ).copy(),
        "proposed_action": audit.get("proposed_action"),
        "executed_action": int(audit["executed_action"]),
        "performance_reward": float(reward),
        "safety_cost": float(cost),
        "legal_action_mask": np.asarray(
            metadata["legal_action_mask"], dtype=np.float32
        ).copy(),
        "safety_action_mask": np.asarray(
            metadata["safety_action_mask"], dtype=np.float32
        ).copy(),
        "final_action_mask": np.asarray(
            metadata["final_action_mask"], dtype=np.float32
        ).copy(),
        "shield_modified": bool(audit["action_modified"]),
        "fallback_triggered": bool(
            metadata["fallback_triggered"]
        ),
        "fuzzy_safety_margin": float(
            metadata["fuzzy_safety_margin"]
        ),
        "predicted_risk_finish": float(
            metadata["predicted_risk_finish"]
        ),
        "violation_flag": bool(metadata["violation_flag"]),
        "manager_phase_id": int(metadata["manager_phase_id"]),
        "action_source": str(audit["action_source"]),
        "policy_selection_type": str(
            audit["policy_selection_type"]
        ),
        "performance_reward_components": dict(reward_components),
    }


def _commit_pending(
    trajectory: list[SafeReplayTransition],
    pending: dict | None,
    *,
    next_state,
    next_mask,
    done: bool,
    near_boundary_margin: float,
) -> None:
    if pending is None:
        return
    trajectory.append(
        SafeReplayTransition(
            **pending,
            next_state=np.asarray(
                next_state, dtype=np.float32
            ).copy(),
            next_final_action_mask=np.asarray(
                next_mask, dtype=np.float32
            ).copy(),
            done=float(bool(done)),
            near_boundary_margin=float(near_boundary_margin),
        )
    )


def _resolve_fixed_resource_actions(env) -> tuple[int, int, bool]:
    """Return host action, local VM action, and whether fallback is used."""
    context = env._current_safety_shield_context
    if context is None:
        raise RuntimeError("missing task safety context")
    global_final = np.asarray(
        context["vm_masks_global"]["final_action_mask"],
        dtype=np.float32,
    ).reshape(-1)
    safe_global_indices = np.flatnonzero(global_final > 0.5)
    fallback = safe_global_indices.size == 0
    if fallback:
        global_vm_index = context.get(
            "fallback_vm_global_index"
        )
        if global_vm_index is None:
            raise RuntimeError(
                "empty safe VM set has no deterministic fallback"
            )
        global_vm_index = int(global_vm_index)
    else:
        safe_vm_ids = [
            int(env.vm_ids[int(index)])
            for index in safe_global_indices
        ]
        vm_id, _ = env.select_vm_deterministic(
            env._cur_tid,
            candidate_vm_ids=safe_vm_ids,
        )
        global_vm_index = int(env.vm_ids.index(int(vm_id)))

    vm_id = int(env.vm_ids[global_vm_index])
    host_id = int(env.vms[vm_id].host_id)
    host_action = int(env.host_ids.index(host_id))
    vm_action = int(
        env.host_to_vm_indices[host_id].index(global_vm_index)
    )
    return host_action, vm_action, fallback


def _final_episode_metrics(env) -> dict:
    workflow_count = len(env.workflows)
    expected_workflow_count = int(
        env.workflows_per_episode
        if env.workflows_per_episode is not None
        else workflow_count
    )
    completed = 0
    violation_count = 0
    lateness_values = []
    for workflow_id, workflow in enumerate(env.workflows):
        if int(env.wf_remaining_tasks.get(workflow_id, 1)) != 0:
            continue
        completed += 1
        finish_tfn = env._workflow_finish_tfn(workflow_id)
        risk_finish = float(env.fuzzy_deadline_measure(finish_tfn))
        deadline = float(workflow.deadline)
        lateness = max(0.0, risk_finish - deadline)
        lateness_values.append(lateness)
        violation_count += int(lateness > 1e-9)

    energy = env.get_fuzzy_energy_summary()
    all_completed = bool(
        env.done_flag
        and expected_workflow_count > 0
        and workflow_count == expected_workflow_count
        and completed == expected_workflow_count
    )
    violation_rate = float(
        violation_count / max(workflow_count, 1)
    )
    max_lateness = float(max(lateness_values, default=0.0))
    return {
        "workflow_count": int(workflow_count),
        "expected_workflow_count": int(
            expected_workflow_count
        ),
        "completed_workflow_count": int(completed),
        "deadline_violation_count": int(violation_count),
        "deadline_violation_rate": violation_rate,
        "max_fuzzy_lateness": max_lateness,
        "constraint_feasible": bool(
            all_completed
            and violation_count == 0
            and max_lateness <= 1e-9
        ),
        "all_workflows_completed": all_completed,
        "fuzzy_energy_mean": float(
            energy["fuzzy_total_energy_mean"]
        ),
        "fuzzy_energy_std": float(
            energy["fuzzy_total_energy_std"]
        ),
        "fuzzy_energy_score": float(
            energy["fuzzy_total_energy_score"]
        ),
    }


def generate_safe_demonstration_episode(
    env,
    options: DemonstrationGenerationOptions,
    *,
    safety_standard: DemonstrationSafetyStandard | None = None,
) -> SafeDemonstrationEpisode:
    """Run one complete real environment episode and return replay rows."""
    if not bool(getattr(env, "safe_rl_enabled", False)):
        raise ValueError("demonstration generation requires safe_rl")
    if not bool(getattr(env, "safe_rl_shield_enabled", False)):
        raise ValueError(
            "demonstration generation requires the safety shield"
        )
    if not bool(getattr(env, "safe_rl_state_enabled", False)):
        raise ValueError(
            "demonstration generation requires safe observations"
        )
    if getattr(env, "manager_mode", None) != HEURISTIC_SELECTION_MODE:
        raise ValueError(
            "demonstrations require heuristic_selection_mode"
        )
    if int(env.random_seed) != int(options.workflow_seed):
        raise ValueError(
            "environment random_seed must equal workflow_seed"
        )
    if int(env.fuzzy_resource_seed) != int(options.resource_seed):
        raise ValueError(
            "environment fuzzy_resource_seed must equal resource_seed"
        )

    heuristic_indices = [
        index
        for index, heuristic in enumerate(env.manager_heuristics)
        if heuristic.heuristic_id == options.heuristic_id
    ]
    if len(heuristic_indices) != 1:
        raise ValueError(
            f"unknown or duplicate heuristic_id: "
            f"{options.heuristic_id}"
        )
    heuristic_index = int(heuristic_indices[0])
    heuristic = env.manager_heuristics[heuristic_index]
    if not heuristic.available:
        raise ValueError(
            f"heuristic is not admitted: {heuristic.heuristic_id}"
        )
    if env.get_manager_action_mask()[heuristic_index] <= 0.5:
        raise ValueError(
            "heuristic is unavailable for this workflow seed"
        )

    env.reset()
    trajectories = {
        layer: [] for layer in DEMONSTRATION_LAYERS
    }
    pending_host = None
    pending_vm = None
    phases = 0

    while not env.done_flag:
        phases += 1
        if phases > int(options.max_manager_phases):
            raise RuntimeError(
                "demonstration exceeded max_manager_phases"
            )
        manager_state = np.asarray(
            env.get_manager_state(), dtype=np.float32
        ).copy()
        manager_mask = (
            np.asarray(
                env.get_manager_action_mask(), dtype=np.float32
            ).reshape(-1)
            > 0.5
        ).astype(np.float32)
        if manager_mask[heuristic_index] <= 0.5:
            raise RuntimeError(
                "selected heuristic became unavailable mid-episode"
            )
        env.apply_manager_heuristic(heuristic_index)

        while True:
            host_state, has_host = (
                env.get_host_state_for_next_assignment()
            )
            if not has_host:
                break
            if pending_host is not None:
                _commit_pending(
                    trajectories["host"],
                    pending_host,
                    next_state=host_state["obs"],
                    next_mask=_copy_mask(
                        host_state, "final_action_mask"
                    ),
                    done=False,
                    near_boundary_margin=(
                        options.near_boundary_margin
                    ),
                )
                pending_host = None

            host_action, vm_action, fallback = (
                _resolve_fixed_resource_actions(env)
            )
            host_selection = _demonstration_selection(
                host_action, fallback=fallback
            )
            env.host_select(
                host_action,
                action_selection=host_selection,
            )
            vm_state, has_vm = (
                env.get_vm_state_for_current_task()
            )
            if not has_vm:
                raise RuntimeError(
                    "fixed resource choice produced no VM state"
                )
            if pending_vm is not None:
                _commit_pending(
                    trajectories["vm"],
                    pending_vm,
                    next_state=vm_state["obs"],
                    next_mask=_copy_mask(
                        vm_state, "final_action_mask"
                    ),
                    done=False,
                    near_boundary_margin=(
                        options.near_boundary_margin
                    ),
                )
                pending_vm = None

            vm_selection = _demonstration_selection(
                vm_action, fallback=fallback
            )
            r_host, r_vm, task_info = env.vm_assign(
                vm_action,
                action_selection=vm_selection,
            )
            host_audit = finalize_action_audit(
                host_selection,
                task_info.get("host_shield_decision", {}),
            )
            vm_audit = finalize_action_audit(
                vm_selection,
                task_info.get("vm_shield_decision", {}),
            )
            host_reward = float(
                task_info.get(
                    "total_performance_reward",
                    task_info.get(
                        "performance_reward_host", r_host
                    ),
                )
            )
            vm_reward = float(
                task_info.get(
                    "total_performance_reward",
                    task_info.get("performance_reward_vm", r_vm),
                )
            )
            safety_cost = float(
                task_info.get("safety_cost", 0.0)
            )
            pending_host = _pending_transition(
                host_state,
                host_audit,
                task_info,
                reward=host_reward,
                cost=safety_cost,
                reward_components=performance_reward_components(
                    task_info,
                    total_performance_reward=host_reward,
                ),
                shield_decision=task_info.get(
                    "host_shield_decision", {}
                ),
            )
            pending_vm = _pending_transition(
                vm_state,
                vm_audit,
                task_info,
                reward=vm_reward,
                cost=safety_cost,
                reward_components=performance_reward_components(
                    task_info,
                    total_performance_reward=vm_reward,
                ),
                shield_decision=task_info.get(
                    "vm_shield_decision", {}
                ),
            )

        manager_reward_raw, phase_info = (
            env.finish_phase_and_advance()
        )
        manager_next_state = np.asarray(
            env.get_manager_state(), dtype=np.float32
        ).copy()
        manager_next_mask = (
            np.asarray(
                env.get_manager_action_mask(), dtype=np.float32
            ).reshape(-1)
            > 0.5
        ).astype(np.float32)
        manager_reward = float(
            phase_info.get(
                "total_performance_reward",
                phase_info.get(
                    "performance_reward", manager_reward_raw
                ),
            )
        )
        manager_cost = float(
            phase_info.get(
                "heuristic_phase_safety_cost",
                phase_info.get("safety_cost", 0.0),
            )
        )
        manager_audit = {
            "proposed_action": heuristic_index,
            "executed_action": heuristic_index,
            "action_modified": False,
            "fallback_triggered": False,
            "action_source": "demonstration_safe_heuristic",
            "policy_selection_type": (
                "demonstration_safe_heuristic"
            ),
        }
        manager_layer_state = {
            "obs": manager_state,
            "mask": manager_mask,
            "legal_action_mask": manager_mask,
            "safety_action_mask": manager_mask,
            "final_action_mask": manager_mask,
        }
        manager_pending = _pending_transition(
            manager_layer_state,
            manager_audit,
            phase_info,
            reward=manager_reward,
            cost=manager_cost,
            reward_components=performance_reward_components(
                phase_info,
                total_performance_reward=manager_reward,
            ),
            shield_decision=None,
        )
        _commit_pending(
            trajectories["manager"],
            manager_pending,
            next_state=manager_next_state,
            next_mask=manager_next_mask,
            done=bool(env.done_flag),
            near_boundary_margin=options.near_boundary_margin,
        )

        if env.done_flag:
            if pending_host is not None:
                _commit_pending(
                    trajectories["host"],
                    pending_host,
                    next_state=np.zeros_like(
                        pending_host["state"]
                    ),
                    next_mask=np.zeros_like(
                        pending_host["final_action_mask"]
                    ),
                    done=True,
                    near_boundary_margin=(
                        options.near_boundary_margin
                    ),
                )
                pending_host = None
            if pending_vm is not None:
                _commit_pending(
                    trajectories["vm"],
                    pending_vm,
                    next_state=np.zeros_like(
                        pending_vm["state"]
                    ),
                    next_mask=np.zeros_like(
                        pending_vm["final_action_mask"]
                    ),
                    done=True,
                    near_boundary_margin=(
                        options.near_boundary_margin
                    ),
                )
                pending_vm = None

    metrics = _final_episode_metrics(env)
    standard = (
        safety_standard
        if safety_standard is not None
        else DemonstrationSafetyStandard()
    )
    episode_id = options.episode_id or (
        f"{options.split}.wf{int(options.workflow_seed)}."
        f"res{int(options.resource_seed)}."
        f"{heuristic.heuristic_id}"
    )
    return SafeDemonstrationEpisode(
        episode_id=episode_id,
        split=str(options.split).lower(),
        generator_policy=(
            SAFE_DEMONSTRATION_GENERATOR_POLICY
        ),
        heuristic_id=heuristic.heuristic_id,
        heuristic_source=heuristic.source,
        heuristic_version=heuristic.version,
        workflow_seed=int(options.workflow_seed),
        resource_seed=int(options.resource_seed),
        ddl_setting={
            "deadline_mode": str(env.deadline_mode),
            "deadline_alpha_small": float(
                env.deadline_alpha_small
            ),
            "deadline_alpha_large": float(
                env.deadline_alpha_large
            ),
            "deadline_alpha_small_prob": float(
                env.deadline_alpha_small_prob
            ),
            "workflow_deadlines": [
                float(workflow.deadline)
                for workflow in env.workflows
            ],
        },
        fuzzy_parameters={
            "delta1": float(env.fuzzy_delta1),
            "delta2": float(env.fuzzy_delta2),
            "deadline_eta": float(env.fuzzy_deadline_eta),
            "energy_uncertainty_weight": float(
                env.fuzzy_energy_uncertainty_weight
            ),
            "resource_seed": int(env.fuzzy_resource_seed),
        },
        observation_schema_versions={
            layer: str(
                env.get_observation_schema(layer)[
                    "schema_version"
                ]
            )
            for layer in DEMONSTRATION_LAYERS
        },
        episode_metrics=metrics,
        trajectories=trajectories,
        safety_standard=standard,
    )


__all__ = [
    "DemonstrationGenerationOptions",
    "generate_safe_demonstration_episode",
]
