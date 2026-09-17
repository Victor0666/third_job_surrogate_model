"""Adapters exposing the current fuzzy scheduler to comparison policies.

The adapter does not reimplement execution, communication, DDL or energy
models. All state transitions and predictions are delegated to
``HrlFcfsCacheEnv``. Safety shielding and safe value learning remain disabled
for these non-safe comparison methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from . import LLM_SAFE_HRL_ROOT  # noqa: F401
from base.hrl_env import HrlFcfsCacheEnv


TASK_FEATURE_NAMES = (
    "min_exec_time",
    "min_comm_time",
    "min_incremental_energy",
    "slack",
    "upward_rank",
    "remaining_work",
    "ready_wait_time",
    "uncertainty",
)


def _signed_log(value: float) -> float:
    scalar = float(value)
    return float(np.sign(scalar) * np.log1p(abs(scalar)))


def _legal_mask(state: dict) -> np.ndarray:
    for key in ("final_action_mask", "legal_action_mask", "mask"):
        if key in state:
            return np.asarray(state[key], dtype=np.float32).reshape(-1)
    raise ValueError("environment state does not contain an action mask")


@dataclass(frozen=True)
class FuzzyAssignmentResult:
    """One assignment result using the existing fuzzy environment model."""

    task_id: int
    vm_id: int
    host_id: int
    reward: float
    ddl_reward: float
    energy_reward: float
    predicted_violation: float
    predicted_safety_margin: float
    fuzzy_energy_delta: float
    env_host_reward: float
    env_vm_reward: float
    info: dict


TaskOrderer = Callable[[Sequence[int], np.ndarray, bool], Sequence[int]]


class FuzzyBaselineEnv(HrlFcfsCacheEnv):
    """Fuzzy scheduling environment shared by IRWS, MARL and PD3QN."""

    GLOBAL_VM_FEATURE_DIM = 8

    def __init__(
        self,
        *args,
        ddl_reward_fraction: float = 0.75,
        energy_reward_fraction: float = 0.25,
        comparison_energy_scale: float = 1e-3,
        comparison_tardiness_normalizer: float = 300.0,
        **kwargs,
    ) -> None:
        if bool(kwargs.get("safe_rl_enabled", False)):
            raise ValueError("comparison baselines must not enable safe_rl")
        super().__init__(*args, **kwargs)
        fractions = (
            float(ddl_reward_fraction),
            float(energy_reward_fraction),
        )
        if any(not np.isfinite(value) or value < 0.0 for value in fractions):
            raise ValueError("comparison reward fractions must be finite and non-negative")
        if not np.isclose(sum(fractions), 1.0):
            raise ValueError("comparison reward fractions must sum to one")
        self.ddl_reward_fraction = fractions[0]
        self.energy_reward_fraction = fractions[1]
        self.comparison_energy_scale = float(comparison_energy_scale)
        self.comparison_tardiness_normalizer = float(
            comparison_tardiness_normalizer
        )
        if self.comparison_energy_scale <= 0.0:
            raise ValueError("comparison_energy_scale must be positive")
        if self.comparison_tardiness_normalizer <= 0.0:
            raise ValueError("comparison_tardiness_normalizer must be positive")
        self._comparison_task_orderer: TaskOrderer | None = None
        self._comparison_training = False
        self.global_vm_obs_dim = int(
            self.manager_obs_dim
            + len(TASK_FEATURE_NAMES)
            + self.num_vms * self.GLOBAL_VM_FEATURE_DIM
        )

    def set_task_orderer(
        self,
        orderer: TaskOrderer | None,
        *,
        training: bool,
    ) -> None:
        self._comparison_task_orderer = orderer
        self._comparison_training = bool(training)

    def ready_task_feature_matrix(
        self,
        task_ids: Sequence[int] | None = None,
    ) -> tuple[list[int], np.ndarray]:
        ids = [
            int(value)
            for value in (
                self.get_ready_tasks() if task_ids is None else task_ids
            )
        ]
        if not ids:
            return [], np.zeros((0, len(TASK_FEATURE_NAMES)), dtype=np.float32)
        features = self.build_task_features(ids)
        matrix = np.column_stack(
            [features[name] for name in TASK_FEATURE_NAMES]
        ).astype(np.float32)
        if matrix.shape != (len(ids), len(TASK_FEATURE_NAMES)):
            raise RuntimeError("ready-task feature shape mismatch")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("ready-task features contain NaN or infinity")
        normalized = np.vectorize(_signed_log, otypes=[float])(matrix)
        return ids, np.asarray(normalized, dtype=np.float32)

    def _phase_prepare_tasks(self) -> None:
        """Build one deterministic phase ordering before any assignment."""
        self._add_workflow_if_arrived()
        ready_ids = [int(value) for value in self.get_ready_tasks()]
        if not ready_ids:
            self._phase_tasks = []
            self._phase_ready_task_ordering = []
            return
        if self._comparison_task_orderer is None:
            ordering = sorted(
                ready_ids,
                key=lambda task_id: (
                    float(self.task_ready_time[task_id]),
                    int(task_id),
                ),
            )
        else:
            _, matrix = self.ready_task_feature_matrix(ready_ids)
            ordering = [
                int(value)
                for value in self._comparison_task_orderer(
                    tuple(ready_ids),
                    matrix.copy(),
                    self._comparison_training,
                )
            ]
            if len(ordering) != len(ready_ids) or set(ordering) != set(ready_ids):
                raise ValueError(
                    "task orderer must return each current ready task exactly once"
                )
        self._phase_tasks = list(ordering)
        self._phase_ready_task_ordering = list(ordering)

    def global_vm_action_mask(self) -> np.ndarray:
        if self._cur_tid is None:
            raise RuntimeError("no current task; request a host state first")
        feasible = set(self.get_feasible_vms(int(self._cur_tid)))
        return np.asarray(
            [
                1.0
                if int(vm_id) in feasible
                and float(self.vm_available_at[index])
                <= float(self.current_time) + 1e-9
                else 0.0
                for index, vm_id in enumerate(self.vm_ids)
            ],
            dtype=np.float32,
        )

    def _current_task_vector(self) -> np.ndarray:
        if self._cur_tid is None:
            raise RuntimeError("no current task")
        features = self.build_task_features([int(self._cur_tid)])
        return np.asarray(
            [_signed_log(features[name][0]) for name in TASK_FEATURE_NAMES],
            dtype=np.float32,
        )

    def global_vm_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Return a fixed-size global VM observation for the current task."""
        if self._cur_tid is None:
            raise RuntimeError("no current task; request a host state first")
        task_id = int(self._cur_tid)
        now = float(self.current_time)
        mean_pc = max(
            float(np.mean([float(self.vms[value].pc) for value in self.vm_ids])),
            1e-9,
        )
        mean_bw = max(
            float(np.mean([float(self.vms[value].bw) for value in self.vm_ids])),
            1e-9,
        )
        vm_rows = []
        for index, vm_id in enumerate(self.vm_ids):
            vm = self.vms[vm_id]
            host_indices = self.host_to_vm_indices[int(vm.host_id)]
            busy = sum(
                float(self.vm_available_at[item]) > now + 1e-9
                for item in host_indices
            )
            host_load = float(busy / max(len(host_indices), 1))
            row = (
                1.0 if float(self.vm_available_at[index]) <= now + 1e-9 else 0.0,
                _signed_log(max(0.0, float(self.vm_available_at[index]) - now)),
                float(vm.pc) / mean_pc,
                float(vm.bw) / mean_bw,
                _signed_log(self.estimate_exec_time(task_id, int(vm_id))),
                _signed_log(self.estimate_comm_time(task_id, int(vm_id))),
                _signed_log(self.estimate_incremental_energy_score(task_id, int(vm_id))),
                host_load,
            )
            vm_rows.extend(row)
        observation = np.concatenate(
            [
                np.asarray(self.get_manager_state(), dtype=np.float32),
                self._current_task_vector(),
                np.asarray(vm_rows, dtype=np.float32),
            ]
        ).astype(np.float32)
        if observation.size != self.global_vm_obs_dim:
            raise RuntimeError("global VM observation dimension mismatch")
        if not np.all(np.isfinite(observation)):
            raise ValueError("global VM observation contains NaN or infinity")
        mask = self.global_vm_action_mask()
        if not np.any(mask > 0.5):
            raise RuntimeError("current assignment has no idle legal VM")
        return observation, mask

    def _assignment_prediction(self, vm_id: int) -> tuple[float, dict]:
        if self._cur_tid is None:
            raise RuntimeError("no current task")
        before = float(
            self.get_fuzzy_energy_summary()["fuzzy_total_energy_score"]
        )
        prediction = self.predict_task_vm_action_risk(
            int(self._cur_tid),
            int(vm_id),
        )
        return before, prediction

    def _finalize_assignment_result(
        self,
        *,
        task_id: int,
        vm_id: int,
        host_id: int,
        before_energy: float,
        prediction: dict,
        env_rewards: tuple[float, float, dict],
    ) -> FuzzyAssignmentResult:
        r_host, r_vm, info = env_rewards
        if int(info.get("invalid", 0)):
            raise RuntimeError(f"environment rejected legal comparison action: {info}")
        after_energy = float(
            self.get_fuzzy_energy_summary()["fuzzy_total_energy_score"]
        )
        energy_delta = max(0.0, after_energy - float(before_energy))
        predicted_violation = max(
            0.0,
            float(prediction["predicted_violation_amount"]),
        )
        ddl_reward = -float(
            np.clip(
                predicted_violation / self.comparison_tardiness_normalizer,
                0.0,
                1.0,
            )
        )
        energy_reward = -energy_delta * self.comparison_energy_scale
        reward = (
            self.ddl_reward_fraction * ddl_reward
            + self.energy_reward_fraction * energy_reward
        )
        details = dict(info)
        details["comparison_reward_components"] = {
            "ddl_reward": float(ddl_reward),
            "energy_reward": float(energy_reward),
            "ddl_fraction": float(self.ddl_reward_fraction),
            "energy_fraction": float(self.energy_reward_fraction),
            "fuzzy_energy_delta": float(energy_delta),
            "predicted_violation": float(predicted_violation),
            "used_for_checkpoint_selection": False,
        }
        return FuzzyAssignmentResult(
            task_id=int(task_id),
            vm_id=int(vm_id),
            host_id=int(host_id),
            reward=float(reward),
            ddl_reward=float(ddl_reward),
            energy_reward=float(energy_reward),
            predicted_violation=float(predicted_violation),
            predicted_safety_margin=float(prediction["safety_margin"]),
            fuzzy_energy_delta=float(energy_delta),
            env_host_reward=float(r_host),
            env_vm_reward=float(r_vm),
            info=details,
        )

    def assign_global_vm(self, vm_global_index: int) -> FuzzyAssignmentResult:
        index = int(vm_global_index)
        mask = self.global_vm_action_mask()
        if index < 0 or index >= self.num_vms or mask[index] <= 0.5:
            raise ValueError("global VM action is not currently legal")
        task_id = int(self._cur_tid)
        vm_id = int(self.vm_ids[index])
        host_id = int(self.vms[vm_id].host_id)
        host_index = int(self.host_ids.index(host_id))
        before, prediction = self._assignment_prediction(vm_id)
        self.host_select(host_index)
        _, available = self.get_vm_state_for_current_task()
        if not available:
            raise RuntimeError("selected host has no local VM state")
        slot = int(self.host_to_vm_indices[host_id].index(index))
        env_rewards = self.vm_assign(slot)
        return self._finalize_assignment_result(
            task_id=task_id,
            vm_id=vm_id,
            host_id=host_id,
            before_energy=before,
            prediction=prediction,
            env_rewards=env_rewards,
        )

    def assign_local_vm(self, vm_slot: int) -> FuzzyAssignmentResult:
        if self._cur_tid is None or self._cur_host_id is None:
            raise RuntimeError("host must be selected before local VM assignment")
        task_id = int(self._cur_tid)
        host_id = int(self._cur_host_id)
        indices = self.host_to_vm_indices[host_id]
        slot = int(vm_slot)
        if slot < 0 or slot >= len(indices):
            raise ValueError("local VM slot is out of range")
        vm_id = int(self.vm_ids[int(indices[slot])])
        before, prediction = self._assignment_prediction(vm_id)
        env_rewards = self.vm_assign(slot)
        return self._finalize_assignment_result(
            task_id=task_id,
            vm_id=vm_id,
            host_id=host_id,
            before_energy=before,
            prediction=prediction,
            env_rewards=env_rewards,
        )


__all__ = [
    "FuzzyAssignmentResult",
    "FuzzyBaselineEnv",
    "TASK_FEATURE_NAMES",
    "_legal_mask",
]
