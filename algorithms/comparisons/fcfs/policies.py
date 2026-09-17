"""Deterministic FCFS policies on the shared fuzzy baseline environment."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


METHOD_DISPLAY_NAMES = {
    "fcfs_fcfs": "FCFS-FCFS",
    "fcfs_fixed": "FCFS-Fixed",
}


def fcfs_task_order(environment, ready_task_ids: Sequence[int]) -> list[int]:
    """Order ready tasks by the common deterministic FCFS key."""

    def order_key(task_id: int) -> tuple[float, float, int, int]:
        workflow_id, _ = environment.task_meta[int(task_id)]
        workflow = environment.workflows[int(workflow_id)]
        return (
            float(environment.task_ready_time[int(task_id)]),
            float(workflow.arrival_time),
            int(workflow_id),
            int(task_id),
        )

    return sorted((int(value) for value in ready_task_ids), key=order_key)


def _legal_vm_indices(environment) -> list[int]:
    return [
        int(index)
        for index in np.flatnonzero(
            np.asarray(environment.global_vm_action_mask()) > 0.5
        )
    ]


def select_vm_fcfs(environment) -> int:
    """Return the legal VM index with earliest availability, then VM id.

    The selector intentionally reads no deadline, energy, risk, shield, RL or
    LLM state. The shared legal-action mask remains authoritative.
    """
    legal = _legal_vm_indices(environment)
    if not legal:
        raise RuntimeError("FCFS-FCFS assignment has no legal idle VM")
    return min(
        legal,
        key=lambda index: (
            float(environment.vm_available_at[index]),
            int(environment.vm_ids[index]),
        ),
    )


def select_vm_fixed(environment) -> int:
    """Delegate VM selection to the shared deterministic fixed rule."""
    if environment._cur_tid is None:
        raise RuntimeError("FCFS-Fixed assignment has no current task")
    legal = _legal_vm_indices(environment)
    if not legal:
        raise RuntimeError("FCFS-Fixed assignment has no legal idle VM")
    legal_vm_ids = [int(environment.vm_ids[index]) for index in legal]
    selected_vm_id, _ = environment.select_vm_deterministic(
        int(environment._cur_tid),
        candidate_vm_ids=legal_vm_ids,
    )
    index_by_vm_id = {
        int(environment.vm_ids[index]): int(index) for index in legal
    }
    try:
        return index_by_vm_id[int(selected_vm_id)]
    except KeyError as exc:
        raise RuntimeError(
            "shared fixed VM rule returned a VM outside the legal set"
        ) from exc


class _BaseFCFSPolicy:
    """Shared lifecycle for evaluation-only FCFS task policies."""

    method_id = ""
    display_name = ""
    vm_selector: Callable[[object], int]

    def begin_episode(self, env, *, training: bool) -> None:
        if training:
            raise ValueError(f"{self.display_name} is evaluation-only")
        env.set_task_orderer(
            lambda ready_ids, _features, _training: fcfs_task_order(
                env, ready_ids
            ),
            training=False,
        )

    def assign(self, env, host_state, *, training: bool):
        del host_state, training
        return env.assign_global_vm(self.vm_selector(env))

    def observe_assignment(self, env, result, *, training: bool) -> None:
        del env, result, training

    def end_phase(self, env, reward, info, *, training: bool) -> None:
        del env, reward, info, training

    def end_episode(self, env, *, training: bool) -> None:
        del env, training

    def save(self, path) -> None:
        del path

    def load(self, path) -> None:
        del path


class FCFSFCFSPolicy(_BaseFCFSPolicy):
    """FCFS task ordering plus earliest-available-first VM selection."""

    method_id = "fcfs_fcfs"
    display_name = METHOD_DISPLAY_NAMES[method_id]
    vm_selector = staticmethod(select_vm_fcfs)


class FCFSFixedPolicy(_BaseFCFSPolicy):
    """FCFS task ordering plus the shared deterministic fixed VM rule."""

    method_id = "fcfs_fixed"
    display_name = METHOD_DISPLAY_NAMES[method_id]
    vm_selector = staticmethod(select_vm_fixed)


POLICY_TYPES = {
    FCFSFCFSPolicy.method_id: FCFSFCFSPolicy,
    FCFSFixedPolicy.method_id: FCFSFixedPolicy,
}


def make_policy(method_id: str) -> _BaseFCFSPolicy:
    """Construct one explicitly named FCFS baseline policy."""
    try:
        policy_type = POLICY_TYPES[str(method_id)]
    except KeyError as exc:
        raise ValueError(
            f"unknown FCFS method_id {method_id!r}; expected one of "
            f"{sorted(POLICY_TYPES)}"
        ) from exc
    return policy_type()


__all__ = [
    "FCFSFCFSPolicy",
    "FCFSFixedPolicy",
    "METHOD_DISPLAY_NAMES",
    "POLICY_TYPES",
    "fcfs_task_order",
    "make_policy",
    "select_vm_fcfs",
    "select_vm_fixed",
]
