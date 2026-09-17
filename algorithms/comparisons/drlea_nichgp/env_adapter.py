"""Narrow problem-level adapter over the current fuzzy CEWS environment."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np

from algorithms.llm_safe_hrl.base.hrl_env import HrlFcfsCacheEnv
from project_paths import PROJECT_ROOT

from .config import ComparisonConfig


class CEWSEnvAdapter:
    """Expose only simulator state and actions needed by DRL-EA/NichGP."""

    def __init__(self, config: ComparisonConfig, seed: int):
        self.config = config
        self.seed = int(seed)
        dax_paths = [
            str(
                path
                if path.is_absolute()
                else PROJECT_ROOT / path
            )
            for value in config.dax_paths
            for path in (Path(value),)
        ]
        cache_path = Path(config.deadline_cache_path)
        if not cache_path.is_absolute():
            cache_path = PROJECT_ROOT / cache_path
        self.env = HrlFcfsCacheEnv(
            dax_paths=dax_paths,
            horizon=float(config.horizon),
            arrival_lambda=float(config.arrival_lambda),
            random_seed=self.seed,
            max_ready_tasks="auto",
            normalize=True,
            workflows_per_episode=int(config.workflows_per_episode),
            num_cloud_hosts=int(config.num_cloud_hosts),
            num_edge_hosts=int(config.num_edge_hosts),
            cloud_vms_per_host=tuple(config.cloud_vms_per_host),
            edge_vms_per_host=tuple(config.edge_vms_per_host),
            cloud_pc_tiers=tuple(config.cloud_pc_tiers),
            edge_pc_tiers=tuple(config.edge_pc_tiers),
            cloud_bw_tiers=tuple(config.cloud_bw_tiers),
            edge_bw_tiers=tuple(config.edge_bw_tiers),
            deadline_mode="cache_fcfs",
            deadline_cache_path=str(cache_path),
            deadline_cache_strict=True,
            deadline_alpha_small=float(config.deadline_alpha_small),
            deadline_alpha_large=float(config.deadline_alpha_large),
            deadline_alpha_small_prob=float(
                config.deadline_alpha_small_prob
            ),
            fuzzy_enabled=True,
            fuzzy_delta1=float(config.fuzzy_delta1),
            fuzzy_delta2=float(config.fuzzy_delta2),
            fuzzy_energy_uncertainty_weight=float(
                config.fuzzy_energy_lambda
            ),
            fuzzy_deadline_eta=float(config.fuzzy_deadline_eta),
            fuzzy_resource_seed=self.seed,
            fuzzy_use_deadline_constraint=True,
            safe_rl_enabled=False,
            safe_rl_shield_enabled=False,
            safe_rl_state_enabled=False,
            scenario_code=config.scenario,
            task_code=config.scenario[0],
            resource_code=config.scenario[1],
        )
        self.no_legal_vm_advances = 0
        self.assignment_count = 0
        self._prediction_cache: dict[tuple[int, int], dict] = {}
        self._feasible_vms_cache: dict[int, tuple[int, ...]] = {}
        self._modal_components_cache: dict[tuple[int, int], dict] = {}
        self._uncertainty_cache: dict[tuple[int, int], float] = {}
        if self.vm_ids != tuple(range(self.num_vms)):
            raise ValueError(
                "RA action indices require contiguous global_vm_id values"
            )

    @property
    def current_time(self) -> float:
        return float(self.env.current_time)

    @property
    def done(self) -> bool:
        return bool(self.env.done_flag)

    @property
    def vm_ids(self) -> tuple[int, ...]:
        return tuple(int(vm_id) for vm_id in self.env.vm_ids)

    @property
    def num_vms(self) -> int:
        return len(self.vm_ids)

    def reset(self) -> None:
        self.env.reset()
        self.no_legal_vm_advances = 0
        self.assignment_count = 0
        self._prediction_cache.clear()
        self._feasible_vms_cache.clear()
        self._modal_components_cache.clear()
        self._uncertainty_cache.clear()

    @staticmethod
    def _cache_audit_enabled() -> bool:
        return os.environ.get("DRLEA_CACHE_AUDIT", "0") == "1"

    def feasible_vms(self, task_id: int) -> list[int]:
        task_id = int(task_id)
        cached = self._feasible_vms_cache.get(task_id)
        if cached is None:
            cached = tuple(
                int(vm_id)
                for vm_id in self.env.get_feasible_vms(task_id)
            )
            self._feasible_vms_cache[task_id] = cached
        elif self._cache_audit_enabled():
            actual = tuple(
                int(vm_id)
                for vm_id in self.env.get_feasible_vms(task_id)
            )
            if actual != cached:
                raise AssertionError("feasible-VM cache changed")
        return list(cached)

    def modal_components(self, task_id: int, vm_id: int) -> dict:
        key = (int(task_id), int(vm_id))
        cached = self._modal_components_cache.get(key)
        if cached is None:
            cached = self.env.estimate_task_duration_components_scenario(
                key[0], key[1], "modal"
            )
            self._modal_components_cache[key] = dict(cached)
        elif self._cache_audit_enabled():
            actual = self.env.estimate_task_duration_components_scenario(
                key[0], key[1], "modal"
            )
            if actual != cached:
                raise AssertionError("modal-components cache changed")
        return dict(cached)

    def uncertainty(self, task_id: int, vm_id: int) -> float:
        key = (int(task_id), int(vm_id))
        cached = self._uncertainty_cache.get(key)
        if cached is None:
            cached = float(self.env.calculate_uncertainty(*key))
            self._uncertainty_cache[key] = cached
        elif self._cache_audit_enabled():
            actual = float(self.env.calculate_uncertainty(*key))
            if actual != cached:
                raise AssertionError("uncertainty cache changed")
        return float(cached)

    def ready_tasks(self) -> list[int]:
        return sorted(
            self.env.get_ready_tasks(),
            key=self.fcfs_key,
        )

    def fcfs_key(self, task_id: int) -> tuple:
        workflow_id, local_id = self.env.task_meta[int(task_id)]
        return (
            float(self.env.task_ready_time[int(task_id)]),
            int(workflow_id),
            int(local_id),
            int(task_id),
        )

    def select_fcfs_task(self) -> int:
        ready = self.ready_tasks()
        if not ready:
            raise ValueError("FCFS selection requires a ready task")
        return int(ready[0])

    def legal_vm_ids(self, task_id: int) -> list[int]:
        feasible = set(self.feasible_vms(int(task_id)))
        legal = []
        for index, vm_id in enumerate(self.env.vm_ids):
            if int(vm_id) not in feasible:
                continue
            if (
                self.config.allow_busy_vm_queueing
                or float(self.env.vm_available_at[index])
                <= self.current_time + 1e-9
            ):
                legal.append(int(vm_id))
        return legal

    def legal_vm_mask(self, task_id: int) -> np.ndarray:
        legal = set(self.legal_vm_ids(task_id))
        return np.asarray(
            [1.0 if vm_id in legal else 0.0 for vm_id in self.vm_ids],
            dtype=np.float32,
        )

    def advance_until_actionable(self, max_advances: int = 1_000_000):
        """Return the next FCFS task or ``None`` at episode completion."""

        advances = 0
        while not self.done:
            ready = self.ready_tasks()
            if ready:
                task_id = int(ready[0])
                if self.legal_vm_ids(task_id):
                    return task_id
                self.env.advance_to_next_resource_event()
                self.no_legal_vm_advances += 1
                self._prediction_cache.clear()
            else:
                self.env.advance_to_next_event()
                self._prediction_cache.clear()
            advances += 1
            if advances > int(max_advances):
                raise RuntimeError("event advancement exceeded safety limit")
        return None

    def task_workflow_id(self, task_id: int) -> int:
        return int(self.env.task_meta[int(task_id)][0])

    def task_object(self, task_id: int):
        workflow_id, local_id = self.env.task_meta[int(task_id)]
        return self.env.workflows[int(workflow_id)].tasks[int(local_id)]

    def workflow(self, workflow_id: int):
        return self.env.workflows[int(workflow_id)]

    def task_deadline(self, task_id: int) -> float:
        task = self.task_object(task_id)
        deadline = getattr(task, "sub_deadline", None)
        if deadline is None or not np.isfinite(float(deadline)):
            deadline = self.workflow(
                self.task_workflow_id(task_id)
            ).deadline
        return float(deadline)

    def workflow_deadline(self, workflow_id: int) -> float:
        return float(self.workflow(workflow_id).deadline)

    def task_vm_prediction(self, task_id: int, vm_id: int) -> dict:
        key = (int(task_id), int(vm_id))
        cached = self._prediction_cache.get(key)
        if cached is not None:
            return dict(cached)
        finish = self.env.estimate_task_finish_tfn(
            int(task_id), int(vm_id)
        )
        risk = self.env.fuzzy_deadline_measure(finish)
        components = self.modal_components(int(task_id), int(vm_id))
        prediction = {
            "task_id": int(task_id),
            "vm_id": int(vm_id),
            "optimistic_finish": float(finish.lower),
            "modal_finish": float(finish.modal),
            "pessimistic_finish": float(finish.upper),
            "risk_finish": float(risk),
            "current_task_communication_time_modal": float(
                components["communication_time"]
            ),
            "current_task_execution_time_modal": float(
                components["execution_time"]
            ),
            "incremental_fuzzy_energy": float(
            self.env.estimate_incremental_energy_score(
                int(task_id), int(vm_id)
            )
            ),
        }
        self._prediction_cache[key] = dict(prediction)
        return prediction

    def all_vm_predictions(self, task_id: int) -> dict[int, dict]:
        return {
            vm_id: self.task_vm_prediction(task_id, vm_id)
            for vm_id in self.vm_ids
        }

    def expected_risk_finish(self, task_id: int) -> float:
        legal = self.legal_vm_ids(task_id)
        if not legal:
            raise ValueError("expected finish requires a legal VM")
        return float(
            np.mean(
                [
                    self.task_vm_prediction(task_id, vm_id)[
                        "risk_finish"
                    ]
                    for vm_id in legal
                ]
            )
        )

    def task_fuzzy_slack(self, task_id: int) -> float:
        feasible = self.feasible_vms(int(task_id))
        if not feasible:
            raise ValueError("fuzzy slack requires a feasible VM")
        earliest_risk = min(
            self.task_vm_prediction(task_id, vm_id)["risk_finish"]
            for vm_id in feasible
        )
        return float(self.task_deadline(task_id) - earliest_risk)

    def execute(self, task_id: int, vm_id: int) -> dict:
        if int(vm_id) not in self.legal_vm_ids(task_id):
            raise ValueError("DRL-EA attempted an illegal VM action")
        prediction = self.task_vm_prediction(task_id, vm_id)
        before = self.env.get_fuzzy_energy_summary()
        assignment = self.env.assign_task(int(task_id), int(vm_id))
        after = self.env.get_fuzzy_energy_summary()
        self.assignment_count += 1
        self._prediction_cache.clear()
        return {
            **assignment,
            **prediction,
            "fuzzy_energy_score_before": float(
                before["fuzzy_total_energy_score"]
            ),
            "fuzzy_energy_score_after": float(
                after["fuzzy_total_energy_score"]
            ),
            "incremental_fuzzy_energy_actual": float(
                after["fuzzy_total_energy_score"]
                - before["fuzzy_total_energy_score"]
            ),
        }

    def workflow_finish_prediction(self, workflow_id: int) -> dict:
        finish = self.env.predict_workflow_finish_tfn(int(workflow_id))
        risk = self.env.fuzzy_deadline_measure(finish)
        deadline = self.workflow_deadline(workflow_id)
        return {
            "lower": float(finish.lower),
            "modal": float(finish.modal),
            "upper": float(finish.upper),
            "risk": float(risk),
            "deadline": float(deadline),
            "margin": float(deadline - risk),
            "lateness": float(max(0.0, risk - deadline)),
        }

    def active_workflow_ids(self) -> list[int]:
        return [
            int(workflow.workflow_id)
            for workflow in self.env.workflows
            if int(self.env.wf_remaining_tasks.get(
                int(workflow.workflow_id), 0
            ))
            > 0
        ]

    def aggregate_risk_slack(self) -> float:
        active = self.active_workflow_ids()
        if not active:
            return 0.0
        margins = [
            self.workflow_finish_prediction(workflow_id)["margin"]
            for workflow_id in active
        ]
        return float(np.mean(margins))

    def host_utilization(self, host_id: int) -> float:
        indices = self.env.host_to_vm_indices[int(host_id)]
        if not indices:
            return 0.0
        busy = sum(
            float(self.env.vm_available_at[index])
            > self.current_time + 1e-9
            for index in indices
        )
        return float(busy / len(indices))

    def workflow_completion_ratio(self, workflow_id: int) -> float:
        task_ids = [
            task_id
            for task_id, (owner, _local) in enumerate(self.env.task_meta)
            if int(owner) == int(workflow_id)
        ]
        if not task_ids:
            return 0.0
        finished = sum(
            self.env.task_state[task_id] == "Finished"
            for task_id in task_ids
        )
        return float(finished / len(task_ids))

    def workflow_task_ids(self, workflow_id: int) -> list[int]:
        return [
            int(task_id)
            for task_id, (owner, _local) in enumerate(self.env.task_meta)
            if int(owner) == int(workflow_id)
        ]

    def instance_descriptor(self) -> dict:
        resources = []
        for vm_id in self.vm_ids:
            vm = self.env.vms[int(vm_id)]
            resources.append(
                {
                    "vm_id": int(vm_id),
                    "host_id": int(vm.host_id),
                    "pc": [
                        float(vm.pc.lower),
                        float(vm.pc.modal),
                        float(vm.pc.upper),
                    ],
                    "bw": [
                        float(vm.bw.lower),
                        float(vm.bw.modal),
                        float(vm.bw.upper),
                    ],
                }
            )
        workflows = []
        for workflow in self.env.workflows:
            workflows.append(
                {
                    "workflow_id": int(workflow.workflow_id),
                    "arrival_time": float(workflow.arrival_time),
                    "dax": Path(
                        str(getattr(workflow, "dax_path", ""))
                    ).name,
                    "deadline": float(workflow.deadline),
                    "task_count": len(workflow.tasks),
                }
            )
        return {
            "scenario": self.config.scenario,
            "ddl": self.config.ddl,
            "seed": self.seed,
            "dax_inputs": list(self.config.dax_paths),
            "arrival_times": [
                float(value) for value in self.env.arrival_times
            ],
            "resources": resources,
            "workflows": workflows,
            "fuzzy": {
                "delta1": self.config.fuzzy_delta1,
                "delta2": self.config.fuzzy_delta2,
                "eta": self.config.fuzzy_deadline_eta,
                "lambda_energy": self.config.fuzzy_energy_lambda,
            },
        }

    def instance_fingerprint(self) -> str:
        payload = json.dumps(
            self.instance_descriptor(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def ensure_finite(self, values: Iterable[float], name: str) -> None:
        array = np.asarray(list(values), dtype=np.float64)
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains NaN or Inf")
