"""Observed workload queues on the unchanged combined-service simulator."""
from pathlib import Path
import math

import numpy as np

from algorithms.comparisons.fuzzy_common.environment import FuzzyBaselineEnv
from common.scheduling_transport import COMMUNICATION_MODEL_VERSION
from common.workflow_opt import LoadRecord, energy_from_records

FAMILIES = ("CyberShake", "Epigenomics", "Ligo", "Montage", "Sipht")


def family_index(path):
    name = Path(path).stem.split("_")[0]
    if name not in FAMILIES:
        raise ValueError(f"unregistered workflow family: {name}")
    return FAMILIES.index(name)


def completed_work(work, compute_start, compute_end, now):
    if not all(map(math.isfinite, (work, compute_start, compute_end, now))) or work < 0 or compute_end <= compute_start:
        raise ValueError("CPU interval and workload must be finite and non-negative with positive duration")
    return float(work) * min(1.0, max(0.0, (now - compute_start) / (compute_end - compute_start)))


class DtecnEnv(FuzzyBaselineEnv):
    """Read-only observations; no alternative timing, power or safety model."""

    def reset(self, *args, **kwargs):
        result = super().reset(*args, **kwargs)
        if COMMUNICATION_MODEL_VERSION != "combined_input_compute_output_v1":
            raise ValueError("DTECN artifacts require combined_input_compute_output_v1")
        self.cpu_intervals = {}
        return result

    def assign_global_vm(self, index):
        result = super().assign_global_vm(index)
        history = self.assignment_history[-1]
        duration = self.estimate_task_duration_components_scenario(result.task_id, result.vm_id, "modal")
        start = history["input_arrival_time"]
        self.cpu_intervals[result.task_id] = (start, start + duration["execution_time"])
        return result

    def snapshot(self):
        """Only arrived work and executed CPU service enter queue conservation."""
        now = float(self.current_time)
        totals = np.zeros(5)
        service = np.zeros(5)
        per_workflow = np.zeros(self.workflows_per_episode)
        qos = np.zeros(self.workflows_per_episode)
        active = np.zeros(self.workflows_per_episode)
        for tid, (wid, _) in enumerate(self.task_meta):
            family = family_index(self.workflows[wid].dax_path)
            work = float(self.task_mi[tid])
            done = completed_work(work, *self.cpu_intervals[tid], now) if tid in self.cpu_intervals else 0.0
            totals[family] += work
            service[family] += done
            per_workflow[wid] += max(0.0, work - done)
        for wf in self.workflows:
            wid = int(wf.workflow_id)
            end = wf.finish_time if wf.finish_time is not None else now
            qos[wid] = 1.0 - (float(end) - wf.arrival_time) / max(wf.deadline - wf.arrival_time, 1e-9)
            active[wid] = float(wf.finish_time is None)
        pc = sum(float(vm.pc) for vm in self.vms.values())
        used = sum(float(self.vms[self.task_assigned_vm[tid]].pc)
                   for tid, (start, end) in self.cpu_intervals.items() if start <= now < end)
        return dict(time=now, totals=totals, service=service, queue=np.maximum(totals - service, 0),
                    remaining=per_workflow, qos=qos, active=active,
                    unused_capacity=max(0.0, 1.0 - used / max(pc, 1e-9)))

    def observed_energy(self, until):
        """Clip committed modal load records at the observed clock, never future finishes."""
        records = [LoadRecord(r.start_time, min(r.end_time, until), r.server_id, r.vm_pc)
                   for r in self._records if r.start_time < until]
        return float(energy_from_records(records, self.hosts))

    def observations(self, snapshot):
        """Fixed workflow slots, zeros for unarrived slots, one FCFS decision slot."""
        count = int(self.workflows_per_episode)
        obs = np.zeros((count, self.observation_dim), dtype=np.float32)
        masks = np.zeros((count, self.num_vms), dtype=np.float32)
        tid = self._cur_tid
        owner = int(self.task_meta[tid][0]) if tid is not None else None
        vm_features = []
        for index, vid in enumerate(self.vm_ids):
            vm = self.vms[vid]
            vm_features.extend((float(vm.pc), float(vm.bw),
                                max(0.0, self.vm_available_at[index] - self.current_time)))
        vm_features = np.log1p(vm_features).tolist()
        for wf in self.workflows:
            wid = int(wf.workflow_id)
            family = np.eye(5)[family_index(wf.dax_path)].tolist()
            local = [snapshot["active"][wid], float(wid == owner),
                     np.log1p(snapshot["remaining"][wid]),
                     (self.current_time - wf.arrival_time) / max(wf.deadline - wf.arrival_time, 1e-9)]
            task = ([np.log1p(self.task_mi[tid]), np.log1p(self.task_in_bits[tid]),
                     np.log1p(self.task_out_bits[tid]),
                     (self._task_deadline(tid) - self.current_time) / max(wf.deadline - wf.arrival_time, 1e-9)]
                    if wid == owner else [0.0] * 4)
            obs[wid] = local + family + task + np.log1p(snapshot["queue"]).tolist() + vm_features
        if owner is not None:
            masks[owner] = self.global_vm_action_mask()
        if not np.isfinite(obs).all():
            raise ValueError("non-finite observed state")
        return obs, masks, owner

    @property
    def observation_dim(self):
        return 18 + 3 * self.num_vms


def interval_feedback(env, before, after, config):
    arrivals = after["totals"] - before["totals"]
    service = after["service"] - before["service"]
    queue = before["queue"]
    if np.any(arrivals < -1e-7) or np.any(service < -1e-7):
        raise ValueError("arrivals/service cannot decrease")
    expected = queue + arrivals - service
    if not np.allclose(after["queue"], expected, atol=1e-6):
        raise ValueError("workload queue conservation failed")
    drift = float(np.dot(queue, arrivals - service))
    energy = env.observed_energy(after["time"]) - env.observed_energy(before["time"])
    arrived = len(env.workflows)
    qos = float(np.mean(after["qos"][:arrived])) if arrived else 0.0
    objective = config["beta"] * drift + energy - qos
    # Eq. 27 and Eq. 24 are separate in the paper; do not replace reward by -objective.
    rewards = after["qos"] - config["eta"] * after["unused_capacity"]
    present = np.arange(len(rewards)) < arrived
    rewards = rewards * present
    return rewards.astype(np.float32), dict(
        elapsed=after["time"] - before["time"], arrivals_mi=arrivals.tolist(),
        service_mi=service.tolist(), queue_before_mi=queue.tolist(),
        queue_after_mi=after["queue"].tolist(), drift_term=drift,
        actual_drift=float((np.dot(after["queue"], after["queue"]) - np.dot(queue, queue)) / 2),
        modal_energy_j=energy, system_qos=qos, objective=float(objective),
        unused_capacity=float(after["unused_capacity"]))
