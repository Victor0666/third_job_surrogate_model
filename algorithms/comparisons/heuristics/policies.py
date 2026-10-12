"""SPT, workflow EDD and descendant-work MWKR with shared fixed VM selection."""

from __future__ import annotations

import math

from algorithms.comparisons.fcfs.policies import FCFSFixedPolicy, fcfs_task_order


TASK_ORDER_DESCRIPTIONS = {
    "fcfs_fcfs": "(ready_time, workflow_arrival_time, workflow_id, task_id)",
    "fcfs_fixed": "(ready_time, workflow_arrival_time, workflow_id, task_id)",
    "spt_fixed": "minimum feasible-VM modal execution time; FCFS tie-break",
    "edd_fixed": "workflow absolute deadline; FCFS tie-break",
    "mwkr_fixed": "descending current-task plus unique descendant work (MI); FCFS tie-break",
}


def heuristic_task_order(env, ready_ids, method_id):
    """Sort raw task values, without using log-normalized observations."""
    ids = fcfs_task_order(env, ready_ids)
    scores = {}
    for task_id in ids:
        if method_id == "spt_fixed":
            vms = env.get_feasible_vms(task_id)
            if not vms:
                raise ValueError(f"Task {task_id} has no feasible VM")
            score = min(env.estimate_exec_time(task_id, vm) for vm in vms)
        elif method_id == "edd_fixed":
            workflow_id, _ = env.task_meta[task_id]
            score = env.workflows[workflow_id].deadline
        elif method_id == "mwkr_fixed":
            score = -env.calculate_remaining_work(task_id)
        else:
            raise ValueError(f"Unknown heuristic: {method_id}")
        if score is None or not math.isfinite(float(score)):
            raise ValueError(f"Invalid {method_id} score for task {task_id}: {score}")
        scores[task_id] = float(score)
    # Stable sorting preserves the shared FCFS order for equal scores.
    return sorted(ids, key=scores.__getitem__)


class SPTFixedPolicy(FCFSFixedPolicy):
    method_id = "spt_fixed"
    display_name = "SPT-Fixed"

    def begin_episode(self, env, *, training):
        if training:
            raise ValueError(f"{self.display_name} is evaluation-only")
        env.set_task_orderer(
            lambda ready_ids, _features, _training: heuristic_task_order(
                env, ready_ids, self.method_id
            ),
            training=False,
        )


class EDDFixedPolicy(SPTFixedPolicy):
    method_id = "edd_fixed"
    display_name = "EDD-Fixed"


class MWKRFixedPolicy(SPTFixedPolicy):
    method_id = "mwkr_fixed"
    display_name = "MWKR-Fixed"


POLICY_TYPES = {
    policy.method_id: policy
    for policy in (FCFSFixedPolicy, SPTFixedPolicy, EDDFixedPolicy, MWKRFixedPolicy)
}
