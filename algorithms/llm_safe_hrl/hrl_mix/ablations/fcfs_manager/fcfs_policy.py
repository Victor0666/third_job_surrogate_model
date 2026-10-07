"""Reuse the formal deterministic baseline, including all tie-breaks."""
from algorithms.comparisons.fcfs.policies import fcfs_task_order


def order_ready_tasks_fcfs(env, task_ids):
    return fcfs_task_order(env, task_ids)
