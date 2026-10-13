"""Joint ready-task decisions on the unchanged FCFS simulator."""
import json
import time

import numpy as np

from algorithms.comparisons.dtecn_llm.policy import DtecnPolicy
from algorithms.comparisons.dtecn_llm.environment import interval_feedback


BATCH_PROMPT = """Allocate the current FCFS task batch jointly. Return only JSON
{\"assignments\":[{\"task_id\":integer,\"vm_id\":integer},...]}, in the given task order.
Include every batch task exactly once. Each VM must be legal for its task and
used at most once. Host follows VM ownership. Do not reorder or select tasks.
Tasks are indivisible; CPU frequency control and busy-VM queueing are unavailable.
Minimize J=beta*sum(Q*(A-S))+observed_modal_energy-system_QoS.
Queues are unfinished arrived CPU work; transmission does not serve CPU queues.
MADDPG examples use reward=QoS-eta*unused_capacity, distinct from J.
Feedback is observed, not a future rollout. Use examples to decide the joint mapping.
Task observation fields: active, owner, log1p(remaining_MI), elapsed/budget,
five workflow-family flags, log1p(task_MI/input_bits/output_bits), deadline_remaining/budget.
"""


def matching(candidates):
    """Check feasibility, never optimize energy or delay."""
    owners = {}

    def augment(task, seen):
        for vm in candidates[task]:
            if vm in seen:
                continue
            seen.add(vm)
            if vm not in owners or augment(owners[vm], seen):
                owners[vm] = task
                return True
        return False

    if not all(augment(task, set()) for task in range(len(candidates))):
        return None
    return [next(vm for vm, task in owners.items() if task == i)
            for i in range(len(candidates))]


def stable_assignment(candidates):
    result, used = [], set()
    for i, legal in enumerate(candidates):
        for vm in legal:
            if vm in used:
                continue
            remaining = [[v for v in row if v not in used and v != vm]
                         for row in candidates[i + 1:]]
            if matching(remaining) is not None:
                result.append(vm)
                used.add(vm)
                break
        else:
            raise ValueError("batch has no complete legal assignment")
    return result


def parse_assignments(content, task_ids, candidates):
    data = json.loads(content)
    if not isinstance(data, dict) or set(data) != {"assignments"}:
        raise ValueError("expected assignments object")
    rows = data["assignments"]
    if not isinstance(rows, list) or len(rows) != len(task_ids):
        raise ValueError("batch length mismatch")
    used, result = set(), []
    for task, legal, row in zip(task_ids, candidates, rows):
        if not isinstance(row, dict) or set(row) != {"task_id", "vm_id"}:
            raise ValueError("invalid assignment fields")
        if type(row["task_id"]) is not int or row["task_id"] != task:
            raise ValueError("task order/identity mismatch")
        vm = row["vm_id"]
        if type(vm) is not int or vm not in legal or vm in used:
            raise ValueError("illegal or conflicting VM")
        used.add(vm)
        result.append(vm)
    return result


class BatchPolicy(DtecnPolicy):
    def __init__(self, *args, batch_callback=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_callback = batch_callback

    def begin_episode(self, env, *, training):
        if training:
            raise ValueError("batch policy is for collection/ICL; MADDPG training is unchanged")
        super().begin_episode(env, training=False)
        self.batch = None
        self.batch_id = 0
        self.stats.update(batches=0, unique_batch_skips=0)

    def _finish_batch(self, env):
        if self.batch is None:
            return
        batch = self.batch
        if batch["next"] != len(batch["tasks"]):
            raise RuntimeError("incomplete batch before feedback")
        rewards, feedback = interval_feedback(env, batch["before"], env.snapshot(), self.config["objective"])
        case = dict(batch_id=batch["id"], state=batch["state"],
                    assignments=batch["assignments"],
                    rewards={str(owner): float(rewards[owner]) for owner in batch["owners"]},
                    feedback=feedback, action_source=batch["source"],
                    seed=int(env.random_seed), scenario=env.scenario_code)
        self.stats["feedback_count"] += 1
        self.stats["lyapunov_objective_sum"] += feedback["objective"]
        case["request"] = batch["request"]
        if self.collect:
            self.collected.append(case)
        if self.client is not None:
            self.online_cases.append(case)
        if self.batch_callback:
            self.batch_callback(case)
        self.batch = None

    def _start_batch(self, env):
        self._finish_batch(env)
        request_before = {key: self.stats[key] for key in (
            "requests", "failures", "fallbacks", "llm_seconds", "reported_tokens", "usage_missing")}
        snapshot = env.snapshot()
        current = int(env._cur_tid)
        ids = [current] + list(env._phase_tasks)
        tasks, candidates, owners = [], [], []
        idle = {int(vid) for i, vid in enumerate(env.vm_ids)
                if env.vm_available_at[i] <= env.current_time + 1e-9}
        try:
            for tid in ids:
                legal = sorted(idle.intersection(env.get_feasible_vms(tid)))
                if matching(candidates + [legal]) is None:
                    break
                env._cur_tid = tid
                observations, _, owner = env.observations(snapshot)
                tasks.append(dict(task_id=int(tid), workflow_slot=owner,
                                  observation=observations[owner, :13].tolist(), legal_vm_ids=legal))
                owners.append(owner)
                candidates.append(legal)
        finally:
            env._cur_tid = current
        if not tasks:
            raise RuntimeError("FCFS head has no legal idle VM")
        state = dict(time=float(env.current_time), tasks=tasks,
                     vm_ids=list(map(int, env.vm_ids)),
                     resources=[dict(vm_id=int(vid), pc=float(env.vms[vid].pc), bw=float(env.vms[vid].bw),
                                     busy_seconds=max(0., float(env.vm_available_at[i] - env.current_time)))
                                for i, vid in enumerate(env.vm_ids)],
                     queue_mi=snapshot["queue"].tolist(),
                     arrived_workflows=[dict(slot=i, remaining_mi=float(snapshot["remaining"][i]),
                                             qos=float(snapshot["qos"][i]), active=float(snapshot["active"][i]))
                                        for i in range(len(env.workflows))])
        self.batch_id += 1
        self.stats["batches"] += 1
        plan, source = None, "maddpg"
        if self.client is not None:
            plan = stable_assignment(candidates)
            unique = all(matching([([v for v in row if v != plan[i]] if j == i else row)
                                   for j, row in enumerate(candidates)]) is None
                         for i in range(len(candidates)))
            source = "unique_batch"
            if unique:
                self.stats["unique_batch_skips"] += 1
            else:
                recent = self.online_cases[-self.config["llm"]["online_example_count"]:]
                initial = list(self.initial_cases[:self.config["llm"]["initial_example_count"]])
                while True:
                    prompt = BATCH_PROMPT + "\nConstants: " + json.dumps(self.config["objective"])
                    prompt += "\nMARL examples: " + json.dumps(initial, separators=(",", ":"))
                    prompt += "\nObserved batch feedback: " + json.dumps(recent, separators=(",", ":"))
                    prompt += "\nCurrent state: " + json.dumps(state, separators=(",", ":"))
                    if len(prompt) <= self.config["llm"]["prompt_char_budget"]:
                        break
                    if recent:
                        recent = recent[1:]
                    elif len(initial) > 1:
                        initial.pop()
                    else:
                        raise ValueError("prompt budget cannot hold current batch and initial example")
                self.stats["requests"] += 1
                started, usage = time.perf_counter(), None
                try:
                    content, usage = self.client(prompt)
                    plan = parse_assignments(content, [t["task_id"] for t in tasks], candidates)
                    source = "llm_batch"
                except Exception as exc:
                    if getattr(exc, "status_code", None) in (400, 401, 403, 404):
                        raise
                    if not isinstance(exc, (ValueError, TimeoutError, ConnectionError)):
                        from openai import APIConnectionError, APIStatusError
                        if not isinstance(exc, (APIConnectionError, APIStatusError)):
                            raise
                    plan = stable_assignment(candidates)
                    source = "legal_batch_fallback"
                    state["request_error"] = type(exc).__name__
                    self.stats["failures"] += 1
                    self.stats["fallbacks"] += 1
                finally:
                    self.stats["llm_seconds"] += time.perf_counter() - started
                    if usage is None or usage.get("total_tokens") is None:
                        self.stats["usage_missing"] += 1
                    else:
                        self.stats["reported_tokens"] += int(usage["total_tokens"])
        self.batch = dict(id=self.batch_id, before=snapshot, state=state, tasks=tasks,
                          candidates=candidates, owners=owners, plan=plan, source=source,
                          assignments=[], next=0, used=set(),
                          request={key: self.stats[key] - value for key, value in request_before.items()})

    def assign(self, env, host_state, *, training):
        if training:
            raise ValueError("online batches do not train MADDPG")
        if self.batch is None or self.batch["next"] == len(self.batch["tasks"]):
            self._start_batch(env)
        batch = self.batch
        i = batch["next"]
        task = batch["tasks"][i]
        if task["task_id"] != env._cur_tid or batch["state"]["time"] != env.current_time:
            raise RuntimeError("FCFS order/time changed inside a batch")
        observations, masks, owner = env.observations(env.snapshot())
        if batch["plan"] is None:
            preferences = self.agent.preferences(observations, owner, False)
            viable = []
            for vm in batch["candidates"][i]:
                if vm in batch["used"]:
                    continue
                remaining = [[v for v in row if v not in batch["used"] and v != vm]
                             for row in batch["candidates"][i + 1:]]
                if matching(remaining) is not None:
                    viable.append(vm)
            vm_id = max(viable, key=lambda v: (preferences[env._vm_index_by_id[v]], -env._vm_index_by_id[v]))
        else:
            vm_id = batch["plan"][i]
            preferences = np.zeros(env.num_vms)
            preferences[env._vm_index_by_id[vm_id]] = 1.
        index = env._vm_index_by_id[vm_id]
        if masks[owner, index] <= .5:
            raise RuntimeError("prevalidated batch action became illegal")
        result = env.assign_global_vm(index)
        batch["used"].add(vm_id)
        batch["assignments"].append(dict(task_id=task["task_id"], vm_id=int(vm_id)))
        batch["next"] += 1
        self.pending = dict(owner=owner, preferences=np.asarray(preferences), state=batch["state"],
                            source=batch["source"], batch_id=batch["id"], batch_size=len(batch["tasks"]))
        self.decision_records.append(dict(batch_id=batch["id"], batch_size=len(batch["tasks"]),
                                          task_id=task["task_id"], executed_vm_id=int(vm_id),
                                          action_source=batch["source"]))
        return result

    def end_episode(self, env, *, training):
        self._finish_batch(env)
