"""Physical event semantics, algorithm identity and shared evaluator parity."""

import heapq
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from algorithms.comparisons.mtgp.core import (
    RulePair, TERMINALS, TREE_SCHEMA, evaluate_tree, crossover, task_terminals,
)
from algorithms.comparisons.drlea_nichgp.gp_primitives import GPProgram, program_depth
from algorithms.comparisons.drlea_nichgp.env_adapter import CEWSEnvAdapter
from algorithms.comparisons.drlea_nichgp.metrics import episode_metrics, aggregate_seed_metrics
from algorithms.llm_safe_hrl.base.hrl_env import HrlHeftEnv
from algorithms.llm_safe_hrl.hrl_mix.safe_metrics import build_episode_metric_record, aggregate_safe_metric_records
from common.scheduling_transport import execution_window, COMMUNICATION_MODEL_VERSION
from common.resource_opt import TriangularFuzzyNumber
from project_paths import PROJECT_ROOT


def environment(inputs=(4,), computes=(8,), outputs=(2,)):
    env = HrlHeftEnv(
        dax_paths=[str(PROJECT_ROOT / "data/dax/CyberShake_30.xml")],
        num_cloud_hosts=1, num_edge_hosts=0, cloud_vms_per_host=(1,), edge_vms_per_host=(1,),
        cloud_pc_tiers=(1.0,), cloud_bw_tiers=(1.0,), workflows_per_episode=1,
        fuzzy_enabled=True,
    )
    env.vms[0].pc = TriangularFuzzyNumber(1.0, 1.0, 1.0)
    env.vms[0].bw = TriangularFuzzyNumber(1.0, 1.0, 1.0)
    count = len(inputs)
    tasks = [SimpleNamespace(task_id=index, state="Ready") for index in range(count)]
    graph = nx.DiGraph()
    graph.add_nodes_from(range(count))
    env.workflows = [SimpleNamespace(workflow_id=0, arrival_time=0.0, deadline=100.0, tasks=tasks, graph=graph)]
    env.task_meta = [(0, index) for index in range(count)]
    env.task_state = ["Ready"] * count
    env.task_parents = [[] for _ in range(count)]
    env.task_children = [[] for _ in range(count)]
    env.task_global_parents = [[] for _ in range(count)]
    env.task_ready_time = [0.0] * count
    env.task_end_time = [0.0] * count
    env.task_mi = list(computes)
    env.task_in_bits = [value * 1e6 for value in inputs]
    env.task_out_bits = [value * 1e6 for value in outputs]
    env.ready_task_ids = list(range(count))
    env.wf_remaining_tasks = {0: count}
    env._workflow_local_to_global_cache = {0: {index: index for index in range(count)}}
    env._workflow_task_ids_cache = {0: tuple(range(count))}
    for scenario in ("optimistic", "pessimistic"):
        env.shadow_task_end_time[scenario] = [0.0] * count
        env.shadow_task_start_time[scenario] = [0.0] * count
    return env


def drain(env):
    while not env.done_flag:
        env.advance_to_next_resource_event()


def tree(*tokens):
    return GPProgram(tokens, TERMINALS, TREE_SCHEMA)


def test_input_transfer_waits_for_vm_and_counts_as_execution_load():
    assert execution_window(10, 4, 20, 8, 2) == (24, 20, 34)
    env = environment()
    env.current_time = 10.0
    env.vm_available_at[0] = 20.0
    for scenario in env.shadow_vm_available_at:
        env.shadow_vm_available_at[scenario][0] = 20.0
    assert env.estimate_task_finish_tfn(0, 0).as_tuple() == (34.0, 34.0, 34.0)
    env.route_task_to_vm(0, 0)
    assert env.task_state == ["Queued"]
    assert env.vm_waiting_queues[0] == [0]
    assert env._records == []
    env.current_time = 20.0
    env._dispatch_vm_queue(0)
    assert [(row.start_time, row.end_time) for row in env._records] == [(20.0, 34.0)]
    drain(env)
    assert env.wf_finish_time[0] == 34.0


def test_sequencing_chooses_vm_service_and_idle_routing_starts_directly():
    env = environment((2, 4, 20), (1, 8, 2), (0, 0, 0))
    env.vm_available_at[0] = 10.0
    for scenario in env.shadow_vm_available_at:
        env.shadow_vm_available_at[scenario][0] = 10.0
    seen = []
    def selector(vm, queue):
        seen.append(queue)
        return max(queue, key=lambda task: env.task_mi[task])
    env.vm_queue_selector = selector
    for task in range(3):
        env.route_task_to_vm(task, 0)
    assert env.vm_waiting_queues[0] == [0, 1, 2]
    env.current_time = 10.0
    env._dispatch_vm_queue(0)
    drain(env)
    assert [row["task_id"] for row in env.assignment_history] == [1, 2, 0]
    assert seen == [(0, 1, 2), (0, 2), (0,)]
    assert [(row.start_time, row.end_time) for row in env._records] == [(10, 22), (22, 44), (44, 47)]
    idle = environment()
    idle.vm_queue_selector = lambda *args: pytest.fail("idle task must start directly")
    idle.route_task_to_vm(0, 0)
    assert idle._records[0].start_time == 0.0


def test_shared_physics_and_metrics_match_for_identical_schedule():
    reserved, queued = environment(), environment()
    reserved._assign_task_to_specific_vm(0, 0)
    queued.route_task_to_vm(0, 0)
    drain(reserved)
    drain(queued)
    assert reserved.task_end_time == queued.task_end_time == [14.0]
    assert reserved.get_fuzzy_energy_summary() == queued.get_fuzzy_energy_summary()
    canonical = build_episode_metric_record(queued, seed=1, scheduling_time_seconds=0.0)
    adapter = object.__new__(CEWSEnvAdapter)
    adapter.env, adapter.seed = queued, 1
    adapter.assignment_count, adapter.no_legal_vm_advances = 1, 0
    adapter.instance_fingerprint = lambda: "fixture"
    comparison = episode_metrics(adapter)
    for field in ("deadline_violation_rate", "max_fuzzy_lateness", "mean_fuzzy_lateness",
                  "fuzzy_energy_mean", "fuzzy_energy_std", "fuzzy_energy_score", "evaluation_completed"):
        assert canonical[field] == comparison[field]
    assert aggregate_seed_metrics([comparison])["fuzzy_energy_score"] == aggregate_safe_metric_records([canonical])["fuzzy_energy_score"]
    old = dict(canonical, communication_model_version="input_transfer_then_compute_output_v1")
    with pytest.raises(ValueError, match="different communication models"):
        aggregate_safe_metric_records([canonical, old])


def test_terminal_wait_is_time_since_vm_queue_entry():
    env = environment()
    env.vm_available_at[0] = 20.0
    env.route_task_to_vm(0, 0)
    env.current_time = 10.0
    env._process_finish_events_at_current_time()
    values = dict(zip(TERMINALS, task_terminals(env, 0, 0)))
    assert values["TWT"] == 10.0
    assert values["NIQ"] == 1
    assert values["WIQ"] == 8.0
    assert values["TTIQ"] == 14.0


def test_tree_swapping_crossover_and_original_protected_division():
    first = RulePair(tree("add", "PT", "TWT"), tree("IN_COMM"))
    second = RulePair(tree("sub", "NIQ", "MRT"), tree("OUT_COMM"))
    for seed in range(20):
        children = crossover(first, second, np.random.default_rng(seed))
        swapped_sequencing = children[0].sequencing == second.sequencing and children[1].sequencing == first.sequencing
        swapped_routing = children[0].routing == second.routing and children[1].routing == first.routing
        assert swapped_sequencing or swapped_routing
        for pair in children:
            assert max(program_depth(pair.sequencing), program_depth(pair.routing)) + 1 <= 8
            assert RulePair.from_dict(pair.to_dict()) == pair
    values = [0.0] * len(TERMINALS)
    values[TERMINALS.index("PT")] = 2.0
    assert evaluate_tree(tree("pdiv", "PT", "TWT"), values) == 1.0
    values[TERMINALS.index("TWT")] = 1e-13
    assert evaluate_tree(tree("pdiv", "PT", "TWT"), values) == 2e13
    with pytest.raises(ValueError):
        execution_window(0.0, float("nan"), 0.0, 1.0, 1.0)
