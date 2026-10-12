"""Rule semantics, shared inputs and executable evaluation of fixed baselines."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from algorithms.comparisons.fcfs.policies import select_vm_fixed
from algorithms.comparisons.fcfs.train_fcfs import (
    build_fcfs_protocol, make_policy, run_formal_evaluation,
)
from algorithms.comparisons.heuristics.policies import POLICY_TYPES, heuristic_task_order
from algorithms.comparisons.fuzzy_common.evaluation import make_environment, run_episode
from algorithms.llm_safe_hrl.hrl_mix.experiment_matrix import build_experiment_manifest


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/deadlines/fcfs/diagnostic/fcfs_fixed_SS_exactmix_formal38.json"


def task_env():
    return SimpleNamespace(
        task_ready_time=[0, 0, 0], task_meta=[(0, 0), (1, 0), (2, 0)],
        workflows=[SimpleNamespace(arrival_time=0, deadline=d) for d in (30, 15, 20)],
        get_feasible_vms=lambda tid: [0, 1],
        estimate_exec_time=lambda tid, vm: [2, 5, 3][tid] + vm,
        calculate_remaining_work=lambda tid: [20, 40, 15][tid],
    )


@pytest.mark.parametrize("method, expected", [
    ("spt_fixed", [0, 2, 1]), ("edd_fixed", [1, 2, 0]),
    ("mwkr_fixed", [1, 0, 2]),
])
def test_rule_semantics_and_stable_ties(method, expected):
    env = task_env()
    ready = [2, 1, 0]
    assert heuristic_task_order(env, ready, method) == expected
    assert ready == [2, 1, 0]
    env.estimate_exec_time = lambda *_: 1
    env.calculate_remaining_work = lambda *_: 1
    for workflow in env.workflows:
        workflow.deadline = 1
    assert heuristic_task_order(env, ready, method) == [0, 1, 2]


@pytest.mark.parametrize("method", ["spt_fixed", "edd_fixed", "mwkr_fixed"])
def test_nonfinite_rule_score_is_rejected(method):
    env = task_env()
    env.estimate_exec_time = lambda *_: float("nan")
    env.calculate_remaining_work = lambda *_: float("inf")
    env.workflows[0].deadline = None
    with pytest.raises(ValueError, match="Invalid"):
        heuristic_task_order(env, [0], method)


@pytest.mark.parametrize("method", POLICY_TYPES)
def test_shared_vm_selector_and_no_training(method):
    policy = make_policy(method)
    assert policy.vm_selector is select_vm_fixed
    with pytest.raises(ValueError, match="evaluation-only"):
        policy.begin_episode(task_env(), training=True)


def test_same_seed_has_identical_inputs_and_no_safety_components():
    protocol = build_fcfs_protocol("SS", "T", workflows_per_episode=1,
                                   deadline_cache_path=str(CACHE))
    snapshots = []
    for method in POLICY_TYPES:
        env = make_environment(protocol, 101)
        env.reset(seed=101)
        make_policy(method).begin_episode(env, training=False)
        snapshots.append((list(env.episode_dax_sequence), list(env.arrival_times),
                          list(env.task_mi), [wf.deadline for wf in env.workflows]))
    assert all(row == snapshots[0] for row in snapshots)
    kwargs = protocol.environment_kwargs(101)
    for key in ("safe_rl_enabled", "safe_rl_shield_enabled", "safe_rl_state_enabled"):
        assert kwargs[key] is False
    assert kwargs["fuzzy_resource_seed"] == 101
    assert kwargs["deadline_cache_strict"] is True
    assert protocol.test_seeds == tuple(range(201, 231))


@pytest.mark.parametrize("method", POLICY_TYPES)
def test_actual_episode_is_deterministic_and_completes(method):
    protocol = build_fcfs_protocol("SS", "T", workflows_per_episode=1,
                                   deadline_cache_path=str(CACHE))
    records = []
    for _ in range(2):
        env = make_environment(protocol, 101)
        record = run_episode(env, make_policy(method), seed=101, training=False)
        record.pop("scheduling_time_seconds")
        assert record["completed_workflow_count"] == 1
        assert np.isfinite(record["fuzzy_energy_score"])
        records.append(record)
    assert records[0] == records[1]


@pytest.mark.parametrize("method", POLICY_TYPES)
def test_formal_runner_records_rule_and_shared_protocol(method, tmp_path):
    result = run_formal_evaluation(method, workflows_per_episode=1,
                                  deadline_cache_paths={"SS": str(CACHE)},
                                  output_path=tmp_path / f"{method}.json")
    assert len(result["seed_records"]) == 30
    assert result["resource_rule"] == "shared_select_vm_deterministic"
    assert result["training_performed"] is False
    assert result["checkpoint_selection_performed"] is False
    assert result["task_order"]


def test_experiment_matrix_registers_all_four_baselines():
    manifest = build_experiment_manifest(
        ROOT / "algorithms/llm_safe_hrl/hrl_mix/config/safe_hrl_experiment_matrix.json",
        smoke=True,
    )
    runs = {row["run_id"]: row for row in manifest["runs"]}
    for method in POLICY_TYPES:
        assert runs[method]["execution_ready"] is True
