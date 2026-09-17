"""Focused regression tests for the two explicit FCFS baselines."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from algorithms.comparisons.fcfs import train_fcfs
from algorithms.comparisons.fcfs.policies import (
    FCFSFCFSPolicy,
    FCFSFixedPolicy,
    fcfs_task_order,
    select_vm_fcfs,
    select_vm_fixed,
)
from algorithms.comparisons.fcfs.train_fcfs import (
    FORMAL_TEST_SEEDS,
    build_fcfs_protocol,
    default_output_path,
    run_formal_evaluation,
)
from algorithms.comparisons.fuzzy_common.environment import FuzzyBaselineEnv
from algorithms.comparisons.fuzzy_common.evaluation import (
    make_environment,
    run_episode,
)
from base.hrl_env import HrlHeftEnv, HrlFcfsCacheEnv
from common.resource_opt import TriangularFuzzyNumber


def _task_environment():
    environment = SimpleNamespace()
    environment.task_ready_time = {0: 4.0, 1: 4.0, 2: 4.0, 3: 2.0}
    environment.task_meta = [(1, 0), (0, 0), (0, 1), (1, 1)]
    environment.workflows = [
        SimpleNamespace(arrival_time=3.0),
        SimpleNamespace(arrival_time=1.0),
    ]
    return environment


def _minimal_fuzzy_fixed_environment():
    environment = object.__new__(HrlHeftEnv)
    environment.fuzzy_enabled = True
    environment._cur_tid = 0
    environment.task_state = ["Ready"]
    environment.task_meta = [(0, 0)]
    environment.task_baseline_finish = [10.0]
    environment.current_time = 0.0
    environment.vm_ids = [10, 20]
    environment.vms = {
        10: SimpleNamespace(vm_id=10),
        20: SimpleNamespace(vm_id=20),
    }
    environment.vm_available_at = np.asarray([0.0, 0.0])
    environment.global_vm_action_mask = lambda: np.asarray([1.0, 1.0])
    environment.get_feasible_vms = lambda task: [10, 20]
    environment.estimate_exec_time = lambda task, vm: 1.0
    environment.estimate_comm_time = lambda task, vm: 0.0
    environment.estimate_incremental_energy = lambda task, vm: 1.0
    environment.estimate_task_finish_tfn = lambda task, vm: (
        TriangularFuzzyNumber(1.0, 2.0, 3.0)
    )
    environment.fuzzy_deadline_measure = lambda finish: float(finish.upper)
    return environment


def _without_timing(record):
    return {
        key: value
        for key, value in record.items()
        if key != "scheduling_time_seconds"
    }


def test_task_order_is_identical_for_both_policies():
    environment = _task_environment()
    expected = [3, 0, 1, 2]
    assert fcfs_task_order(environment, [2, 0, 3, 1]) == expected

    orderers = []
    environment.set_task_orderer = (
        lambda orderer, training: orderers.append((orderer, training))
    )
    FCFSFCFSPolicy().begin_episode(environment, training=False)
    FCFSFixedPolicy().begin_episode(environment, training=False)
    for orderer, training in orderers:
        assert training is False
        assert orderer((2, 0, 3, 1), np.empty((4, 8)), False) == expected


def test_fcfs_fcfs_vm_selection_uses_only_availability_and_vm_id():
    class Environment:
        vm_available_at = [5.0, 2.0, 2.0, 1.0]
        vm_ids = [1, 9, 4, 0]
        deadline = 1.0
        fuzzy_energy = [0.0, 100.0, 1.0, -1.0]

        @staticmethod
        def global_vm_action_mask():
            return [1.0, 1.0, 1.0, 0.0]

    environment = Environment()
    assert select_vm_fcfs(environment) == 2
    environment.deadline = -1e12
    environment.fuzzy_energy = [1e20, -1e20, 1e30, 0.0]
    assert select_vm_fcfs(environment) == 2


def test_fcfs_fixed_delegates_legal_subset_to_shared_rule():
    calls = []

    class Environment:
        _cur_tid = 7
        vm_ids = [8, 4, 9]
        vm_available_at = [0.0, 0.0, 0.0]

        @staticmethod
        def global_vm_action_mask():
            return [1.0, 1.0, 0.0]

        @staticmethod
        def select_vm_deterministic(task, candidate_vm_ids=None):
            calls.append((task, list(candidate_vm_ids)))
            return 4, {"vm_id": 4}

    assert select_vm_fixed(Environment()) == 1
    assert calls == [(7, [8, 4])]


def test_fcfs_fixed_can_change_with_shared_fuzzy_energy_rule():
    environment = _minimal_fuzzy_fixed_environment()
    energy = {10: 1.0, 20: 5.0}
    environment.estimate_incremental_energy_score = (
        lambda task, vm: energy[int(vm)]
    )
    assert select_vm_fixed(environment) == 0
    energy = {10: 9.0, 20: 2.0}
    assert select_vm_fixed(environment) == 1

    finishes = {
        10: TriangularFuzzyNumber(2.0, 4.0, 20.0),
        20: TriangularFuzzyNumber(2.0, 4.0, 8.0),
    }
    environment.estimate_task_finish_tfn = (
        lambda task, vm: finishes[int(vm)]
    )
    energy = {10: 0.01, 20: 100.0}
    assert select_vm_fixed(environment) == 1


@pytest.mark.parametrize(
    "policy_type", [FCFSFCFSPolicy, FCFSFixedPolicy]
)
def test_same_seed_episode_is_deterministic(policy_type):
    protocol = build_fcfs_protocol("SS", "T", workflows_per_episode=1)
    outcomes = []
    for _ in range(2):
        environment = make_environment(protocol, 201)
        record = run_episode(
            environment,
            policy_type(),
            seed=201,
            training=False,
        )
        outcomes.append(
            (
                _without_timing(record),
                json.dumps(
                    environment.assignment_history,
                    sort_keys=True,
                    allow_nan=False,
                ),
            )
        )
    assert outcomes[0] == outcomes[1]


def test_both_methods_share_environment_and_final_test_seeds():
    protocol = build_fcfs_protocol("SS", "T", workflows_per_episode=1)
    assert protocol.test_seeds == FORMAL_TEST_SEEDS == tuple(range(201, 231))
    assert isinstance(make_environment(protocol, 201), FuzzyBaselineEnv)
    assert issubclass(FuzzyBaselineEnv, HrlFcfsCacheEnv)
    fcfs_path = default_output_path("fcfs_fcfs", protocol)
    fixed_path = default_output_path("fcfs_fixed", protocol)
    assert fcfs_path != fixed_path
    assert "fcfs_fcfs" in fcfs_path.parts
    assert "fcfs_fixed" in fixed_path.parts


@pytest.mark.parametrize(
    ("method_id", "display_name"),
    [("fcfs_fcfs", "FCFS-FCFS"), ("fcfs_fixed", "FCFS-Fixed")],
)
def test_formal_runner_smoke(tmp_path: Path, method_id, display_name):
    destination = tmp_path / method_id / "metrics.json"
    cache_path = (
        Path(__file__).resolve().parents[1]
        / "data" / "deadlines" / "fcfs" / "diagnostic"
        / "fcfs_fixed_SS_exactmix_formal38.json"
    )
    payload = run_formal_evaluation(
        method_id,
        "SS",
        "T",
        output_path=destination,
        workflows_per_episode=1,
        deadline_cache_paths={"SS": str(cache_path)},
    )
    assert payload["method_id"] == method_id
    assert payload["display_name"] == display_name
    assert Path(payload["output_path"]) == destination.resolve()
    assert len(payload["seed_records"]) == 30
    assert {
        int(record["seed"]) for record in payload["seed_records"]
    } == set(range(201, 231))
    restored = json.loads(destination.read_text(encoding="utf-8"))
    assert restored["method_id"] == method_id
    assert restored["display_name"] == display_name
    assert restored["deadline_cache_paths"]["SS"] == str(
        cache_path.resolve()
    )


def test_single_runner_requires_and_records_complete_cache_mapping(
    tmp_path: Path,
    monkeypatch,
):
    cache_paths = {
        scenario: str(tmp_path / f"cache_{scenario}.json")
        for scenario in ("SS", "MS", "LS")
    }
    monkeypatch.setattr(
        train_fcfs,
        "evaluate_policy",
        lambda *_args, **_kwargs: SimpleNamespace(
            aggregate={},
            records=(),
        ),
    )
    payload = train_fcfs.run_single_generalization(
        "fcfs_fcfs",
        "SS",
        output_root=tmp_path / "out",
        deadline_cache_paths=cache_paths,
    )
    assert payload["deadline_cache_paths"] == {
        scenario: str(Path(path).resolve())
        for scenario, path in cache_paths.items()
    }
    with pytest.raises(ValueError, match="MS, LS"):
        train_fcfs.run_single_generalization(
            "fcfs_fixed",
            "SS",
            output_root=tmp_path / "missing",
            deadline_cache_paths={"SS": cache_paths["SS"]},
        )


def test_policy_does_not_modify_task_order_input():
    environment = _task_environment()
    ready = [2, 0, 3, 1]
    before = copy.deepcopy(ready)
    fcfs_task_order(environment, ready)
    assert ready == before
