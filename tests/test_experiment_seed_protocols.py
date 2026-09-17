"""Regression tests for the unified comparison and LLM seed protocols."""

from __future__ import annotations

import json
from pathlib import Path

from omegaconf import OmegaConf
import pytest

from algorithms.comparisons.drlea_nichgp.config import (
    build_config,
    ensure_disjoint_seeds,
)
from algorithms.comparisons.fcfs import train_fcfs
from algorithms.comparisons.fuzzy_common.protocol import FuzzyComparisonProtocol
from algorithms.comparisons.fuzzy_common.evaluation import (
    PAPER_FINAL_METRIC_FIELDS,
    aggregate_paper_final_metrics,
)
from algorithms.comparisons.marl.policy import MARLPolicy
from algorithms.comparisons.fcfs.policies import (
    FCFSFCFSPolicy,
    FCFSFixedPolicy,
    fcfs_task_order,
    select_vm_fcfs,
)
from algorithms.comparisons.fuzzy_common.environment import FuzzyBaselineEnv
from algorithms.llm_safe_hrl.LLM.problems.cews_task_constructive.eval import (
    _resolve_seeds,
    load_problem_config,
)
from algorithms.llm_safe_hrl.LLM.seevo import SeEvo
from algorithms.llm_safe_hrl.scenario_registry import (
    WORKLOAD_CATEGORY_REGISTRY,
    deterministic_workload_sequence,
    resolve_experiment_protocol,
    workload_category_counts,
)
from hrl_mix.train_config import build_train_config
from base.hrl_env import HrlFcfsCacheEnv
from algorithms.comparisons.fcfs.generate_deadline_cache import (
    generate_exact_deadline_cache,
)


def test_fcfs_formal_path_uses_shared_fuzzy_environment():
    source = Path(train_fcfs.__file__).read_text(encoding="utf-8")
    assert "env_fcfs" not in source
    assert issubclass(FuzzyBaselineEnv, HrlFcfsCacheEnv)
    protocol = train_fcfs.build_fcfs_protocol("SS", "T")
    assert protocol.test_seeds == tuple(range(201, 231))
    kwargs = protocol.environment_kwargs(201)
    assert kwargs["deadline_mode"] == "cache_fcfs"
    assert kwargs["deadline_cache_strict"] is True
    assert kwargs["fuzzy_enabled"] is True
    assert kwargs["fuzzy_resource_seed"] == 201


def test_fcfs_task_and_resource_ties_are_deterministic():
    class Workflow:
        def __init__(self, arrival_time):
            self.arrival_time = arrival_time

    env = type("TaskEnv", (), {})()
    env.task_ready_time = {8: 4.0, 3: 4.0, 5: 2.0}
    env.task_meta = [(0, 0)] * 9
    env.task_meta[8] = (0, 0)
    env.task_meta[3] = (1, 0)
    env.task_meta[5] = (1, 1)
    env.workflows = [Workflow(2.0), Workflow(1.0)]
    expected = [5, 3, 8]
    assert fcfs_task_order(env, [8, 3, 5]) == expected

    captured = []
    env.set_task_orderer = lambda orderer, training: captured.append(orderer)
    FCFSFCFSPolicy().begin_episode(env, training=False)
    FCFSFixedPolicy().begin_episode(env, training=False)
    assert captured[0]((8, 3, 5), None, False) == expected
    assert captured[1]((8, 3, 5), None, False) == expected

    class AssignmentEnv:
        vm_available_at = [2.0, 2.0, 3.0]
        vm_ids = [9, 4, 1]

        @staticmethod
        def global_vm_action_mask():
            return [1.0, 1.0, 1.0]

    assert select_vm_fcfs(AssignmentEnv()) == 1


def test_drlea_environment_seeds_do_not_depend_on_algorithm_seed():
    first = build_config("SS", "T", 7, smoke=True)
    second = build_config("SS", "T", 99, smoke=True)
    expected = (
        (1, 2, 3, 4, 5),
        (101, 102, 103),
        tuple(range(201, 231))
    )
    assert (
        first.train_seeds,
        first.validation_seeds,
        first.test_seeds,
    ) == expected
    assert (
        second.train_seeds,
        second.validation_seeds,
        second.test_seeds,
    ) == expected
    assert (first.algorithm_seed, second.algorithm_seed) == (7, 99)


def test_llm_safe_default_split_and_direct_evaluator_guard():
    config = load_problem_config()
    assert config["dataset"]["train_seeds"] == [1, 2, 3]
    assert config["dataset"]["validation_seeds"] == [4, 5]
    assert _resolve_seeds(config, "train", None) == [1, 2, 3]
    with pytest.raises(ValueError, match="reserved comparison"):
        _resolve_seeds(config, "train", [101])
    with pytest.raises(ValueError, match="reserved comparison"):
        _resolve_seeds(config, "train", [203])


def test_seevo_rejects_reserved_seed_override_before_cma():
    algorithm = object.__new__(SeEvo)
    algorithm.cfg = OmegaConf.create(
        {
            "problem": {
                "dataset": {
                    "train_seeds": [1, 2, 3],
                    "validation_seeds": [4, 5],
                    "test_seeds": [100],
                }
            }
        }
    )
    algorithm.case_num = [201]
    with pytest.raises(ValueError, match="reserved comparison"):
        algorithm._optimization_seed_sets()


def test_formal_seed_contracts_cannot_be_overridden():
    context = resolve_experiment_protocol("single", source_scenario="SS")
    assert context.llm_train_seeds == (1, 2, 3)
    assert context.llm_validation_seeds == (4, 5)
    assert context.safe_hrl_train_seeds == (1, 2, 3, 4, 5)
    assert context.safe_hrl_validation_seeds == (101, 102, 103)
    assert context.final_test_seeds == tuple(range(201, 231))
    with pytest.raises(ValueError, match="formal LLM offline seeds"):
        resolve_experiment_protocol(
            "single",
            source_scenario="SS",
            train_seeds=(0, 1, 2),
            validation_seeds=(3, 4),
        )


def test_formal_rl_lengths_validation_cadence_and_curriculum_ablation():
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr("hrl_mix.train_config.os.makedirs", lambda *args, **kwargs: None)
        full = build_train_config(protocol="single", source_scenario="SS")
        ablation = build_train_config(
            protocol="single",
            source_scenario="SS",
            safe_rl_curriculum_enabled=False,
        )
    for config in (full, ablation):
        assert config.max_episodes == 600
        assert config.validation_interval == 25
        assert config.train_seeds == (1, 2, 3, 4, 5)
        assert config.validation_seeds == (101, 102, 103)
        assert config.final_test_seeds == tuple(range(201, 231))
    assert full.curriculum_enabled is True
    assert ablation.curriculum_enabled is False


def test_safe_hrl_curriculum_and_algorithm_seed_use_distinct_artifact_names():
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr("hrl_mix.train_config.os.makedirs", lambda *args, **kwargs: None)
        full = build_train_config(
            protocol="single",
            source_scenario="SS",
            safe_rl_enabled=True,
            optimizer_seed=7,
        )
        ablation = build_train_config(
            protocol="single",
            source_scenario="SS",
            safe_rl_enabled=True,
            safe_rl_curriculum_enabled=False,
            optimizer_seed=7,
        )
        other_algorithm_seed = build_train_config(
            protocol="single",
            source_scenario="SS",
            safe_rl_enabled=True,
            optimizer_seed=8,
        )
    assert len({full.run_name, ablation.run_name, other_algorithm_seed.run_name}) == 3


def test_drlea_formal_defaults_are_single_300_plus_300_and_25():
    config = build_config("SM", "T", 13)
    assert config.protocol == "single"
    assert config.training_scenarios == ("SM",)
    assert config.test_scenarios == ("SM", "MM", "LM")
    assert (config.ra_episodes, config.sa_episodes) == (300, 300)
    assert config.validation_interval == 25


def test_safe_hrl_best_checkpoint_uses_fixed_formal_source_validation():
    source = Path("algorithms/llm_safe_hrl/hrl_mix/train_runner.py").read_text(
        encoding="utf-8"
    )
    formal_call = (
        "scenarios=(str(cfg.source_scenario or "
        "cfg.training_scenarios[0]).upper(),)"
    )
    assert formal_call in source
    assert "controller=None" in source
    assert 'source="curriculum_validation"' in source


def test_shared_paper_final_metric_view_has_exact_six_fields():
    metrics = aggregate_paper_final_metrics(
        [
            {
                "workflow_count": 4,
                "deadline_violation_count": 0,
                "deadline_violation_rate": 0.0,
                "max_fuzzy_lateness": 0.0,
                "mean_fuzzy_lateness": 0.0,
                "fuzzy_energy_mean": 10.0,
                "fuzzy_energy_std": 2.0,
                "fuzzy_energy_score": 12.0,
            },
            {
                "workflow_count": 6,
                "deadline_violation_count": 1,
                "deadline_violation_rate": 1.0 / 6.0,
                "max_fuzzy_lateness": 3.0,
                "mean_fuzzy_lateness": 0.5,
                "fuzzy_energy_mean": 14.0,
                "fuzzy_energy_std": 4.0,
                "fuzzy_energy_score": 18.0,
            },
        ]
    )
    assert tuple(metrics) == PAPER_FINAL_METRIC_FIELDS
    assert metrics["deadline_violation_rate"] == pytest.approx(0.1)
    assert metrics["mean_fuzzy_lateness"] == pytest.approx(0.3)
    assert metrics["fuzzy_energy_mean"] == pytest.approx(12.0)

def test_learning_baseline_seeds_before_policy_network_creation():
    source = Path("algorithms/comparisons/run_fuzzy_baseline.py").read_text(
        encoding="utf-8"
    )
    assert source.index("seed_everything(int(args.optimizer_seed))") < source.index(
        "policy = _policy("
    )


def test_global_paper_final_seed_reservation_is_fail_closed():
    with pytest.raises(ValueError, match="201-230"):
        FuzzyComparisonProtocol(
            scenario="SS",
            ddl_setting="T",
            train_seeds=(202,),
            validation_seeds=(101,),
            test_seeds=(201,),
        )
    with pytest.raises(ValueError, match="201-230"):
        ensure_disjoint_seeds((1,), (229,), (201,))


def test_explicit_workload_categories_and_ratios_are_registry_defined():
    category_30 = set(WORKLOAD_CATEGORY_REGISTRY["30"])
    assert {
        "CyberShake_30.xml",
        "Epigenomics_24.xml",
        "Ligo_30.xml",
        "Montage_25.xml",
        "Sipht_29.xml",
    } <= category_30
    assert workload_category_counts("S", 50) == {"30": 50}
    assert workload_category_counts("M", 50) == {"30": 15, "50": 35}
    assert workload_category_counts("L", 50) == {
        "30": 10,
        "50": 15,
        "100": 25,
    }
    first = deterministic_workload_sequence("M", 17, 50)
    assert first == deterministic_workload_sequence("M", 17, 50)
    assert first != deterministic_workload_sequence("M", 18, 50)
    category_by_dax = {
        dax: category
        for category, dax_names in WORKLOAD_CATEGORY_REGISTRY.items()
        for dax in dax_names
    }
    for task_scale, expected in (
        ("S", {"30": 50}),
        ("M", {"30": 15, "50": 35}),
        ("L", {"30": 10, "50": 15, "100": 25}),
    ):
        actual = {}
        for dax_name in deterministic_workload_sequence(task_scale, 17, 50):
            category = category_by_dax[dax_name]
            actual[category] = actual.get(category, 0) + 1
        assert actual == expected


def test_exact_mix_cache_generator_uses_real_fcfs_environment(tmp_path):
    destination = generate_exact_deadline_cache(
        "MS",
        seeds=(1,),
        workflows_per_episode=2,
        output_path=tmp_path / "cache.json",
    )
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["meta"]["schema_version"] == "exact_workload_mix_v1"
    assert payload["meta"]["category_counts"] == {"30": 1, "50": 1}
    assert payload["meta"]["environment_seeds"] == [1]
    assert len(payload["data"][0]["wf_dax_names"]) == 2
    assert len(payload["data"][0]["wf_makespans"]) == 2


def test_strict_deadline_cache_rejects_dax_mismatch():
    env = object.__new__(HrlFcfsCacheEnv)
    env.deadline_cache_strict = True
    env._deadline_cache = {
        1: {"wf_makespans": [12.0], "wf_dax_names": ["Ligo_50.xml"]}
    }
    with pytest.raises(ValueError, match="regenerate the FCFS cache"):
        env._cache_lookup_makespan(1, 0, "Montage_25.xml")


def test_marl_vm_bootstrap_consumes_only_same_host_pending_transition():
    calls = []
    policy = object.__new__(MARLPolicy)
    policy.vm_agent = type(
        "Agent",
        (),
        {"update_transition": lambda self, *args: calls.append(args)},
    )()
    policy._pending_vm_by_host = {
        0: {
            "observation": [0.0],
            "mask": [1.0],
            "action": 0,
            "reward": 1.0,
        },
        1: {
            "observation": [1.0],
            "mask": [1.0],
            "action": 0,
            "reward": 2.0,
        },
    }
    policy._update_pending_vm(1, [2.0], done=0.0)
    assert set(policy._pending_vm_by_host) == {0}
    assert len(calls) == 1
    assert calls[0][-1] == 1
