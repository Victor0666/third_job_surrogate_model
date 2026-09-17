from __future__ import annotations

import inspect
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from algorithms.comparisons.fuzzy_common.evaluation import (
    make_environment,
    run_episode,
)
from algorithms.comparisons.fuzzy_common.protocol import (
    FuzzyComparisonProtocol,
)
from algorithms.comparisons.fuzzy_common.training import train_baseline
from algorithms.comparisons.irws import IRWSPolicy
from algorithms.comparisons.marl import MARLPolicy
from algorithms.comparisons.pd3qn import PD3QNPolicy
from algorithms.comparisons import run_fuzzy_baseline
from hrl_mix.model_selection import (
    FeasibilityFirstModelMetrics,
    is_better_model,
)


def _protocol(workflows: int = 1) -> FuzzyComparisonProtocol:
    return FuzzyComparisonProtocol(
        scenario="SS",
        ddl_setting="T",
        train_seeds=(1,),
        validation_seeds=(101,),
        test_seeds=(201,),
        workflows_per_episode=workflows,
    )


class FuzzyComparisonProtocolTests(unittest.TestCase):
    def test_protocol_freezes_fuzzy_and_seed_roles(self):
        protocol = _protocol()
        self.assertEqual(protocol.ddl_setting, "Tight")
        self.assertEqual(protocol.ddl_small_probability, 0.8)
        kwargs = protocol.environment_kwargs(1)
        self.assertTrue(kwargs["fuzzy_enabled"])
        self.assertFalse(kwargs["safe_rl_enabled"])
        self.assertFalse(kwargs["safe_rl_shield_enabled"])
        self.assertEqual(kwargs["fuzzy_energy_uncertainty_weight"], 1.0)
        self.assertEqual(kwargs["fuzzy_deadline_eta"], 0.95)
        self.assertEqual(kwargs["fuzzy_resource_seed"], 1)
        with self.assertRaises(ValueError):
            protocol.assert_not_test_seed(201, "training")

    def test_protocol_rejects_seed_leakage(self):
        with self.assertRaises(ValueError):
            FuzzyComparisonProtocol(
                scenario="SS",
                ddl_setting="T",
                train_seeds=(1, 2),
                validation_seeds=(2, 3),
                test_seeds=(4,),
            )

    def test_single_target_scenario_uses_its_own_cache(self):
        cache_root = Path("data/deadlines/fcfs/diagnostic")
        protocol = FuzzyComparisonProtocol(
            scenario="SS",
            ddl_setting="T",
            train_seeds=(1,),
            validation_seeds=(101,),
            test_seeds=(201,),
            deadline_cache_paths={
                "SS": str(cache_root / "fcfs_fixed_SS_exactmix_formal38.json"),
                "MS": str(cache_root / "fcfs_fixed_MS_exactmix_formal38.json"),
            },
        )
        target = protocol.for_scenario("MS")
        env = make_environment(target, 201)
        self.assertTrue(
            str(env.deadline_cache_path).endswith(
                "fcfs_fixed_MS_exactmix_formal38.json"
            )
        )
    def test_checkpoint_selection_is_strictly_feasibility_first(self):
        feasible = FeasibilityFirstModelMetrics(
            deadline_violation_rate=0.0,
            max_fuzzy_lateness=0.0,
            mean_fuzzy_lateness=0.0,
            fuzzy_energy_score=100.0,
            all_seed_feasible=True,
            feasible_seed_rate=1.0,
            worst_seed_violation=0.0,
            worst_seed_lateness=0.0,
            validation_seed_count=1,
        )
        infeasible_low_energy = FeasibilityFirstModelMetrics(
            deadline_violation_rate=0.01,
            max_fuzzy_lateness=1.0,
            mean_fuzzy_lateness=0.1,
            fuzzy_energy_score=1.0,
            all_seed_feasible=False,
            feasible_seed_rate=0.0,
            worst_seed_violation=0.01,
            worst_seed_lateness=1.0,
            validation_seed_count=1,
        )
        self.assertFalse(is_better_model(infeasible_low_energy, feasible))
        self.assertTrue(is_better_model(feasible, infeasible_low_energy))


def test_repeated_deadline_cache_cli_uses_source_path(monkeypatch):
    captured = {}
    monkeypatch.setattr(run_fuzzy_baseline, "make_environment", lambda *a, **k: object())
    monkeypatch.setattr(run_fuzzy_baseline, "_policy", lambda *a, **k: object())
    monkeypatch.setattr(run_fuzzy_baseline, "seed_everything", lambda *a, **k: None)

    def fake_train(*args, **kwargs):
        captured["output_dir"] = kwargs["output_dir"]
        return SimpleNamespace(
            checkpoint_path="checkpoint.pt",
            manifest_path="manifest.json",
            best_validation={},
        )

    monkeypatch.setattr(run_fuzzy_baseline, "train_baseline", fake_train)
    run_fuzzy_baseline.main(
        [
            "--method", "irws",
            "--scenario", "SS",
            "--smoke",
            "--deadline-cache", "SS=cache_SS.json",
            "--deadline-cache", "MS=cache_MS.json",
            "--deadline-cache", "LS=cache_LS.json",
        ]
    )
    assert Path(captured["output_dir"]).name == "cache_SS"


class FuzzyComparisonPolicyTests(unittest.TestCase):
    def test_policy_roles_and_action_spaces(self):
        env = make_environment(_protocol(), 1)
        irws = IRWSPolicy(env)
        marl = MARLPolicy(env)
        pd3qn = PD3QNPolicy(env)
        self.assertEqual(irws.task_ranker.network.network[0].in_features, 8)
        self.assertEqual(marl.host_agent.action_dim, env.num_hosts)
        self.assertEqual(marl.vm_agent.host_count, env.num_hosts)
        self.assertEqual(pd3qn.agent.output_dim, env.num_vms)
        self.assertTrue(pd3qn.agent.use_per)
        self.assertFalse(pd3qn.agent.safe_rl_enabled)

    def test_real_environment_global_vm_assignment_uses_fuzzy_model(self):
        env = make_environment(_protocol(), 1)
        env.reset(seed=1)
        env.set_task_orderer(None, training=False)
        host_state, available = env.get_host_state_for_next_assignment()
        self.assertTrue(available)
        observation, mask = env.global_vm_state()
        self.assertEqual(observation.shape, (env.global_vm_obs_dim,))
        self.assertEqual(mask.shape, (env.num_vms,))
        vm_index = int(np.flatnonzero(mask > 0.5)[0])
        result = env.assign_global_vm(vm_index)
        self.assertTrue(np.isfinite(result.reward))
        self.assertTrue(np.isfinite(result.fuzzy_energy_delta))
        self.assertIn("comparison_reward_components", result.info)
        self.assertFalse(
            result.info["comparison_reward_components"][
                "used_for_checkpoint_selection"
            ]
        )

    def test_task_ordering_is_deterministic_in_evaluation(self):
        env = make_environment(_protocol(), 1)
        policy = IRWSPolicy(env)
        task_ids = (10, 5, 7)
        features = np.asarray(
            [
                [1.0] * 8,
                [2.0] * 8,
                [3.0] * 8,
            ],
            dtype=np.float32,
        )
        first = list(policy._order_tasks(task_ids, features, False))
        second = list(policy._order_tasks(task_ids, features, False))
        self.assertEqual(first, second)
        self.assertEqual(set(first), set(task_ids))

    def test_comparison_packages_do_not_import_offline_llm_or_cma(self):
        modules = (IRWSPolicy, MARLPolicy, PD3QNPolicy)
        forbidden = ("counterfactual_feedback", "rule_optimization", "cma", "LLM.seevo")
        for policy_class in modules:
            source = inspect.getsource(inspect.getmodule(policy_class))
            for token in forbidden:
                self.assertNotIn(token, source)

    def test_small_real_training_produces_isolated_manifest(self):
        protocol = _protocol()
        prototype = make_environment(protocol, 1)
        policy = PD3QNPolicy(prototype)
        with tempfile.TemporaryDirectory() as temp_dir:
            result = train_baseline(
                protocol,
                policy,
                output_dir=temp_dir,
                episodes=1,
                validation_interval=1,
                max_assignment_steps=10000,
            )
            self.assertTrue(Path(result.checkpoint_path).is_file())
            manifest = Path(result.manifest_path).read_text(encoding="utf-8")
            self.assertIn('"final_test_used_for_selection": false', manifest)
            self.assertIn('"method_id": "fuzzy_pd3qn"', manifest)

    def test_full_real_episode_completes(self):
        protocol = _protocol()
        env = make_environment(protocol, 1)
        policy = MARLPolicy(env)
        record = run_episode(
            env,
            policy,
            seed=1,
            training=False,
            max_assignment_steps=10000,
        )
        self.assertTrue(record["evaluation_completed"])
        self.assertEqual(record["completed_workflow_count"], 1)
        self.assertGreater(record["assignment_steps"], 0)


if __name__ == "__main__":
    unittest.main()
