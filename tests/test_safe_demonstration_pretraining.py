"""Stage 13 safe demonstration dataset and offline pretraining tests."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from algorithms.llm_safe_hrl.paths import LLM_ROOT
from base.hrl_env import HrlFcfsCacheEnv
from base.safe_demonstration import (
    SAFE_DEMONSTRATION_GENERATOR_POLICY,
    DemonstrationSafetyStandard,
    SafeDemonstrationEpisode,
    StrictSeedSplit,
    append_demonstration_episode,
    load_demonstration_split,
)
from base.safe_replay import SafeReplayTransition
from hrl_mix.train_config import (
    OfflinePretrainingConfig,
    SafeRLConfig,
)

try:
    import torch

    from base.d3qn_agent import D3QNAgent
    from base.offline_pretraining import (
        OfflinePretrainingOptions,
        pretrain_agents_from_demonstrations,
    )
    from hrl_mix.safe_demonstrations import (
        DemonstrationGenerationOptions,
        generate_safe_demonstration_episode,
    )

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None
    D3QNAgent = None
    OfflinePretrainingOptions = None
    pretrain_agents_from_demonstrations = None
    DemonstrationGenerationOptions = None
    generate_safe_demonstration_episode = None
    TORCH_AVAILABLE = False


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIBRARY = (
    LLM_ROOT
    / "problems"
    / "cews_task_constructive"
    / "safe_heuristic_library_resS.json"
)

LAYER_DIMS = {
    "manager": (3, 2),
    "host": (4, 2),
    "vm": (5, 2),
}
SCHEMAS = {
    "manager": "test_manager_schema_v1",
    "host": "test_host_schema_v1",
    "vm": "test_vm_schema_v1",
}


def _transition(input_dim, action_dim, *, cost=0.0):
    final_mask = np.ones(action_dim, dtype=np.float32)
    return SafeReplayTransition(
        state=np.linspace(
            0.1, 0.9, input_dim, dtype=np.float32
        ),
        proposed_action=0,
        executed_action=0,
        performance_reward=-1.0,
        safety_cost=float(cost),
        next_state=np.zeros(input_dim, dtype=np.float32),
        done=1.0,
        legal_action_mask=final_mask,
        safety_action_mask=final_mask,
        final_action_mask=final_mask,
        next_final_action_mask=np.zeros(
            action_dim, dtype=np.float32
        ),
        shield_modified=False,
        fallback_triggered=False,
        fuzzy_safety_margin=10.0,
        predicted_risk_finish=1.0,
        violation_flag=bool(cost > 0.0),
        manager_phase_id=0,
        action_source="demonstration_fixed_vm_rule",
        policy_selection_type=(
            "demonstration_fixed_vm_rule"
        ),
    )


def _episode(
    episode_id,
    split,
    workflow_seed,
    resource_seed,
    *,
    safe=True,
):
    metrics = {
        "workflow_count": 1,
        "completed_workflow_count": 1,
        "deadline_violation_count": 0 if safe else 1,
        "deadline_violation_rate": 0.0 if safe else 1.0,
        "max_fuzzy_lateness": 0.0 if safe else 2.0,
        "constraint_feasible": bool(safe),
        "all_workflows_completed": True,
        "fuzzy_energy_mean": 10.0,
        "fuzzy_energy_std": 2.0,
        "fuzzy_energy_score": 12.0,
    }
    return SafeDemonstrationEpisode(
        episode_id=episode_id,
        split=split,
        generator_policy=(
            SAFE_DEMONSTRATION_GENERATOR_POLICY
        ),
        heuristic_id="traditional_edf",
        heuristic_source="traditional",
        heuristic_version="builtin_v1",
        workflow_seed=workflow_seed,
        resource_seed=resource_seed,
        ddl_setting={"name": "test"},
        fuzzy_parameters={
            "deadline_eta": 0.95,
            "energy_uncertainty_weight": 1.0,
            "delta1": 0.75,
            "delta2": 1.2,
        },
        observation_schema_versions=SCHEMAS,
        episode_metrics=metrics,
        trajectories={
            layer: [
                _transition(*LAYER_DIMS[layer])
            ]
            for layer in LAYER_DIMS
        },
        safety_standard=DemonstrationSafetyStandard(),
    )


def _seed_split():
    return StrictSeedSplit(
        train_workflow_seeds=(1,),
        validation_workflow_seeds=(2,),
        final_test_workflow_seeds=(3,),
        train_resource_seeds=(11,),
        validation_resource_seeds=(12,),
        final_test_resource_seeds=(13,),
    )


def _agent(layer):
    input_dim, action_dim = LAYER_DIMS[layer]
    return D3QNAgent(
        input_dim=input_dim,
        output_dim=action_dim,
        hidden_dims=(16,),
        head_hidden_dims=(8,),
        device="cpu",
        batch_size=1,
        buffer_size=16,
        observation_schema_version=SCHEMAS[layer],
        safe_rl_enabled=True,
    )


def _parameters(module):
    return {
        name: value.detach().clone()
        for name, value in module.state_dict().items()
    }


def _changed(before, module):
    after = module.state_dict()
    return any(
        not torch.equal(before[name], after[name])
        for name in before
    )


class DemonstrationDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(
            dir=PROJECT_ROOT
        )
        self.manifest_path = (
            Path(self.temporary.name) / "manifest.json"
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_manifest_records_required_fields_and_safe_label(self):
        episode = _episode("train.safe", "train", 1, 11)
        manifest = append_demonstration_episode(
            self.manifest_path,
            episode,
            seed_split=_seed_split(),
        )
        record = manifest["episodes"][0]
        for field in (
            "generator_policy",
            "heuristic_id",
            "resource_seed",
            "workflow_seed",
            "ddl_setting",
            "fuzzy_parameters",
            "feasibility",
            "fuzzy_energy_score",
            "trajectory_count",
        ):
            self.assertIn(field, record)
        self.assertTrue(
            record["feasibility"]["safe_demonstration"]
        )
        self.assertEqual(record["trajectory_count"]["total"], 3)

    def test_unsafe_episode_is_not_a_safe_demonstration(self):
        unsafe = _episode(
            "train.unsafe", "train", 1, 11, safe=False
        )
        self.assertFalse(unsafe.safe_demonstration)
        manifest = append_demonstration_episode(
            self.manifest_path,
            unsafe,
            seed_split=_seed_split(),
        )
        self.assertEqual(manifest["safe_episode_count"], 0)
        with self.assertRaisesRegex(
            ValueError, "no eligible train"
        ):
            load_demonstration_split(
                self.manifest_path,
                "train",
                require_safe=True,
            )

    def test_strict_split_rejects_cross_role_seed_leakage(self):
        with self.assertRaisesRegex(ValueError, "seed leakage"):
            StrictSeedSplit(
                train_workflow_seeds=(1,),
                validation_workflow_seeds=(2,),
                final_test_workflow_seeds=(3,),
                train_resource_seeds=(11,),
                validation_resource_seeds=(1,),
                final_test_resource_seeds=(13,),
            )

    def test_final_test_split_cannot_be_loaded_for_pretraining(self):
        with self.assertRaisesRegex(
            ValueError, "only train or validation"
        ):
            load_demonstration_split(
                self.manifest_path,
                "final_test",
            )

    def test_episode_hash_tamper_is_rejected(self):
        append_demonstration_episode(
            self.manifest_path,
            _episode("train.safe", "train", 1, 11),
            seed_split=_seed_split(),
        )
        manifest = json.loads(
            self.manifest_path.read_text(encoding="utf-8")
        )
        episode_path = (
            self.manifest_path.parent
            / manifest["episodes"][0]["episode_file"]
        )
        episode_path.write_text(
            episode_path.read_text(encoding="utf-8") + " ",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            ValueError, "episode file hash mismatch"
        ):
            load_demonstration_split(
                self.manifest_path, "train"
            )


class OfflinePretrainingConfigTests(unittest.TestCase):
    def test_default_config_does_not_change_legacy_semantics(self):
        config = SafeRLConfig()
        self.assertFalse(config.enabled)
        self.assertFalse(config.offline_pretraining.enabled)
        self.assertIsNone(
            config.offline_pretraining.dataset_manifest_path
        )

    def test_enabled_pretraining_requires_versioned_manifest(self):
        with self.assertRaisesRegex(
            ValueError, "requires a dataset manifest"
        ):
            OfflinePretrainingConfig(enabled=True)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class OfflinePretrainingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(
            dir=PROJECT_ROOT
        )
        self.manifest_path = (
            Path(self.temporary.name) / "manifest.json"
        )
        split = _seed_split()
        append_demonstration_episode(
            self.manifest_path,
            _episode("train.safe", "train", 1, 11),
            seed_split=split,
        )
        append_demonstration_episode(
            self.manifest_path,
            _episode(
                "validation.safe", "validation", 2, 12
            ),
            seed_split=split,
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_qr_qc_pretraining_preserves_online_update_state(self):
        agents = {
            layer: _agent(layer) for layer in LAYER_DIMS
        }
        before = {
            layer: {
                "q_r": _parameters(agent.q_r_online),
                "q_c": _parameters(agent.q_c_online),
                "q_r_target": _parameters(agent.q_r_target),
                "q_c_target": _parameters(agent.q_c_target),
                "updates": agent._updates,
                "epsilon_steps": agent._eps_steps,
                "replay_size": len(agent.buffer),
                "target_tau": agent.target_update_tau,
                "target_freq": agent.target_update_freq,
            }
            for layer, agent in agents.items()
        }
        report = pretrain_agents_from_demonstrations(
            agents,
            OfflinePretrainingOptions(
                enabled=True,
                dataset_manifest_path=str(
                    self.manifest_path
                ),
                epochs=1,
                batch_size=1,
                performance_learning_rate=1e-3,
                safety_learning_rate=1e-3,
                train_q_r=True,
                train_q_c=True,
                behavior_cloning_enabled=True,
                behavior_cloning_weight=0.1,
                sync_targets_after_pretraining=False,
                random_seed=5,
            ),
        )
        self.assertTrue(report["enabled"])
        for layer, agent in agents.items():
            self.assertTrue(
                _changed(before[layer]["q_r"], agent.q_r_online)
            )
            self.assertTrue(
                _changed(before[layer]["q_c"], agent.q_c_online)
            )
            self.assertFalse(
                _changed(
                    before[layer]["q_r_target"],
                    agent.q_r_target,
                )
            )
            self.assertFalse(
                _changed(
                    before[layer]["q_c_target"],
                    agent.q_c_target,
                )
            )
            self.assertEqual(
                agent._updates, before[layer]["updates"]
            )
            self.assertEqual(
                agent._eps_steps,
                before[layer]["epsilon_steps"],
            )
            self.assertEqual(
                len(agent.buffer),
                before[layer]["replay_size"],
            )
            self.assertEqual(
                agent.target_update_tau,
                before[layer]["target_tau"],
            )
            self.assertEqual(
                agent.target_update_freq,
                before[layer]["target_freq"],
            )


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class RealGenerationSmokeTests(unittest.TestCase):
    def test_real_environment_generates_replay_compatible_episode(self):
        dax = PROJECT_ROOT / "data" / "dax" / "Montage_25.xml"
        environment = HrlFcfsCacheEnv(
            dax_paths=[str(dax)],
            horizon=1e6,
            arrival_lambda=0.03,
            random_seed=31,
            max_ready_tasks="auto",
            normalize=True,
            workflows_per_episode=1,
            num_cloud_hosts=1,
            num_edge_hosts=1,
            cloud_vms_per_host=(2,),
            edge_vms_per_host=(2,),
            cloud_pc_tiers=(1.0, 2.0, 4.0, 6.0, 8.0),
            edge_pc_tiers=(1.0, 2.0, 4.0, 6.0, 8.0),
            cloud_bw_tiers=(
                1000.0,
                2000.0,
                4000.0,
                6000.0,
                8000.0,
            ),
            edge_bw_tiers=(
                1000.0,
                2000.0,
                4000.0,
                6000.0,
                8000.0,
            ),
            deadline_mode="none",
            manager_alpha_delay=0.75,
            manager_delay_mode="tardiness",
            safe_rl_enabled=True,
            safe_rl_shield_enabled=True,
            safe_rl_state_enabled=True,
            manager_mode="heuristic_selection_mode",
            manager_heuristic_library_path=str(
                DEFAULT_LIBRARY
            ),
            fuzzy_enabled=True,
            fuzzy_deadline_eta=0.95,
            fuzzy_energy_uncertainty_weight=1.0,
            fuzzy_resource_seed=31,
            fuzzy_use_deadline_constraint=True,
        )
        episode = generate_safe_demonstration_episode(
            environment,
            DemonstrationGenerationOptions(
                heuristic_id="traditional_edf",
                workflow_seed=31,
                resource_seed=31,
                split="train",
                max_manager_phases=1000,
            ),
        )
        self.assertTrue(episode.safe_demonstration)
        self.assertEqual(
            episode.generator_policy,
            SAFE_DEMONSTRATION_GENERATOR_POLICY,
        )
        for layer in ("manager", "host", "vm"):
            self.assertTrue(episode.trajectories[layer])
            for transition in episode.trajectories[layer]:
                self.assertIsInstance(
                    transition, SafeReplayTransition
                )
        resource_sources = {
            row.action_source
            for layer in ("host", "vm")
            for row in episode.trajectories[layer]
        }
        self.assertTrue(
            resource_sources
            <= {
                "demonstration_fixed_vm_rule",
                "fallback_action",
            }
        )
        self.assertEqual(episode.heuristic_source, "traditional")
        self.assertAlmostEqual(
            episode.fuzzy_parameters["deadline_eta"], 0.95
        )
        self.assertAlmostEqual(
            episode.fuzzy_parameters[
                "energy_uncertainty_weight"
            ],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
