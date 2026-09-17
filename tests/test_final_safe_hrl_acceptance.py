"""Small end-to-end acceptance tests for legacy and safe HRL modes."""

from __future__ import annotations

import contextlib
import csv
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from algorithms.llm_safe_hrl.paths import LLM_ROOT
from base.d3qn_agent import D3QNAgent
from hrl_mix.train_config import build_train_config
import hrl_mix.train_runner as train_runner
from LLM.problems.cews_task_constructive.eval import (
    evaluate_candidate,
    load_problem_config,
)


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DAX = ROOT / "data" / "dax" / "Montage_25.xml"


class FinalSafeHRLAcceptanceTests(unittest.TestCase):
    def test_small_cews_reference_evaluation_uses_frozen_formulas(self):
        problem_config = load_problem_config(
            LLM_ROOT
            / "cfg"
            / "problem"
            / "cews_task_constructive.yaml"
        )
        problem_config["dataset"]["workflows_per_instance"] = 1
        result = evaluate_candidate(
            (
                LLM_ROOT
                / "problems"
                / "cews_task_constructive"
                / "reference.py"
            ),
            problem_config,
            [1],
        )
        self.assertTrue(result["interface_valid"])
        self.assertTrue(result["all_evaluation_seeds_completed"])
        self.assertEqual(result["completed_seed_count"], 1)
        self.assertAlmostEqual(
            result["objective"],
            result["fuzzy_total_energy_mean"]
            + result["fuzzy_total_energy_std"],
        )
        self.assertEqual(
            problem_config["fuzzy"]["deadline_eta"],
            0.95,
        )
        self.assertEqual(
            problem_config["fuzzy"][
                "energy_uncertainty_weight"
            ],
            1.0,
        )

    def test_llm_entry_does_not_embed_a_qwen_api_key(self):
        source = (LLM_ROOT / "main.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("QWEN_API_KEY", source)

    @staticmethod
    def _small_agent_config(config):
        return replace(
            config,
            batch_size=2,
            buffer_size=64,
            eps_start=0.2,
            eps_end=0.2,
            eps_decay_steps=10,
            hidden_dims=(32, 16),
        )

    def _small_config(
        self,
        output_root: Path,
        *,
        safe_rl_enabled: bool,
    ):
        build_kwargs = {
            "safe_rl_enabled": safe_rl_enabled,
            "safe_rl_shield_enabled": safe_rl_enabled,
            "safe_rl_state_enabled": safe_rl_enabled,
            "safe_rl_dynamic_lambda_enabled": safe_rl_enabled,
            "safe_rl_heuristic_manager_enabled": safe_rl_enabled,
        }
        # build_train_config normally creates its production output
        # directories. This test substitutes temporary paths immediately,
        # so suppress those two unrelated directory writes.
        with patch("hrl_mix.train_config.os.makedirs"):
            base = build_train_config(
                "SS",
                "T",
                1,
                **build_kwargs,
            )
        return replace(
            base,
            dax_list=[str(REFERENCE_DAX)],
            task_code="",
            workflows_per_episode=1,
            max_episodes=1,
            save_interval=10**9,
            eval_seeds=(1,),
            warmup_frac=0.0,
            hard_max_steps=100_000,
            save_dir=str(output_root / "checkpoints"),
            log_path=str(output_root / "logs" / "train.csv"),
            vm_agent=self._small_agent_config(base.vm_agent),
            host_agent=self._small_agent_config(base.host_agent),
            manager_agent=self._small_agent_config(
                base.manager_agent
            ),
        )

    @staticmethod
    def _restore_checkpoint(path: Path) -> D3QNAgent:
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
        restored = D3QNAgent(
            input_dim=int(payload["input_dim"]),
            output_dim=int(payload["output_dim"]),
            hidden_dims=tuple(payload["hidden_dims"]),
            head_hidden_dims=(
                tuple(payload["head_hidden_dims"])
                if payload["head_hidden_dims"] is not None
                else None
            ),
            device="cpu",
            observation_schema_version=str(
                payload["observation_schema_version"]
            ),
            safe_rl_enabled=bool(payload["safe_rl_enabled"]),
            safety_discount=float(payload["safety_discount"]),
            safety_learning_rate=float(
                payload["safety_learning_rate"]
            ),
            safety_loss_weight=float(
                payload["safety_loss_weight"]
            ),
            initial_lagrange_multiplier=float(
                payload["lagrange_multiplier"]
            ),
            use_per=bool(payload["use_per"]),
            per_alpha=float(payload["per_alpha"]),
            per_beta_start=float(payload["per_beta_start"]),
            per_beta_end=float(payload["per_beta_end"]),
            per_beta_steps=int(payload["per_beta_steps"]),
            safe_replay_near_boundary_margin=float(
                payload["safe_replay_near_boundary_margin"]
            ),
            safe_per_combined_priority=bool(
                payload["safe_per_combined_priority"]
            ),
            safe_per_performance_td_weight=float(
                payload["safe_per_performance_td_weight"]
            ),
            safe_per_safety_td_weight=float(
                payload["safe_per_safety_td_weight"]
            ),
        )
        restored.load(str(path))
        return restored

    def _run_one_episode(self, *, safe_rl_enabled: bool):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        output_root = Path(temporary.name)
        config = self._small_config(
            output_root,
            safe_rl_enabled=safe_rl_enabled,
        )
        Path(config.save_dir).mkdir(parents=True, exist_ok=True)
        Path(config.log_path).parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        kwargs = {
            "safe_rl_enabled": safe_rl_enabled,
            "safe_rl_shield_enabled": safe_rl_enabled,
            "safe_rl_state_enabled": safe_rl_enabled,
            "safe_rl_dynamic_lambda_enabled": safe_rl_enabled,
            "safe_rl_heuristic_manager_enabled": safe_rl_enabled,
        }
        stdout = io.StringIO()
        with (
            patch.object(
                train_runner,
                "build_train_config",
                return_value=config,
            ),
            contextlib.redirect_stdout(stdout),
        ):
            train_runner.train("SS", "T", 1, **kwargs)
        return output_root, stdout.getvalue()

    def test_original_hrl_one_episode_and_checkpoint_restore(self):
        output_root, output = self._run_one_episode(
            safe_rl_enabled=False
        )
        self.assertIn("Training finished.", output)
        self.assertNotIn("[safe Manager stage11] mode=heuristic_selection_mode", output)

        with (output_root / "logs" / "train.csv").open(
            "r",
            encoding="utf-8",
            newline="",
        ) as handle:
            rows = list(csv.DictReader(handle))
        self.assertTrue(any(row["type"] == "episode" for row in rows))
        self.assertNotIn("safety_cost", rows[0])

        for layer in ("manager", "host", "vm"):
            checkpoint = (
                output_root
                / "checkpoints"
                / f"{layer}_final.pth"
            )
            self.assertTrue(checkpoint.is_file())
            restored = self._restore_checkpoint(checkpoint)
            self.assertFalse(restored.safe_rl_enabled)
            self.assertIsNone(restored.q_c_online)

    def test_safe_hrl_one_episode_manifest_and_checkpoint_restore(self):
        output_root, output = self._run_one_episode(
            safe_rl_enabled=True
        )
        self.assertIn("Training finished.", output)
        self.assertIn(
            "[safe Manager stage11] mode=heuristic_selection_mode",
            output,
        )
        checkpoint_dir = output_root / "checkpoints"
        manifest = json.loads(
            (
                checkpoint_dir
                / "best_checkpoint_manifest.json"
            ).read_text(encoding="utf-8")
        )
        metrics = manifest["model_selection_metrics"]
        for key in (
            "all_seed_feasible",
            "feasible_seed_rate",
            "worst_seed_violation",
            "worst_seed_lateness",
        ):
            self.assertIn(key, metrics)
        self.assertEqual(
            manifest["selection_policy"],
            "feasibility_first_lexicographic",
        )
        self.assertIn("replay_metadata", manifest)
        self.assertIn("config_snapshot", manifest)
        self.assertIn("heuristic_library_version", manifest)

        with (output_root / "logs" / "train.csv").open(
            "r",
            encoding="utf-8",
            newline="",
        ) as handle:
            rows = list(csv.DictReader(handle))
        episode_rows = [
            row for row in rows if row["type"] == "episode"
        ]
        self.assertTrue(episode_rows)
        for key in (
            "total_performance_reward",
            "safety_cost",
            "shield_intervention_count",
            "fallback_action_count",
            "current_lambda",
        ):
            self.assertIn(key, episode_rows[0])

        for split in ("training", "validation"):
            for suffix in ("csv", "jsonl"):
                self.assertTrue(
                    (
                        checkpoint_dir
                        / "safe_metrics"
                        / f"{split}_metrics.{suffix}"
                    ).is_file()
                )

        for layer in ("manager", "host", "vm"):
            checkpoint = checkpoint_dir / f"{layer}_final.pth"
            restored = self._restore_checkpoint(checkpoint)
            self.assertTrue(restored.safe_rl_enabled)
            self.assertIsNotNone(restored.q_c_online)
            self.assertIsNotNone(restored.q_c_target)
            self.assertIsNotNone(restored.q_c_optim)
            self.assertIsInstance(
                restored.lagrange_controller_state,
                dict,
            )
            self.assertEqual(
                restored.lagrange_multiplier,
                restored.lagrange_controller_state[
                    "current_lambda"
                ],
            )


if __name__ == "__main__":
    unittest.main()
