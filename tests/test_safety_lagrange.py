"""阶段 8 共享 episode/EMA 动态拉格朗日控制器测试。"""

from __future__ import annotations

from pathlib import Path
import math
import tempfile
import unittest
from unittest.mock import patch

from base.safety_lagrange import (
    LagrangeSafetyController,
    synchronize_lagrange_multiplier,
)
from hrl_mix.train_config import build_train_config


try:
    import torch

    from base.d3qn_agent import D3QNAgent
    from hrl_mix.train_eval import (
        FINAL_EVALUATION_VIOLATION_BUDGET,
        evaluate_hrl_three_layer_multi_seed,
    )
    from hrl_mix.train_runner import (
        _save_agent_checkpoint,
        _update_shared_lagrange_at_episode_end,
    )

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None
    D3QNAgent = None
    TORCH_AVAILABLE = False


class LagrangeSafetyControllerTests(unittest.TestCase):
    @staticmethod
    def _controller(**overrides):
        options = {
            "enabled": True,
            "lambda_init": 1.0,
            "lambda_lr": 0.5,
            "lambda_min": 0.0,
            "lambda_max": 10.0,
            "cost_budget": 0.0,
            "update_interval": 1,
            "cost_ema_factor": 0.0,
            "warmup_steps": 0,
        }
        options.update(overrides)
        return LagrangeSafetyController(**options)

    def test_cost_above_budget_increases_lambda(self):
        controller = self._controller()
        result = controller.observe_episode(
            episode_safety_cost=4.0,
            safety_transition_count=2,
        )
        self.assertAlmostEqual(result["mean_safety_cost"], 2.0)
        self.assertAlmostEqual(result["constraint_gap"], 2.0)
        self.assertAlmostEqual(result["current_lambda"], 2.0)
        self.assertEqual(result["lambda_update_count"], 1)

    def test_cost_below_budget_decreases_lambda(self):
        controller = self._controller(
            lambda_init=2.0,
            cost_budget=2.0,
        )
        result = controller.observe_episode(
            episode_safety_cost=2.0,
            safety_transition_count=2,
        )
        self.assertAlmostEqual(result["mean_safety_cost"], 1.0)
        self.assertAlmostEqual(result["constraint_gap"], -1.0)
        self.assertAlmostEqual(result["current_lambda"], 1.5)

    def test_lambda_respects_both_bounds(self):
        upper = self._controller(
            lambda_init=1.0,
            lambda_lr=100.0,
            lambda_max=3.0,
        )
        upper.observe_episode(
            episode_safety_cost=10.0,
            safety_transition_count=1,
        )
        self.assertEqual(upper.current_lambda, 3.0)

        lower = self._controller(
            lambda_init=1.0,
            lambda_lr=100.0,
            cost_budget=10.0,
        )
        lower.observe_episode(
            episode_safety_cost=0.0,
            safety_transition_count=1,
        )
        self.assertEqual(lower.current_lambda, 0.0)

    def test_warmup_interval_and_ema_are_episode_based(self):
        controller = self._controller(
            lambda_lr=1.0,
            warmup_steps=2,
            update_interval=2,
            cost_ema_factor=0.5,
        )
        for cost in (2.0, 4.0, 6.0):
            result = controller.observe_episode(
                episode_safety_cost=cost,
                safety_transition_count=1,
            )
            self.assertFalse(result["lambda_update_applied"])
        result = controller.observe_episode(
            episode_safety_cost=8.0,
            safety_transition_count=1,
        )
        # EMA: 2 -> 3 -> 4.5 -> 6.25。
        self.assertAlmostEqual(result["mean_safety_cost"], 6.25)
        self.assertAlmostEqual(result["current_lambda"], 7.25)
        self.assertEqual(result["lambda_update_count"], 1)
        self.assertEqual(result["observed_episode_count"], 4)

    def test_state_dict_restores_controller_exactly(self):
        source = self._controller(cost_ema_factor=0.5)
        source.observe_episode(
            episode_safety_cost=3.0,
            safety_transition_count=2,
        )
        restored = self._controller(cost_ema_factor=0.5)
        restored.load_state_dict(source.state_dict())
        self.assertEqual(restored.state_dict(), source.state_dict())

    def test_safe_rl_disabled_keeps_lambda_fixed(self):
        controller = self._controller(enabled=False)
        result = controller.observe_episode(
            episode_safety_cost=100.0,
            safety_transition_count=1,
        )
        self.assertEqual(result["current_lambda"], 1.0)
        self.assertEqual(result["lambda_update_count"], 0)
        self.assertEqual(result["observed_episode_count"], 0)
        self.assertEqual(result["lambda_update_reason"], "disabled")

        legacy_config = build_train_config(
            safe_rl_enabled=False,
            safe_rl_dynamic_lambda_enabled=False,
        )
        self.assertFalse(legacy_config.safe_rl.enabled)
        self.assertFalse(legacy_config.safe_rl.lagrangian.enabled)
        with self.assertRaisesRegex(
            ValueError,
            "requires safe_rl_enabled=True",
        ):
            build_train_config(
                safe_rl_enabled=False,
                safe_rl_dynamic_lambda_enabled=True,
            )

    def test_dynamic_config_exposes_all_required_fields(self):
        with patch("hrl_mix.train_config.os.makedirs"):
            config = build_train_config(
                safe_rl_enabled=True,
                safe_rl_dynamic_lambda_enabled=True,
            )
        lagrange = config.safe_rl.lagrangian
        self.assertTrue(lagrange.enabled)
        self.assertEqual(lagrange.lambda_init, 1.0)
        self.assertEqual(lagrange.lambda_lr, 0.01)
        self.assertEqual(lagrange.lambda_min, 0.0)
        self.assertEqual(lagrange.lambda_max, 100.0)
        self.assertEqual(lagrange.cost_budget, 0.0)
        self.assertEqual(lagrange.update_interval, 1)
        self.assertEqual(lagrange.cost_ema_factor, 0.9)
        self.assertEqual(lagrange.warmup_steps, 5)
        self.assertTrue(config.run_name.startswith("safe-ss-t-"))
        self.assertLessEqual(len(Path(config.save_dir).name), 48)

    def test_run_names_are_short_stable_and_safe_modes_are_distinct(self):
        with patch("hrl_mix.train_config.os.makedirs"):
            legacy = build_train_config(safe_rl_enabled=False)
            safe_fixed = build_train_config(safe_rl_enabled=True)
            safe_dynamic = build_train_config(
                safe_rl_enabled=True,
                safe_rl_dynamic_lambda_enabled=True,
            )
        self.assertEqual(
            legacy.run_name,
            "hrl-ss-t-s1",
        )
        self.assertNotEqual(safe_fixed.run_name, safe_dynamic.run_name)
        self.assertTrue(safe_fixed.run_name.startswith("safe-ss-t-"))
        self.assertTrue(safe_dynamic.run_name.startswith("safe-ss-t-"))
        self.assertEqual(
            Path(legacy.save_dir).name,
            legacy.run_name,
        )
        self.assertEqual(
            Path(legacy.log_path).name,
            "train.csv",
        )

    def test_non_finite_cost_is_rejected_without_nan(self):
        controller = self._controller()
        result = controller.observe_episode(
            episode_safety_cost=float("nan"),
            safety_transition_count=1,
        )
        self.assertTrue(math.isfinite(result["current_lambda"]))
        self.assertTrue(math.isfinite(result["mean_safety_cost"]))
        self.assertEqual(result["invalid_sample_count"], 1)
        self.assertEqual(result["lambda_update_count"], 0)

    def test_one_shared_lambda_is_synchronized_to_all_layers(self):
        class FakeAgent:
            safe_rl_enabled = True
            lagrange_multiplier = 0.0

        agents = [FakeAgent(), FakeAgent(), FakeAgent()]
        controller = self._controller()
        controller.observe_episode(
            episode_safety_cost=2.0,
            safety_transition_count=1,
        )
        value = synchronize_lagrange_multiplier(
            controller,
            agents,
        )
        self.assertEqual(value, 2.0)
        self.assertEqual(
            [agent.lagrange_multiplier for agent in agents],
            [2.0, 2.0, 2.0],
        )

@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class LagrangeCheckpointIntegrationTests(unittest.TestCase):
    @staticmethod
    def _agent(initial_lambda=1.0):
        return D3QNAgent(
            input_dim=2,
            output_dim=2,
            hidden_dims=(8,),
            head_hidden_dims=(8,),
            device="cpu",
            safe_rl_enabled=True,
            initial_lagrange_multiplier=initial_lambda,
        )

    def test_episode_end_updates_and_synchronizes_three_agents(self):
        class EpisodeEnvironment:
            _safety_cumulative_cost = 6.0
            _safety_cumulative_transition_count = 3

        controller = LagrangeSafetyController(
            enabled=True,
            lambda_init=1.0,
            lambda_lr=0.25,
            lambda_min=0.0,
            lambda_max=5.0,
            cost_budget=0.0,
            update_interval=1,
            cost_ema_factor=0.0,
            warmup_steps=0,
        )
        agents = [self._agent() for _ in range(3)]
        result = _update_shared_lagrange_at_episode_end(
            controller,
            EpisodeEnvironment(),
            agents,
        )
        self.assertAlmostEqual(result["mean_safety_cost"], 2.0)
        self.assertAlmostEqual(result["current_lambda"], 1.5)
        self.assertEqual(
            [agent.lagrange_multiplier for agent in agents],
            [1.5, 1.5, 1.5],
        )

    def test_final_evaluation_uses_zero_violation_standard(self):
        class FinishedEvalEnvironment:
            def __init__(self, **_kwargs):
                self.done_flag = True
                self.total_energy = 5.0

            def reset(self):
                self._safety_cumulative_deadline_violation_count = 1
                self._safety_cumulative_completed_workflow_count = 2

            @staticmethod
            def get_manager_state():
                return [0.0]

            @staticmethod
            def get_manager_action_mask():
                return [1.0]

            @staticmethod
            def apply_manager_delta(_delta):
                return None

        class FixedAgent:
            @staticmethod
            def select_action(
                _state,
                _mask,
                deterministic,
                count_step,
            ):
                self.assertTrue(deterministic)
                self.assertFalse(count_step)
                return 0

        result = evaluate_hrl_three_layer_multi_seed(
            FinishedEvalEnvironment,
            {},
            FixedAgent(),
            FixedAgent(),
            FixedAgent(),
            seeds=(1,),
            return_safety_metrics=True,
        )
        safety = result[-1]
        self.assertEqual(FINAL_EVALUATION_VIOLATION_BUDGET, 0.0)
        self.assertEqual(
            safety["evaluation_violation_budget"],
            0.0,
        )
        self.assertAlmostEqual(
            safety["deadline_violation_rate"],
            0.5,
        )
        self.assertFalse(safety["zero_violation_pass"])

    def test_agent_checkpoint_embeds_and_restores_controller(self):
        controller = LagrangeSafetyController(
            enabled=True,
            lambda_init=1.0,
            lambda_lr=0.25,
            lambda_min=0.0,
            lambda_max=5.0,
            cost_budget=0.0,
            update_interval=1,
            cost_ema_factor=0.5,
            warmup_steps=0,
        )
        controller.observe_episode(
            episode_safety_cost=4.0,
            safety_transition_count=2,
        )
        agent = self._agent(
            initial_lambda=controller.current_lambda
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dynamic_lambda.pth"
            _save_agent_checkpoint(
                agent,
                str(path),
                controller,
            )
            restored_agent = self._agent()
            restored_agent.load(str(path))

            restored_controller = LagrangeSafetyController(
                enabled=True,
                lambda_init=1.0,
                lambda_lr=0.25,
                lambda_min=0.0,
                lambda_max=5.0,
                cost_budget=0.0,
                update_interval=1,
                cost_ema_factor=0.5,
                warmup_steps=0,
            )
            restored_controller.load_state_dict(
                restored_agent.lagrange_controller_state
            )
            self.assertEqual(
                restored_controller.state_dict(),
                controller.state_dict(),
            )
            self.assertEqual(
                restored_agent.lagrange_multiplier,
                controller.current_lambda,
            )


if __name__ == "__main__":
    unittest.main()
