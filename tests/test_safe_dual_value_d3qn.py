"""阶段 7 独立 Q_r/Q_c、mask、Bellman 目标和 checkpoint 测试。"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np


try:
    import torch
    import torch.nn as nn

    from base.d3qn_agent import D3QNAgent
    from hrl_mix.train_runner import (
        _commit_safe_pending_transition,
    )
    from hrl_mix.train_utils import select_layer_action

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None
    nn = None
    D3QNAgent = None
    _commit_safe_pending_transition = None
    select_layer_action = None
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class FixedQ(nn.Module):
        """为动作选择和 Bellman target 测试提供固定 Q 向量。"""

        def __init__(self, values):
            super().__init__()
            self.register_buffer(
                "values",
                torch.tensor(values, dtype=torch.float32),
            )

        def forward(self, states):
            return self.values.unsqueeze(0).repeat(
                states.shape[0],
                1,
            )


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class SafeDualValueD3QNTests(unittest.TestCase):
    @staticmethod
    def _agent(
        *,
        safe_rl_enabled=True,
        gamma=0.9,
        safety_discount=0.8,
        initial_lagrange_multiplier=1.0,
    ):
        return D3QNAgent(
            input_dim=2,
            output_dim=3,
            hidden_dims=(8,),
            head_hidden_dims=(8,),
            device="cpu",
            batch_size=1,
            buffer_size=8,
            eps_start=0.0,
            eps_end=0.0,
            target_update_tau=0.0,
            target_update_freq=100,
            safe_rl_enabled=safe_rl_enabled,
            gamma=gamma,
            safety_discount=safety_discount,
            safety_learning_rate=1e-3,
            safety_loss_weight=0.5,
            initial_lagrange_multiplier=(
                initial_lagrange_multiplier
            ),
        )

    @staticmethod
    def _parameter_pointers(module):
        return {
            int(parameter.data_ptr())
            for parameter in module.parameters()
        }

    def test_three_agents_and_qr_qc_parameters_are_independent(self):
        agents = [self._agent() for _ in range(3)]
        all_network_pointer_sets = []
        for agent in agents:
            network_pointer_sets = [
                self._parameter_pointers(agent.online),
                self._parameter_pointers(agent.target),
                self._parameter_pointers(agent.q_c_online),
                self._parameter_pointers(agent.q_c_target),
            ]
            for left_index, left in enumerate(network_pointer_sets):
                for right in network_pointer_sets[left_index + 1 :]:
                    self.assertTrue(left.isdisjoint(right))
            all_network_pointer_sets.extend(network_pointer_sets)

        for left_index, left in enumerate(all_network_pointer_sets):
            for right in all_network_pointer_sets[left_index + 1 :]:
                self.assertTrue(left.isdisjoint(right))

    def test_reward_and_safety_bellman_targets_use_masked_policy(self):
        agent = self._agent(
            gamma=0.9,
            safety_discount=0.8,
            initial_lagrange_multiplier=1.0,
        )
        agent.online = FixedQ([1.0, 5.0, 100.0])
        agent.target = FixedQ([10.0, 20.0, 999.0])
        agent.q_c_online = FixedQ([0.0, 2.0, -100.0])
        agent.q_c_target = FixedQ([2.0, 3.0, 999.0])

        reward_target, safety_target, next_actions = (
            agent._compute_double_dqn_targets(
                torch.zeros((1, 2), dtype=torch.float32),
                torch.tensor(
                    [[1.0, 1.0, 0.0]],
                    dtype=torch.float32,
                ),
                torch.tensor([0.0], dtype=torch.float32),
                torch.tensor([1.0], dtype=torch.float32),
                torch.tensor([0.5], dtype=torch.float32),
            )
        )

        # 无 mask 时第三个动作综合分数最高；final mask 必须排除它。
        self.assertEqual(int(next_actions.item()), 1)
        self.assertAlmostEqual(
            float(reward_target.item()),
            1.0 + 0.9 * 20.0,
        )
        self.assertAlmostEqual(
            float(safety_target.item()),
            0.5 + 0.8 * 3.0,
            places=6,
        )

    def test_action_selection_uses_qr_minus_lambda_qc_and_mask(self):
        agent = self._agent(initial_lagrange_multiplier=1.0)
        agent.online = FixedQ([5.0, 4.0, 100.0])
        agent.q_c_online = FixedQ([10.0, 0.0, -100.0])
        action = agent.select_action(
            np.zeros(2, dtype=np.float32),
            np.asarray([1.0, 1.0, 0.0], dtype=np.float32),
            deterministic=True,
        )
        self.assertEqual(action, 1)
        with self.assertRaisesRegex(
            ValueError,
            "empty final_action_mask",
        ):
            agent.select_action(
                np.zeros(2, dtype=np.float32),
                np.zeros(3, dtype=np.float32),
                deterministic=True,
            )

    def test_qc_update_uses_cost_not_performance_reward(self):
        agent = self._agent(gamma=0.0, safety_discount=0.0)
        state = np.zeros(2, dtype=np.float32)
        mask = np.ones(3, dtype=np.float32)
        agent.remember(
            state,
            mask,
            0,
            2.0,
            state,
            np.zeros(3, dtype=np.float32),
            1.0,
            cost=7.0,
        )
        result = agent.update()

        self.assertIsInstance(result, float)
        self.assertAlmostEqual(
            agent.last_update_info["performance_target_mean"],
            2.0,
        )
        self.assertAlmostEqual(
            agent.last_update_info["safety_target_mean"],
            7.0,
        )
        self.assertIsNotNone(
            agent.last_update_info["safety_loss"]
        )

    def test_safe_checkpoint_saves_and_restores_both_values(self):
        agent = self._agent()
        state = np.zeros(2, dtype=np.float32)
        mask = np.ones(3, dtype=np.float32)
        agent.remember(
            state,
            mask,
            0,
            1.0,
            state,
            np.zeros(3, dtype=np.float32),
            1.0,
            cost=2.0,
        )
        agent.update()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dual_value.pth"
            agent.save(str(path))
            checkpoint = torch.load(
                str(path),
                map_location="cpu",
                weights_only=True,
            )
            self.assertTrue(checkpoint["safe_rl_enabled"])
            self.assertIsNotNone(checkpoint["online"])
            self.assertIsNotNone(checkpoint["target"])
            self.assertIsNotNone(checkpoint["optim"])
            self.assertIsNotNone(checkpoint["q_c_online"])
            self.assertIsNotNone(checkpoint["q_c_target"])
            self.assertIsNotNone(checkpoint["q_c_optim"])

            restored = self._agent()
            restored.load(str(path))
            for key, value in agent.online.state_dict().items():
                self.assertTrue(
                    torch.equal(
                        value,
                        restored.online.state_dict()[key],
                    )
                )
            for key, value in agent.q_c_online.state_dict().items():
                self.assertTrue(
                    torch.equal(
                        value,
                        restored.q_c_online.state_dict()[key],
                    )
                )
            self.assertTrue(restored.q_c_optim.state_dict()["state"])
            self.assertEqual(
                restored.lagrange_multiplier,
                agent.lagrange_multiplier,
            )
            with self.assertRaisesRegex(
                ValueError,
                "cannot be loaded into a performance-only agent",
            ):
                self._agent(safe_rl_enabled=False).load(str(path))

            legacy_path = Path(directory) / "performance_only.pth"
            self._agent(safe_rl_enabled=False).save(
                str(legacy_path)
            )
            with self.assertRaisesRegex(
                ValueError,
                "has no Q_c state",
            ):
                self._agent().load(str(legacy_path))

    def test_safe_disabled_keeps_legacy_replay_and_qr_policy(self):
        training_agent = self._agent(safe_rl_enabled=False)
        self.assertIsNone(training_agent.q_c_online)
        self.assertIsNone(training_agent.q_c_target)
        self.assertIsNone(training_agent.q_c_optim)

        state = np.zeros(2, dtype=np.float32)
        mask = np.ones(3, dtype=np.float32)
        training_agent.remember(
            state,
            mask,
            1,
            3.0,
            state,
            np.zeros(3, dtype=np.float32),
            1.0,
        )
        self.assertEqual(len(training_agent.buffer[0]), 7)
        self.assertIsInstance(training_agent.update(), float)
        self.assertIsNone(
            training_agent.last_update_info["safety_loss"]
        )

        policy_agent = self._agent(safe_rl_enabled=False)
        policy_agent.online = FixedQ([1.0, 5.0, 2.0])
        action = policy_agent.select_action(
            state,
            mask,
            deterministic=True,
        )
        self.assertEqual(action, 1)

    def test_layer_helper_passes_final_mask_and_bypasses_empty_set(self):
        class RecordingAgent:
            def __init__(self):
                self.calls = []

            def select_action(
                self,
                state,
                mask,
                deterministic,
                count_step,
            ):
                self.calls.append(np.array(mask, copy=True))
                return int(np.argmax(mask))

        agent = RecordingAgent()
        state = {
            "obs": np.zeros(2, dtype=np.float32),
            "mask": np.ones(3, dtype=np.float32),
            "final_action_mask": np.asarray(
                [0.0, 1.0, 0.0],
                dtype=np.float32,
            ),
            "fallback_action": None,
        }
        action, selected, learning_mask = select_layer_action(
            agent,
            state,
            safe_rl_enabled=True,
            deterministic=True,
            count_step=False,
        )
        self.assertEqual(action, 1)
        self.assertTrue(selected)
        np.testing.assert_array_equal(
            learning_mask,
            state["final_action_mask"],
        )
        np.testing.assert_array_equal(
            agent.calls[0],
            state["final_action_mask"],
        )

        empty_state = {
            **state,
            "final_action_mask": np.zeros(3, dtype=np.float32),
            "fallback_action": 2,
        }
        fallback, selected, _ = select_layer_action(
            agent,
            empty_state,
            safe_rl_enabled=True,
            deterministic=True,
            count_step=False,
        )
        self.assertEqual(fallback, 2)
        self.assertFalse(selected)
        self.assertEqual(len(agent.calls), 1)

    def test_pending_layer_transition_bootstraps_to_next_decision(self):
        agent = self._agent()
        pending = {
            "state": np.asarray([1.0, 2.0], dtype=np.float32),
            "mask": np.asarray(
                [1.0, 0.0, 1.0],
                dtype=np.float32,
            ),
            "action": 2,
            "reward": 3.0,
            "cost": 4.0,
        }
        next_state = np.asarray([5.0, 6.0], dtype=np.float32)
        next_mask = np.asarray(
            [0.0, 1.0, 0.0],
            dtype=np.float32,
        )
        _commit_safe_pending_transition(
            agent,
            pending,
            next_state=next_state,
            next_mask=next_mask,
            done=0.0,
            warmup_frac=1.0,
        )
        transition = agent.buffer[0]
        self.assertEqual(transition.executed_action, 2)
        self.assertEqual(transition.performance_reward, 3.0)
        self.assertEqual(transition.safety_cost, 4.0)
        np.testing.assert_array_equal(
            transition.next_state,
            next_state,
        )
        np.testing.assert_array_equal(
            transition.next_final_action_mask,
            next_mask,
        )
        self.assertEqual(transition.done, 0.0)
        self.assertEqual(transition.proposed_action, 2)
        self.assertEqual(
            transition.action_source,
            "unspecified_safe_action",
        )
        self.assertFalse(transition.shield_modified)


if __name__ == "__main__":
    unittest.main()
