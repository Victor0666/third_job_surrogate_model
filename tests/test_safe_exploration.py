"""阶段 9：三层安全 epsilon-greedy 与动作审计测试。"""

from __future__ import annotations

import unittest

import numpy as np


try:
    import torch
    import torch.nn as nn

    from base.d3qn_agent import D3QNAgent
    from base.safety_shield import FuzzyDDLSafetyShield
    from hrl_mix.train_utils import (
        finalize_action_audit,
        select_layer_action_with_info,
    )

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None
    nn = None
    D3QNAgent = None
    FuzzyDDLSafetyShield = None
    finalize_action_audit = None
    select_layer_action_with_info = None
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class FixedQ(nn.Module):
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
class SafeExplorationTests(unittest.TestCase):
    @staticmethod
    def _agent(*, epsilon):
        return D3QNAgent(
            input_dim=2,
            output_dim=4,
            hidden_dims=(8,),
            head_hidden_dims=(8,),
            device="cpu",
            batch_size=1,
            buffer_size=16,
            eps_start=epsilon,
            eps_end=epsilon,
            target_update_tau=0.0,
            target_update_freq=100,
            safe_rl_enabled=True,
            initial_lagrange_multiplier=1.0,
        )

    def test_epsilon_one_random_exploration_never_crosses_final_mask(self):
        np.random.seed(7)
        agent = self._agent(epsilon=1.0)
        final_mask = np.asarray(
            [0.0, 1.0, 0.0, 1.0],
            dtype=np.float32,
        )
        observed = set()
        for _ in range(100):
            selection = agent.select_action_with_info(
                np.zeros(2, dtype=np.float32),
                final_mask,
            )
            observed.add(selection["action"])
            self.assertEqual(
                selection["selection_type"],
                "random_safe_exploration",
            )
            self.assertGreater(
                final_mask[selection["action"]],
                0.5,
            )
        self.assertEqual(observed, {1, 3})

    def test_epsilon_zero_greedy_uses_safe_score_and_final_mask(self):
        agent = self._agent(epsilon=0.0)
        # 动作 0 的 Q_r 最大但被 mask；动作 1 的
        # Q_r-lambda*Q_c=5，高于动作 3 的 4。
        agent.online = FixedQ([100.0, 5.0, 1.0, 7.0])
        agent.q_c_online = FixedQ([0.0, 0.0, 0.0, 3.0])
        selection = agent.select_action_with_info(
            np.zeros(2, dtype=np.float32),
            np.asarray(
                [0.0, 1.0, 0.0, 1.0],
                dtype=np.float32,
            ),
        )
        self.assertEqual(selection["action"], 1)
        self.assertEqual(
            selection["selection_type"],
            "greedy_safe_action",
        )
        self.assertFalse(selection["random_exploration"])

    def test_empty_final_set_bypasses_agent_and_triggers_fallback(self):
        agent = self._agent(epsilon=1.0)
        state = {
            "obs": np.zeros(2, dtype=np.float32),
            "mask": np.ones(4, dtype=np.float32),
            "final_action_mask": np.zeros(
                4,
                dtype=np.float32,
            ),
            "fallback_action": 2,
        }
        (
            action,
            selected_by_agent,
            learning_mask,
            selection,
        ) = select_layer_action_with_info(
            agent,
            state,
            safe_rl_enabled=True,
            deterministic=False,
            count_step=True,
        )
        self.assertEqual(action, 2)
        self.assertFalse(selected_by_agent)
        self.assertEqual(agent._action_calls, 0)
        self.assertFalse(np.any(learning_mask))
        self.assertEqual(
            selection["action_source"],
            "fallback_action",
        )
        self.assertIsNone(selection["proposed_action"])
        self.assertEqual(selection["executed_action"], 2)

    def test_shield_correction_preserves_proposed_and_executed(self):
        shield = FuzzyDDLSafetyShield(enabled=True)
        masks = shield.combine_masks(
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
        )
        decision = shield.resolve_action(
            1,
            masks,
            action_metrics=[
                {
                    "predicted_violation_amount": 0.0,
                    "fuzzy_safety_margin": 2.0,
                    "fuzzy_finish_risk": 8.0,
                },
                {
                    "predicted_violation_amount": 3.0,
                    "fuzzy_safety_margin": -3.0,
                    "fuzzy_finish_risk": 13.0,
                },
                {
                    "predicted_violation_amount": 0.0,
                    "fuzzy_safety_margin": 1.0,
                    "fuzzy_finish_risk": 9.0,
                },
                {},
            ],
            layer="vm",
        )
        audit = finalize_action_audit(
            {
                "action": 1,
                "proposed_action": 1,
                "selected_by_agent": True,
                "selection_type": "random_safe_exploration",
                "policy_selection_type": (
                    "random_safe_exploration"
                ),
            },
            decision,
        )
        self.assertEqual(audit["proposed_action"], 1)
        self.assertEqual(audit["executed_action"], 0)
        self.assertTrue(audit["action_modified"])
        self.assertEqual(
            audit["policy_selection_type"],
            "random_safe_exploration",
        )
        self.assertEqual(
            audit["action_source"],
            "shield_correction",
        )

    def test_replay_trains_executed_action_and_keeps_proposal_for_audit(self):
        agent = self._agent(epsilon=0.0)
        state = np.asarray([1.0, 2.0], dtype=np.float32)
        mask = np.asarray(
            [1.0, 0.0, 1.0, 0.0],
            dtype=np.float32,
        )
        agent.remember(
            state,
            mask,
            2,
            1.0,
            np.zeros_like(state),
            np.zeros_like(mask),
            1.0,
            cost=0.5,
            proposed_action=1,
            action_source="shield_correction",
            policy_selection_type="random_safe_exploration",
            action_modified=True,
        )
        transition = agent.buffer[0]
        self.assertEqual(transition.executed_action, 2)
        self.assertEqual(transition.proposed_action, 1)
        self.assertEqual(
            transition.action_source,
            "shield_correction",
        )
        self.assertEqual(
            transition.policy_selection_type,
            "random_safe_exploration",
        )
        self.assertTrue(transition.shield_modified)

        agent.update()
        self.assertEqual(
            agent.last_update_info["executed_action_mean"],
            2.0,
        )
        self.assertEqual(
            agent.last_update_info["proposed_action_mean"],
            1.0,
        )
        self.assertEqual(
            agent.last_update_info["action_modified_rate"],
            1.0,
        )

    def test_safe_rl_disabled_keeps_legacy_epsilon_greedy_and_replay(self):
        agent = D3QNAgent(
            input_dim=2,
            output_dim=4,
            hidden_dims=(8,),
            head_hidden_dims=(8,),
            device="cpu",
            batch_size=1,
            buffer_size=4,
            eps_start=1.0,
            eps_end=1.0,
            safe_rl_enabled=False,
        )
        action = agent.select_action(
            np.zeros(2, dtype=np.float32),
            np.asarray([0.0, 1.0, 0.0, 0.0]),
        )
        self.assertEqual(action, 1)
        self.assertEqual(
            agent.last_action_selection["selection_type"],
            "random_exploration",
        )
        agent.remember(
            np.zeros(2, dtype=np.float32),
            np.asarray([0.0, 1.0, 0.0, 0.0]),
            1,
            1.0,
            np.zeros(2, dtype=np.float32),
            np.zeros(4, dtype=np.float32),
            1.0,
        )
        self.assertEqual(len(agent.buffer[0]), 7)


if __name__ == "__main__":
    unittest.main()
