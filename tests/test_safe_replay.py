"""阶段 10 版本化安全经验回放、风险分类和 PER 测试。"""

from __future__ import annotations

import json
import unittest

import numpy as np

from base.safe_replay import (
    RISK_CATEGORIES,
    SafeReplayTransition,
    stack_safe_replay_transitions,
)
from hrl_mix.train_config import SafetyReplayConfig


try:
    import torch

    from base.d3qn_agent import D3QNAgent

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None
    D3QNAgent = None
    TORCH_AVAILABLE = False


def _components(total=1.0):
    return {
        "energy_reward": float(total),
        "completion_reward": 0.0,
        "waiting_reward": 0.0,
        "utilization_reward": 0.0,
        "communication_reward": 0.0,
        "total_performance_reward": float(total),
    }


def _transition(
    *,
    proposed_action=0,
    executed_action=0,
    final_mask=(1.0, 0.0, 0.0),
    legal_mask=(1.0, 1.0, 0.0),
    safety_mask=(1.0, 0.0, 0.0),
    shield_modified=False,
    fallback_triggered=False,
    margin=2.0,
    violation=False,
    phase_id=3,
):
    return SafeReplayTransition(
        state=np.asarray([1.0, 2.0], dtype=np.float32),
        proposed_action=proposed_action,
        executed_action=executed_action,
        performance_reward=1.0,
        safety_cost=0.5,
        next_state=np.asarray([3.0, 4.0], dtype=np.float32),
        done=0.0,
        legal_action_mask=np.asarray(
            legal_mask,
            dtype=np.float32,
        ),
        safety_action_mask=np.asarray(
            safety_mask,
            dtype=np.float32,
        ),
        final_action_mask=np.asarray(
            final_mask,
            dtype=np.float32,
        ),
        next_final_action_mask=np.asarray(
            [0.0, 1.0, 0.0],
            dtype=np.float32,
        ),
        shield_modified=shield_modified,
        fallback_triggered=fallback_triggered,
        fuzzy_safety_margin=margin,
        predicted_risk_finish=12.0,
        violation_flag=violation,
        manager_phase_id=phase_id,
        near_boundary_margin=1.0,
        action_source="greedy_safe_action",
        policy_selection_type="greedy_safe_action",
        performance_reward_components=_components(),
    )


class SafeReplayTransitionTests(unittest.TestCase):
    def test_replay_config_validates_schema_threshold_and_td_weights(self):
        config = SafetyReplayConfig()
        self.assertEqual(config.transition_schema_version, 1)
        self.assertFalse(config.combined_per_priority)
        with self.assertRaises(ValueError):
            SafetyReplayConfig(transition_schema_version=99)
        with self.assertRaises(ValueError):
            SafetyReplayConfig(near_boundary_margin=-1.0)
        with self.assertRaises(ValueError):
            SafetyReplayConfig(
                combined_per_priority=True,
                performance_td_weight=0.0,
                safety_td_weight=0.0,
            )

    def test_transition_contains_required_fields_and_executed_training_action(self):
        transition = _transition(
            proposed_action=1,
            executed_action=0,
            shield_modified=True,
        )
        required = {
            "state",
            "proposed_action",
            "executed_action",
            "performance_reward",
            "safety_cost",
            "next_state",
            "done",
            "legal_action_mask",
            "safety_action_mask",
            "final_action_mask",
            "shield_modified",
            "fallback_triggered",
            "fuzzy_safety_margin",
            "predicted_risk_finish",
            "violation_flag",
            "manager_phase_id",
        }
        self.assertTrue(required.issubset(transition.to_dict()))
        self.assertEqual(transition.training_fields()[2], 0)
        self.assertEqual(transition.proposed_action, 1)

    def test_all_five_risk_categories_are_mutually_classified(self):
        rows = [
            _transition(margin=2.0),
            _transition(margin=1.0),
            _transition(
                proposed_action=1,
                shield_modified=True,
            ),
            _transition(
                proposed_action=None,
                executed_action=1,
                final_mask=(0.0, 0.0, 0.0),
                fallback_triggered=True,
                margin=-1.0,
            ),
            _transition(violation=True),
        ]
        self.assertEqual(
            {row.risk_category for row in rows},
            RISK_CATEGORIES,
        )

    def test_transition_json_serialization_round_trip(self):
        original = _transition(
            proposed_action=1,
            shield_modified=True,
            phase_id=9,
        )
        restored = SafeReplayTransition.deserialize(
            original.serialize()
        )
        self.assertEqual(
            restored.to_dict(),
            original.to_dict(),
        )
        json.dumps(original.to_dict(), allow_nan=False)

    def test_stack_shapes_and_metadata_are_explicit(self):
        batch = stack_safe_replay_transitions(
            [_transition(), _transition(phase_id=4)],
            input_dim=2,
            action_dim=3,
        )
        self.assertEqual(batch["state"].shape, (2, 2))
        self.assertEqual(
            batch["final_action_mask"].shape,
            (2, 3),
        )
        self.assertEqual(
            batch["legal_action_mask"].shape,
            (2, 3),
        )
        self.assertEqual(
            batch["safety_action_mask"].shape,
            (2, 3),
        )
        self.assertEqual(batch["executed_action"].shape, (2,))
        self.assertEqual(
            batch["fuzzy_safety_margin"].shape,
            (2,),
        )
        self.assertEqual(
            batch["predicted_risk_finish"].shape,
            (2,),
        )
        self.assertEqual(
            batch["manager_phase_id"].tolist(),
            [3, 4],
        )
        self.assertEqual(
            batch["performance_reward_components"][
                "total_performance_reward"
            ].shape,
            (2,),
        )
        np.testing.assert_array_equal(
            batch["executed_action"],
            [0, 0],
        )

    def test_fallback_can_use_hard_legal_action_when_final_mask_is_empty(self):
        transition = _transition(
            proposed_action=None,
            executed_action=1,
            final_mask=(0.0, 0.0, 0.0),
            fallback_triggered=True,
            margin=-2.0,
        )
        self.assertEqual(transition.risk_category, "fallback")
        self.assertEqual(transition.executed_action, 1)
        self.assertIsNone(transition.proposed_action)

    def test_invalid_dimensions_and_nonfallback_boundary_are_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "same dimension",
        ):
            SafeReplayTransition(
                **{
                    key: value
                    for key, value in {
                        **_transition().to_dict(),
                        "safety_action_mask": [1.0, 0.0],
                    }.items()
                    if key
                    not in {"schema_version", "risk_category"}
                }
            )
        with self.assertRaisesRegex(
            ValueError,
            "final_action_mask",
        ):
            _transition(
                executed_action=1,
                final_mask=(1.0, 0.0, 0.0),
            )

    def test_old_unversioned_replay_is_explicitly_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "legacy tuple",
        ):
            SafeReplayTransition.from_dict(
                (np.zeros(2), np.ones(3), 0)
            )
        with self.assertRaisesRegex(
            ValueError,
            "schema mismatch",
        ):
            SafeReplayTransition.from_dict(
                {
                    key: value
                    for key, value in _transition().to_dict().items()
                    if key != "schema_version"
                }
            )
        with self.assertRaisesRegex(
            ValueError,
            "legacy or unversioned",
        ):
            stack_safe_replay_transitions(
                [
                    (
                        np.zeros(2),
                        np.ones(3),
                        0,
                        1.0,
                    )
                ],
                input_dim=2,
                action_dim=3,
            )

    def test_reward_component_total_mismatch_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "must equal the sum",
        ):
            SafeReplayTransition(
                **{
                    key: value
                    for key, value in {
                        **_transition().to_dict(),
                        "performance_reward_components": {
                            **_components(),
                            "total_performance_reward": 2.0,
                        },
                    }.items()
                    if key
                    not in {"schema_version", "risk_category"}
                }
            )


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class SafeReplayAgentTests(unittest.TestCase):
    @staticmethod
    def _agent(*, use_per=False, combined=False):
        return D3QNAgent(
            input_dim=2,
            output_dim=3,
            hidden_dims=(8,),
            head_hidden_dims=(8,),
            device="cpu",
            batch_size=1,
            buffer_size=4,
            eps_start=0.0,
            eps_end=0.0,
            target_update_tau=0.0,
            target_update_freq=100,
            safe_rl_enabled=True,
            use_per=use_per,
            safe_per_combined_priority=combined,
            safe_per_performance_td_weight=1.0,
            safe_per_safety_td_weight=3.0,
            safe_replay_near_boundary_margin=1.0,
        )

    @staticmethod
    def _remember(agent):
        agent.remember(
            np.asarray([1.0, 2.0], dtype=np.float32),
            np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            0,
            2.0,
            np.asarray([3.0, 4.0], dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            1.0,
            cost=4.0,
            proposed_action=0,
            legal_action_mask=np.asarray([1.0, 1.0, 0.0]),
            safety_action_mask=np.asarray([1.0, 0.0, 0.0]),
            fuzzy_safety_margin=2.0,
            predicted_risk_finish=10.0,
            manager_phase_id=5,
            performance_reward_components=_components(2.0),
        )

    def test_replay_state_dict_round_trip_and_schema_guard(self):
        source = self._agent()
        self._remember(source)
        payload = source.replay_state_dict()
        json.dumps(payload, allow_nan=False)

        restored = self._agent()
        restored.load_replay_state_dict(payload)
        self.assertEqual(len(restored.buffer), 1)
        self.assertEqual(
            restored.buffer[0].to_dict(),
            source.buffer[0].to_dict(),
        )

        without_version = dict(payload)
        without_version.pop("replay_buffer_schema_version")
        with self.assertRaisesRegex(
            ValueError,
            "schema mismatch",
        ):
            restored.load_replay_state_dict(without_version)

        wrong_dimension = dict(payload)
        wrong_dimension["input_dim"] = 99
        with self.assertRaisesRegex(
            ValueError,
            "observation dimension",
        ):
            restored.load_replay_state_dict(wrong_dimension)

        wrong_replay_config = dict(payload)
        wrong_replay_config[
            "safe_replay_near_boundary_margin"
        ] = 2.0
        with self.assertRaisesRegex(
            ValueError,
            "config mismatch",
        ):
            restored.load_replay_state_dict(wrong_replay_config)

    def test_agent_stores_empty_final_mask_fallback_for_risk_replay(self):
        agent = self._agent()
        agent.remember(
            np.asarray([1.0, 2.0], dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            1,
            -1.0,
            np.asarray([3.0, 4.0], dtype=np.float32),
            np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            0.0,
            cost=2.0,
            proposed_action=None,
            legal_action_mask=np.asarray([1.0, 1.0, 0.0]),
            safety_action_mask=np.zeros(3, dtype=np.float32),
            fallback_triggered=True,
            fuzzy_safety_margin=-3.0,
            predicted_risk_finish=20.0,
            manager_phase_id=7,
            performance_reward_components=_components(-1.0),
        )
        transition = agent.buffer[0]
        self.assertEqual(transition.risk_category, "fallback")
        self.assertEqual(transition.executed_action, 1)
        self.assertIsNone(transition.proposed_action)

    def test_combined_per_priority_uses_performance_and_safety_td_errors(self):
        agent = self._agent(use_per=True, combined=True)
        for network in (
            agent.online,
            agent.target,
            agent.q_c_online,
            agent.q_c_target,
        ):
            for parameter in network.parameters():
                torch.nn.init.constant_(parameter, 0.0)
        self._remember(agent)
        agent.update()
        # done=1 且初始 Q=0，因此 |delta_r|=2、|delta_c|=4。
        # (1*2 + 3*4) / (1+3) = 3.5。
        self.assertAlmostEqual(
            agent.priorities[0],
            3.500001,
            places=5,
        )
        self.assertEqual(
            agent.last_update_info["per_priority_mode"],
            "combined_performance_safety_td",
        )


if __name__ == "__main__":
    unittest.main()
