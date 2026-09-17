"""阶段 10 风险调整模糊能耗增量 performance reward 测试。"""

from __future__ import annotations

import unittest

from base.hrl_env import HrlHeftEnv


def _summary(score, *, mean=None, std=None):
    return {
        "fuzzy_total_energy_score": float(score),
        "fuzzy_total_energy_mean": float(
            score if mean is None else mean
        ),
        "fuzzy_total_energy_std": float(
            0.0 if std is None else std
        ),
    }


def _reward_stub(after_summaries):
    environment = object.__new__(HrlHeftEnv)
    environment.safe_rl_enabled = True
    environment.energy_reward_scale = 1e-3
    environment.fuzzy_energy_uncertainty_weight = 1.0
    environment._safe_fuzzy_energy_score = 0.0
    summaries = iter(after_summaries)
    environment.get_fuzzy_energy_summary = lambda: next(summaries)
    return environment


class SafePerformanceRewardTests(unittest.TestCase):
    def test_action_before_after_baseline_matches_fuzzy_objective_delta(self):
        environment = _reward_stub(
            [_summary(130.0, mean=120.0, std=10.0)]
        )
        result = environment._safe_performance_reward_breakdown(
            _summary(100.0, mean=95.0, std=5.0)
        )
        self.assertEqual(
            result["fuzzy_energy_score_before"],
            100.0,
        )
        self.assertEqual(
            result["fuzzy_energy_score_after"],
            130.0,
        )
        self.assertEqual(result["performance_energy_delta"], 30.0)
        self.assertEqual(result["raw_energy_reward"], -30.0)
        self.assertAlmostEqual(result["energy_reward"], -0.03)
        self.assertAlmostEqual(
            result["total_performance_reward"],
            -0.03,
        )
        self.assertEqual(result["fuzzy_energy_mean_after"], 120.0)
        self.assertEqual(result["fuzzy_energy_std_after"], 10.0)

    def test_incremental_rewards_telescope_without_final_energy_duplication(self):
        environment = _reward_stub(
            [_summary(130.0), _summary(150.0)]
        )
        first = environment._safe_performance_reward_breakdown(
            _summary(100.0)
        )
        second = environment._safe_performance_reward_breakdown(
            _summary(130.0)
        )
        self.assertEqual(
            first["performance_energy_delta"]
            + second["performance_energy_delta"],
            50.0,
        )
        self.assertEqual(
            first["raw_energy_reward"]
            + second["raw_energy_reward"],
            -50.0,
        )
        self.assertAlmostEqual(
            first["total_performance_reward"]
            + second["total_performance_reward"],
            -0.05,
        )

    def test_shaping_fields_are_separate_and_zero_by_default(self):
        environment = _reward_stub([_summary(12.0)])
        result = environment._safe_performance_reward_breakdown(
            _summary(10.0)
        )
        for field in (
            "completion_reward",
            "waiting_reward",
            "utilization_reward",
            "communication_reward",
        ):
            self.assertEqual(result[field], 0.0)
        self.assertEqual(
            result["total_performance_reward"],
            sum(
                result[field]
                for field in (
                    "energy_reward",
                    "completion_reward",
                    "waiting_reward",
                    "utilization_reward",
                    "communication_reward",
                )
            ),
        )

    def test_safety_cost_is_not_part_of_total_performance_reward(self):
        environment = _reward_stub([_summary(11.0)])
        environment._safety_cumulative_cost = 999.0
        result = environment._safe_performance_reward_breakdown(
            _summary(10.0)
        )
        self.assertAlmostEqual(
            result["total_performance_reward"],
            -0.001,
        )
        self.assertFalse(
            result[
                "performance_reward_safety_cost_included"
            ]
        )
        self.assertNotIn("safety_cost", result)


if __name__ == "__main__":
    unittest.main()
