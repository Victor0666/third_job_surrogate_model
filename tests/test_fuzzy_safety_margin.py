"""动态模糊工作流安全裕量的阶段 2 回归测试。"""

from __future__ import annotations

from types import SimpleNamespace
import unittest

from base.hrl_env import HrlHeftEnv
from common.resource_opt import TriangularFuzzyNumber


def _margin_environment(deadlines, predictions, *, current_time=0.0):
    """构造只依赖统一安全裕量接口的最小环境桩。"""
    environment = object.__new__(HrlHeftEnv)
    environment.safe_rl_enabled = True
    environment.fuzzy_enabled = True
    environment.fuzzy_deadline_eta = 0.95
    environment.current_time = float(current_time)
    environment.workflows = [
        SimpleNamespace(
            deadline=float(deadline),
            arrival_time=0.0,
        )
        for deadline in deadlines
    ]
    environment.wf_finish_time = {}
    environment.task_meta = [
        (workflow_id, 0) for workflow_id in range(len(deadlines))
    ]
    environment.task_state = ["Ready"] * len(deadlines)
    environment.task_mi = [
        100.0 + workflow_id for workflow_id in range(len(deadlines))
    ]
    environment.task_up_rank = [
        10.0 + workflow_id for workflow_id in range(len(deadlines))
    ]
    environment.ready_task_ids = list(range(len(deadlines)))
    environment.predict_workflow_finish_tfn = (
        lambda workflow_id: predictions[int(workflow_id)]
    )
    return environment


class DynamicFuzzySafetyMarginTests(unittest.TestCase):
    """覆盖正、零、负裕量和工作流级聚合。"""

    def test_loose_deadline_has_positive_margin(self):
        environment = _margin_environment(
            [20.0],
            [TriangularFuzzyNumber(8.0, 10.0, 12.0)],
        )
        result = environment.get_dynamic_fuzzy_safety_margins()
        row = result["workflow_safety_margins"][0]
        required = {
            "workflow_id",
            "fuzzy_finish_estimate_lower",
            "fuzzy_finish_estimate_modal",
            "fuzzy_finish_estimate_upper",
            "fuzzy_finish_risk",
            "deadline",
            "fuzzy_safety_margin",
            "normalized_fuzzy_safety_margin",
            "is_predicted_at_risk",
        }
        self.assertTrue(required.issubset(row))
        self.assertAlmostEqual(row["fuzzy_finish_risk"], 11.9)
        self.assertAlmostEqual(row["fuzzy_safety_margin"], 8.1)
        self.assertAlmostEqual(
            row["normalized_fuzzy_safety_margin"],
            8.1 / 20.0,
        )
        self.assertFalse(row["is_predicted_at_risk"])
        self.assertEqual(row["safety_status"], "positive_margin")

    def test_deadline_equal_to_risk_finish_is_boundary(self):
        environment = _margin_environment(
            [11.9],
            [TriangularFuzzyNumber(8.0, 10.0, 12.0)],
        )
        result = environment.get_dynamic_fuzzy_safety_margins()
        row = result["workflow_safety_margins"][0]
        self.assertAlmostEqual(row["fuzzy_safety_margin"], 0.0)
        self.assertAlmostEqual(
            row["normalized_fuzzy_safety_margin"],
            0.0,
        )
        self.assertTrue(row["is_at_safety_boundary"])
        self.assertTrue(row["is_predicted_at_risk"])
        self.assertFalse(row["is_predicted_violation"])
        self.assertEqual(result["risk_workflow_ratio"], 1.0)
        self.assertEqual(result["predicted_violation_rate"], 0.0)

    def test_clearly_infeasible_deadline_has_negative_margin(self):
        environment = _margin_environment(
            [5.0],
            [TriangularFuzzyNumber(8.0, 10.0, 12.0)],
        )
        result = environment.get_dynamic_fuzzy_safety_margins()
        row = result["workflow_safety_margins"][0]
        self.assertAlmostEqual(row["fuzzy_safety_margin"], -6.9)
        self.assertEqual(
            row["normalized_fuzzy_safety_margin"],
            -1.0,
        )
        self.assertTrue(row["is_predicted_at_risk"])
        self.assertTrue(row["is_predicted_violation"])
        self.assertEqual(row["safety_status"], "predicted_violation")
        self.assertEqual(result["predicted_violation_rate"], 1.0)

    def test_modal_and_pessimistic_estimates_remain_distinct(self):
        environment = _margin_environment(
            [20.0],
            [TriangularFuzzyNumber(7.0, 8.0, 12.0)],
        )
        row = environment.get_dynamic_fuzzy_safety_margins()[
            "workflow_safety_margins"
        ][0]
        self.assertEqual(row["fuzzy_finish_estimate_modal"], 8.0)
        self.assertEqual(row["fuzzy_finish_estimate_upper"], 12.0)
        self.assertNotEqual(
            row["fuzzy_finish_estimate_modal"],
            row["fuzzy_finish_estimate_upper"],
        )
        self.assertAlmostEqual(row["fuzzy_finish_risk"], 11.8)

    def test_no_active_workflow_returns_zero_aggregates(self):
        environment = _margin_environment([], [])
        result = environment.get_dynamic_fuzzy_safety_margins()
        self.assertEqual(result["workflow_safety_margins"], [])
        self.assertEqual(result["unfinished_workflow_count"], 0)
        self.assertEqual(result["minimum_safety_margin"], 0.0)
        self.assertEqual(result["mean_safety_margin"], 0.0)
        self.assertEqual(result["risk_workflow_ratio"], 0.0)
        self.assertEqual(result["predicted_violation_rate"], 0.0)
        self.assertFalse(result["prediction_is_safety_guarantee"])

        disabled_environment = _margin_environment(
            [5.0],
            [TriangularFuzzyNumber(8.0, 10.0, 12.0)],
        )
        disabled_environment.safe_rl_enabled = False
        disabled_result = (
            disabled_environment.get_dynamic_fuzzy_safety_margins()
        )
        self.assertEqual(
            disabled_result["workflow_safety_margins"],
            [],
        )
        self.assertEqual(disabled_result["unfinished_workflow_count"], 0)

    def test_multiple_workflows_aggregate_margins_and_rates(self):
        common_prediction = TriangularFuzzyNumber(5.0, 6.0, 8.0)
        environment = _margin_environment(
            [10.0, 7.9, 7.0],
            [common_prediction, common_prediction, common_prediction],
        )
        result = environment.get_dynamic_fuzzy_safety_margins()
        margins = [
            row["fuzzy_safety_margin"]
            for row in result["workflow_safety_margins"]
        ]
        self.assertEqual(result["unfinished_workflow_count"], 3)
        self.assertAlmostEqual(min(margins), -0.9)
        self.assertAlmostEqual(result["minimum_safety_margin"], -0.9)
        self.assertAlmostEqual(result["mean_safety_margin"], 0.4)
        self.assertAlmostEqual(result["risk_workflow_ratio"], 2.0 / 3.0)
        self.assertAlmostEqual(
            result["predicted_violation_rate"],
            1.0 / 3.0,
        )
        self.assertEqual(result["risk_workflow_count"], 2)
        self.assertEqual(result["predicted_violation_count"], 1)


if __name__ == "__main__":
    unittest.main()
