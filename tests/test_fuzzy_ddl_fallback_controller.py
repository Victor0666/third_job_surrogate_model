"""阶段 5 空安全集确定性模糊 DDL 回退控制器测试。"""

from __future__ import annotations

import unittest

from base.safety_fallback import (
    DeterministicFuzzyDDLFallbackController,
)


def _candidate(
    vm_id,
    host_id,
    violation,
    energy,
    risk_finish,
):
    return {
        "vm_id": int(vm_id),
        "host_id": int(host_id),
        "predicted_violation_amount": float(violation),
        "fuzzy_marginal_energy": float(energy),
        "risk_finish": float(risk_finish),
    }


class DeterministicFuzzyDDLFallbackControllerTests(unittest.TestCase):
    def setUp(self):
        self.controller = (
            DeterministicFuzzyDDLFallbackController(enabled=True)
        )

    def test_no_safe_vm_triggers_minimum_violation_fallback(self):
        result = self.controller.select(
            [
                _candidate(10, 0, 3.0, 8.0, 20.0),
                _candidate(11, 0, 1.0, 9.0, 19.0),
            ],
            safe_action_count=0,
        )
        self.assertTrue(result["fallback_triggered"])
        self.assertEqual(
            result["fallback_reason"],
            "empty_safe_action_set",
        )
        self.assertEqual(result["candidate_count"], 2)
        self.assertEqual(result["minimum_violation"], 1.0)
        self.assertEqual(result["selected_host"], 0)
        self.assertEqual(result["selected_vm"], 11)
        self.assertEqual(
            result["tie_break_stage"],
            "minimum_violation",
        )

    def test_violation_tie_is_resolved_by_fuzzy_energy(self):
        result = self.controller.select(
            [
                _candidate(10, 0, 1.0, 9.0, 18.0),
                _candidate(11, 0, 1.0, 7.0, 20.0),
            ],
            safe_action_count=0,
        )
        self.assertEqual(result["selected_vm"], 11)
        self.assertEqual(
            result["tie_break_stage"],
            "fuzzy_marginal_energy",
        )

    def test_energy_tie_is_resolved_by_risk_finish(self):
        result = self.controller.select(
            [
                _candidate(10, 0, 1.0, 7.0, 20.0),
                _candidate(11, 0, 1.0, 7.0, 18.0),
            ],
            safe_action_count=0,
        )
        self.assertEqual(result["selected_vm"], 11)
        self.assertEqual(
            result["tie_break_stage"],
            "risk_finish_time",
        )

    def test_finish_time_tie_is_resolved_by_stable_vm_id(self):
        result = self.controller.select(
            [
                _candidate(12, 0, 1.0, 7.0, 18.0),
                _candidate(9, 0, 1.0, 7.0, 18.0),
            ],
            safe_action_count=0,
        )
        self.assertEqual(result["selected_vm"], 9)
        self.assertEqual(
            result["tie_break_stage"],
            "stable_vm_id",
        )

    def test_stable_id_result_does_not_depend_on_input_order(self):
        candidates = [
            _candidate(12, 0, 1.0, 7.0, 18.0),
            _candidate(9, 0, 1.0, 7.0, 18.0),
            _candidate(11, 0, 1.0, 7.0, 18.0),
        ]
        forward = self.controller.select(
            candidates,
            safe_action_count=0,
        )
        reverse = self.controller.select(
            list(reversed(candidates)),
            safe_action_count=0,
        )
        self.assertEqual(forward["selected_vm"], 9)
        self.assertEqual(reverse["selected_vm"], 9)
        self.assertEqual(
            forward["tie_break_stage"],
            "stable_vm_id",
        )

    def test_multi_host_selection_uses_globally_best_internal_vm(self):
        result = self.controller.select(
            [
                _candidate(10, 0, 2.0, 5.0, 18.0),
                _candidate(11, 0, 3.0, 4.0, 17.0),
                _candidate(20, 1, 1.0, 9.0, 20.0),
                _candidate(21, 1, 4.0, 3.0, 16.0),
            ],
            safe_action_count=0,
        )
        self.assertEqual(result["selected_vm"], 20)
        self.assertEqual(result["selected_host"], 1)
        self.assertEqual(result["minimum_violation"], 1.0)
        self.assertEqual(
            {
                row["host_id"]: row["vm_id"]
                for row in result["host_best_candidates"]
            },
            {0: 10, 1: 20},
        )

    def test_safe_action_available_does_not_trigger_fallback(self):
        result = self.controller.select(
            [_candidate(10, 0, 0.0, 5.0, 10.0)],
            safe_action_count=1,
        )
        self.assertFalse(result["fallback_triggered"])
        self.assertEqual(
            result["fallback_reason"],
            "safe_action_available",
        )
        self.assertIsNone(result["selected_host"])
        self.assertIsNone(result["selected_vm"])

    def test_disabled_controller_does_not_override_legacy_behavior(self):
        controller = DeterministicFuzzyDDLFallbackController(
            enabled=False
        )
        result = controller.select(
            [_candidate(10, 0, 2.0, 5.0, 10.0)],
            safe_action_count=0,
        )
        self.assertFalse(result["fallback_triggered"])
        self.assertEqual(
            result["fallback_reason"],
            "fallback_controller_disabled",
        )
        self.assertIsNone(result["selected_vm"])


if __name__ == "__main__":
    unittest.main()
