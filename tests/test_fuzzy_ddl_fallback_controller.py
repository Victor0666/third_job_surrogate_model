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

    def test_predicted_unsafe_candidates_do_not_trigger_fallback(self):
        for candidates in (
            [_candidate(10, 0, 3.0, 8.0, 20.0)],
            [_candidate(10, 0, 3.0, 8.0, 20.0),
             _candidate(20, 1, 1.0, 4.0, 19.0)],
        ):
            result = self.controller.select(candidates, safe_action_count=0)
            self.assertFalse(result["fallback_triggered"])
            self.assertEqual(result["fallback_reason"], "legal_action_available")
            self.assertEqual(result["candidate_count"], len(candidates))
            self.assertIsNone(result["selected_host"])
            self.assertIsNone(result["selected_vm"])

    def test_safe_action_available_does_not_trigger_fallback(self):
        result = self.controller.select(
            [_candidate(10, 0, 0.0, 5.0, 10.0)], safe_action_count=1,
        )
        self.assertFalse(result["fallback_triggered"])

    def test_no_legal_candidate_delegates_to_environment(self):
        result = self.controller.select([], safe_action_count=0)
        self.assertFalse(result["fallback_triggered"])
        self.assertEqual(result["fallback_reason"], "no_legal_action")
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
