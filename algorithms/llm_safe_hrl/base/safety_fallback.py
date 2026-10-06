"""预测风险回退的兼容诊断接口，以及 CEWS 共用的固定 VM 排序。

合法候选始终交给 RL；资源不可用时由环境推进时间/等待。
"""

from __future__ import annotations

from math import isfinite
from typing import Mapping, Sequence


_FALLBACK_RECORD_FIELDS = (
    "fallback_triggered",
    "fallback_reason",
    "candidate_count",
    "minimum_violation",
    "selected_host",
    "selected_vm",
    "tie_break_stage",
)

_FALLBACK_METRIC_KEYS = (
    "predicted_violation_amount",
    "fuzzy_marginal_energy",
    "risk_finish",
)


def select_vm_candidate_by_fixed_rule_order(
    candidates: Sequence[Mapping],
    metric_keys: Sequence[str],
) -> dict:
    """按给定指标顺序比较，并始终以稳定 ``vm_id`` 作最后平局键。

    CEWS 固定 VM 规则和安全回退控制器共同调用该函数。调用方决定前置指标，
    因而 CEWS 原有按时/延期分支语义不变。
    """
    rows = list(candidates)
    if not rows:
        raise ValueError("at least one VM candidate is required")
    selected = min(
        rows,
        key=lambda row: tuple(
            float(row[key]) for key in metric_keys
        )
        + (int(row["vm_id"]),),
    )
    return dict(selected)


def _inactive_record(reason: str, candidate_count: int) -> dict:
    return {
        "fallback_triggered": False,
        "fallback_reason": str(reason),
        "candidate_count": int(candidate_count),
        "minimum_violation": 0.0,
        "selected_host": None,
        "selected_vm": None,
        "tie_break_stage": "not_triggered",
        "selected_candidate": None,
    }


class DeterministicFuzzyDDLFallbackController:
    """保留原候选校验和诊断接口，但不按 predicted risk 接管 RL。

    候选必须提供 ``vm_id``、``host_id``、
    ``predicted_violation_amount``、``fuzzy_marginal_energy`` 和
    ``risk_finish``。控制器不读取环境状态，也不执行动作，便于独立测试。
    """

    def __init__(self, *, enabled: bool = False):
        self.enabled = bool(enabled)

    @staticmethod
    def _normalize_candidates(
        candidates: Sequence[Mapping],
    ) -> list[dict]:
        normalized = []
        seen_vm_ids = set()
        for candidate in candidates:
            row = dict(candidate)
            missing = {
                key
                for key in (
                    "vm_id",
                    "host_id",
                    "predicted_violation_amount",
                    "fuzzy_marginal_energy",
                    "risk_finish",
                )
                if key not in row
            }
            if missing:
                raise ValueError(
                    "fallback candidate is missing fields: "
                    + ", ".join(sorted(missing))
                )

            vm_id = int(row["vm_id"])
            host_id = int(row["host_id"])
            violation = float(row["predicted_violation_amount"])
            energy = float(row["fuzzy_marginal_energy"])
            risk_finish = float(row["risk_finish"])
            if vm_id in seen_vm_ids:
                raise ValueError(
                    f"fallback candidates contain duplicate vm_id: {vm_id}"
                )
            if not all(
                isfinite(value)
                for value in (violation, energy, risk_finish)
            ):
                raise ValueError(
                    "fallback candidate metrics must be finite"
                )
            if violation < 0.0:
                raise ValueError(
                    "predicted_violation_amount must be non-negative"
                )

            seen_vm_ids.add(vm_id)
            row.update(
                {
                    "vm_id": vm_id,
                    "host_id": host_id,
                    "predicted_violation_amount": violation,
                    "fuzzy_marginal_energy": energy,
                    "risk_finish": risk_finish,
                }
            )
            normalized.append(row)
        return normalized

    @staticmethod
    def _tie_break_stage(candidates: Sequence[Mapping]) -> str:
        if len(candidates) == 1:
            return "single_candidate"

        minimum_violation = min(
            row["predicted_violation_amount"] for row in candidates
        )
        violation_ties = [
            row
            for row in candidates
            if row["predicted_violation_amount"] == minimum_violation
        ]
        if len(violation_ties) == 1:
            return "minimum_violation"

        minimum_energy = min(
            row["fuzzy_marginal_energy"] for row in violation_ties
        )
        energy_ties = [
            row
            for row in violation_ties
            if row["fuzzy_marginal_energy"] == minimum_energy
        ]
        if len(energy_ties) == 1:
            return "fuzzy_marginal_energy"

        minimum_risk_finish = min(
            row["risk_finish"] for row in energy_ties
        )
        finish_ties = [
            row
            for row in energy_ties
            if row["risk_finish"] == minimum_risk_finish
        ]
        if len(finish_ties) == 1:
            return "risk_finish_time"
        return "stable_vm_id"

    def select(
        self,
        candidates: Sequence[Mapping],
        *,
        safe_action_count: int,
        fallback_reason: str = "empty_safe_action_set",
    ) -> dict:
        """保留诊断接口；无 predicted-safe 动作时仍由 RL 选择合法动作。"""
        normalized = self._normalize_candidates(candidates)
        if not self.enabled:
            return _inactive_record(
                "fallback_controller_disabled",
                len(normalized),
            )
        # Candidates are hard-legal actions: predicted risk never bypasses RL.
        if normalized:
            return _inactive_record("legal_action_available", len(normalized))
        return _inactive_record("no_legal_action", 0)

    @staticmethod
    def record_fields(record: Mapping) -> dict:
        """提取供环境 info/日志使用的稳定回退记录字段。"""
        return {
            key: record.get(key)
            for key in _FALLBACK_RECORD_FIELDS
        }


__all__ = [
    "DeterministicFuzzyDDLFallbackController",
    "select_vm_candidate_by_fixed_rule_order",
]
