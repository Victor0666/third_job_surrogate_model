"""空安全动作集合下的确定性模糊 DDL 回退控制器。

该控制器复用 CEWS 固定 VM 规则在“候选全部延期”时的字典序核心：
违反量、边际能耗、完成时间、稳定 VM ID。区别仅在于本模块消费阶段 3
基于动态 ``D_safe`` 计算的违反量和风险完成时间，不修改 CEWS 原规则。
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
    """按固定字典序选择一个硬合法 VM，并由该 VM 唯一确定 Host。

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
        """在且仅在安全动作集合为空时选择确定性回退 VM。"""
        normalized = self._normalize_candidates(candidates)
        if not self.enabled:
            return _inactive_record(
                "fallback_controller_disabled",
                len(normalized),
            )
        if int(safe_action_count) > 0:
            return _inactive_record(
                "safe_action_available",
                len(normalized),
            )
        if not normalized:
            raise RuntimeError(
                "safe action set is empty but no hard-legal fallback "
                "candidate is available"
            )

        candidates_by_host = {}
        for row in normalized:
            candidates_by_host.setdefault(row["host_id"], []).append(
                row
            )
        host_best_candidates = [
            select_vm_candidate_by_fixed_rule_order(
                host_candidates,
                _FALLBACK_METRIC_KEYS,
            )
            for _, host_candidates in sorted(
                candidates_by_host.items()
            )
        ]
        # 先求每个 Host 的内部最佳 VM，再用完全相同的全序比较这些代表项。
        # 该两级选择与直接取全局最小等价，但显式保证 Host 不是任意选取。
        selected = select_vm_candidate_by_fixed_rule_order(
            host_best_candidates,
            _FALLBACK_METRIC_KEYS,
        )
        return {
            "fallback_triggered": True,
            "fallback_reason": str(fallback_reason),
            "candidate_count": int(len(normalized)),
            "minimum_violation": float(
                selected["predicted_violation_amount"]
            ),
            "selected_host": int(selected["host_id"]),
            "selected_vm": int(selected["vm_id"]),
            "tie_break_stage": self._tie_break_stage(normalized),
            "selected_candidate": dict(selected),
            "host_best_candidates": [
                dict(row) for row in host_best_candidates
            ],
            "ranking_rule": (
                "predicted_violation_amount -> "
                "fuzzy_marginal_energy -> risk_finish -> stable_vm_id"
            ),
        }

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
