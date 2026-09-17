"""独立的模糊 DDL 动作屏蔽与修正逻辑。

本模块只处理动作掩码、确定性修正和审计记录，不依赖 D3QN、runner 或环境
内部状态。任务完成时间和安全边界仍由环境预测，本模块消费统一的预测字段。
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def _binary_mask(values, *, name: str) -> np.ndarray:
    """把一维动作掩码规范化为只读语义的 ``float32`` 二值数组。"""
    mask = np.asarray(values, dtype=np.float32)
    if mask.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional mask")
    if not np.all(np.isfinite(mask)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return (mask > 0.5).astype(np.float32)


class FuzzyDDLSafetyShield:
    """构造安全动作集合并修正不安全的 Host/VM 动作。

    ``enabled=False`` 时，``final_action_mask`` 严格等于硬合法性掩码，
    ``resolve_action`` 原样返回 RL 动作，因而不会改变旧执行行为。
    """

    def __init__(self, *, enabled: bool = False):
        self.enabled = bool(enabled)

    def combine_masks(
        self,
        legal_action_mask,
        safety_action_mask,
    ) -> dict:
        """分别保留硬合法、安全 DDL 与最终动作掩码。"""
        legal = _binary_mask(
            legal_action_mask,
            name="legal_action_mask",
        )
        safety = _binary_mask(
            safety_action_mask,
            name="safety_action_mask",
        )
        if legal.shape != safety.shape:
            raise ValueError(
                "legal_action_mask and safety_action_mask must have "
                "the same shape"
            )
        safe_intersection = (
            (legal > 0.5) & (safety > 0.5)
        ).astype(np.float32)
        final = safe_intersection if self.enabled else legal.copy()
        return {
            "legal_action_mask": legal,
            "safety_action_mask": safety,
            "final_action_mask": final,
            "safe_legal_action_mask": safe_intersection,
            "safe_action_count": int(np.sum(safe_intersection)),
            "shield_enabled": bool(self.enabled),
        }

    def build_vm_masks(
        self,
        legal_action_mask,
        action_vm_ids: Sequence[int | None],
        vm_predictions: Sequence[Mapping],
    ) -> dict:
        """按候选 VM 的 ``is_predicted_safe`` 生成 VM safety mask。"""
        legal = _binary_mask(
            legal_action_mask,
            name="legal_action_mask",
        )
        if len(action_vm_ids) != legal.size:
            raise ValueError(
                "action_vm_ids length must match legal_action_mask"
            )
        predictions_by_vm = {
            int(prediction["vm_id"]): prediction
            for prediction in vm_predictions
        }
        safety = np.zeros_like(legal)
        for action, vm_id in enumerate(action_vm_ids):
            if vm_id is None:
                continue
            prediction = predictions_by_vm.get(int(vm_id))
            if prediction is not None and bool(
                prediction.get("is_predicted_safe", False)
            ):
                safety[action] = 1.0
        result = self.combine_masks(legal, safety)
        result["action_vm_ids"] = [
            None if vm_id is None else int(vm_id)
            for vm_id in action_vm_ids
        ]
        return result

    def build_host_masks(
        self,
        legal_action_mask,
        safe_legal_vm_count_by_host: Sequence[int],
    ) -> dict:
        """Host 内至少存在一个硬合法且安全的 VM 时标记为安全。"""
        legal = _binary_mask(
            legal_action_mask,
            name="legal_action_mask",
        )
        if len(safe_legal_vm_count_by_host) != legal.size:
            raise ValueError(
                "safe_legal_vm_count_by_host length must match "
                "legal_action_mask"
            )
        counts = np.asarray(
            safe_legal_vm_count_by_host,
            dtype=np.int64,
        )
        safety = (counts > 0).astype(np.float32)
        result = self.combine_masks(legal, safety)
        result["safe_legal_vm_count_by_host"] = [
            int(value) for value in counts
        ]
        return result

    @staticmethod
    def _metric(
        action_metrics: Sequence[Mapping] | None,
        action: int,
    ) -> dict:
        if action_metrics is None or action >= len(action_metrics):
            return {}
        value = action_metrics[action]
        return dict(value or {})

    def _replacement_key(
        self,
        action: int,
        action_metrics: Sequence[Mapping] | None,
    ) -> tuple:
        metric = self._metric(action_metrics, action)
        return (
            float(
                metric.get(
                    "predicted_violation_amount",
                    float("inf"),
                )
            ),
            -float(metric.get("safety_margin", -float("inf"))),
            float(metric.get("predicted_risk", float("inf"))),
            int(action),
        )

    def resolve_action(
        self,
        proposed_action: int,
        mask_bundle: Mapping,
        *,
        action_metrics: Sequence[Mapping] | None = None,
        fallback_action: int | None = None,
        layer: str,
    ) -> dict:
        """接受安全提议、修正不安全提议，或在空安全集时交给回退动作。"""
        legal = _binary_mask(
            mask_bundle["legal_action_mask"],
            name="legal_action_mask",
        )
        safety = _binary_mask(
            mask_bundle["safety_action_mask"],
            name="safety_action_mask",
        )
        final = _binary_mask(
            mask_bundle["final_action_mask"],
            name="final_action_mask",
        )
        if not (legal.shape == safety.shape == final.shape):
            raise ValueError("shield masks must have the same shape")

        proposed = int(proposed_action)
        proposed_in_range = 0 <= proposed < legal.size
        proposed_legal = bool(
            proposed_in_range and legal[proposed] > 0.5
        )
        proposed_safe = bool(
            proposed_in_range and safety[proposed] > 0.5
        )
        proposed_final = bool(
            proposed_in_range and final[proposed] > 0.5
        )

        fallback_applied = False
        proposal_accepted = False
        if not self.enabled:
            executed = proposed
            reason = "shield_disabled"
            proposal_accepted = True
        elif proposed_final:
            executed = proposed
            reason = "proposed_action_safe"
            proposal_accepted = True
        else:
            final_actions = np.flatnonzero(final > 0.5)
            if final_actions.size > 0:
                executed = min(
                    (int(action) for action in final_actions),
                    key=lambda action: self._replacement_key(
                        action,
                        action_metrics,
                    ),
                )
                reason = (
                    "proposed_action_hard_illegal"
                    if not proposed_legal
                    else "proposed_action_predicted_unsafe"
                )
            else:
                if fallback_action is None:
                    raise RuntimeError(
                        f"{layer} safety action set is empty and no "
                        "fallback action was provided"
                    )
                fallback = int(fallback_action)
                if fallback < 0 or fallback >= legal.size:
                    raise ValueError(
                        f"{layer} fallback action is out of range: "
                        f"{fallback}"
                    )
                if legal[fallback] <= 0.5:
                    raise ValueError(
                        f"{layer} fallback action violates the hard "
                        f"legal mask: {fallback}"
                    )
                executed = fallback
                reason = "no_safe_action_fallback"
                fallback_applied = True

        executed_metric = self._metric(action_metrics, executed)
        proposed_metric = (
            self._metric(action_metrics, proposed)
            if proposed_in_range
            else {}
        )
        intervention = bool(
            self.enabled and not proposal_accepted
        )
        action_modified = bool(int(executed) != int(proposed))
        return {
            "layer": str(layer),
            "shield_enabled": bool(self.enabled),
            "rl_proposed_action": int(proposed),
            "executed_action": int(executed),
            "action_modified": bool(action_modified),
            "shield_intervened": bool(intervention),
            "proposal_accepted": bool(proposal_accepted),
            "modification_reason": str(reason),
            "fallback_applied": bool(fallback_applied),
            "proposed_action_legal": bool(proposed_legal),
            "proposed_action_safe": bool(proposed_safe),
            "predicted_risk": float(
                executed_metric.get("predicted_risk", 0.0)
            ),
            "safety_margin": float(
                executed_metric.get("safety_margin", 0.0)
            ),
            "predicted_violation_amount": float(
                executed_metric.get(
                    "predicted_violation_amount",
                    0.0,
                )
            ),
            "proposed_predicted_risk": float(
                proposed_metric.get("predicted_risk", 0.0)
            ),
            "proposed_safety_margin": float(
                proposed_metric.get("safety_margin", 0.0)
            ),
            "legal_action_mask": legal.tolist(),
            "safety_action_mask": safety.tolist(),
            "final_action_mask": final.tolist(),
        }


__all__ = ["FuzzyDDLSafetyShield"]
