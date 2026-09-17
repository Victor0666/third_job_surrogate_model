"""版本化的安全 HRL 经验回放结构。

本模块只定义 transition、风险分类、序列化和 batch 堆叠，不依赖 D3QN
网络。安全 replay 不再使用容易错位的裸 tuple。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Mapping, Sequence

import numpy as np


SAFE_REPLAY_TRANSITION_SCHEMA_VERSION = 1
SAFE_REPLAY_BUFFER_SCHEMA_VERSION = 1

RISK_CATEGORIES = frozenset(
    {
        "normal_safe",
        "near_boundary",
        "shield_intervention",
        "fallback",
        "actual_violation",
    }
)

PERFORMANCE_REWARD_COMPONENTS = (
    "energy_reward",
    "completion_reward",
    "waiting_reward",
    "utilization_reward",
    "communication_reward",
    "total_performance_reward",
)


def _finite_vector(values, *, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32).reshape(-1)
    if vector.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return vector.copy()


def _binary_mask(values, *, name: str) -> np.ndarray:
    mask = _finite_vector(values, name=name)
    if np.any((mask < 0.0) | (mask > 1.0)):
        raise ValueError(f"{name} values must be in [0, 1]")
    return (mask > 0.5).astype(np.float32)


def _finite_scalar(value, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def classify_safe_replay_risk(
    *,
    violation_flag: bool,
    fallback_triggered: bool,
    shield_modified: bool,
    fuzzy_safety_margin: float,
    near_boundary_margin: float,
) -> str:
    """按互斥优先级分类风险经验。"""
    threshold = _finite_scalar(
        near_boundary_margin,
        name="near_boundary_margin",
    )
    if threshold < 0.0:
        raise ValueError(
            "near_boundary_margin must be non-negative"
        )
    margin = _finite_scalar(
        fuzzy_safety_margin,
        name="fuzzy_safety_margin",
    )
    if bool(violation_flag):
        return "actual_violation"
    if bool(fallback_triggered):
        return "fallback"
    if bool(shield_modified):
        return "shield_intervention"
    if margin <= threshold:
        return "near_boundary"
    return "normal_safe"


@dataclass(frozen=True)
class SafeReplayTransition:
    """一个可验证、可序列化的安全 transition。

    Q_r/Q_c 训练只能使用 ``executed_action``。``proposed_action`` 和
    shield/fallback 字段仅描述控制器如何改变动作。
    """

    state: np.ndarray
    proposed_action: int | None
    executed_action: int
    performance_reward: float
    safety_cost: float
    next_state: np.ndarray
    done: float
    legal_action_mask: np.ndarray
    safety_action_mask: np.ndarray
    final_action_mask: np.ndarray
    next_final_action_mask: np.ndarray
    shield_modified: bool
    fallback_triggered: bool
    fuzzy_safety_margin: float
    predicted_risk_finish: float
    violation_flag: bool
    manager_phase_id: int
    near_boundary_margin: float = 0.0
    action_source: str = "unspecified_safe_action"
    policy_selection_type: str = "unspecified_safe_action"
    performance_reward_components: Mapping[str, float] = field(
        default_factory=dict
    )
    risk_category: str = field(init=False)
    schema_version: int = field(
        init=False,
        default=SAFE_REPLAY_TRANSITION_SCHEMA_VERSION,
    )

    def __post_init__(self) -> None:
        state = _finite_vector(self.state, name="state")
        next_state = _finite_vector(
            self.next_state,
            name="next_state",
        )
        if state.shape != next_state.shape:
            raise ValueError(
                "state and next_state dimensions must match"
            )

        legal = _binary_mask(
            self.legal_action_mask,
            name="legal_action_mask",
        )
        safety = _binary_mask(
            self.safety_action_mask,
            name="safety_action_mask",
        )
        final = _binary_mask(
            self.final_action_mask,
            name="final_action_mask",
        )
        next_final = _binary_mask(
            self.next_final_action_mask,
            name="next_final_action_mask",
        )
        if not (
            legal.shape
            == safety.shape
            == final.shape
            == next_final.shape
        ):
            raise ValueError(
                "all action masks must have the same dimension"
            )
        if np.any((final > 0.5) & (legal <= 0.5)):
            raise ValueError(
                "final_action_mask cannot contain a hard-illegal action"
            )

        executed = int(self.executed_action)
        if executed < 0 or executed >= legal.size:
            raise ValueError("executed_action is out of range")
        if bool(self.fallback_triggered):
            if legal[executed] <= 0.5:
                raise ValueError(
                    "fallback executed_action must be hard legal"
                )
        elif final[executed] <= 0.5:
            raise ValueError(
                "non-fallback executed_action must be in "
                "final_action_mask"
            )

        proposed = self.proposed_action
        if proposed is not None:
            proposed = int(proposed)
            if proposed < 0 or proposed >= legal.size:
                raise ValueError("proposed_action is out of range")

        reward = _finite_scalar(
            self.performance_reward,
            name="performance_reward",
        )
        cost = _finite_scalar(
            self.safety_cost,
            name="safety_cost",
        )
        if cost < 0.0:
            raise ValueError("safety_cost must be non-negative")
        done = _finite_scalar(self.done, name="done")
        if done < 0.0 or done > 1.0:
            raise ValueError("done must be in [0, 1]")
        margin = _finite_scalar(
            self.fuzzy_safety_margin,
            name="fuzzy_safety_margin",
        )
        risk_finish = _finite_scalar(
            self.predicted_risk_finish,
            name="predicted_risk_finish",
        )
        phase_id = int(self.manager_phase_id)
        if phase_id < 0:
            raise ValueError(
                "manager_phase_id must be non-negative"
            )
        near_boundary_margin = _finite_scalar(
            self.near_boundary_margin,
            name="near_boundary_margin",
        )
        if near_boundary_margin < 0.0:
            raise ValueError(
                "near_boundary_margin must be non-negative"
            )

        supplied_components = dict(
            self.performance_reward_components or {}
        )
        if not supplied_components:
            supplied_components = {
                "energy_reward": reward,
                "completion_reward": 0.0,
                "waiting_reward": 0.0,
                "utilization_reward": 0.0,
                "communication_reward": 0.0,
                "total_performance_reward": reward,
            }
        missing = [
            key
            for key in PERFORMANCE_REWARD_COMPONENTS
            if key not in supplied_components
        ]
        if missing:
            raise ValueError(
                "performance_reward_components is missing: "
                + ", ".join(missing)
            )
        components = {
            key: _finite_scalar(
                supplied_components[key],
                name=f"performance_reward_components[{key}]",
            )
            for key in PERFORMANCE_REWARD_COMPONENTS
        }
        component_sum = sum(
            components[key]
            for key in PERFORMANCE_REWARD_COMPONENTS
            if key != "total_performance_reward"
        )
        if not np.isclose(
            components["total_performance_reward"],
            component_sum,
            rtol=1e-6,
            atol=1e-7,
        ):
            raise ValueError(
                "total_performance_reward must equal the sum of "
                "performance reward components"
            )
        if not np.isclose(
            components["total_performance_reward"],
            reward,
            rtol=1e-6,
            atol=1e-7,
        ):
            raise ValueError(
                "performance_reward must equal "
                "total_performance_reward"
            )

        category = classify_safe_replay_risk(
            violation_flag=bool(self.violation_flag),
            fallback_triggered=bool(self.fallback_triggered),
            shield_modified=bool(self.shield_modified),
            fuzzy_safety_margin=margin,
            near_boundary_margin=near_boundary_margin,
        )
        if category not in RISK_CATEGORIES:
            raise RuntimeError(
                f"unexpected risk category: {category}"
            )

        object.__setattr__(self, "state", state)
        object.__setattr__(self, "next_state", next_state)
        object.__setattr__(self, "legal_action_mask", legal)
        object.__setattr__(self, "safety_action_mask", safety)
        object.__setattr__(self, "final_action_mask", final)
        object.__setattr__(
            self,
            "next_final_action_mask",
            next_final,
        )
        object.__setattr__(self, "proposed_action", proposed)
        object.__setattr__(self, "executed_action", executed)
        object.__setattr__(self, "performance_reward", reward)
        object.__setattr__(self, "safety_cost", cost)
        object.__setattr__(self, "done", done)
        object.__setattr__(self, "fuzzy_safety_margin", margin)
        object.__setattr__(
            self,
            "predicted_risk_finish",
            risk_finish,
        )
        object.__setattr__(self, "manager_phase_id", phase_id)
        object.__setattr__(
            self,
            "near_boundary_margin",
            near_boundary_margin,
        )
        object.__setattr__(
            self,
            "performance_reward_components",
            components,
        )
        object.__setattr__(self, "risk_category", category)

    def training_fields(self) -> tuple:
        """返回 Q_r/Q_c 训练所需字段，动作固定为 executed_action。"""
        return (
            self.state,
            self.final_action_mask,
            self.executed_action,
            self.performance_reward,
            self.safety_cost,
            self.next_state,
            self.next_final_action_mask,
            self.done,
        )

    def to_dict(self) -> dict:
        """转换为 JSON 兼容且带 schema version 的字典。"""
        return {
            "schema_version": int(self.schema_version),
            "state": self.state.tolist(),
            "proposed_action": self.proposed_action,
            "executed_action": int(self.executed_action),
            "performance_reward": float(
                self.performance_reward
            ),
            "safety_cost": float(self.safety_cost),
            "next_state": self.next_state.tolist(),
            "done": float(self.done),
            "legal_action_mask": self.legal_action_mask.tolist(),
            "safety_action_mask": (
                self.safety_action_mask.tolist()
            ),
            "final_action_mask": self.final_action_mask.tolist(),
            "next_final_action_mask": (
                self.next_final_action_mask.tolist()
            ),
            "shield_modified": bool(self.shield_modified),
            "fallback_triggered": bool(
                self.fallback_triggered
            ),
            "fuzzy_safety_margin": float(
                self.fuzzy_safety_margin
            ),
            "predicted_risk_finish": float(
                self.predicted_risk_finish
            ),
            "violation_flag": bool(self.violation_flag),
            "manager_phase_id": int(self.manager_phase_id),
            "near_boundary_margin": float(
                self.near_boundary_margin
            ),
            "action_source": str(self.action_source),
            "policy_selection_type": str(
                self.policy_selection_type
            ),
            "performance_reward_components": dict(
                self.performance_reward_components
            ),
            "risk_category": str(self.risk_category),
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> "SafeReplayTransition":
        """严格读取当前 schema；无版本旧 tuple/dict 不会被猜测。"""
        if not isinstance(payload, Mapping):
            raise ValueError(
                "safe replay transition payload must be a mapping; "
                "legacy tuple replay requires an explicit migration"
            )
        version = payload.get("schema_version")
        if version != SAFE_REPLAY_TRANSITION_SCHEMA_VERSION:
            raise ValueError(
                "safe replay transition schema mismatch: "
                f"checkpoint={version!r}, current="
                f"{SAFE_REPLAY_TRANSITION_SCHEMA_VERSION}"
            )
        transition = cls(
            state=payload["state"],
            proposed_action=payload.get("proposed_action"),
            executed_action=payload["executed_action"],
            performance_reward=payload["performance_reward"],
            safety_cost=payload["safety_cost"],
            next_state=payload["next_state"],
            done=payload["done"],
            legal_action_mask=payload["legal_action_mask"],
            safety_action_mask=payload["safety_action_mask"],
            final_action_mask=payload["final_action_mask"],
            next_final_action_mask=payload[
                "next_final_action_mask"
            ],
            shield_modified=payload["shield_modified"],
            fallback_triggered=payload["fallback_triggered"],
            fuzzy_safety_margin=payload[
                "fuzzy_safety_margin"
            ],
            predicted_risk_finish=payload[
                "predicted_risk_finish"
            ],
            violation_flag=payload["violation_flag"],
            manager_phase_id=payload["manager_phase_id"],
            near_boundary_margin=payload.get(
                "near_boundary_margin",
                0.0,
            ),
            action_source=payload.get(
                "action_source",
                "unspecified_safe_action",
            ),
            policy_selection_type=payload.get(
                "policy_selection_type",
                "unspecified_safe_action",
            ),
            performance_reward_components=payload[
                "performance_reward_components"
            ],
        )
        stored_category = payload.get("risk_category")
        if (
            stored_category is not None
            and str(stored_category) != transition.risk_category
        ):
            raise ValueError(
                "stored risk_category does not match transition fields"
            )
        return transition

    def serialize(self) -> bytes:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
        ).encode("utf-8")

    @classmethod
    def deserialize(cls, data: bytes) -> "SafeReplayTransition":
        try:
            payload = json.loads(bytes(data).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                "invalid serialized safe replay transition"
            ) from exc
        return cls.from_dict(payload)


def stack_safe_replay_transitions(
    transitions: Sequence[SafeReplayTransition],
    *,
    input_dim: int,
    action_dim: int,
) -> dict:
    """校验并堆叠安全 batch，显式拒绝旧 tuple replay。"""
    rows = list(transitions)
    if not rows:
        raise ValueError("safe replay batch must not be empty")
    for row in rows:
        if not isinstance(row, SafeReplayTransition):
            raise ValueError(
                "legacy or unversioned safe replay transition "
                "detected; migrate it explicitly before sampling"
            )
        if row.state.size != int(input_dim):
            raise ValueError(
                "safe replay state dimension mismatch"
            )
        if row.final_action_mask.size != int(action_dim):
            raise ValueError(
                "safe replay action-mask dimension mismatch"
            )

    training_columns = tuple(
        zip(*(row.training_fields() for row in rows))
    )
    state, final_mask, action, reward, cost, next_state, next_mask, done = (
        map(np.asarray, training_columns)
    )
    return {
        "state": state.astype(np.float32, copy=False),
        "legal_action_mask": np.stack(
            [row.legal_action_mask for row in rows]
        ).astype(np.float32, copy=False),
        "safety_action_mask": np.stack(
            [row.safety_action_mask for row in rows]
        ).astype(np.float32, copy=False),
        "final_action_mask": final_mask.astype(
            np.float32,
            copy=False,
        ),
        "executed_action": action.astype(
            np.int64,
            copy=False,
        ),
        "performance_reward": reward.astype(
            np.float32,
            copy=False,
        ),
        "safety_cost": cost.astype(np.float32, copy=False),
        "next_state": next_state.astype(
            np.float32,
            copy=False,
        ),
        "next_final_action_mask": next_mask.astype(
            np.float32,
            copy=False,
        ),
        "done": done.astype(np.float32, copy=False),
        "proposed_action": np.asarray(
            [
                -1
                if row.proposed_action is None
                else row.proposed_action
                for row in rows
            ],
            dtype=np.int64,
        ),
        "shield_modified": np.asarray(
            [row.shield_modified for row in rows],
            dtype=np.float32,
        ),
        "fallback_triggered": np.asarray(
            [row.fallback_triggered for row in rows],
            dtype=np.float32,
        ),
        "fuzzy_safety_margin": np.asarray(
            [row.fuzzy_safety_margin for row in rows],
            dtype=np.float32,
        ),
        "predicted_risk_finish": np.asarray(
            [row.predicted_risk_finish for row in rows],
            dtype=np.float32,
        ),
        "violation_flag": np.asarray(
            [row.violation_flag for row in rows],
            dtype=np.float32,
        ),
        "manager_phase_id": np.asarray(
            [row.manager_phase_id for row in rows],
            dtype=np.int64,
        ),
        "action_source": np.asarray(
            [row.action_source for row in rows],
            dtype=object,
        ),
        "policy_selection_type": np.asarray(
            [row.policy_selection_type for row in rows],
            dtype=object,
        ),
        "risk_category": np.asarray(
            [row.risk_category for row in rows],
            dtype=object,
        ),
        "performance_reward_components": {
            key: np.asarray(
                [
                    row.performance_reward_components[key]
                    for row in rows
                ],
                dtype=np.float32,
            )
            for key in PERFORMANCE_REWARD_COMPONENTS
        },
    }


__all__ = [
    "PERFORMANCE_REWARD_COMPONENTS",
    "RISK_CATEGORIES",
    "SAFE_REPLAY_BUFFER_SCHEMA_VERSION",
    "SAFE_REPLAY_TRANSITION_SCHEMA_VERSION",
    "SafeReplayTransition",
    "classify_safe_replay_risk",
    "stack_safe_replay_transitions",
]
