"""Auditable cache for successful local counterfactual comparisons."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Any

from .schemas import canonical_hash


LOCAL_ESTIMATOR_VERSION = "local_one_step_v2"


@dataclass(frozen=True)
class CounterfactualCacheKey:
    structure_hash: str
    frozen_rule_hash: str
    scenario_id: str
    seed: int
    decision_state_hash: str
    selected_task_id: int
    alternative_task_id: int
    counterfactual_config_hash: str
    resource_config_hash: str
    estimator_version: str = LOCAL_ESTIMATOR_VERSION

    @property
    def digest(self) -> str:
        return canonical_hash(asdict(self))


class CounterfactualCache:
    """JSON cache that never stores exceptions or incomplete results."""

    def __init__(self, *, enabled: bool = True, path: str | Path | None = None):
        self.enabled = bool(enabled)
        self.path = Path(path).resolve() if path else None
        self._items: dict[str, dict[str, Any]] = {}
        self.hits = 0
        self.misses = 0
        if self.enabled and self.path and self.path.is_file():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                self._items = payload

    def get(self, key: CounterfactualCacheKey) -> dict[str, Any] | None:
        if not self.enabled:
            self.misses += 1
            return None
        value = self._items.get(key.digest)
        if value is None:
            self.misses += 1
            return None
        self.hits += 1
        return json.loads(json.dumps(value))

    def put(self, key: CounterfactualCacheKey, value: dict[str, Any]) -> None:
        if not self.enabled:
            return
        if not isinstance(value, dict) or not value.get("state_immutable", False):
            return
        self._items[key.digest] = json.loads(json.dumps(value, allow_nan=False))
        self.flush()

    def flush(self) -> None:
        if not self.enabled or self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self._items, ensure_ascii=True, allow_nan=False, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(self.path)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return float(self.hits / total) if total else 0.0
