"""Thread-safe exact-label dataset with strict experiment-context binding."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import threading
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class SurrogateContext:
    schema_version: str
    simulator_fingerprint: str
    evaluation_config_hash: str
    resource_config_hash: str
    protocol_identity: str
    scenario: str
    domain: str
    ddl: str = ""

    def as_dict(self) -> dict[str, str]:
        return asdict(self)

    @property
    def context_hash(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class SurrogateDataset:
    """Append-only records.  Public insertion accepts exact observations only."""

    def __init__(self, path: str | os.PathLike[str] | None, context: SurrogateContext):
        self.path = Path(path) if path else None
        self.context = context
        self._lock = threading.RLock()
        self._records: list[dict[str, Any]] = []
        self.ignored_context_records = 0
        self._load()

    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        with self._lock, self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except (json.JSONDecodeError, TypeError):
                    continue
                if row.get("context_hash") != self.context.context_hash:
                    self.ignored_context_records += 1
                    continue
                if row.get("label_source") != "exact":
                    continue
                self._records.append(row)

    def add_exact(
        self,
        kind: str,
        features: Sequence[float],
        label: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
        *,
        label_source: str = "exact",
    ) -> None:
        if label_source != "exact":
            raise ValueError("surrogate predictions cannot be added as training labels")
        row = {
            "schema_version": self.context.schema_version,
            "context_hash": self.context.context_hash,
            "context": self.context.as_dict(),
            "kind": str(kind),
            "features": [float(value) for value in features],
            "label": dict(label),
            "metadata": dict(metadata or {}),
            "label_source": "exact",
        }
        encoded = json.dumps(row, ensure_ascii=True, allow_nan=False, sort_keys=True)
        with self._lock:
            self._records.append(row)
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(encoded + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())

    def records(self, kind: str) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(row) for row in self._records if row.get("kind") == kind]

    def count(self, kind: str | None = None) -> int:
        with self._lock:
            if kind is None:
                return len(self._records)
            return sum(row.get("kind") == kind for row in self._records)
