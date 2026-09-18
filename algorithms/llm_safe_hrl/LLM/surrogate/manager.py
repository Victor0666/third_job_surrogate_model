"""Fail-open orchestration for structure and CMA parameter screening."""

from __future__ import annotations

from collections import deque
import hashlib
import math
import os
from pathlib import Path
import pickle
import threading
from typing import Any, Mapping, Sequence

from .config import SurrogateConfig
from .dataset import SurrogateContext, SurrogateDataset
from .features import extract_parameter_features
from .models import ExtraTreesMetricModel, SurrogatePrediction


def _label(metrics: Mapping[str, Any]) -> dict[str, Any]:
    violation = metrics.get(
        "deadline_violation_count",
        metrics.get(
            "max_deadline_violation_rate_across_seeds",
            metrics.get("deadline_violation_rate", metrics.get("constraint_violation", 1e300)),
        ),
    )
    return {
        "constraint_feasible": bool(metrics.get("constraint_feasible", False)),
        "violation": float(violation),
        "lateness": float(metrics.get("total_lateness", metrics.get("constraint_secondary_violation", 1e300))),
        "energy": float(metrics.get("fuzzy_total_energy_score", metrics.get("objective", metrics.get("energy", 1e300)))),
    }


class SurrogateManager:
    """Own exact datasets and two independent model bundles.

    Every public gate catches model failures and returns all candidates for exact
    evaluation.  Predictions never pass through the dataset insertion API.
    """

    def __init__(
        self,
        config: SurrogateConfig,
        context: SurrogateContext,
        *,
        artifact_root: str | os.PathLike[str] | None = None,
    ) -> None:
        self.config = config
        self.context = context
        root = Path(artifact_root or ".")
        dataset_path = Path(config.dataset_path)
        checkpoint_path = Path(config.checkpoint_path)
        self.dataset = SurrogateDataset(
            dataset_path if dataset_path.is_absolute() else root / dataset_path,
            context,
        )
        self.checkpoint_path = checkpoint_path if checkpoint_path.is_absolute() else root / checkpoint_path
        self.structure_model = ExtraTreesMetricModel(
            random_seed=config.random_seed,
            conservative_sigma=config.conservative_sigma,
        )
        self.parameter_model = ExtraTreesMetricModel(
            random_seed=config.random_seed + 1009,
            conservative_sigma=config.conservative_sigma,
        )
        self._lock = threading.RLock()
        self._new_exact = 0
        self._audit_promising = {
            kind: deque(maxlen=config.fail_safe.audit_window)
            for kind in ("structure", "parameter")
        }
        self._stable_audit_rounds = {"structure": 0, "parameter": 0}
        self.gate_disabled_reasons = {"structure": "", "parameter": ""}
        self._pending_structure_audits: dict[int, dict[str, Any]] = {}
        self.disabled_reason = "" if config.enabled else "disabled"
        if config.enabled:
            self._load_checkpoint()
            if not self.disabled_reason and self.dataset.count():
                self.retrain(force=True)

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    @property
    def healthy(self) -> bool:
        return self.enabled and not self.disabled_reason

    def gate_healthy(self, kind: str) -> bool:
        return self.healthy and not self.gate_disabled_reasons[kind]

    @staticmethod
    def _priority_key(metrics: Mapping[str, Any]) -> tuple[float, ...]:
        try:
            from rule_optimization.cmaes_optimizer import constraint_priority_key
        except ImportError:  # pragma: no cover
            from ..rule_optimization.cmaes_optimizer import constraint_priority_key
        return constraint_priority_key(metrics)

    def _model_ready(self, kind: str) -> bool:
        model = self.structure_model if kind == "structure" else self.parameter_model
        minimum = (
            self.config.warmup.min_structures
            if kind == "structure"
            else self.config.parameter_gate.min_exact_pairs
        )
        return (
            self.gate_healthy(kind)
            and model.healthy
            and self.dataset.count(kind) >= minimum
            and self.dataset.count() >= self.config.warmup.min_exact_evaluations
        )

    def _load_checkpoint(self) -> None:
        if not self.checkpoint_path.exists():
            return
        try:
            with self.checkpoint_path.open("rb") as handle:
                payload = pickle.load(handle)  # noqa: S301 - trusted local artifact
            if payload.get("context_hash") != self.context.context_hash:
                self.disabled_reason = "checkpoint_context_mismatch"
                return
            self.structure_model = payload["structure_model"]
            self.parameter_model = payload["parameter_model"]
            self.gate_disabled_reasons.update(
                payload.get("gate_disabled_reasons", {})
            )
        except Exception as exc:  # fail open
            self.disabled_reason = f"checkpoint_load_error:{type(exc).__name__}"

    def _save_checkpoint(self) -> None:
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.checkpoint_path.with_suffix(self.checkpoint_path.suffix + ".tmp")
        payload = {
            "context_hash": self.context.context_hash,
            "context": self.context.as_dict(),
            "structure_model": self.structure_model,
            "parameter_model": self.parameter_model,
            "gate_disabled_reasons": dict(self.gate_disabled_reasons),
        }
        with temporary.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.checkpoint_path)

    def retrain(self, *, force: bool = False) -> None:
        if not self.enabled:
            return
        with self._lock:
            if not force and self._new_exact < self.config.retrain_every_exact_evaluations:
                return
            for kind, model in (
                ("structure", self.structure_model),
                ("parameter", self.parameter_model),
            ):
                try:
                    model.fit(self.dataset.records(kind))
                    if not model.healthy:
                        self.gate_disabled_reasons[kind] = str(
                            model.failure_reason
                        )
                        continue
                    promising_recall = model.validation_metrics.get(
                        "promising_recall",
                        0.0,
                    )
                    validation_passed = (
                        promising_recall
                        >= self.config.fail_safe.min_promising_recall
                    )
                    self.gate_disabled_reasons[kind] = (
                        "" if validation_passed
                        else "model_validation_recall_below_threshold"
                    )
                    if validation_passed:
                        self._audit_promising[kind].clear()
                        self._stable_audit_rounds[kind] = 0
                except Exception as exc:
                    self.gate_disabled_reasons[kind] = (
                        f"training_error:{type(exc).__name__}"
                    )
            if any(model.healthy for model in (self.structure_model, self.parameter_model)):
                self._save_checkpoint()
            self._new_exact = 0

    def _record(self, kind: str, features, metrics, metadata) -> None:
        with self._lock:
            self.dataset.add_exact(kind, features, _label(metrics), metadata)
            self._new_exact += 1
        self.retrain()

    def record_structure_exact(
        self,
        candidate,
        parameters: Mapping[str, float],
        quick_metrics: Mapping[str, Any],
        final_metrics: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return
        features = extract_parameter_features(candidate, parameters, quick_metrics)
        final_label = _label(final_metrics)
        quick_label = _label(quick_metrics)
        final_label["energy_improvement"] = (
            quick_label["energy"] - final_label["energy"]
        )
        final_label["feasibility_improved"] = (
            not quick_label["constraint_feasible"]
            and final_label["constraint_feasible"]
        )
        with self._lock:
            self.dataset.add_exact(
                "structure",
                features,
                final_label,
                metadata,
            )
            self._new_exact += 1
            iteration = int((metadata or {}).get("iteration", -1))
            pending = self._pending_structure_audits.get(iteration)
            if pending is not None:
                pending["actual"][str((metadata or {}).get("structure_hash", ""))] = dict(final_metrics)
                if len(pending["actual"]) >= pending["expected"]:
                    actual = pending["actual"]
                    ordered = sorted(actual, key=lambda key: self._priority_key(actual[key]))
                    top_count = max(
                        1,
                        min(
                            len(ordered),
                            self.config.structure_gate.min_true_structures,
                        ),
                    )
                    top = set(ordered[:top_count])
                    for structure_hash in actual:
                        self.record_audit(
                            gate="structure",
                            actual_promising=structure_hash in top,
                            selected_for_exact=structure_hash in pending["would_select"],
                        )
                    self.finish_audit("structure")
                    self._pending_structure_audits.pop(iteration, None)
        self.retrain()

    def record_parameter_exact(
        self,
        candidate,
        parameters: Mapping[str, float],
        quick_metrics: Mapping[str, Any],
        refine_metrics: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return
        features = extract_parameter_features(candidate, parameters, quick_metrics)
        self._record("parameter", features, refine_metrics, metadata)

    def _deterministic_index(self, identities: Sequence[str], salt: str) -> int:
        ranked = []
        for index, identity in enumerate(identities):
            digest = hashlib.sha256(
                f"{self.config.random_seed}:{salt}:{identity}".encode("utf-8")
            ).hexdigest()
            ranked.append((digest, index))
        return min(ranked)[1]

    def _select(
        self,
        predictions: Sequence[SurrogatePrediction],
        identities: Sequence[str],
        budget: int,
        salt: str,
    ) -> tuple[set[int], dict[int, list[str]]]:
        count = len(predictions)
        budget = min(count, max(1, int(budget)))
        ordered = sorted(
            range(count),
            key=lambda index: (self._priority_key(predictions[index].metrics), index),
        )
        selected = set(ordered[:budget])
        reasons = {index: ["predicted_promising"] for index in selected}
        uncertainty_quota = (
            max(
                1,
                int(math.ceil(budget * self.config.active_learning.uncertainty_fraction)),
            )
            if self.config.active_learning.uncertainty_fraction > 0.0
            else 0
        )
        random_quota = (
            max(
                1,
                int(math.ceil(budget * self.config.active_learning.random_fraction)),
            )
            if self.config.active_learning.random_fraction > 0.0
            else 0
        )
        uncertainty_order = sorted(
            range(count),
            key=lambda index: (-predictions[index].total_uncertainty, index),
        )
        random_order = sorted(
            range(count),
            key=lambda index: hashlib.sha256(
                f"{self.config.random_seed}:{salt}:{identities[index]}".encode(
                    "utf-8"
                )
            ).hexdigest(),
        )
        anchors = [
            *((index, "high_uncertainty") for index in uncertainty_order[:uncertainty_quota]),
            *((index, "deterministic_random") for index in random_order[:random_quota]),
        ]
        protected = {ordered[0]}
        for anchor, reason in anchors:
            if anchor not in selected and len(selected) >= budget:
                replaceable = sorted(selected - protected, key=lambda index: (ordered.index(index), index), reverse=True)
                if replaceable:
                    removed = replaceable[0]
                    selected.remove(removed)
                    reasons.pop(removed, None)
            if len(selected) < budget or anchor in selected:
                selected.add(anchor)
                reasons.setdefault(anchor, []).append(reason)
                protected.add(anchor)
        if not any(
            "deterministic_random" in item_reasons
            for item_reasons in reasons.values()
        ):
            selected_list = sorted(selected)
            local = self._deterministic_index(
                [identities[index] for index in selected_list],
                salt + ":selected",
            )
            reasons.setdefault(selected_list[local], []).append(
                "deterministic_random"
            )
        return selected, reasons

    @staticmethod
    def _validate_predictions(predictions: Sequence[SurrogatePrediction]) -> None:
        for prediction in predictions:
            values = [
                prediction.total_uncertainty,
                prediction.metrics.get("deadline_violation_rate"),
                prediction.metrics.get("total_lateness"),
                prediction.metrics.get("fuzzy_total_energy_score"),
            ]
            if not all(math.isfinite(float(value)) for value in values):
                raise ValueError("surrogate returned a non-finite prediction")

    def _audit_interval(self, gate: str) -> int:
        if self._stable_audit_rounds[gate] >= self.config.fail_safe.stable_audit_rounds:
            return self.config.fail_safe.stable_audit_every_generations
        return self.config.fail_safe.audit_every_generations

    def select_structures(
        self,
        candidates: Sequence[Any],
        parameter_maps: Sequence[Mapping[str, float]],
        quick_metrics: Sequence[Mapping[str, Any]],
        *,
        iteration: int,
    ) -> dict[str, Any]:
        count = len(candidates)
        fallback = {
            "selected_indices": list(range(count)),
            "predictions": [None] * count,
            "reasons": {index: ["exact_fallback"] for index in range(count)},
            "gate_used": False,
            "failure_reason": "not_ready",
        }
        if not self._model_ready("structure") or count == 0:
            return fallback
        try:
            predictions = [
                self.structure_model.predict(
                    extract_parameter_features(candidate, parameters, metrics),
                    robustness=float(metrics.get("objective_std_across_seeds", 0.0)),
                )
                for candidate, parameters, metrics in zip(candidates, parameter_maps, quick_metrics)
            ]
            self._validate_predictions(predictions)
            fraction = (
                self.config.structure_gate.initial_full_cma_fraction
                if self.dataset.count("structure") < 2 * self.config.warmup.min_structures
                else self.config.structure_gate.mature_full_cma_fraction
            )
            budget = max(
                self.config.structure_gate.min_true_structures,
                int(math.ceil(count * fraction)),
            )
            identities = [candidate.structure_hash for candidate in candidates]
            selected, reasons = self._select(predictions, identities, budget, f"structure:{iteration}")
            would_select = set(selected)
            audit_generation = (
                (int(iteration) + 1) % self._audit_interval("structure") == 0
            )
            if audit_generation:
                with self._lock:
                    self._pending_structure_audits[int(iteration)] = {
                        "expected": count,
                        "would_select": {
                            identities[index] for index in would_select
                        },
                        "actual": {},
                    }
                selected = set(range(count))
                for index in selected:
                    reasons.setdefault(index, []).append("recall_audit")
            return {
                "selected_indices": sorted(selected),
                "would_select_indices": sorted(would_select),
                "predictions": predictions,
                "reasons": reasons,
                "gate_used": True,
                "audit_generation": audit_generation,
                "failure_reason": "",
            }
        except Exception as exc:
            fallback["failure_reason"] = f"{type(exc).__name__}:{exc}"
            return fallback

    def select_parameters(
        self,
        candidate,
        parameter_maps: Sequence[Mapping[str, float]],
        quick_metrics: Sequence[Mapping[str, Any]],
        *,
        generation: int,
        iteration: int = 0,
    ) -> dict[str, Any]:
        count = len(parameter_maps)
        fallback = {
            "selected_indices": list(range(count)),
            "predictions": [None] * count,
            "reasons": {index: ["exact_fallback"] for index in range(count)},
            "gate_used": False,
            "failure_reason": "not_ready",
        }
        if not self._model_ready("parameter") or count == 0:
            return fallback
        try:
            predictions = [
                self.parameter_model.predict(
                    extract_parameter_features(candidate, parameters, metrics),
                    robustness=float(metrics.get("objective_std_across_seeds", 0.0)),
                )
                for parameters, metrics in zip(parameter_maps, quick_metrics)
            ]
            self._validate_predictions(predictions)
            budget = max(
                1,
                min(count, self.config.parameter_gate.min_true_candidates_per_generation),
            )
            identities = [
                ",".join(f"{key}={float(value):.12g}" for key, value in sorted(parameters.items()))
                for parameters in parameter_maps
            ]
            selected, reasons = self._select(predictions, identities, budget, f"parameter:{candidate.structure_hash}:{generation}")
            would_select = set(selected)
            # Both counters are stable inputs.  Adding them makes audits occur
            # across short CMA runs too (the production search has only four
            # generations, so ``generation + 1`` alone never reaches 10).
            sequence = int(iteration) + int(generation) + 1
            audit_generation = (
                sequence % self._audit_interval("parameter") == 0
            )
            if audit_generation:
                selected = set(range(count))
                for index in selected:
                    reasons.setdefault(index, []).append("recall_audit")
            return {
                "selected_indices": sorted(selected),
                "would_select_indices": sorted(would_select),
                "predictions": predictions,
                "reasons": reasons,
                "gate_used": True,
                "audit_generation": audit_generation,
                "failure_reason": "",
            }
        except Exception as exc:
            fallback["failure_reason"] = f"{type(exc).__name__}:{exc}"
            return fallback

    def record_audit(
        self,
        *,
        gate: str = "parameter",
        actual_promising: bool | None = None,
        selected_for_exact: bool | None = None,
        promising_found: bool | None = None,
    ) -> None:
        """Record DDL-first Top-K recall evidence from a fully exact audit."""
        if gate not in self.gate_disabled_reasons:
            raise ValueError(f"unknown surrogate gate: {gate}")
        with self._lock:
            if promising_found is not None:
                self._audit_promising[gate].append(bool(promising_found))
            elif actual_promising:
                self._audit_promising[gate].append(bool(selected_for_exact))

    def finish_audit(self, gate: str) -> None:
        """Apply one gate's recall result and adapt only that gate's interval."""
        if gate not in self.gate_disabled_reasons:
            raise ValueError(f"unknown surrogate gate: {gate}")
        with self._lock:
            minimum = self.config.fail_safe.min_audit_samples
            if len(self._audit_promising[gate]) < minimum:
                return
            promising_recall = sum(self._audit_promising[gate]) / len(
                self._audit_promising[gate]
            )
            if promising_recall < self.config.fail_safe.min_promising_recall:
                self.gate_disabled_reasons[gate] = "promising_recall_below_threshold"
                self._stable_audit_rounds[gate] = 0
            else:
                self.gate_disabled_reasons[gate] = ""
                self._stable_audit_rounds[gate] += 1

    def stats(self) -> dict[str, Any]:
        gate_stats = {}
        for gate in ("structure", "parameter"):
            promising = self._audit_promising[gate]
            gate_stats[gate] = {
                "healthy": self.gate_healthy(gate),
                "disabled_reason": self.gate_disabled_reasons[gate],
                "promising_recall": (
                    sum(promising) / len(promising) if promising else None
                ),
                "promising_audit_positives": len(promising),
                "stable_audit_rounds": self._stable_audit_rounds[gate],
                "audit_interval": self._audit_interval(gate),
            }
        return {
            "enabled": self.enabled,
            "healthy": self.healthy,
            "disabled_reason": self.disabled_reason,
            "exact_structure_labels": self.dataset.count("structure"),
            "exact_parameter_labels": self.dataset.count("parameter"),
            "gates": gate_stats,
            "structure_validation": dict(self.structure_model.validation_metrics),
            "parameter_validation": dict(self.parameter_model.validation_metrics),
            "models": {
                "structure": {
                    "healthy": bool(self.structure_model.healthy),
                    "mode": getattr(
                        self.structure_model,
                        "mode",
                        "unknown",
                    ),
                    "failure_reason": str(self.structure_model.failure_reason),
                },
                "parameter": {
                    "healthy": bool(self.parameter_model.healthy),
                    "mode": getattr(
                        self.parameter_model,
                        "mode",
                        "unknown",
                    ),
                    "failure_reason": str(self.parameter_model.failure_reason),
                },
            },
            "context_hash": self.context.context_hash,
        }
