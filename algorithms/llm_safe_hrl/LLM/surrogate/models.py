"""Lazy ExtraTrees models with tree-spread and residual uncertainty."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class SurrogatePrediction:
    metrics: dict[str, Any]
    uncertainty: dict[str, float]

    @property
    def total_uncertainty(self) -> float:
        return float(sum(self.uncertainty.values()))


class ExtraTreesMetricModel:
    """DDL-first ranking model with separate constraint/energy regressors."""

    TARGETS = ("violation", "lateness", "energy")

    def __init__(self, *, random_seed: int = 0, conservative_sigma: float = 1.0):
        self.random_seed = int(random_seed)
        self.conservative_sigma = float(conservative_sigma)
        self.regressors: dict[str, Any] = {}
        self.residual_scale: dict[str, float] = {}
        self.validation_metrics: dict[str, float] = {}
        self.mode = "ranking_only"
        self.healthy = False
        self.failure_reason = "not_trained"

    @staticmethod
    def _regressor_class():
        try:
            from sklearn.ensemble import ExtraTreesRegressor
        except ImportError as exc:
            raise RuntimeError("scikit-learn is required when surrogate.enabled=true") from exc
        return ExtraTreesRegressor

    def fit(self, records: Sequence[Mapping[str, Any]]) -> None:
        self.healthy = False
        self.regressors = {}
        self.residual_scale = {}
        self.validation_metrics = {}
        if len(records) < 2:
            self.failure_reason = "insufficient_samples"
            return
        X = np.asarray([row["features"] for row in records], dtype=np.float64)
        labels = [row["label"] for row in records]
        if not np.isfinite(X).all():
            self.failure_reason = "nonfinite_features"
            return
        ExtraTreesRegressor = self._regressor_class()
        random = np.random.RandomState(self.random_seed)
        permutation = random.permutation(len(records))
        validation_count = min(max(1, len(records) // 5), len(records) - 1)
        validation_indices = permutation[:validation_count]
        training_indices = permutation[validation_count:]
        regressors = {}
        calibration_regressors = {}
        residual_scale = {}
        for offset, target in enumerate(self.TARGETS):
            y = np.asarray([float(label[target]) for label in labels], dtype=np.float64)
            if not np.isfinite(y).all():
                self.failure_reason = f"nonfinite_{target}_labels"
                return
            calibration_model = ExtraTreesRegressor(
                n_estimators=96,
                min_samples_leaf=2,
                random_state=self.random_seed + offset + 1,
                n_jobs=1,
            )
            calibration_model.fit(X[training_indices], y[training_indices])
            validation_error = (
                calibration_model.predict(X[validation_indices])
                - y[validation_indices]
            )
            residual_scale[target] = float(
                np.sqrt(np.mean(validation_error ** 2))
            )
            calibration_regressors[target] = calibration_model
            model = ExtraTreesRegressor(
                n_estimators=96,
                min_samples_leaf=2,
                random_state=self.random_seed + offset + 1,
                n_jobs=1,
            )
            model.fit(X, y)
            regressors[target] = model
        self.regressors = regressors
        self.residual_scale = residual_scale
        conservative_rows = []
        for row_index in validation_indices:
            row = X[row_index : row_index + 1]
            predicted = {}
            for target, calibration_model in calibration_regressors.items():
                tree_values = self._tree_values(calibration_model, row)
                value = float(
                    np.mean(tree_values)
                    + self.conservative_sigma
                    * math.hypot(
                        np.std(tree_values),
                        residual_scale[target],
                    )
                )
                predicted[target] = max(0.0, value) if target != "energy" else value
            predicted["constraint_feasible"] = bool(predicted["violation"] <= 0.0)
            conservative_rows.append(predicted)
        actual_validation = [labels[index] for index in validation_indices]

        def priority(row):
            violation = max(0.0, float(row["violation"]))
            ddl_violated = violation > 0.0
            return (
                1.0 if ddl_violated else 0.0,
                violation if ddl_violated else 0.0,
                float(row["lateness"]) if ddl_violated else 0.0,
                float(row["energy"]),
            )
        top_count = max(1, int(math.ceil(len(actual_validation) * 0.25)))
        actual_top = set(
            sorted(
                range(len(actual_validation)),
                key=lambda index: (priority(actual_validation[index]), index),
            )[:top_count]
        )
        predicted_top = set(
            sorted(
                range(len(conservative_rows)),
                key=lambda index: (priority(conservative_rows[index]), index),
            )[:top_count]
        )
        self.validation_metrics = {
            "promising_recall": float(len(actual_top & predicted_top) / len(actual_top)),
            "validation_samples": float(len(actual_validation)),
        }
        self.healthy = True
        self.failure_reason = ""

    @staticmethod
    def _tree_values(model: Any, row: np.ndarray) -> np.ndarray:
        return np.asarray(
            [float(tree.predict(row)[0]) for tree in model.estimators_],
            dtype=np.float64,
        )

    def predict(self, features: Sequence[float], *, robustness: float = 0.0) -> SurrogatePrediction:
        if not self.healthy or set(self.regressors) != set(self.TARGETS):
            raise RuntimeError("surrogate model is not healthy: " + self.failure_reason)
        row = np.asarray(features, dtype=np.float64).reshape(1, -1)
        predictions: dict[str, float] = {}
        uncertainties: dict[str, float] = {}
        for target, model in self.regressors.items():
            values = self._tree_values(model, row)
            predictions[target] = float(np.mean(values))
            uncertainties[target] = float(math.hypot(np.std(values), self.residual_scale[target]))
        sigma = self.conservative_sigma
        violation = max(0.0, predictions["violation"] + sigma * uncertainties["violation"])
        lateness = max(0.0, predictions["lateness"] + sigma * uncertainties["lateness"])
        energy = predictions["energy"] + sigma * uncertainties["energy"]
        metrics = {
            "constraint_feasible": bool(violation <= 0.0),
            "deadline_violation_rate": violation,
            "max_deadline_violation_rate_across_seeds": violation,
            "constraint_violation": violation,
            "total_lateness": lateness,
            "constraint_secondary_violation": lateness,
            "objective": energy,
            "fuzzy_total_energy_score": energy,
            "objective_std_across_seeds": max(0.0, float(robustness)),
            "objective_cv_across_seeds": max(0.0, float(robustness)),
            "evaluation_source": "surrogate_conservative",
            "per_seed_metrics": [],
        }
        return SurrogatePrediction(metrics, uncertainties)
