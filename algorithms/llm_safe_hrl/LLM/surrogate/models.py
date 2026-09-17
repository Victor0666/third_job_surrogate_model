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
    feasible_probability: float

    @property
    def total_uncertainty(self) -> float:
        return float(sum(self.uncertainty.values()))


class ExtraTreesMetricModel:
    """One feasibility classifier and separate constraint/energy regressors."""

    TARGETS = ("violation", "lateness", "energy")

    def __init__(self, *, random_seed: int = 0, conservative_sigma: float = 1.0):
        self.random_seed = int(random_seed)
        self.conservative_sigma = float(conservative_sigma)
        self.classifier = None
        self.regressors: dict[str, Any] = {}
        self.residual_scale: dict[str, float] = {}
        self.classifier_residual = 0.0
        self.validation_metrics: dict[str, float] = {}
        self.healthy = False
        self.failure_reason = "not_trained"

    @staticmethod
    def _classes():
        try:
            from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
        except ImportError as exc:
            raise RuntimeError("scikit-learn is required when surrogate.enabled=true") from exc
        return ExtraTreesClassifier, ExtraTreesRegressor

    def fit(self, records: Sequence[Mapping[str, Any]]) -> None:
        self.healthy = False
        if len(records) < 2:
            self.failure_reason = "insufficient_samples"
            return
        X = np.asarray([row["features"] for row in records], dtype=np.float64)
        labels = [row["label"] for row in records]
        feasible = np.asarray([bool(label["constraint_feasible"]) for label in labels], dtype=int)
        if not np.isfinite(X).all() or len(np.unique(feasible)) < 2:
            self.failure_reason = "nonfinite_features_or_single_feasibility_class"
            return
        ExtraTreesClassifier, ExtraTreesRegressor = self._classes()
        validation_count = max(1, len(records) // 5)
        permutation = np.random.RandomState(self.random_seed).permutation(
            len(records)
        )
        validation_indices = permutation[:validation_count]
        training_indices = permutation[validation_count:]
        if (
            len(training_indices) < 2
            or len(np.unique(feasible[training_indices])) < 2
        ):
            self.failure_reason = "insufficient_validation_split_classes"
            return
        calibration_classifier = ExtraTreesClassifier(
            n_estimators=96,
            min_samples_leaf=2,
            random_state=self.random_seed,
            n_jobs=1,
        )
        calibration_classifier.fit(X[training_indices], feasible[training_indices])
        calibration_probability = calibration_classifier.predict_proba(
            X[validation_indices]
        )[:, list(calibration_classifier.classes_).index(1)]
        self.classifier_residual = float(
            np.sqrt(
                np.mean(
                    (calibration_probability - feasible[validation_indices]) ** 2
                )
            )
        )
        classifier = ExtraTreesClassifier(
            n_estimators=96,
            min_samples_leaf=2,
            random_state=self.random_seed,
            n_jobs=1,
        )
        classifier.fit(X, feasible)
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
        self.classifier = classifier
        self.regressors = regressors
        self.residual_scale = residual_scale
        conservative_rows = []
        for local_index, row_index in enumerate(validation_indices):
            row = X[row_index : row_index + 1]
            class_values = self._tree_values(
                calibration_classifier,
                row,
                probability=True,
            )
            probability = float(np.mean(class_values))
            class_uncertainty = float(
                math.hypot(np.std(class_values), self.classifier_residual)
            )
            predicted = {
                "constraint_feasible": bool(
                    probability
                    - self.conservative_sigma * class_uncertainty
                    >= 0.5
                )
            }
            for target, calibration_model in calibration_regressors.items():
                tree_values = self._tree_values(calibration_model, row)
                predicted[target] = float(
                    np.mean(tree_values)
                    + self.conservative_sigma
                    * math.hypot(
                        np.std(tree_values),
                        residual_scale[target],
                    )
                )
            conservative_rows.append(predicted)
        actual_validation = [labels[index] for index in validation_indices]
        positive_feasible = [
            index
            for index, label in enumerate(actual_validation)
            if bool(label["constraint_feasible"])
        ]
        feasible_recall = (
            sum(
                bool(conservative_rows[index]["constraint_feasible"])
                for index in positive_feasible
            )
            / len(positive_feasible)
            if positive_feasible
            else 0.0
        )
        def priority(row):
            return (
                0.0 if bool(row["constraint_feasible"]) else 1.0,
                0.0 if bool(row["constraint_feasible"]) else float(row["violation"]),
                0.0 if bool(row["constraint_feasible"]) else float(row["lateness"]),
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
            "feasible_recall": float(feasible_recall),
            "promising_recall": float(len(actual_top & predicted_top) / len(actual_top)),
            "validation_samples": float(len(actual_validation)),
        }
        self.healthy = True
        self.failure_reason = ""

    @staticmethod
    def _tree_values(model: Any, row: np.ndarray, *, probability: bool = False) -> np.ndarray:
        values = []
        for tree in model.estimators_:
            if probability:
                index = list(tree.classes_).index(1) if 1 in tree.classes_ else None
                values.append(0.0 if index is None else float(tree.predict_proba(row)[0, index]))
            else:
                values.append(float(tree.predict(row)[0]))
        return np.asarray(values, dtype=np.float64)

    def predict(self, features: Sequence[float], *, robustness: float = 0.0) -> SurrogatePrediction:
        if not self.healthy or self.classifier is None:
            raise RuntimeError("surrogate model is not healthy: " + self.failure_reason)
        row = np.asarray(features, dtype=np.float64).reshape(1, -1)
        class_values = self._tree_values(self.classifier, row, probability=True)
        probability = float(np.mean(class_values))
        class_uncertainty = float(math.hypot(np.std(class_values), self.classifier_residual))
        predictions: dict[str, float] = {}
        uncertainties: dict[str, float] = {"feasibility": class_uncertainty}
        for target, model in self.regressors.items():
            values = self._tree_values(model, row)
            predictions[target] = float(np.mean(values))
            uncertainties[target] = float(math.hypot(np.std(values), self.residual_scale[target]))
        sigma = self.conservative_sigma
        conservative_probability = probability - sigma * class_uncertainty
        violation = max(0.0, predictions["violation"] + sigma * uncertainties["violation"])
        lateness = max(0.0, predictions["lateness"] + sigma * uncertainties["lateness"])
        energy = predictions["energy"] + sigma * uncertainties["energy"]
        metrics = {
            "constraint_feasible": bool(conservative_probability >= 0.5),
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
        return SurrogatePrediction(metrics, uncertainties, probability)
