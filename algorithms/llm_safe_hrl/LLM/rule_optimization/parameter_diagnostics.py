"""Machine-readable diagnostics that inform later LLM structure evolution."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from .cmaes_optimizer import OptimizationResult, constraint_priority_key
from .parameter_schema import ParameterDefinition, ParameterSchema


def _confidence(sample_count: int, *, medium: int = 4, high: int = 10) -> str:
    if sample_count >= high:
        return "high"
    if sample_count >= medium:
        return "medium"
    return "low"


def _metric_snapshot(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "constraint_feasible": bool(metrics.get("constraint_feasible", False)),
        "deadline_violation_rate": float(
            metrics.get(
                "max_deadline_violation_rate_across_seeds",
                metrics.get("deadline_violation_rate", float("inf")),
            )
        ),
        "total_lateness": float(
            metrics.get(
                "total_lateness",
                metrics.get("constraint_secondary_violation", float("inf")),
            )
        ),
        "fuzzy_energy": float(
            metrics.get(
                "fuzzy_total_energy_score",
                metrics.get("objective", float("inf")),
            )
        ),
        "robustness": float(
            metrics.get(
                "objective_std_across_seeds",
                metrics.get("objective_cv_across_seeds", 0.0),
            )
        ),
    }


def _near_boundary(value: float, definition: ParameterDefinition, epsilon: float) -> tuple[bool, bool]:
    width = definition.upper_bound - definition.lower_bound
    margin = max(float(epsilon) * width, 1e-12)
    return (
        value <= definition.lower_bound + margin,
        value >= definition.upper_bound - margin,
    )


def _parameter_samples(
    entries: Sequence[Mapping[str, Any]],
    name: str,
) -> np.ndarray:
    return np.asarray(
        [float(entry["parameters"][name]) for entry in entries],
        dtype=float,
    )


def _local_effects(
    perturbations: Sequence[Mapping[str, Any]],
    name: str,
    best_metrics: Mapping[str, Any],
) -> list[dict[str, Any]]:
    baseline_key = constraint_priority_key(best_metrics)
    effects = []
    baseline_energy = float(
        best_metrics.get(
            "fuzzy_total_energy_score",
            best_metrics.get("objective", 0.0),
        )
    )
    for item in perturbations:
        if item.get("parameter") != name:
            continue
        metrics = item["metrics"]
        energy = float(
            metrics.get(
                "fuzzy_total_energy_score",
                metrics.get("objective", baseline_energy),
            )
        )
        effects.append(
            {
                "direction": int(item.get("direction", 0)),
                "delta": float(item.get("delta", 0.0)),
                "comparison_key_changed": constraint_priority_key(metrics) != baseline_key,
                "feasibility_changed": bool(metrics.get("constraint_feasible", False))
                != bool(best_metrics.get("constraint_feasible", False)),
                "relative_energy_change": float(
                    abs(energy - baseline_energy) / max(abs(baseline_energy), 1e-12)
                ),
                "metrics": _metric_snapshot(metrics),
                "evidence_source": str(
                    item.get(
                        "evaluation_source",
                        metrics.get("evidence_source", "real"),
                    )
                ),
            }
        )
    return effects


def _scenario_rows(metrics: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = metrics.get("per_seed_metrics", [])
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, Mapping)]


def generate_parameter_diagnostics(
    schema: ParameterSchema,
    optimization: OptimizationResult,
    *,
    boundary_epsilon: float = 0.02,
    sensitivity_epsilon: float = 0.02,
    correlation_threshold: float = 0.85,
) -> dict[str, Any]:
    """Generate evidence, confidence, and non-causal structural suggestions."""
    history = optimization.history
    elites = optimization.elite_samples
    perturbations = optimization.local_perturbations
    diagnostic_baseline = (
        optimization.diagnostic_baseline_metrics or optimization.best_metrics
    )
    boundary_analysis = []
    inactivity_analysis = []
    fragility_parameters = []

    for definition in schema.parameters:
        all_values = _parameter_samples(history, definition.name)
        elite_values = _parameter_samples(elites, definition.name)
        all_bounds = [
            _near_boundary(value, definition, boundary_epsilon)
            for value in all_values
        ]
        elite_bounds = [
            _near_boundary(value, definition, boundary_epsilon)
            for value in elite_values
        ]
        lower_rate = float(np.mean([item[0] for item in all_bounds])) if all_bounds else 0.0
        upper_rate = float(np.mean([item[1] for item in all_bounds])) if all_bounds else 0.0
        elite_lower_rate = (
            float(np.mean([item[0] for item in elite_bounds])) if elite_bounds else 0.0
        )
        elite_upper_rate = (
            float(np.mean([item[1] for item in elite_bounds])) if elite_bounds else 0.0
        )
        persistent_boundary = max(elite_lower_rate, elite_upper_rate) >= 0.6
        boundary_analysis.append(
            {
                "parameter": definition.name,
                "evidence": {
                    "lower_boundary_rate": lower_rate,
                    "upper_boundary_rate": upper_rate,
                    "elite_lower_boundary_rate": elite_lower_rate,
                    "elite_upper_boundary_rate": elite_upper_rate,
                    "elite_mean": float(np.mean(elite_values)),
                    "lower_bound": definition.lower_bound,
                    "upper_bound": definition.upper_bound,
                    "persistent_boundary_contact": persistent_boundary,
                },
                "confidence": _confidence(len(elite_values)),
                "suggested_structural_action": (
                    [
                        "reconsider_parameter_bounds_or_replace_linear_term_with_piecewise_or_gated_form"
                    ]
                    if persistent_boundary
                    else ["retain_current_parameterization"]
                ),
            }
        )

        width = definition.upper_bound - definition.lower_bound
        variance = float(np.var(elite_values)) if len(elite_values) else 0.0
        normalized_variance = variance / max(width * width, 1e-24)
        near_zero_rate = float(
            np.mean(np.abs(elite_values) <= max(0.02 * width, 1e-12))
        ) if len(elite_values) else 0.0
        local_effects = _local_effects(
            perturbations,
            definition.name,
            diagnostic_baseline,
        )
        max_energy_effect = max(
            (item["relative_energy_change"] for item in local_effects),
            default=0.0,
        )
        any_key_effect = any(
            item["comparison_key_changed"] for item in local_effects
        )
        inactive = (
            normalized_variance <= 1e-4
            and not any_key_effect
            and max_energy_effect <= max(float(sensitivity_epsilon), 1e-6)
        )
        inactivity_analysis.append(
            {
                "parameter": definition.name,
                "evidence": {
                    "elite_variance": variance,
                    "normalized_elite_variance": normalized_variance,
                    "elite_near_zero_rate": near_zero_rate,
                    "local_perturbation_effects": local_effects,
                    "inactive": inactive,
                },
                "confidence": _confidence(min(len(elite_values), len(local_effects) + 2)),
                "suggested_structural_action": (
                    ["remove_redundant_feature_or_enable_it_only_under_a_condition"]
                    if inactive or near_zero_rate >= 0.8
                    else ["retain_and_monitor_across_generations"]
                ),
            }
        )

        feasibility_changes = sum(
            bool(item["feasibility_changed"]) for item in local_effects
        )
        energy_fragile = max_energy_effect >= max(
            5.0 * float(sensitivity_epsilon),
            0.1,
        )
        fragile = feasibility_changes > 0 or energy_fragile
        fragility_parameters.append(
            {
                "parameter": definition.name,
                "evidence": {
                    "perturbation_count": len(local_effects),
                    "feasibility_change_count": feasibility_changes,
                    "max_relative_energy_change": max_energy_effect,
                    "energy_fragile": energy_fragile,
                    "local_perturbation_effects": local_effects,
                    "fragile": fragile,
                },
                "confidence": _confidence(len(local_effects), medium=2, high=4),
                "suggested_structural_action": (
                    ["add_smoothing_normalization_or_a_bounded_gate"]
                    if fragile
                    else ["retain_current_local_shape"]
                ),
            }
        )

    elite_matrix = np.asarray(
        [
            [float(entry["parameters"][name]) for name in schema.names]
            for entry in elites
        ],
        dtype=float,
    )
    correlation_matrix: dict[str, dict[str, float | None]] = {
        name: {} for name in schema.names
    }
    flagged_pairs = []
    enough_correlation_samples = elite_matrix.shape[0] >= 4
    if enough_correlation_samples and elite_matrix.shape[1] > 0:
        with np.errstate(invalid="ignore", divide="ignore"):
            matrix = np.corrcoef(elite_matrix, rowvar=False)
        matrix = np.atleast_2d(matrix)
        for left_index, left in enumerate(schema.names):
            for right_index, right in enumerate(schema.names):
                value = float(matrix[left_index, right_index])
                correlation_matrix[left][right] = value if math.isfinite(value) else None
            for right_index in range(left_index + 1, len(schema.names)):
                value = float(matrix[left_index, right_index])
                if math.isfinite(value) and abs(value) >= correlation_threshold:
                    flagged_pairs.append(
                        {
                            "parameters": [left, schema.names[right_index]],
                            "correlation": value,
                            "evidence": {
                                "elite_sample_count": elite_matrix.shape[0],
                                "threshold": correlation_threshold,
                            },
                            "confidence": _confidence(elite_matrix.shape[0]),
                            "suggested_structural_action": [
                                "inspect_redundancy_or_add_an_explicit_interaction",
                                "treat_correlation_as_a_non_causal_structure_hint_only",
                            ],
                        }
                    )
    else:
        for left in schema.names:
            for right in schema.names:
                correlation_matrix[left][right] = None
    correlation_analysis = {
        "evidence": {
            "elite_sample_count": int(elite_matrix.shape[0]),
            "correlation_threshold": float(correlation_threshold),
            "matrix": correlation_matrix,
            "flagged_pairs": flagged_pairs,
            "insufficient_samples": not enough_correlation_samples,
        },
        "confidence": _confidence(int(elite_matrix.shape[0])),
        "suggested_structural_action": (
            ["collect_more_elite_samples_before_structural_change"]
            if not enough_correlation_samples
            else ["review_only_stable_cross_seed_or_cross_generation_correlations"]
        ),
    }

    best_rows = _scenario_rows(diagnostic_baseline)
    baseline_by_context = {
        (
            str(row.get("scenario_id", "unknown")),
            int(row.get("seed", -1)),
        ): row
        for row in best_rows
    }
    scenario_labels = sorted(
        {
            str(row.get("scenario_id", "unknown"))
            for row in best_rows
        }
    )
    scenario_sensitivity = []
    for definition in schema.parameters:
        perturbation_rows = []
        for item in perturbations:
            if item.get("parameter") != definition.name:
                continue
            for row in _scenario_rows(item["metrics"]):
                perturbation_rows.append(
                    {
                        "scenario_id": str(row.get("scenario_id", "unknown")),
                        "seed": int(row.get("seed", -1)),
                        "direction": int(item.get("direction", 0)),
                        "constraint_feasible": bool(row.get("constraint_feasible", False)),
                        "deadline_violation_rate": float(
                            row.get("deadline_violation_rate", float("inf"))
                        ),
                        "fuzzy_energy": float(
                            row.get(
                                "fuzzy_total_energy_score",
                                row.get("objective", float("inf")),
                            )
                        ),
                    }
                )
        enough_scenarios = len(scenario_labels) >= 2
        seed_labels = sorted({row["seed"] for row in perturbation_rows})
        scenario_effects = []
        for scenario_id in scenario_labels:
            rows = [
                row
                for row in perturbation_rows
                if row["scenario_id"] == scenario_id
            ]
            feasibility_changes = 0
            ddl_changes = []
            energy_changes = []
            for row in rows:
                baseline = baseline_by_context.get(
                    (scenario_id, int(row["seed"]))
                )
                if baseline is None:
                    continue
                feasibility_changes += int(
                    bool(row["constraint_feasible"])
                    != bool(baseline.get("constraint_feasible", False))
                )
                ddl_changes.append(
                    abs(
                        float(row["deadline_violation_rate"])
                        - float(
                            baseline.get(
                                "deadline_violation_rate",
                                float("inf"),
                            )
                        )
                    )
                )
                baseline_energy = float(
                    baseline.get(
                        "fuzzy_total_energy_score",
                        baseline.get("objective", float("inf")),
                    )
                )
                energy_changes.append(
                    abs(float(row["fuzzy_energy"]) - baseline_energy)
                    / max(abs(baseline_energy), 1e-12)
                )
            scenario_effects.append(
                {
                    "scenario_id": scenario_id,
                    "sample_count": len(rows),
                    "feasibility_change_count": feasibility_changes,
                    "mean_absolute_ddl_change": (
                        float(np.mean(ddl_changes)) if ddl_changes else 0.0
                    ),
                    "mean_relative_energy_change": (
                        float(np.mean(energy_changes)) if energy_changes else 0.0
                    ),
                }
            )
        feasibility_counts = {
            item["feasibility_change_count"] for item in scenario_effects
        }
        ddl_effects = [
            item["mean_absolute_ddl_change"] for item in scenario_effects
        ]
        energy_effects = [
            item["mean_relative_energy_change"] for item in scenario_effects
        ]
        scenario_sensitive = enough_scenarios and (
            len(feasibility_counts) > 1
            or max(ddl_effects, default=0.0) - min(ddl_effects, default=0.0)
            >= float(sensitivity_epsilon)
            or max(energy_effects, default=0.0) - min(energy_effects, default=0.0)
            >= float(sensitivity_epsilon)
        )
        seed_feasibility = {}
        for row in perturbation_rows:
            baseline = baseline_by_context.get(
                (row["scenario_id"], int(row["seed"]))
            )
            if baseline is None:
                continue
            seed_feasibility.setdefault(int(row["seed"]), set()).add(
                bool(row["constraint_feasible"])
                != bool(baseline.get("constraint_feasible", False))
            )
        seed_sensitive = (
            len(seed_labels) >= 2
            and len(
                {
                    tuple(sorted(values))
                    for values in seed_feasibility.values()
                }
            ) > 1
        )
        scenario_sensitivity.append(
            {
                "parameter": definition.name,
                "evidence": {
                    "scenario_labels": scenario_labels,
                    "seed_labels": seed_labels,
                    "seed_level_perturbations": perturbation_rows,
                    "scenario_effects": scenario_effects,
                    "insufficient_scenarios": not enough_scenarios,
                    "scenario_sensitive": scenario_sensitive,
                    "seed_sensitive": seed_sensitive,
                },
                "confidence": (
                    _confidence(len(perturbation_rows))
                    if enough_scenarios or len(seed_labels) >= 2
                    else "low"
                ),
                "suggested_structural_action": (
                    ["enable_the_feature_only_in_the_sensitive_scenario_with_a_gate"]
                    if scenario_sensitive
                    else ["add_robust_normalization_or_smoothing_across_seeds"]
                    if seed_sensitive
                    else ["collect_cross_scenario_evidence_before_adding_a_scenario_gate"]
                ),
            }
        )

    action_counts: dict[str, int] = {}
    for collection in (
        boundary_analysis,
        inactivity_analysis,
        flagged_pairs,
        scenario_sensitivity,
        fragility_parameters,
    ):
        for item in collection:
            for action in item.get("suggested_structural_action", []):
                action_counts[action] = action_counts.get(action, 0) + 1

    return {
        "schema_version": "parameter_diagnostics_v1",
        "boundary_analysis": boundary_analysis,
        "inactivity_analysis": inactivity_analysis,
        "correlation_analysis": correlation_analysis,
        "scenario_sensitivity": scenario_sensitivity,
        "fragility_analysis": {
            "evidence": {
                "baseline": _metric_snapshot(diagnostic_baseline),
                "parameter_results": fragility_parameters,
                "rule_fragile": any(
                    item["evidence"]["fragile"] for item in fragility_parameters
                ),
            },
            "confidence": _confidence(len(perturbations), medium=2, high=8),
            "suggested_structural_action": (
                ["simplify_or_smooth_fragile_thresholds_and_gates"]
                if any(item["evidence"]["fragile"] for item in fragility_parameters)
                else ["retain_current_local_parameterization"]
            ),
        },
        "summary": {
            "elite_sample_count": len(elites),
            "history_sample_count": len(history),
            "perturbation_count": len(perturbations),
            "real_evidence_count": sum(
                item.get("evaluation_source", "real") == "real"
                for item in perturbations
            ),
            "exact_identical_evidence_count": sum(
                item.get("evaluation_source") == "exact_identical"
                for item in perturbations
            ),
            "baseline_contexts": [
                [scenario, seed]
                for scenario, seed in optimization.diagnostic_baseline_contexts
            ],
            "suggested_action_counts": action_counts,
            "correlation_is_not_causation": True,
            "strong_changes_require_repeated_cross_generation_seed_or_scenario_evidence": True,
        },
    }


def accumulate_cross_generation_diagnostics(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate repeated diagnostic signals for one structure hash."""
    generation_count = len(records)
    signal_counts: dict[tuple[str, str], dict[str, Any]] = {}

    def add_signal(
        signal_type: str,
        identity: str,
        iteration: int,
        actions: Sequence[str],
    ) -> None:
        key = (signal_type, identity)
        entry = signal_counts.setdefault(
            key,
            {
                "signal_type": signal_type,
                "identity": identity,
                "iterations": [],
                "suggested_structural_action": set(),
            },
        )
        entry["iterations"].append(int(iteration))
        entry["suggested_structural_action"].update(str(item) for item in actions)

    for index, record in enumerate(records):
        iteration = int(record.get("iteration", index))
        diagnostics = record.get("diagnostics", record)
        if not isinstance(diagnostics, Mapping):
            continue
        for item in diagnostics.get("boundary_analysis", []):
            if item.get("evidence", {}).get("persistent_boundary_contact"):
                add_signal(
                    "persistent_boundary_contact",
                    str(item.get("parameter", "unknown")),
                    iteration,
                    item.get("suggested_structural_action", []),
                )
        for item in diagnostics.get("inactivity_analysis", []):
            if item.get("evidence", {}).get("inactive"):
                add_signal(
                    "inactive_parameter",
                    str(item.get("parameter", "unknown")),
                    iteration,
                    item.get("suggested_structural_action", []),
                )
        fragility = diagnostics.get("fragility_analysis", {}).get("evidence", {})
        for item in fragility.get("parameter_results", []):
            if item.get("evidence", {}).get("fragile"):
                add_signal(
                    "fragile_parameter",
                    str(item.get("parameter", "unknown")),
                    iteration,
                    item.get("suggested_structural_action", []),
                )
        correlation = diagnostics.get("correlation_analysis", {}).get("evidence", {})
        for item in correlation.get("flagged_pairs", []):
            pair = sorted(str(value) for value in item.get("parameters", []))
            if len(pair) == 2:
                add_signal(
                    "correlated_parameter_pair",
                    "|".join(pair),
                    iteration,
                    item.get("suggested_structural_action", []),
                )
        for item in diagnostics.get("scenario_sensitivity", []):
            evidence = item.get("evidence", {})
            if evidence.get("scenario_sensitive") or evidence.get("seed_sensitive"):
                add_signal(
                    "scenario_or_seed_sensitive_parameter",
                    str(item.get("parameter", "unknown")),
                    iteration,
                    item.get("suggested_structural_action", []),
                )

    stable_signals = []
    all_signals = []
    for entry in signal_counts.values():
        occurrences = len(set(entry["iterations"]))
        rate = float(occurrences / generation_count) if generation_count else 0.0
        payload = {
            "signal_type": entry["signal_type"],
            "identity": entry["identity"],
            "occurrences": occurrences,
            "generation_count": generation_count,
            "occurrence_rate": rate,
            "iterations": sorted(set(entry["iterations"])),
            "confidence": (
                "high" if occurrences >= 3 and rate >= 0.6
                else "medium" if occurrences >= 2
                else "low"
            ),
            "suggested_structural_action": sorted(
                entry["suggested_structural_action"]
            ),
        }
        all_signals.append(payload)
        if occurrences >= 2:
            stable_signals.append(payload)

    return {
        "schema_version": "cross_generation_parameter_diagnostics_v1",
        "generation_count": generation_count,
        "explicit_statistics_available": generation_count >= 2,
        "signals": sorted(
            all_signals,
            key=lambda item: (item["signal_type"], item["identity"]),
        ),
        "stable_signals": sorted(
            stable_signals,
            key=lambda item: (item["signal_type"], item["identity"]),
        ),
        "strong_structural_change_supported": any(
            item["confidence"] in {"medium", "high"}
            for item in stable_signals
        ),
    }
