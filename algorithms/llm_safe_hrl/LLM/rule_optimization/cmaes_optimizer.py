"""Bounded CMA-ES ask/tell optimization with feasibility-first ranking."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from concurrent.futures import ThreadPoolExecutor
import copy
import math
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .parameter_schema import (
    ParameterDefinition,
    ParameterSchema,
    canonical_json_sha256,
)


MetricEvaluator = Callable[[dict[str, float], str, Sequence[int]], Mapping[str, Any]]
BatchMetricEvaluator = Callable[
    [Sequence[dict[str, float]], str, Sequence[int]],
    Sequence[Mapping[str, Any]],
]
ReplayGate = Callable[
    [dict[str, float], Sequence[tuple[str, int]]],
    Mapping[tuple[str, int], Any],
]


@dataclass(frozen=True)
class DiagnosticReplayGateConfig:
    enabled: bool = False
    replay_workers: int = 20
    max_trace_mb_per_structure: int = 128
    max_trace_mb_total: int = 1024
    delete_traces_after_use: bool = True


def _finite(value: Any, default: float = float("inf")) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def constraint_priority_key(
    metrics: Mapping[str, Any],
    *,
    performance_tolerance: float = 0.0,
) -> tuple[float, ...]:
    """Return a strict DDL-first key; no constraint/energy weighted sum is used."""
    feasible = bool(metrics.get("constraint_feasible", False))
    violation = _finite(
        metrics.get(
            "deadline_violation_count",
            metrics.get(
                "max_deadline_violation_rate_across_seeds",
                metrics.get(
                    "deadline_violation_rate",
                    metrics.get("constraint_violation"),
                ),
            ),
        )
    )
    tardiness = _finite(
        metrics.get(
            "total_lateness",
            metrics.get("constraint_secondary_violation"),
        )
    )
    energy = _finite(
        metrics.get(
            "fuzzy_total_energy_score",
            metrics.get("objective", metrics.get("energy")),
        )
    )
    robustness = _finite(
        metrics.get(
            "objective_std_across_seeds",
            metrics.get("objective_cv_across_seeds", 0.0),
        ),
        0.0,
    )
    tolerance = max(0.0, float(performance_tolerance))
    energy_bucket = (
        float(round(energy / tolerance))
        if tolerance > 0.0 and math.isfinite(energy)
        else energy
    )
    if feasible:
        return (0.0, 0.0, 0.0, energy_bucket, robustness, energy)
    return (1.0, violation, tardiness, energy_bucket, robustness, energy)


def rank_fitness(
    results: Sequence[Mapping[str, Any]],
    *,
    performance_tolerance: float = 0.0,
) -> list[float]:
    """Map lexicographic constraint keys to scalar ranks for CMA-ES tell()."""
    keys = [
        constraint_priority_key(
            result,
            performance_tolerance=performance_tolerance,
        )
        for result in results
    ]
    order = sorted(range(len(keys)), key=lambda index: (keys[index], index))
    ranks = [0.0] * len(keys)
    previous_key = None
    previous_rank = 0.0
    for position, index in enumerate(order):
        if previous_key is None or keys[index] != previous_key:
            previous_rank = float(position)
            previous_key = keys[index]
        ranks[index] = previous_rank
    return ranks


@dataclass(frozen=True)
class OptimizerConfig:
    """Central configuration for bounded, staged CMA-ES optimization."""

    enabled: bool = False
    optimizer_seed: int = 0
    max_parameters: int = 12
    population_size: int = 8
    max_generations: int = 6
    initial_sigma: float = 0.25
    stage_seed_counts: Mapping[str, int] = field(
        default_factory=lambda: {"quick": 1, "refine": 3, "confirm": 0}
    )
    elite_fraction: float = 0.25
    boundary_epsilon: float = 1e-6
    sensitivity_epsilon: float = 0.02
    correlation_threshold: float = 0.85
    early_stop_patience: int = 3
    cache_enabled: bool = True
    cache_path: str = "parameter_evaluation_cache.json"
    cache_precision: int = 12
    max_parallel_evaluations: int = 1
    max_branches: int = 6
    max_ast_depth: int = 18
    max_interactions: int = 8
    performance_tolerance: float = 0.0
    diagnostic_perturbations: bool = True
    diagnostic_seed_count: int = 2
    diagnostic_replay_gate: DiagnosticReplayGateConfig = field(
        default_factory=DiagnosticReplayGateConfig
    )
    scenario_ids: Sequence[str] = ()
    auto_admission_enabled: bool = False
    admission_required: bool = True
    admission_config_path: str = ""
    admission_manifest_path: str = ""

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "OptimizerConfig":
        if value is None:
            return cls()
        defaults = cls()
        known = {name for name in cls.__dataclass_fields__}
        kwargs = {name: value[name] for name in known if name in value}
        replay_value = kwargs.get("diagnostic_replay_gate")
        if isinstance(replay_value, Mapping):
            kwargs["diagnostic_replay_gate"] = DiagnosticReplayGateConfig(
                **dict(replay_value)
            )
        config = cls(**kwargs)
        if config.max_parameters < 1:
            raise ValueError("max_parameters must be positive")
        if config.population_size < 2:
            raise ValueError("population_size must be at least 2")
        if config.max_generations < 1:
            raise ValueError("max_generations must be positive")
        if not 0.0 < config.initial_sigma <= 1.0:
            raise ValueError("initial_sigma must be in (0, 1]")
        if not 0.0 < config.elite_fraction <= 1.0:
            raise ValueError("elite_fraction must be in (0, 1]")
        if not 0.0 < config.boundary_epsilon < 0.5:
            raise ValueError("boundary_epsilon must be in (0, 0.5)")
        if not 0.0 < config.sensitivity_epsilon < 0.5:
            raise ValueError("sensitivity_epsilon must be in (0, 0.5)")
        if not 0.0 <= config.correlation_threshold <= 1.0:
            raise ValueError("correlation_threshold must be in [0, 1]")
        if config.early_stop_patience < 1:
            raise ValueError("early_stop_patience must be positive")
        if int(config.max_parallel_evaluations) < 1:
            raise ValueError("max_parallel_evaluations must be positive")
        if int(config.diagnostic_seed_count) < 1:
            raise ValueError("diagnostic_seed_count must be positive")
        replay = config.diagnostic_replay_gate
        if replay.enabled and not config.diagnostic_perturbations:
            raise ValueError("diagnostic replay requires diagnostic_perturbations=true")
        if min(
            int(replay.replay_workers),
            int(replay.max_trace_mb_per_structure),
            int(replay.max_trace_mb_total),
        ) < 1:
            raise ValueError("diagnostic replay worker and trace limits must be positive")
        if replay.max_trace_mb_total < replay.max_trace_mb_per_structure:
            raise ValueError("total trace budget must cover at least one structure")
        scenario_ids = tuple(
            str(item).strip().upper() for item in config.scenario_ids
        )
        if any(not item for item in scenario_ids):
            raise ValueError("scenario_ids must contain non-empty identifiers")
        if len(set(scenario_ids)) != len(scenario_ids):
            raise ValueError("scenario_ids must not contain duplicates")
        object.__setattr__(config, "scenario_ids", scenario_ids)
        counts = dict(config.stage_seed_counts)
        if any(int(counts.get(stage, 0)) < 0 for stage in ("quick", "refine", "confirm")):
            raise ValueError("stage_seed_counts must be non-negative")
        if int(counts.get("quick", 0)) < 1:
            raise ValueError("quick stage must use at least one training seed")
        if int(counts.get("refine", 0)) < 1:
            raise ValueError("refine stage must use at least one training seed")
        del defaults
        return config

    def as_dict(self) -> dict[str, Any]:
        return {
            name: (
                asdict(value)
                if name == "diagnostic_replay_gate"
                else dict(value)
                if isinstance(value, Mapping)
                else list(value)
                if name == "scenario_ids"
                else value
            )
            for name, value in self.__dict__.items()
        }


@dataclass
class OptimizationResult:
    """Complete replayable output of one parameter search."""

    best_parameters: dict[str, float]
    best_metrics: dict[str, Any]
    history: list[dict[str, Any]]
    elite_samples: list[dict[str, Any]]
    local_perturbations: list[dict[str, Any]]
    generations: int
    evaluations: int
    stop_reason: str
    stage_seeds: dict[str, list[int]]
    final_stage: str = ""
    surrogate_stats: dict[str, Any] = field(default_factory=dict)
    diagnostic_baseline_metrics: dict[str, Any] = field(default_factory=dict)
    diagnostic_baseline_contexts: list[tuple[str, int]] = field(default_factory=list)
    replay_gate_stats: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "best_parameters": dict(self.best_parameters),
            "best_metrics": dict(self.best_metrics),
            "history": list(self.history),
            "elite_samples": list(self.elite_samples),
            "local_perturbations": list(self.local_perturbations),
            "generations": self.generations,
            "evaluations": self.evaluations,
            "stop_reason": self.stop_reason,
            "stage_seeds": {key: list(value) for key, value in self.stage_seeds.items()},
            "final_stage": self.final_stage,
        }
        if self.surrogate_stats:
            payload["surrogate_stats"] = dict(self.surrogate_stats)
        if self.diagnostic_baseline_metrics:
            payload["diagnostic_baseline_metrics"] = dict(
                self.diagnostic_baseline_metrics
            )
            payload["diagnostic_baseline_contexts"] = [
                [scenario, seed]
                for scenario, seed in self.diagnostic_baseline_contexts
            ]
        if self.replay_gate_stats:
            payload["replay_gate_stats"] = dict(self.replay_gate_stats)
        return payload


def _unit_from_actual(definition: ParameterDefinition, value: float) -> float:
    lower, upper = definition.lower_bound, definition.upper_bound
    if definition.transform == "log":
        return float((math.log(value) - math.log(lower)) / (math.log(upper) - math.log(lower)))
    if definition.transform == "logit":
        fraction = min(max((value - lower) / (upper - lower), 1e-12), 1.0 - 1e-12)
        logit = math.log(fraction / (1.0 - fraction))
        return float((logit + 12.0) / 24.0)
    return float((value - lower) / (upper - lower))


def _actual_from_unit(definition: ParameterDefinition, unit_value: float) -> float:
    lower, upper = definition.lower_bound, definition.upper_bound
    value = min(max(float(unit_value), 0.0), 1.0)
    if definition.transform == "log":
        return float(math.exp(math.log(lower) + value * (math.log(upper) - math.log(lower))))
    if definition.transform == "logit":
        logit = -12.0 + 24.0 * value
        fraction = 1.0 / (1.0 + math.exp(-logit))
        return float(lower + fraction * (upper - lower))
    return float(lower + value * (upper - lower))


class CMAESOptimizer:
    """Use pycma ask/tell while keeping all scheduling semantics in the evaluator."""

    def __init__(self, config: OptimizerConfig):
        self.config = config

    @staticmethod
    def _import_cma():
        try:
            import cma
        except ImportError as exc:
            raise RuntimeError(
                "CMA-ES parameter optimization is enabled but the 'cma' package is "
                "not installed. Install algorithms/llm_safe_hrl/LLM/requirements.txt."
            ) from exc
        return cma

    @staticmethod
    def _select_seeds(seeds: Sequence[int], count: int, stage: str) -> list[int]:
        available = [int(seed) for seed in seeds]
        if count <= 0:
            return []
        if not available:
            raise ValueError(f"{stage} stage has no available seeds")
        return available[: min(int(count), len(available))]

    @staticmethod
    def _safe_evaluate(
        evaluator: MetricEvaluator,
        parameters: dict[str, float],
        stage: str,
        seeds: Sequence[int],
    ) -> dict[str, Any]:
        """Record abnormal vectors as deterministic worst ranks, never cache them here."""
        try:
            return dict(evaluator(parameters, stage, seeds))
        except Exception as exc:
            worst = 1e300
            return {
                "constraint_feasible": False,
                "deadline_violation_rate": worst,
                "max_deadline_violation_rate_across_seeds": worst,
                "constraint_violation": worst,
                "total_lateness": worst,
                "constraint_secondary_violation": worst,
                "objective": worst,
                "fuzzy_total_energy_score": worst,
                "objective_std_across_seeds": worst,
                "objective_cv_across_seeds": worst,
                "evaluation_error": f"{type(exc).__name__}: {exc}",
                "failed_stage": stage,
                "failed_seeds": [int(seed) for seed in seeds],
                "per_seed_metrics": [],
            }

    @classmethod
    def _safe_evaluate_many(
        cls,
        evaluator: MetricEvaluator,
        batch_evaluator: BatchMetricEvaluator | None,
        parameter_maps: Sequence[dict[str, float]],
        stage: str,
        seeds: Sequence[int],
    ) -> list[dict[str, Any]]:
        """Evaluate a batch while preserving the legacy per-vector fallback."""
        if not parameter_maps:
            return []
        if batch_evaluator is None:
            return [
                cls._safe_evaluate(evaluator, parameters, stage, seeds)
                for parameters in parameter_maps
            ]
        try:
            results = list(batch_evaluator(parameter_maps, stage, seeds))
            if len(results) != len(parameter_maps):
                raise ValueError(
                    "batch evaluator must return one result per parameter vector"
                )
            return [dict(result) for result in results]
        except Exception:
            # Isolate a batch transport failure using the established per-vector
            # deterministic error handling instead of invalidating the whole batch.
            return [
                cls._safe_evaluate(evaluator, parameters, stage, seeds)
                for parameters in parameter_maps
            ]

    def optimize(
        self,
        schema: ParameterSchema,
        evaluator: MetricEvaluator,
        *,
        batch_evaluator: BatchMetricEvaluator | None = None,
        train_seeds: Sequence[int],
        validation_seeds: Sequence[int] = (),
        final_test_seeds: Sequence[int] = (),
        warm_start: Mapping[str, float] | None = None,
        surrogate_manager: Any | None = None,
        surrogate_candidate: Any | None = None,
        surrogate_iteration: int = 0,
        replay_gate: ReplayGate | None = None,
        trace_capture_evaluator: MetricEvaluator | None = None,
        diagnostic_scenario_ids: Sequence[str] = (),
    ) -> OptimizationResult:
        """Run staged search and deterministic local perturbation diagnostics."""
        if not self.config.enabled:
            raise RuntimeError("CMA-ES optimizer was called while disabled")
        if len(schema.parameters) > self.config.max_parameters:
            raise ValueError("parameter schema exceeds configured max_parameters")
        train = [int(seed) for seed in train_seeds]
        validation = [int(seed) for seed in validation_seeds]
        tests = {int(seed) for seed in final_test_seeds}
        if tests.intersection(train) or tests.intersection(validation):
            raise ValueError("final test seeds must not be used for parameter optimization")
        if set(train).intersection(validation):
            raise ValueError("training and validation seeds must be disjoint")
        counts = dict(self.config.stage_seed_counts)
        stage_seeds = {
            "quick": self._select_seeds(train, int(counts.get("quick", 1)), "quick"),
            "refine": self._select_seeds(train, int(counts.get("refine", len(train))), "refine"),
            # confirm=0 means all available held-out validation seeds.
            "confirm": (
                self._select_seeds(
                    validation,
                    int(counts.get("confirm", 0)) or len(validation),
                    "confirm",
                )
                if validation
                else []
            ),
        }
        if (
            self.config.diagnostic_perturbations
            and len(stage_seeds["refine"]) < self.config.diagnostic_seed_count
        ):
            raise ValueError(
                "refine seeds are fewer than diagnostic_seed_count"
            )
        cma = self._import_cma()
        initial_values = dict(
            warm_start
            if warm_start is not None
            else zip(schema.names, schema.initial_values)
        )
        x0 = [
            _unit_from_actual(definition, float(initial_values[definition.name]))
            for definition in schema.parameters
        ]
        epsilon = self.config.boundary_epsilon
        x0 = np.clip(np.asarray(x0, dtype=float), epsilon, 1.0 - epsilon).tolist()
        strategy = cma.CMAEvolutionStrategy(
            x0,
            self.config.initial_sigma,
            {
                "bounds": [epsilon, 1.0 - epsilon],
                "popsize": self.config.population_size,
                "seed": self.config.optimizer_seed,
                "verbose": -9,
                "verb_disp": 0,
                "verb_log": 0,
            },
        )
        history: list[dict[str, Any]] = []
        best_key: tuple[float, ...] | None = None
        current_stage: str | None = None
        no_improvement = 0
        stop_reason = "max_generations"
        generations_completed = 0
        surrogate_active = bool(
            surrogate_manager is not None
            and getattr(surrogate_manager, "enabled", False)
            and surrogate_candidate is not None
        )
        surrogate_stats = {
            "enabled": surrogate_active,
            "gate_generations": 0,
            "fallback_generations": 0,
            "exact_quick_vectors": 0,
            "exact_refine_vectors": 0,
            "predicted_refine_vectors": 0,
            "elite_supplement_vectors": 0,
            "estimated_true_seed_calls": 0,
            "estimated_seed_calls_avoided": 0,
            "failure_reasons": [],
        }
        quick_generations = (
            min(max(1, self.config.max_generations // 3), self.config.max_generations - 1)
            if self.config.max_generations > 1
            else 0
        )

        for generation in range(self.config.max_generations):
            stage = "quick" if generation < quick_generations else "refine"
            if stage != current_stage:
                current_stage = stage
                best_key = None
                no_improvement = 0
            seeds = stage_seeds[stage]
            asked = strategy.ask()
            repaired = [
                np.clip(np.asarray(vector, dtype=float), epsilon, 1.0 - epsilon).tolist()
                for vector in asked
            ]
            parameter_maps = [
                {
                    definition.name: _actual_from_unit(definition, value)
                    for definition, value in zip(schema.parameters, vector)
                }
                for vector in repaired
            ]
            evaluation_sources = ["exact"] * len(parameter_maps)
            selection_reasons = {index: ["exact_stage"] for index in range(len(parameter_maps))}
            if stage == "refine" and surrogate_active:
                # Every vector first receives a real quick observation.  In the
                # production evaluator, the overlapping first refine seed is then
                # served from EvaluationCache rather than executed twice.
                quick_metrics = self._safe_evaluate_many(
                    evaluator,
                    batch_evaluator,
                    parameter_maps,
                    "quick",
                    stage_seeds["quick"],
                )
                surrogate_stats["exact_quick_vectors"] += len(parameter_maps)
                surrogate_stats["estimated_true_seed_calls"] += (
                    len(parameter_maps) * len(stage_seeds["quick"])
                )
                try:
                    decision = surrogate_manager.select_parameters(
                        surrogate_candidate,
                        parameter_maps,
                        quick_metrics,
                        generation=generation,
                        iteration=int(surrogate_iteration),
                    )
                except Exception as exc:  # fail open to full exact refine
                    decision = {
                        "selected_indices": list(range(len(parameter_maps))),
                        "predictions": [None] * len(parameter_maps),
                        "reasons": {
                            index: ["manager_error"]
                            for index in range(len(parameter_maps))
                        },
                        "gate_used": False,
                        "failure_reason": f"{type(exc).__name__}:{exc}",
                    }
                selected = sorted(
                    {
                        int(index)
                        for index in decision.get("selected_indices", [])
                        if 0 <= int(index) < len(parameter_maps)
                    }
                )
                if not selected:
                    selected = list(range(len(parameter_maps)))
                    decision["gate_used"] = False
                    decision["failure_reason"] = "empty_selection"
                if not decision.get("gate_used", False):
                    selected = list(range(len(parameter_maps)))
                    surrogate_stats["fallback_generations"] += 1
                    reason = str(decision.get("failure_reason", "not_ready"))
                    if reason:
                        surrogate_stats["failure_reasons"].append(reason)
                else:
                    surrogate_stats["gate_generations"] += 1
                selected_maps = [parameter_maps[index] for index in selected]
                selected_metrics = self._safe_evaluate_many(
                    evaluator,
                    batch_evaluator,
                    selected_maps,
                    "refine",
                    seeds,
                )
                exact_by_index = dict(zip(selected, selected_metrics))
                metrics = []
                predictions = list(decision.get("predictions", []))
                for index in range(len(parameter_maps)):
                    if index in exact_by_index:
                        result = exact_by_index[index]
                        metrics.append(result)
                        if "evaluation_error" not in result:
                            try:
                                surrogate_manager.record_parameter_exact(
                                    surrogate_candidate,
                                    parameter_maps[index],
                                    quick_metrics[index],
                                    result,
                                    {
                                        "generation": generation,
                                        "candidate_index": index,
                                        "quick_seeds": list(stage_seeds["quick"]),
                                        "refine_seeds": list(seeds),
                                    },
                                )
                            except Exception as exc:
                                surrogate_stats["failure_reasons"].append(
                                    "label_write_error:"
                                    f"{type(exc).__name__}:{exc}"
                                )
                    else:
                        prediction = predictions[index] if index < len(predictions) else None
                        if prediction is None:
                            # A malformed prediction is never allowed to skip work.
                            result = self._safe_evaluate(
                                evaluator,
                                parameter_maps[index],
                                "refine",
                                seeds,
                            )
                            metrics.append(result)
                            exact_by_index[index] = result
                            selected.append(index)
                        else:
                            metrics.append(dict(prediction.metrics))
                            evaluation_sources[index] = "surrogate"
                audit = getattr(surrogate_manager, "record_audit", None)
                if (
                    callable(audit)
                    and decision.get("gate_used", False)
                    and decision.get("audit_generation", False)
                ):
                    exact_indices = sorted(exact_by_index)
                    exact_order = sorted(
                        exact_indices,
                        key=lambda item: constraint_priority_key(
                            exact_by_index[item],
                            performance_tolerance=self.config.performance_tolerance,
                        ),
                    )
                    promising_exact = set(
                        exact_order[: max(1, int(math.ceil(len(exact_order) / 2)))]
                    )
                    would_select = set(
                        int(index)
                        for index in decision.get("would_select_indices", selected)
                    )
                    for index in exact_indices:
                        audit(
                            gate="parameter",
                            actual_promising=index in promising_exact,
                            selected_for_exact=index in would_select,
                        )
                    finish_audit = getattr(surrogate_manager, "finish_audit", None)
                    if callable(finish_audit):
                        finish_audit("parameter")
                selection_reasons = {
                    int(index): list(reasons)
                    for index, reasons in dict(decision.get("reasons", {})).items()
                }
                for index in range(len(parameter_maps)):
                    selection_reasons.setdefault(
                        index,
                        ["surrogate_not_selected"]
                        if evaluation_sources[index] != "exact"
                        else ["exact_fallback"],
                    )
                exact_count = sum(source == "exact" for source in evaluation_sources)
                predicted_count = len(parameter_maps) - exact_count
                surrogate_stats["exact_refine_vectors"] += exact_count
                surrogate_stats["predicted_refine_vectors"] += predicted_count
                # Refine includes the quick seed; production cache reuse means
                # only the remaining seeds launch for selected vectors.
                extra_refine_seeds = max(0, len(seeds) - len(stage_seeds["quick"]))
                surrogate_stats["estimated_true_seed_calls"] += exact_count * extra_refine_seeds
                surrogate_stats["estimated_seed_calls_avoided"] += predicted_count * extra_refine_seeds
            else:
                metrics = self._safe_evaluate_many(
                    evaluator,
                    batch_evaluator,
                    parameter_maps,
                    stage,
                    seeds,
                )
            fitness = rank_fitness(
                metrics,
                performance_tolerance=self.config.performance_tolerance,
            )
            strategy.tell(repaired, fitness)
            keys = [
                constraint_priority_key(
                    item,
                    performance_tolerance=self.config.performance_tolerance,
                )
                for item in metrics
            ]
            generation_best = min(keys)
            for index, (parameters, result, key, rank) in enumerate(
                zip(parameter_maps, metrics, keys, fitness)
            ):
                history.append(
                    {
                        "generation": generation,
                        "candidate_index": index,
                        "stage": stage,
                        "seeds": list(seeds),
                        "parameters": parameters,
                        "metrics": result,
                        "comparison_key": list(key),
                        "rank": float(rank),
                        **(
                            {
                                "evaluation_source": evaluation_sources[index],
                                "surrogate_selection_reasons": selection_reasons.get(index, []),
                                "surrogate_uncertainty": (
                                    dict(predictions[index].uncertainty)
                                    if index < len(predictions)
                                    and predictions[index] is not None
                                    else {}
                                ),
                                "quick_metrics": dict(quick_metrics[index]),
                            }
                            if surrogate_active and stage == "refine"
                            else {}
                        ),
                    }
                )
            generations_completed = generation + 1
            if best_key is None or generation_best < best_key:
                best_key = generation_best
                no_improvement = 0
            else:
                no_improvement += 1
            if (
                stage == "refine"
                and no_improvement >= self.config.early_stop_patience
            ):
                stop_reason = "early_stop_patience"
                break
            if stage == "refine" and strategy.stop():
                stop_reason = "cma_stop:" + ",".join(sorted(strategy.stop()))
                break

        final_search_stage = (
            "refine"
            if any(item["stage"] == "refine" for item in history)
            else "quick"
        )
        final_stage_history = [
            item for item in history if item["stage"] == final_search_stage
        ]
        if surrogate_active and final_search_stage == "refine":
            # Any surrogate-only vector currently ranked inside the possible
            # elite set receives a full exact refine evaluation before best/
            # elite selection.  Predictions guide search but never decide the
            # reported result.
            possible_order = sorted(
                final_stage_history,
                key=lambda item: tuple(item["comparison_key"]),
            )
            possible_elite_count = max(
                1,
                int(math.ceil(len(possible_order) * self.config.elite_fraction)),
            )
            supplement = [
                item
                for item in possible_order[:possible_elite_count]
                if item.get("evaluation_source") == "surrogate"
            ]
            if supplement:
                supplement_results = self._safe_evaluate_many(
                    evaluator,
                    batch_evaluator,
                    [item["parameters"] for item in supplement],
                    "refine",
                    stage_seeds["refine"],
                )
                for item, result in zip(supplement, supplement_results):
                    item["metrics"] = result
                    item["comparison_key"] = list(
                        constraint_priority_key(
                            result,
                            performance_tolerance=self.config.performance_tolerance,
                        )
                    )
                    item["evaluation_source"] = "exact"
                    item.setdefault("surrogate_selection_reasons", []).append(
                        "elite_exact_supplement"
                    )
                    if "evaluation_error" not in result:
                        try:
                            surrogate_manager.record_parameter_exact(
                                surrogate_candidate,
                                item["parameters"],
                                item.get("quick_metrics", {}),
                                result,
                                {
                                    "generation": item["generation"],
                                    "candidate_index": item["candidate_index"],
                                    "reason": "elite_exact_supplement",
                                },
                            )
                        except Exception as exc:
                            surrogate_stats["failure_reasons"].append(
                                "supplement_label_write_error:"
                                f"{type(exc).__name__}:{exc}"
                            )
                extra_refine_seeds = max(
                    0,
                    len(stage_seeds["refine"]) - len(stage_seeds["quick"]),
                )
                surrogate_stats["elite_supplement_vectors"] += len(supplement)
                surrogate_stats["estimated_true_seed_calls"] += (
                    len(supplement) * extra_refine_seeds
                )
                surrogate_stats["estimated_seed_calls_avoided"] -= (
                    len(supplement) * extra_refine_seeds
                )
            final_stage_history = [
                item
                for item in final_stage_history
                if item.get("evaluation_source", "exact") == "exact"
            ]
            if not final_stage_history:
                raise RuntimeError("parameter surrogate produced no exact refine candidate")
        ordered = sorted(
            final_stage_history,
            key=lambda item: tuple(item["comparison_key"]),
        )
        elite_count = max(1, int(math.ceil(len(ordered) * self.config.elite_fraction)))
        elites = ordered[:elite_count]

        confirm_evaluations = 0
        if stage_seeds["confirm"]:
            confirm_candidates = []
            seen = set()
            confirm_limit = max(1, int(math.ceil(self.config.population_size * self.config.elite_fraction)))
            for elite in ordered:
                vector_key = tuple(round(elite["parameters"][name], 12) for name in schema.names)
                if vector_key in seen:
                    continue
                seen.add(vector_key)
                confirm_candidates.append(elite)
                if len(confirm_candidates) >= confirm_limit:
                    break
            confirm_results = self._safe_evaluate_many(
                evaluator,
                batch_evaluator,
                [item["parameters"] for item in confirm_candidates],
                "confirm",
                stage_seeds["confirm"],
            )
            confirmed = []
            for elite, result in zip(confirm_candidates, confirm_results):
                confirmed.append(
                    {
                        **elite,
                        "stage": "confirm",
                        "seeds": list(stage_seeds["confirm"]),
                        "metrics": result,
                        "comparison_key": list(
                            constraint_priority_key(
                                result,
                                performance_tolerance=self.config.performance_tolerance,
                            )
                        ),
                    }
                )
                confirm_evaluations += 1
            if confirmed:
                ordered = sorted(confirmed, key=lambda item: tuple(item["comparison_key"]))
                elites = ordered
                final_search_stage = "confirm"

        best = ordered[0]
        diagnostic_seeds = list(stage_seeds["refine"])[
            : self.config.diagnostic_seed_count
        ]
        scenarios = tuple(str(item) for item in diagnostic_scenario_ids)
        if not scenarios:
            scenarios = tuple(self.config.scenario_ids) or ("unknown",)
        diagnostic_contexts = [
            (scenario_id, int(seed))
            for scenario_id in scenarios
            for seed in diagnostic_seeds
        ]
        diagnostic_baseline_metrics: dict[str, Any] = {}
        replay_gate_stats: dict[str, Any] = {}
        local_perturbations = []
        if self.config.diagnostic_perturbations:
            baseline_evaluator = trace_capture_evaluator or evaluator
            if trace_capture_evaluator is not None:
                diagnostic_baseline_metrics = dict(
                    self._safe_evaluate(
                        baseline_evaluator,
                        best["parameters"],
                        "diagnostic_baseline",
                        diagnostic_seeds,
                    )
                )
            else:
                diagnostic_baseline_metrics = dict(
                    self._safe_evaluate_many(
                        evaluator,
                        batch_evaluator,
                        [best["parameters"]],
                        "diagnostic_baseline",
                        diagnostic_seeds,
                    )[0]
                )
            perturbation_requests = []
            for definition in schema.parameters:
                center = float(best["parameters"][definition.name])
                delta = self.config.sensitivity_epsilon * (
                    definition.upper_bound - definition.lower_bound
                )
                for direction in (-1, 1):
                    perturbed = dict(best["parameters"])
                    perturbed[definition.name] = min(
                        definition.upper_bound,
                        max(definition.lower_bound, center + direction * delta),
                    )
                    perturbation_requests.append(
                        (definition, direction, center, perturbed)
                    )
            use_replay = replay_gate is not None and trace_capture_evaluator is not None
            replay_results: list[Mapping[tuple[str, int], Any] | None] = [
                None
            ] * len(perturbation_requests)
            if use_replay:
                replay_started = time.perf_counter()
                workers = min(
                    len(perturbation_requests),
                    self.config.diagnostic_replay_gate.replay_workers,
                )
                with ThreadPoolExecutor(
                    max_workers=max(1, workers),
                    thread_name_prefix="diagnostic-replay",
                ) as executor:
                    replay_results = list(
                        executor.map(
                            lambda request: replay_gate(
                                request[3],
                                diagnostic_contexts,
                            ),
                            perturbation_requests,
                        )
                    )
                replay_wall_seconds = time.perf_counter() - replay_started
            else:
                replay_wall_seconds = 0.0
            skipped = []
            real = []
            for index, verdicts in enumerate(replay_results):
                context_verdicts = (
                    [verdicts.get(context) for context in diagnostic_contexts]
                    if verdicts is not None
                    else []
                )
                baseline_hashes = {
                    verdict.baseline_frozen_rule_hash
                    for verdict in context_verdicts
                    if verdict is not None and verdict.baseline_frozen_rule_hash
                }
                candidate_hashes = {
                    verdict.candidate_frozen_rule_hash
                    for verdict in context_verdicts
                    if verdict is not None and verdict.candidate_frozen_rule_hash
                }
                if (
                    verdicts is not None
                    and len(verdicts) == len(diagnostic_contexts)
                    and len(baseline_hashes) == 1
                    and len(candidate_hashes) == 1
                    and all(
                        verdict is not None
                        and verdict.status == "ok"
                        and verdict.identical is True
                        for verdict in context_verdicts
                    )
                ):
                    skipped.append(index)
                else:
                    real.append(index)
            real_results = self._safe_evaluate_many(
                evaluator,
                batch_evaluator,
                [perturbation_requests[index][3] for index in real],
                "diagnostic",
                diagnostic_seeds,
            ) if real else []
            results_by_index = dict(zip(real, real_results))
            for index in skipped:
                verdicts = replay_results[index] or {}
                derived = copy.deepcopy(diagnostic_baseline_metrics)
                candidate_hashes = {
                    verdict.candidate_frozen_rule_hash
                    for verdict in verdicts.values()
                }
                trace_hashes = sorted(
                    verdict.trace_sha256
                    for verdict in verdicts.values()
                    if verdict.trace_sha256
                )
                if len(candidate_hashes) == 1:
                    candidate_hash = next(iter(candidate_hashes))
                    derived["candidate_sha256"] = candidate_hash
                    for row in derived.get("per_seed_metrics", []):
                        if isinstance(row, dict):
                            row["candidate_sha256"] = candidate_hash
                derived["evidence_source"] = "exact_identical"
                derived["baseline_evaluation_hash"] = canonical_json_sha256(
                    diagnostic_baseline_metrics
                )
                derived["trace_hashes"] = trace_hashes
                results_by_index[index] = derived
            perturbation_results = [
                results_by_index[index]
                for index in range(len(perturbation_requests))
            ]
            if use_replay:
                all_verdicts = [
                    verdict
                    for verdicts in replay_results
                    if verdicts is not None
                    for verdict in verdicts.values()
                ]
                replay_gate_stats = {
                    "enabled": True,
                    "context_judgement_count": len(all_verdicts),
                    "context_identical_count": sum(
                        verdict.status == "ok" and verdict.identical
                        for verdict in all_verdicts
                    ),
                    "vector_count": len(perturbation_requests),
                    "skipped_vector_count": len(skipped),
                    "skipped_episode_count": len(skipped) * len(diagnostic_contexts),
                    "real_vector_count": len(real),
                    "rule_error_context_count": sum(
                        verdict.status == "rule_error" for verdict in all_verdicts
                    ),
                    "trace_mismatch_context_count": sum(
                        verdict.status == "trace_mismatch" for verdict in all_verdicts
                    ),
                    "replay_cpu_seconds": sum(
                        verdict.replay_seconds for verdict in all_verdicts
                    ),
                    "replay_wall_seconds": replay_wall_seconds,
                }
            for request, result in zip(
                perturbation_requests,
                perturbation_results,
            ):
                definition, direction, center, perturbed = request
                local_perturbations.append(
                    {
                        "parameter": definition.name,
                        "direction": direction,
                        "delta": perturbed[definition.name] - center,
                        "parameters": perturbed,
                        "metrics": result,
                        "comparison_key": list(
                            constraint_priority_key(
                                result,
                                performance_tolerance=self.config.performance_tolerance,
                            )
                        ),
                        "seeds": list(diagnostic_seeds),
                        "evaluation_source": result.get(
                            "evidence_source",
                            "real" if use_replay else "exact",
                        ),
                    }
                )

        if surrogate_active:
            manager_stats = getattr(surrogate_manager, "stats", None)
            if callable(manager_stats):
                surrogate_stats["manager"] = dict(manager_stats())
            surrogate_stats["reused_quick_seed_evaluations"] = (
                (
                    surrogate_stats["exact_refine_vectors"]
                    + surrogate_stats["elite_supplement_vectors"]
                )
                * min(len(stage_seeds["quick"]), len(stage_seeds["refine"]))
            )
            surrogate_stats["true_seed_level_evaluations"] = surrogate_stats[
                "estimated_true_seed_calls"
            ]
            surrogate_stats["estimated_seed_level_evaluations_saved"] = (
                surrogate_stats["estimated_seed_calls_avoided"]
            )
        return OptimizationResult(
            best_parameters=dict(best["parameters"]),
            best_metrics=dict(best["metrics"]),
            history=history,
            elite_samples=elites,
            local_perturbations=local_perturbations,
            generations=generations_completed,
            evaluations=(
                len(history) + confirm_evaluations + len(local_perturbations)
            ),
            stop_reason=stop_reason,
            stage_seeds=stage_seeds,
            final_stage=final_search_stage,
            surrogate_stats=surrogate_stats if surrogate_active else {},
            diagnostic_baseline_metrics=diagnostic_baseline_metrics,
            diagnostic_baseline_contexts=diagnostic_contexts,
            replay_gate_stats=replay_gate_stats,
        )
