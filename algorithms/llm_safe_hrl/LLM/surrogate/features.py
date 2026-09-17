"""Fixed-size, AST-only features for structure and parameter surrogates."""

from __future__ import annotations

import ast
import math
from typing import Any, Mapping

import numpy as np

try:
    from rule_optimization.parameter_schema import PRIORITY_ARGUMENTS, RuleCandidate
except ImportError:  # pragma: no cover - package import layout
    from ..rule_optimization.parameter_schema import PRIORITY_ARGUMENTS, RuleCandidate


MAX_PARAMETERS = 12
STRUCTURE_FEATURE_NAMES = (
    "ast_node_count",
    "ast_depth",
    "branch_count",
    "interaction_count",
    *(f"uses_{name}" for name in PRIORITY_ARGUMENTS),
    "nonlinear_operation_count",
    "gate_operation_count",
    "parameter_count",
    "identity_parameter_count",
    "log_parameter_count",
    "logit_parameter_count",
)
QUICK_FEATURE_NAMES = (
    "quick_feasible",
    "quick_violation",
    "quick_lateness",
    "quick_energy",
    "quick_robustness",
)
PARAMETER_FEATURE_NAMES = (
    *STRUCTURE_FEATURE_NAMES,
    *(f"parameter_{index}_normalized" for index in range(MAX_PARAMETERS)),
    *(f"parameter_{index}_present" for index in range(MAX_PARAMETERS)),
    *QUICK_FEATURE_NAMES,
)


def _finite(value: Any, default: float = 1e12) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(number):
        return float(default)
    return float(min(max(number, -1e12), 1e12))


def _depth(node: ast.AST) -> int:
    children = list(ast.iter_child_nodes(node))
    return 1 if not children else 1 + max(_depth(child) for child in children)


def extract_structure_features(candidate: RuleCandidate) -> np.ndarray:
    """Extract static features without importing or executing candidate source."""
    tree = ast.parse(candidate.parameterized_rule_source)
    nodes = list(ast.walk(tree))
    used_names = {node.id for node in nodes if isinstance(node, ast.Name)}
    nonlinear_names = {
        "abs", "exp", "log", "log1p", "sqrt", "power", "tanh", "sigmoid"
    }
    gate_names = {"where", "minimum", "maximum", "clip"}
    nonlinear = 0
    gate = 0
    for node in nodes:
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            nonlinear += 1
        if isinstance(node, ast.Call):
            name = (
                node.func.id
                if isinstance(node.func, ast.Name)
                else node.func.attr
                if isinstance(node.func, ast.Attribute)
                else ""
            )
            nonlinear += int(name in nonlinear_names)
            gate += int(name in gate_names)
        gate += int(isinstance(node, (ast.If, ast.IfExp, ast.Compare, ast.BoolOp)))
    complexity = candidate.complexity or {}
    schema = candidate.parameter_schema
    parameters = tuple(schema.parameters) if schema is not None else ()
    transforms = [definition.transform for definition in parameters]
    values = [
        _finite(complexity.get("ast_node_count", len(nodes))),
        _finite(complexity.get("ast_depth", _depth(tree))),
        _finite(complexity.get("branch_count", 0)),
        _finite(complexity.get("interaction_count", 0)),
        *(float(name in used_names) for name in PRIORITY_ARGUMENTS),
        float(nonlinear),
        float(gate),
        float(len(parameters)),
        float(transforms.count("identity")),
        float(transforms.count("log")),
        float(transforms.count("logit")),
    ]
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (len(STRUCTURE_FEATURE_NAMES),) or not np.isfinite(result).all():
        raise ValueError("structure surrogate features are not finite and fixed-size")
    return result


def _normalized_parameter(definition, value: Any) -> float:
    number = _finite(value, definition.initial_value)
    lower = float(definition.lower_bound)
    upper = float(definition.upper_bound)
    if definition.transform == "log":
        number = min(max(number, lower), upper)
        return (math.log(number) - math.log(lower)) / (math.log(upper) - math.log(lower))
    if definition.transform == "logit":
        fraction = min(max((number - lower) / (upper - lower), 1e-12), 1.0 - 1e-12)
        return min(max((math.log(fraction / (1.0 - fraction)) + 12.0) / 24.0, 0.0), 1.0)
    return min(max((number - lower) / (upper - lower), 0.0), 1.0)


def extract_parameter_features(
    candidate: RuleCandidate,
    parameters: Mapping[str, float],
    quick_metrics: Mapping[str, Any],
) -> np.ndarray:
    schema = candidate.parameter_schema
    if schema is None:
        raise ValueError("parameter features require a parameter schema")
    normalized = np.zeros(MAX_PARAMETERS, dtype=np.float64)
    present = np.zeros(MAX_PARAMETERS, dtype=np.float64)
    for index, definition in enumerate(schema.parameters[:MAX_PARAMETERS]):
        normalized[index] = _normalized_parameter(
            definition,
            parameters.get(definition.name, definition.initial_value),
        )
        present[index] = 1.0
    violation = quick_metrics.get(
        "deadline_violation_count",
        quick_metrics.get(
            "max_deadline_violation_rate_across_seeds",
            quick_metrics.get("deadline_violation_rate", quick_metrics.get("constraint_violation")),
        ),
    )
    quick = np.asarray(
        [
            float(bool(quick_metrics.get("constraint_feasible", False))),
            _finite(violation),
            _finite(quick_metrics.get("total_lateness", quick_metrics.get("constraint_secondary_violation"))),
            _finite(quick_metrics.get("fuzzy_total_energy_score", quick_metrics.get("objective", quick_metrics.get("energy")))),
            _finite(quick_metrics.get("objective_std_across_seeds", quick_metrics.get("objective_cv_across_seeds", 0.0)), 0.0),
        ],
        dtype=np.float64,
    )
    result = np.concatenate((extract_structure_features(candidate), normalized, present, quick))
    if result.shape != (len(PARAMETER_FEATURE_NAMES),) or not np.isfinite(result).all():
        raise ValueError("parameter surrogate features are not finite and fixed-size")
    return result
