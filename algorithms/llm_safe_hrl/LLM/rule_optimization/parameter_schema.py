"""Safe representation, validation, hashing, and freezing of parameterized rules."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Mapping, Sequence


PARAMETER_SCHEMA_NAME = "PARAMETER_SCHEMA"
PARAMETER_CONTAINER_NAME = "PARAMS"
RULE_METADATA_NAME = "RULE_METADATA"
_ALLOWED_RULE_METADATA_FIELDS = frozenset(
    {
        "structure_hash",
        "parameter_schema_hash",
        "best_parameter_hash",
        "best_parameters",
        "optimizer_config_hash",
        "parameter_diagnostics_hash",
        "optimizer_seed",
        "training_seeds",
        "validation_seeds",
    }
)
PRIORITY_FUNCTION_NAME = "get_task_priority_v2"
PRIORITY_ARGUMENTS = (
    "min_exec_time",
    "min_comm_time",
    "min_incremental_energy",
    "slack",
    "upward_rank",
    "remaining_work",
    "ready_wait_time",
    "uncertainty",
)

_ALLOWED_IMPORTS = frozenset({"numpy"})
_FORBIDDEN_NODES = (
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Delete,
    ast.For,
    ast.GeneratorExp,
    ast.Global,
    ast.Nonlocal,
    ast.Try,
    ast.While,
    ast.With,
    ast.AsyncWith,
    ast.Await,
    ast.Yield,
    ast.YieldFrom,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
)
_FORBIDDEN_CALL_NAMES = frozenset(
    {
        "__import__",
        "breakpoint",
        "compile",
        "eval",
        "exec",
        "globals",
        "getattr",
        "input",
        "locals",
        "open",
        "print",
        "setattr",
        "vars",
    }
)
_FORBIDDEN_ATTRIBUTE_NAMES = frozenset(
    {
        "connect",
        "dump",
        "dumps",
        "load",
        "loads",
        "open",
        "popen",
        "read",
        "remove",
        "rename",
        "replace",
        "system",
        "unlink",
        "write",
    }
)
_ALLOWED_STRUCTURAL_NUMBERS = frozenset({-2.0, -1.0, 0.0, 1.0, 2.0})


class RuleValidationError(ValueError):
    """Raised when an LLM rule violates the parameter or safety contract."""


def canonical_json_sha256(value: Any) -> str:
    """Hash a JSON-compatible value using a deterministic representation."""
    payload = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _finite_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise RuleValidationError(f"{field_name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RuleValidationError(f"{field_name} must be a finite number") from exc
    if not math.isfinite(result):
        raise RuleValidationError(f"{field_name} must be finite")
    return result


@dataclass(frozen=True)
class ParameterDefinition:
    """One bounded continuous parameter exposed to CMA-ES."""

    name: str
    initial_value: float
    lower_bound: float
    upper_bound: float
    semantic_description: str
    parameter_type: str = "float"
    transform: str = "identity"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ParameterDefinition":
        required = {
            "name",
            "initial_value",
            "lower_bound",
            "upper_bound",
            "semantic_description",
        }
        missing = sorted(required.difference(value))
        if missing:
            raise RuleValidationError(
                "parameter definition is missing: " + ", ".join(missing)
            )
        name = str(value["name"]).strip()
        if not name.isidentifier() or name.startswith("_"):
            raise RuleValidationError(f"invalid parameter name: {name!r}")
        initial = _finite_number(value["initial_value"], f"{name}.initial_value")
        lower = _finite_number(value["lower_bound"], f"{name}.lower_bound")
        upper = _finite_number(value["upper_bound"], f"{name}.upper_bound")
        if not lower < upper:
            raise RuleValidationError(
                f"{name}.lower_bound must be less than upper_bound"
            )
        if not lower <= initial <= upper:
            raise RuleValidationError(
                f"{name}.initial_value must lie inside its bounds"
            )
        description = str(value["semantic_description"]).strip()
        if not description:
            raise RuleValidationError(
                f"{name}.semantic_description must not be empty"
            )
        parameter_type = str(value.get("parameter_type", "float")).strip()
        transform = str(value.get("transform", "identity")).strip()
        if parameter_type != "float":
            raise RuleValidationError(
                f"{name}.parameter_type must be 'float' for CMA-ES"
            )
        if transform not in {"identity", "log", "logit"}:
            raise RuleValidationError(
                f"{name}.transform must be identity, log, or logit"
            )
        if transform == "log" and lower <= 0.0:
            raise RuleValidationError(
                f"{name}.lower_bound must be positive for log transform"
            )
        return cls(
            name=name,
            initial_value=initial,
            lower_bound=lower,
            upper_bound=upper,
            semantic_description=description,
            parameter_type=parameter_type,
            transform=transform,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "initial_value": self.initial_value,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "semantic_description": self.semantic_description,
            "parameter_type": self.parameter_type,
            "transform": self.transform,
        }


@dataclass(frozen=True)
class ParameterSchema:
    """Validated ordered parameter schema embedded in an LLM rule module."""

    parameters: tuple[ParameterDefinition, ...]
    schema_version: str = "rule_parameters_v1"

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        max_parameters: int = 12,
    ) -> "ParameterSchema":
        if "parameters" not in value:
            raise RuleValidationError("PARAMETER_SCHEMA is missing parameters")
        raw_parameters = value["parameters"]
        if not isinstance(raw_parameters, (list, tuple)):
            raise RuleValidationError("PARAMETER_SCHEMA.parameters must be a list")
        if not raw_parameters:
            raise RuleValidationError("PARAMETER_SCHEMA.parameters must not be empty")
        if len(raw_parameters) > int(max_parameters):
            raise RuleValidationError(
                f"parameter count {len(raw_parameters)} exceeds limit {max_parameters}"
            )
        parsed = []
        for item in raw_parameters:
            if not isinstance(item, Mapping):
                raise RuleValidationError("each parameter definition must be a mapping")
            parsed.append(ParameterDefinition.from_mapping(item))
        names = [item.name for item in parsed]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise RuleValidationError(
                "duplicate parameter name(s): " + ", ".join(duplicates)
            )
        version = str(value.get("schema_version", "rule_parameters_v1")).strip()
        if not version:
            raise RuleValidationError("parameter schema_version must not be empty")
        return cls(tuple(parsed), version)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.parameters)

    @property
    def initial_values(self) -> list[float]:
        return [item.initial_value for item in self.parameters]

    @property
    def lower_bounds(self) -> list[float]:
        return [item.lower_bound for item in self.parameters]

    @property
    def upper_bounds(self) -> list[float]:
        return [item.upper_bound for item in self.parameters]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "parameters": [item.as_dict() for item in self.parameters],
        }

    @property
    def schema_hash(self) -> str:
        return canonical_json_sha256(self.as_dict())

    def values_dict(self, values: Sequence[float]) -> dict[str, float]:
        if len(values) != len(self.parameters):
            raise RuleValidationError(
                f"expected {len(self.parameters)} values, received {len(values)}"
            )
        result = {}
        for definition, value in zip(self.parameters, values):
            number = _finite_number(value, definition.name)
            if not definition.lower_bound <= number <= definition.upper_bound:
                raise RuleValidationError(
                    f"{definition.name}={number} is outside declared bounds"
                )
            result[definition.name] = number
        return result


@dataclass
class RuleCandidate:
    """Rule structure plus its schema, optimized parameters, and frozen artifact."""

    structure_id: str
    parameterized_rule_source: str
    parameter_schema: ParameterSchema | None
    best_parameters: dict[str, float] = field(default_factory=dict)
    frozen_rule_source: str = ""
    evaluation_result: dict[str, Any] = field(default_factory=dict)
    parameter_diagnostics: dict[str, Any] = field(default_factory=dict)
    optimizer_config: dict[str, Any] = field(default_factory=dict)
    optimizer_seed: int | None = None
    final_code_hash: str = ""
    parameter_hash: str = ""
    complexity: dict[str, int] = field(default_factory=dict)

    @property
    def structure_hash(self) -> str:
        return self.structure_id

    @property
    def parameter_schema_hash(self) -> str:
        if self.parameter_schema is None:
            return ""
        return self.parameter_schema.schema_hash

    def as_dict(self) -> dict[str, Any]:
        return {
            "structure_id": self.structure_id,
            "structure_hash": self.structure_hash,
            "parameterized_rule_source": self.parameterized_rule_source,
            "parameter_schema": (
                self.parameter_schema.as_dict()
                if self.parameter_schema is not None
                else None
            ),
            "best_parameters": dict(self.best_parameters),
            "frozen_rule_source": self.frozen_rule_source,
            "evaluation_result": dict(self.evaluation_result),
            "parameter_diagnostics": dict(self.parameter_diagnostics),
            "optimizer_config": dict(self.optimizer_config),
            "optimizer_seed": self.optimizer_seed,
            "final_code_hash": self.final_code_hash,
            "parameter_hash": self.parameter_hash,
            "parameter_schema_hash": self.parameter_schema_hash,
            "complexity": dict(self.complexity),
        }


def _assignment_literal(tree: ast.Module, name: str) -> tuple[Any, ast.Assign] | None:
    matches = []
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == name:
            matches.append(node)
    if not matches:
        return None
    if len(matches) > 1:
        raise RuleValidationError(f"{name} must be assigned exactly once")
    try:
        return ast.literal_eval(matches[0].value), matches[0]
    except (TypeError, ValueError) as exc:
        raise RuleValidationError(f"{name} must be an AST literal") from exc


def _priority_function(tree: ast.Module) -> ast.FunctionDef:
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == PRIORITY_FUNCTION_NAME
    ]
    if len(matches) != 1:
        raise RuleValidationError(
            f"source must define exactly one {PRIORITY_FUNCTION_NAME} function"
        )
    return matches[0]


def _validate_signature(function: ast.FunctionDef) -> None:
    args = function.args
    names = tuple(argument.arg for argument in args.args)
    if (
        names != PRIORITY_ARGUMENTS
        or args.posonlyargs
        or args.kwonlyargs
        or args.vararg is not None
        or args.kwarg is not None
        or args.defaults
        or args.kw_defaults
    ):
        raise RuleValidationError(
            f"{PRIORITY_FUNCTION_NAME} must keep the established eight-argument signature"
        )


def _ast_depth(node: ast.AST) -> int:
    children = list(ast.iter_child_nodes(node))
    return 1 if not children else 1 + max(_ast_depth(child) for child in children)


def _contains_feature(node: ast.AST) -> bool:
    return any(
        isinstance(child, ast.Name) and child.id in PRIORITY_ARGUMENTS
        for child in ast.walk(node)
    )


def _complexity(function: ast.FunctionDef) -> dict[str, int]:
    branch_count = sum(
        isinstance(node, (ast.If, ast.IfExp)) for node in ast.walk(function)
    )
    interaction_count = sum(
        isinstance(node, ast.BinOp)
        and isinstance(node.op, (ast.Mult, ast.Div, ast.Pow))
        and _contains_feature(node.left)
        and _contains_feature(node.right)
        for node in ast.walk(function)
    )
    return {
        "branch_count": int(branch_count),
        "ast_depth": int(_ast_depth(function)),
        "interaction_count": int(interaction_count),
        "ast_node_count": int(sum(1 for _ in ast.walk(function))),
    }


def _validate_imports_and_safety(tree: ast.Module, function: ast.FunctionDef) -> None:
    for node in tree.body:
        if isinstance(node, ast.Import):
            if any(alias.name not in _ALLOWED_IMPORTS for alias in node.names):
                raise RuleValidationError("only numpy imports are allowed")
        elif isinstance(node, ast.ImportFrom):
            if node.module not in _ALLOWED_IMPORTS:
                raise RuleValidationError("only numpy imports are allowed")
        elif isinstance(node, ast.FunctionDef):
            continue
        elif isinstance(node, ast.Assign):
            if (
                len(node.targets) != 1
                or not isinstance(node.targets[0], ast.Name)
                or node.targets[0].id
                not in {PARAMETER_SCHEMA_NAME, RULE_METADATA_NAME}
            ):
                raise RuleValidationError(
                    "module-level assignments are limited to literal rule metadata"
                )
            try:
                ast.literal_eval(node.value)
            except (TypeError, ValueError) as exc:
                raise RuleValidationError("module-level metadata must be literal") from exc
        elif (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            continue
        else:
            raise RuleValidationError(
                f"unsupported module-level statement: {type(node).__name__}"
            )

    for node in ast.walk(function):
        if isinstance(node, _FORBIDDEN_NODES):
            raise RuleValidationError(
                f"forbidden rule construct: {type(node).__name__}"
            )
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == function.name:
                raise RuleValidationError("recursive priority rules are forbidden")
            if isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_CALL_NAMES:
                raise RuleValidationError(f"forbidden call: {node.func.id}")
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr.lower() in _FORBIDDEN_ATTRIBUTE_NAMES
            ):
                raise RuleValidationError(f"forbidden call: {node.func.attr}")

    class _RecursionVisitor(ast.NodeVisitor):
        def __init__(self):
            self.function_stack: list[str] = []

        def visit_FunctionDef(self, node: ast.FunctionDef):  # noqa: N802
            self.function_stack.append(node.name)
            for statement in node.body:
                self.visit(statement)
            self.function_stack.pop()

        def visit_Call(self, node: ast.Call):  # noqa: N802
            if (
                self.function_stack
                and isinstance(node.func, ast.Name)
                and node.func.id == self.function_stack[-1]
            ):
                raise RuleValidationError("recursive priority rules are forbidden")
            self.generic_visit(node)

    _RecursionVisitor().visit(function)


def _parameter_reference_name(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Subscript):
        return None
    if not isinstance(node.value, ast.Name) or node.value.id != PARAMETER_CONTAINER_NAME:
        return None
    slice_node = node.slice
    if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, str):
        return slice_node.value
    return None


def _validate_parameter_references(
    function: ast.FunctionDef,
    schema: ParameterSchema,
) -> None:
    declared = set(schema.names)
    referenced = {
        name
        for node in ast.walk(function)
        if (name := _parameter_reference_name(node)) is not None
    }
    unknown = sorted(referenced.difference(declared))
    unused = sorted(declared.difference(referenced))
    if unknown:
        raise RuleValidationError(
            "rule references undeclared parameter(s): " + ", ".join(unknown)
        )
    if unused:
        raise RuleValidationError(
            "schema declares unused parameter(s): " + ", ".join(unused)
        )
    for node in ast.walk(function):
        if isinstance(node, ast.Name) and node.id == PARAMETER_CONTAINER_NAME:
            parent_is_valid = any(
                _parameter_reference_name(candidate) is not None
                and candidate.value is node
                for candidate in ast.walk(function)
                if isinstance(candidate, ast.Subscript)
            )
            if not parent_is_valid:
                raise RuleValidationError("PARAMS may only be accessed as PARAMS['name']")

    for node in ast.walk(function):
        if not isinstance(node, ast.Constant):
            continue
        value = node.value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if float(value) not in _ALLOWED_STRUCTURAL_NUMBERS:
            raise RuleValidationError(
                f"hidden numeric constant {value!r}; declare it in PARAMETER_SCHEMA"
            )


class _StructureNormalizer(ast.NodeTransformer):
    def visit_Constant(self, node: ast.Constant):  # noqa: N802
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            return node
        return ast.copy_location(ast.Constant(value=0), node)


def _structure_hash(tree: ast.Module, schema_assignment: ast.Assign | None) -> str:
    copied = ast.parse(ast.unparse(tree))
    copied.body = [
        node
        for node in copied.body
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in {PARAMETER_SCHEMA_NAME, RULE_METADATA_NAME}
        )
    ]
    normalized = _StructureNormalizer().visit(copied)
    ast.fix_missing_locations(normalized)
    return hashlib.sha256(
        ast.dump(normalized, annotate_fields=True, include_attributes=False).encode(
            "utf-8"
        )
    ).hexdigest()


def parse_rule_candidate(
    source: str,
    *,
    max_parameters: int = 12,
    max_branches: int = 6,
    max_ast_depth: int = 18,
    max_interactions: int = 8,
) -> RuleCandidate:
    """Parse a parameterized or legacy rule without executing its source."""
    if not isinstance(source, str) or not source.strip():
        raise RuleValidationError("rule source must be non-empty text")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise RuleValidationError(f"invalid Python source: {exc}") from exc
    function = _priority_function(tree)
    complexity = _complexity(function)

    raw_schema = _assignment_literal(tree, PARAMETER_SCHEMA_NAME)
    schema = None
    schema_assignment = None
    if raw_schema is not None:
        _validate_imports_and_safety(tree, function)
        if complexity["branch_count"] > int(max_branches):
            raise RuleValidationError("rule exceeds maximum branch count")
        if complexity["ast_depth"] > int(max_ast_depth):
            raise RuleValidationError("rule exceeds maximum AST depth")
        if complexity["interaction_count"] > int(max_interactions):
            raise RuleValidationError("rule exceeds maximum interaction count")
        raw_value, schema_assignment = raw_schema
        if not isinstance(raw_value, Mapping):
            raise RuleValidationError("PARAMETER_SCHEMA must be a mapping literal")
        schema = ParameterSchema.from_mapping(
            raw_value,
            max_parameters=max_parameters,
        )
        _validate_signature(function)
        _validate_parameter_references(function, schema)

    return RuleCandidate(
        structure_id=_structure_hash(tree, schema_assignment),
        parameterized_rule_source=source.strip(),
        parameter_schema=schema,
        complexity=complexity,
    )


class _ParameterFreezer(ast.NodeTransformer):
    def __init__(self, values: Mapping[str, float]):
        self.values = values

    def visit_Subscript(self, node: ast.Subscript):  # noqa: N802
        name = _parameter_reference_name(node)
        if name is None:
            return self.generic_visit(node)
        if name not in self.values:
            raise RuleValidationError(f"no frozen value supplied for {name}")
        return ast.copy_location(ast.Constant(value=float(self.values[name])), node)


def _metadata_assignment(metadata: Mapping[str, Any]) -> ast.Assign:
    expression = ast.parse(repr(dict(metadata)), mode="eval").body
    return ast.Assign(
        targets=[ast.Name(id=RULE_METADATA_NAME, ctx=ast.Store())],
        value=expression,
    )


def freeze_rule_source(
    source: str,
    schema: ParameterSchema,
    values: Sequence[float] | Mapping[str, float],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> str:
    """Replace declared PARAMS references with finite literals using AST nodes."""
    if isinstance(values, Mapping):
        ordered = [values[name] for name in schema.names]
    else:
        ordered = list(values)
    value_map = schema.values_dict(ordered)
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise RuleValidationError(f"invalid Python source: {exc}") from exc
    parsed = _assignment_literal(tree, PARAMETER_SCHEMA_NAME)
    if parsed is None:
        raise RuleValidationError("cannot freeze a rule without PARAMETER_SCHEMA")
    function = _priority_function(tree)
    _validate_signature(function)
    transformed = _ParameterFreezer(value_map).visit(tree)
    transformed.body = [
        node
        for node in transformed.body
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in {PARAMETER_SCHEMA_NAME, RULE_METADATA_NAME}
        )
    ]
    if metadata:
        insert_at = 0
        while insert_at < len(transformed.body) and isinstance(
            transformed.body[insert_at], (ast.Import, ast.ImportFrom)
        ):
            insert_at += 1
        transformed.body.insert(insert_at, _metadata_assignment(metadata))
    ast.fix_missing_locations(transformed)
    frozen = ast.unparse(transformed).strip() + "\n"
    validate_frozen_rule_source(frozen, require_metadata=bool(metadata))
    return frozen


def extract_rule_metadata(source: str) -> dict[str, Any]:
    """Read literal frozen-rule metadata without importing or executing code."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise RuleValidationError(f"invalid Python source: {exc}") from exc
    parsed = _assignment_literal(tree, RULE_METADATA_NAME)
    if parsed is None:
        return {}
    value, _assignment = parsed
    if not isinstance(value, Mapping):
        raise RuleValidationError("RULE_METADATA must be a mapping literal")
    return dict(value)


def validate_frozen_rule_source(
    source: str,
    *,
    require_metadata: bool = False,
) -> dict[str, int]:
    """Validate a deterministic frozen rule before evaluation or Manager loading."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise RuleValidationError(f"invalid Python source: {exc}") from exc
    function = _priority_function(tree)
    _validate_signature(function)
    _validate_imports_and_safety(tree, function)
    if any(
        isinstance(node, ast.Name) and node.id == PARAMETER_CONTAINER_NAME
        for node in ast.walk(function)
    ):
        raise RuleValidationError("frozen rule still references PARAMS")
    if _assignment_literal(tree, PARAMETER_SCHEMA_NAME) is not None:
        raise RuleValidationError("frozen rule must not retain PARAMETER_SCHEMA")
    metadata = extract_rule_metadata(source)
    if require_metadata and not metadata:
        raise RuleValidationError("frozen rule is missing RULE_METADATA")
    unknown_metadata = sorted(set(metadata).difference(_ALLOWED_RULE_METADATA_FIELDS))
    if unknown_metadata:
        raise RuleValidationError(
            "unsupported RULE_METADATA field(s): " + ", ".join(unknown_metadata)
        )
    for field_name in (
        "structure_hash",
        "parameter_schema_hash",
        "best_parameter_hash",
        "optimizer_config_hash",
        "parameter_diagnostics_hash",
    ):
        if field_name not in metadata:
            continue
        value = str(metadata[field_name]).lower()
        if len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise RuleValidationError(f"invalid RULE_METADATA hash: {field_name}")
    for field_name in ("training_seeds", "validation_seeds"):
        if field_name not in metadata:
            continue
        seeds = metadata[field_name]
        if not isinstance(seeds, list) or any(
            isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds
        ):
            raise RuleValidationError(
                f"RULE_METADATA.{field_name} must be a list of integers"
            )
    if (
        "training_seeds" in metadata
        and "validation_seeds" in metadata
        and set(metadata["training_seeds"]).intersection(metadata["validation_seeds"])
    ):
        raise RuleValidationError("RULE_METADATA training/validation seeds overlap")
    if "optimizer_seed" in metadata and (
        isinstance(metadata["optimizer_seed"], bool)
        or not isinstance(metadata["optimizer_seed"], int)
    ):
        raise RuleValidationError("RULE_METADATA.optimizer_seed must be an integer")
    if "best_parameters" in metadata:
        best_parameters = metadata["best_parameters"]
        if not isinstance(best_parameters, Mapping) or any(
            not isinstance(name, str)
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for name, value in best_parameters.items()
        ):
            raise RuleValidationError(
                "RULE_METADATA.best_parameters must map names to finite numbers"
            )
        if (
            "best_parameter_hash" in metadata
            and canonical_json_sha256(best_parameters)
            != metadata["best_parameter_hash"]
        ):
            raise RuleValidationError("RULE_METADATA best parameter hash mismatch")
    return _complexity(function)
