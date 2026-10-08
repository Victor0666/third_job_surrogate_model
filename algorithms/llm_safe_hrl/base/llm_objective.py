"""Shared offline LLM objective contract; downstream Safe-HRL is unchanged."""
import ast
import math

FORBIDDEN_FEATURES = frozenset({
    'slack', 'uncertainty', 'deadline', 'predicted_violation',
    'safety_margin', 'violation_risk', 'risk', 'Qc', 'q_c', 'lambda',
})
ENERGY_POLICY = 'energy_only_topk_v1'
ENERGY_RANKING_FIELDS = ['fuzzy_total_energy_score', 'candidate_sha256']


def objective_mode(value='original'):
    if value not in ('original', 'energy_only'):
        raise ValueError(f'unsupported llm_objective: {value}')
    return value


def energy_fitness(metrics):
    if isinstance(metrics['fuzzy_total_energy_score'], bool):
        raise ValueError('energy_only requires numeric energy, not bool')
    value = float(metrics['fuzzy_total_energy_score'])
    if not math.isfinite(value):
        raise ValueError('energy_only requires finite fuzzy_total_energy_score')
    return value


def objective_identity():
    return dict(llm_objective='energy_only', ranking_metric='mean_fuzzy_energy_score',
                safety_admission_enabled=False,
                forbidden_llm_features=sorted(FORBIDDEN_FEATURES),
                energy_only_feature_schema='energy_only_v1')


def validate_energy_only_source(source):
    """Keep the eight-argument API, but disallow reading safety argument slots."""
    tree = ast.parse(source)
    forbidden = set(FORBIDDEN_FEATURES)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith('get_task_priority'):
            args = node.args.posonlyargs + node.args.args
            if len(args) != 8 or node.args.vararg or node.args.kwarg:
                raise ValueError('energy_only requires the eight-argument priority API')
            forbidden.update(args[i].arg for i in (3, 7))
    for node in ast.walk(tree):
        name = node.id if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) else None
        if name in forbidden or (isinstance(node, ast.Attribute) and node.attr in forbidden):
            raise ValueError(f'energy_only heuristic uses forbidden safety feature: {name or node.attr}')
        if isinstance(node, ast.Name) and node.id in {'locals', 'globals', 'eval', 'exec', 'getattr', 'vars', '__import__'}:
            raise ValueError(f'energy_only forbids dynamic feature access: {node.id}')


def validate_objective_identity(payload, expected=None):
    mode = objective_mode(payload.get('llm_objective', 'original'))
    if expected is not None and mode != objective_mode(expected):
        raise ValueError(f'llm_objective mismatch: expected {expected}, got {mode}')
    if mode == 'energy_only':
        for key, value in objective_identity().items():
            if payload.get(key) != value:
                raise ValueError(f'energy_only objective identity mismatch: {key}')
    return mode
