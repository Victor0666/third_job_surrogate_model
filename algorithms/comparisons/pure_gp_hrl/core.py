"""GP trees evolve features and constants together; no external parameter search."""
from dataclasses import dataclass
import math

from algorithms.comparisons.drlea_nichgp.gp_primitives import (
    GPProgram, OPERATORS, _subtree_end, program_depth, crossover as subtree_crossover,
)
from algorithms.comparisons.gp_hrl.core import terminals_for_objective
from algorithms.llm_safe_hrl.LLM.rule_optimization.parameter_schema import (
    PRIORITY_ARGUMENTS, canonical_json_sha256, validate_frozen_rule_source,
)
from algorithms.llm_safe_hrl.base.llm_objective import validate_energy_only_source


@dataclass(frozen=True)
class PureGPProgram(GPProgram):
    tokens: tuple[str | float, ...]
    version: str = 'pure_gp_erc_v1'

    def _evaluate_at(self, index, values):
        if index >= len(self.tokens):
            raise ValueError('truncated GP expression')
        token = self.tokens[index]
        if isinstance(token, float):
            if not math.isfinite(token) or not -4. <= token <= 4.:
                raise ValueError('GP constant must be finite and in [-4,4]')
            return token, index + 1
        if not isinstance(token, str):
            raise ValueError('GP tokens must be named operators/features or float constants')
        return super()._evaluate_at(index, values)

    def validate_structure(self):
        if self.version != 'pure_gp_erc_v1':
            raise ValueError('pure GP schema mismatch')
        super().validate_structure()
        if not any(token in self.terminal_names for token in self.tokens):
            raise ValueError('GP rule must use at least one task feature')

    @classmethod
    def from_dict(cls, payload):
        program = cls(tuple(payload['tokens']), tuple(payload['terminal_names']), payload['version'])
        program.validate_structure()
        return program


def random_program(rng, terminal_names, *, max_depth, force_operator=True):
    def grow(depth, force=False):
        if depth >= max_depth or (not force and rng.random() < .25):
            return ([float(rng.uniform(-4., 4.))] if rng.random() < .25
                    else [str(rng.choice(terminal_names))])
        return [str(rng.choice(OPERATORS)), *grow(depth + 1), *grow(depth + 1)]
    return PureGPProgram(tuple(grow(0, force_operator)), tuple(terminal_names))


def crossover(first, second, rng, max_depth):
    if first.terminal_names != second.terminal_names:
        raise ValueError('GP crossover requires the same feature contract')
    children = subtree_crossover(first, second, rng, max_depth)
    return tuple(PureGPProgram(child.tokens, child.terminal_names) for child in children)


def mutate(program, rng, max_depth):
    start = int(rng.integers(len(program.tokens)))
    end = _subtree_end(program.tokens, start)
    replacement = random_program(rng, program.terminal_names,
                                 max_depth=max(1, min(2, max_depth)), force_operator=False)
    child = PureGPProgram(program.tokens[:start] + replacement.tokens + program.tokens[end:],
                          program.terminal_names)
    return child if program_depth(child) <= max_depth else program


def frozen_source(program, mode, search_config, context):
    program.validate_structure()
    allowed = terminals_for_objective(mode)
    if program.terminal_names != allowed:
        raise ValueError('GP features differ from the objective contract')
    constants = {f'erc_{index}': token for index, token in enumerate(program.tokens)
                 if isinstance(token, float)}
    structure = ['ERC' if isinstance(token, float) else token for token in program.tokens]
    metadata = dict(structure_hash=canonical_json_sha256(structure),
                    parameter_schema_hash=canonical_json_sha256({'kind': 'gp_constant_leaves',
                                                                 'positions': list(constants)}),
                    best_parameter_hash=canonical_json_sha256(constants), best_parameters=constants,
                    optimizer_config_hash=canonical_json_sha256(search_config),
                    parameter_diagnostics_hash=canonical_json_sha256({'status': 'not_run', 'method': 'pure_gp'}),
                    optimizer_seed=search_config['algorithm_seed'],
                    training_seeds=list(context.llm_train_seeds),
                    validation_seeds=list(context.llm_validation_seeds))
    used = tuple(name for name in allowed if name in program.tokens)
    lines = [f'    x_{name} = np.asarray({name}, dtype=float)' for name in used]
    binary = {'add': '+', 'sub': '-', 'mul': '*'}

    def render(index):
        token = program.tokens[index]
        if isinstance(token, float):
            return repr(token), index + 1
        if token not in OPERATORS:
            return f'x_{token}', index + 1
        left, position = render(index + 1)
        right, position = render(position)
        expression = (f'{left} {binary[token]} {right}' if token in binary else
                      f'np.minimum({left}, {right})' if token == 'min' else
                      f'np.maximum({left}, {right})' if token == 'max' else
                      f'pdiv({left}, {right})')
        variable = f'v_{index}'
        lines.append(f'    {variable} = bounded({expression})')
        return variable, position

    result, _ = render(0)
    source = (
        'import numpy as np\n'
        f'RULE_METADATA = {metadata!r}\n'
        'def bounded(value):\n'
        '    return np.where(np.isfinite(value) & (np.abs(value) <= 1e12), value, np.nan)\n'
        'def pdiv(left, right):\n'
        '    left, right = np.broadcast_arrays(left, right)\n'
        '    value = np.divide(left, right, out=np.ones_like(left, dtype=float), where=np.abs(right) > 1e-12)\n'
        '    return np.where(np.isfinite(left) & np.isfinite(right), value, np.nan)\n'
        f"def get_task_priority_v2({', '.join(PRIORITY_ARGUMENTS)}):\n"
        + '\n'.join(lines) + f'\n    return {result}\n'
    )
    validate_frozen_rule_source(source, require_metadata=True)
    if mode == 'energy_only':
        validate_energy_only_source(source)
    return source
