"""Compile existing arithmetic GP trees to the production priority contract."""
from algorithms.comparisons.drlea_nichgp.gp_primitives import GPProgram, OPERATORS
from algorithms.llm_safe_hrl.LLM.rule_optimization.parameter_schema import PRIORITY_ARGUMENTS
from algorithms.llm_safe_hrl.base.llm_objective import objective_mode, validate_energy_only_source


def terminals_for_objective(mode):
    objective_mode(mode)
    return tuple(name for name in PRIORITY_ARGUMENTS
                 if mode != 'energy_only' or name not in ('slack', 'uncertainty'))


def parameterized_source(program: GPProgram, mode='original'):
    program.validate_structure()
    allowed = terminals_for_objective(mode)
    if any(token not in (*OPERATORS, *allowed) for token in program.tokens):
        raise ValueError('GP tree contains a terminal outside the priority contract')
    used = tuple(name for name in allowed if name in program.tokens)
    schema = {'parameters': [dict(name=f'w_{name}', initial_value=1.0,
                                  lower_bound=-4.0, upper_bound=4.0,
                                  semantic_description=f'GP coefficient for {name}')
                             for name in used]}
    # Flat assignments preserve tree evaluation order without deep Python ASTs.
    lines = [f"    x_{name} = np.asarray({name}, dtype=float) * PARAMS['w_{name}']"
             for name in used]
    binary = {'add': '+', 'sub': '-', 'mul': '*'}

    def render(index):
        token = program.tokens[index]
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
        f'PARAMETER_SCHEMA = {schema!r}\n'
        'def bounded(value):\n'
        '    return np.where(np.isfinite(value) & (np.abs(value) <= 1e12), value, np.nan)\n'
        'def pdiv(left, right):\n'
        '    value = np.divide(left, right, out=np.ones_like(left), where=np.abs(right) > 1e-12)\n'
        '    return np.where(np.isfinite(left) & np.isfinite(right), value, np.nan)\n'
        f"def get_task_priority_v2({', '.join(PRIORITY_ARGUMENTS)}):\n"
        + '\n'.join(lines) + f'\n    return {result}\n'
    )
    if mode == 'energy_only':
        validate_energy_only_source(source)
    return source
