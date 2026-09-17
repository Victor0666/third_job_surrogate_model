"""Small dependency-free arithmetic GP with protected division."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


OPERATORS = ("add", "sub", "mul", "pdiv", "min", "max")
ARITY = {name: 2 for name in OPERATORS}
VALUE_LIMIT = 1e12


def protected_division(left: float, right: float) -> float:
    if not math.isfinite(left) or not math.isfinite(right):
        return float("nan")
    if abs(right) <= 1e-12:
        return 1.0
    return float(left / right)


def _apply(operator: str, left: float, right: float) -> float:
    if operator == "add":
        value = left + right
    elif operator == "sub":
        value = left - right
    elif operator == "mul":
        value = left * right
    elif operator == "pdiv":
        value = protected_division(left, right)
    elif operator == "min":
        value = min(left, right)
    elif operator == "max":
        value = max(left, right)
    else:
        raise ValueError(f"unknown GP operator: {operator}")
    if not math.isfinite(value) or abs(value) > VALUE_LIMIT:
        return float("nan")
    return float(value)


@dataclass(frozen=True)
class GPProgram:
    """Prefix expression using arithmetic operators and named terminals."""

    tokens: tuple[str, ...]
    terminal_names: tuple[str, ...]
    version: str = "drlea_gp14_v1"

    def _evaluate_at(
        self,
        index: int,
        values: dict[str, float],
    ) -> tuple[float, int]:
        if index >= len(self.tokens):
            raise ValueError("truncated GP expression")
        token = self.tokens[index]
        if token in ARITY:
            left, next_index = self._evaluate_at(index + 1, values)
            right, next_index = self._evaluate_at(next_index, values)
            return _apply(token, left, right), next_index
        if token not in values:
            raise ValueError(f"unknown GP terminal: {token}")
        return float(values[token]), index + 1

    def __call__(self, terminals: Sequence[float]) -> float:
        if len(terminals) != len(self.terminal_names):
            raise ValueError("GP terminal dimension mismatch")
        array = np.asarray(terminals, dtype=np.float64)
        if not np.all(np.isfinite(array)):
            return float("nan")
        values = dict(zip(self.terminal_names, array.tolist()))
        try:
            result, final_index = self._evaluate_at(0, values)
        except (ArithmeticError, OverflowError, ValueError):
            return float("nan")
        if final_index != len(self.tokens) or not math.isfinite(result):
            return float("nan")
        return float(result)

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "tokens": list(self.tokens),
            "terminal_names": list(self.terminal_names),
            "expression": self.expression(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "GPProgram":
        program = cls(
            tokens=tuple(str(value) for value in payload["tokens"]),
            terminal_names=tuple(
                str(value) for value in payload["terminal_names"]
            ),
            version=str(payload.get("version", "drlea_gp14_v1")),
        )
        program.validate_structure()
        return program

    def validate_structure(self) -> None:
        placeholder = {
            name: 0.0 for name in self.terminal_names
        }
        _value, final_index = self._evaluate_at(0, placeholder)
        if final_index != len(self.tokens):
            raise ValueError("GP expression has trailing tokens")

    def expression(self) -> str:
        def render(index: int) -> tuple[str, int]:
            token = self.tokens[index]
            if token not in ARITY:
                return token, index + 1
            left, next_index = render(index + 1)
            right, next_index = render(next_index)
            return f"{token}({left},{right})", next_index

        text, final_index = render(0)
        if final_index != len(self.tokens):
            raise ValueError("GP expression has trailing tokens")
        return text


def random_program(
    rng: np.random.Generator,
    terminal_names: tuple[str, ...],
    *,
    max_depth: int,
    force_operator: bool = True,
) -> GPProgram:
    def grow(depth: int, force: bool = False) -> list[str]:
        if depth >= max_depth or (
            not force and rng.random() < 0.25
        ):
            return [str(rng.choice(terminal_names))]
        operator = str(rng.choice(OPERATORS))
        return [
            operator,
            *grow(depth + 1),
            *grow(depth + 1),
        ]

    return GPProgram(
        tuple(grow(0, force_operator)),
        tuple(terminal_names),
    )


def _subtree_end(tokens: tuple[str, ...], start: int) -> int:
    token = tokens[start]
    if token not in ARITY:
        return start + 1
    position = start + 1
    for _ in range(ARITY[token]):
        position = _subtree_end(tokens, position)
    return position


def program_depth(program: GPProgram) -> int:
    def depth(index: int) -> tuple[int, int]:
        token = program.tokens[index]
        if token not in ARITY:
            return 0, index + 1
        left, position = depth(index + 1)
        right, position = depth(position)
        return 1 + max(left, right), position

    value, final = depth(0)
    if final != len(program.tokens):
        raise ValueError("invalid program structure")
    return value


def crossover(
    first: GPProgram,
    second: GPProgram,
    rng: np.random.Generator,
    max_depth: int,
) -> tuple[GPProgram, GPProgram]:
    first_start = int(rng.integers(len(first.tokens)))
    second_start = int(rng.integers(len(second.tokens)))
    first_end = _subtree_end(first.tokens, first_start)
    second_end = _subtree_end(second.tokens, second_start)
    child_a = GPProgram(
        first.tokens[:first_start]
        + second.tokens[second_start:second_end]
        + first.tokens[first_end:],
        first.terminal_names,
    )
    child_b = GPProgram(
        second.tokens[:second_start]
        + first.tokens[first_start:first_end]
        + second.tokens[second_end:],
        first.terminal_names,
    )
    if program_depth(child_a) > max_depth:
        child_a = first
    if program_depth(child_b) > max_depth:
        child_b = second
    return child_a, child_b


def mutate(
    program: GPProgram,
    rng: np.random.Generator,
    max_depth: int,
) -> GPProgram:
    start = int(rng.integers(len(program.tokens)))
    end = _subtree_end(program.tokens, start)
    replacement = random_program(
        rng,
        program.terminal_names,
        max_depth=max(1, min(2, max_depth)),
        force_operator=False,
    )
    result = GPProgram(
        program.tokens[:start]
        + replacement.tokens
        + program.tokens[end:],
        program.terminal_names,
    )
    return result if program_depth(result) <= max_depth else program
