"""Stable filesystem roots for the proposed algorithm.

Algorithm source is intentionally colocated below ``algorithms/`` while data,
logs, tests, and shared domain models remain project-level resources. Code
must use these constants instead of inferring the project root from the local
depth of ``base``, ``hrl_mix``, or ``LLM``.
"""

from __future__ import annotations

from pathlib import Path

from project_paths import PROJECT_ROOT

ALGORITHM_ROOT = Path(__file__).resolve().parent
LLM_ROOT = ALGORITHM_ROOT / "LLM"
HRL_ROOT = ALGORITHM_ROOT / "hrl_mix"
BASE_ROOT = ALGORITHM_ROOT / "base"
COMPARISON_ROOT = PROJECT_ROOT / "algorithms" / "comparisons"
