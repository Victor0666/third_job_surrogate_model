"""Shared fuzzy scheduling protocol for learning-based baselines."""

from __future__ import annotations

import sys
from pathlib import Path


LLM_SAFE_HRL_ROOT = Path(__file__).resolve().parents[2] / "llm_safe_hrl"
if str(LLM_SAFE_HRL_ROOT) not in sys.path:
    sys.path.insert(0, str(LLM_SAFE_HRL_ROOT))


__all__ = ["LLM_SAFE_HRL_ROOT"]
