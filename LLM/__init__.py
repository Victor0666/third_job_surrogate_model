"""Compatibility package for the relocated SeEvo/CEWS modules.

New source files live in ``algorithms/llm_safe_hrl/LLM``. Existing imports
such as ``LLM.problems.cews_task_constructive.eval`` remain valid.
"""

from pathlib import Path


_IMPLEMENTATION_DIR = (
    Path(__file__).resolve().parents[1]
    / "algorithms"
    / "llm_safe_hrl"
    / "LLM"
)
if not _IMPLEMENTATION_DIR.is_dir():
    raise ImportError(
        f"SeEvo/CEWS implementation not found: {_IMPLEMENTATION_DIR}"
    )
__path__.append(str(_IMPLEMENTATION_DIR))

