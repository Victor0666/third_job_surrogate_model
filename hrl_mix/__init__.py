"""Compatibility package for the relocated safe-HRL training modules.

New source files live in ``algorithms/llm_safe_hrl/hrl_mix``. Existing
commands such as ``python -m hrl_mix.train`` remain valid.
"""

from pathlib import Path


_IMPLEMENTATION_DIR = (
    Path(__file__).resolve().parents[1]
    / "algorithms"
    / "llm_safe_hrl"
    / "hrl_mix"
)
if not _IMPLEMENTATION_DIR.is_dir():
    raise ImportError(
        f"safe-HRL training implementation not found: {_IMPLEMENTATION_DIR}"
    )
__path__.append(str(_IMPLEMENTATION_DIR))

