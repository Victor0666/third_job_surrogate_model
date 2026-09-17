"""Compatibility package for the relocated legacy FCFS comparison.

The implementation now lives in ``algorithms/comparisons/fcfs``.
"""

from pathlib import Path


_IMPLEMENTATION_DIR = (
    Path(__file__).resolve().parents[1]
    / "algorithms"
    / "comparisons"
    / "fcfs"
)
if not _IMPLEMENTATION_DIR.is_dir():
    raise ImportError(
        f"legacy FCFS comparison not found: {_IMPLEMENTATION_DIR}"
    )
__path__.append(str(_IMPLEMENTATION_DIR))

