"""Project-level filesystem roots shared by all algorithm families."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
ALGORITHMS_ROOT = PROJECT_ROOT / "algorithms"

