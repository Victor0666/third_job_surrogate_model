"""Offline surrogate screening for SeEvo and CMA-ES.

The package is deliberately isolated from the online scheduling path.  Model
dependencies are imported lazily so a disabled or unavailable surrogate leaves
the exact evaluation path usable.
"""

from .config import SurrogateConfig
from .dataset import SurrogateContext, SurrogateDataset
from .features import extract_parameter_features, extract_structure_features
from .manager import SurrogateManager
from .models import SurrogatePrediction
from .replay_gate import DecisionTraceCapture, ReplayVerdict, replay_frozen_rule

__all__ = [
    "SurrogateConfig",
    "SurrogateContext",
    "SurrogateDataset",
    "SurrogateManager",
    "SurrogatePrediction",
    "DecisionTraceCapture",
    "ReplayVerdict",
    "replay_frozen_rule",
    "extract_parameter_features",
    "extract_structure_features",
]
