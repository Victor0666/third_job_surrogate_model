"""Schema constants for non-admission Top-K heuristic libraries.

Top-K selection deliberately has a schema distinct from safe admission. A
rule selected by ranking is available to the Manager, but must never be
reported as having passed the safety-admission policy.
"""

SAFE_ADMISSION_MODE = "safe_admission"
TOPK_SELECTION_MODE = "top_k"

TOPK_MANIFEST_SCHEMA_VERSION = 4
TOPK_RECORD_SCHEMA_VERSION = 1
TOPK_SELECTION_POLICY_VERSION = "feasibility_first_topk_v1"


__all__ = [
    "SAFE_ADMISSION_MODE",
    "TOPK_MANIFEST_SCHEMA_VERSION",
    "TOPK_RECORD_SCHEMA_VERSION",
    "TOPK_SELECTION_MODE",
    "TOPK_SELECTION_POLICY_VERSION",
]
