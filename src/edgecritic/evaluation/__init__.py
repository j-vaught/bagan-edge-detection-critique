"""Evaluation metrics and analysis tools."""

from edgecritic.evaluation.benchmarks import runtime_comparison
from edgecritic.evaluation.conditioning import (
    analyze_condition_numbers,
    wvf_orientation_profile,
)
from edgecritic.evaluation.metrics import (
    angular_error_deg,
    compute_edges_at_threshold,
    compute_ods_ois,
)

__all__ = [
    "compute_ods_ois",
    "compute_edges_at_threshold",
    "angular_error_deg",
    "wvf_orientation_profile",
    "runtime_comparison",
    "analyze_condition_numbers",
]
