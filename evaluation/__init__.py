"""Evaluation metrics, grouped splits, duplicate audits, and reporting."""

from .duplicate_audit import (
    CFRDuplicateAudit,
    CFRDuplicatePair,
    TemporalOverlap,
    audit_cfr_near_duplicates,
    find_cross_split_temporal_overlaps,
)
from .independent_splits import (
    IndependentSplit,
    audit_group_leakage,
    freeze_split_lock,
    make_independent_split,
)
from .metrics import (
    binary_metrics,
    block_bootstrap_auc,
    group_split,
    holdout_domain,
    in_domain_split,
)
from .spatial_audit import SpatialLeakageAudit, SpatialLeakagePair, audit_spatial_buffer
from .statistics import paired_group_regression_comparison, paired_seed_delta
from .uncertainty import evaluate_grouped_regression_uncertainty
from .uncertainty_io import evaluate_prediction_file

__all__ = [
    "CFRDuplicateAudit",
    "CFRDuplicatePair",
    "IndependentSplit",
    "SpatialLeakageAudit",
    "SpatialLeakagePair",
    "TemporalOverlap",
    "audit_cfr_near_duplicates",
    "audit_group_leakage",
    "audit_spatial_buffer",
    "binary_metrics",
    "block_bootstrap_auc",
    "find_cross_split_temporal_overlaps",
    "evaluate_grouped_regression_uncertainty",
    "evaluate_prediction_file",
    "freeze_split_lock",
    "group_split",
    "holdout_domain",
    "in_domain_split",
    "make_independent_split",
    "paired_seed_delta",
    "paired_group_regression_comparison",
]
