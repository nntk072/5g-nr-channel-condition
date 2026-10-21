"""Evaluation metrics, grouped splits, and reporting utilities."""

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
from .statistics import paired_seed_delta

__all__ = [
    "IndependentSplit",
    "audit_group_leakage",
    "binary_metrics",
    "block_bootstrap_auc",
    "freeze_split_lock",
    "group_split",
    "holdout_domain",
    "in_domain_split",
    "make_independent_split",
    "paired_seed_delta",
]
