"""Evaluation metrics, group-aware statistics, and reporting."""

from .metrics import (
    binary_metrics,
    block_bootstrap_auc,
    group_split,
    holdout_domain,
    in_domain_split,
)
from .statistics import paired_seed_delta

__all__ = [
    "binary_metrics",
    "block_bootstrap_auc",
    "group_split",
    "holdout_domain",
    "in_domain_split",
    "paired_seed_delta",
]
