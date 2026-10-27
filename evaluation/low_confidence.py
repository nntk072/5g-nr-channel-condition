"""Descriptive abstention-coverage audit at a frozen operating threshold."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np


def build_low_confidence_audit(
    probabilities: Sequence[float] | np.ndarray,
    independent_group_ids: Sequence[str] | np.ndarray,
    family_ids: Sequence[str] | np.ndarray,
    domain_ids: Sequence[str] | np.ndarray,
    *,
    threshold: float,
    margin: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Flag scores within a caller-declared margin of the frozen threshold.

    This reports descriptive coverage only. It does not choose the operating
    threshold or margin, estimate error risk, or infer physical independence.
    """

    scores = np.asarray(probabilities, dtype=np.float64)
    vectors = [
        np.asarray(independent_group_ids, dtype=object),
        np.asarray(family_ids, dtype=object),
        np.asarray(domain_ids, dtype=object),
    ]
    if (
        scores.ndim != 1
        or scores.size == 0
        or not np.all(np.isfinite(scores))
        or np.any((scores < 0) | (scores > 1))
    ):
        raise ValueError("probabilities must be a nonempty finite vector in [0, 1]")
    if any(vector.ndim != 1 or vector.shape != scores.shape for vector in vectors):
        raise ValueError("group, family, and domain vectors must align with probabilities")
    if any(
        not isinstance(value, str) or not value or value != value.strip()
        for vector in vectors
        for value in vector.tolist()
    ):
        raise ValueError("group, family, and domain IDs must be canonical nonempty strings")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
        or not 0 <= float(threshold) <= 1
    ):
        raise ValueError("threshold must be finite and in [0, 1]")
    if (
        isinstance(margin, bool)
        or not isinstance(margin, (int, float))
        or not math.isfinite(float(margin))
        or not 0 <= float(margin) < 0.5
    ):
        raise ValueError("margin must be finite and in [0, 0.5)")

    groups, families, domains = (vector.astype(str) for vector in vectors)
    flagged = np.abs(scores - float(threshold)) <= float(margin)

    def summarize(indices: np.ndarray) -> dict[str, int | float]:
        selected_groups = groups[indices]
        selected_flagged = flagged[indices]
        return {
            "record_count": int(len(indices)),
            "independent_group_count": int(len(np.unique(selected_groups))),
            "low_confidence_record_count": int(np.sum(selected_flagged)),
            "low_confidence_independent_group_count": int(
                len(np.unique(selected_groups[selected_flagged]))
            ),
            "low_confidence_record_fraction": float(np.mean(selected_flagged)),
        }

    def summarize_by(values: np.ndarray) -> list[dict[str, Any]]:
        return [
            {
                "stratum_id": stratum,
                **summarize(np.flatnonzero(values == stratum)),
            }
            for stratum in sorted(set(values.tolist()))
        ]

    report = {
        "schema": "loslab.low_confidence_audit.v1",
        "partition": "test",
        "threshold_source": "validation",
        "threshold": float(threshold),
        "margin": float(margin),
        "criterion": "absolute_probability_distance_to_threshold_at_most_margin",
        "independence_identity_source": "caller_supplied_group_ids",
        **summarize(np.arange(len(scores), dtype=np.int64)),
        "by_family": summarize_by(families),
        "by_domain": summarize_by(domains),
    }
    return flagged, report
