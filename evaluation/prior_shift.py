"""Declared prior-shift adjustment fitted only from training groups."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from typing import Any

import numpy as np

_SPEC_SCHEMA = "loslab.prior_shift.v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SPEC_FIELDS = frozenset(
    {
        "schema",
        "training_prior_kind",
        "training_prior_id",
        "target_prior_kind",
        "target_prior_id",
        "target_positive_prevalence",
        "target_prior_evidence_sha256",
    }
)


def validate_prior_shift_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return a canonical natural-target prior-shift declaration."""

    if not isinstance(spec, Mapping):
        raise TypeError("prior_shift must be a mapping")
    if any(not isinstance(field, str) for field in spec):
        raise ValueError("prior_shift field names must be strings")
    fields = set(spec)
    if fields != _SPEC_FIELDS:
        raise ValueError(
            "prior_shift fields must match the versioned contract exactly; "
            f"missing={sorted(_SPEC_FIELDS - fields)}, "
            f"unknown={sorted(fields - _SPEC_FIELDS)}"
        )
    if spec["schema"] != _SPEC_SCHEMA:
        raise ValueError(f"prior_shift schema must be {_SPEC_SCHEMA!r}")
    if spec["training_prior_kind"] not in ("natural", "enriched_stress"):
        raise ValueError("training_prior_kind must be natural or enriched_stress")
    if spec["target_prior_kind"] != "natural":
        raise ValueError("target_prior_kind must be natural; stress priors are evaluation strata")
    result = dict(spec)
    for field_name in ("training_prior_id", "target_prior_id"):
        value = result[field_name]
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError(f"{field_name} must be a nonempty canonical string")
    prevalence = result["target_positive_prevalence"]
    if isinstance(prevalence, bool) or not isinstance(prevalence, (int, float)):
        raise ValueError("target_positive_prevalence must be a real number")
    prevalence = float(prevalence)
    if not math.isfinite(prevalence) or not 0.0 < prevalence < 1.0:
        raise ValueError("target_positive_prevalence must be finite and strictly between 0 and 1")
    result["target_positive_prevalence"] = prevalence
    evidence_digest = result["target_prior_evidence_sha256"]
    if not isinstance(evidence_digest, str) or not _SHA256_RE.fullmatch(evidence_digest):
        raise ValueError("target_prior_evidence_sha256 must be a lowercase SHA-256 digest")
    return result


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def adjust_probabilities_between_priors(
    probabilities: np.ndarray,
    source_positive_prevalence: float,
    target_positive_prevalence: float,
) -> np.ndarray:
    """Apply label-shift odds correction between two declared class priors."""

    scores = np.asarray(probabilities, dtype=np.float64)
    if (
        scores.ndim != 1
        or scores.size == 0
        or not np.all(np.isfinite(scores))
        or np.any((scores < 0) | (scores > 1))
    ):
        raise ValueError("probabilities must be a finite one-dimensional vector in [0, 1]")
    priors = (source_positive_prevalence, target_positive_prevalence)
    if any(
        isinstance(prior, bool)
        or not isinstance(prior, (int, float))
        or not math.isfinite(float(prior))
        or not 0.0 < float(prior) < 1.0
        for prior in priors
    ):
        raise ValueError("source and target positive prevalence must be finite and in (0, 1)")
    source = float(source_positive_prevalence)
    target = float(target_positive_prevalence)
    numerator = scores * target * (1.0 - source)
    denominator = numerator + (1.0 - scores) * source * (1.0 - target)
    return numerator / denominator


def adjust_probabilities_for_prior_shift(
    probabilities: np.ndarray,
    training_labels: np.ndarray,
    training_group_ids: np.ndarray,
    prior_shift: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Adjust posterior odds from a group-weighted train prior to a natural prior.

    Every independent training group receives equal weight when its empirical
    positive fraction is aggregated. Validation/test labels are not accepted by
    this API, preventing them from influencing prior estimation.
    """

    spec = validate_prior_shift_spec(prior_shift)
    scores = np.asarray(probabilities, dtype=np.float64)
    labels_raw = np.asarray(training_labels)
    groups = np.asarray(training_group_ids, dtype=object)
    if scores.ndim != 1 or not np.all(np.isfinite(scores)) or np.any((scores < 0) | (scores > 1)):
        raise ValueError("probabilities must be a finite one-dimensional vector in [0, 1]")
    if labels_raw.ndim != 1 or groups.ndim != 1 or labels_raw.shape != groups.shape:
        raise ValueError(
            "training labels and group IDs must be equal-length one-dimensional arrays"
        )
    if labels_raw.size == 0 or not np.all(np.isin(labels_raw, (0, 1))):
        raise ValueError("training labels must be a nonempty binary vector")
    group_values = groups.tolist()
    if any(
        not isinstance(group_id, str) or not group_id or group_id != group_id.strip()
        for group_id in group_values
    ):
        raise ValueError("training group IDs must be nonempty canonical strings")

    labels = labels_raw.astype(np.int8, copy=False)
    group_rates: dict[str, list[int]] = {}
    for group_id, label in zip(group_values, labels.tolist(), strict=True):
        group_rates.setdefault(group_id, []).append(label)
    group_prevalences = [float(np.mean(group_rates[group_id])) for group_id in sorted(group_rates)]
    training_prevalence = float(np.mean(group_prevalences))
    if not 0.0 < training_prevalence < 1.0:
        raise ValueError("training-group prevalence must contain both classes")

    target_prevalence = spec["target_positive_prevalence"]
    adjusted = adjust_probabilities_between_priors(
        scores,
        source_positive_prevalence=training_prevalence,
        target_positive_prevalence=target_prevalence,
    )
    roster_payload = {
        "training_labels": labels.tolist(),
        "training_group_ids": group_values,
    }
    report = {
        "schema": "loslab.prior_shift_adjustment.v1",
        "prior_shift_spec_sha256": _canonical_sha256(spec),
        "training_prior_kind": spec["training_prior_kind"],
        "training_prior_id": spec["training_prior_id"],
        "calibration_group_weighted_positive_prevalence": training_prevalence,
        "calibration_independent_group_count": len(group_prevalences),
        "calibration_labels_and_groups_sha256": _canonical_sha256(roster_payload),
        "target_prior_kind": spec["target_prior_kind"],
        "target_prior_id": spec["target_prior_id"],
        "target_positive_prevalence": target_prevalence,
        "target_prior_evidence_sha256": spec["target_prior_evidence_sha256"],
        "adjustment": "prior_probability_shift_odds_v1",
        "fit_partition": "calibration",
        "probability_evaluation_partitions": ["validation", "test"],
    }
    return adjusted.astype(np.float64, copy=False), report
