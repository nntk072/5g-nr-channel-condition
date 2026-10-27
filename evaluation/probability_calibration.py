"""Group-balanced Platt calibration fitted on a held-out training partition."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression


def _calibration_roster_digest(
    probabilities: np.ndarray, labels: np.ndarray, groups: list[str]
) -> str:
    payload = {
        "raw_probabilities": probabilities.tolist(),
        "labels": labels.tolist(),
        "independent_group_ids": groups,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def apply_platt_calibrator(probabilities: np.ndarray, report: dict[str, Any]) -> np.ndarray:
    """Apply a frozen positive-slope Platt transform to raw model probabilities."""

    scores = np.asarray(probabilities, dtype=np.float64)
    if scores.ndim != 1 or not np.all(np.isfinite(scores)) or np.any((scores < 0) | (scores > 1)):
        raise ValueError("probabilities must be a finite one-dimensional vector in [0, 1]")
    if not isinstance(report, dict) or report.get("schema") != "loslab.group_platt_calibrator.v1":
        raise ValueError("unsupported Platt calibrator report")
    slope = report.get("slope")
    intercept = report.get("intercept")
    if (
        isinstance(slope, bool)
        or not isinstance(slope, (int, float))
        or not math.isfinite(float(slope))
        or slope <= 0
        or isinstance(intercept, bool)
        or not isinstance(intercept, (int, float))
        or not math.isfinite(float(intercept))
    ):
        raise ValueError("Platt calibrator parameters must be finite with positive slope")
    clipped = np.clip(scores, 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped) - np.log1p(-clipped)
    return expit(float(slope) * logits + float(intercept))


def invert_platt_calibrator(probabilities: np.ndarray, report: dict[str, Any]) -> np.ndarray:
    """Map calibrated probabilities back to the raw model probability scale."""

    calibrated = np.asarray(probabilities, dtype=np.float64)
    if (
        calibrated.ndim != 1
        or not np.all(np.isfinite(calibrated))
        or np.any((calibrated < 0) | (calibrated > 1))
    ):
        raise ValueError("calibrated probabilities must be finite one-dimensional values in [0, 1]")
    if not isinstance(report, dict) or report.get("schema") != "loslab.group_platt_calibrator.v1":
        raise ValueError("unsupported Platt calibrator report")
    slope = report.get("slope")
    intercept = report.get("intercept")
    if (
        isinstance(slope, bool)
        or not isinstance(slope, (int, float))
        or not math.isfinite(float(slope))
        or slope <= 0
        or isinstance(intercept, bool)
        or not isinstance(intercept, (int, float))
        or not math.isfinite(float(intercept))
    ):
        raise ValueError("Platt calibrator parameters must be finite with positive slope")
    clipped = np.clip(calibrated, 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped) - np.log1p(-clipped)
    return expit((logits - float(intercept)) / float(slope))


def fit_group_balanced_platt_calibrator(
    raw_probabilities: Sequence[float] | np.ndarray,
    calibration_labels: Sequence[int] | np.ndarray,
    independent_group_ids: Sequence[str] | np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit a Platt transform using only the explicit calibration partition.

    Each independent group receives equal total fitting weight. Calibration
    labels must contain both classes; callers must ensure these rows are
    disjoint from model fitting, threshold selection, and final evaluation.
    """

    scores = np.asarray(raw_probabilities, dtype=np.float64)
    labels_raw = np.asarray(calibration_labels)
    groups = np.asarray(independent_group_ids, dtype=object)
    if scores.ndim != 1 or not np.all(np.isfinite(scores)) or np.any((scores < 0) | (scores > 1)):
        raise ValueError("raw_probabilities must be finite one-dimensional values in [0, 1]")
    if labels_raw.ndim != 1 or groups.ndim != 1 or scores.shape != labels_raw.shape:
        raise ValueError(
            "calibration probabilities, labels, and groups must be equal-length vectors"
        )
    if groups.shape != labels_raw.shape:
        raise ValueError(
            "calibration probabilities, labels, and groups must be equal-length vectors"
        )
    if labels_raw.size < 2 or not np.all(np.isin(labels_raw, (0, 1))):
        raise ValueError("calibration labels must contain both binary classes")
    labels = labels_raw.astype(np.int8, copy=False)
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("calibration labels must contain both binary classes")
    group_values = groups.tolist()
    if any(
        not isinstance(group_id, str) or not group_id or group_id != group_id.strip()
        for group_id in group_values
    ):
        raise ValueError("calibration group IDs must be nonempty canonical strings")
    group_sizes = Counter(group_values)
    if len(group_sizes) < 2:
        raise ValueError("calibration requires at least two independent groups")
    clipped = np.clip(scores, 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped) - np.log1p(-clipped)
    sample_weights = np.asarray([1.0 / group_sizes[group_id] for group_id in group_values])
    estimator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    estimator.fit(logits.reshape(-1, 1), labels, sample_weight=sample_weights)
    slope = float(estimator.coef_[0, 0])
    intercept = float(estimator.intercept_[0])
    if not math.isfinite(slope) or slope <= 0 or not math.isfinite(intercept):
        raise ValueError("calibration produced a nonfinite or nonpositive Platt slope")
    report = {
        "schema": "loslab.group_platt_calibrator.v1",
        "method": "group_weighted_logistic_on_clipped_probability_logit_v1",
        "slope": slope,
        "intercept": intercept,
        "calibration_rows": int(len(labels)),
        "calibration_independent_group_count": len(group_sizes),
        "calibration_positive_prevalence": float(
            np.mean(
                [np.mean(labels[np.asarray(group_values) == group_id]) for group_id in group_sizes]
            )
        ),
        "calibration_input_sha256": _calibration_roster_digest(scores, labels, group_values),
        "fit_partition": "calibration",
        "excluded_partitions": ["validation", "test"],
    }
    return apply_platt_calibrator(scores, report), report
