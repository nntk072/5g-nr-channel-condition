"""Explicit post-solver relative path-power threshold masks."""

from __future__ import annotations

import math

import numpy as np


def relative_path_power_db(gains: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Return per-path coefficient power in dB relative to the strongest valid path.

    Invalid/padded paths map to negative infinity. This reports coefficient
    power ``|a|^2`` before coherent CFR summation; it is not received power or
    a receiver detectability threshold.
    """
    coefficients = np.asarray(gains)
    mask = np.asarray(valid)
    if coefficients.ndim != 1 or not np.iscomplexobj(coefficients):
        raise ValueError("gains must be a one-dimensional complex path vector")
    if mask.shape != coefficients.shape or mask.dtype.kind != "b":
        raise ValueError("valid must be a boolean mask matching the path vector")
    if not np.isfinite(coefficients).all():
        raise ValueError("gains must be finite, including padded entries")
    powers = np.abs(coefficients) ** 2
    if not np.isfinite(powers).all():
        raise ValueError("squared path powers must be finite")
    if np.any(powers[mask] < 0.0):
        raise ValueError("path powers must be nonnegative")
    if not np.any(mask) or not np.any(powers[mask] > 0.0):
        raise ValueError("at least one valid path must have positive coefficient power")
    peak_power = float(np.max(powers[mask]))
    result = np.full(coefficients.shape, -np.inf, dtype=np.float64)
    positive = mask & (powers > 0.0)
    result[positive] = 10.0 * np.log10(powers[positive] / peak_power)
    return result


def relative_power_threshold_mask(
    gains: np.ndarray, valid: np.ndarray, threshold_db: float
) -> np.ndarray:
    """Keep valid paths no weaker than ``threshold_db`` dB from the peak path."""
    if isinstance(threshold_db, bool) or not math.isfinite(threshold_db) or threshold_db > 0.0:
        raise ValueError("threshold_db must be finite and nonpositive")
    levels = relative_path_power_db(gains, valid)
    return np.asarray(valid, dtype=bool) & (levels >= threshold_db - 1e-12)
