"""Confidence-controlled zero-mean and proper-complex checks for G4 channel taps."""

from __future__ import annotations

import math
from numbers import Integral, Real
from statistics import NormalDist
from typing import Any

import numpy as np


def evaluate_complex_circularity_family(
    channel_coefficients: np.ndarray,
    *,
    familywise_alpha: float = 0.01,
    maximum_coordinate_half_width: float = 0.05,
    family_claim_count: int | None = None,
) -> dict[str, Any]:
    """Evaluate zero complex mean and zero pseudocovariance across Rayleigh taps.

    ``channel_coefficients`` is shaped ``[independent_draw, tap]``. Draws are
    the statistical units; taps within a draw remain paired. The confidence
    family has four two-sided real-coordinate claims per tap: real and
    imaginary normalized mean, and real and imaginary normalized
    pseudocovariance. Bonferroni-normal intervals are used exactly as in the
    frozen fixed/scalable TDL protocols.
    """
    if not isinstance(channel_coefficients, np.ndarray):
        raise TypeError("channel_coefficients must be a NumPy array")
    values = channel_coefficients
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("channel_coefficients must have [draw>=2, tap>=1] axes")
    if values.dtype.kind != "c":
        raise TypeError("channel_coefficients must have a complex dtype")
    if not np.isfinite(values).all():
        raise ValueError("channel_coefficients must contain only finite values")
    if isinstance(familywise_alpha, bool) or not isinstance(familywise_alpha, Real):
        raise TypeError("familywise_alpha must be a real scalar")
    if not math.isfinite(float(familywise_alpha)) or not 0.0 < familywise_alpha < 1.0:
        raise ValueError("familywise_alpha must be finite and in (0, 1)")
    if isinstance(maximum_coordinate_half_width, bool) or not isinstance(
        maximum_coordinate_half_width, Real
    ):
        raise TypeError("maximum_coordinate_half_width must be a real scalar")
    if (
        not math.isfinite(float(maximum_coordinate_half_width))
        or maximum_coordinate_half_width <= 0.0
    ):
        raise ValueError("maximum_coordinate_half_width must be finite and positive")

    minimum_claim_count = 4 * values.shape[1]
    if family_claim_count is None:
        claim_count = minimum_claim_count
    elif (
        isinstance(family_claim_count, bool)
        or not isinstance(family_claim_count, Integral)
        or family_claim_count < minimum_claim_count
    ):
        raise ValueError(
            "family_claim_count must be an integer covering at least four claims per tap"
        )
    else:
        claim_count = int(family_claim_count)

    # Per-tap scaling keeps power and squared coefficients bounded without
    # changing normalized means, pseudocovariances, or their standard errors.
    complex_values = values.astype(np.complex128, copy=False)
    scale = np.max(np.maximum(np.abs(complex_values.real), np.abs(complex_values.imag)), axis=0)
    if np.any(scale == 0.0):
        raise ValueError("each tap must have positive finite power")
    scaled = complex_values / scale[None, :]
    power = np.mean(np.abs(scaled) ** 2, axis=0)
    if not np.isfinite(power).all() or np.any(power <= 0.0):
        raise ValueError("each tap must have positive finite power")

    draw_count = values.shape[0]
    critical_z = NormalDist().inv_cdf(1.0 - float(familywise_alpha) / (2.0 * claim_count))
    normalized_mean = scaled.mean(axis=0) / np.sqrt(power)
    mean_half_width = (
        critical_z
        * np.stack((scaled.real.std(axis=0, ddof=1), scaled.imag.std(axis=0, ddof=1))).T
        / (np.sqrt(power)[:, None] * math.sqrt(draw_count))
    )

    normalized_pseudocovariance = np.mean(scaled**2, axis=0) / power
    influence = scaled**2 - normalized_pseudocovariance[None, :] * power[None, :]
    pseudocovariance_half_width = (
        critical_z
        * np.stack((influence.real.std(axis=0, ddof=1), influence.imag.std(axis=0, ddof=1))).T
        / (power[:, None] * math.sqrt(draw_count))
    )

    mean_coordinates = np.stack((normalized_mean.real, normalized_mean.imag), axis=1)
    pseudocovariance_coordinates = np.stack(
        (normalized_pseudocovariance.real, normalized_pseudocovariance.imag), axis=1
    )
    mean_contains_zero = np.abs(mean_coordinates) <= mean_half_width
    pseudocovariance_contains_zero = (
        np.abs(pseudocovariance_coordinates) <= pseudocovariance_half_width
    )
    maximum_half_width = max(float(mean_half_width.max()), float(pseudocovariance_half_width.max()))
    return {
        "schema": "loslab.g4_complex_circularity_family.v1",
        "draw_count": draw_count,
        "tap_count": values.shape[1],
        "claim_count": claim_count,
        "familywise_alpha": float(familywise_alpha),
        "normalized_mean": normalized_mean,
        "mean_coordinate_half_width": mean_half_width,
        "mean_contains_zero": mean_contains_zero,
        "normalized_pseudocovariance": normalized_pseudocovariance,
        "pseudocovariance_coordinate_half_width": pseudocovariance_half_width,
        "pseudocovariance_contains_zero": pseudocovariance_contains_zero,
        "maximum_coordinate_half_width": maximum_half_width,
        "maximum_coordinate_half_width_limit": float(maximum_coordinate_half_width),
        "pass": bool(
            np.all(mean_contains_zero)
            and np.all(pseudocovariance_contains_zero)
            and maximum_half_width <= maximum_coordinate_half_width
        ),
    }
