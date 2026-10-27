"""Family-wise DKW checks for ETSI-labeled TDL Rayleigh tap marginals."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class RayleighTapMarginalResult:
    """Empirical normalized-power distance and simultaneous DKW upper bound."""

    tap_index: int
    draw_count: int
    expected_mean_power: float
    empirical_ks_distance: float
    confidence_radius: float
    simultaneous_upper_bound: float
    passed: bool


def rayleigh_marginal_dkw_radius(
    draw_count: int, family_tap_count: int, *, alpha: float = 0.01
) -> float:
    """Return the Bonferroni-composed DKW radius for a family of tap CDFs."""
    if isinstance(draw_count, bool) or not isinstance(draw_count, int) or draw_count < 1:
        raise ValueError("draw_count must be a positive integer")
    if (
        isinstance(family_tap_count, bool)
        or not isinstance(family_tap_count, int)
        or family_tap_count < 1
    ):
        raise ValueError("family_tap_count must be a positive integer")
    if isinstance(alpha, bool) or not isinstance(alpha, Real):
        raise TypeError("alpha must be a real number")
    alpha_value = float(alpha)
    if not math.isfinite(alpha_value) or not 0.0 < alpha_value < 1.0:
        raise ValueError("alpha must be finite and in (0, 1)")
    return math.sqrt(math.log(2.0 * family_tap_count / alpha_value) / (2.0 * draw_count))


def rayleigh_normalized_power_ks_distance(
    samples: NDArray[np.generic], *, expected_mean_power: float
) -> float:
    """Compute the two-sided empirical-CDF distance from Exp(mean=expected power)."""
    values = np.asarray(samples)
    if values.ndim != 1 or values.size == 0 or values.dtype.kind != "c":
        raise ValueError("samples must be a nonempty complex vector")
    if not np.isfinite(values).all():
        raise ValueError("samples must contain finite complex values")
    if isinstance(expected_mean_power, bool) or not isinstance(expected_mean_power, Real):
        raise TypeError("expected_mean_power must be a real number")
    mean_power = float(expected_mean_power)
    if not math.isfinite(mean_power) or mean_power <= 0.0:
        raise ValueError("expected_mean_power must be finite and positive")

    normalized_power = np.square(np.abs(values.astype(np.complex128, copy=False))) / mean_power
    if not np.isfinite(normalized_power).all():
        raise ValueError("normalized sample powers must be finite")
    ordered = np.sort(normalized_power)
    expected_cdf = -np.expm1(-ordered)
    count = ordered.size
    upper_empirical = np.arange(1, count + 1, dtype=np.float64) / count
    lower_empirical = np.arange(0, count, dtype=np.float64) / count
    return float(
        max(
            np.max(upper_empirical - expected_cdf),
            np.max(expected_cdf - lower_empirical),
        )
    )


def score_rayleigh_tap_marginal_family(
    path_samples: NDArray[np.generic],
    tap_power_fractions: Sequence[tuple[int, float]],
    *,
    alpha: float = 0.01,
    maximum_simultaneous_cdf_error: float = 0.05,
) -> tuple[RayleighTapMarginalResult, ...]:
    """Score independent-draw TDL coefficients against table-scaled Rayleigh marginals.

    ``path_samples`` uses ``[draw, native_tap]`` axes. Expected tap powers must
    come from the separately pinned ETSI profile reference; they are not
    estimated from these samples. The DKW bound is simultaneous over all x
    and all listed taps using a Bonferroni union bound.
    """
    samples = np.asarray(path_samples)
    if samples.ndim != 2 or samples.shape[0] == 0 or samples.shape[1] == 0:
        raise ValueError("path_samples must have nonempty [draw, native_tap] axes")
    if samples.dtype.kind != "c" or not np.isfinite(samples).all():
        raise ValueError("path_samples must contain finite complex values")
    if isinstance(maximum_simultaneous_cdf_error, bool) or not isinstance(
        maximum_simultaneous_cdf_error, Real
    ):
        raise TypeError("maximum_simultaneous_cdf_error must be a real number")
    maximum_error = float(maximum_simultaneous_cdf_error)
    if not math.isfinite(maximum_error) or not 0.0 < maximum_error < 1.0:
        raise ValueError("maximum_simultaneous_cdf_error must be finite and in (0, 1)")
    if not isinstance(tap_power_fractions, Sequence) or not tap_power_fractions:
        raise ValueError("tap_power_fractions must be a nonempty sequence")

    normalized: list[tuple[int, float]] = []
    seen: set[int] = set()
    for row in tap_power_fractions:
        if not isinstance(row, tuple) or len(row) != 2:
            raise ValueError(
                "tap power rows must be (native_tap_index, expected_mean_power) tuples"
            )
        tap_index, mean_power = row
        if isinstance(tap_index, bool) or not isinstance(tap_index, int) or tap_index < 0:
            raise ValueError("native tap indices must be nonnegative integers")
        if tap_index in seen:
            raise ValueError("native tap indices must be unique")
        if tap_index >= samples.shape[1]:
            raise ValueError("native tap index exceeds the path-sample axis")
        if isinstance(mean_power, bool) or not isinstance(mean_power, Real):
            raise TypeError("expected tap powers must be real numbers")
        mean_value = float(mean_power)
        if not math.isfinite(mean_value) or mean_value <= 0.0:
            raise ValueError("expected tap powers must be finite and positive")
        normalized.append((tap_index, mean_value))
        seen.add(tap_index)
    normalized.sort()

    radius = rayleigh_marginal_dkw_radius(samples.shape[0], len(normalized), alpha=alpha)
    return tuple(
        RayleighTapMarginalResult(
            tap_index=tap_index,
            draw_count=samples.shape[0],
            expected_mean_power=mean_power,
            empirical_ks_distance=distance,
            confidence_radius=radius,
            simultaneous_upper_bound=distance + radius,
            passed=distance + radius <= maximum_error,
        )
        for tap_index, mean_power in normalized
        for distance in (
            rayleigh_normalized_power_ks_distance(
                samples[:, tap_index], expected_mean_power=mean_power
            ),
        )
    )
