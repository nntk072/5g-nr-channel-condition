"""Bounded Jakes-spectrum summaries for independent standardized TDL draws."""

from __future__ import annotations

import math
from numbers import Real
from typing import Sequence

import numpy as np


def jakes_doppler_cdf(normalized_frequency: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return the isotropic-scattering Doppler CDF for frequency divided by ``f_D``."""
    values = np.asarray(normalized_frequency)
    if (
        values.dtype == np.bool_
        or np.iscomplexobj(values)
        or not np.issubdtype(values.dtype, np.number)
    ):
        raise TypeError("normalized_frequency must contain real numeric values")
    values = values.astype(np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("normalized_frequency must be finite")
    clipped = np.clip(values, -1.0, 1.0)
    result = 0.5 + np.arcsin(clipped) / math.pi
    return np.where(values <= -1.0, 0.0, np.where(values >= 1.0, 1.0, result))


def cluster_spectrum_cdf(
    channel_coefficients: np.ndarray,
    *,
    sampling_frequency_hz: float,
    maximum_doppler_hz: float,
    normalized_frequency_cutoffs: Sequence[float],
) -> np.ndarray:
    """Estimate one two-sided, windowed Doppler-spectrum CDF per channel draw.

    ``channel_coefficients`` has axes ``[independent_draw, diffuse_tap, time]``.
    Each tap receives equal weight after per-tap periodogram normalization;
    taps are therefore not counted as independent replicates. A symmetric Hann
    window limits leakage, and the full complex FFT preserves positive and
    negative Doppler frequencies.
    """
    values = np.asarray(channel_coefficients)
    if values.ndim != 3 or any(size <= 0 for size in values.shape[:2]) or values.shape[2] < 16:
        raise ValueError("channel_coefficients must have [draw, tap, time>=16] axes")
    if not np.iscomplexobj(values) or values.dtype == np.dtype("O"):
        raise TypeError("channel_coefficients must be a dense complex array")
    if not np.isfinite(values).all():
        raise ValueError("channel_coefficients must be finite")
    for name, scalar in (
        ("sampling_frequency_hz", sampling_frequency_hz),
        ("maximum_doppler_hz", maximum_doppler_hz),
    ):
        if isinstance(scalar, bool) or not isinstance(scalar, Real):
            raise TypeError(f"{name} must be a real scalar")
        if not math.isfinite(float(scalar)) or scalar <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if sampling_frequency_hz <= 2.0 * maximum_doppler_hz:
        raise ValueError("sampling_frequency_hz must exceed twice maximum_doppler_hz")
    cutoffs = np.asarray(normalized_frequency_cutoffs)
    if (
        cutoffs.ndim != 1
        or cutoffs.size == 0
        or cutoffs.dtype == np.bool_
        or np.iscomplexobj(cutoffs)
        or not np.issubdtype(cutoffs.dtype, np.number)
    ):
        raise TypeError("normalized_frequency_cutoffs must be a nonempty real sequence")
    cutoffs = cutoffs.astype(np.float64, copy=False)
    if not np.isfinite(cutoffs).all() or np.any(np.diff(cutoffs) <= 0):
        raise ValueError("normalized_frequency_cutoffs must be finite and strictly increasing")
    normalized_nyquist = sampling_frequency_hz / (2.0 * maximum_doppler_hz)
    if cutoffs[0] < -normalized_nyquist or cutoffs[-1] > normalized_nyquist:
        raise ValueError("normalized frequency cutoffs must lie within the sampled Nyquist band")

    time_count = values.shape[-1]
    window = np.hanning(time_count).astype(np.float64)
    normalized_frequency = np.fft.fftshift(
        np.fft.fftfreq(time_count, d=1.0 / float(sampling_frequency_hz))
    ) / float(maximum_doppler_hz)
    cdf = np.empty((values.shape[0], cutoffs.size), dtype=np.float64)
    window_power = float(np.square(window).sum())
    # Chunking bounds temporary FFT memory independently of the requested draw
    # count; the frozen native protocol uses this default 16-draw working set.
    for start in range(0, values.shape[0], 16):
        stop = min(start + 16, values.shape[0])
        centered = values[start:stop].astype(np.complex128, copy=True)
        centered -= centered.mean(axis=-1, keepdims=True)
        tapered = centered * window
        periodogram = np.abs(np.fft.fftshift(np.fft.fft(tapered, axis=-1), axes=-1)) ** 2
        periodogram /= window_power
        tap_power = periodogram.sum(axis=-1, keepdims=True)
        if not np.isfinite(tap_power).all() or np.any(tap_power <= 0):
            raise ValueError("every draw/tap must have positive finite spectral power")
        draw_spectrum = (periodogram / tap_power).mean(axis=1)
        for cutoff_index, cutoff in enumerate(cutoffs):
            cdf[start:stop, cutoff_index] = draw_spectrum[:, normalized_frequency <= cutoff].sum(
                axis=1
            )
    return cdf


def simultaneous_spectrum_cdf_bounds(
    per_draw_cdfs: np.ndarray,
    *,
    expected_cdf: Sequence[float] | np.ndarray,
    familywise_alpha: float,
    maximum_absolute_error: float,
    family_claim_count: int | None = None,
) -> dict[str, np.ndarray | int | bool]:
    """Compute Bonferroni-normal bounds over independent channel-draw clusters."""
    values = np.asarray(per_draw_cdfs)
    expected = np.asarray(expected_cdf)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] == 0:
        raise ValueError("per_draw_cdfs must have [draw>=2, cutoff>=1] axes")
    if expected.shape != (values.shape[1],):
        raise ValueError("expected_cdf must have one value per frequency cutoff")
    if (
        values.dtype == np.bool_
        or expected.dtype == np.bool_
        or np.iscomplexobj(values)
        or np.iscomplexobj(expected)
        or not np.issubdtype(values.dtype, np.number)
        or not np.issubdtype(expected.dtype, np.number)
    ):
        raise TypeError("per_draw_cdfs and expected_cdf must contain real numeric values")
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("per_draw_cdfs must be finite probabilities")
    if not np.isfinite(expected).all() or np.any((expected < 0) | (expected > 1)):
        raise ValueError("expected_cdf must be finite probabilities")
    for name, scalar in (
        ("familywise_alpha", familywise_alpha),
        ("maximum_absolute_error", maximum_absolute_error),
    ):
        if isinstance(scalar, bool) or not isinstance(scalar, Real):
            raise TypeError(f"{name} must be a real scalar")
        if not math.isfinite(float(scalar)):
            raise ValueError(f"{name} must be finite")
    if not 0.0 < familywise_alpha < 1.0:
        raise ValueError("familywise_alpha must be in (0, 1)")
    if maximum_absolute_error <= 0.0:
        raise ValueError("maximum_absolute_error must be positive")

    from statistics import NormalDist

    if family_claim_count is None:
        claim_count = values.shape[1]
    elif (
        isinstance(family_claim_count, bool)
        or not isinstance(family_claim_count, int)
        or family_claim_count < values.shape[1]
    ):
        raise ValueError("family_claim_count must be an integer at least as large as this family")
    else:
        claim_count = family_claim_count
    critical_z = NormalDist().inv_cdf(1.0 - familywise_alpha / (2.0 * claim_count))
    mean = values.mean(axis=0)
    standard_error = values.std(axis=0, ddof=1) / math.sqrt(values.shape[0])
    half_width = critical_z * standard_error
    absolute_error = np.abs(mean - expected)
    return {
        "draw_count": values.shape[0],
        "claim_count": claim_count,
        "mean_cdf": mean,
        "expected_cdf": expected,
        "absolute_error": absolute_error,
        "simultaneous_half_width": half_width,
        "status": bool(np.all(absolute_error + half_width <= maximum_absolute_error)),
    }
