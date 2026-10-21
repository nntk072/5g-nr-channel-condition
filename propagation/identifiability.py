"""Finite-bandwidth diagnostics for delay-path steering ambiguities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class PathSteeringDiagnostics:
    """Conditioning of complex path-gain estimation at fixed candidate delays.

    These diagnostics do not solve the nonlinear joint delay/gain estimation
    problem and are not, by themselves, a CRLB or a guaranteed delay-resolution
    bound.
    """

    normalized_gram: np.ndarray
    singular_values: np.ndarray
    rank: int
    condition_number: float
    mutual_coherence: float


def path_steering_diagnostics(
    delays_s: np.ndarray,
    frequencies_hz: np.ndarray,
    *,
    rank_tolerance: float | None = None,
) -> PathSteeringDiagnostics:
    """Measure column ambiguity for `exp(-j 2π f τ)` over sampled tones.

    Frequencies are absolute baseband tone values in Hz; delays are seconds.
    Columns are normalized to unit norm before computing the Gram matrix.
    ``condition_number`` is infinite when the candidate steering matrix is
    rank deficient under the selected numerical tolerance.
    """
    delays = np.asarray(delays_s, dtype=np.float64)
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    if delays.ndim != 1 or delays.size < 1 or not np.isfinite(delays).all():
        raise ValueError("delays_s must be a non-empty finite 1D array")
    if frequencies.ndim != 1 or frequencies.size < 2 or not np.isfinite(frequencies).all():
        raise ValueError("frequencies_hz must be a finite 1D array with at least two tones")
    if np.any(delays < 0):
        raise ValueError("delays_s must be nonnegative")
    if rank_tolerance is not None and (not np.isfinite(rank_tolerance) or rank_tolerance <= 0):
        raise ValueError("rank_tolerance must be finite and positive")

    steering = np.exp(-2j * np.pi * frequencies[:, None] * delays[None, :])
    steering /= np.sqrt(float(frequencies.size))
    gram = steering.conj().T @ steering
    singular_values = np.linalg.svd(steering, compute_uv=False)
    tolerance = (
        rank_tolerance
        if rank_tolerance is not None
        else max(steering.shape) * np.finfo(np.float64).eps * singular_values[0]
    )
    rank = int(np.count_nonzero(singular_values > tolerance))
    condition_number = (
        float(singular_values[0] / singular_values[-1]) if rank == delays.size else float("inf")
    )
    if delays.size == 1:
        coherence = 0.0
    else:
        off_diagonal = np.abs(gram - np.eye(delays.size))
        coherence = float(np.max(off_diagonal))
    return PathSteeringDiagnostics(
        normalized_gram=gram,
        singular_values=singular_values,
        rank=rank,
        condition_number=condition_number,
        mutual_coherence=coherence,
    )
