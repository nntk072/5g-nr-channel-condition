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


@dataclass(frozen=True, slots=True)
class PathParameterFisherDiagnostics:
    """Local Fisher information for complex gains and excess delays.

    Parameters are ordered as all gain real parts, all gain imaginary parts,
    then all delays. The Fisher matrix uses normalized delay coordinates
    ``tau / delay_scale_s`` so its condition number is not dominated by the
    arbitrary choice of seconds versus nanoseconds. The covariance bound is in
    physical units (seconds for delay rows/columns) and is absent when the
    local model is rank deficient; a pseudoinverse must not be presented as
    finite certainty along an unobservable parameter direction.
    """

    fisher_information: np.ndarray
    eigenvalues: np.ndarray
    rank: int
    condition_number: float
    crlb_covariance: np.ndarray | None
    delay_scale_s: float


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


def path_parameter_fisher_diagnostics(
    complex_gains: np.ndarray,
    excess_delays_s: np.ndarray,
    frequencies_hz: np.ndarray,
    *,
    noise_variance: float,
    rank_tolerance: float | None = None,
) -> PathParameterFisherDiagnostics:
    """Compute the local real-parameter Fisher matrix for a complex CFR model.

    The assumed observation is ``H(f)=sum_l a_l exp(-j 2π f τ_l) + n`` with
    independent circular complex Gaussian noise whose variance is
    ``E[|n|²]`` per tone. This is a local CRLB diagnostic at the supplied true
    gains and delays, not a global estimator guarantee or a receiver.
    """
    gains = np.asarray(complex_gains, dtype=np.complex128)
    delays = np.asarray(excess_delays_s, dtype=np.float64)
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    if gains.ndim != 1 or gains.size < 1 or not np.isfinite(gains).all():
        raise ValueError("complex_gains must be a non-empty finite 1D array")
    if delays.ndim != 1 or delays.size != gains.size or not np.isfinite(delays).all():
        raise ValueError("excess_delays_s must be finite and match complex_gains")
    if np.any(delays < 0):
        raise ValueError("excess_delays_s must be nonnegative")
    if frequencies.ndim != 1 or frequencies.size < 2 or not np.isfinite(frequencies).all():
        raise ValueError("frequencies_hz must be a finite 1D array with at least two tones")
    if not np.isfinite(noise_variance) or noise_variance <= 0:
        raise ValueError("noise_variance must be finite and positive")
    if rank_tolerance is not None and (not np.isfinite(rank_tolerance) or rank_tolerance <= 0):
        raise ValueError("rank_tolerance must be finite and positive")

    # A common frequency offset only rotates each unknown complex path gain.
    # Centering the frequencies therefore preserves identifiability while
    # avoiding a needless coupling between gain phase and delay.
    centered_frequencies = frequencies - np.mean(frequencies)
    frequency_span = float(np.ptp(frequencies))
    if frequency_span <= 0:
        raise ValueError("frequencies_hz must contain at least two distinct tones")
    delay_scale_s = 1.0 / frequency_span
    steering = np.exp(-2j * np.pi * centered_frequencies[:, None] * delays[None, :])
    delay_jacobian = (
        -2j * np.pi * centered_frequencies[:, None] * gains[None, :] * steering * delay_scale_s
    )
    jacobian = np.concatenate((steering, 1j * steering, delay_jacobian), axis=1)
    fisher = (2.0 / noise_variance) * np.real(jacobian.conj().T @ jacobian)
    fisher = (fisher + fisher.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(fisher)
    tolerance = (
        rank_tolerance
        if rank_tolerance is not None
        else max(fisher.shape) * np.finfo(np.float64).eps * max(float(eigenvalues[-1]), 0.0)
    )
    rank = int(np.count_nonzero(eigenvalues > tolerance))
    if rank == fisher.shape[0]:
        condition_number = float(eigenvalues[-1] / eigenvalues[0])
        covariance_normalized = np.linalg.inv(fisher)
        parameter_scale = np.ones(fisher.shape[0], dtype=np.float64)
        parameter_scale[2 * gains.size :] = delay_scale_s
        covariance = parameter_scale[:, None] * covariance_normalized * parameter_scale[None, :]
    else:
        condition_number = float("inf")
        covariance = None
    return PathParameterFisherDiagnostics(
        fisher_information=fisher,
        eigenvalues=eigenvalues,
        rank=rank,
        condition_number=condition_number,
        crlb_covariance=covariance,
        delay_scale_s=delay_scale_s,
    )
