"""Classical sparse multipath fitting with data-driven model-order selection.

The estimator is an intentionally transparent baseline for receiver-visible
complex CFR samples. It uses a delay-grid OMP initialization, continuous delay
refinement, and a BIC score that estimates the residual noise level instead of
requiring the simulator's noise variance as an input.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares


@dataclass(frozen=True, slots=True)
class MultipathFit:
    """Selected sparse channel model and diagnostics for one CFR observation."""

    delays_s: NDArray[np.float64]
    gains: NDArray[np.complex128]
    predicted_cfr: NDArray[np.complex128]
    model_order: int
    residual_sum_squares: float
    estimated_noise_variance: float
    bic: float
    tested_orders: tuple[int, ...]


def _render(
    frequencies_hz: NDArray[np.float64],
    delays_s: NDArray[np.float64],
    gains: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    if delays_s.size == 0:
        return np.zeros(frequencies_hz.size, dtype=np.complex128)
    steering = np.exp(-2j * np.pi * frequencies_hz[:, None] * delays_s[None, :])
    return steering @ gains


def fit_sparse_multipath(
    observed_cfr: NDArray[np.complexfloating],
    frequencies_hz: NDArray[np.floating],
    pilot_mask: NDArray[np.bool_],
    *,
    maximum_paths: int = 4,
    maximum_delay_s: float,
    delay_grid_size: int = 513,
) -> MultipathFit:
    """Fit a sparse complex CFR and select path count with an unknown-noise BIC.

    Only entries selected by ``pilot_mask`` affect path detection, fitting, or
    model-order selection. The returned CFR is evaluated at every supplied
    frequency. The noise estimate is the per-complex-sample residual power on
    observed pilots; it is a nuisance estimate, not receiver calibration.

    The BIC parameter count is three real parameters per path (complex gain and
    delay) plus one unknown complex-noise variance. A zero-path candidate is
    included. This is a baseline under a finite delay-grid initialization and
    a single specular-path model; it does not represent diffuse residuals or
    establish that selected mathematical paths are physically resolvable.
    """
    y = np.asarray(observed_cfr, dtype=np.complex128)
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    mask = np.asarray(pilot_mask)
    if y.ndim != 1 or frequencies.ndim != 1 or mask.ndim != 1:
        raise ValueError("observed_cfr, frequencies_hz, and pilot_mask must be vectors")
    if y.size != frequencies.size or mask.size != y.size or y.size < 4:
        raise ValueError("input vectors must have matching lengths and at least four tones")
    if mask.dtype != np.bool_:
        raise TypeError("pilot_mask must have boolean dtype")
    if np.count_nonzero(mask) < 4:
        raise ValueError("at least four observed pilot tones are required")
    if not np.isfinite(y.real).all() or not np.isfinite(y.imag).all():
        raise ValueError("observed CFR values must be finite")
    if not np.isfinite(frequencies).all():
        raise ValueError("frequencies_hz must be finite")
    if np.unique(frequencies).size < 4:
        raise ValueError("at least four distinct frequencies are required")
    if isinstance(maximum_paths, bool) or not isinstance(maximum_paths, int):
        raise TypeError("maximum_paths must be an integer")
    if maximum_paths < 1:
        raise ValueError("maximum_paths must be positive")
    if not math.isfinite(maximum_delay_s) or maximum_delay_s <= 0:
        raise ValueError("maximum_delay_s must be finite and positive")
    if isinstance(delay_grid_size, bool) or not isinstance(delay_grid_size, int):
        raise TypeError("delay_grid_size must be an integer")
    if delay_grid_size < 9:
        raise ValueError("delay_grid_size must be at least 9")

    f = frequencies[mask]
    measured = y[mask]
    observation_count = 2 * measured.size
    maximum_identifiable_order = max(0, (observation_count - 2) // 3)
    fit_order = min(maximum_paths, maximum_identifiable_order)
    delays_grid = np.linspace(0.0, maximum_delay_s, delay_grid_size, dtype=np.float64)
    dictionary = np.exp(-2j * np.pi * f[:, None] * delays_grid[None, :])
    dictionary /= np.linalg.norm(dictionary, axis=0, keepdims=True)

    def score(
        delays: NDArray[np.float64], gains: NDArray[np.complex128]
    ) -> tuple[float, float, float]:
        residual = measured - _render(f, delays, gains)
        rss = max(float(np.vdot(residual, residual).real), np.finfo(np.float64).tiny)
        variance = rss / measured.size
        parameter_count = 3 * delays.size + 1
        bic = observation_count * math.log(rss / measured.size) + parameter_count * math.log(
            observation_count
        )
        return rss, variance, bic

    candidates: list[tuple[float, NDArray[np.float64], NDArray[np.complex128], float, float]] = []
    empty_delays = np.empty(0, dtype=np.float64)
    empty_gains = np.empty(0, dtype=np.complex128)
    rss, variance, bic = score(empty_delays, empty_gains)
    candidates.append((bic, empty_delays, empty_gains, rss, variance))

    selected_atoms: list[int] = []
    residual = measured.copy()
    for order in range(1, fit_order + 1):
        correlations = np.abs(dictionary.conj().T @ residual)
        if selected_atoms:
            correlations[np.asarray(selected_atoms, dtype=np.int64)] = -np.inf
        atom = int(np.argmax(correlations))
        selected_atoms.append(atom)
        initial_delays = delays_grid[np.asarray(selected_atoms, dtype=np.int64)]
        initial_steering = np.exp(-2j * np.pi * f[:, None] * initial_delays[None, :])
        initial_gains = np.linalg.lstsq(initial_steering, measured, rcond=None)[0]

        def residual_vector(parameters: NDArray[np.float64]) -> NDArray[np.float64]:
            path_delays = parameters[:order]
            path_gains = parameters[order : 2 * order] + 1j * parameters[2 * order :]
            error = measured - _render(f, path_delays, path_gains)
            return np.concatenate((error.real, error.imag))

        initial_parameters = np.concatenate(
            (initial_delays, initial_gains.real, initial_gains.imag)
        )
        lower = np.concatenate((np.zeros(order), np.full(2 * order, -np.inf)))
        upper = np.concatenate((np.full(order, maximum_delay_s), np.full(2 * order, np.inf)))
        refined = least_squares(
            residual_vector,
            initial_parameters,
            bounds=(lower, upper),
            max_nfev=500,
            x_scale="jac",
        )
        path_delays = refined.x[:order]
        path_gains = refined.x[order : 2 * order] + 1j * refined.x[2 * order :]
        sort_indices = np.argsort(path_delays)
        path_delays = path_delays[sort_indices]
        path_gains = path_gains[sort_indices]
        rss, variance, bic = score(path_delays, path_gains)
        candidates.append((bic, path_delays, path_gains, rss, variance))
        residual = measured - _render(f, path_delays, path_gains)

    best_bic, best_delays, best_gains, best_rss, best_variance = min(
        candidates, key=lambda candidate: candidate[0]
    )
    predicted = _render(frequencies, best_delays, best_gains)
    return MultipathFit(
        delays_s=np.asarray(best_delays, dtype=np.float64),
        gains=np.asarray(best_gains, dtype=np.complex128),
        predicted_cfr=predicted,
        model_order=int(best_delays.size),
        residual_sum_squares=float(best_rss),
        estimated_noise_variance=float(best_variance),
        bic=float(best_bic),
        tested_orders=tuple(range(len(candidates))),
    )
