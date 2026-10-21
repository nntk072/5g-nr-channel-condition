"""Local maximum-likelihood checks for two-path relative-delay CRLBs."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from artifacts.provenance import package_source_sha256, sha256_file, versions
from propagation.identifiability import path_parameter_fisher_diagnostics
from scipy.optimize import least_squares


def _fit_two_path_local_mle(
    observed: np.ndarray,
    frequencies_hz: np.ndarray,
    delay_scale_s: float,
    initial_gains: np.ndarray,
    initial_delays_s: np.ndarray,
) -> tuple[float, int, bool]:
    """Fit unknown complex gains and delays from a truth-centered start."""
    path_count = initial_gains.size
    initial = np.concatenate(
        (initial_gains.real, initial_gains.imag, initial_delays_s / delay_scale_s)
    )

    def prediction(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gains = parameters[:path_count] + 1j * parameters[path_count : 2 * path_count]
        delays = parameters[2 * path_count :] * delay_scale_s
        steering = np.exp(-2j * np.pi * frequencies_hz[:, None] * delays[None, :])
        estimate = steering @ gains
        complex_jacobian = np.concatenate(
            (
                steering,
                1j * steering,
                -2j * np.pi * frequencies_hz[:, None] * delay_scale_s * gains[None, :] * steering,
            ),
            axis=1,
        )
        real_jacobian = np.concatenate((complex_jacobian.real, complex_jacobian.imag), axis=0)
        return estimate, real_jacobian

    def residual(parameters: np.ndarray) -> np.ndarray:
        estimate, _ = prediction(parameters)
        error = estimate - observed
        return np.concatenate((error.real, error.imag))

    def jacobian(parameters: np.ndarray) -> np.ndarray:
        return prediction(parameters)[1]

    fitted = least_squares(
        residual,
        initial,
        jac=jacobian,
        method="lm",
        max_nfev=300,
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
    )
    delays_hat_s = fitted.x[2 * path_count :] * delay_scale_s
    return float(delays_hat_s[1] - delays_hat_s[0]), int(fitted.nfev), bool(fitted.success)


def build_monte_carlo(
    *,
    seed: int,
    trials: int,
    tone_counts: Sequence[int],
    tone_spacing_hz: float,
    snr_db_values: Sequence[float],
    separation_fractions: Sequence[float],
    gain_phase_differences_deg: Sequence[float],
) -> dict[str, object]:
    """Compare local MLE relative-delay dispersion with nuisance-aware CRLBs.

    The optimizer is deliberately initialized near the true parameters. This
    audits the local Fisher calculation, not a deployable receiver or global
    path-resolution algorithm. Common timing is randomized but path gains and
    delays receive truth-centered starts; this privileged initialization is
    recorded in the result limitations.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(trials, bool) or not isinstance(trials, int) or trials < 2:
        raise ValueError("trials must be an integer of at least 2")
    if not tone_counts or any(
        isinstance(count, bool) or not isinstance(count, int) or count < 16 for count in tone_counts
    ):
        raise ValueError("tone_counts must contain integers >= 16")
    if not math.isfinite(tone_spacing_hz) or tone_spacing_hz <= 0:
        raise ValueError("tone_spacing_hz must be finite and positive")
    if not snr_db_values or any(not math.isfinite(value) for value in snr_db_values):
        raise ValueError("snr_db_values must contain finite values")
    if not separation_fractions or any(
        not math.isfinite(value) or value <= 0 for value in separation_fractions
    ):
        raise ValueError("separation_fractions must contain finite positive values")
    if not gain_phase_differences_deg or any(
        not math.isfinite(value) for value in gain_phase_differences_deg
    ):
        raise ValueError("gain_phase_differences_deg must contain finite values")

    rng = np.random.default_rng(seed)
    cases: list[dict[str, object]] = []
    for tone_count in tone_counts:
        frequencies = (np.arange(tone_count, dtype=np.float64) - tone_count // 2) * tone_spacing_hz
        bandwidth_hz = float(np.ptp(frequencies))
        delay_scale_s = 1.0 / bandwidth_hz
        for snr_db in snr_db_values:
            for separation_fraction in separation_fractions:
                relative_delay_s = separation_fraction * delay_scale_s
                for phase_difference_deg in gain_phase_differences_deg:
                    gains = np.asarray(
                        [1.0 + 0j, 0.7 * np.exp(1j * np.deg2rad(phase_difference_deg))],
                        dtype=np.complex128,
                    )
                    reference_steering = np.exp(
                        -2j
                        * np.pi
                        * frequencies[:, None]
                        * np.asarray([0.0, relative_delay_s])[None, :]
                    )
                    signal_power = float(np.mean(np.abs(reference_steering @ gains) ** 2))
                    noise_variance = signal_power / (10.0 ** (snr_db / 10.0))
                    delay_diagnostic = path_parameter_fisher_diagnostics(
                        gains,
                        np.asarray([0.0, relative_delay_s]),
                        frequencies,
                        noise_variance=noise_variance,
                        unknown_common_timing_offset=True,
                    )
                    relative_covariance = delay_diagnostic.relative_delay_crlb_covariance_s2
                    crlb_std_ns = (
                        None
                        if relative_covariance is None
                        else float(np.sqrt(max(float(relative_covariance[0, 0]), 0.0)) * 1e9)
                    )
                    fit_errors_s: list[float] = []
                    fit_nfev: list[int] = []
                    fit_successes = 0
                    order_reversals = 0
                    for _ in range(trials):
                        timing_offset_s = float(rng.uniform(0.1, 0.3) * delay_scale_s)
                        true_delays_s = np.asarray(
                            [timing_offset_s, timing_offset_s + relative_delay_s]
                        )
                        steering = np.exp(
                            -2j * np.pi * frequencies[:, None] * true_delays_s[None, :]
                        )
                        noiseless = steering @ gains
                        noise = np.sqrt(noise_variance / 2.0) * (
                            rng.standard_normal(tone_count) + 1j * rng.standard_normal(tone_count)
                        )
                        observed = noiseless + noise
                        initial_gains = gains + 0.01 * (
                            rng.standard_normal(2) + 1j * rng.standard_normal(2)
                        )
                        initial_delays_s = true_delays_s + rng.normal(
                            0.0, 0.01 * delay_scale_s, size=2
                        )
                        estimated_relative_delay_s, nfev, success = _fit_two_path_local_mle(
                            observed,
                            frequencies,
                            delay_scale_s,
                            initial_gains,
                            initial_delays_s,
                        )
                        fit_nfev.append(nfev)
                        fit_successes += int(success)
                        order_reversals += int(estimated_relative_delay_s < 0.0)
                        if success:
                            fit_errors_s.append(abs(estimated_relative_delay_s) - relative_delay_s)
                    errors_ns = np.asarray(fit_errors_s, dtype=np.float64) * 1e9
                    cases.append(
                        {
                            "tone_count": tone_count,
                            "tone_spacing_hz": tone_spacing_hz,
                            "bandwidth_hz": bandwidth_hz,
                            "snr_db": snr_db,
                            "separation_fraction_of_reciprocal_bandwidth": separation_fraction,
                            "relative_delay_truth_ns": relative_delay_s * 1e9,
                            "gain_phase_difference_deg": phase_difference_deg,
                            "trials": trials,
                            "optimizer_successes": fit_successes,
                            "path_order_reversals": order_reversals,
                            "median_function_evaluations": float(np.median(fit_nfev)),
                            "relative_delay_crlb_std_ns": crlb_std_ns,
                            "empirical_bias_ns": (
                                float(np.mean(errors_ns)) if errors_ns.size else None
                            ),
                            "empirical_std_ns": (
                                float(np.std(errors_ns, ddof=1)) if errors_ns.size > 1 else None
                            ),
                            "empirical_rmse_ns": (
                                float(np.sqrt(np.mean(errors_ns**2))) if errors_ns.size else None
                            ),
                            "empirical_std_to_crlb_ratio": (
                                float(np.std(errors_ns, ddof=1) / crlb_std_ns)
                                if errors_ns.size > 1 and crlb_std_ns not in (None, 0.0)
                                else None
                            ),
                            "per_trial_relative_delay_errors_ns": errors_ns.tolist(),
                        }
                    )
    runtime = versions()
    return {
        "schema": "loslab.identifiability_monte_carlo.v1",
        "evidence_grade": "ORACLE_INITIALIZED_LOCAL_MLE_MONTE_CARLO_NOT_RECEIVER_QUALIFICATION",
        "model": "H(f)=sum_l a_l exp(-j 2 pi f (tau_l + common_clock)) + circular complex AWGN",
        "seed": seed,
        "initialization": "truth-centered local start with small seeded perturbations",
        "source_revision": runtime.get("git_sha"),
        "package_source_sha256": package_source_sha256(),
        "script_sha256": sha256_file(__file__),
        "versions": runtime,
        "cases": cases,
        "limitations": [
            "The optimizer receives truth-centered initial gains and delays, so results audit local CRLB behavior and are not receiver performance.",
            "Paths are two static specular components with known tone frequencies and iid circular complex Gaussian noise.",
            "There is no path-count selection, global search, synchronization algorithm, channel-estimation error, mobility, or scene variation.",
            "Path permutation and global ambiguity are not resolved by a local Fisher or local maximum-likelihood comparison.",
            "Monte Carlo dispersion is conditional on these fixed gains, bandwidths, SNRs and local starts.",
        ],
    }
