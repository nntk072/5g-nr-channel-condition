"""Matched profile-likelihood and OMP uncertainty estimators for two-path CFRs."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.stats import binomtest

from experiments.identifiability_monte_carlo import (
    _fit_two_path_local_mle,
    _omp_two_path_initialization,
)
from experiments.identifiability_profile_likelihood import (
    _prepare_delay_grid,
    _profile_prepared_grid,
    _split_conformal_threshold,
)


def run_matched_uncertainty_estimators(
    *,
    seed: int,
    trials: int,
    calibration_trials: int,
    tone_counts: Sequence[int],
    tone_spacing_hz: float,
    snr_db_values: Sequence[float],
    separation_fractions: Sequence[float],
    gain_phase_differences_deg: Sequence[float],
    profile_grid_step_fraction: float = 1.0 / 128.0,
    omp_grid_step_fraction: float = 1.0 / 32.0,
    maximum_delay_fraction: float = 2.0,
    confidence_level: float = 0.95,
    bootstrap_draws: int = 5_000,
) -> dict[str, Any]:
    """Compare calibrated delay sets on identical calibration/evaluation trials.

    The profile method forms a set from a two-path profile-likelihood ratio.
    The classical baseline estimates two paths by OMP followed by local
    nonlinear least-squares refinement; split conformal calibrates a symmetric
    interval around its permutation-invariant relative-delay estimate. Both
    methods receive the same observations, known path count, noise variance,
    and per-condition calibration/evaluation split.
    """
    for name, value, minimum in (
        ("seed", seed, 0),
        ("trials", trials, 2),
        ("calibration_trials", calibration_trials, 19),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value < minimum:
            raise ValueError(f"{name} must be at least {minimum}")
    if not tone_counts or any(
        isinstance(count, bool) or not isinstance(count, int) or count < 16 for count in tone_counts
    ):
        raise ValueError("tone_counts must contain integers >= 16")
    if not math.isfinite(tone_spacing_hz) or tone_spacing_hz <= 0:
        raise ValueError("tone_spacing_hz must be finite and positive")
    for name, values in (
        ("snr_db_values", snr_db_values),
        ("separation_fractions", separation_fractions),
        ("gain_phase_differences_deg", gain_phase_differences_deg),
    ):
        if not values or any(not math.isfinite(float(value)) for value in values):
            raise ValueError(f"{name} must contain finite values")
    if any(value <= 0 for value in separation_fractions):
        raise ValueError("separation_fractions must be positive")
    for name, numeric_value in (
        ("profile_grid_step_fraction", profile_grid_step_fraction),
        ("omp_grid_step_fraction", omp_grid_step_fraction),
        ("maximum_delay_fraction", maximum_delay_fraction),
    ):
        if not math.isfinite(numeric_value) or numeric_value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if maximum_delay_fraction <= max(separation_fractions) + 0.3:
        raise ValueError("maximum_delay_fraction must exceed max separation plus timing offset")
    if not math.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be finite and in (0, 1)")
    if isinstance(bootstrap_draws, bool) or not isinstance(bootstrap_draws, int):
        raise TypeError("bootstrap_draws must be an integer")
    if bootstrap_draws <= 0:
        raise ValueError("bootstrap_draws must be positive")

    rng = np.random.default_rng(seed)
    cases: list[dict[str, Any]] = []
    for tone_count in tone_counts:
        frequencies = (np.arange(tone_count, dtype=np.float64) - tone_count // 2) * float(
            tone_spacing_hz
        )
        bandwidth_hz = float(np.ptp(frequencies))
        delay_scale_s = 1.0 / bandwidth_hz
        profile_grid_s = (
            np.arange(
                0.0,
                maximum_delay_fraction + profile_grid_step_fraction / 2.0,
                profile_grid_step_fraction,
            )
            * delay_scale_s
        )
        omp_grid_s = (
            np.arange(
                0.0,
                maximum_delay_fraction + omp_grid_step_fraction / 2.0,
                omp_grid_step_fraction,
            )
            * delay_scale_s
        )
        prepared_grid = _prepare_delay_grid(frequencies, profile_grid_s)
        for snr_db in snr_db_values:
            for separation_fraction in separation_fractions:
                relative_truth_s = float(separation_fraction) * delay_scale_s
                for phase_difference_deg in gain_phase_differences_deg:
                    gains = np.asarray(
                        [1.0 + 0j, 0.7 * np.exp(1j * np.deg2rad(phase_difference_deg))],
                        dtype=np.complex128,
                    )
                    reference = np.exp(
                        -2j
                        * np.pi
                        * frequencies[:, None]
                        * np.asarray([0.0, relative_truth_s])[None, :]
                    )
                    signal_power = float(np.mean(np.abs(reference @ gains) ** 2))
                    noise_variance = signal_power / (10.0 ** (float(snr_db) / 10.0))
                    calibration: list[dict[str, Any]] = []
                    evaluation: list[dict[str, Any]] = []
                    for split, sample_count in (
                        ("calibration", calibration_trials),
                        ("evaluation", trials),
                    ):
                        for trial_index in range(sample_count):
                            common_timing_s = float(rng.uniform(0.1, 0.3) * delay_scale_s)
                            true_delays_s = np.asarray(
                                [common_timing_s, common_timing_s + relative_truth_s]
                            )
                            steering = np.exp(
                                -2j * np.pi * frequencies[:, None] * true_delays_s[None, :]
                            )
                            noiseless = steering @ gains
                            noise = np.sqrt(noise_variance / 2.0) * (
                                rng.standard_normal(tone_count)
                                + 1j * rng.standard_normal(tone_count)
                            )
                            observed = noiseless + noise

                            profile = _profile_prepared_grid(
                                observed, noise_variance, prepared_grid
                            )
                            relative_grid_s = np.asarray(
                                profile["relative_delay_grid_s"], dtype=np.float64
                            )
                            log_likelihood_ratio = np.asarray(
                                profile["relative_delay_log_likelihood_ratio"],
                                dtype=np.float64,
                            )
                            truth_index = int(np.argmin(np.abs(relative_grid_s - relative_truth_s)))
                            profile_score = max(0.0, -float(log_likelihood_ratio[truth_index]))

                            omp_gains, omp_delays_s = _omp_two_path_initialization(
                                observed, frequencies, omp_grid_s
                            )
                            omp_relative_s, nfev, optimizer_success = _fit_two_path_local_mle(
                                observed,
                                frequencies,
                                delay_scale_s,
                                omp_gains,
                                omp_delays_s,
                            )
                            omp_estimate_s = abs(omp_relative_s)
                            omp_score = abs(omp_estimate_s - relative_truth_s)
                            row: dict[str, Any] = {
                                "split": split,
                                "trial": trial_index,
                                "common_timing_offset_fraction_of_reciprocal_bandwidth": (
                                    common_timing_s / delay_scale_s
                                ),
                                "true_relative_delay_s": relative_truth_s,
                                "profile_map_relative_delay_s": profile["map_relative_delay_s"],
                                "profile_nonconformity": profile_score,
                                "profile_log_likelihood_ratio": log_likelihood_ratio.tolist(),
                                "omp_refined_relative_delay_s": omp_estimate_s,
                                "omp_absolute_error_s": omp_score,
                                "omp_optimizer_success": optimizer_success,
                                "omp_function_evaluations": nfev,
                                "omp_path_order_reversal": omp_relative_s < 0.0,
                            }
                            (calibration if split == "calibration" else evaluation).append(row)

                    profile_threshold = _split_conformal_threshold(
                        [float(row["profile_nonconformity"]) for row in calibration],
                        confidence_level,
                    )
                    omp_threshold_s = _split_conformal_threshold(
                        [float(row["omp_absolute_error_s"]) for row in calibration],
                        confidence_level,
                    )
                    profile_grid_s_values = np.asarray(
                        profile["relative_delay_grid_s"], dtype=np.float64
                    )
                    profile_coverage: list[bool] = []
                    omp_coverage: list[bool] = []
                    profile_spans_ns: list[float] = []
                    omp_spans_ns: list[float] = []
                    profile_components: list[int] = []
                    profile_map_errors_ns: list[float] = []
                    omp_errors_ns: list[float] = []
                    for row in evaluation:
                        row_log_ratio = np.asarray(
                            row.pop("profile_log_likelihood_ratio"), dtype=np.float64
                        )
                        selected = np.flatnonzero(row_log_ratio >= -profile_threshold)
                        profile_covers = bool(
                            np.any(
                                np.abs(profile_grid_s_values[selected] - relative_truth_s)
                                <= profile_grid_step_fraction * delay_scale_s / 2.0 + 1e-15
                            )
                        )
                        profile_span_ns = float(
                            (selected[-1] - selected[0] + 1)
                            * profile_grid_step_fraction
                            * delay_scale_s
                            * 1e9
                        )
                        components = int(1 + np.count_nonzero(np.diff(selected) > 1))
                        estimate_s = float(row["omp_refined_relative_delay_s"])
                        lower_s = max(0.0, estimate_s - omp_threshold_s)
                        upper_s = estimate_s + omp_threshold_s
                        omp_covers = lower_s <= relative_truth_s <= upper_s
                        omp_span_ns = (upper_s - lower_s) * 1e9
                        row.update(
                            {
                                "profile_conformal_covers_truth": profile_covers,
                                "profile_conformal_hull_span_ns": profile_span_ns,
                                "profile_conformal_components": components,
                                "omp_conformal_covers_truth": omp_covers,
                                "omp_conformal_interval_ns": [lower_s * 1e9, upper_s * 1e9],
                                "omp_conformal_interval_width_ns": omp_span_ns,
                            }
                        )
                        profile_coverage.append(profile_covers)
                        omp_coverage.append(omp_covers)
                        profile_spans_ns.append(profile_span_ns)
                        omp_spans_ns.append(omp_span_ns)
                        profile_components.append(components)
                        profile_map_errors_ns.append(
                            (float(row["profile_map_relative_delay_s"]) - relative_truth_s) * 1e9
                        )
                        omp_errors_ns.append((estimate_s - relative_truth_s) * 1e9)

                    profile_covered = np.asarray(profile_coverage, dtype=bool)
                    omp_covered = np.asarray(omp_coverage, dtype=bool)
                    coverage_delta = profile_covered.astype(np.float64) - omp_covered.astype(
                        np.float64
                    )
                    span_delta = np.asarray(profile_spans_ns) - np.asarray(omp_spans_ns)
                    bootstrap_rng = np.random.default_rng(
                        np.random.SeedSequence([seed, len(cases), 20261029])
                    )
                    coverage_bootstrap = bootstrap_rng.choice(
                        coverage_delta,
                        size=(bootstrap_draws, coverage_delta.size),
                        replace=True,
                    ).mean(axis=1)
                    span_bootstrap = bootstrap_rng.choice(
                        span_delta,
                        size=(bootstrap_draws, span_delta.size),
                        replace=True,
                    ).mean(axis=1)
                    profile_only = int(np.count_nonzero(profile_covered & ~omp_covered))
                    omp_only = int(np.count_nonzero(~profile_covered & omp_covered))
                    discordant = profile_only + omp_only
                    paired_p = (
                        float(binomtest(min(profile_only, omp_only), discordant, 0.5).pvalue)
                        if discordant
                        else 1.0
                    )
                    cases.append(
                        {
                            "tone_count": tone_count,
                            "tone_spacing_hz": float(tone_spacing_hz),
                            "bandwidth_hz": bandwidth_hz,
                            "snr_db": float(snr_db),
                            "separation_fraction_of_reciprocal_bandwidth": float(
                                separation_fraction
                            ),
                            "gain_phase_difference_deg": float(phase_difference_deg),
                            "relative_delay_truth_ns": relative_truth_s * 1e9,
                            "trials": trials,
                            "calibration_trials": calibration_trials,
                            "confidence_level": confidence_level,
                            "profile_grid_step_fraction": profile_grid_step_fraction,
                            "omp_grid_step_fraction": omp_grid_step_fraction,
                            "maximum_delay_fraction": maximum_delay_fraction,
                            "profile_conformal_threshold": profile_threshold,
                            "omp_conformal_absolute_error_threshold_ns": omp_threshold_s * 1e9,
                            "profile_conformal_coverage": float(profile_covered.mean()),
                            "omp_conformal_coverage": float(omp_covered.mean()),
                            "paired_profile_minus_omp_coverage_change": float(
                                coverage_delta.mean()
                            ),
                            "paired_profile_minus_omp_coverage_bootstrap_ci95": [
                                float(value)
                                for value in np.percentile(coverage_bootstrap, [2.5, 97.5])
                            ],
                            "paired_profile_only_coverage": profile_only,
                            "paired_omp_only_coverage": omp_only,
                            "paired_coverage_mcnemar_p_value": paired_p,
                            "profile_median_hull_span_ns": float(np.median(profile_spans_ns)),
                            "omp_median_interval_width_ns": float(np.median(omp_spans_ns)),
                            "paired_profile_minus_omp_mean_span_change_ns": float(
                                span_delta.mean()
                            ),
                            "paired_profile_minus_omp_mean_span_bootstrap_ci95_ns": [
                                float(value) for value in np.percentile(span_bootstrap, [2.5, 97.5])
                            ],
                            "profile_median_components": float(np.median(profile_components)),
                            "profile_map_rmse_ns": float(
                                np.sqrt(np.mean(np.square(profile_map_errors_ns)))
                            ),
                            "omp_refined_rmse_ns": float(
                                np.sqrt(np.mean(np.square(omp_errors_ns)))
                            ),
                            "omp_optimizer_successes": int(
                                sum(bool(row["omp_optimizer_success"]) for row in evaluation)
                            ),
                            "omp_path_order_reversals": int(
                                sum(bool(row["omp_path_order_reversal"]) for row in evaluation)
                            ),
                            "calibration_trials_records": calibration,
                            "evaluation_trials_records": evaluation,
                        }
                    )

    raw_p_values = np.asarray(
        [float(case["paired_coverage_mcnemar_p_value"]) for case in cases], dtype=np.float64
    )
    order = np.argsort(raw_p_values)
    adjusted = np.empty(raw_p_values.size, dtype=np.float64)
    running_max = 0.0
    for rank, index in enumerate(order):
        running_max = max(running_max, (raw_p_values.size - rank) * raw_p_values[int(index)])
        adjusted[int(index)] = min(1.0, running_max)
    for case, adjusted_p_value in zip(cases, adjusted, strict=True):
        case["paired_coverage_mcnemar_holm_p_value"] = float(adjusted_p_value)

    return {
        "schema": "loslab.identifiability_matched_estimators.v1",
        "evidence_grade": "PAIRED_ANALYTICAL_TWO_PATH_ESTIMATOR_SIMULATION",
        "seed": seed,
        "model": "two unordered complex specular paths in centered CFR tones with known iid circular complex AWGN variance",
        "methods": {
            "profile": "two-path profile likelihood over unordered delay pairs; split-conformal likelihood-ratio set",
            "classical_baseline": "two-path complex OMP with known path count, followed by local nonlinear least-squares refinement; split-conformal symmetric interval from absolute relative-delay calibration errors",
        },
        "calibration_protocol": "per-condition calibration and evaluation use disjoint seeded trials; both methods see identical complex observations within each trial",
        "paired_bootstrap_draws": bootstrap_draws,
        "cases": cases,
        "limitations": [
            "Both methods are given the true path count (two), tone grid, carrier-relative frequencies, and noise variance; no model-order selection is tested.",
            "The OMP baseline searches a finite 0-to-maximum-delay dictionary then uses a local optimizer; global minima outside its initialization basin remain possible.",
            "Calibration is per exact synthetic condition. Coverage does not transfer to unseen SNR, path phase, bandwidth, model mismatch, RT scenes, or measurements.",
            "The profile confidence support and OMP intervals are evaluated on a quantized relative-delay grid and a fixed two-specular-path model; diffuse energy and path-count uncertainty are absent.",
            "Per-condition conformal calibration may be operationally unavailable when the true channel condition is unknown; cross-condition calibration is a separate open study.",
        ],
    }
