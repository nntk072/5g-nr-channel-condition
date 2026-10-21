"""Profile-likelihood uncertainty diagnostics for unordered two-path CFRs."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypedDict

import numpy as np
from artifacts.provenance import package_source_sha256, sha256_file, versions


class ProfileLikelihoodResult(TypedDict):
    relative_delay_grid_s: list[float]
    relative_delay_mass: list[float]
    relative_delay_log_likelihood_ratio: list[float]
    map_relative_delay_s: float
    mean_relative_delay_s: float
    interval_68_s: list[float]
    interval_68_mass: float
    interval_95_s: list[float]
    interval_95_mass: float
    relative_delay_entropy_nats: float
    effective_relative_delay_bins: float
    profiled_pair_count: int
    excluded_numerically_collinear_pairs: int


@dataclass(frozen=True)
class _PreparedDelayGrid:
    delays_s: np.ndarray
    steering: np.ndarray
    first: np.ndarray
    second: np.ndarray
    correlation: np.ndarray
    denominator: np.ndarray


def _prepare_delay_grid(frequencies_hz: np.ndarray, delay_grid_s: np.ndarray) -> _PreparedDelayGrid:
    steering = np.exp(-2j * np.pi * frequencies_hz[:, None] * delay_grid_s[None, :])
    steering /= np.sqrt(frequencies_hz.size)
    gram = steering.conj().T @ steering
    first, second = np.triu_indices(delay_grid_s.size, k=1)
    correlation = gram[first, second]
    denominator = 1.0 - np.abs(correlation) ** 2
    identifiable_pairs = denominator > 1e-12
    first = first[identifiable_pairs]
    second = second[identifiable_pairs]
    correlation = correlation[identifiable_pairs]
    denominator = denominator[identifiable_pairs]
    if first.size == 0:
        raise ValueError("delay grid has no numerically distinguishable path pairs")
    return _PreparedDelayGrid(
        delays_s=delay_grid_s,
        steering=steering,
        first=first,
        second=second,
        correlation=correlation,
        denominator=denominator,
    )


def _profile_prepared_grid(
    observed: np.ndarray, noise_variance: float, grid: _PreparedDelayGrid
) -> ProfileLikelihoodResult:
    matched = grid.steering.conj().T @ observed
    first_match = matched[grid.first]
    second_match = matched[grid.second]
    explained_energy = (
        np.abs(first_match) ** 2
        + np.abs(second_match) ** 2
        - 2.0 * np.real(first_match.conj() * grid.correlation * second_match)
    ) / grid.denominator
    observed_energy = float(np.vdot(observed, observed).real)
    explained_energy = np.clip(explained_energy, 0.0, observed_energy)
    residual_energy = np.maximum(observed_energy - explained_energy, 0.0)
    relative_indices = grid.second - grid.first
    pair_log_likelihood = -residual_energy / noise_variance
    relative_log_likelihood = np.full(grid.delays_s.size, -np.inf, dtype=np.float64)
    np.maximum.at(relative_log_likelihood, relative_indices, pair_log_likelihood)
    finite_relative = np.isfinite(relative_log_likelihood)
    relative_mass = np.zeros(grid.delays_s.size, dtype=np.float64)
    relative_log_likelihood_ratio = np.full(grid.delays_s.size, -1e6, dtype=np.float64)
    relative_log_likelihood_ratio[finite_relative] = relative_log_likelihood[
        finite_relative
    ] - float(np.max(relative_log_likelihood[finite_relative]))
    relative_mass[finite_relative] = np.exp(
        np.maximum(
            relative_log_likelihood[finite_relative]
            - float(np.max(relative_log_likelihood[finite_relative])),
            -745.0,
        )
    )
    relative_mass /= relative_mass.sum()
    relative_delays = grid.delays_s - grid.delays_s[0]
    cumulative_mass = np.cumsum(relative_mass)

    def interval(level: float) -> tuple[float, float, float]:
        tail = (1.0 - level) / 2.0
        lower_index = min(
            int(np.searchsorted(cumulative_mass, tail, side="left")), grid.delays_s.size - 1
        )
        upper_index = min(
            int(np.searchsorted(cumulative_mass, 1.0 - tail, side="left")),
            grid.delays_s.size - 1,
        )
        mass = float(relative_mass[lower_index : upper_index + 1].sum())
        return (
            float(relative_delays[lower_index]),
            float(relative_delays[upper_index]),
            mass,
        )

    interval_68 = interval(0.68)
    interval_95 = interval(0.95)
    map_index = int(np.argmax(relative_mass))
    positive_mass = relative_mass[relative_mass > 0]
    entropy = float(-np.sum(positive_mass * np.log(positive_mass)))
    return {
        "relative_delay_grid_s": relative_delays.tolist(),
        "relative_delay_mass": relative_mass.tolist(),
        "relative_delay_log_likelihood_ratio": relative_log_likelihood_ratio.tolist(),
        "map_relative_delay_s": float(relative_delays[map_index]),
        "mean_relative_delay_s": float(np.dot(relative_delays, relative_mass)),
        "interval_68_s": [interval_68[0], interval_68[1]],
        "interval_68_mass": interval_68[2],
        "interval_95_s": [interval_95[0], interval_95[1]],
        "interval_95_mass": interval_95[2],
        "relative_delay_entropy_nats": entropy,
        "effective_relative_delay_bins": float(np.exp(entropy)),
        "profiled_pair_count": int(grid.first.size),
        "excluded_numerically_collinear_pairs": int(
            grid.delays_s.size * (grid.delays_s.size - 1) // 2 - grid.first.size
        ),
    }


def profile_relative_delay_likelihood(
    observed_cfr: np.ndarray,
    frequencies_hz: np.ndarray,
    noise_variance: float,
    delay_grid_s: np.ndarray,
) -> ProfileLikelihoodResult:
    """Profile complex path gains and common timing over unordered delay pairs.

    The returned normalized likelihood mass is a diagnostic over a finite,
    uniformly weighted grid. It is not a Bayesian posterior or a calibrated
    confidence interval. Path count is fixed at two and noise variance is
    treated as known.
    """
    observed = np.asarray(observed_cfr, dtype=np.complex128)
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    delays = np.asarray(delay_grid_s, dtype=np.float64)
    if observed.ndim != 1 or observed.size < 2:
        raise ValueError("observed_cfr must be a one-dimensional vector of length >= 2")
    if frequencies.shape != observed.shape or not np.all(np.isfinite(frequencies)):
        raise ValueError("frequencies_hz must be finite and match observed_cfr")
    if not np.all(np.isfinite(observed.real)) or not np.all(np.isfinite(observed.imag)):
        raise ValueError("observed_cfr must contain finite complex values")
    if not math.isfinite(noise_variance) or noise_variance <= 0:
        raise ValueError("noise_variance must be finite and positive")
    if delays.ndim != 1 or delays.size < 3 or not np.all(np.isfinite(delays)):
        raise ValueError("delay_grid_s must be a finite one-dimensional grid of length >= 3")
    if np.any(np.diff(delays) <= 0):
        raise ValueError("delay_grid_s must be strictly increasing")
    if not np.allclose(np.diff(delays), np.diff(delays)[0], rtol=1e-9, atol=0.0):
        raise ValueError("delay_grid_s must be uniformly spaced")

    grid = _prepare_delay_grid(frequencies, delays)
    return _profile_prepared_grid(observed, noise_variance, grid)


def _wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    if trials < 1:
        return (math.nan, math.nan)
    z = 1.959963984540054
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    half_width = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials**2))
        / denominator
    )
    return (center - half_width, center + half_width)


def _split_conformal_threshold(scores: Sequence[float], level: float) -> float:
    """Finite-sample order-statistic threshold for exchangeable calibration scores."""
    ordered = np.sort(np.asarray(scores, dtype=np.float64))
    if ordered.ndim != 1 or ordered.size == 0:
        raise ValueError("calibration scores must be a nonempty one-dimensional sequence")
    if not math.isfinite(level) or not 0.0 < level < 1.0:
        raise ValueError("level must be finite and in (0, 1)")
    rank = math.ceil((ordered.size + 1) * level)
    if rank > ordered.size:
        return math.inf
    return float(ordered[rank - 1])


def build_profile_likelihood_sweep(
    *,
    seed: int,
    trials: int,
    tone_counts: Sequence[int],
    tone_spacing_hz: float,
    snr_db_values: Sequence[float],
    separation_fractions: Sequence[float],
    gain_phase_differences_deg: Sequence[float],
    calibration_trials: int = 128,
    delay_grid_step_fraction: float = 1.0 / 128.0,
    maximum_delay_fraction: float = 2.0,
) -> dict[str, Any]:
    """Measure profile-likelihood interval coverage on paired two-path CFRs."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(trials, bool) or not isinstance(trials, int) or trials < 2:
        raise ValueError("trials must be an integer of at least 2")
    if (
        isinstance(calibration_trials, bool)
        or not isinstance(calibration_trials, int)
        or calibration_trials < 19
    ):
        raise ValueError("calibration_trials must be an integer of at least 19")
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
    if not math.isfinite(delay_grid_step_fraction) or delay_grid_step_fraction <= 0:
        raise ValueError("delay_grid_step_fraction must be finite and positive")
    if not math.isfinite(maximum_delay_fraction) or maximum_delay_fraction <= 0:
        raise ValueError("maximum_delay_fraction must be finite and positive")
    if maximum_delay_fraction <= max(separation_fractions) + 0.3:
        raise ValueError("maximum_delay_fraction must exceed max separation plus timing offset")

    rng = np.random.default_rng(seed)
    cases: list[dict[str, object]] = []
    for tone_count in tone_counts:
        frequencies = (np.arange(tone_count, dtype=np.float64) - tone_count // 2) * tone_spacing_hz
        bandwidth_hz = float(np.ptp(frequencies))
        delay_scale_s = 1.0 / bandwidth_hz
        delay_grid = (
            np.arange(
                0.0,
                maximum_delay_fraction + delay_grid_step_fraction / 2.0,
                delay_grid_step_fraction,
            )
            * delay_scale_s
        )
        prepared_grid = _prepare_delay_grid(frequencies, delay_grid)
        for snr_db in snr_db_values:
            for separation_fraction in separation_fractions:
                relative_delay_s = separation_fraction * delay_scale_s
                for phase_difference_deg in gain_phase_differences_deg:
                    gains = np.asarray(
                        [1.0 + 0j, 0.7 * np.exp(1j * np.deg2rad(phase_difference_deg))],
                        dtype=np.complex128,
                    )
                    reference = np.exp(
                        -2j
                        * np.pi
                        * frequencies[:, None]
                        * np.asarray([0.0, relative_delay_s])[None, :]
                    )
                    signal_power = float(np.mean(np.abs(reference @ gains) ** 2))
                    noise_variance = signal_power / (10.0 ** (snr_db / 10.0))
                    calibration_scores: list[float] = []
                    calibration_records: list[dict[str, float | int]] = []
                    test_profile_likelihood_ratios: list[np.ndarray] = []
                    test_covered_68 = 0
                    test_covered_95 = 0
                    map_errors_ns: list[float] = []
                    widths_68_ns: list[float] = []
                    widths_95_ns: list[float] = []
                    effective_bins: list[float] = []
                    per_trial: list[dict[str, object]] = []
                    for trial_index in range(calibration_trials + trials):
                        timing_offset_s = float(rng.uniform(0.1, 0.3) * delay_scale_s)
                        timing_offset_fraction = timing_offset_s / delay_scale_s
                        true_delays = np.asarray(
                            [timing_offset_s, timing_offset_s + relative_delay_s]
                        )
                        steering = np.exp(-2j * np.pi * frequencies[:, None] * true_delays[None, :])
                        noiseless = steering @ gains
                        noise = np.sqrt(noise_variance / 2.0) * (
                            rng.standard_normal(tone_count) + 1j * rng.standard_normal(tone_count)
                        )
                        observed = noiseless + noise
                        fit = _profile_prepared_grid(observed, noise_variance, prepared_grid)
                        relative_grid = np.asarray(fit["relative_delay_grid_s"])
                        profile_likelihood_ratio = np.asarray(
                            fit["relative_delay_log_likelihood_ratio"]
                        )
                        nearest_truth_index = int(
                            np.argmin(np.abs(relative_grid - relative_delay_s))
                        )
                        calibration_score = max(
                            0.0, -float(profile_likelihood_ratio[nearest_truth_index])
                        )
                        if trial_index < calibration_trials:
                            calibration_scores.append(calibration_score)
                            calibration_records.append(
                                {
                                    "trial": trial_index,
                                    "common_timing_offset_fraction_of_reciprocal_bandwidth": timing_offset_fraction,
                                    "profile_likelihood_nonconformity": calibration_score,
                                }
                            )
                            continue

                        interval_68 = fit["interval_68_s"]
                        interval_95 = fit["interval_95_s"]
                        lower_68, upper_68 = (float(value) for value in interval_68)
                        lower_95, upper_95 = (float(value) for value in interval_95)
                        hit_68 = lower_68 <= relative_delay_s <= upper_68
                        hit_95 = lower_95 <= relative_delay_s <= upper_95
                        test_covered_68 += int(hit_68)
                        test_covered_95 += int(hit_95)
                        map_error_ns = (float(fit["map_relative_delay_s"]) - relative_delay_s) * 1e9
                        width_68_ns = (upper_68 - lower_68) * 1e9
                        width_95_ns = (upper_95 - lower_95) * 1e9
                        map_errors_ns.append(map_error_ns)
                        widths_68_ns.append(width_68_ns)
                        widths_95_ns.append(width_95_ns)
                        effective_bins.append(float(fit["effective_relative_delay_bins"]))
                        test_profile_likelihood_ratios.append(profile_likelihood_ratio)
                        per_trial.append(
                            {
                                "trial": trial_index - calibration_trials,
                                "split": "evaluation",
                                "common_timing_offset_fraction_of_reciprocal_bandwidth": timing_offset_fraction,
                                "map_error_ns": map_error_ns,
                                "interval_68_width_ns": width_68_ns,
                                "interval_95_width_ns": width_95_ns,
                                "interval_68_covers_truth": hit_68,
                                "interval_95_covers_truth": hit_95,
                                "effective_relative_delay_bins": fit[
                                    "effective_relative_delay_bins"
                                ],
                                "profile_likelihood_nonconformity": calibration_score,
                            }
                        )
                    threshold_68 = _split_conformal_threshold(calibration_scores, 0.68)
                    threshold_95 = _split_conformal_threshold(calibration_scores, 0.95)
                    calibrated_covered_68 = 0
                    calibrated_covered_95 = 0
                    calibrated_widths_68_ns: list[float] = []
                    calibrated_widths_95_ns: list[float] = []
                    calibrated_components_95: list[int] = []
                    for evaluation_index, log_likelihood_ratio in enumerate(
                        test_profile_likelihood_ratios
                    ):
                        selected_68 = np.flatnonzero(log_likelihood_ratio >= -threshold_68)
                        selected_95 = np.flatnonzero(log_likelihood_ratio >= -threshold_95)
                        grid_cell_half_width_s = delay_grid_step_fraction * delay_scale_s / 2.0
                        hit_68 = bool(
                            np.any(
                                np.abs(relative_grid[selected_68] - relative_delay_s)
                                <= grid_cell_half_width_s + 1e-15
                            )
                        )
                        hit_95 = bool(
                            np.any(
                                np.abs(relative_grid[selected_95] - relative_delay_s)
                                <= grid_cell_half_width_s + 1e-15
                            )
                        )
                        span_68_ns = float(
                            (selected_68[-1] - selected_68[0] + 1)
                            * delay_grid_step_fraction
                            * delay_scale_s
                            * 1e9
                        )
                        span_95_ns = float(
                            (selected_95[-1] - selected_95[0] + 1)
                            * delay_grid_step_fraction
                            * delay_scale_s
                            * 1e9
                        )
                        components_95 = int(1 + np.count_nonzero(np.diff(selected_95) > 1))
                        calibrated_covered_68 += int(hit_68)
                        calibrated_covered_95 += int(hit_95)
                        calibrated_widths_68_ns.append(span_68_ns)
                        calibrated_widths_95_ns.append(span_95_ns)
                        calibrated_components_95.append(components_95)
                        per_trial[evaluation_index].update(
                            {
                                "split_conformal_68_covers_truth": hit_68,
                                "split_conformal_68_set_span_ns": span_68_ns,
                                "split_conformal_95_covers_truth": hit_95,
                                "split_conformal_95_set_span_ns": span_95_ns,
                                "split_conformal_95_connected_components": components_95,
                            }
                        )
                    lo_68, hi_68 = _wilson_interval(test_covered_68, trials)
                    lo_95, hi_95 = _wilson_interval(test_covered_95, trials)
                    calibrated_lo_68, calibrated_hi_68 = _wilson_interval(
                        calibrated_covered_68, trials
                    )
                    calibrated_lo_95, calibrated_hi_95 = _wilson_interval(
                        calibrated_covered_95, trials
                    )
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
                            "calibration_trials": calibration_trials,
                            "calibration_records": calibration_records,
                            "delay_grid_step_fraction_of_reciprocal_bandwidth": (
                                delay_grid_step_fraction
                            ),
                            "maximum_delay_fraction_of_reciprocal_bandwidth": (
                                maximum_delay_fraction
                            ),
                            "interval_68_empirical_coverage": test_covered_68 / trials,
                            "interval_68_wilson_95_ci": [lo_68, hi_68],
                            "interval_95_empirical_coverage": test_covered_95 / trials,
                            "interval_95_wilson_95_ci": [lo_95, hi_95],
                            "split_conformal_68_threshold": threshold_68,
                            "split_conformal_68_empirical_coverage": calibrated_covered_68 / trials,
                            "split_conformal_68_wilson_95_ci": [
                                calibrated_lo_68,
                                calibrated_hi_68,
                            ],
                            "split_conformal_68_median_set_span_ns": float(
                                np.median(calibrated_widths_68_ns)
                            ),
                            "split_conformal_95_threshold": threshold_95,
                            "split_conformal_95_empirical_coverage": calibrated_covered_95 / trials,
                            "split_conformal_95_wilson_95_ci": [
                                calibrated_lo_95,
                                calibrated_hi_95,
                            ],
                            "split_conformal_95_median_set_span_ns": float(
                                np.median(calibrated_widths_95_ns)
                            ),
                            "split_conformal_95_median_connected_components": float(
                                np.median(calibrated_components_95)
                            ),
                            "map_error_rmse_ns": float(np.sqrt(np.mean(np.square(map_errors_ns)))),
                            "median_interval_68_width_ns": float(np.median(widths_68_ns)),
                            "median_interval_95_width_ns": float(np.median(widths_95_ns)),
                            "median_effective_relative_delay_bins": float(
                                np.median(effective_bins)
                            ),
                            "per_trial": per_trial,
                        }
                    )
    runtime = versions()
    return {
        "schema": "loslab.identifiability_profile_likelihood.v1",
        "evidence_grade": "CONDITION_CALIBRATED_ANALYTICAL_TWO_PATH_PROFILE_LIKELIHOOD_DIAGNOSTIC",
        "model": "two unordered complex specular paths in centered CFR tones with iid circular complex AWGN",
        "likelihood": "complex gains and common timing profiled by least squares over unordered delay pairs; normalized across relative-delay grid",
        "seed": seed,
        "source_revision": runtime.get("git_sha"),
        "package_source_sha256": package_source_sha256(),
        "experiment_module_sha256": sha256_file(__file__),
        "versions": runtime,
        "cases": cases,
        "limitations": [
            "The normalized profile-likelihood mass is not a Bayesian posterior or a calibrated confidence distribution.",
            "Path count is fixed at two and noise variance is treated as known; diffuse paths, channel-estimation error, and model mismatch are absent.",
            "Common timing and complex gains are profiled over a finite grid; uncertainty widths depend on that grid and are not Bayesian credible intervals.",
            "Only one static analytical CFR family is tested; there is no receiver acquisition, independent geometry, Sionna scene transfer, or learned baseline.",
            "Profile intervals and conformal supports are quantized to delay-grid cells; central intervals may conceal multimodal or disconnected ambiguity sets.",
            "Split-conformal thresholds are calibrated separately at each known synthetic condition; they do not establish transfer to unseen SNRs, path phases, geometries, or scenes.",
        ],
    }
