"""Leave-one-condition-out calibration stress tests for multipath delay sets."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from artifacts.provenance import package_source_sha256, sha256_file, versions

from experiments.identifiability_profile_likelihood import (
    ProfileLikelihoodResult,
    _prepare_delay_grid,
    _profile_prepared_grid,
    _split_conformal_threshold,
    _wilson_interval,
)


def _nonconformity(
    fit: ProfileLikelihoodResult, relative_delay_s: float
) -> tuple[float, np.ndarray, np.ndarray]:
    grid = np.asarray(fit["relative_delay_grid_s"], dtype=np.float64)
    log_likelihood_ratio = np.asarray(fit["relative_delay_log_likelihood_ratio"], dtype=np.float64)
    nearest = int(np.argmin(np.abs(grid - relative_delay_s)))
    score = max(0.0, -float(log_likelihood_ratio[nearest]))
    return score, grid, log_likelihood_ratio


def _profile_set_summary(
    log_likelihood_ratio: np.ndarray,
    relative_grid_s: np.ndarray,
    threshold: float,
    truth_s: float,
    grid_step_s: float,
) -> tuple[bool, float, int]:
    selected = np.flatnonzero(log_likelihood_ratio >= -threshold)
    covers = bool(np.any(np.abs(relative_grid_s[selected] - truth_s) <= grid_step_s / 2.0 + 1e-15))
    span_ns = float((selected[-1] - selected[0] + 1) * grid_step_s * 1e9)
    components = int(1 + np.count_nonzero(np.diff(selected) > 1))
    return covers, span_ns, components


def build_cross_condition_calibration_sweep(
    *,
    seed: int,
    trials: int,
    calibration_trials_per_condition: int,
    tone_counts: Sequence[int],
    tone_spacing_hz: float,
    snr_db_values: Sequence[float],
    separation_fractions: Sequence[float],
    gain_phase_differences_deg: Sequence[float],
    delay_grid_step_fraction: float = 1.0 / 128.0,
    maximum_delay_fraction: float = 2.0,
) -> dict[str, Any]:
    """Evaluate profile-set calibration with the target condition excluded.

    Calibration scores are pooled across all other SNR/separation/phase cells
    at the same tone count. The target cell's calibration sample is retained
    as a matched in-condition control. Cross-condition coverage is an empirical
    stress test; split-conformal guarantees do not survive this distribution
    shift without exchangeability.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(trials, bool) or not isinstance(trials, int) or trials < 2:
        raise ValueError("trials must be an integer of at least 2")
    if (
        isinstance(calibration_trials_per_condition, bool)
        or not isinstance(calibration_trials_per_condition, int)
        or calibration_trials_per_condition < 19
    ):
        raise ValueError("calibration_trials_per_condition must be an integer of at least 19")
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
    if len(set(tone_counts)) != len(tone_counts):
        raise ValueError("tone_counts must not contain duplicates")
    if len(set(snr_db_values)) != len(snr_db_values):
        raise ValueError("snr_db_values must not contain duplicates")
    if len(set(separation_fractions)) != len(separation_fractions):
        raise ValueError("separation_fractions must not contain duplicates")
    if len(set(gain_phase_differences_deg)) != len(gain_phase_differences_deg):
        raise ValueError("gain_phase_differences_deg must not contain duplicates")
    if min(len(snr_db_values), len(separation_fractions), len(gain_phase_differences_deg)) < 2:
        raise ValueError("each condition axis must contain at least two levels")

    rng = np.random.default_rng(seed)
    cases: list[dict[str, Any]] = []
    for tone_count in tone_counts:
        frequencies = (np.arange(tone_count, dtype=np.float64) - tone_count // 2) * tone_spacing_hz
        bandwidth_hz = float(np.ptp(frequencies))
        delay_scale_s = 1.0 / bandwidth_hz
        grid_step_s = delay_grid_step_fraction * delay_scale_s
        delay_grid = (
            np.arange(
                0.0,
                maximum_delay_fraction + delay_grid_step_fraction / 2.0,
                delay_grid_step_fraction,
            )
            * delay_scale_s
        )
        prepared_grid = _prepare_delay_grid(frequencies, delay_grid)
        keys = [
            (float(snr), float(separation), float(phase))
            for snr in snr_db_values
            for separation in separation_fractions
            for phase in gain_phase_differences_deg
        ]
        calibration: dict[tuple[float, float, float], list[dict[str, float | int]]] = {}
        evaluation: dict[tuple[float, float, float], list[dict[str, Any]]] = {}

        for snr_db, separation_fraction, phase_difference_deg in keys:
            key = (snr_db, separation_fraction, phase_difference_deg)
            relative_delay_s = separation_fraction * delay_scale_s
            gains = np.asarray(
                [1.0 + 0j, 0.7 * np.exp(1j * np.deg2rad(phase_difference_deg))],
                dtype=np.complex128,
            )
            reference = np.exp(
                -2j * np.pi * frequencies[:, None] * np.asarray([0.0, relative_delay_s])[None, :]
            )
            signal_power = float(np.mean(np.abs(reference @ gains) ** 2))
            noise_variance = signal_power / (10.0 ** (snr_db / 10.0))
            calibration[key] = []
            evaluation[key] = []
            for split, sample_count in (
                ("calibration", calibration_trials_per_condition),
                ("evaluation", trials),
            ):
                for trial_index in range(sample_count):
                    timing_offset_s = float(rng.uniform(0.1, 0.3) * delay_scale_s)
                    true_delays_s = np.asarray(
                        [timing_offset_s, timing_offset_s + relative_delay_s]
                    )
                    steering = np.exp(-2j * np.pi * frequencies[:, None] * true_delays_s[None, :])
                    noiseless = steering @ gains
                    noise = np.sqrt(noise_variance / 2.0) * (
                        rng.standard_normal(tone_count) + 1j * rng.standard_normal(tone_count)
                    )
                    observed = noiseless + noise
                    fit = _profile_prepared_grid(observed, noise_variance, prepared_grid)
                    score, relative_grid, log_likelihood_ratio = _nonconformity(
                        fit, relative_delay_s
                    )
                    if split == "calibration":
                        calibration[key].append(
                            {
                                "trial": trial_index,
                                "common_timing_offset_fraction_of_reciprocal_bandwidth": (
                                    timing_offset_s / delay_scale_s
                                ),
                                "profile_likelihood_nonconformity": score,
                            }
                        )
                    else:
                        evaluation[key].append(
                            {
                                "trial": trial_index,
                                "common_timing_offset_fraction_of_reciprocal_bandwidth": (
                                    timing_offset_s / delay_scale_s
                                ),
                                "profile_likelihood_nonconformity": score,
                                "map_error_ns": (
                                    float(fit["map_relative_delay_s"]) - relative_delay_s
                                )
                                * 1e9,
                                "central_95_interval_covers_truth": (
                                    float(fit["interval_95_s"][0])
                                    <= relative_delay_s
                                    <= float(fit["interval_95_s"][1])
                                ),
                                "central_95_interval_width_ns": (
                                    float(fit["interval_95_s"][1]) - float(fit["interval_95_s"][0])
                                )
                                * 1e9,
                                "relative_delay_grid_s": relative_grid,
                                "relative_delay_log_likelihood_ratio": log_likelihood_ratio,
                            }
                        )

        for key in keys:
            snr_db, separation_fraction, phase_difference_deg = key
            relative_delay_s = separation_fraction * delay_scale_s
            own_scores = [
                float(record["profile_likelihood_nonconformity"]) for record in calibration[key]
            ]
            source_keys: dict[str, list[tuple[float, float, float]]] = {
                "leave_one_condition_out": [other for other in keys if other != key],
                "leave_one_snr_level_out": [other for other in keys if other[0] != snr_db],
                "leave_one_separation_level_out": [
                    other for other in keys if other[1] != separation_fraction
                ],
                "leave_one_gain_phase_level_out": [
                    other for other in keys if other[2] != phase_difference_deg
                ],
            }
            policy_scores: dict[str, list[float]] = {
                name: [
                    float(record["profile_likelihood_nonconformity"])
                    for other in source_conditions
                    for record in calibration[other]
                ]
                for name, source_conditions in source_keys.items()
            }
            policy_scores["within_condition"] = own_scores
            policy_thresholds = {
                name: _split_conformal_threshold(scores, 0.95)
                for name, scores in policy_scores.items()
            }
            policy_hits = dict.fromkeys(policy_thresholds, 0)
            policy_spans: dict[str, list[float]] = {name: [] for name in policy_thresholds}
            policy_components: dict[str, list[int]] = {name: [] for name in policy_thresholds}
            evaluation_records: list[dict[str, Any]] = []
            for record in evaluation[key]:
                grid = np.asarray(record.pop("relative_delay_grid_s"), dtype=np.float64)
                log_ratio = np.asarray(
                    record.pop("relative_delay_log_likelihood_ratio"), dtype=np.float64
                )
                set_results: dict[str, dict[str, Any]] = {}
                for name, threshold in policy_thresholds.items():
                    hit, span, components = _profile_set_summary(
                        log_ratio, grid, threshold, relative_delay_s, grid_step_s
                    )
                    policy_hits[name] += int(hit)
                    policy_spans[name].append(span)
                    policy_components[name].append(components)
                    set_results[name] = {
                        "covers_truth": hit,
                        "set_span_ns": span,
                        "connected_components": components,
                    }
                evaluation_records.append(
                    {
                        **record,
                        "calibrated_sets_95": set_results,
                    }
                )
            policy_results: dict[str, dict[str, Any]] = {}
            for name, threshold in policy_thresholds.items():
                confidence_interval = _wilson_interval(policy_hits[name], trials)
                source_conditions = source_keys.get(name, [key])
                policy_results[name] = {
                    "threshold": threshold,
                    "calibration_records": len(policy_scores[name]),
                    "source_conditions": [
                        {
                            "snr_db": source[0],
                            "separation_fraction": source[1],
                            "gain_phase_difference_deg": source[2],
                        }
                        for source in source_conditions
                    ],
                    "empirical_coverage": policy_hits[name] / trials,
                    "wilson_95_ci": list(confidence_interval),
                    "median_set_span_ns": float(np.median(policy_spans[name])),
                    "median_connected_components": float(np.median(policy_components[name])),
                }
            condition_holdout = policy_results["leave_one_condition_out"]
            within_condition = policy_results["within_condition"]
            cases.append(
                {
                    "tone_count": tone_count,
                    "tone_spacing_hz": tone_spacing_hz,
                    "bandwidth_hz": bandwidth_hz,
                    "snr_db": snr_db,
                    "separation_fraction_of_reciprocal_bandwidth": separation_fraction,
                    "gain_phase_difference_deg": phase_difference_deg,
                    "trials": trials,
                    "calibration_trials_per_condition": calibration_trials_per_condition,
                    "leave_one_condition_out_95_threshold": condition_holdout["threshold"],
                    "leave_one_condition_out_95_empirical_coverage": condition_holdout[
                        "empirical_coverage"
                    ],
                    "leave_one_condition_out_95_wilson_ci": condition_holdout["wilson_95_ci"],
                    "leave_one_condition_out_95_median_set_span_ns": condition_holdout[
                        "median_set_span_ns"
                    ],
                    "within_condition_95_threshold": within_condition["threshold"],
                    "within_condition_95_empirical_coverage": within_condition[
                        "empirical_coverage"
                    ],
                    "within_condition_95_wilson_ci": within_condition["wilson_95_ci"],
                    "within_condition_95_median_set_span_ns": within_condition[
                        "median_set_span_ns"
                    ],
                    "calibration_transfer_policies": policy_results,
                    "map_error_rmse_ns": float(
                        np.sqrt(
                            np.mean(
                                [
                                    float(record["map_error_ns"]) ** 2
                                    for record in evaluation_records
                                ]
                            )
                        )
                    ),
                    "central_95_interval_empirical_coverage": float(
                        np.mean(
                            [
                                bool(record["central_95_interval_covers_truth"])
                                for record in evaluation_records
                            ]
                        )
                    ),
                    "evaluation_records": evaluation_records,
                    "calibration_nonconformity_scores": [
                        float(record["profile_likelihood_nonconformity"])
                        for record in calibration[key]
                    ],
                }
            )

    runtime = versions()
    return {
        "schema": "loslab.identifiability_cross_condition.v2",
        "evidence_grade": "FACTOR_LEVEL_HOLDOUT_ANALYTICAL_CALIBRATION_STRESS_TEST",
        "model": "two unordered complex specular paths in centered CFR tones with iid circular complex AWGN",
        "path_gain_magnitudes": [1.0, 0.7],
        "common_timing_offset_fraction_of_reciprocal_bandwidth": [0.1, 0.3],
        "delay_grid_step_fraction_of_reciprocal_bandwidth": delay_grid_step_fraction,
        "maximum_delay_fraction_of_reciprocal_bandwidth": maximum_delay_fraction,
        "calibration_protocol": "For each tone count, evaluate four calibration-transfer policies: hold out the exact SNR/separation/phase cell, hold out each target SNR level, hold out each target separation level, and hold out each target gain-phase level; compare all with a same-cell calibration control.",
        "seed": seed,
        "source_revision": runtime.get("git_sha"),
        "package_source_sha256": package_source_sha256(),
        "experiment_module_sha256": sha256_file(__file__),
        "cases": cases,
        "versions": runtime,
        "limitations": [
            "Cross-condition evaluation violates the exchangeability assumption needed for a split-conformal coverage guarantee; report empirical coverage only.",
            "Calibration pools keep tone count fixed; factor-level holdouts exclude one entire SNR, separation, or phase level. This is a synthetic parameter shift, not independent-scene transfer.",
            "Path count, noise variance, tone grid, and channel family are known; diffuse energy, channel-estimation errors, receiver acquisition, and model-order uncertainty are absent.",
            "Coverage is evaluated on quantized delay-grid support cells and depends on the 1/(128B) resolution and 0–2/B support.",
            "No learned method or neural baseline is included; results test uncertainty calibration only.",
        ],
    }
