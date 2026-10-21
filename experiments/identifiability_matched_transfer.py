"""Matched cross-condition calibration stress test for profile and OMP sets."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.stats import binomtest

from experiments.identifiability_profile_likelihood import _split_conformal_threshold

Condition = tuple[int, float, float, float]


def _holm_adjust(p_values: Sequence[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = [1.0] * len(p_values)
    running_max = 0.0
    for rank, index in enumerate(order):
        running_max = max(running_max, (len(p_values) - rank) * float(p_values[index]))
        adjusted[int(index)] = min(1.0, running_max)
    return adjusted


def analyze_matched_cross_condition_transfer(
    matched_result: dict[str, Any], *, seed: int, bootstrap_draws: int = 5_000
) -> dict[str, Any]:
    """Recalibrate each estimator on other conditions and evaluate paired rows.

    Calibration remains within tone count so the frequency grid is fixed. OMP
    delay errors are normalized by bandwidth before pooling, then converted back
    to seconds for the held-out interval. All cross-condition results are
    empirical shift diagnostics; conformal coverage guarantees do not transfer.
    """
    if matched_result.get("schema") != "loslab.identifiability_matched_estimators.v1":
        raise ValueError("matched_result must use the matched-estimator schema")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(bootstrap_draws, bool) or not isinstance(bootstrap_draws, int):
        raise TypeError("bootstrap_draws must be an integer")
    if bootstrap_draws <= 0:
        raise ValueError("bootstrap_draws must be positive")

    cases = matched_result.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("matched_result must contain cases")
    by_condition: dict[Condition, dict[str, Any]] = {}
    for case in cases:
        condition = (
            int(case["tone_count"]),
            float(case["snr_db"]),
            float(case["separation_fraction_of_reciprocal_bandwidth"]),
            float(case["gain_phase_difference_deg"]),
        )
        if condition in by_condition:
            raise ValueError(f"duplicate condition: {condition}")
        if float(case["bandwidth_hz"]) <= 0:
            raise ValueError("case bandwidth_hz must be positive")
        by_condition[condition] = case

    tone_counts = sorted({key[0] for key in by_condition})
    policies = (
        "leave_one_condition_out",
        "leave_one_snr_level_out",
        "leave_one_separation_level_out",
        "leave_one_gain_phase_level_out",
        "within_condition",
    )
    source_conditions: dict[tuple[str, Condition], list[Condition]] = {}
    for tone_count in tone_counts:
        tone_keys = [key for key in by_condition if key[0] == tone_count]
        snr_levels = {key[1] for key in tone_keys}
        separation_levels = {key[2] for key in tone_keys}
        phase_levels = {key[3] for key in tone_keys}
        expected = len(snr_levels) * len(separation_levels) * len(phase_levels)
        if min(len(snr_levels), len(separation_levels), len(phase_levels)) < 2:
            raise ValueError("each tone count needs at least two levels on every condition axis")
        if len(tone_keys) != expected:
            raise ValueError("conditions must form a complete factorial grid within tone count")
        for target in tone_keys:
            source_conditions[(policies[0], target)] = [key for key in tone_keys if key != target]
            source_conditions[(policies[1], target)] = [
                key for key in tone_keys if key[1] != target[1]
            ]
            source_conditions[(policies[2], target)] = [
                key for key in tone_keys if key[2] != target[2]
            ]
            source_conditions[(policies[3], target)] = [
                key for key in tone_keys if key[3] != target[3]
            ]
            source_conditions[(policies[4], target)] = [target]

    p_values: list[float] = []
    results: list[dict[str, Any]] = []
    bootstrap_rng = np.random.default_rng(seed)
    for policy_index, policy in enumerate(policies):
        for condition_index, condition in enumerate(sorted(by_condition)):
            case = by_condition[condition]
            sources = source_conditions[(policy, condition)]
            profile_scores = [
                float(row["profile_nonconformity"])
                for source in sources
                for row in by_condition[source]["calibration_trials_records"]
            ]
            bandwidth_hz = float(case["bandwidth_hz"])
            omp_normalized_errors = [
                float(row["omp_absolute_error_s"]) * float(by_condition[source]["bandwidth_hz"])
                for source in sources
                for row in by_condition[source]["calibration_trials_records"]
            ]
            profile_threshold = _split_conformal_threshold(profile_scores, 0.95)
            omp_normalized_threshold = _split_conformal_threshold(omp_normalized_errors, 0.95)
            evaluation = case["evaluation_trials_records"]
            profile_hits = np.asarray(
                [row["profile_nonconformity"] <= profile_threshold for row in evaluation],
                dtype=bool,
            )
            omp_hits = np.asarray(
                [
                    row["omp_absolute_error_s"] * bandwidth_hz <= omp_normalized_threshold
                    for row in evaluation
                ],
                dtype=bool,
            )
            difference = profile_hits.astype(np.float64) - omp_hits.astype(np.float64)
            draws = bootstrap_rng.choice(
                difference,
                size=(bootstrap_draws, difference.size),
                replace=True,
            ).mean(axis=1)
            profile_only = int(np.count_nonzero(profile_hits & ~omp_hits))
            omp_only = int(np.count_nonzero(omp_hits & ~profile_hits))
            discordant = profile_only + omp_only
            paired_p = (
                float(binomtest(profile_only, discordant, p=0.5).pvalue) if discordant else 1.0
            )
            p_values.append(paired_p)
            results.append(
                {
                    "policy": policy,
                    "tone_count": condition[0],
                    "snr_db": condition[1],
                    "separation_fraction_of_reciprocal_bandwidth": condition[2],
                    "gain_phase_difference_deg": condition[3],
                    "calibration_conditions": [list(source) for source in sources],
                    "calibration_trials": len(profile_scores),
                    "evaluation_trials": len(evaluation),
                    "profile_coverage": float(profile_hits.mean()),
                    "omp_coverage": float(omp_hits.mean()),
                    "paired_profile_minus_omp_coverage_change": float(difference.mean()),
                    "paired_coverage_bootstrap_ci95": [
                        float(value) for value in np.percentile(draws, [2.5, 97.5])
                    ],
                    "profile_only_coverage": profile_only,
                    "omp_only_coverage": omp_only,
                    "paired_mcnemar_p_value": paired_p,
                }
            )

    adjusted = _holm_adjust(p_values)
    for result, adjusted_p in zip(results, adjusted, strict=True):
        result["paired_mcnemar_holm_p_value"] = adjusted_p
    return {
        "schema": "loslab.identifiability_matched_transfer.v1",
        "evidence_grade": "MATCHED_ANALYTICAL_CROSS_CONDITION_CALIBRATION_STRESS_TEST",
        "calibration_protocol": (
            "Within each tone count, compare leave-one-condition, leave-one-factor-level, "
            "and within-condition split-conformal calibration on the exact paired evaluation rows."
        ),
        "input_schema": matched_result["schema"],
        "input_source_revision": matched_result.get("source_revision"),
        "seed": seed,
        "bootstrap_draws_per_case": bootstrap_draws,
        "multiplicity_correction": "Holm adjustment across every policy-condition paired test",
        "cases": results,
        "limitations": [
            "Calibration/evaluation are two-path analytical CFR simulations with known path count, noise variance, and frequency grid.",
            "Cross-condition scores are not exchangeable; empirical coverage is a stress test, not a conformal guarantee.",
            "Bootstrap intervals resample paired evaluation rows conditional on the observed calibration scores; they do not include calibration-sample uncertainty.",
            "Tone count is held fixed while pooling calibration; no tone-count transfer or independent-scene transfer is tested.",
            "OMP delay errors are normalized by reciprocal bandwidth; only the finite-grid OMP plus local-refinement baseline is evaluated.",
            "No diffuse multipath, model-order uncertainty, receiver estimation nuisance, learned method, RT scene, or measured channel is included.",
        ],
    }
