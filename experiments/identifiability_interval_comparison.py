"""Paired calibration/efficiency comparison for two-path delay uncertainty sets."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import binomtest


def compare_profile_intervals(
    study: dict[str, Any], *, seed: int = 20261028, bootstrap_draws: int = 10_000
) -> dict[str, Any]:
    """Compare nominal profile intervals with same-condition conformal sets.

    The two methods are evaluated on identical held-out trials in every
    analytical condition. Exact paired tests compare coverage; bootstrap
    intervals summarize paired span differences. Holm adjustment covers all
    supplied condition cells. These calculations do not establish scene-level
    or real-channel calibration.
    """
    if study.get("schema") != "loslab.identifiability_profile_likelihood.v1":
        raise ValueError("unsupported profile-likelihood study schema")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(bootstrap_draws, bool) or not isinstance(bootstrap_draws, int):
        raise TypeError("bootstrap_draws must be an integer")
    if bootstrap_draws <= 0:
        raise ValueError("bootstrap_draws must be positive")
    cases = study.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("study must contain a nonempty cases list")

    rng = np.random.default_rng(seed)
    comparisons: list[dict[str, Any]] = []
    raw_p_values: list[float] = []
    all_profile_covered: list[np.ndarray] = []
    all_conformal_covered: list[np.ndarray] = []
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("every case must be an object")
        records = case.get("per_trial")
        if not isinstance(records, list) or not records:
            raise ValueError("every case must contain nonempty per_trial records")
        evaluation = [row for row in records if row.get("split") == "evaluation"]
        trial_ids = [row.get("trial") for row in evaluation]
        if not evaluation or len(set(trial_ids)) != len(trial_ids):
            raise ValueError("each case needs unique held-out evaluation trials")
        if len(evaluation) != case.get("trials"):
            raise ValueError("held-out trial count does not match case metadata")
        required_fields = (
            "interval_95_covers_truth",
            "interval_95_width_ns",
            "split_conformal_95_covers_truth",
            "split_conformal_95_set_span_ns",
        )
        if any(any(field not in row for field in required_fields) for row in evaluation):
            raise ValueError("evaluation trials are missing paired coverage or width outcomes")
        if any(
            not isinstance(row[field], bool)
            for row in evaluation
            for field in ("interval_95_covers_truth", "split_conformal_95_covers_truth")
        ):
            raise ValueError("paired coverage outcomes must be booleans")
        if any(
            isinstance(row[field], bool) or not isinstance(row[field], (int, float))
            for row in evaluation
            for field in ("interval_95_width_ns", "split_conformal_95_set_span_ns")
        ):
            raise ValueError("paired uncertainty widths must be numeric")
        profile_covered = np.asarray(
            [row["interval_95_covers_truth"] for row in evaluation], dtype=bool
        )
        conformal_covered = np.asarray(
            [row["split_conformal_95_covers_truth"] for row in evaluation], dtype=bool
        )
        profile_width = np.asarray(
            [row["interval_95_width_ns"] for row in evaluation], dtype=np.float64
        )
        conformal_width = np.asarray(
            [row["split_conformal_95_set_span_ns"] for row in evaluation], dtype=np.float64
        )
        if (
            not np.isfinite(profile_width).all()
            or not np.isfinite(conformal_width).all()
            or np.any(profile_width < 0)
            or np.any(conformal_width < 0)
        ):
            raise ValueError("paired uncertainty widths must be finite and nonnegative")

        improvement = int(np.count_nonzero(~profile_covered & conformal_covered))
        regression = int(np.count_nonzero(profile_covered & ~conformal_covered))
        discordant = improvement + regression
        p_value = (
            float(binomtest(min(improvement, regression), discordant, 0.5).pvalue)
            if discordant
            else 1.0
        )
        raw_p_values.append(p_value)
        coverage_delta = conformal_covered.astype(np.float64) - profile_covered.astype(np.float64)
        span_delta = conformal_width - profile_width
        coverage_bootstrap = rng.choice(
            coverage_delta, size=(bootstrap_draws, coverage_delta.size), replace=True
        ).mean(axis=1)
        span_bootstrap = rng.choice(
            span_delta, size=(bootstrap_draws, span_delta.size), replace=True
        ).mean(axis=1)
        all_profile_covered.append(profile_covered)
        all_conformal_covered.append(conformal_covered)
        comparisons.append(
            {
                "tone_count": int(case["tone_count"]),
                "snr_db": float(case["snr_db"]),
                "separation_fraction_of_reciprocal_bandwidth": float(
                    case["separation_fraction_of_reciprocal_bandwidth"]
                ),
                "gain_phase_difference_deg": float(case["gain_phase_difference_deg"]),
                "evaluation_trials": len(evaluation),
                "profile_interval_coverage": float(profile_covered.mean()),
                "conformal_set_coverage": float(conformal_covered.mean()),
                "paired_coverage_rate_change": float(coverage_delta.mean()),
                "paired_coverage_rate_change_bootstrap_ci95": [
                    float(value) for value in np.percentile(coverage_bootstrap, [2.5, 97.5])
                ],
                "conformal_only_covers": improvement,
                "profile_only_covers": regression,
                "mcnemar_exact_two_sided_p_value": p_value,
                "profile_interval_median_width_ns": float(np.median(profile_width)),
                "conformal_set_median_span_ns": float(np.median(conformal_width)),
                "paired_mean_span_change_ns": float(span_delta.mean()),
                "paired_mean_span_change_bootstrap_ci95_ns": [
                    float(value) for value in np.percentile(span_bootstrap, [2.5, 97.5])
                ],
            }
        )

    order = np.argsort(raw_p_values)
    adjusted = np.empty(len(raw_p_values), dtype=np.float64)
    running_max = 0.0
    for rank, index in enumerate(order):
        running_max = max(running_max, (len(raw_p_values) - rank) * raw_p_values[int(index)])
        adjusted[int(index)] = min(1.0, running_max)
    for row, value in zip(comparisons, adjusted, strict=True):
        row["mcnemar_holm_adjusted_p_value"] = float(value)

    profile_all = np.concatenate(all_profile_covered)
    conformal_all = np.concatenate(all_conformal_covered)
    return {
        "schema": "loslab.identifiability_interval_comparison.v1",
        "evidence_grade": "PAIRED_SYNTHETIC_ANALYTICAL_UNCERTAINTY_COMPARISON",
        "seed": study.get("seed"),
        "bootstrap_seed": seed,
        "bootstrap_draws": bootstrap_draws,
        "input_source_revision": study.get("source_revision"),
        "condition_cells": len(comparisons),
        "pooled_evaluation_trials": int(profile_all.size),
        "pooled_profile_interval_coverage": float(profile_all.mean()),
        "pooled_conformal_set_coverage": float(conformal_all.mean()),
        "conditions": comparisons,
        "limitations": [
            "Both methods use the same finite-grid profile-likelihood score; this compares calibration, not independent estimator families.",
            "Same-condition split-conformal calibration assumes exchangeable calibration/evaluation draws from the specified synthetic cell.",
            "The finite grid, known two-path count, known noise variance, and exact analytical model omit channel-estimation error, diffuse energy, scene variation, and receiver impairments.",
            "Set span measures the hull and may include disconnected low-likelihood gaps; coverage is grid-cell quantized.",
            "Paired Monte Carlo uncertainty is conditional on these analytical distributions and is not evidence of real-scene or measured-channel calibration.",
        ],
    }
