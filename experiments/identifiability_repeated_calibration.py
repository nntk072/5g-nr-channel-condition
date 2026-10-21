"""Repeated-split diagnostics for calibrated profile-likelihood delay sets."""

from __future__ import annotations

from typing import Any, Mapping, cast

import numpy as np
from artifacts.provenance import package_source_sha256, sha256_file, versions

from experiments.identifiability_profile_likelihood import build_profile_likelihood_sweep

_CONDITION_FIELDS = (
    "tone_count",
    "tone_spacing_hz",
    "snr_db",
    "separation_fraction_of_reciprocal_bandwidth",
    "gain_phase_difference_deg",
    "assumed_to_true_noise_variance_ratio",
    "third_path_amplitude_ratio_to_strongest",
)


def build_repeated_calibration_study(
    sweep_config: Mapping[str, Any],
    *,
    repetitions: int = 20,
    bootstrap_draws: int = 10_000,
) -> dict[str, Any]:
    """Repeat independent calibration/test splits and resample at split level.

    The provided sweep configuration is reused unchanged except that each
    repetition receives a distinct deterministic seed. Confidence limits
    bootstrap whole split-level coverage rates, preserving dependence induced
    by the shared calibration threshold within each repetition.
    """
    if isinstance(repetitions, bool) or not isinstance(repetitions, int):
        raise TypeError("repetitions must be an integer")
    if repetitions < 5:
        raise ValueError("repetitions must be at least 5 for split-level resampling")
    if isinstance(bootstrap_draws, bool) or not isinstance(bootstrap_draws, int):
        raise TypeError("bootstrap_draws must be an integer")
    if bootstrap_draws < 100:
        raise ValueError("bootstrap_draws must be at least 100")
    base_seed = sweep_config.get("seed")
    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed < 0:
        raise ValueError("sweep_config must contain a nonnegative integer seed")

    accumulated: dict[tuple[object, ...], dict[str, Any]] = {}
    condition_values: dict[tuple[object, ...], dict[str, object]] = {}
    for repetition in range(repetitions):
        config = dict(sweep_config)
        config["seed"] = base_seed + repetition
        report = build_profile_likelihood_sweep(**cast(Any, config))
        for case in report["cases"]:
            key = tuple(case[field] for field in _CONDITION_FIELDS)
            condition_values[key] = {field: case[field] for field in _CONDITION_FIELDS}
            accumulated.setdefault(
                key,
                {
                    "repetitions": [],
                    "coverage": [],
                    "median_set_span_ns": [],
                },
            )
            accumulated[key]["coverage"].append(
                float(case["split_conformal_95_empirical_coverage"])
            )
            accumulated[key]["median_set_span_ns"].append(
                float(case["split_conformal_95_median_set_span_ns"])
            )
            accumulated[key]["repetitions"].append(
                {
                    "seed": config["seed"],
                    "calibration_trials": case["calibration_trials"],
                    "evaluation_trials": case["trials"],
                    "calibration_rank": case["split_conformal_95_calibration_rank"],
                    "threshold": case["split_conformal_95_threshold"],
                    "test_coverage": case["split_conformal_95_empirical_coverage"],
                    "test_wilson_interval_conditional_on_threshold": case[
                        "split_conformal_95_wilson_95_ci"
                    ],
                    "median_set_span_ns": case["split_conformal_95_median_set_span_ns"],
                    "path_order_permutations_applied": case["path_order_permutations_applied"],
                }
            )

    rng = np.random.default_rng(base_seed + repetitions)
    cases: list[dict[str, object]] = []
    for key, values in accumulated.items():
        rates = np.asarray(values["coverage"], dtype=np.float64)
        bootstrap_indices = rng.integers(0, repetitions, size=(bootstrap_draws, repetitions))
        split_bootstrap = rates[bootstrap_indices].mean(axis=1)
        spans = np.asarray(values["median_set_span_ns"], dtype=np.float64)
        repeated = values["repetitions"]
        if len(repeated) != repetitions:
            raise RuntimeError("stress condition was not observed in every repetition")
        cases.append(
            {
                **condition_values[key],
                "repetition_count": repetitions,
                "calibration_trials_per_repetition": repeated[0]["calibration_trials"],
                "evaluation_trials_per_repetition": repeated[0]["evaluation_trials"],
                "mean_split_coverage": float(rates.mean()),
                "split_bootstrap_percentile_interval_95": [
                    float(value) for value in np.percentile(split_bootstrap, [2.5, 97.5])
                ],
                "min_split_coverage": float(rates.min()),
                "max_split_coverage": float(rates.max()),
                "median_of_split_median_set_spans_ns": float(np.median(spans)),
                "repetitions": repeated,
            }
        )

    runtime = versions()
    runner_source_hash = sha256_file(__file__)
    return {
        "schema": "loslab.identifiability_repeated_calibration.v1",
        "evidence_grade": "REPEATED_SYNTHETIC_CALIBRATION_SPLIT_DIAGNOSTIC_NOT_EXTERNAL_VALIDATION",
        "seed": base_seed,
        "repetitions": repetitions,
        "bootstrap_draws": bootstrap_draws,
        "bootstrap_unit": "independent_calibration_evaluation_split",
        "source_revision": runtime.get("git_sha"),
        "package_source_sha256": package_source_sha256(),
        "experiment_module_sha256": runner_source_hash,
        "versions": runtime,
        "cases": cases,
        "limitations": [
            "This estimates repeated-split behavior only for the specified analytical synthetic distributions.",
            "Split-level bootstrap intervals are descriptive Monte Carlo uncertainty summaries, not a finite-sample conformal theorem.",
            "Each split's calibration and evaluation draws are exchangeable only within the same declared condition.",
            "No conclusion transfers to measured channels, arbitrary receiver noise, diffuse multipath, or independent RT scenes.",
        ],
    }


def summarize_repeated_calibration_study(
    report: Mapping[str, Any], *, raw_report_sha256: str
) -> dict[str, Any]:
    """Return a compact repeated-split summary while retaining every split metric."""
    if report.get("schema") != "loslab.identifiability_repeated_calibration.v1":
        raise ValueError("unsupported repeated-calibration report schema")
    cases = report.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("repeated-calibration report must have nonempty cases")
    if len(raw_report_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in raw_report_sha256.lower()
    ):
        raise ValueError("raw_report_sha256 must be a SHA256 hex digest")
    compact_cases = []
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("repetitions"), list):
            raise ValueError("each repeated-calibration case must include repetition records")
        compact_cases.append(
            {
                key: value
                for key, value in case.items()
                if key in _CONDITION_FIELDS
                or key
                in {
                    "repetition_count",
                    "calibration_trials_per_repetition",
                    "evaluation_trials_per_repetition",
                    "mean_split_coverage",
                    "split_bootstrap_percentile_interval_95",
                    "min_split_coverage",
                    "max_split_coverage",
                    "median_of_split_median_set_spans_ns",
                    "repetitions",
                }
            }
        )
    return {
        "schema": "loslab.identifiability_repeated_calibration_summary.v1",
        "evidence_grade": report.get("evidence_grade"),
        "seed": report.get("seed"),
        "repetition_count": report.get("repetitions"),
        "bootstrap_draws": report.get("bootstrap_draws"),
        "bootstrap_unit": report.get("bootstrap_unit"),
        "source_revision": report.get("source_revision"),
        "package_source_sha256": report.get("package_source_sha256"),
        "experiment_module_sha256": report.get("experiment_module_sha256"),
        "runner_sha256": report.get("runner_sha256"),
        "versions": report.get("versions"),
        "raw_report_sha256": raw_report_sha256,
        "cases": compact_cases,
        "limitations": report.get("limitations", []),
    }
