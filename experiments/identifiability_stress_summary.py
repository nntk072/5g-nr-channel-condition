"""Compact, integrity-linked summaries for profile-likelihood stress runs."""

from __future__ import annotations

from typing import Any, Mapping


def summarize_profile_likelihood_stress(
    report: Mapping[str, Any], *, raw_report_sha256: str
) -> dict[str, Any]:
    """Validate and compact a raw profile-likelihood stress report."""
    if report.get("schema") != "loslab.identifiability_profile_likelihood.v1":
        raise ValueError("unsupported profile-likelihood report schema")
    if report.get("evidence_grade") != (
        "CONDITION_CALIBRATED_ANALYTICAL_PROFILE_LIKELIHOOD_STRESS_DIAGNOSTIC"
    ):
        raise ValueError("report is not a profile-likelihood stress diagnostic")
    cases = report.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("report must contain at least one stress case")
    required = (
        "assumed_to_true_noise_variance_ratio",
        "third_path_amplitude_ratio_to_strongest",
        "true_path_count",
        "fitted_path_count",
        "path_order_randomized",
        "path_order_permutations_applied",
        "trials",
        "calibration_trials",
        "split_conformal_95_empirical_coverage",
        "split_conformal_95_wilson_95_ci",
        "split_conformal_95_threshold",
        "split_conformal_95_calibration_rank",
        "split_conformal_95_median_set_span_ns",
    )
    compact_cases = []
    for index, item in enumerate(cases):
        if not isinstance(item, dict):
            raise ValueError(f"case {index} must be an object")
        missing = [name for name in required if name not in item]
        if missing:
            raise ValueError(f"case {index} missing fields: {', '.join(missing)}")
        compact_cases.append({name: item[name] for name in required})
    if len(raw_report_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in raw_report_sha256.lower()
    ):
        raise ValueError("raw_report_sha256 must be a SHA256 hex digest")
    return {
        "schema": "loslab.identifiability_profile_likelihood_stress_summary.v1",
        "evidence_grade": report["evidence_grade"],
        "seed": report.get("seed"),
        "source_revision": report.get("source_revision"),
        "package_source_sha256": report.get("package_source_sha256"),
        "experiment_module_sha256": report.get("experiment_module_sha256"),
        "runner_sha256": report.get("runner_sha256"),
        "versions": report.get("versions"),
        "raw_report_sha256": raw_report_sha256,
        "calibration_claim": (
            "finite-sample split-conformal marginal coverage applies only within each "
            "separately calibrated, exchangeable synthetic condition; reported test Wilson "
            "intervals condition on the fitted calibration threshold and omit calibration-split "
            "variability"
        ),
        "cases": compact_cases,
        "limitations": report.get("limitations", []),
    }
