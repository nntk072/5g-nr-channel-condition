"""Criterion-oriented paired delay/angular checks for bounded G3 reviews.

These diagnostics compare declared candidate/challenger channel metrics using
the project's frozen physical tolerances. They never qualify a scene family or
authorize production ray tracing.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
from scipy.stats import beta

from evaluation.g3_joint_acceptance import (
    family_alpha_for_claim,
    validate_g3_joint_acceptance_family,
)

_METRICS = (
    "first_arrival_delay_s",
    "rms_delay_spread_s",
    "arrival_angular_spread_deg",
    "departure_angular_spread_deg",
)
_FAMILYWISE_ALPHA = 0.05
_BOUNDS_PER_METRIC = 2  # p95 upper confidence bound and cluster-failure bound
MIN_INDEPENDENT_CLUSTERS_PER_STRATUM = 2
_MAX_BOOTSTRAP_INDEX_CELLS = 2_000_000
_CLUSTER_FAILURE_LIMIT = 0.05


def _required_number(value: Any, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise ValueError(
            f"{name} must be a finite {'nonnegative ' if nonnegative else ''}real number"
        )
    return result


def _normalised_metric_error(
    name: str,
    candidate: float,
    challenger: float,
    *,
    effective_bandwidth_hz: float,
) -> float:
    difference = abs(candidate - challenger)
    if name == "first_arrival_delay_s":
        tolerance = 0.1 / effective_bandwidth_hz
    elif name == "rms_delay_spread_s":
        tolerance = max(1e-9, 0.05 * max(candidate, challenger))
    else:
        tolerance = max(1.0, 0.05 * max(candidate, challenger))
    return difference / tolerance


def _upper_clopper_pearson(failures: int, clusters: int, alpha: float) -> float:
    if failures == clusters:
        return 1.0
    return float(beta.ppf(1.0 - alpha, failures + 1, clusters - failures))


def _bootstrap_p95_upper(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    draws: int,
    level: float,
) -> float:
    cluster_count = len(values)
    rows_per_chunk = max(1, _MAX_BOOTSTRAP_INDEX_CELLS // cluster_count)
    estimates = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, rows_per_chunk):
        stop = min(start + rows_per_chunk, draws)
        indices = rng.integers(0, cluster_count, size=(stop - start, cluster_count))
        estimates[start:stop] = np.quantile(values[indices], 0.95, axis=1)
    return float(np.quantile(estimates, level))


def evaluate_propagation_metrics(
    records: Sequence[dict[str, Any]],
    *,
    expected_strata: Sequence[tuple[str, str, str]],
    effective_bandwidth_hz: float,
    joint_acceptance_family: dict[str, Any] | None = None,
    bootstrap_seed: int = 20261024,
    bootstrap_draws: int = 20_000,
) -> dict[str, Any]:
    """Evaluate paired delay/angular errors using independent clusters.

    Every row is one paired episode. It carries ``family_id``, ``condition``
    (LOS/NLOS), ``component`` (total/specular/diffuse), ``cluster_id``,
    ``episode_id``, candidate/challenger configuration IDs, and a ``metrics``
    mapping for all four names in ``_METRICS``. Each metric item has candidate
    and challenger values, an availability state (resolved/unresolved/not_applicable),
    and a reason when masked. Expected strata are preregistered and must exactly
    match the observed family × condition × component set.

    Errors are divided by their frozen physical tolerance. Per-cluster medians
    and p95s receive equal weight; whole clusters are resampled for a one-sided
    95% p95 upper bound. A cluster fails a metric if any resolved episode has
    normalized error > 1. A numeric pass remains conditional and is never a
    scientific qualification decision.
    """
    if not records:
        raise ValueError("records must be nonempty")
    if not expected_strata:
        raise ValueError("expected_strata must be nonempty and preregistered")
    if (
        isinstance(bootstrap_seed, bool)
        or not isinstance(bootstrap_seed, int)
        or bootstrap_seed < 0
    ):
        raise ValueError("bootstrap_seed must be a nonnegative integer")
    if (
        isinstance(bootstrap_draws, bool)
        or not isinstance(bootstrap_draws, int)
        or bootstrap_draws < 1000
    ):
        raise ValueError("bootstrap_draws must be an integer >= 1000")
    bandwidth = _required_number(effective_bandwidth_hz, "effective_bandwidth_hz")
    if bandwidth <= 0.0:
        raise ValueError("effective_bandwidth_hz must be positive")

    expected: set[tuple[str, str, str]] = set()
    for item in expected_strata:
        if (
            not isinstance(item, tuple)
            or len(item) != 3
            or not isinstance(item[0], str)
            or not item[0].strip()
            or not isinstance(item[1], str)
            or item[1] not in {"LOS", "NLOS"}
            or not isinstance(item[2], str)
            or item[2] not in {"total", "specular", "diffuse"}
        ):
            raise ValueError(
                "expected strata must be (family_id, LOS/NLOS, total/specular/diffuse)"
            )
        if item in expected:
            raise ValueError(f"duplicate expected stratum: {item}")
        expected.add(item)

    if joint_acceptance_family is None:
        familywise_claim_count = len(expected) * len(_METRICS) * _BOUNDS_PER_METRIC
        per_claim_alpha = _FAMILYWISE_ALPHA / familywise_claim_count
        family_reference = None
        confidence_scope = (
            "Bonferroni family across p95 and cluster-failure bounds for every "
            "metric × preregistered family × condition × component stratum"
        )
    else:
        normalized_family = validate_g3_joint_acceptance_family(joint_acceptance_family)
        per_claim_alpha, familywise_claim_count = family_alpha_for_claim(
            normalized_family,
            evaluator="propagation_metrics",
            expected_strata=sorted(expected),
        )
        family_reference = {
            "family_id": normalized_family["family_id"],
            "family_sha256": normalized_family["family_sha256"],
        }
        confidence_scope = "shared G3 family across material, complex-CSI, and propagation claims"
    per_claim_confidence_level = 1.0 - per_claim_alpha

    # stratum -> metric -> cluster -> episode-level normalized errors
    grouped: dict[tuple[str, str, str], dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: {name: defaultdict(list) for name in _METRICS}
    )
    availability: dict[tuple[tuple[str, str, str], str], dict[str, int]] = defaultdict(
        lambda: {"resolved": 0, "unresolved": 0, "not_applicable": 0}
    )
    reasons: dict[tuple[tuple[str, str, str], str], dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    seen: set[tuple[str, str, str, str, str]] = set()
    design: tuple[str, str] | None = None

    for row in records:
        family = row.get("family_id")
        condition = row.get("condition")
        component = row.get("component")
        cluster = row.get("cluster_id")
        episode = row.get("episode_id")
        if not all(
            isinstance(value, str) and value.strip() for value in (family, cluster, episode)
        ):
            raise ValueError("family_id, cluster_id, and episode_id must be nonempty strings")
        if not isinstance(condition, str) or condition not in {"LOS", "NLOS"}:
            raise ValueError("condition must be LOS or NLOS")
        if not isinstance(component, str) or component not in {"total", "specular", "diffuse"}:
            raise ValueError("component must be total, specular, or diffuse")
        family_id, condition_id, component_id, cluster_id, episode_id = cast(
            tuple[str, str, str, str, str],
            (family, condition, component, cluster, episode),
        )
        key = (family_id, condition_id, component_id)
        identity = (*key, cluster_id, episode_id)
        if identity in seen:
            raise ValueError(f"duplicate episode ID in stratum/cluster: {identity}")
        seen.add(identity)
        candidate_config = row.get("candidate_configuration_id")
        challenger_config = row.get("challenger_configuration_id")
        if not all(
            isinstance(value, str) and value.strip()
            for value in (candidate_config, challenger_config)
        ):
            raise ValueError("candidate/challenger configuration IDs are required")
        current_design = (cast(str, candidate_config), cast(str, challenger_config))
        if design is not None and current_design != design:
            raise ValueError("candidate/challenger configuration IDs must match across records")
        design = current_design
        metric_data = row.get("metrics")
        if not isinstance(metric_data, dict) or set(metric_data) != set(_METRICS):
            raise ValueError(f"metrics must contain exactly: {', '.join(_METRICS)}")

        for metric_name in _METRICS:
            item = metric_data[metric_name]
            if not isinstance(item, dict):
                raise ValueError(f"{metric_name} must be an object")
            state = item.get("availability")
            candidate = item.get("candidate")
            challenger = item.get("challenger")
            reason = item.get("reason")
            if not isinstance(state, str) or state not in {
                "resolved",
                "unresolved",
                "not_applicable",
            }:
                raise ValueError(f"{metric_name}.availability is invalid")
            if state == "resolved":
                if candidate is None or challenger is None or reason is not None:
                    raise ValueError(
                        f"resolved {metric_name} needs two values and no masking reason"
                    )
                candidate_value = _required_number(
                    candidate,
                    f"{metric_name}.candidate",
                    nonnegative=metric_name != "first_arrival_delay_s",
                )
                challenger_value = _required_number(
                    challenger,
                    f"{metric_name}.challenger",
                    nonnegative=metric_name != "first_arrival_delay_s",
                )
                error = _normalised_metric_error(
                    metric_name,
                    candidate_value,
                    challenger_value,
                    effective_bandwidth_hz=bandwidth,
                )
                if not math.isfinite(error):
                    raise ValueError(f"{metric_name} normalized error is not finite")
                grouped[key][metric_name][cluster_id].append(error)
            else:
                if candidate is not None or challenger is not None:
                    raise ValueError(f"masked {metric_name} values must both be null")
                if not isinstance(reason, str) or not reason.strip():
                    raise ValueError(f"masked {metric_name} requires a nonempty reason")
                if state == "not_applicable" and component_id == "total":
                    raise ValueError("total-component metrics cannot be not_applicable")
                reasons[(key, metric_name)][reason] += 1
            availability[(key, metric_name)][cast(str, state)] += 1

    observed = {stratum for stratum, _ in availability}
    if observed != expected:
        raise ValueError(
            f"observed strata must exactly match expected_strata; missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )

    rng = np.random.default_rng(bootstrap_seed)
    strata_reports: list[dict[str, Any]] = []
    for key in sorted(expected):
        family, condition, component = key
        metric_reports: list[dict[str, Any]] = []
        for metric_name in _METRICS:
            counts = availability[(key, metric_name)]
            clusters = grouped[key][metric_name]
            resolved_clusters = sorted(clusters)
            metric_report: dict[str, Any] = {
                "metric": metric_name,
                "availability_counts": dict(counts),
                "mask_reasons": dict(sorted(reasons[(key, metric_name)].items())),
            }
            if not resolved_clusters:
                metric_report["criteria_status"] = (
                    "NOT_APPLICABLE" if counts["not_applicable"] else "UNRESOLVED"
                )
                metric_reports.append(metric_report)
                continue
            cluster_medians = np.asarray(
                [np.median(clusters[cluster_id]) for cluster_id in resolved_clusters],
                dtype=np.float64,
            )
            cluster_p95s = np.asarray(
                [np.quantile(clusters[cluster_id], 0.95) for cluster_id in resolved_clusters],
                dtype=np.float64,
            )
            metric_report["independent_clusters"] = len(resolved_clusters)
            metric_report["resolved_episodes"] = sum(
                len(clusters[cluster_id]) for cluster_id in resolved_clusters
            )
            metric_report["median_normalized_error"] = float(np.median(cluster_medians))
            metric_report["p95_normalized_error"] = float(np.quantile(cluster_p95s, 0.95))
            if len(resolved_clusters) < MIN_INDEPENDENT_CLUSTERS_PER_STRATUM:
                metric_report["criteria_status"] = "UNRESOLVED"
                metric_report["unresolved_reason"] = "fewer than two resolved independent clusters"
                metric_reports.append(metric_report)
                continue
            p95_upper = _bootstrap_p95_upper(
                cluster_p95s,
                rng=rng,
                draws=bootstrap_draws,
                level=per_claim_confidence_level,
            )
            failures = sum(max(clusters[cluster_id]) > 1.0 for cluster_id in resolved_clusters)
            failure_upper = _upper_clopper_pearson(
                failures, len(resolved_clusters), per_claim_alpha
            )
            checks = {
                "median_within_tolerance": metric_report["median_normalized_error"] <= 1.0,
                "p95_within_tolerance": metric_report["p95_normalized_error"] <= 1.0,
                "p95_upper_bound_within_tolerance": p95_upper <= 1.0,
                "cluster_failure_upper_bound_within_tolerance": failure_upper <= 0.05,
            }
            metric_report.update(
                {
                    "p95_one_sided_bootstrap_upper_confidence_bound": p95_upper,
                    "confidence_level": per_claim_confidence_level,
                    "per_claim_alpha_bonferroni": per_claim_alpha,
                    "clusters_with_any_error_above_tolerance": failures,
                    "cluster_failure_one_sided_exact_upper_bound": failure_upper,
                    "criterion_checks": checks,
                    "criteria_status": (
                        "UNRESOLVED"
                        if counts["unresolved"]
                        else "PASS"
                        if all(checks.values())
                        else "FAIL"
                    ),
                }
            )
            metric_reports.append(metric_report)
        strata_reports.append(
            {
                "family_id": family,
                "condition": condition,
                "component": component,
                "metrics": metric_reports,
            }
        )

    statuses = [metric["criteria_status"] for item in strata_reports for metric in item["metrics"]]
    any_assessable = "PASS" in statuses or "FAIL" in statuses
    all_pass = any_assessable and all(state in {"PASS", "NOT_APPLICABLE"} for state in statuses)
    return {
        "schema": "loslab.g3_propagation_metric_comparison.v1",
        "status": "NUMERICAL_CRITERIA_ONLY_NOT_QUALIFIED",
        "admission_possible": False,
        "any_metric_assessable": any_assessable,
        "all_numeric_criteria_pass": all_pass,
        "all_criteria_resolved_and_pass": all_pass,
        "configuration": {
            "expected_strata": [list(item) for item in sorted(expected)],
            "effective_bandwidth_hz": bandwidth,
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_draws": bootstrap_draws,
            "familywise_alpha": _FAMILYWISE_ALPHA,
            "familywise_confidence_level": 1.0 - _FAMILYWISE_ALPHA,
            "familywise_claim_count": familywise_claim_count,
            "per_claim_alpha_bonferroni": per_claim_alpha,
            "joint_acceptance_family": family_reference,
            "bootstrap_confidence_level": per_claim_confidence_level,
            "bootstrap_unit": "independent outer cluster; episodes remain nested",
            "confidence_scope": confidence_scope,
            "point_estimator": "equal-weight median and p95 of per-cluster episode summaries",
            "candidate_configuration_id": design[0] if design else None,
            "challenger_configuration_id": design[1] if design else None,
        },
        "frozen_tolerances": {
            "first_arrival_delay_s": "0.1 / effective_bandwidth_hz",
            "rms_delay_spread_s": "max(1e-9 seconds, 0.05 * max(candidate, challenger))",
            "arrival_angular_spread_deg": "max(1 degree, 0.05 * max(candidate, challenger))",
            "departure_angular_spread_deg": "max(1 degree, 0.05 * max(candidate, challenger))",
            "normalized_median_and_p95_max": 1.0,
            "cluster_failure_rate_upper_bound_max": _CLUSTER_FAILURE_LIMIT,
        },
        "strata": strata_reports,
        "qualification_gaps": [
            "Physical path/statistic resolution and component decomposition require independent audit.",
            "Effective bandwidth, candidate/challenger settings, and source hashes must be bound to the frozen run manifest.",
            "Independent scene-family sampling, path-cap status, and representativeness are not verified by this scorer.",
            "A numerical pass does not establish propagation truth or authorize production RT generation.",
        ],
    }
