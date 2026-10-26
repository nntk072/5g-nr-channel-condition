"""Criterion-oriented complex-CSI comparisons for bounded G3 reviews.

The scorer computes the frozen symmetric, unnormalized CFR error and reports
cluster-aware per-stratum diagnostics. Passing these numerical checks is not
qualification: source provenance, receiver noise-floor derivation, scene
independence, preregistration, and external review remain separate gates.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from scipy.stats import beta

from evaluation.g3_joint_acceptance import (
    family_alpha_for_claim,
    validate_g3_joint_acceptance_family,
)

COMPLEX_CSI_MEDIAN_LIMIT = 0.01
COMPLEX_CSI_P95_LIMIT = 0.05
COMPLEX_CSI_CLUSTER_FAILURE_LIMIT = 0.05
MIN_INDEPENDENT_CLUSTERS_PER_STRATUM = 2
_MAX_BOOTSTRAP_INDEX_CELLS = 2_000_000


def symmetric_complex_csi_error(
    candidate: NDArray[np.complexfloating[Any, Any]],
    challenger: NDArray[np.complexfloating[Any, Any]],
    *,
    h_floor: float,
) -> float:
    """Compute the frozen e_H metric without phase alignment or normalization."""
    left = np.asarray(candidate)
    right = np.asarray(challenger)
    if left.ndim == 0 or left.size == 0 or left.shape != right.shape:
        raise ValueError("candidate and challenger CFRs must have the same nonempty shape")
    if not np.iscomplexobj(left) or not np.iscomplexobj(right):
        raise ValueError("candidate and challenger CFRs must be complex-valued")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("candidate and challenger CFRs must contain only finite values")
    if isinstance(h_floor, bool) or not isinstance(h_floor, (int, float)):
        raise ValueError("h_floor must be a finite positive real number")
    floor = float(h_floor)
    if not math.isfinite(floor) or floor <= 0.0:
        raise ValueError("h_floor must be a finite positive real number")

    candidate_norm = float(np.linalg.norm(left.ravel()))
    challenger_norm = float(np.linalg.norm(right.ravel()))
    denominator = max((candidate_norm + challenger_norm) / 2.0, floor)
    error = float(np.linalg.norm((left - right).ravel())) / denominator
    if not math.isfinite(error):
        raise ValueError("complex-CSI error is not finite")
    return error


def _upper_clopper_pearson(failures: int, clusters: int, alpha: float) -> float:
    if failures == clusters:
        return 1.0
    return float(beta.ppf(1.0 - alpha, failures + 1, clusters - failures))


def _cluster_bootstrap_p95_upper(
    values: NDArray[np.float64], *, rng: np.random.Generator, draws: int, level: float
) -> float:
    """Resample whole cluster summaries with bounded temporary index memory."""
    cluster_count = len(values)
    chunk_size = max(1, _MAX_BOOTSTRAP_INDEX_CELLS // cluster_count)
    estimates = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, chunk_size):
        stop = min(start + chunk_size, draws)
        indices = rng.integers(0, cluster_count, size=(stop - start, cluster_count))
        estimates[start:stop] = np.quantile(values[indices], 0.95, axis=1)
    return float(np.quantile(estimates, level))


def evaluate_complex_csi(
    records: Sequence[dict[str, Any]],
    *,
    expected_strata: Sequence[tuple[str, str]],
    bootstrap_seed: int = 20261024,
    bootstrap_draws: int = 20_000,
    confidence_level: float = 0.95,
    joint_acceptance_family: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score matched complex CFR episodes, resampling independent clusters.

    Each record is one paired episode and must provide ``family_id``,
    ``condition`` (``LOS`` or ``NLOS``), ``cluster_id``, both complex CFR
    arrays, stable grid/configuration IDs, and the predeclared ``h_floor``.
    To prevent clusters with more recorded links from dominating the estimand,
    each cluster contributes its median and 95th-percentile episode error;
    stratum summaries and bootstrap intervals operate on those cluster
    summaries. A cluster fails when any episode has e_H > 0.05.

    The returned numerical assessment is explicitly not a qualification
    decision. The caller must independently establish that the CFR axes,
    noise-derived floor, matched configuration, input provenance, and physical
    independence satisfy the frozen protocol.
    """
    if not records:
        raise ValueError("records must be nonempty")
    if not expected_strata:
        raise ValueError("expected_strata must be nonempty and preregistered")
    expected: set[tuple[str, str]] = set()
    for item in expected_strata:
        if (
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not item[0].strip()
            or not isinstance(item[1], str)
            or item[1] not in {"LOS", "NLOS"}
        ):
            raise ValueError("each expected stratum must be (nonempty family_id, LOS or NLOS)")
        if item in expected:
            raise ValueError(f"duplicate expected stratum: {item}")
        expected.add(item)
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
    if (
        isinstance(confidence_level, bool)
        or not isinstance(confidence_level, (int, float))
        or not math.isfinite(float(confidence_level))
        or not 0.0 < float(confidence_level) < 1.0
    ):
        raise ValueError("confidence_level must be a finite value in (0, 1)")
    if float(confidence_level) != 0.95:
        raise ValueError("confidence_level is frozen at 0.95 by the G3 acceptance criteria")
    familywise_alpha = 1.0 - float(confidence_level)
    if joint_acceptance_family is None:
        familywise_claim_count = 2 * len(expected)
        per_claim_alpha = familywise_alpha / familywise_claim_count
        family_reference = None
    else:
        normalized_family = validate_g3_joint_acceptance_family(joint_acceptance_family)
        per_claim_alpha, familywise_claim_count = family_alpha_for_claim(
            normalized_family,
            evaluator="complex_csi",
            expected_strata=sorted(expected),
        )
        family_reference = {
            "family_id": normalized_family["family_id"],
            "family_sha256": normalized_family["family_sha256"],
        }
    per_claim_confidence = 1.0 - per_claim_alpha

    grouped: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    stratum_design: dict[tuple[str, str], tuple[tuple[int, ...], str, str, str, float]] = {}
    seen_episode_ids: set[tuple[str, str, str, str]] = set()
    for row in records:
        family = row.get("family_id")
        condition = row.get("condition")
        cluster = row.get("cluster_id")
        episode = row.get("episode_id")
        if not all(
            isinstance(value, str) and value.strip() for value in (family, cluster, episode)
        ):
            raise ValueError("family_id, cluster_id, and episode_id must be nonempty strings")
        if not isinstance(condition, str) or condition not in {"LOS", "NLOS"}:
            raise ValueError("condition must be LOS or NLOS")
        grid_id = row.get("evaluation_grid_id")
        candidate_config = row.get("candidate_configuration_id")
        challenger_config = row.get("challenger_configuration_id")
        if not all(
            isinstance(value, str) and value.strip()
            for value in (grid_id, candidate_config, challenger_config)
        ):
            raise ValueError(
                "evaluation_grid_id and candidate/challenger configuration IDs are required"
            )
        family_id = cast(str, family)
        condition_id = cast(str, condition)
        cluster_id = cast(str, cluster)
        episode_id = cast(str, episode)
        identity = (family_id, condition_id, cluster_id, episode_id)
        if identity in seen_episode_ids:
            raise ValueError(f"duplicate episode ID in stratum/cluster: {identity}")
        seen_episode_ids.add(identity)
        candidate_cfr = cast(
            NDArray[np.complexfloating[Any, Any]], np.asarray(row.get("candidate_cfr"))
        )
        challenger_cfr = cast(
            NDArray[np.complexfloating[Any, Any]], np.asarray(row.get("challenger_cfr"))
        )
        h_floor = row.get("h_floor")
        if isinstance(h_floor, bool) or not isinstance(h_floor, (int, float)):
            raise ValueError("h_floor must be a finite positive real number")
        error = symmetric_complex_csi_error(
            candidate_cfr,
            challenger_cfr,
            h_floor=float(h_floor),
        )
        stratum_key = (family_id, condition_id)
        design = (
            candidate_cfr.shape,
            cast(str, grid_id),
            cast(str, candidate_config),
            cast(str, challenger_config),
            float(h_floor),
        )
        if stratum_key in stratum_design and stratum_design[stratum_key] != design:
            raise ValueError(
                f"evaluation grid, configuration IDs, CFR shape, and h_floor must match within {stratum_key}"
            )
        stratum_design[stratum_key] = design
        grouped[stratum_key][cluster_id].append(error)

    observed = set(grouped)
    if observed != expected:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        raise ValueError(
            f"observed strata must exactly match expected_strata; missing={missing}, "
            f"unexpected={unexpected}"
        )
    if len(set(stratum_design.values())) != 1:
        raise ValueError(
            "grid, candidate/challenger configurations, CFR shape, and h_floor must match across strata"
        )

    rng = np.random.default_rng(bootstrap_seed)
    strata: list[dict[str, Any]] = []
    for (family, condition), clusters in sorted(grouped.items()):
        cluster_ids = sorted(clusters)
        if len(cluster_ids) < MIN_INDEPENDENT_CLUSTERS_PER_STRATUM:
            raise ValueError(
                f"stratum {family}/{condition} needs at least "
                f"{MIN_INDEPENDENT_CLUSTERS_PER_STRATUM} clusters"
            )
        per_cluster_median = np.asarray(
            [np.median(clusters[key]) for key in cluster_ids], dtype=np.float64
        )
        per_cluster_p95 = np.asarray(
            [np.quantile(clusters[key], 0.95) for key in cluster_ids], dtype=np.float64
        )
        p95_upper = _cluster_bootstrap_p95_upper(
            per_cluster_p95,
            rng=rng,
            draws=bootstrap_draws,
            level=per_claim_confidence,
        )
        failures = sum(
            max(clusters[key]) > COMPLEX_CSI_CLUSTER_FAILURE_LIMIT for key in cluster_ids
        )
        failure_upper = _upper_clopper_pearson(failures, len(cluster_ids), per_claim_alpha)
        median_error = float(np.median(per_cluster_median))
        p95_error = float(np.quantile(per_cluster_p95, 0.95))
        checks = {
            "median_e_h": median_error <= COMPLEX_CSI_MEDIAN_LIMIT,
            "p95_e_h": p95_error <= COMPLEX_CSI_P95_LIMIT,
            "p95_e_h_upper_confidence_bound": p95_upper <= COMPLEX_CSI_P95_LIMIT,
            "cluster_failure_upper_confidence_bound": (
                failure_upper <= COMPLEX_CSI_CLUSTER_FAILURE_LIMIT
            ),
        }
        strata.append(
            {
                "family_id": family,
                "condition": condition,
                "evaluation_grid_id": stratum_design[(family, condition)][1],
                "candidate_configuration_id": stratum_design[(family, condition)][2],
                "challenger_configuration_id": stratum_design[(family, condition)][3],
                "cfr_shape": list(stratum_design[(family, condition)][0]),
                "independent_clusters": len(cluster_ids),
                "episodes": sum(len(clusters[key]) for key in cluster_ids),
                "cluster_ids": cluster_ids,
                "median_e_h_equal_cluster_weight": median_error,
                "p95_e_h_equal_cluster_weight": p95_error,
                "p95_e_h_upper_confidence_bound": p95_upper,
                "p95_e_h_upper_confidence_level": per_claim_confidence,
                "cluster_failure_count": failures,
                "cluster_failure_rate": failures / len(cluster_ids),
                "cluster_failure_rate_one_sided_upper_confidence_bound": failure_upper,
                "cluster_failure_upper_confidence_level": per_claim_confidence,
                "criterion_checks": checks,
                "numeric_criteria_pass": all(checks.values()),
            }
        )

    return {
        "schema": "loslab.g3_complex_csi_comparison.v2",
        "status": "NUMERICAL_CRITERIA_ONLY_NOT_QUALIFIED",
        "admission_possible": False,
        "criteria": {
            "symmetric_e_h": "||H_candidate-H_challenger||_F / max((||H_candidate||_F + ||H_challenger||_F)/2, H_floor)",
            "median_e_h_max": COMPLEX_CSI_MEDIAN_LIMIT,
            "p95_e_h_max": COMPLEX_CSI_P95_LIMIT,
            "p95_upper_confidence_bound_max": COMPLEX_CSI_P95_LIMIT,
            "cluster_failure_definition": "any episode in a cluster has e_H > 0.05",
            "cluster_failure_rate_upper_bound_max": COMPLEX_CSI_CLUSTER_FAILURE_LIMIT,
            "familywise_confidence_level": float(confidence_level),
        },
        "configuration": {
            "expected_strata": [list(item) for item in sorted(expected)],
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_draws": bootstrap_draws,
            "bootstrap_unit": "independent cluster; all episode-level values remain together",
            "confidence_scope": (
                "Bonferroni family-wise across the cluster-bootstrap p95 bound and exact-binomial "
                "cluster-failure bound for every expected family × LOS/NLOS stratum; "
                + (
                    "jointly corrected with the material-calibration claims by the referenced G3 family"
                    if family_reference is not None
                    else "not jointly corrected with other G3 estimands"
                )
            ),
            "familywise_claim_count": familywise_claim_count,
            "familywise_alpha": familywise_alpha,
            "per_claim_alpha_bonferroni": per_claim_alpha,
            "per_claim_confidence_level": per_claim_confidence,
            "joint_acceptance_family": family_reference,
            "point_estimator": "equal-weight median and p95 of per-cluster episode summaries",
            "phase_alignment": False,
            "channel_normalization": False,
        },
        "strata": strata,
        "all_numeric_criteria_pass": all(item["numeric_criteria_pass"] for item in strata),
        "qualification_gaps": [
            "CFR resource/port/symbol axes and absolute delay/phase references are not independently verified by this scorer.",
            "H_floor must be derived from the frozen receiver noise/link budget and verified externally.",
            "Scene-family representativeness, physical cluster independence, matched run configuration, provenance, and path-cap status require independent audit.",
            "No measurement or independently qualified scene-family evidence is created by synthetic fixture data.",
        ],
    }
