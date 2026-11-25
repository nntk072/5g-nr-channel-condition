"""Complex-CSI and bootstrap assurance planning."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.stats import beta

from evaluation.acceptance.joint_acceptance import (
    family_alpha_for_claim,
    validate_joint_acceptance_family,
)
from evaluation.sample_size.binomial import (
    _probability,
)
from evaluation.sample_size.constants import (
    _BOOTSTRAP_METRICS,
    _COMPLEX_CSI_FAILURE_LIMIT,
    _COMPLEX_CSI_P95_LIMIT,
    _MAX_BOOTSTRAP_PEAK_CELLS,
    _MAX_BOOTSTRAP_RESAMPLING_CELLS,
    MAX_INTERVAL_COVERAGE,
    MAX_MEDIAN_INTERVAL_WIDTH_DB,
    MAX_POWER_ERROR_MEDIAN_DB,
    MAX_POWER_ERROR_P95_DB,
    MIN_INDEPENDENT_MEASUREMENT_UNITS,
    MIN_INTERVAL_COVERAGE,
)


def plan_complex_csi_bootstrap_assurance(
    *,
    joint_acceptance_family: dict[str, Any],
    assumed_iid_scenarios_by_stratum: Mapping[tuple[str, str], Mapping[str, Any]],
    candidate_clusters_per_stratum: Sequence[int],
    bootstrap_replicates: int = 2_000,
    assurance_replicates: int = 500,
    target_assurance: float = 0.80,
    monte_carlo_confidence: float = 0.95,
    seed: int = 1,
) -> dict[str, Any]:
    """Estimate CSI evaluator pass assurance under explicit IID cluster scenarios.

    Each scenario atom supplies a hypothetical cluster median, cluster p95, and
    the evaluator's cluster-failure indicator. The simulated study applies the
    same equal-cluster point statistics, Bonferroni percentile cluster bootstrap
    upper bound, and exact-binomial failure bound as ``evaluate_complex_csi``.
    The family-bound per-claim alpha is shared with material calibration.
    Sampling remains conditional on the caller's finite-support assumptions and
    this function can never qualify a scene family or dataset.
    """
    family = validate_joint_acceptance_family(joint_acceptance_family)
    expected_strata = {
        (str(item[0]), str(item[1])) for item in family["complex_csi_expected_strata"]
    }
    if not isinstance(assumed_iid_scenarios_by_stratum, Mapping):
        raise ValueError("assumed_iid_scenarios_by_stratum must be a mapping")
    if set(assumed_iid_scenarios_by_stratum) != expected_strata:
        raise ValueError("CSI bootstrap scenarios must exactly match the joint-family strata")
    per_claim_alpha, family_claim_count = family_alpha_for_claim(
        family,
        evaluator="complex_csi",
        expected_strata=sorted(expected_strata),
    )

    candidates = list(candidate_clusters_per_stratum)
    if not candidates or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 2 for value in candidates
    ):
        raise ValueError("candidate_clusters_per_stratum must contain integers >= 2")
    if len(set(candidates)) != len(candidates):
        raise ValueError("candidate_clusters_per_stratum must be unique")
    if candidates != sorted(candidates):
        raise ValueError("candidate_clusters_per_stratum must be sorted ascending")
    if (
        isinstance(bootstrap_replicates, bool)
        or not isinstance(bootstrap_replicates, int)
        or bootstrap_replicates < 1_000
    ):
        raise ValueError("bootstrap_replicates must be an integer >= 1000")
    if (
        isinstance(assurance_replicates, bool)
        or not isinstance(assurance_replicates, int)
        or assurance_replicates < 100
    ):
        raise ValueError("assurance_replicates must be an integer >= 100")
    target = _probability(target_assurance, "target_assurance")
    mc_confidence = _probability(monte_carlo_confidence, "monte_carlo_confidence")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    scenario = _validate_complex_csi_assurance_scenarios(assumed_iid_scenarios_by_stratum)
    work_cells = (
        len(expected_strata) * assurance_replicates * bootstrap_replicates * sum(candidates)
    )
    if work_cells > _MAX_BOOTSTRAP_RESAMPLING_CELLS:
        raise ValueError("requested CSI bootstrap assurance exceeds the bounded work limit")
    if bootstrap_replicates * max(candidates) > _MAX_BOOTSTRAP_PEAK_CELLS:
        raise ValueError("a single CSI bootstrap resample exceeds the bounded memory limit")

    alpha_per_assurance_bound = (1.0 - mc_confidence) / (len(expected_strata) * len(candidates))
    rng = np.random.default_rng(seed)
    confidence_level = 1.0 - per_claim_alpha
    candidate_reports: list[dict[str, Any]] = []
    for cluster_count in candidates:
        stratum_reports: list[dict[str, Any]] = []
        lower_bounds: list[float] = []
        for stratum in sorted(expected_strata):
            atoms, probabilities = scenario[stratum]
            median_values = np.asarray([atom["cluster_median_e_h"] for atom in atoms])
            p95_values = np.asarray([atom["cluster_p95_e_h"] for atom in atoms])
            failure_values = np.asarray([atom["cluster_failed"] for atom in atoms], dtype=np.bool_)
            successes = 0
            for _ in range(assurance_replicates):
                sampled = rng.choice(len(atoms), size=cluster_count, p=probabilities)
                study_medians = median_values[sampled]
                study_p95 = p95_values[sampled]
                study_failures = failure_values[sampled]
                point_median = float(np.median(study_medians))
                point_p95 = float(np.quantile(study_p95, 0.95))
                bootstrap_upper_values = np.empty(bootstrap_replicates, dtype=np.float64)
                chunk_size = max(1, _MAX_BOOTSTRAP_PEAK_CELLS // cluster_count)
                for start in range(0, bootstrap_replicates, chunk_size):
                    stop = min(start + chunk_size, bootstrap_replicates)
                    indices = rng.integers(0, cluster_count, size=(stop - start, cluster_count))
                    bootstrap_upper_values[start:stop] = np.quantile(
                        study_p95[indices], 0.95, axis=1
                    )
                bootstrap_upper = float(np.quantile(bootstrap_upper_values, confidence_level))
                failure_count = int(study_failures.sum())
                failure_upper = (
                    1.0
                    if failure_count == cluster_count
                    else float(
                        beta.ppf(
                            1.0 - per_claim_alpha,
                            failure_count + 1,
                            cluster_count - failure_count,
                        )
                    )
                )
                successes += int(
                    point_median <= 0.01
                    and point_p95 <= _COMPLEX_CSI_P95_LIMIT
                    and bootstrap_upper <= _COMPLEX_CSI_P95_LIMIT
                    and failure_upper <= _COMPLEX_CSI_FAILURE_LIMIT
                )
            lower = (
                0.0
                if successes == 0
                else float(
                    beta.ppf(
                        alpha_per_assurance_bound,
                        successes,
                        assurance_replicates - successes + 1,
                    )
                )
            )
            lower_bounds.append(lower)
            stratum_reports.append(
                {
                    "family_id": stratum[0],
                    "condition": stratum[1],
                    "cluster_count": cluster_count,
                    "joint_pass_count": successes,
                    "assurance_estimate": successes / assurance_replicates,
                    "assurance_one_sided_exact_lower_bound": lower,
                    "monte_carlo_confidence_level_bonferroni": 1.0 - alpha_per_assurance_bound,
                }
            )
        joint_lower = max(0.0, 1.0 - sum(1.0 - value for value in lower_bounds))
        candidate_reports.append(
            {
                "candidate_clusters_per_stratum": cluster_count,
                "stratum_assurance": stratum_reports,
                "joint_assurance_lower_bound_union_bound": joint_lower,
                "meets_target_assurance": joint_lower >= target,
            }
        )
    selected = next(
        (
            item["candidate_clusters_per_stratum"]
            for item in candidate_reports
            if item["meets_target_assurance"]
        ),
        None,
    )
    scenario_record = [
        {
            "stratum": list(key),
            "atoms": scenario[key][0],
            "probabilities": scenario[key][1].tolist(),
        }
        for key in sorted(scenario)
    ]
    scenario_digest = hashlib.sha256(
        json.dumps(
            scenario_record,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema": "loslab.complex_csi_bootstrap_assurance_planning.v1",
        "status": "SCENARIO_CONDITIONAL_PLANNING_ONLY_NOT_QUALIFICATION",
        "qualification_status": "NOT_QUALIFIED",
        "admission_possible": False,
        "scenario_provenance_verified": False,
        "joint_acceptance_family": {
            "family_id": family["family_id"],
            "family_sha256": family["family_sha256"],
            "familywise_alpha": family["familywise_alpha"],
            "claim_count": family_claim_count,
            "per_claim_alpha_bonferroni": per_claim_alpha,
            "per_claim_confidence_level": confidence_level,
        },
        "criteria": {
            "cluster_median_e_h_max": 0.01,
            "cluster_p95_e_h_max": _COMPLEX_CSI_P95_LIMIT,
            "bootstrap_p95_upper_confidence_bound_max": _COMPLEX_CSI_P95_LIMIT,
            "cluster_failure_upper_confidence_bound_max": _COMPLEX_CSI_FAILURE_LIMIT,
        },
        "candidate_designs": candidate_reports,
        "selected_clusters_per_stratum_among_evaluated_candidates": selected,
        "target_assurance": target,
        "monte_carlo_confidence": mc_confidence,
        "monte_carlo_alpha_bonferroni": alpha_per_assurance_bound,
        "bootstrap_replicates_per_study": bootstrap_replicates,
        "assurance_replicates_per_candidate_and_stratum": assurance_replicates,
        "candidate_search_count": len(candidates),
        "independent_strata_count": len(expected_strata),
        "scenario_sha256": scenario_digest,
        "seed": seed,
        "complete_sample_size": None,
        "sampling_units_are_scene_clusters": True,
        "limitations": [
            "All cluster-summary distributions are hypothetical and unverified; no measurements or pilot data are read.",
            "The assurance estimate is conditional on the supplied finite-support IID scenario and exact evaluator rules.",
            "Scene-cluster counts remain separate from material acquisition-unit counts; no combined sample size is inferred.",
            "This planner never qualifies , authorizes RT production, or verifies physical independence or scene representativeness.",
        ],
    }


def _validate_complex_csi_assurance_scenarios(
    scenarios: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[tuple[str, str], tuple[list[dict[str, Any]], np.ndarray]]:
    validated: dict[tuple[str, str], tuple[list[dict[str, Any]], np.ndarray]] = {}
    required = {"assumed_cluster_summaries", "probabilities"}
    atom_fields = {"cluster_median_e_h", "cluster_p95_e_h", "cluster_failed"}
    for stratum, raw_scenario in scenarios.items():
        if (
            not isinstance(stratum, tuple)
            or len(stratum) != 2
            or any(not isinstance(part, str) or not part.strip() for part in stratum)
            or stratum[1] not in {"LOS", "NLOS"}
        ):
            raise ValueError("CSI scenario keys must be (nonempty family_id, LOS or NLOS) tuples")
        if not isinstance(raw_scenario, Mapping) or set(raw_scenario) != required:
            raise ValueError("each CSI scenario must contain cluster summaries and probabilities")
        atoms_raw = raw_scenario["assumed_cluster_summaries"]
        probabilities_raw = raw_scenario["probabilities"]
        if (
            not isinstance(atoms_raw, Sequence)
            or isinstance(atoms_raw, (str, bytes))
            or not atoms_raw
        ):
            raise ValueError("assumed_cluster_summaries must be a nonempty sequence")
        if (
            not isinstance(probabilities_raw, Sequence)
            or isinstance(probabilities_raw, (str, bytes))
            or len(probabilities_raw) != len(atoms_raw)
        ):
            raise ValueError("CSI scenario probabilities must match the summary atoms")
        atoms: list[dict[str, Any]] = []
        for atom in atoms_raw:
            if not isinstance(atom, Mapping) or set(atom) != atom_fields:
                raise ValueError("each assumed CSI cluster summary has an invalid field set")
            median = atom["cluster_median_e_h"]
            p95 = atom["cluster_p95_e_h"]
            failed = atom["cluster_failed"]
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
                for value in (median, p95)
            ):
                raise ValueError("CSI cluster median and p95 assumptions must be finite and >= 0")
            if median > p95:
                raise ValueError("cluster_p95_e_h must be >= cluster_median_e_h")
            if not isinstance(failed, bool):
                raise ValueError("cluster_failed assumptions must be boolean")
            atoms.append(
                {
                    "cluster_median_e_h": float(median),
                    "cluster_p95_e_h": float(p95),
                    "cluster_failed": failed,
                }
            )
        probabilities = np.asarray(probabilities_raw, dtype=np.float64)
        if (
            probabilities.ndim != 1
            or not np.all(np.isfinite(probabilities))
            or np.any(probabilities <= 0.0)
            or abs(float(probabilities.sum()) - 1.0) > 1e-9
        ):
            raise ValueError("CSI scenario probabilities must be positive and sum to one")
        validated[stratum] = atoms, probabilities
    return validated


def _bootstrap_acceptance_checks(
    unit_values: np.ndarray,
    *,
    rng: np.random.Generator,
    bootstrap_replicates: int,
    per_comparison_alpha: float,
) -> dict[str, bool]:
    """Apply the equal-unit point and percentile-bootstrap margins."""
    unit_count = unit_values.shape[0]
    point = np.asarray(
        [
            np.median(unit_values[:, 0]),
            np.percentile(unit_values[:, 1], 95),
            np.max(unit_values[:, 2]),
            np.mean(unit_values[:, 3]),
            np.median(unit_values[:, 4]),
        ],
        dtype=np.float64,
    )
    sampled = unit_values[rng.integers(0, unit_count, size=(bootstrap_replicates, unit_count))]
    bootstrap_statistics = np.column_stack(
        (
            np.median(sampled[:, :, 0], axis=1),
            np.percentile(sampled[:, :, 1], 95, axis=1),
            np.max(sampled[:, :, 2], axis=1),
            np.mean(sampled[:, :, 3], axis=1),
            np.median(sampled[:, :, 4], axis=1),
        )
    )
    tail_percent = 100.0 * per_comparison_alpha / 2.0
    interval = np.percentile(
        bootstrap_statistics,
        [tail_percent, 100.0 - tail_percent],
        axis=0,
    )
    return {
        "median_absolute_power_error_db": bool(
            point[0] <= MAX_POWER_ERROR_MEDIAN_DB and interval[1, 0] <= MAX_POWER_ERROR_MEDIAN_DB
        ),
        "p95_absolute_power_error_db": bool(
            point[1] <= MAX_POWER_ERROR_P95_DB and interval[1, 1] <= MAX_POWER_ERROR_P95_DB
        ),
        "maximum_angular_cdf_ks": bool(point[2] <= 0.10 and interval[1, 2] <= 0.10),
        "nominal_95_interval_coverage": bool(
            MIN_INTERVAL_COVERAGE <= point[3] <= MAX_INTERVAL_COVERAGE
            and interval[0, 3] >= MIN_INTERVAL_COVERAGE
            and interval[1, 3] <= MAX_INTERVAL_COVERAGE
        ),
        "median_prediction_interval_width_db": bool(
            point[4] <= MAX_MEDIAN_INTERVAL_WIDTH_DB
            and interval[1, 4] <= MAX_MEDIAN_INTERVAL_WIDTH_DB
        ),
    }


def _validate_assumption_scenarios(
    scenarios: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], str]:
    if not isinstance(scenarios, Mapping) or not scenarios:
        raise ValueError("assumed_iid_scenarios_by_stratum must be a nonempty mapping")
    if any(not isinstance(stratum, str) or not stratum.strip() for stratum in scenarios):
        raise ValueError("scenario stratum names must be nonempty strings")
    validated: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    canonical: dict[str, Any] = {}
    for stratum in sorted(scenarios):
        scenario = scenarios[stratum]
        if not isinstance(scenario, Mapping) or set(scenario) != {
            "assumed_unit_summaries",
            "probabilities",
        }:
            raise ValueError(
                f"scenario {stratum!r} must contain exactly assumed_unit_summaries and probabilities"
            )
        atoms = scenario["assumed_unit_summaries"]
        try:
            raw_probabilities = np.asarray(scenario["probabilities"])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"scenario {stratum!r} probabilities must be a numeric sequence"
            ) from error
        if raw_probabilities.dtype.kind == "b":
            raise ValueError(f"scenario {stratum!r} probabilities must be numeric weights")
        try:
            probabilities = np.asarray(raw_probabilities, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"scenario {stratum!r} probabilities must be numeric weights"
            ) from error
        if (
            isinstance(atoms, (str, bytes))
            or not isinstance(atoms, Sequence)
            or not atoms
            or probabilities.ndim != 1
            or probabilities.shape != (len(atoms),)
        ):
            raise ValueError(f"scenario {stratum!r} has incompatible atom and probability counts")
        if (
            not np.all(np.isfinite(probabilities))
            or np.any(probabilities <= 0.0)
            or not math.isclose(float(probabilities.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12)
        ):
            raise ValueError(f"scenario {stratum!r} probabilities must be positive and sum to one")

        rows: list[list[float]] = []
        for atom_index, atom in enumerate(atoms):
            if not isinstance(atom, Mapping) or set(atom) != set(_BOOTSTRAP_METRICS):
                raise ValueError(
                    f"scenario {stratum!r} atom {atom_index} must specify exactly the five metrics"
                )
            values: list[float] = []
            for metric in _BOOTSTRAP_METRICS:
                value = atom[metric]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"scenario {stratum!r} {metric} must be a finite number")
                numeric = float(value)
                if not math.isfinite(numeric):
                    raise ValueError(f"scenario {stratum!r} {metric} must be a finite number")
                values.append(numeric)
            if (
                values[0] < 0.0
                or values[1] < 0.0
                or not 0.0 <= values[2] <= 1.0
                or not 0.0 <= values[3] <= 1.0
                or values[4] < 0.0
            ):
                raise ValueError(f"scenario {stratum!r} atom {atom_index} is outside metric ranges")
            rows.append(values)

        matrix = np.asarray(rows, dtype=np.float64)
        weights = probabilities / probabilities.sum()
        validated[stratum] = matrix, weights
        canonical[stratum] = {
            "assumed_unit_summaries": [
                {
                    metric: float(value)
                    for metric, value in zip(_BOOTSTRAP_METRICS, row, strict=True)
                }
                for row in matrix
            ],
            "probabilities": [float(value) for value in weights],
        }
    digest = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return validated, digest


def plan_bootstrap_assurance(
    *,
    assumed_iid_scenarios_by_stratum: Mapping[str, Mapping[str, Any]],
    candidate_units_per_stratum: Sequence[int],
    joint_acceptance_family: Mapping[str, Any] | None = None,
    stratum_roster_by_id: Mapping[str, Sequence[str]] | None = None,
    bootstrap_replicates: int = 2_000,
    assurance_replicates: int = 500,
    target_assurance: float = 0.80,
    familywise_alpha: float = 0.05,
    familywise_claim_count: int | None = None,
    monte_carlo_confidence: float = 0.95,
    seed: int = 1,
) -> dict[str, Any]:
    """Estimate bootstrap pass assurance under explicit discrete IID scenarios.

    Each named stratum receives a caller-specified finite-support distribution of
    *hypothetical independent-unit summaries*. One Monte Carlo study draws the
    requested number of units from that distribution, then runs the same
    equal-unit point statistics and Bonferroni percentile bootstrap checks as
    the evaluator. A one-sided Clopper-Pearson lower bound is computed for
    each stratum/candidate; a union bound combines strata without assuming
    cross-stratum independence. Bonferroni also covers searching the supplied
    candidate counts.

    This is conditional assurance for the supplied scenario distributions,
    not power or precision evidence for an unknown physical population. Inputs
    are assumptions, not measurements; their provenance cannot be verified
    here. The result never qualifies a material, scene, or dataset and never
    replaces external acquisition-unit, calibration, or legal review. When the
    acceptance family and material stratum-ID roster are supplied, the report
    binds every named scenario stratum to the exact frozen material roster for
    downstream joint-design composition.
    """
    validated, scenario_digest = _validate_assumption_scenarios(assumed_iid_scenarios_by_stratum)
    family_identity: dict[str, str] | None = None
    normalized_roster: dict[str, list[str]] | None = None
    if (joint_acceptance_family is None) != (stratum_roster_by_id is None):
        raise ValueError(
            "joint_acceptance_family and stratum_roster_by_id must be supplied together"
        )
    if joint_acceptance_family is not None and stratum_roster_by_id is not None:
        family = validate_joint_acceptance_family(dict(joint_acceptance_family))
        expected = {tuple(row) for row in family["material_expected_strata"]}
        if set(stratum_roster_by_id) != set(validated):
            raise ValueError("material stratum roster IDs must match the scenario strata")
        normalized_roster = {}
        observed: set[tuple[str, ...]] = set()
        for stratum_id, stratum in stratum_roster_by_id.items():
            key = tuple(stratum)
            if len(key) != 3 or any(not isinstance(part, str) or not part for part in key):
                raise ValueError("material stratum roster entries must have three nonempty strings")
            if key in observed:
                raise ValueError("material stratum roster contains duplicate strata")
            observed.add(key)
            normalized_roster[stratum_id] = list(key)
        if observed != expected:
            raise ValueError("material stratum roster differs from acceptance-family strata")
        family_identity = {
            "family_id": family["family_id"],
            "family_sha256": family["family_sha256"],
        }
    if (
        isinstance(candidate_units_per_stratum, (str, bytes))
        or not isinstance(candidate_units_per_stratum, Sequence)
        or not candidate_units_per_stratum
    ):
        raise ValueError("candidate_units_per_stratum must be a nonempty integer sequence")
    candidates = list(candidate_units_per_stratum)
    if any(
        isinstance(count, bool)
        or not isinstance(count, int)
        or count < MIN_INDEPENDENT_MEASUREMENT_UNITS
        for count in candidates
    ):
        raise ValueError(
            f"candidate unit counts must be integers >= {MIN_INDEPENDENT_MEASUREMENT_UNITS}"
        )
    if len(set(candidates)) != len(candidates):
        raise ValueError("candidate unit counts must be unique")
    candidates.sort()
    for name, value, minimum in (
        ("bootstrap_replicates", bootstrap_replicates, 1_000),
        ("assurance_replicates", assurance_replicates, 100),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")

    target = _probability(target_assurance, "target_assurance")
    alpha = _probability(familywise_alpha, "familywise_alpha")
    mc_confidence = _probability(monte_carlo_confidence, "monte_carlo_confidence")
    resampling_cells = (
        len(validated) * assurance_replicates * bootstrap_replicates * sum(candidates)
    )
    if resampling_cells > _MAX_BOOTSTRAP_RESAMPLING_CELLS:
        raise ValueError(
            "requested design exceeds the bounded bootstrap planning work limit; "
            "reduce candidates or replication counts"
        )
    peak_bootstrap_cells = bootstrap_replicates * max(candidates)
    if peak_bootstrap_cells > _MAX_BOOTSTRAP_PEAK_CELLS:
        raise ValueError(
            "a single bootstrap resample exceeds the bounded memory limit; "
            "reduce the largest candidate count or bootstrap replicates"
        )

    local_claim_count = len(validated) * len(_BOOTSTRAP_METRICS)
    if familywise_claim_count is None:
        familywise_claim_count = local_claim_count
    if (
        isinstance(familywise_claim_count, bool)
        or not isinstance(familywise_claim_count, int)
        or familywise_claim_count < local_claim_count
    ):
        raise ValueError("familywise_claim_count must cover every material bootstrap claim")
    per_comparison_alpha = alpha / familywise_claim_count
    candidate_alpha = (1.0 - mc_confidence) / (len(candidates) * len(validated))
    candidate_results: list[dict[str, Any]] = []
    for candidate_index, unit_count in enumerate(candidates):
        stratum_results: dict[str, Any] = {}
        lower_bounds: list[float] = []
        for stratum_index, (stratum, (atoms, weights)) in enumerate(validated.items()):
            rng = np.random.default_rng(
                np.random.SeedSequence([seed, candidate_index, stratum_index])
            )
            metric_successes = {metric: 0 for metric in _BOOTSTRAP_METRICS}
            joint_successes = 0
            for _ in range(assurance_replicates):
                sampled = atoms[
                    rng.choice(atoms.shape[0], size=unit_count, replace=True, p=weights)
                ]
                checks = _bootstrap_acceptance_checks(
                    sampled,
                    rng=rng,
                    bootstrap_replicates=bootstrap_replicates,
                    per_comparison_alpha=per_comparison_alpha,
                )
                for metric, passed in checks.items():
                    metric_successes[metric] += int(passed)
                joint_successes += int(all(checks.values()))

            lower = (
                0.0
                if joint_successes == 0
                else float(
                    beta.ppf(
                        candidate_alpha, joint_successes, assurance_replicates - joint_successes + 1
                    )
                )
            )
            lower_bounds.append(lower)
            stratum_results[stratum] = {
                "assumed_support_atom_count": int(atoms.shape[0]),
                "joint_pass_count": joint_successes,
                "joint_assurance_estimate": joint_successes / assurance_replicates,
                "joint_assurance_one_sided_exact_lower_bound": lower,
                "metric_pass_probability_estimates": {
                    metric: count / assurance_replicates
                    for metric, count in metric_successes.items()
                },
            }
        joint_lower_bound = max(0.0, 1.0 - sum(1.0 - bound for bound in lower_bounds))
        candidate_results.append(
            {
                "independent_units_per_stratum": unit_count,
                "stratum_assurance": stratum_results,
                "joint_assurance_lower_bound_union_bound": joint_lower_bound,
                "meets_target_assurance": joint_lower_bound >= target,
            }
        )

    selected = next(
        (
            row["independent_units_per_stratum"]
            for row in candidate_results
            if row["meets_target_assurance"]
        ),
        None,
    )
    return {
        "schema": "loslab.bootstrap_assurance_planning.v1",
        "status": "SCENARIO_CONDITIONAL_PLANNING_ONLY_NOT_QUALIFICATION",
        "qualification_status": "NOT_QUALIFIED",
        "admission_possible": False,
        "directly_reads_measurement_files": False,
        "scenario_distribution_sha256": scenario_digest,
        "joint_acceptance_family": family_identity,
        "stratum_roster_by_id": normalized_roster,
        "assumption_model": (
            "Within each stratum, independent acquisition-unit summaries are IID draws "
            "from the supplied finite-support distribution. Cross-stratum dependence is "
            "unrestricted; a union bound combines stratum pass probabilities."
        ),
        "scenario_provenance_verified": False,
        "familywise_alpha": alpha,
        "familywise_claim_count": familywise_claim_count,
        "per_comparison_alpha": per_comparison_alpha,
        "target_assurance": target,
        "monte_carlo_confidence": mc_confidence,
        "monte_carlo_bound": "one-sided exact Clopper-Pearson, Bonferroni across candidates and strata",
        "assurance_replicates": assurance_replicates,
        "bootstrap_replicates_per_study": bootstrap_replicates,
        "candidate_designs": candidate_results,
        "selected_units_per_stratum_among_evaluated_candidates": selected,
        "selected_candidate_is_smallest_tested_not_global_minimum": True,
        "complete_sample_size": None,
        "complete_design_status": "UNRESOLVED_EXTERNAL_POPULATION_AND_ACQUISITION_EVIDENCE",
        "limitations": [
            "Assurance applies only to the supplied hypothetical finite-support distributions.",
            "Scenario assumptions and their representativeness are not externally verified.",
            "Complex-CSI, path-truth, NR-impact, legal-rights, and independent-acquisition gates are not qualified here.",
            "A candidate grid search does not establish the minimum over untested sample sizes.",
        ],
        "bounded_resampling_cell_count": resampling_cells,
        "peak_bootstrap_sample_cell_count": peak_bootstrap_cells,
    }
