"""Assumption-driven sample-size planning for G3 material and complex CSI.

This module is separate from the G3 evaluators and consumes no measurement
data. It provides distribution-free and exact-binomial designs plus nested
simulations of the evaluator-aligned bootstrap rules under explicitly
hypothetical finite-support distributions. The maximum-over-units angular-KS
criterion uses an explicit IID per-unit pass-probability assumption; its
literal finite-sample maximum gives a pass-probability upper bound on n, not a
population-maximum guarantee. Receiver-impact and propagation assurance live in
sibling modules and are orchestrated by the acquisition-design CLI. Component
designs use distinct physical-unit scales and never become a complete G3
sample-size claim: external acquisition authority, representativeness, rights,
path truth, and a unified cost/design model remain unresolved. No function in
this planner qualifies material or complex-CSI performance.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.stats import beta, binom

from evaluation.g3_joint_acceptance import (
    COMPLEX_CSI_CLAIMS,
    FAMILY_SCHEMA_V2,
    FAMILY_SCHEMA_V3,
    FAMILY_SCHEMA_V4,
    MATERIAL_CLAIMS,
    PROPAGATION_CLAIMS,
    RECEIVER_IMPACT_CLAIMS,
    family_alpha_for_claim,
    validate_g3_joint_acceptance_family,
)

# Mirrored from the G3.03 evaluator's frozen project-defined margins. Kept
# local so this data-free planner has no dependency on evaluator code/data.
MAX_MEDIAN_INTERVAL_WIDTH_DB = 3.0
MAX_POWER_ERROR_MEDIAN_DB = 1.5
MAX_POWER_ERROR_P95_DB = 3.0
MAX_INTERVAL_COVERAGE = 0.98
MIN_INTERVAL_COVERAGE = 0.90
MIN_INDEPENDENT_MEASUREMENT_UNITS = 20

_QUANTILE_TARGETS: dict[str, tuple[float, float]] = {
    # (population quantile to bound, project-defined upper margin)
    "median_absolute_power_error_db": (0.50, MAX_POWER_ERROR_MEDIAN_DB),
    "p95_absolute_power_error_db": (0.95, MAX_POWER_ERROR_P95_DB),
    "median_prediction_interval_width_db": (0.50, MAX_MEDIAN_INTERVAL_WIDTH_DB),
}
_COMPARISONS_PER_STRATUM = 5
_ALL_G3_CLAIMS_PER_STRATUM = _COMPARISONS_PER_STRATUM + 1
_COMPLEX_CSI_FAILURE_LIMIT = 0.05
_COMPLEX_CSI_P95_LIMIT = 0.05
_PROPAGATION_METRICS = (
    "first_arrival_delay_s",
    "rms_delay_spread_s",
    "arrival_angular_spread_deg",
    "departure_angular_spread_deg",
)
_PROPAGATION_P95_LIMIT = 1.0
_PROPAGATION_FAILURE_LIMIT = 0.05
_EXACT_BINOMIAL_SEARCH_LIMIT = 10_000_000
_EXACT_BINOMIAL_CHUNK_SIZE = 16_384
_BOOTSTRAP_METRICS = (
    "median_absolute_power_error_db",
    "p95_absolute_power_error_db",
    "maximum_angular_cdf_ks",
    "nominal_95_interval_coverage",
    "median_prediction_interval_width_db",
)
_MAX_BOOTSTRAP_RESAMPLING_CELLS = 100_000_000
_MAX_BOOTSTRAP_PEAK_CELLS = 2_000_000


def _probability(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite probability")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0 or result >= 1.0:
        raise ValueError(f"{name} must be in (0, 1)")
    return result


def _required_quantile_units(
    *,
    quantile: float,
    planned_pass_probability: float,
    alpha: float,
    beta: float,
) -> int:
    """Hoeffding design for proving P(unit summary <= margin) > quantile.

    The pass indicator is Bernoulli. The one-sided lower confidence bound and
    its power are both distribution-free. Requiring the sum of their Hoeffding
    deviations to fit inside the assumed pass-rate separation is conservative.
    """
    separation = planned_pass_probability - quantile
    if separation <= 0.0:
        raise ValueError("planned pass probability must exceed the target quantile")
    deviation_sum = math.sqrt(math.log(1.0 / alpha)) + math.sqrt(math.log(1.0 / beta))
    return math.ceil((deviation_sum / separation) ** 2 / 2.0)


def _required_bounded_mean_units(
    *,
    planned_mean: float,
    lower_margin: float,
    upper_margin: float,
    outcome_lower: float,
    outcome_upper: float,
    alpha: float,
    beta: float,
) -> int:
    """Hoeffding design for a two-sided CI wholly inside the allowed margins."""
    if not outcome_lower <= planned_mean <= outcome_upper:
        raise ValueError("planned_mean must lie within the bounded outcome range")
    if not lower_margin < upper_margin:
        raise ValueError("coverage margins must be strictly ordered")
    if outcome_lower > lower_margin or outcome_upper < upper_margin:
        raise ValueError("outcome range must contain the complete acceptance interval")
    slack = min(planned_mean - lower_margin, upper_margin - planned_mean)
    if slack <= 0.0:
        raise ValueError("planned_mean must be strictly inside both coverage margins")

    # With probability >= 1-beta the estimate is within e of its planned mean;
    # its two-sided (1-alpha) Hoeffding CI has radius r. Requiring e+r <= slack
    # places that CI wholly inside the acceptance interval.
    deviations = math.sqrt(math.log(2.0 / alpha)) + math.sqrt(math.log(2.0 / beta))
    return math.ceil(deviations**2 / (2.0 * slack**2))


def _maximum_angular_units_for_pass_probability(
    *,
    planned_unit_pass_probability: float,
    minimum_pass_probability: float,
) -> int | None:
    """Largest n for which an IID sample maximum passes with target probability.

    For per-unit angular summary U and frozen threshold t, the evaluator's
    point rule is max(U_1, ..., U_n) <= t. Its bootstrap statistic is the max
    of a resample of those same units, so it cannot exceed the observed max.
    Thus the conjunction of point and bootstrap-upper-CI rules is equivalent
    to the point rule, and its pass probability is exactly q**n where
    q=P(U<=t). Return the largest integer n with q**n >= target. None means
    q=1 and there is no finite upper bound; 0 means no positive n meets target.
    """
    q = planned_unit_pass_probability
    target = minimum_pass_probability
    if not math.isfinite(q) or not 0.0 <= q <= 1.0:
        raise ValueError("planned_unit_pass_probability must be in [0, 1]")
    if not math.isfinite(target) or not 0.0 < target < 1.0:
        raise ValueError("minimum_pass_probability must be in (0, 1)")
    if q == 1.0:
        return None
    if q == 0.0 or q < target:
        return 0

    result = math.floor(math.log(target) / math.log(q))
    # Correct floating-point rounding at integer boundaries.
    while result > 0 and q**result < target:
        result -= 1
    while q ** (result + 1) >= target:
        result += 1
    return result


def _exact_binomial_critical_failures(
    cluster_counts: np.ndarray,
    *,
    per_comparison_alpha: float,
) -> np.ndarray:
    """Return the largest accepted failure count under a one-sided exact test.

    A failure count ``x`` is accepted exactly when its one-sided
    Clopper-Pearson upper bound is at most 0.05, equivalently when the
    null-tail probability ``P(Binomial(n, 0.05) <= x)`` is no greater than
    the allocated alpha. SciPy's binomial CDF/quantile use stable special
    functions; the CDF checks correct the quantile boundary explicitly.
    """
    quantile = np.asarray(
        binom.ppf(per_comparison_alpha, cluster_counts, _COMPLEX_CSI_FAILURE_LIMIT)
    )
    critical = quantile.astype(np.int64) - 1

    valid = critical >= 0
    if np.any(valid):
        upper_tail = np.asarray(
            binom.cdf(critical[valid], cluster_counts[valid], _COMPLEX_CSI_FAILURE_LIMIT)
        )
        critical[valid] -= (upper_tail > per_comparison_alpha).astype(np.int64)

    next_count = critical + 1
    valid = next_count <= cluster_counts
    if np.any(valid):
        next_tail = np.asarray(
            binom.cdf(next_count[valid], cluster_counts[valid], _COMPLEX_CSI_FAILURE_LIMIT)
        )
        critical[valid] += (next_tail <= per_comparison_alpha).astype(np.int64)
    return critical


def _exact_binomial_failure_design(
    *,
    planned_failure_probability: float,
    per_comparison_alpha: float,
    per_comparison_power: float,
) -> dict[str, int | float | bool | str | None]:
    """Find the first exact-binomial design meeting the allocated power."""
    start = 1
    while start <= _EXACT_BINOMIAL_SEARCH_LIMIT:
        stop = min(start + _EXACT_BINOMIAL_CHUNK_SIZE, _EXACT_BINOMIAL_SEARCH_LIMIT + 1)
        counts = np.arange(start, stop, dtype=np.int64)
        critical = _exact_binomial_critical_failures(
            counts,
            per_comparison_alpha=per_comparison_alpha,
        )
        powers = np.zeros(counts.shape, dtype=np.float64)
        admissible = critical >= 0
        if np.any(admissible):
            powers[admissible] = binom.cdf(
                critical[admissible], counts[admissible], planned_failure_probability
            )
        passing = np.flatnonzero(powers >= per_comparison_power)
        if passing.size:
            index = int(passing[0])
            return {
                "sample_size_found": True,
                "required_independent_outer_clusters_per_stratum": int(counts[index]),
                "maximum_admissible_observed_failures_per_stratum": int(critical[index]),
                "achieved_per_comparison_power": float(powers[index]),
                "search_limit_per_stratum": _EXACT_BINOMIAL_SEARCH_LIMIT,
                "search_status": "EXACT_BINOMIAL_DESIGN_FOUND",
            }
        start = stop

    return {
        "sample_size_found": False,
        "required_independent_outer_clusters_per_stratum": None,
        "maximum_admissible_observed_failures_per_stratum": None,
        "achieved_per_comparison_power": None,
        "search_limit_per_stratum": _EXACT_BINOMIAL_SEARCH_LIMIT,
        "search_status": "SEARCH_LIMIT_REACHED_NO_DESIGN_RETURNED",
    }


def plan_g3_independent_acquisition_units(
    *,
    stratum_count: int,
    planned_quantile_pass_probabilities: Mapping[str, float],
    planned_interval_coverage: float,
    planned_complex_csi_failure_probability: float,
    planned_angular_unit_pass_probability: float | None = None,
    familywise_alpha: float = 0.05,
    familywise_power: float = 0.80,
) -> dict[str, Any]:
    """Plan balanced independent acquisition units per G3.03 stratum.

    Assumptions are design inputs, not pilot estimates. Each shared input must
    be a conservative worst-stratum value; the planner does not accept or
    infer stratum-specific differences. Each quantile input is
    the preregistered probability that an independent acquisition unit's
    already-defined summary is at or below that metric's frozen margin. The
    interval-coverage input is the planned mean of independent-unit coverage
    summaries. One Bonferroni allocation covers all five distributional
    metric claims and the complex-CSI cluster-failure claim in every declared
    stratum. Familywise power is allocated across that same complete family.
    The returned counts remain planning assumptions and cannot qualify
    material performance.

    Interpreting these inputs as population quantities requires a representative
    sampling frame or an explicit common-distribution assumption across the
    independent physical acquisition units. The angular maximum can be planned
    only if ``planned_angular_unit_pass_probability`` supplies an explicit IID
    assumption for P(per-unit maximum angular CDF KS <= 0.10). Its exact pass
    probability is q**n and yields an upper, not lower, bound on n. Omitting
    the assumption leaves that criterion unresolved. This function does not
    inspect data, modify evaluator behavior, or issue qualification/admission.
    """
    if isinstance(stratum_count, bool) or not isinstance(stratum_count, int) or stratum_count < 1:
        raise ValueError("stratum_count must be a positive integer")
    alpha = _probability(familywise_alpha, "familywise_alpha")
    power = _probability(familywise_power, "familywise_power")
    beta = 1.0 - power
    if beta <= 0.0:
        raise ValueError("familywise_power must be less than 1")
    if not isinstance(planned_quantile_pass_probabilities, Mapping):
        raise ValueError("planned_quantile_pass_probabilities must be a mapping")
    if set(planned_quantile_pass_probabilities) != set(_QUANTILE_TARGETS):
        raise ValueError(
            "planned_quantile_pass_probabilities must specify exactly: "
            + ", ".join(sorted(_QUANTILE_TARGETS))
        )

    coverage_target = _probability(planned_interval_coverage, "planned_interval_coverage")
    if not MIN_INTERVAL_COVERAGE < coverage_target < MAX_INTERVAL_COVERAGE:
        raise ValueError("planned_interval_coverage must be strictly inside G3.03 margins")

    comparison_count = stratum_count * _ALL_G3_CLAIMS_PER_STRATUM
    alpha_per_comparison = alpha / comparison_count
    beta_per_comparison = beta / comparison_count

    planned_failure_probability = _probability(
        planned_complex_csi_failure_probability,
        "planned_complex_csi_failure_probability",
    )
    if planned_failure_probability >= _COMPLEX_CSI_FAILURE_LIMIT:
        raise ValueError(
            "planned_complex_csi_failure_probability must be below the frozen 0.05 limit"
        )
    complex_csi_failure_plan = _exact_binomial_failure_design(
        planned_failure_probability=planned_failure_probability,
        per_comparison_alpha=alpha_per_comparison,
        per_comparison_power=1.0 - beta_per_comparison,
    )
    complex_csi_failure_plan.update(
        {
            "criterion": (
                "For each independent outer cluster, failure is any episode whose e_H exceeds 0.05; "
                "the stratum failure probability's one-sided exact upper bound must be <= 0.05."
            ),
            "failure_probability_limit": _COMPLEX_CSI_FAILURE_LIMIT,
            "one_sided_confidence_level_per_comparison": 1.0 - alpha_per_comparison,
            "familywise_alpha": alpha,
            "familywise_power_target": power,
            "stratum_count": stratum_count,
            "per_comparison_alpha_bonferroni": alpha_per_comparison,
            "per_comparison_power_target_bonferroni": 1.0 - beta_per_comparison,
            "familywise_claim_count": comparison_count,
            "familywise_scope": (
                "five distributional metrics and complex-CSI cluster failure across all strata"
            ),
            "planned_failure_probability": planned_failure_probability,
            "sampling_assumption": (
                "Outer clusters are independent and representative of the target population, "
                "or follow a preregistered common distribution within each stratum."
            ),
            "failure_count_rule": (
                "Accept only observed failure counts whose one-sided Clopper-Pearson upper "
                "bound is <= 0.05 at the Bonferroni-adjusted family-wise confidence level."
            ),
            "single_stratum_one_sided_95pct_zero_failure_minimum_clusters": 59,
            "admission_possible": False,
            "is_complete_g3_03_design": False,
        }
    )

    quantile_plans: dict[str, dict[str, float | int]] = {}
    for metric, (quantile, margin) in _QUANTILE_TARGETS.items():
        planned_pass_probability = _probability(
            planned_quantile_pass_probabilities[metric],
            f"planned_quantile_pass_probabilities[{metric!r}]",
        )
        units = _required_quantile_units(
            quantile=quantile,
            planned_pass_probability=planned_pass_probability,
            alpha=alpha_per_comparison,
            beta=beta_per_comparison,
        )
        quantile_plans[metric] = {
            "target_quantile": quantile,
            "upper_margin": margin,
            "planned_unit_pass_probability": planned_pass_probability,
            "required_independent_units_per_stratum": units,
        }

    coverage_units = _required_bounded_mean_units(
        planned_mean=coverage_target,
        lower_margin=MIN_INTERVAL_COVERAGE,
        upper_margin=MAX_INTERVAL_COVERAGE,
        outcome_lower=0.0,
        outcome_upper=1.0,
        alpha=alpha_per_comparison,
        beta=beta_per_comparison,
    )
    supported_counts = [
        int(item["required_independent_units_per_stratum"]) for item in quantile_plans.values()
    ] + [coverage_units]
    supported_lower_bound = max(supported_counts)
    csi_sample_size = complex_csi_failure_plan["required_independent_outer_clusters_per_stratum"]
    all_claims_lower_bound = (
        max(supported_lower_bound, int(csi_sample_size))
        if isinstance(csi_sample_size, int)
        else None
    )
    measurement_unit_lower_bound = (
        max(all_claims_lower_bound, MIN_INDEPENDENT_MEASUREMENT_UNITS)
        if all_claims_lower_bound is not None
        else None
    )

    angular_plan: dict[str, Any]
    if planned_angular_unit_pass_probability is None:
        angular_plan = {
            "assumption_supplied": False,
            "planned_unit_pass_probability": None,
            "minimum_per_stratum_pass_probability_target": None,
            "maximum_independent_units_per_stratum": None,
            "maximum_units_status": "UNRESOLVED_MISSING_EXPLICIT_PASS_PROBABILITY",
            "compatible_with_combined_six_claim_lower_bound": None,
            "interpretation": (
                "Supply an explicit IID probability that one independent unit's maximum "
                "angular CDF KS is <= 0.10; no value is inferred from measurement or pilot data."
            ),
        }
        complete_design_status = "UNRESOLVED_MISSING_ANGULAR_PASS_PROBABILITY_ASSUMPTION"
    else:
        angular_pass_probability = planned_angular_unit_pass_probability
        if (
            isinstance(angular_pass_probability, bool)
            or not isinstance(angular_pass_probability, (int, float))
            or not math.isfinite(float(angular_pass_probability))
            or not 0.0 <= float(angular_pass_probability) <= 1.0
        ):
            raise ValueError("planned_angular_unit_pass_probability must be in [0, 1]")
        angular_pass_probability = float(angular_pass_probability)
        angular_power_target = 1.0 - beta_per_comparison
        angular_maximum_units = _maximum_angular_units_for_pass_probability(
            planned_unit_pass_probability=angular_pass_probability,
            minimum_pass_probability=angular_power_target,
        )
        angular_compatible = (
            None
            if measurement_unit_lower_bound is None
            else angular_maximum_units is None
            or angular_maximum_units >= measurement_unit_lower_bound
        )
        angular_plan = {
            "assumption_supplied": True,
            "planned_unit_pass_probability": angular_pass_probability,
            "minimum_per_stratum_pass_probability_target": angular_power_target,
            "maximum_independent_units_per_stratum": angular_maximum_units,
            "maximum_units_status": (
                "UNBOUNDED_Q_EQUALS_ONE"
                if angular_maximum_units is None
                else "FINITE_UPPER_BOUND"
                if angular_maximum_units > 0
                else "NO_POSITIVE_SAMPLE_SIZE_MEETS_POWER_TARGET"
            ),
            "pilot_floor_units_per_stratum": MIN_INDEPENDENT_MEASUREMENT_UNITS,
            "compatible_with_pilot_floor": (
                angular_maximum_units is None
                or angular_maximum_units >= MIN_INDEPENDENT_MEASUREMENT_UNITS
            ),
            "minimum_units_for_combined_six_claim_lower_bound": measurement_unit_lower_bound,
            "compatible_with_combined_six_claim_lower_bound": angular_compatible,
            "pass_probability_at_maximum_units": (
                None
                if angular_maximum_units is None or angular_compatible is not True
                else angular_pass_probability**angular_maximum_units
            ),
            "acceptance_event": (
                "Every independent unit's maximum angular CDF KS is <= 0.10; "
                "conditional on that point rule, every unit-resampled bootstrap "
                "maximum and its percentile upper endpoint are also <= 0.10."
            ),
            "pass_probability_formula": "planned_unit_pass_probability ** n",
            "sampling_assumption": (
                "Independent measurement units are IID within each stratum, and the supplied "
                "pass probability is a conservative weakest-stratum value."
            ),
            "interpretation": (
                "Exact probability of passing this finite-sample maximum rule under the stated "
                "IID assumption; larger n can reduce pass probability. This is not a confidence "
                "bound for a population essential supremum or a qualification result."
            ),
        }
        complete_design_status = (
            "UNRESOLVED_COMPLEX_CSI_SAMPLE_SIZE_SEARCH"
            if measurement_unit_lower_bound is None
            else "INFEASIBLE_UNDER_SUPPLIED_ANGULAR_ASSUMPTION"
            if angular_compatible is False
            else "PARTIAL_ONLY_EVALUATOR_ALIGNED_BOOTSTRAP_AND_EXTERNAL_GATES_REMAIN"
        )

    return {
        "schema": "loslab.g3_sample_size_planning.v3",
        "status": "PLANNING_ONLY_NOT_QUALIFICATION",
        "uses_measurement_or_pilot_data": False,
        "admission_possible": False,
        "stratum_count": stratum_count,
        "familywise_alpha": alpha,
        "familywise_power_target": power,
        "complex_csi_cluster_failure_plan": complex_csi_failure_plan,
        "distributional_metrics_per_stratum": _COMPARISONS_PER_STRATUM,
        "all_claims_per_stratum": _ALL_G3_CLAIMS_PER_STRATUM,
        "bonferroni_comparison_count": comparison_count,
        "per_comparison_alpha": alpha_per_comparison,
        "per_comparison_beta": beta_per_comparison,
        "independent_unit_definition": "externally verified physical acquisition unit",
        "shared_assumptions_scope": (
            "Each planned input must conservatively represent the weakest stratum; "
            "inputs are currently shared across all strata."
        ),
        "population_sampling_assumption": (
            "Independent acquisition units must represent the target population, "
            "or follow a preregistered common distribution."
        ),
        "quantile_metric_plans": quantile_plans,
        "bounded_mean_metric_plans": {
            "nominal_95_interval_coverage": {
                "planned_mean": coverage_target,
                "acceptance_interval": [MIN_INTERVAL_COVERAGE, MAX_INTERVAL_COVERAGE],
                "outcome_range": [0.0, 1.0],
                "required_independent_units_per_stratum": coverage_units,
            }
        },
        "angular_maximum_metric_plan": angular_plan,
        "hypothetical_required_units_per_stratum_for_supported_metrics": supported_lower_bound,
        "hypothetical_required_units_per_stratum_for_all_six_claims": all_claims_lower_bound,
        "minimum_units_for_all_six_claims_and_pilot_floor": measurement_unit_lower_bound,
        "pilot_floor_units_per_stratum": MIN_INDEPENDENT_MEASUREMENT_UNITS,
        "pilot_floor_is_not_qualification_sample_size": True,
        "complete_g3_03_sample_size": None,
        "complete_design_status": complete_design_status,
        "evaluation_alignment": (
            "These Hoeffding bounds are conservative design calculations, not a sample-size "
            "guarantee for the evaluator's Bonferroni percentile cluster-bootstrap intervals. "
            "The angular maximum calculation exactly plans the finite-sample point/bootstrap "
            "acceptance event only under its explicit IID pass-probability assumption."
        ),
        "methods": {
            "quantiles": "one-sided Hoeffding lower bound and power on independent-unit pass indicators",
            "coverage": "two-sided Hoeffding confidence interval with conservative power allocation",
            "multiplicity": (
                "Bonferroni across five distributional metrics and the complex-CSI claim "
                "per expected stratum"
            ),
        },
        "assumptions_must_be_preregistered_and_not_estimated_from_pilot": True,
    }


def plan_g3_joint_independent_units(
    *,
    joint_acceptance_family: dict[str, Any],
    planned_quantile_pass_probabilities: Mapping[str, float],
    planned_interval_coverage: float,
    planned_complex_csi_p95_pass_probability: float,
    planned_complex_csi_failure_probability: float,
    planned_angular_unit_pass_probability: float | None,
    planned_propagation_p95_pass_probabilities: Mapping[str, float] | None = None,
    planned_propagation_cluster_failure_probabilities: Mapping[str, float] | None = None,
    familywise_power: float = 0.80,
) -> dict[str, Any]:
    """Plan separate material-unit and scene-cluster counts under one G3 family.

    The shared family determines the Bonferroni alpha and power allocation.
    Material units and propagation/CSI scene clusters remain separate sampling
    units. Version-2 and v3 families require explicit worst-stratum
    pass/failure assumptions for each propagation metric and return a separate
    count per declared propagation stratum. V3 also records receiver strata;
    evaluator-aligned receiver assurance is supplied by its nested-bootstrap
    planner. No single combined sample-size claim is made.
    """
    family = validate_g3_joint_acceptance_family(joint_acceptance_family)
    material_strata = family["material_expected_strata"]
    csi_strata = family["complex_csi_expected_strata"]
    propagation_strata = family.get("propagation_expected_strata", [])
    per_claim_alpha, family_claim_count = family_alpha_for_claim(
        family,
        evaluator="material_calibration",
        expected_strata=material_strata,
    )
    csi_alpha, csi_family_claim_count = family_alpha_for_claim(
        family,
        evaluator="complex_csi",
        expected_strata=csi_strata,
    )
    if per_claim_alpha != csi_alpha or family_claim_count != csi_family_claim_count:
        raise ValueError("joint G3 claim allocation is internally inconsistent")
    expected_claim_count = len(material_strata) * len(MATERIAL_CLAIMS) + len(csi_strata) * len(
        COMPLEX_CSI_CLAIMS
    )
    if family["schema"] == FAMILY_SCHEMA_V2:
        propagation_alpha, propagation_claim_count = family_alpha_for_claim(
            family,
            evaluator="propagation_metrics",
            expected_strata=propagation_strata,
        )
        if propagation_alpha != per_claim_alpha or propagation_claim_count != family_claim_count:
            raise ValueError("joint G3 claim allocation is internally inconsistent")
        expected_claim_count += len(propagation_strata) * len(PROPAGATION_CLAIMS)
    elif family["schema"] in {FAMILY_SCHEMA_V3, FAMILY_SCHEMA_V4}:
        propagation_alpha, propagation_claim_count = family_alpha_for_claim(
            family,
            evaluator="propagation_metrics",
            expected_strata=propagation_strata,
        )
        receiver_strata = family["receiver_impact_expected_strata"]
        receiver_alpha, receiver_claim_count = family_alpha_for_claim(
            family,
            evaluator="receiver_impact",
            expected_strata=receiver_strata,
        )
        if any(value != per_claim_alpha for value in (propagation_alpha, receiver_alpha)) or any(
            value != family_claim_count for value in (propagation_claim_count, receiver_claim_count)
        ):
            raise ValueError("joint G3 claim allocation is internally inconsistent")
        expected_claim_count += len(propagation_strata) * len(PROPAGATION_CLAIMS)
        expected_claim_count += len(receiver_strata) * len(RECEIVER_IMPACT_CLAIMS)
    elif (
        planned_propagation_p95_pass_probabilities is not None
        or planned_propagation_cluster_failure_probabilities is not None
    ):
        raise ValueError("propagation planning assumptions require a v2 or v3 joint G3 family")
    if family_claim_count != expected_claim_count:
        raise ValueError("joint G3 family claim count does not match its declared strata")

    alpha = float(family["familywise_alpha"])
    power = _probability(familywise_power, "familywise_power")
    beta_per_claim = (1.0 - power) / family_claim_count
    per_claim_power = 1.0 - beta_per_claim
    coverage_target = _probability(planned_interval_coverage, "planned_interval_coverage")
    if not MIN_INTERVAL_COVERAGE < coverage_target < MAX_INTERVAL_COVERAGE:
        raise ValueError("planned_interval_coverage must be strictly inside G3.03 margins")
    if not isinstance(planned_quantile_pass_probabilities, Mapping):
        raise ValueError("planned_quantile_pass_probabilities must be a mapping")
    if set(planned_quantile_pass_probabilities) != set(_QUANTILE_TARGETS):
        raise ValueError(
            "planned_quantile_pass_probabilities must specify exactly: "
            + ", ".join(sorted(_QUANTILE_TARGETS))
        )

    material_quantile_plans: dict[str, dict[str, float | int]] = {}
    for metric, (quantile, margin) in _QUANTILE_TARGETS.items():
        pass_probability = _probability(
            planned_quantile_pass_probabilities[metric],
            f"planned_quantile_pass_probabilities[{metric!r}]",
        )
        material_quantile_plans[metric] = {
            "target_quantile": quantile,
            "acceptance_margin": margin,
            "planned_unit_pass_probability": pass_probability,
            "required_independent_measurement_units_per_stratum": _required_quantile_units(
                quantile=quantile,
                planned_pass_probability=pass_probability,
                alpha=per_claim_alpha,
                beta=beta_per_claim,
            ),
        }
    coverage_units = _required_bounded_mean_units(
        planned_mean=coverage_target,
        lower_margin=MIN_INTERVAL_COVERAGE,
        upper_margin=MAX_INTERVAL_COVERAGE,
        outcome_lower=0.0,
        outcome_upper=1.0,
        alpha=per_claim_alpha,
        beta=beta_per_claim,
    )
    material_supported = max(
        [
            int(value["required_independent_measurement_units_per_stratum"])
            for value in material_quantile_plans.values()
        ]
        + [coverage_units, MIN_INDEPENDENT_MEASUREMENT_UNITS]
    )

    csi_p95_probability = _probability(
        planned_complex_csi_p95_pass_probability,
        "planned_complex_csi_p95_pass_probability",
    )
    csi_p95_units = _required_quantile_units(
        quantile=0.95,
        planned_pass_probability=csi_p95_probability,
        alpha=per_claim_alpha,
        beta=beta_per_claim,
    )
    csi_failure_probability = _probability(
        planned_complex_csi_failure_probability,
        "planned_complex_csi_failure_probability",
    )
    if csi_failure_probability >= _COMPLEX_CSI_FAILURE_LIMIT:
        raise ValueError("planned_complex_csi_failure_probability must be below 0.05")
    csi_failure_design = _exact_binomial_failure_design(
        planned_failure_probability=csi_failure_probability,
        per_comparison_alpha=per_claim_alpha,
        per_comparison_power=per_claim_power,
    )
    csi_failure_units = csi_failure_design["required_independent_outer_clusters_per_stratum"]
    csi_supported = (
        max(csi_p95_units, int(csi_failure_units)) if isinstance(csi_failure_units, int) else None
    )

    propagation_design: dict[str, Any] | None = None
    if family["schema"] in {FAMILY_SCHEMA_V2, FAMILY_SCHEMA_V3, FAMILY_SCHEMA_V4}:
        if not isinstance(planned_propagation_p95_pass_probabilities, Mapping) or not isinstance(
            planned_propagation_cluster_failure_probabilities, Mapping
        ):
            raise ValueError(
                "v2 G3 planning requires propagation p95-pass and cluster-failure assumptions"
            )
        if set(planned_propagation_p95_pass_probabilities) != set(_PROPAGATION_METRICS):
            raise ValueError(
                "planned_propagation_p95_pass_probabilities must specify exactly: "
                + ", ".join(_PROPAGATION_METRICS)
            )
        if set(planned_propagation_cluster_failure_probabilities) != set(_PROPAGATION_METRICS):
            raise ValueError(
                "planned_propagation_cluster_failure_probabilities must specify exactly: "
                + ", ".join(_PROPAGATION_METRICS)
            )
        propagation_metric_plans: dict[str, dict[str, Any]] = {}
        propagation_supported_counts: list[int] = []
        for metric in _PROPAGATION_METRICS:
            p95_pass_probability = _probability(
                planned_propagation_p95_pass_probabilities[metric],
                f"planned_propagation_p95_pass_probabilities[{metric!r}]",
            )
            p95_units = _required_quantile_units(
                quantile=0.95,
                planned_pass_probability=p95_pass_probability,
                alpha=per_claim_alpha,
                beta=beta_per_claim,
            )
            failure_probability = _probability(
                planned_propagation_cluster_failure_probabilities[metric],
                f"planned_propagation_cluster_failure_probabilities[{metric!r}]",
            )
            if failure_probability >= _PROPAGATION_FAILURE_LIMIT:
                raise ValueError(
                    f"planned cluster-failure probability for {metric} must be below 0.05"
                )
            failure_design = _exact_binomial_failure_design(
                planned_failure_probability=failure_probability,
                per_comparison_alpha=per_claim_alpha,
                per_comparison_power=per_claim_power,
            )
            failure_units = failure_design["required_independent_outer_clusters_per_stratum"]
            supported = (
                max(p95_units, int(failure_units), 2) if isinstance(failure_units, int) else None
            )
            propagation_metric_plans[metric] = {
                "p95_claim": {
                    "acceptance_margin_normalized_error": _PROPAGATION_P95_LIMIT,
                    "planned_cluster_pass_probability": p95_pass_probability,
                    "required_independent_clusters_per_stratum": p95_units,
                },
                "cluster_failure_claim": {
                    "acceptance_probability_limit": _PROPAGATION_FAILURE_LIMIT,
                    "planned_failure_probability": failure_probability,
                    **failure_design,
                },
                "required_clusters_for_both_claims_per_stratum": supported,
            }
            if isinstance(supported, int):
                propagation_supported_counts.append(supported)
        propagation_design = {
            "stratum_count": len(propagation_strata),
            "strata": propagation_strata,
            "claim_count": len(propagation_strata) * len(PROPAGATION_CLAIMS),
            "assumptions_are_worst_stratum_across_all_declared_strata": True,
            "metric_plans": propagation_metric_plans,
            "required_independent_clusters_per_stratum_for_supported_claims": (
                max(propagation_supported_counts)
                if len(propagation_supported_counts) == len(_PROPAGATION_METRICS)
                else None
            ),
            "complete_precision_powered_design": False,
        }

    receiver_design = None
    if family["schema"] in {FAMILY_SCHEMA_V3, FAMILY_SCHEMA_V4}:
        receiver_strata = family["receiver_impact_expected_strata"]
        receiver_design = {
            "stratum_count": len(receiver_strata),
            "strata": receiver_strata,
            "claim_count": len(receiver_strata) * len(RECEIVER_IMPACT_CLAIMS),
            "assurance_design_supplied_separately": True,
            "complete_precision_powered_design": False,
        }

    angular_plan: dict[str, Any]
    if planned_angular_unit_pass_probability is None:
        angular_plan = {
            "assumption_supplied": False,
            "maximum_measurement_units_per_stratum": None,
            "compatible_with_material_design": None,
        }
    else:
        angular_probability = planned_angular_unit_pass_probability
        if (
            isinstance(angular_probability, bool)
            or not isinstance(angular_probability, (int, float))
            or not math.isfinite(float(angular_probability))
            or not 0.0 <= float(angular_probability) <= 1.0
        ):
            raise ValueError("planned_angular_unit_pass_probability must be in [0, 1]")
        angular_maximum = _maximum_angular_units_for_pass_probability(
            planned_unit_pass_probability=float(angular_probability),
            minimum_pass_probability=per_claim_power,
        )
        angular_plan = {
            "assumption_supplied": True,
            "planned_unit_pass_probability": float(angular_probability),
            "minimum_per_stratum_pass_probability_target": per_claim_power,
            "maximum_measurement_units_per_stratum": angular_maximum,
            "compatible_with_material_design": (
                True if angular_maximum is None else angular_maximum >= material_supported
            ),
            "interpretation": (
                "Finite-sample maximum-rule pass probability under an explicit IID assumption; "
                "this is not a population-maximum guarantee."
            ),
        }

    return {
        "schema": "loslab.g3_joint_sample_size_planning.v1",
        "status": "SCENARIO_CONDITIONAL_PLANNING_ONLY_NOT_QUALIFICATION",
        "qualification_status": "NOT_QUALIFIED",
        "admission_possible": False,
        "joint_acceptance_family": {
            "schema": family["schema"],
            "family_id": family["family_id"],
            "family_sha256": family["family_sha256"],
            "familywise_alpha": alpha,
            "familywise_power_target": power,
            "claim_count": family_claim_count,
            "per_claim_alpha_bonferroni": per_claim_alpha,
            "per_claim_power_target_bonferroni": per_claim_power,
            **(
                {"path_equivalence_expected_scenarios": family[
                    "path_equivalence_expected_scenarios"
                ]}
                if family["schema"] == FAMILY_SCHEMA_V4
                else {}
            ),
        },
        "material_design": {
            "stratum_count": len(material_strata),
            "claim_count": len(material_strata) * len(MATERIAL_CLAIMS),
            "required_independent_measurement_units_per_stratum_for_supported_claims": material_supported,
            "quantile_metric_plans": material_quantile_plans,
            "coverage_claim_units_per_stratum": coverage_units,
            "angular_maximum_claim": angular_plan,
            "pilot_floor_units_per_stratum": MIN_INDEPENDENT_MEASUREMENT_UNITS,
            "complete_precision_powered_design": False,
        },
        "complex_csi_design": {
            "stratum_count": len(csi_strata),
            "claim_count": len(csi_strata) * len(COMPLEX_CSI_CLAIMS),
            "p95_claim": {
                "target_quantile": 0.95,
                "acceptance_margin": _COMPLEX_CSI_P95_LIMIT,
                "planned_cluster_pass_probability": csi_p95_probability,
                "required_independent_scene_clusters_per_stratum": csi_p95_units,
            },
            "cluster_failure_claim": csi_failure_design,
            "required_independent_scene_clusters_per_stratum_for_supported_claims": csi_supported,
            "complete_precision_powered_design": False,
        },
        "propagation_design": propagation_design,
        "receiver_impact_design": receiver_design,
        "sampling_units_are_distinct": True,
        "single_combined_sample_size_is_reported": False,
        "uses_measurement_or_pilot_data": False,
        "complete_g3_sample_size": None,
        "limitations": [
            "All pass probabilities and interval-coverage assumptions are unverified scenario inputs.",
            "Material acquisition units and complex-CSI scene clusters are different sampling units.",
            "Propagation p95 and failure designs use explicit shared worst-stratum assumptions; receiver-impact cluster/block-bootstrap assurance is reported separately and remains incomplete unless its evaluator-aligned scenario design passes.",
            "The Hoeffding plans are conservative and do not replace evaluator-aligned assurance or review.",
            "Angular maximum planning is conditional on the explicit IID per-unit pass probability.",
            "External preregistration, physical-independence evidence, calibration, and scene representativeness remain open.",
        ],
    }


def compose_g3_joint_sample_size_design(
    *,
    joint_acceptance_family: dict[str, Any],
    joint_plan: Mapping[str, Any],
    material_assurance_plan: Mapping[str, Any],
    complex_csi_assurance_plan: Mapping[str, Any],
    propagation_assurance_plan: Mapping[str, Any],
    receiver_assurance_plan: Mapping[str, Any],
    target_joint_g3_assurance: float = 0.80,
) -> dict[str, Any]:
    """Join branch-specific G3 planning outputs without merging sampling units.

    The component planners remain authoritative for their own criteria and
    assumptions. This function verifies their family/stratum alignment and
    assembles per-stratum required counts for a future preregistration. It
    never upgrades scenario-conditional planning to qualification or reports
    a single combined G3 sample size.
    """
    family = validate_g3_joint_acceptance_family(joint_acceptance_family)
    target_joint = _probability(
        target_joint_g3_assurance, "target_joint_g3_assurance"
    )
    if family.get("schema") != FAMILY_SCHEMA_V4:
        raise ValueError("joint design composition requires acceptance-family v4")
    expected_family = {
        "family_id": family["family_id"],
        "family_sha256": family["family_sha256"],
    }
    components = {
        "joint": (joint_plan, "loslab.g3_joint_sample_size_planning.v1"),
        "material_assurance": (
            material_assurance_plan,
            "loslab.g3_bootstrap_assurance_planning.v1",
        ),
        "complex_csi_assurance": (
            complex_csi_assurance_plan,
            "loslab.g3_complex_csi_bootstrap_assurance_planning.v1",
        ),
        "propagation_assurance": (
            propagation_assurance_plan,
            "loslab.g3_propagation_bootstrap_assurance_planning.v1",
        ),
        "receiver_assurance": (
            receiver_assurance_plan,
            "loslab.g3_receiver_impact_bootstrap_assurance_planning.v1",
        ),
    }
    component_digests: dict[str, dict[str, str]] = {}
    for name, (plan, expected_schema) in components.items():
        if not isinstance(plan, Mapping) or plan.get("schema") != expected_schema:
            raise ValueError(f"{name} plan has an unsupported schema")
        if plan.get("admission_possible") is not False or plan.get("qualification_status") != (
            "NOT_QUALIFIED"
        ):
            raise ValueError(f"{name} plan must remain explicitly non-qualifying")
        if name != "joint" and plan.get("scenario_provenance_verified") is not False:
            raise ValueError(f"{name} plan must not claim verified scenario provenance")
        family_identity = plan.get("joint_acceptance_family")
        if not isinstance(family_identity, Mapping) or any(
            family_identity.get(key) != value for key, value in expected_family.items()
        ):
            raise ValueError(f"{name} plan does not match the supplied acceptance family")
        component_digests[name] = {
            "schema": expected_schema,
            "sha256": hashlib.sha256(
                json.dumps(
                    dict(plan),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest(),
        }

    strata_by_branch = {
        "material_calibration": family["material_expected_strata"],
        "complex_csi": family["complex_csi_expected_strata"],
        "propagation_metrics": family["propagation_expected_strata"],
        "receiver_impact": family["receiver_impact_expected_strata"],
        "path_equivalence": family["path_equivalence_expected_scenarios"],
    }
    units_by_branch = {
        "material_calibration": "independent_acquisition_unit",
        "complex_csi": "independent_physical_episode",
        "propagation_metrics": "independent_physical_episode",
        "receiver_impact": "independent_transport_block_cluster",
        "path_equivalence": "physical_episode",
    }

    def positive_count(value: Any, label: str) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{label} must be a positive integer or null")
        return value

    def selected_assurance_count(
        plan: Mapping[str, Any],
        expected_strata: list[Any],
        label: str,
        *,
        stratum_key_fields: tuple[str, ...] = ("family_id", "condition"),
    ) -> tuple[int | None, float | None]:
        selected = positive_count(
            plan.get("selected_clusters_per_stratum_among_evaluated_candidates"),
            f"{label}.selected_clusters_per_stratum_among_evaluated_candidates",
        )
        if selected is None:
            return None, None
        target = _probability(plan.get("target_assurance"), f"{label}.target_assurance")
        candidates = plan.get("candidate_designs")
        if not isinstance(candidates, list):
            raise ValueError(f"{label}.candidate_designs must be an array")
        matches = [
            row
            for row in candidates
            if isinstance(row, Mapping)
            and row.get("candidate_clusters_per_stratum") == selected
        ]
        if len(matches) != 1 or matches[0].get("meets_target_assurance") is not True:
            raise ValueError(f"{label} selected count is not a passing candidate design")
        joint_lower = matches[0].get("joint_assurance_lower_bound_union_bound")
        if (
            isinstance(joint_lower, bool)
            or not isinstance(joint_lower, (int, float))
            or not math.isfinite(float(joint_lower))
            or float(joint_lower) < target
            or float(joint_lower) > 1.0
        ):
            raise ValueError(f"{label} selected candidate does not meet joint assurance")
        rows = matches[0].get("stratum_assurance")
        expected_keys = {tuple(row) for row in expected_strata}
        observed: set[tuple[str, ...]] = set()
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"{label} selected candidate has no stratum assurance rows")
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError(f"{label} stratum assurance rows must be objects")
            key = tuple(row.get(field) for field in stratum_key_fields)
            if any(not isinstance(part, str) or not part for part in key) or key in observed:
                raise ValueError(f"{label} has invalid or duplicate assurance strata")
            observed.add(key)
            if row.get("cluster_count") != selected:
                raise ValueError(f"{label} stratum cluster count differs from selected count")
            bound = row.get("assurance_one_sided_exact_lower_bound")
            if (
                isinstance(bound, bool)
                or not isinstance(bound, (int, float))
                or not math.isfinite(float(bound))
                or float(bound) < target
                or float(bound) > 1.0
            ):
                raise ValueError(f"{label} stratum bound does not meet target assurance")
        if observed != expected_keys:
            raise ValueError(f"{label} assurance strata differ from the acceptance family")
        recomputed_joint_lower = max(
            0.0,
            1.0
            - sum(
                1.0 - float(row["assurance_one_sided_exact_lower_bound"])
                for row in rows
            ),
        )
        if (
            isinstance(joint_lower, bool)
            or not isinstance(joint_lower, (int, float))
            or not math.isfinite(float(joint_lower))
            or not 0.0 <= float(joint_lower) <= 1.0
            or not math.isclose(
                float(joint_lower), recomputed_joint_lower, rel_tol=1e-12, abs_tol=1e-12
            )
        ):
            raise ValueError(f"{label} joint assurance bound is inconsistent with its strata")
        return selected, float(joint_lower)

    def selected_material_assurance_count(
        plan: Mapping[str, Any],
    ) -> tuple[int | None, float | None]:
        if plan.get("joint_acceptance_family") != expected_family:
            raise ValueError("material assurance plan does not match the supplied acceptance family")
        roster = plan.get("stratum_roster_by_id")
        if not isinstance(roster, Mapping):
            raise ValueError("material assurance plan must bind material stratum IDs")
        normalized_roster: dict[str, tuple[str, ...]] = {}
        for stratum_id, stratum in roster.items():
            if (
                not isinstance(stratum_id, str)
                or not stratum_id
                or isinstance(stratum, (str, bytes))
                or not isinstance(stratum, list)
                or len(stratum) != 3
                or any(not isinstance(part, str) or not part for part in stratum)
            ):
                raise ValueError("material assurance plan contains an invalid stratum binding")
            normalized_roster[stratum_id] = tuple(stratum)
        expected_material_strata = {
            tuple(row) for row in strata_by_branch["material_calibration"]
        }
        if (
            len(normalized_roster) != len(expected_material_strata)
            or set(normalized_roster.values()) != expected_material_strata
        ):
            raise ValueError("material assurance plan strata differ from the acceptance family")
        selected = positive_count(
            plan.get("selected_units_per_stratum_among_evaluated_candidates"),
            "material_assurance_plan.selected_units_per_stratum_among_evaluated_candidates",
        )
        if selected is None:
            return None, None
        target = _probability(plan.get("target_assurance"), "material_assurance_plan.target_assurance")
        candidates = plan.get("candidate_designs")
        if not isinstance(candidates, list):
            raise ValueError("material_assurance_plan.candidate_designs must be an array")
        matches = [
            row
            for row in candidates
            if isinstance(row, Mapping)
            and row.get("independent_units_per_stratum") == selected
        ]
        if len(matches) != 1 or matches[0].get("meets_target_assurance") is not True:
            raise ValueError("material assurance selection is not a passing candidate")
        rows = matches[0].get("stratum_assurance")
        if not isinstance(rows, Mapping) or set(rows) != set(normalized_roster):
            raise ValueError("material assurance rows do not cover the bound stratum IDs")
        lower_bounds: list[float] = []
        for stratum_id, row in rows.items():
            if not isinstance(row, Mapping):
                raise ValueError("material assurance stratum rows must be objects")
            bound = row.get("joint_assurance_one_sided_exact_lower_bound")
            atom_count = row.get("assumed_support_atom_count")
            if (
                isinstance(bound, bool)
                or not isinstance(bound, (int, float))
                or not math.isfinite(float(bound))
                or float(bound) < target
                or float(bound) > 1.0
                or isinstance(atom_count, bool)
                or not isinstance(atom_count, int)
                or atom_count < 1
            ):
                raise ValueError(f"material assurance bound is invalid for {stratum_id}")
            lower_bounds.append(float(bound))
        joint_lower = matches[0].get("joint_assurance_lower_bound_union_bound")
        recomputed = max(0.0, 1.0 - sum(1.0 - value for value in lower_bounds))
        if (
            isinstance(joint_lower, bool)
            or not isinstance(joint_lower, (int, float))
            or not math.isfinite(float(joint_lower))
            or not 0.0 <= float(joint_lower) <= 1.0
            or not math.isclose(float(joint_lower), recomputed, rel_tol=1e-12, abs_tol=1e-12)
            or float(joint_lower) < target
        ):
            raise ValueError("material assurance joint bound is inconsistent or below target")
        return selected, float(joint_lower)

    material = joint_plan.get("material_design")
    csi = joint_plan.get("complex_csi_design")
    propagation = joint_plan.get("propagation_design")
    if not isinstance(material, Mapping) or not isinstance(csi, Mapping):
        raise ValueError("joint plan is missing material or complex-CSI design")
    if (
        joint_plan.get("single_combined_sample_size_is_reported") is not False
        or joint_plan.get("complete_g3_sample_size") is not None
    ):
        raise ValueError("joint plan must keep sampling units separate and combined size unresolved")
    material_base_units = positive_count(
        material.get("required_independent_measurement_units_per_stratum_for_supported_claims"),
        "joint_plan.material_design.required_independent_measurement_units_per_stratum_for_supported_claims",
    )
    material_assurance_units, material_assurance_lower = selected_material_assurance_count(
        material_assurance_plan
    )
    material_units = (
        max(material_base_units, material_assurance_units)
        if material_base_units is not None and material_assurance_units is not None
        else None
    )
    csi_base_units = positive_count(
        csi.get("required_independent_scene_clusters_per_stratum_for_supported_claims"),
        "joint_plan.complex_csi_design.required_independent_scene_clusters_per_stratum_for_supported_claims",
    )
    csi_assurance_units, csi_assurance_lower = selected_assurance_count(
        complex_csi_assurance_plan,
        strata_by_branch["complex_csi"],
        "complex_csi_assurance_plan",
    )
    csi_units = (
        max(csi_base_units, csi_assurance_units)
        if csi_base_units is not None and csi_assurance_units is not None
        else None
    )
    propagation_base_units: int | None = None
    if propagation is not None:
        if not isinstance(propagation, Mapping):
            raise ValueError("joint_plan.propagation_design must be an object or null")
        propagation_base_units = positive_count(
            propagation.get("required_independent_clusters_per_stratum_for_supported_claims"),
            "joint_plan.propagation_design.required_independent_clusters_per_stratum_for_supported_claims",
        )
    propagation_assurance_units, propagation_assurance_lower = selected_assurance_count(
        propagation_assurance_plan,
        strata_by_branch["propagation_metrics"],
        "propagation_assurance_plan",
        stratum_key_fields=("family_id", "condition", "component"),
    )
    propagation_units = (
        max(propagation_base_units, propagation_assurance_units)
        if propagation_base_units is not None and propagation_assurance_units is not None
        else None
    )
    receiver_component = joint_plan.get("receiver_impact_design")
    if not isinstance(propagation, Mapping) or not isinstance(receiver_component, Mapping):
        raise ValueError("family v4 joint plan must include propagation and receiver designs")
    for branch, component in (
        ("material_calibration", material),
        ("complex_csi", csi),
        ("propagation_metrics", propagation),
        ("receiver_impact", receiver_component),
    ):
        declared_strata = component.get("stratum_count")
        if (
            isinstance(declared_strata, bool)
            or not isinstance(declared_strata, int)
            or declared_strata != len(strata_by_branch[branch])
        ):
            raise ValueError(f"joint plan {branch} stratum count differs from the family")
        declared_roster = component.get("strata")
        if declared_roster is not None:
            if not isinstance(declared_roster, (list, tuple)) or any(
                not isinstance(row, (list, tuple)) for row in declared_roster
            ):
                raise ValueError(f"joint plan {branch} stratum roster is invalid")
            if {tuple(row) for row in declared_roster} != {
                tuple(row) for row in strata_by_branch[branch]
            }:
                raise ValueError(f"joint plan {branch} stratum roster differs from the family")
    receiver_units, receiver_assurance_lower = selected_assurance_count(
        receiver_assurance_plan,
        strata_by_branch["receiver_impact"],
        "receiver_assurance_plan",
    )

    branch_counts = {
        "material_calibration": material_units,
        "complex_csi": csi_units,
        "propagation_metrics": propagation_units,
        "receiver_impact": receiver_units,
        "path_equivalence": 1,
    }
    branch_designs: dict[str, Any] = {}
    for branch, roster in strata_by_branch.items():
        count = branch_counts[branch]
        branch_designs[branch] = {
            "sampling_unit": units_by_branch[branch],
            "required_units_by_stratum": [
                {"stratum": list(stratum), "required_units": count} for stratum in roster
            ],
            "count_available": count is not None,
            "component_status": (
                "EXACT_EPISODE_CENSUS" if branch == "path_equivalence" else
                "SCENARIO_CONDITIONAL_COUNT_AVAILABLE" if count is not None else
                "REQUIRED_ASSURANCE_COUNT_UNRESOLVED"
            ),
        }

    all_counts_available = all(
        design["count_available"] for design in branch_designs.values()
    )
    component_design_complete = all(
        count is not None and lower is not None
        for count, lower in (
            (material_assurance_units, material_assurance_lower),
            (csi_assurance_units, csi_assurance_lower),
            (propagation_assurance_units, propagation_assurance_lower),
            (receiver_units, receiver_assurance_lower),
        )
    )
    assurance_bounds = {
        "material_calibration": material_assurance_lower,
        "complex_csi": csi_assurance_lower,
        "propagation_metrics": propagation_assurance_lower,
        "receiver_impact": receiver_assurance_lower,
    }
    joint_assurance_lower = (
        max(0.0, 1.0 - sum(1.0 - bound for bound in assurance_bounds.values()))
        if all(bound is not None for bound in assurance_bounds.values())
        else None
    )
    blockers = [
        "planner assumptions are scenario inputs without verified physical/population provenance"
    ]
    if not component_design_complete:
        blockers.append("one or more evaluator-aligned assurance plans have no selected candidate")
    if not all_counts_available:
        blockers.append("one or more branch-specific required counts remain unresolved")
    if joint_assurance_lower is None:
        blockers.append("one or more evaluator-aligned branch assurance bounds remain unresolved")
    elif joint_assurance_lower < target_joint:
        blockers.append(
            "cross-branch union-bound assurance is below the requested joint G3 target"
        )
    angular = material.get("angular_maximum_claim")
    angular_supplied = isinstance(angular, Mapping) and angular.get("assumption_supplied") is True
    angular_compatible = (
        angular.get("compatible_with_material_design") if isinstance(angular, Mapping) else None
    )
    angular_maximum = (
        angular.get("maximum_measurement_units_per_stratum")
        if isinstance(angular, Mapping)
        else None
    )
    if (
        angular_supplied
        and isinstance(angular_maximum, int)
        and material_units is not None
        and material_units > angular_maximum
    ):
        angular_compatible = False
    if not angular_supplied:
        blockers.append("material angular-maximum pass-probability assumption is missing")
    elif angular_compatible is False:
        blockers.append("material angular-maximum assumption conflicts with the planned unit count")
    elif angular_compatible is not True:
        blockers.append("material angular-maximum compatibility remains unresolved")
    complete_powered_design = bool(
        all_counts_available
        and component_design_complete
        and joint_assurance_lower is not None
        and joint_assurance_lower >= target_joint
        and angular_supplied
        and angular_compatible is True
    )

    result: dict[str, Any] = {
        "schema": "loslab.g3_joint_sample_size_design_composition.v1",
        "status": "SCENARIO_CONDITIONAL_DESIGN_NOT_QUALIFICATION",
        "qualification_status": "NOT_QUALIFIED",
        "admission_possible": False,
        "joint_acceptance_family": expected_family,
        "component_plans": component_digests,
        "sampling_units_are_distinct": True,
        "single_combined_sample_size_is_reported": False,
        "branch_designs": branch_designs,
        "all_branch_counts_available": all_counts_available,
        "component_designs_claim_complete_powered_design": component_design_complete,
        "branch_assurance_lower_bounds": assurance_bounds,
        "joint_g3_assurance_lower_bound_union_bound": joint_assurance_lower,
        "target_joint_g3_assurance": target_joint,
        "material_angular_maximum_assumption": {
            "supplied": angular_supplied,
            "compatible_with_material_design": angular_compatible,
            "maximum_units_per_stratum": angular_maximum,
        },
        "complete_precision_powered_g3_design": complete_powered_design,
        "complete_g3_sample_size": None,
        "blockers": blockers,
        "limitations": [
            "Counts are conditional on the exact assumptions and family-bound component plans.",
            "The cross-branch assurance bound covers material, complex-CSI, propagation, and receiver branches; path equivalence is a frozen episode census and carries no population-generalization claim.",
            "Different branches use independent acquisition units, physical episodes, and receiver clusters; their counts must not be summed.",
            "This composition artifact does not verify scenario assumptions, physical independence, measurement provenance, or external review.",
        ],
    }
    result["design_sha256"] = hashlib.sha256(
        json.dumps(
            result,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return result


def verify_g3_joint_sample_size_design(design: Mapping[str, Any]) -> dict[str, Any]:
    """Verify composition-artifact integrity while preserving non-admission status."""
    if not isinstance(design, Mapping):
        raise TypeError("G3 joint sample-size design must be an object")
    saved = dict(design)
    declared_digest = saved.pop("design_sha256", None)
    if (
        not isinstance(declared_digest, str)
        or len(declared_digest) != 64
        or any(character not in "0123456789abcdef" for character in declared_digest)
    ):
        raise ValueError("design_sha256 must be a lowercase SHA-256 digest")
    actual_digest = hashlib.sha256(
        json.dumps(
            saved,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if actual_digest != declared_digest:
        raise ValueError("G3 joint sample-size design digest does not match its contents")
    expected_fields = {
        "schema",
        "status",
        "qualification_status",
        "admission_possible",
        "joint_acceptance_family",
        "component_plans",
        "sampling_units_are_distinct",
        "single_combined_sample_size_is_reported",
        "branch_designs",
        "all_branch_counts_available",
        "component_designs_claim_complete_powered_design",
        "branch_assurance_lower_bounds",
        "joint_g3_assurance_lower_bound_union_bound",
        "target_joint_g3_assurance",
        "material_angular_maximum_assumption",
        "complete_precision_powered_g3_design",
        "complete_g3_sample_size",
        "blockers",
        "limitations",
    }
    if set(saved) != expected_fields:
        raise ValueError("G3 joint sample-size design fields differ from its v1 schema")
    if (
        saved.get("schema") != "loslab.g3_joint_sample_size_design_composition.v1"
        or saved.get("status") != "SCENARIO_CONDITIONAL_DESIGN_NOT_QUALIFICATION"
        or saved.get("qualification_status") != "NOT_QUALIFIED"
        or saved.get("admission_possible") is not False
        or saved.get("sampling_units_are_distinct") is not True
        or saved.get("single_combined_sample_size_is_reported") is not False
        or not isinstance(saved.get("component_designs_claim_complete_powered_design"), bool)
        or not isinstance(saved.get("complete_precision_powered_g3_design"), bool)
        or saved.get("complete_g3_sample_size") is not None
    ):
        raise ValueError("G3 joint sample-size design violates its non-qualification contract")
    angular_assumption = saved.get("material_angular_maximum_assumption")
    if not isinstance(angular_assumption, Mapping) or set(angular_assumption) != {
        "supplied",
        "compatible_with_material_design",
        "maximum_units_per_stratum",
    }:
        raise ValueError("material angular-maximum design declaration is invalid")
    if not isinstance(angular_assumption.get("supplied"), bool):
        raise ValueError("material angular-maximum supplied flag must be Boolean")
    compatible = angular_assumption.get("compatible_with_material_design")
    if compatible is not None and not isinstance(compatible, bool):
        raise ValueError("material angular-maximum compatibility must be Boolean or null")
    maximum = angular_assumption.get("maximum_units_per_stratum")
    if maximum is not None and (
        isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1
    ):
        raise ValueError("material angular-maximum count must be a positive integer or null")
    if not angular_assumption["supplied"] and (
        compatible is not None or maximum is not None
    ):
        raise ValueError("unsupplied angular-maximum assumptions cannot declare derived values")
    component_plans = saved.get("component_plans")
    expected_component_schemas = {
        "joint": "loslab.g3_joint_sample_size_planning.v1",
        "material_assurance": "loslab.g3_bootstrap_assurance_planning.v1",
        "complex_csi_assurance": "loslab.g3_complex_csi_bootstrap_assurance_planning.v1",
        "propagation_assurance": "loslab.g3_propagation_bootstrap_assurance_planning.v1",
        "receiver_assurance": "loslab.g3_receiver_impact_bootstrap_assurance_planning.v1",
    }
    if not isinstance(component_plans, Mapping) or set(component_plans) != set(
        expected_component_schemas
    ):
        raise ValueError("component_plans does not cover the exact required planning outputs")
    for name, expected_schema in expected_component_schemas.items():
        component = component_plans[name]
        digest = component.get("sha256") if isinstance(component, Mapping) else None
        if (
            not isinstance(component, Mapping)
            or set(component) != {"schema", "sha256"}
            or component.get("schema") != expected_schema
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"component_plans.{name} is invalid")
    branches = saved.get("branch_designs")
    expected_branches = {
        "material_calibration",
        "complex_csi",
        "propagation_metrics",
        "receiver_impact",
        "path_equivalence",
    }
    if not isinstance(branches, Mapping) or set(branches) != expected_branches:
        raise ValueError("branch_designs does not cover the exact G3 evaluator branches")
    count_availability: list[bool] = []
    for branch, branch_design in branches.items():
        if not isinstance(branch_design, Mapping):
            raise ValueError(f"branch_designs.{branch} must be an object")
        if set(branch_design) != {
            "sampling_unit",
            "required_units_by_stratum",
            "count_available",
            "component_status",
        }:
            raise ValueError(f"branch_designs.{branch} fields are invalid")
        roster = branch_design.get("required_units_by_stratum")
        if not isinstance(roster, list) or not roster:
            raise ValueError(f"branch_designs.{branch} must contain a nonempty stratum roster")
        keys: set[tuple[str, ...]] = set()
        counts: list[int | None] = []
        for row in roster:
            if not isinstance(row, Mapping) or set(row) != {"stratum", "required_units"}:
                raise ValueError(f"branch_designs.{branch} has an invalid stratum row")
            key = row["stratum"]
            if (
                not isinstance(key, list)
                or not key
                or any(not isinstance(part, str) or not part for part in key)
                or tuple(key) in keys
            ):
                raise ValueError(f"branch_designs.{branch} has an invalid or duplicate stratum")
            keys.add(tuple(key))
            count = row["required_units"]
            if count is not None and (
                isinstance(count, bool) or not isinstance(count, int) or count < 1
            ):
                raise ValueError(f"branch_designs.{branch} required_units is invalid")
            if branch == "path_equivalence" and count != 1:
                raise ValueError("path_equivalence must remain a one-per-episode census")
            counts.append(count)
        count_available = all(count is not None for count in counts)
        if branch_design.get("count_available") is not count_available:
            raise ValueError(f"branch_designs.{branch} count_available is inconsistent")
        count_availability.append(count_available)
    if saved.get("all_branch_counts_available") is not all(count_availability):
        raise ValueError("all_branch_counts_available is inconsistent with branch rosters")
    assurance_bounds = saved.get("branch_assurance_lower_bounds")
    assurance_branches = {
        "material_calibration",
        "complex_csi",
        "propagation_metrics",
        "receiver_impact",
    }
    if not isinstance(assurance_bounds, Mapping) or set(assurance_bounds) != assurance_branches:
        raise ValueError("branch_assurance_lower_bounds does not cover the evaluator branches")
    normalized_bounds: list[float] = []
    for branch, bound in assurance_bounds.items():
        if bound is None:
            continue
        if (
            isinstance(bound, bool)
            or not isinstance(bound, (int, float))
            or not math.isfinite(float(bound))
            or not 0.0 <= float(bound) <= 1.0
        ):
            raise ValueError(f"branch assurance lower bound is invalid for {branch}")
        normalized_bounds.append(float(bound))
    declared_joint_assurance = saved.get("joint_g3_assurance_lower_bound_union_bound")
    if len(normalized_bounds) == len(assurance_branches):
        expected_joint_assurance = max(
            0.0, 1.0 - sum(1.0 - bound for bound in normalized_bounds)
        )
        if (
            isinstance(declared_joint_assurance, bool)
            or not isinstance(declared_joint_assurance, (int, float))
            or not math.isclose(
                float(declared_joint_assurance),
                expected_joint_assurance,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("joint G3 assurance bound is inconsistent with branch bounds")
    elif declared_joint_assurance is not None:
        raise ValueError("joint G3 assurance bound must be null when a branch bound is unresolved")
    component_design_complete = all(
        branches[branch]["count_available"] and assurance_bounds[branch] is not None
        for branch in assurance_branches
    )
    if saved["component_designs_claim_complete_powered_design"] is not component_design_complete:
        raise ValueError(
            "component powered-design claim is inconsistent with branch counts and assurance bounds"
        )
    target_joint_assurance = _probability(
        saved.get("target_joint_g3_assurance"), "target_joint_g3_assurance"
    )
    if angular_assumption.get("compatible_with_material_design") is True:
        maximum_units = angular_assumption.get("maximum_units_per_stratum")
        material_rows = branches["material_calibration"]["required_units_by_stratum"]
        if maximum_units is not None and any(
            row["required_units"] is not None
            and row["required_units"] > maximum_units
            for row in material_rows
        ):
            raise ValueError("material angular-maximum assumption conflicts with required counts")
    expected_complete = bool(
        saved["all_branch_counts_available"]
        and component_design_complete
        and declared_joint_assurance is not None
        and float(declared_joint_assurance) >= target_joint_assurance
        and angular_assumption.get("supplied") is True
        and angular_assumption.get("compatible_with_material_design") is True
    )
    if saved["complete_precision_powered_g3_design"] is not expected_complete:
        raise ValueError("complete_precision_powered_g3_design is inconsistent with component plans")
    return {
        "schema": "loslab.g3_joint_sample_size_design_verification.v1",
        "status": "PASS_DESIGN_INTEGRITY_NOT_QUALIFIED",
        "qualification_status": "NOT_QUALIFIED",
        "admission_possible": False,
        "design_sha256": declared_digest,
        "complete_precision_powered_g3_design": expected_complete,
    }


def plan_g3_complex_csi_bootstrap_assurance(
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
    family = validate_g3_joint_acceptance_family(joint_acceptance_family)
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
        "schema": "loslab.g3_complex_csi_bootstrap_assurance_planning.v1",
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
        "complete_g3_sample_size": None,
        "sampling_units_are_scene_clusters": True,
        "limitations": [
            "All cluster-summary distributions are hypothetical and unverified; no measurements or pilot data are read.",
            "The assurance estimate is conditional on the supplied finite-support IID scenario and exact evaluator rules.",
            "Scene-cluster counts remain separate from material acquisition-unit counts; no combined G3 sample size is inferred.",
            "This planner never qualifies G3, authorizes RT production, or verifies physical independence or scene representativeness.",
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
    """Apply the G3.03 equal-unit point and percentile-bootstrap margins."""
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


def _validate_g3_assumption_scenarios(
    scenarios: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], str]:
    if not isinstance(scenarios, Mapping) or not scenarios:
        raise ValueError("assumed_iid_scenarios_by_stratum must be a nonempty mapping")
    if any(not isinstance(stratum, str) or not stratum.strip() for stratum in scenarios):
        raise ValueError("scenario stratum names must be nonempty strings")
    validated: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    canonical: dict[str, Any] = {}
    for stratum in sorted(scenarios):
        if not isinstance(stratum, str) or not stratum.strip():
            raise ValueError("scenario stratum names must be nonempty strings")
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
                    f"scenario {stratum!r} atom {atom_index} must specify exactly the five G3 metrics"
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


def plan_g3_bootstrap_assurance(
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
    """Estimate G3.03 bootstrap pass assurance under explicit discrete IID scenarios.

    Each named stratum receives a caller-specified finite-support distribution of
    *hypothetical independent-unit summaries*. One Monte Carlo study draws the
    requested number of units from that distribution, then runs the same
    equal-unit point statistics and Bonferroni percentile bootstrap checks as
    the G3.03 evaluator. A one-sided Clopper-Pearson lower bound is computed for
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
    validated, scenario_digest = _validate_g3_assumption_scenarios(assumed_iid_scenarios_by_stratum)
    family_identity: dict[str, str] | None = None
    normalized_roster: dict[str, list[str]] | None = None
    if (joint_acceptance_family is None) != (stratum_roster_by_id is None):
        raise ValueError(
            "joint_acceptance_family and stratum_roster_by_id must be supplied together"
        )
    if joint_acceptance_family is not None and stratum_roster_by_id is not None:
        family = validate_g3_joint_acceptance_family(dict(joint_acceptance_family))
        expected = {tuple(row) for row in family["material_expected_strata"]}
        if set(stratum_roster_by_id) != set(validated):
            raise ValueError("material stratum roster IDs must match the scenario strata")
        normalized_roster = {}
        observed: set[tuple[str, ...]] = set()
        for stratum_id, stratum in stratum_roster_by_id.items():
            if not isinstance(stratum_id, str) or not stratum_id.strip():
                raise ValueError("material stratum roster IDs must be nonempty strings")
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
        "schema": "loslab.g3_bootstrap_assurance_planning.v1",
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
        "complete_g3_03_sample_size": None,
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
