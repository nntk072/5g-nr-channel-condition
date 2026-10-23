"""Conservative, assumption-driven planning for G3.03 acquisition-unit counts.

This module is deliberately separate from the G3.03 evaluator. It uses no
measurement data and can never qualify material performance. It computes
distribution-free Hoeffding planning bounds for three per-unit quantile
criteria and the bounded per-unit interval-coverage mean, plus an exact
binomial design for the frozen complex-CSI cluster-failure criterion. The evaluator's
maximum-over-units angular-KS criterion is not covered: boundedness in [0, 1]
alone is insufficient for a population-maximum guarantee; a preregistered tail
model or a changed tolerated-exceedance estimand is needed. Consequently this
utility never returns a complete G3.03 sample size.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.stats import binom

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
_COMPLEX_CSI_FAILURE_LIMIT = 0.05
_EXACT_BINOMIAL_SEARCH_LIMIT = 10_000_000
_EXACT_BINOMIAL_CHUNK_SIZE = 16_384


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


def _exact_binomial_critical_failures(
    cluster_counts: np.ndarray,
    *,
    per_stratum_alpha: float,
) -> np.ndarray:
    """Return the largest accepted failure count under a one-sided exact test.

    A failure count ``x`` is accepted exactly when its one-sided
    Clopper-Pearson upper bound is at most 0.05, equivalently when the
    null-tail probability ``P(Binomial(n, 0.05) <= x)`` is no greater than
    the allocated alpha. SciPy's binomial CDF/quantile use stable special
    functions; the CDF checks correct the quantile boundary explicitly.
    """
    quantile = np.asarray(binom.ppf(per_stratum_alpha, cluster_counts, _COMPLEX_CSI_FAILURE_LIMIT))
    critical = quantile.astype(np.int64) - 1

    valid = critical >= 0
    if np.any(valid):
        upper_tail = np.asarray(
            binom.cdf(critical[valid], cluster_counts[valid], _COMPLEX_CSI_FAILURE_LIMIT)
        )
        critical[valid] -= (upper_tail > per_stratum_alpha).astype(np.int64)

    next_count = critical + 1
    valid = next_count <= cluster_counts
    if np.any(valid):
        next_tail = np.asarray(
            binom.cdf(next_count[valid], cluster_counts[valid], _COMPLEX_CSI_FAILURE_LIMIT)
        )
        critical[valid] += (next_tail <= per_stratum_alpha).astype(np.int64)
    return critical


def _exact_binomial_failure_design(
    *,
    planned_failure_probability: float,
    per_stratum_alpha: float,
    per_stratum_power: float,
) -> dict[str, int | float | bool | str | None]:
    """Find the first exact-binomial design meeting the allocated power."""
    start = 1
    while start <= _EXACT_BINOMIAL_SEARCH_LIMIT:
        stop = min(start + _EXACT_BINOMIAL_CHUNK_SIZE, _EXACT_BINOMIAL_SEARCH_LIMIT + 1)
        counts = np.arange(start, stop, dtype=np.int64)
        critical = _exact_binomial_critical_failures(
            counts,
            per_stratum_alpha=per_stratum_alpha,
        )
        powers = np.zeros(counts.shape, dtype=np.float64)
        admissible = critical >= 0
        if np.any(admissible):
            powers[admissible] = binom.cdf(
                critical[admissible], counts[admissible], planned_failure_probability
            )
        passing = np.flatnonzero(powers >= per_stratum_power)
        if passing.size:
            index = int(passing[0])
            return {
                "sample_size_found": True,
                "required_independent_outer_clusters_per_stratum": int(counts[index]),
                "maximum_admissible_observed_failures_per_stratum": int(critical[index]),
                "achieved_per_stratum_power": float(powers[index]),
                "search_limit_per_stratum": _EXACT_BINOMIAL_SEARCH_LIMIT,
                "search_status": "EXACT_BINOMIAL_DESIGN_FOUND",
            }
        start = stop

    return {
        "sample_size_found": False,
        "required_independent_outer_clusters_per_stratum": None,
        "maximum_admissible_observed_failures_per_stratum": None,
        "achieved_per_stratum_power": None,
        "search_limit_per_stratum": _EXACT_BINOMIAL_SEARCH_LIMIT,
        "search_status": "SEARCH_LIMIT_REACHED_NO_DESIGN_RETURNED",
    }


def plan_g3_independent_acquisition_units(
    *,
    stratum_count: int,
    planned_quantile_pass_probabilities: Mapping[str, float],
    planned_interval_coverage: float,
    planned_complex_csi_failure_probability: float,
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
    summaries. A Bonferroni allocation covers all five G3.03 metric claims in
    every declared stratum; familywise power is conservatively Bonferroni
    allocated in the same way. The complex-CSI cluster-failure claim is
    separately Bonferroni-allocated over the requested strata; this does not
    claim joint familywise control across that claim and the five existing
    metric claims.

    Interpreting these inputs as population quantities requires a representative
    sampling frame or an explicit common-distribution assumption across the
    independent physical acquisition units. The returned supported lower bound
    is not sufficient for G3.03 as a whole:
    maximum_angular_cdf_ks has no distribution-free finite-n plan under the
    current maximum-over-units estimand. This function does not inspect data,
    modify evaluator behavior, or issue any qualification/admission result.
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

    comparison_count = stratum_count * _COMPARISONS_PER_STRATUM
    alpha_per_comparison = alpha / comparison_count
    beta_per_comparison = beta / comparison_count

    failure_alpha_per_stratum = alpha / stratum_count
    failure_beta_per_stratum = beta / stratum_count
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
        per_stratum_alpha=failure_alpha_per_stratum,
        per_stratum_power=1.0 - failure_beta_per_stratum,
    )
    complex_csi_failure_plan.update(
        {
            "criterion": (
                "For each independent outer cluster, failure is any episode whose e_H exceeds 0.05; "
                "the stratum failure probability's one-sided exact upper bound must be <= 0.05."
            ),
            "failure_probability_limit": _COMPLEX_CSI_FAILURE_LIMIT,
            "one_sided_confidence_level_per_stratum": 1.0 - failure_alpha_per_stratum,
            "familywise_alpha": alpha,
            "familywise_power_target": power,
            "stratum_count": stratum_count,
            "per_stratum_alpha_bonferroni": failure_alpha_per_stratum,
            "per_stratum_power_target_bonferroni": 1.0 - failure_beta_per_stratum,
            "planned_failure_probability": planned_failure_probability,
            "sampling_assumption": (
                "Outer clusters are independent and representative of the target population, "
                "or follow a preregistered common distribution within each stratum."
            ),
            "failure_count_rule": (
                "Accept only observed failure counts whose one-sided Clopper-Pearson upper "
                "bound is <= 0.05 at the Bonferroni-adjusted per-stratum confidence level."
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

    return {
        "schema": "loslab.g3_sample_size_planning.v1",
        "status": "PLANNING_ONLY_NOT_QUALIFICATION",
        "uses_measurement_or_pilot_data": False,
        "admission_possible": False,
        "stratum_count": stratum_count,
        "familywise_alpha": alpha,
        "familywise_power_target": power,
        "complex_csi_cluster_failure_plan": complex_csi_failure_plan,
        "comparisons_per_stratum": _COMPARISONS_PER_STRATUM,
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
        "unsupported_estimands": {
            "maximum_angular_cdf_ks": (
                "The evaluator uses a maximum over sampled independent-unit summaries. "
                "Boundedness of the statistic alone does not yield a finite-n "
                "population-maximum guarantee; a preregistered tail model or a "
                "tolerated-exceedance estimand is needed."
            )
        },
        "hypothetical_required_units_per_stratum_for_supported_metrics": supported_lower_bound,
        "pilot_floor_units_per_stratum": MIN_INDEPENDENT_MEASUREMENT_UNITS,
        "pilot_floor_is_not_qualification_sample_size": True,
        "complete_g3_03_sample_size": None,
        "complete_design_status": "UNRESOLVED_UNSUPPORTED_MAXIMUM_ANGULAR_ESTIMAND",
        "evaluation_alignment": (
            "These Hoeffding bounds are conservative design calculations, not a sample-size "
            "guarantee for the evaluator's Bonferroni percentile cluster-bootstrap intervals."
        ),
        "methods": {
            "quantiles": "one-sided Hoeffding lower bound and power on independent-unit pass indicators",
            "coverage": "two-sided Hoeffding confidence interval with conservative power allocation",
            "multiplicity": "Bonferroni across five metric claims per expected stratum",
        },
        "assumptions_must_be_preregistered_and_not_estimated_from_pilot": True,
    }
