"""Conservative, assumption-driven planning for G3.03 acquisition-unit counts.

This module is deliberately separate from the G3.03 evaluator. It uses no
measurement data and can never qualify material performance. It computes
distribution-free Hoeffding planning bounds for three per-unit quantile
criteria and the bounded per-unit interval-coverage mean, plus an exact
binomial design for the frozen complex-CSI cluster-failure criterion. The
maximum-over-units angular-KS criterion is handled only under an explicit IID
per-unit pass-probability assumption. Its literal finite-sample maximum gives
an upper bound on n for a desired pass probability, not a population-maximum
guarantee. A separate nested simulation estimates evaluator-aligned bootstrap
assurance only under explicit hypothetical finite-support distributions. These
utilities never return a complete G3.03 sample size.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.stats import beta, binom

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
    summaries. A Bonferroni allocation covers all five G3.03 metric claims in
    every declared stratum; familywise power is conservatively Bonferroni
    allocated in the same way. The complex-CSI cluster-failure claim is
    separately Bonferroni-allocated over the requested strata; this does not
    claim joint familywise control across that claim and the five existing
    metric claims.

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
    measurement_unit_lower_bound = max(supported_lower_bound, MIN_INDEPENDENT_MEASUREMENT_UNITS)

    angular_plan: dict[str, Any]
    if planned_angular_unit_pass_probability is None:
        angular_plan = {
            "assumption_supplied": False,
            "planned_unit_pass_probability": None,
            "minimum_per_stratum_pass_probability_target": None,
            "maximum_independent_units_per_stratum": None,
            "maximum_units_status": "UNRESOLVED_MISSING_EXPLICIT_PASS_PROBABILITY",
            "compatible_with_supported_metrics_lower_bound": None,
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
            angular_maximum_units is None or angular_maximum_units >= measurement_unit_lower_bound
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
            "compatible_with_supported_metrics_lower_bound": angular_compatible,
            "minimum_units_for_bootstrap_eligible_joint_design": measurement_unit_lower_bound,
            "compatible_with_bootstrap_eligible_joint_design": angular_compatible,
            "pass_probability_at_maximum_units": (
                None
                if angular_maximum_units is None
                or angular_maximum_units < MIN_INDEPENDENT_MEASUREMENT_UNITS
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
            "INFEASIBLE_UNDER_SUPPLIED_ANGULAR_ASSUMPTION"
            if not angular_compatible
            else "PARTIAL_ONLY_BOOTSTRAP_PRECISION_COMPLEX_CSI_AND_EXTERNAL_GATES_REMAIN"
        )

    return {
        "schema": "loslab.g3_sample_size_planning.v2",
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
        "angular_maximum_metric_plan": angular_plan,
        "hypothetical_required_units_per_stratum_for_supported_metrics": supported_lower_bound,
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
            "multiplicity": "Bonferroni across five metric claims per expected stratum",
        },
        "assumptions_must_be_preregistered_and_not_estimated_from_pilot": True,
    }


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
    bootstrap_replicates: int = 2_000,
    assurance_replicates: int = 500,
    target_assurance: float = 0.80,
    familywise_alpha: float = 0.05,
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
    replaces external acquisition-unit, calibration, or legal review.
    """
    validated, scenario_digest = _validate_g3_assumption_scenarios(assumed_iid_scenarios_by_stratum)
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

    per_comparison_alpha = alpha / (len(validated) * len(_BOOTSTRAP_METRICS))
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
        "admission_possible": False,
        "directly_reads_measurement_files": False,
        "scenario_distribution_sha256": scenario_digest,
        "assumption_model": (
            "Within each stratum, independent acquisition-unit summaries are IID draws "
            "from the supplied finite-support distribution. Cross-stratum dependence is "
            "unrestricted; a union bound combines stratum pass probabilities."
        ),
        "scenario_provenance_verified": False,
        "familywise_alpha": alpha,
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
