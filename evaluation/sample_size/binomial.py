"""Binomial and bounded-mean design helpers for sample-size planning."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.stats import binom

from evaluation.sample_size.constants import (
    _COMPLEX_CSI_FAILURE_LIMIT,
    _EXACT_BINOMIAL_CHUNK_SIZE,
    _EXACT_BINOMIAL_SEARCH_LIMIT,
)


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
