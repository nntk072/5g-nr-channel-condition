"""Numeric limits and claim-count constants for sample-size planning."""

from __future__ import annotations

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


_ALL_CLAIMS_PER_STRATUM = _COMPARISONS_PER_STRATUM + 1


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
