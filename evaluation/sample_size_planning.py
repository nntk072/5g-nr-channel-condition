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

from evaluation.sample_size.binomial import (
    _exact_binomial_critical_failures,
    _exact_binomial_failure_design,
    _maximum_angular_units_for_pass_probability,
    _probability,
    _required_bounded_mean_units,
    _required_quantile_units,
)
from evaluation.sample_size.bootstrap import (
    _bootstrap_acceptance_checks,
    _validate_assumption_scenarios,
    _validate_complex_csi_assurance_scenarios,
    plan_bootstrap_assurance,
    plan_complex_csi_bootstrap_assurance,
)
from evaluation.sample_size.constants import (
    _ALL_CLAIMS_PER_STRATUM,
    _BOOTSTRAP_METRICS,
    _COMPARISONS_PER_STRATUM,
    _COMPLEX_CSI_FAILURE_LIMIT,
    _COMPLEX_CSI_P95_LIMIT,
    _EXACT_BINOMIAL_CHUNK_SIZE,
    _EXACT_BINOMIAL_SEARCH_LIMIT,
    _MAX_BOOTSTRAP_PEAK_CELLS,
    _MAX_BOOTSTRAP_RESAMPLING_CELLS,
    _PROPAGATION_FAILURE_LIMIT,
    _PROPAGATION_METRICS,
    _PROPAGATION_P95_LIMIT,
    _QUANTILE_TARGETS,
    MAX_INTERVAL_COVERAGE,
    MAX_MEDIAN_INTERVAL_WIDTH_DB,
    MAX_POWER_ERROR_MEDIAN_DB,
    MAX_POWER_ERROR_P95_DB,
    MIN_INDEPENDENT_MEASUREMENT_UNITS,
    MIN_INTERVAL_COVERAGE,
)
from evaluation.sample_size.independent_units import (
    plan_independent_acquisition_units,
    plan_joint_independent_units,
)
from evaluation.sample_size.joint_design import (
    compose_joint_sample_size_design,
    verify_joint_sample_size_design,
)

__all__ = [
    "MAX_INTERVAL_COVERAGE",
    "MAX_MEDIAN_INTERVAL_WIDTH_DB",
    "MAX_POWER_ERROR_MEDIAN_DB",
    "MAX_POWER_ERROR_P95_DB",
    "MIN_INDEPENDENT_MEASUREMENT_UNITS",
    "MIN_INTERVAL_COVERAGE",
    "_ALL_CLAIMS_PER_STRATUM",
    "_BOOTSTRAP_METRICS",
    "_COMPARISONS_PER_STRATUM",
    "_COMPLEX_CSI_FAILURE_LIMIT",
    "_COMPLEX_CSI_P95_LIMIT",
    "_EXACT_BINOMIAL_CHUNK_SIZE",
    "_EXACT_BINOMIAL_SEARCH_LIMIT",
    "_MAX_BOOTSTRAP_PEAK_CELLS",
    "_MAX_BOOTSTRAP_RESAMPLING_CELLS",
    "_PROPAGATION_FAILURE_LIMIT",
    "_PROPAGATION_METRICS",
    "_PROPAGATION_P95_LIMIT",
    "_QUANTILE_TARGETS",
    "_bootstrap_acceptance_checks",
    "_exact_binomial_critical_failures",
    "_exact_binomial_failure_design",
    "_maximum_angular_units_for_pass_probability",
    "_probability",
    "_required_bounded_mean_units",
    "_required_quantile_units",
    "_validate_assumption_scenarios",
    "_validate_complex_csi_assurance_scenarios",
    "compose_joint_sample_size_design",
    "plan_bootstrap_assurance",
    "plan_complex_csi_bootstrap_assurance",
    "plan_independent_acquisition_units",
    "plan_joint_independent_units",
    "verify_joint_sample_size_design",
]
