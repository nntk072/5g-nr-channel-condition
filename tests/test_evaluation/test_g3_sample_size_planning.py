"""Tests for the standalone, non-admitting G3.03 sample-size planner."""

from __future__ import annotations

import math

import numpy as np
import pytest
from evaluation.g3_material_calibration import (
    MAX_INTERVAL_COVERAGE as EVALUATOR_MAX_INTERVAL_COVERAGE,
)
from evaluation.g3_material_calibration import (
    MAX_MEDIAN_INTERVAL_WIDTH_DB as EVALUATOR_MAX_MEDIAN_INTERVAL_WIDTH_DB,
)
from evaluation.g3_material_calibration import (
    MAX_POWER_ERROR_MEDIAN_DB as EVALUATOR_MAX_POWER_ERROR_MEDIAN_DB,
)
from evaluation.g3_material_calibration import (
    MAX_POWER_ERROR_P95_DB as EVALUATOR_MAX_POWER_ERROR_P95_DB,
)
from evaluation.g3_material_calibration import (
    MIN_INDEPENDENT_MEASUREMENT_UNITS as EVALUATOR_MIN_INDEPENDENT_MEASUREMENT_UNITS,
)
from evaluation.g3_material_calibration import (
    MIN_INTERVAL_COVERAGE as EVALUATOR_MIN_INTERVAL_COVERAGE,
)
from evaluation.g3_material_calibration import _cluster_bootstrap_intervals
from evaluation.g3_sample_size_planning import (
    MAX_INTERVAL_COVERAGE,
    MAX_MEDIAN_INTERVAL_WIDTH_DB,
    MAX_POWER_ERROR_MEDIAN_DB,
    MAX_POWER_ERROR_P95_DB,
    MIN_INDEPENDENT_MEASUREMENT_UNITS,
    MIN_INTERVAL_COVERAGE,
    _exact_binomial_critical_failures,
    _maximum_angular_units_for_pass_probability,
    _required_bounded_mean_units,
    _required_quantile_units,
    plan_g3_independent_acquisition_units,
)
from scipy.stats import binom


def _assumptions() -> dict[str, float]:
    return {
        "median_absolute_power_error_db": 0.90,
        "p95_absolute_power_error_db": 0.99,
        "median_prediction_interval_width_db": 0.90,
    }


def test_local_planning_contract_margins_match_evaluator_without_runtime_coupling() -> None:
    assert MAX_INTERVAL_COVERAGE == EVALUATOR_MAX_INTERVAL_COVERAGE
    assert MAX_MEDIAN_INTERVAL_WIDTH_DB == EVALUATOR_MAX_MEDIAN_INTERVAL_WIDTH_DB
    assert MAX_POWER_ERROR_MEDIAN_DB == EVALUATOR_MAX_POWER_ERROR_MEDIAN_DB
    assert MAX_POWER_ERROR_P95_DB == EVALUATOR_MAX_POWER_ERROR_P95_DB
    assert MIN_INTERVAL_COVERAGE == EVALUATOR_MIN_INTERVAL_COVERAGE
    assert MIN_INDEPENDENT_MEASUREMENT_UNITS == EVALUATOR_MIN_INDEPENDENT_MEASUREMENT_UNITS


def test_planner_is_data_free_bonferroni_adjusted_and_never_admits() -> None:
    result = plan_g3_independent_acquisition_units(
        stratum_count=4,
        planned_quantile_pass_probabilities=_assumptions(),
        planned_interval_coverage=0.95,
        planned_complex_csi_failure_probability=0.01,
    )

    assert result["schema"] == "loslab.g3_sample_size_planning.v2"
    assert result["status"] == "PLANNING_ONLY_NOT_QUALIFICATION"
    assert result["uses_measurement_or_pilot_data"] is False
    assert result["admission_possible"] is False
    assert result["complete_g3_03_sample_size"] is None
    assert (
        result["complete_design_status"] == "UNRESOLVED_MISSING_ANGULAR_PASS_PROBABILITY_ASSUMPTION"
    )
    assert "not a sample-size guarantee" in result["evaluation_alignment"]
    assert result["angular_maximum_metric_plan"]["assumption_supplied"] is False
    assert result["angular_maximum_metric_plan"]["maximum_independent_units_per_stratum"] is None
    assert "weakest stratum" in result["shared_assumptions_scope"]
    assert "represent the target population" in result["population_sampling_assumption"]
    assert result["bonferroni_comparison_count"] == 20
    assert result["per_comparison_alpha"] == pytest.approx(0.05 / 20)
    assert result["per_comparison_beta"] == pytest.approx(0.20 / 20)
    failure_plan = result["complex_csi_cluster_failure_plan"]
    assert failure_plan["admission_possible"] is False
    assert failure_plan["is_complete_g3_03_design"] is False
    assert failure_plan["stratum_count"] == 4
    assert failure_plan["per_stratum_alpha_bonferroni"] == pytest.approx(0.05 / 4)
    assert failure_plan["per_stratum_power_target_bonferroni"] == pytest.approx(0.95)
    assert failure_plan["sample_size_found"] is True
    assert failure_plan["achieved_per_stratum_power"] >= 0.95
    assert failure_plan["single_stratum_one_sided_95pct_zero_failure_minimum_clusters"] == 59
    assert "independent and representative" in failure_plan["sampling_assumption"]
    assert "any episode" in failure_plan["criterion"]
    assert result["hypothetical_required_units_per_stratum_for_supported_metrics"] == max(
        [
            plan["required_independent_units_per_stratum"]
            for plan in result["quantile_metric_plans"].values()
        ]
        + [
            result["bounded_mean_metric_plans"]["nominal_95_interval_coverage"][
                "required_independent_units_per_stratum"
            ]
        ]
    )


def test_quantile_planning_formula_is_conservative_hoeffding_bound() -> None:
    alpha = 0.01
    beta = 0.05
    quantile = 0.95
    planned_pass = 0.99
    expected = math.ceil(
        (math.sqrt(math.log(1.0 / alpha)) + math.sqrt(math.log(1.0 / beta))) ** 2
        / (2.0 * (planned_pass - quantile) ** 2)
    )

    assert (
        _required_quantile_units(
            quantile=quantile,
            planned_pass_probability=planned_pass,
            alpha=alpha,
            beta=beta,
        )
        == expected
    )


def test_quantile_sample_size_increases_as_assumed_margin_shrinks() -> None:
    broad = _required_quantile_units(
        quantile=0.50,
        planned_pass_probability=0.90,
        alpha=0.01,
        beta=0.05,
    )
    narrow = _required_quantile_units(
        quantile=0.50,
        planned_pass_probability=0.80,
        alpha=0.01,
        beta=0.05,
    )

    assert narrow > broad


def test_bounded_mean_formula_uses_full_unit_range_and_interior_slack() -> None:
    units = _required_bounded_mean_units(
        planned_mean=0.95,
        lower_margin=0.90,
        upper_margin=0.98,
        outcome_lower=0.0,
        outcome_upper=1.0,
        alpha=0.01,
        beta=0.05,
    )
    slack = 0.03
    expected = math.ceil(
        (math.sqrt(math.log(2.0 / 0.01)) + math.sqrt(math.log(2.0 / 0.05))) ** 2 / (2.0 * slack**2)
    )

    assert units == expected


def test_angular_maximum_plan_is_an_upper_bound_under_exact_iid_pass_probability() -> None:
    assert (
        _maximum_angular_units_for_pass_probability(
            planned_unit_pass_probability=0.90,
            minimum_pass_probability=0.80,
        )
        == 2
    )
    assert 0.90**2 >= 0.80
    assert 0.90**3 < 0.80
    assert (
        _maximum_angular_units_for_pass_probability(
            planned_unit_pass_probability=0.90,
            minimum_pass_probability=0.95,
        )
        == 0
    )
    assert (
        _maximum_angular_units_for_pass_probability(
            planned_unit_pass_probability=1.0,
            minimum_pass_probability=0.95,
        )
        is None
    )


def test_bootstrap_maximum_cannot_exceed_the_observed_unit_maximum() -> None:
    units = {
        f"unit-{index:02d}": [
            {
                "absolute_power_error_db": 0.1,
                "angular_cdf_ks": index / 100.0,
                "interval_width_db": 1.0,
                "covered": 1.0,
            }
        ]
        for index in range(20)
    }
    report = _cluster_bootstrap_intervals(
        units,
        seed=41023,
        replicates=2_000,
        simultaneous_comparisons=5,
    )
    observed_maximum = max(rows[0]["angular_cdf_ks"] for rows in units.values())
    bootstrap_upper = report["percentile_95_ci"]["maximum_angular_cdf_ks"][1]
    assert bootstrap_upper <= observed_maximum


def test_angular_maximum_plan_fails_closed_without_assumption_and_flags_conflict() -> None:
    result = plan_g3_independent_acquisition_units(
        stratum_count=1,
        planned_quantile_pass_probabilities=_assumptions(),
        planned_interval_coverage=0.95,
        planned_complex_csi_failure_probability=0.01,
        planned_angular_unit_pass_probability=0.90,
    )
    angular = result["angular_maximum_metric_plan"]
    assert angular["maximum_independent_units_per_stratum"] == 0
    assert angular["compatible_with_pilot_floor"] is False
    assert angular["compatible_with_supported_metrics_lower_bound"] is False
    assert result["complete_design_status"] == "INFEASIBLE_UNDER_SUPPLIED_ANGULAR_ASSUMPTION"
    assert result["complete_g3_03_sample_size"] is None
    assert result["admission_possible"] is False


@pytest.mark.parametrize("value", [True, -0.1, 1.1, float("nan"), float("inf")])
def test_angular_pass_probability_rejects_invalid_assumptions(value: float) -> None:
    with pytest.raises(ValueError, match="planned_angular_unit_pass_probability"):
        plan_g3_independent_acquisition_units(
            stratum_count=1,
            planned_quantile_pass_probabilities=_assumptions(),
            planned_interval_coverage=0.95,
            planned_complex_csi_failure_probability=0.01,
            planned_angular_unit_pass_probability=value,
        )


def test_exact_one_sided_95_percent_bound_requires_59_zero_failure_clusters() -> None:
    critical = _exact_binomial_critical_failures(
        np.array([58, 59], dtype=np.int64),
        per_stratum_alpha=0.05,
    )

    assert critical.tolist() == [-1, 0]
    assert binom.cdf(0, 58, 0.05) > 0.05
    assert binom.cdf(0, 59, 0.05) <= 0.05


def test_exact_binomial_design_is_minimal_for_preregistered_power() -> None:
    result = plan_g3_independent_acquisition_units(
        stratum_count=1,
        planned_quantile_pass_probabilities=_assumptions(),
        planned_interval_coverage=0.95,
        planned_complex_csi_failure_probability=0.01,
    )["complex_csi_cluster_failure_plan"]
    required = result["required_independent_outer_clusters_per_stratum"]
    cutoff = result["maximum_admissible_observed_failures_per_stratum"]

    assert result["sample_size_found"] is True
    assert isinstance(required, int)
    assert isinstance(cutoff, int)
    assert result["achieved_per_stratum_power"] >= 0.80
    assert binom.cdf(cutoff, required, 0.01) >= 0.80
    previous_cutoff = int(
        _exact_binomial_critical_failures(
            np.array([required - 1], dtype=np.int64),
            per_stratum_alpha=0.05,
        )[0]
    )
    previous_power = (
        float(binom.cdf(previous_cutoff, required - 1, 0.01)) if previous_cutoff >= 0 else 0.0
    )
    assert previous_power < 0.80
    four_strata = plan_g3_independent_acquisition_units(
        stratum_count=4,
        planned_quantile_pass_probabilities=_assumptions(),
        planned_interval_coverage=0.95,
        planned_complex_csi_failure_probability=0.01,
    )["complex_csi_cluster_failure_plan"]
    assert four_strata["required_independent_outer_clusters_per_stratum"] > required


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"stratum_count": 0}, "stratum_count"),
        ({"stratum_count": True}, "stratum_count"),
        ({"planned_interval_coverage": 0.90}, "strictly inside"),
        ({"planned_complex_csi_failure_probability": 0.05}, "must be below"),
        ({"planned_complex_csi_failure_probability": 0.06}, "must be below"),
        ({"planned_quantile_pass_probabilities": {}}, "must specify exactly"),
        (
            {
                "planned_quantile_pass_probabilities": {
                    **_assumptions(),
                    "unplanned_metric": 0.9,
                }
            },
            "must specify exactly",
        ),
        (
            {
                "planned_quantile_pass_probabilities": {
                    **_assumptions(),
                    "p95_absolute_power_error_db": 0.95,
                }
            },
            "must exceed the target quantile",
        ),
    ],
)
def test_planner_rejects_incomplete_or_incompatible_design_assumptions(
    kwargs: dict[str, object], message: str
) -> None:
    arguments: dict[str, object] = {
        "stratum_count": 2,
        "planned_quantile_pass_probabilities": _assumptions(),
        "planned_interval_coverage": 0.95,
        "planned_complex_csi_failure_probability": 0.01,
    }
    arguments.update(kwargs)

    with pytest.raises(ValueError, match=message):
        plan_g3_independent_acquisition_units(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize("planned_mean", [0.90, 0.98, 0.89, 0.99])
def test_bounded_mean_rejects_target_on_or_outside_acceptance_margin(
    planned_mean: float,
) -> None:
    with pytest.raises(ValueError, match="strictly inside"):
        _required_bounded_mean_units(
            planned_mean=planned_mean,
            lower_margin=0.90,
            upper_margin=0.98,
            outcome_lower=0.0,
            outcome_upper=1.0,
            alpha=0.01,
            beta=0.05,
        )
