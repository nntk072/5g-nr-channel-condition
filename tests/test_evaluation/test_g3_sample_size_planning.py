"""Tests for the standalone, non-admitting G3.03 sample-size planner."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from evaluation.g3_joint_acceptance import (
    build_g3_joint_acceptance_family,
    build_g3_joint_acceptance_family_v2,
    build_g3_joint_acceptance_family_v3,
    build_g3_joint_acceptance_family_v4,
)
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
    _bootstrap_acceptance_checks,
    _exact_binomial_critical_failures,
    _maximum_angular_units_for_pass_probability,
    _required_bounded_mean_units,
    _required_quantile_units,
    compose_g3_joint_sample_size_design,
    plan_g3_bootstrap_assurance,
    plan_g3_complex_csi_bootstrap_assurance,
    plan_g3_independent_acquisition_units,
    plan_g3_joint_independent_units,
    verify_g3_joint_sample_size_design,
)
from scipy.stats import binom


def _assumptions() -> dict[str, float]:
    return {
        "median_absolute_power_error_db": 0.90,
        "p95_absolute_power_error_db": 0.99,
        "median_prediction_interval_width_db": 0.90,
    }


def _bootstrap_atoms() -> list[dict[str, float]]:
    return [
        {
            "median_absolute_power_error_db": 0.5,
            "p95_absolute_power_error_db": 1.0,
            "maximum_angular_cdf_ks": 0.05,
            "nominal_95_interval_coverage": 0.95,
            "median_prediction_interval_width_db": 1.0,
        },
        {
            "median_absolute_power_error_db": 0.7,
            "p95_absolute_power_error_db": 1.2,
            "maximum_angular_cdf_ks": 0.06,
            "nominal_95_interval_coverage": 0.96,
            "median_prediction_interval_width_db": 1.2,
        },
    ]


def _bootstrap_scenario(atoms: list[dict[str, float]]) -> dict[str, Any]:
    return {"assumed_unit_summaries": atoms, "probabilities": [0.5, 0.5]}


def _joint_g3_family() -> dict[str, Any]:
    return build_g3_joint_acceptance_family(
        family_id="planning-test",
        material_expected_strata=[("brick", "V", "normal")],
        complex_csi_expected_strata=[("urban", "LOS")],
    )


def _joint_g3_family_v2() -> dict[str, Any]:
    return build_g3_joint_acceptance_family_v2(
        family_id="planning-test-v2",
        material_expected_strata=[("brick", "V", "normal")],
        complex_csi_expected_strata=[("urban", "LOS")],
        propagation_expected_strata=[
            ("urban", "LOS", "total"),
            ("urban", "LOS", "specular"),
        ],
    )


def _csi_bootstrap_scenario(*, median: float, p95: float, failed: bool) -> dict[str, Any]:
    return {
        "assumed_cluster_summaries": [
            {
                "cluster_median_e_h": median,
                "cluster_p95_e_h": p95,
                "cluster_failed": failed,
            }
        ],
        "probabilities": [1.0],
    }


def _joint_v2_plan(family: dict[str, Any]) -> dict[str, Any]:
    metrics = {
        "first_arrival_delay_s",
        "rms_delay_spread_s",
        "arrival_angular_spread_deg",
        "departure_angular_spread_deg",
    }
    return plan_g3_joint_independent_units(
        joint_acceptance_family=family,
        planned_quantile_pass_probabilities=_assumptions(),
        planned_interval_coverage=0.95,
        planned_complex_csi_p95_pass_probability=0.99,
        planned_complex_csi_failure_probability=0.01,
        planned_angular_unit_pass_probability=None,
        planned_propagation_p95_pass_probabilities={metric: 0.99 for metric in metrics},
        planned_propagation_cluster_failure_probabilities={metric: 0.01 for metric in metrics},
    )


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

    assert result["schema"] == "loslab.g3_sample_size_planning.v3"
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
    assert result["distributional_metrics_per_stratum"] == 5
    assert result["all_claims_per_stratum"] == 6
    assert result["bonferroni_comparison_count"] == 24
    assert result["per_comparison_alpha"] == pytest.approx(0.05 / 24)
    assert result["per_comparison_beta"] == pytest.approx(0.20 / 24)
    failure_plan = result["complex_csi_cluster_failure_plan"]
    assert failure_plan["admission_possible"] is False
    assert failure_plan["is_complete_g3_03_design"] is False
    assert failure_plan["stratum_count"] == 4
    assert failure_plan["per_comparison_alpha_bonferroni"] == pytest.approx(0.05 / 24)
    assert failure_plan["per_comparison_power_target_bonferroni"] == pytest.approx(1.0 - 0.20 / 24)
    assert failure_plan["familywise_claim_count"] == 24
    assert "complex-CSI cluster failure" in failure_plan["familywise_scope"]
    assert failure_plan["sample_size_found"] is True
    assert failure_plan["achieved_per_comparison_power"] >= 1.0 - 0.20 / 24
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
    assert result["hypothetical_required_units_per_stratum_for_all_six_claims"] == max(
        result["hypothetical_required_units_per_stratum_for_supported_metrics"],
        failure_plan["required_independent_outer_clusters_per_stratum"],
    )
    assert result["minimum_units_for_all_six_claims_and_pilot_floor"] == max(
        result["hypothetical_required_units_per_stratum_for_all_six_claims"],
        MIN_INDEPENDENT_MEASUREMENT_UNITS,
    )


def test_bootstrap_assurance_is_scenario_conditional_and_never_qualifies() -> None:
    scenarios = {
        "brick-H-0-30": _bootstrap_scenario(_bootstrap_atoms()),
        "concrete-V-30-60": _bootstrap_scenario(_bootstrap_atoms()),
    }
    result = plan_g3_bootstrap_assurance(
        assumed_iid_scenarios_by_stratum=scenarios,
        candidate_units_per_stratum=[20, 30],
        bootstrap_replicates=1_000,
        assurance_replicates=100,
        target_assurance=0.80,
        seed=701,
    )

    assert result["schema"] == "loslab.g3_bootstrap_assurance_planning.v1"
    assert result["status"] == "SCENARIO_CONDITIONAL_PLANNING_ONLY_NOT_QUALIFICATION"
    assert result["admission_possible"] is False
    assert result["scenario_provenance_verified"] is False
    assert result["complete_g3_03_sample_size"] is None
    assert result["selected_units_per_stratum_among_evaluated_candidates"] == 20
    first = result["candidate_designs"][0]
    assert first["joint_assurance_lower_bound_union_bound"] >= 0.80
    assert set(first["stratum_assurance"]) == set(scenarios)
    assert all(
        entry["joint_assurance_estimate"] == 1.0 for entry in first["stratum_assurance"].values()
    )
    assert (
        result["scenario_distribution_sha256"]
        == plan_g3_bootstrap_assurance(
            assumed_iid_scenarios_by_stratum=scenarios,
            candidate_units_per_stratum=[20, 30],
            bootstrap_replicates=1_000,
            assurance_replicates=100,
            target_assurance=0.80,
            seed=701,
        )["scenario_distribution_sha256"]
    )


def test_complex_csi_bootstrap_assurance_matches_joint_family_and_exact_cluster_rule() -> None:
    scenarios = {("urban", "LOS"): _csi_bootstrap_scenario(median=0.005, p95=0.02, failed=False)}
    kwargs: dict[str, Any] = {
        "joint_acceptance_family": _joint_g3_family(),
        "assumed_iid_scenarios_by_stratum": scenarios,
        "candidate_clusters_per_stratum": [20, 120],
        "bootstrap_replicates": 1_000,
        "assurance_replicates": 100,
        "target_assurance": 0.80,
        "seed": 701,
    }
    result = plan_g3_complex_csi_bootstrap_assurance(**kwargs)

    assert result["schema"] == "loslab.g3_complex_csi_bootstrap_assurance_planning.v1"
    assert result["qualification_status"] == "NOT_QUALIFIED"
    assert result["admission_possible"] is False
    assert result["complete_g3_sample_size"] is None
    assert result["joint_acceptance_family"]["claim_count"] == 7
    assert result["candidate_designs"][0]["joint_assurance_lower_bound_union_bound"] == 0.0
    assert result["candidate_designs"][0]["meets_target_assurance"] is False
    assert result["candidate_designs"][1]["joint_assurance_lower_bound_union_bound"] >= 0.80
    assert result["candidate_designs"][1]["meets_target_assurance"] is True
    assert result["selected_clusters_per_stratum_among_evaluated_candidates"] == 120
    assert result["sampling_units_are_scene_clusters"] is True
    assert (
        result["scenario_sha256"]
        == plan_g3_complex_csi_bootstrap_assurance(**kwargs)["scenario_sha256"]
    )


def test_complex_csi_bootstrap_assurance_rejects_missing_joint_family_strata() -> None:
    with pytest.raises(ValueError, match="exactly match the joint-family strata"):
        plan_g3_complex_csi_bootstrap_assurance(
            joint_acceptance_family=_joint_g3_family(),
            assumed_iid_scenarios_by_stratum={},
            candidate_clusters_per_stratum=[20],
        )


def test_bootstrap_assurance_does_not_select_a_design_for_failing_assumptions() -> None:
    failing_atoms = _bootstrap_atoms()
    failing_atoms[1] = {**failing_atoms[1], "p95_absolute_power_error_db": 4.0}
    result = plan_g3_bootstrap_assurance(
        assumed_iid_scenarios_by_stratum={"brick-H-0-30": _bootstrap_scenario(failing_atoms)},
        candidate_units_per_stratum=[20],
        bootstrap_replicates=1_000,
        assurance_replicates=100,
        target_assurance=0.80,
        seed=702,
    )

    assert result["selected_units_per_stratum_among_evaluated_candidates"] is None
    assert result["candidate_designs"][0]["meets_target_assurance"] is False


def test_bootstrap_assurance_acceptance_rules_match_the_evaluator() -> None:
    errors = np.linspace(0.2, 2.0, 20)
    coverage = np.asarray([True] * 19 + [False])
    values = np.column_stack(
        (
            errors,
            errors,
            np.linspace(0.01, 0.12, 20),
            coverage.astype(np.float64),
            np.linspace(0.5, 3.5, 20),
        )
    )
    units = {
        f"unit-{index:02d}": [
            {
                "absolute_power_error_db": float(errors[index]),
                "angular_cdf_ks": float(values[index, 2]),
                "interval_width_db": float(values[index, 4]),
                "covered": bool(coverage[index]),
            }
        ]
        for index in range(20)
    }
    seed = 812
    replicates = 1_000
    evaluator_result = _cluster_bootstrap_intervals(
        units,
        seed=seed,
        replicates=replicates,
        simultaneous_comparisons=5,
    )
    actual = _bootstrap_acceptance_checks(
        values,
        rng=np.random.default_rng(seed),
        bootstrap_replicates=replicates,
        per_comparison_alpha=0.05 / 5,
    )
    ci = evaluator_result["percentile_95_ci"]
    expected = {
        "median_absolute_power_error_db": np.median(errors) <= MAX_POWER_ERROR_MEDIAN_DB
        and ci["median_absolute_power_error_db"][1] <= MAX_POWER_ERROR_MEDIAN_DB,
        "p95_absolute_power_error_db": np.percentile(errors, 95) <= MAX_POWER_ERROR_P95_DB
        and ci["p95_absolute_power_error_db"][1] <= MAX_POWER_ERROR_P95_DB,
        "maximum_angular_cdf_ks": np.max(values[:, 2]) <= 0.10
        and ci["maximum_angular_cdf_ks"][1] <= 0.10,
        "nominal_95_interval_coverage": np.mean(coverage) >= MIN_INTERVAL_COVERAGE
        and np.mean(coverage) <= MAX_INTERVAL_COVERAGE
        and ci["nominal_95_interval_coverage"][0] >= MIN_INTERVAL_COVERAGE
        and ci["nominal_95_interval_coverage"][1] <= MAX_INTERVAL_COVERAGE,
        "median_prediction_interval_width_db": np.median(values[:, 4])
        <= MAX_MEDIAN_INTERVAL_WIDTH_DB
        and ci["median_prediction_interval_width_db"][1] <= MAX_MEDIAN_INTERVAL_WIDTH_DB,
    }

    assert actual == expected


@pytest.mark.parametrize(
    ("scenarios", "counts", "error"),
    [
        ({}, [20], "nonempty mapping"),
        ({"stratum": _bootstrap_scenario(_bootstrap_atoms())}, [19], "integers >= 20"),
        (
            {
                "stratum": {
                    "assumed_unit_summaries": _bootstrap_atoms(),
                    "probabilities": [0.4, 0.4],
                }
            },
            [20],
            "sum to one",
        ),
    ],
)
def test_bootstrap_assurance_rejects_invalid_design_inputs(
    scenarios: dict[str, dict[str, Any]], counts: list[int], error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        plan_g3_bootstrap_assurance(
            assumed_iid_scenarios_by_stratum=scenarios,
            candidate_units_per_stratum=counts,
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
    assert angular["compatible_with_combined_six_claim_lower_bound"] is False
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
        per_comparison_alpha=0.05,
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
    assert (
        result["achieved_per_comparison_power"] >= result["per_comparison_power_target_bonferroni"]
    )
    assert binom.cdf(cutoff, required, 0.01) >= result["per_comparison_power_target_bonferroni"]
    previous_cutoff = int(
        _exact_binomial_critical_failures(
            np.array([required - 1], dtype=np.int64),
            per_comparison_alpha=result["per_comparison_alpha_bonferroni"],
        )[0]
    )
    previous_power = (
        float(binom.cdf(previous_cutoff, required - 1, 0.01)) if previous_cutoff >= 0 else 0.0
    )
    assert previous_power < result["per_comparison_power_target_bonferroni"]
    four_strata = plan_g3_independent_acquisition_units(
        stratum_count=4,
        planned_quantile_pass_probabilities=_assumptions(),
        planned_interval_coverage=0.95,
        planned_complex_csi_failure_probability=0.01,
    )["complex_csi_cluster_failure_plan"]
    assert four_strata["required_independent_outer_clusters_per_stratum"] > required


def test_v2_joint_planner_designs_separate_propagation_strata_under_shared_alpha_power():
    family = _joint_g3_family_v2()

    report = _joint_v2_plan(family)

    assert report["joint_acceptance_family"]["claim_count"] == family["claim_count"] == 23
    assert report["joint_acceptance_family"]["per_claim_alpha_bonferroni"] == pytest.approx(
        0.05 / 23
    )
    propagation = report["propagation_design"]
    assert propagation["stratum_count"] == 2
    assert propagation["claim_count"] == 16
    assert propagation["assumptions_are_worst_stratum_across_all_declared_strata"] is True
    assert isinstance(
        propagation["required_independent_clusters_per_stratum_for_supported_claims"], int
    )
    assert set(propagation["metric_plans"]) == {
        "first_arrival_delay_s",
        "rms_delay_spread_s",
        "arrival_angular_spread_deg",
        "departure_angular_spread_deg",
    }
    assert report["single_combined_sample_size_is_reported"] is False
    assert report["complete_g3_sample_size"] is None


def test_v2_joint_planner_requires_all_propagation_assumptions():
    family = _joint_g3_family_v2()
    with pytest.raises(ValueError, match="requires propagation p95-pass and cluster-failure"):
        plan_g3_joint_independent_units(
            joint_acceptance_family=family,
            planned_quantile_pass_probabilities=_assumptions(),
            planned_interval_coverage=0.95,
            planned_complex_csi_p95_pass_probability=0.99,
            planned_complex_csi_failure_probability=0.01,
            planned_angular_unit_pass_probability=None,
        )


def test_v2_joint_planner_rejects_unfavorable_propagation_assumptions():
    family = _joint_g3_family_v2()
    metrics = {
        "first_arrival_delay_s",
        "rms_delay_spread_s",
        "arrival_angular_spread_deg",
        "departure_angular_spread_deg",
    }
    with pytest.raises(ValueError, match="planned pass probability must exceed"):
        plan_g3_joint_independent_units(
            joint_acceptance_family=family,
            planned_quantile_pass_probabilities=_assumptions(),
            planned_interval_coverage=0.95,
            planned_complex_csi_p95_pass_probability=0.99,
            planned_complex_csi_failure_probability=0.01,
            planned_angular_unit_pass_probability=None,
            planned_propagation_p95_pass_probabilities={
                **{metric: 0.99 for metric in metrics},
                "first_arrival_delay_s": 0.95,
            },
            planned_propagation_cluster_failure_probabilities={metric: 0.01 for metric in metrics},
        )


def test_joint_planner_counts_v3_receiver_claims_without_claiming_complete_design():
    family = build_g3_joint_acceptance_family_v3(
        family_id="planning-test-v3",
        material_expected_strata=[("brick", "V", "normal")],
        complex_csi_expected_strata=[("urban", "LOS")],
        propagation_expected_strata=[("urban", "LOS", "total")],
        receiver_impact_expected_strata=[("urban", "LOS")],
    )

    metrics = {
        "first_arrival_delay_s",
        "rms_delay_spread_s",
        "arrival_angular_spread_deg",
        "departure_angular_spread_deg",
    }
    result = plan_g3_joint_independent_units(
        joint_acceptance_family=family,
        planned_quantile_pass_probabilities=_assumptions(),
        planned_interval_coverage=0.95,
        planned_complex_csi_p95_pass_probability=0.99,
        planned_complex_csi_failure_probability=0.01,
        planned_angular_unit_pass_probability=None,
        planned_propagation_p95_pass_probabilities={metric: 0.99 for metric in metrics},
        planned_propagation_cluster_failure_probabilities={metric: 0.01 for metric in metrics},
    )

    assert result["joint_acceptance_family"]["claim_count"] == 19
    assert result["receiver_impact_design"]["claim_count"] == 4
    assert result["receiver_impact_design"]["complete_precision_powered_design"] is False
    assert result["complete_g3_sample_size"] is None


def test_joint_planner_accepts_v4_path_roster_without_counting_path_episodes_as_claims():
    family = build_g3_joint_acceptance_family_v4(
        family_id="planning-test-v4",
        material_expected_strata=[("brick", "V", "normal")],
        complex_csi_expected_strata=[("urban", "LOS")],
        propagation_expected_strata=[("urban", "LOS", "total")],
        receiver_impact_expected_strata=[("urban", "LOS")],
        path_equivalence_expected_scenarios=[
            ("urban", "LOS", "episode-1"),
            ("urban", "LOS", "episode-2"),
        ],
    )
    metrics = {
        "first_arrival_delay_s",
        "rms_delay_spread_s",
        "arrival_angular_spread_deg",
        "departure_angular_spread_deg",
    }

    result = plan_g3_joint_independent_units(
        joint_acceptance_family=family,
        planned_quantile_pass_probabilities=_assumptions(),
        planned_interval_coverage=0.95,
        planned_complex_csi_p95_pass_probability=0.99,
        planned_complex_csi_failure_probability=0.01,
        planned_angular_unit_pass_probability=None,
        planned_propagation_p95_pass_probabilities={metric: 0.99 for metric in metrics},
        planned_propagation_cluster_failure_probabilities={metric: 0.01 for metric in metrics},
    )

    assert result["joint_acceptance_family"]["claim_count"] == 19
    assert result["joint_acceptance_family"]["path_equivalence_expected_scenarios"] == [
        ["urban", "LOS", "episode-1"],
        ["urban", "LOS", "episode-2"],
    ]
    assert result["receiver_impact_design"]["claim_count"] == 4
    assert result["propagation_design"]["claim_count"] == 8
    assert result["complete_g3_sample_size"] is None


def test_joint_design_composer_aligns_branch_counts_without_claiming_qualification():
    family = build_g3_joint_acceptance_family_v4(
        family_id="composition-v4",
        material_expected_strata=[("brick", "V", "normal")],
        complex_csi_expected_strata=[("urban", "LOS")],
        propagation_expected_strata=[("urban", "LOS", "total")],
        receiver_impact_expected_strata=[("urban", "LOS")],
        path_equivalence_expected_scenarios=[
            ("urban", "LOS", "episode-1"),
            ("urban", "LOS", "episode-2"),
        ],
    )
    identity = {
        "family_id": family["family_id"],
        "family_sha256": family["family_sha256"],
    }
    common = {"admission_possible": False, "qualification_status": "NOT_QUALIFIED"}
    joint = {
        "schema": "loslab.g3_joint_sample_size_planning.v1",
        **common,
        "single_combined_sample_size_is_reported": False,
        "complete_g3_sample_size": None,
        "joint_acceptance_family": identity,
        "material_design": {
            "stratum_count": 1,
            "required_independent_measurement_units_per_stratum_for_supported_claims": 40,
            "complete_precision_powered_design": False,
            "angular_maximum_claim": {"assumption_supplied": False},
        },
        "complex_csi_design": {
            "stratum_count": 1,
            "required_independent_scene_clusters_per_stratum_for_supported_claims": 25,
            "complete_precision_powered_design": False,
        },
        "propagation_design": {
            "stratum_count": 1,
            "strata": [["urban", "LOS", "total"]],
            "required_independent_clusters_per_stratum_for_supported_claims": 30,
            "complete_precision_powered_design": False,
        },
        "receiver_impact_design": {
            "stratum_count": 1,
            "strata": [["urban", "LOS"]],
            "complete_precision_powered_design": False,
        },
    }

    def assurance(
        schema: str, stratum: list[str], count: int, key_fields: tuple[str, ...]
    ) -> dict[str, Any]:
        return {
            "schema": schema,
            **common,
            "scenario_provenance_verified": False,
            "joint_acceptance_family": identity,
            "selected_clusters_per_stratum_among_evaluated_candidates": count,
            "target_assurance": 0.8,
            "candidate_designs": [
                {
                    "candidate_clusters_per_stratum": count,
                    "joint_assurance_lower_bound_union_bound": 0.81,
                    "meets_target_assurance": True,
                    "stratum_assurance": [
                        {
                            **dict(zip(key_fields, stratum, strict=True)),
                            "cluster_count": count,
                            "assurance_one_sided_exact_lower_bound": 0.81,
                        }
                    ],
                }
            ],
        }

    result = compose_g3_joint_sample_size_design(
        joint_acceptance_family=family,
        joint_plan=joint,
        material_assurance_plan={
            "schema": "loslab.g3_bootstrap_assurance_planning.v1",
            **common,
            "scenario_provenance_verified": False,
            "joint_acceptance_family": identity,
            "stratum_roster_by_id": {"material-1": ["brick", "V", "normal"]},
            "selected_units_per_stratum_among_evaluated_candidates": 50,
            "target_assurance": 0.8,
            "candidate_designs": [
                {
                    "independent_units_per_stratum": 50,
                    "joint_assurance_lower_bound_union_bound": 0.81,
                    "meets_target_assurance": True,
                    "stratum_assurance": {
                        "material-1": {
                            "assumed_support_atom_count": 2,
                            "joint_assurance_one_sided_exact_lower_bound": 0.81,
                        }
                    },
                }
            ],
        },
        complex_csi_assurance_plan=assurance(
            "loslab.g3_complex_csi_bootstrap_assurance_planning.v1",
            ["urban", "LOS"],
            35,
            ("family_id", "condition"),
        ),
        propagation_assurance_plan=assurance(
            "loslab.g3_propagation_bootstrap_assurance_planning.v1",
            ["urban", "LOS", "total"],
            45,
            ("family_id", "condition", "component"),
        ),
        receiver_assurance_plan=assurance(
            "loslab.g3_receiver_impact_bootstrap_assurance_planning.v1",
            ["urban", "LOS"],
            45,
            ("family_id", "condition"),
        ),
    )

    branches = result["branch_designs"]
    assert branches["material_calibration"]["required_units_by_stratum"][0][
        "required_units"
    ] == 50
    assert branches["complex_csi"]["required_units_by_stratum"][0]["required_units"] == 35
    assert branches["propagation_metrics"]["required_units_by_stratum"][0][
        "required_units"
    ] == 45
    assert branches["receiver_impact"]["required_units_by_stratum"][0][
        "required_units"
    ] == 45
    assert [
        row["required_units"] for row in branches["path_equivalence"]["required_units_by_stratum"]
    ] == [1, 1]
    assert result["all_branch_counts_available"] is True
    assert result["single_combined_sample_size_is_reported"] is False
    assert result["joint_g3_assurance_lower_bound_union_bound"] == pytest.approx(0.24)
    assert result["target_joint_g3_assurance"] == pytest.approx(0.8)
    assert result["complete_precision_powered_g3_design"] is False
    assert result["admission_possible"] is False
    assert result["complete_g3_sample_size"] is None
    schema_path = (
        Path(__file__).parents[2]
        / "datasets/schema/g3-joint-sample-size-design-composition.v1.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(result)
    verification = verify_g3_joint_sample_size_design(result)
    assert verification["status"] == "PASS_DESIGN_INTEGRITY_NOT_QUALIFIED"
    assert verification["admission_possible"] is False

    tampered_bound = dict(result)
    tampered_bound["joint_g3_assurance_lower_bound_union_bound"] = 0.99
    tampered_bound.pop("design_sha256")
    tampered_bound["design_sha256"] = hashlib.sha256(
        json.dumps(
            tampered_bound,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    with pytest.raises(ValueError, match="joint G3 assurance bound is inconsistent"):
        verify_g3_joint_sample_size_design(tampered_bound)

    tampered = dict(result)
    tampered["admission_possible"] = True
    tampered.pop("design_sha256")
    tampered["design_sha256"] = hashlib.sha256(
        json.dumps(
            tampered,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    with pytest.raises(ValueError, match="non-qualification contract"):
        verify_g3_joint_sample_size_design(tampered)

    tampered_component_claim = dict(result)
    tampered_component_claim["component_designs_claim_complete_powered_design"] = False
    tampered_component_claim.pop("design_sha256")
    tampered_component_claim["design_sha256"] = hashlib.sha256(
        json.dumps(
            tampered_component_claim,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    with pytest.raises(ValueError, match="component powered-design claim is inconsistent"):
        verify_g3_joint_sample_size_design(tampered_component_claim)


def test_joint_design_composer_rejects_assurance_plan_from_another_family():
    family = build_g3_joint_acceptance_family_v4(
        family_id="composition-v4",
        material_expected_strata=[("brick", "V", "normal")],
        complex_csi_expected_strata=[("urban", "LOS")],
        propagation_expected_strata=[("urban", "LOS", "total")],
        receiver_impact_expected_strata=[("urban", "LOS")],
        path_equivalence_expected_scenarios=[("urban", "LOS", "episode-1")],
    )
    bad = {
        "schema": "loslab.g3_complex_csi_bootstrap_assurance_planning.v1",
        "admission_possible": False,
        "qualification_status": "NOT_QUALIFIED",
        "scenario_provenance_verified": False,
        "joint_acceptance_family": {"family_id": "other", "family_sha256": "0" * 64},
    }
    receiver = dict(bad)
    receiver["schema"] = "loslab.g3_receiver_impact_bootstrap_assurance_planning.v1"
    joint = {
        "schema": "loslab.g3_joint_sample_size_planning.v1",
        "admission_possible": False,
        "qualification_status": "NOT_QUALIFIED",
        "joint_acceptance_family": {
            "family_id": family["family_id"],
            "family_sha256": family["family_sha256"],
        },
        "single_combined_sample_size_is_reported": False,
        "complete_g3_sample_size": None,
    }
    with pytest.raises(ValueError, match="does not match the supplied acceptance family"):
        compose_g3_joint_sample_size_design(
            joint_acceptance_family=family,
            joint_plan=joint,
            material_assurance_plan={
                "schema": "loslab.g3_bootstrap_assurance_planning.v1",
                "admission_possible": False,
                "qualification_status": "NOT_QUALIFIED",
                "scenario_provenance_verified": False,
                "joint_acceptance_family": {
                    "family_id": family["family_id"],
                    "family_sha256": family["family_sha256"],
                },
            },
            complex_csi_assurance_plan=bad,
            propagation_assurance_plan=receiver,
            receiver_assurance_plan=receiver,
        )


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
