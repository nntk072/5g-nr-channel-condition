"""Independent acquisition-unit planning for the joint evidence."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from evaluation.joint_acceptance import (
    COMPLEX_CSI_CLAIMS,
    FAMILY_SCHEMA_V2,
    FAMILY_SCHEMA_V3,
    FAMILY_SCHEMA_V4,
    MATERIAL_CLAIMS,
    PROPAGATION_CLAIMS,
    RECEIVER_IMPACT_CLAIMS,
    family_alpha_for_claim,
    validate_joint_acceptance_family,
)
from evaluation.sample_size.binomial import (
    _exact_binomial_failure_design,
    _maximum_angular_units_for_pass_probability,
    _probability,
    _required_bounded_mean_units,
    _required_quantile_units,
)
from evaluation.sample_size.constants import (
    _ALL_CLAIMS_PER_STRATUM,
    _COMPARISONS_PER_STRATUM,
    _COMPLEX_CSI_FAILURE_LIMIT,
    _COMPLEX_CSI_P95_LIMIT,
    _PROPAGATION_FAILURE_LIMIT,
    _PROPAGATION_METRICS,
    _PROPAGATION_P95_LIMIT,
    _QUANTILE_TARGETS,
    MAX_INTERVAL_COVERAGE,
    MIN_INDEPENDENT_MEASUREMENT_UNITS,
    MIN_INTERVAL_COVERAGE,
)


def plan_independent_acquisition_units(
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

    comparison_count = stratum_count * _ALL_CLAIMS_PER_STRATUM
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
        "all_claims_per_stratum": _ALL_CLAIMS_PER_STRATUM,
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


def plan_joint_independent_units(
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
    family = validate_joint_acceptance_family(joint_acceptance_family)
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
                {
                    "path_equivalence_expected_scenarios": family[
                        "path_equivalence_expected_scenarios"
                    ]
                }
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
