"""Joint sample-size design composition: component identity and non-qualification checks.

Only the component-level admission rules are exercised here, in the order the composer
applies them. Full planner outputs are covered by the sample-size planning tests.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any

import pytest
from evaluation.acceptance.joint_acceptance import (
    build_joint_acceptance_family_v3,
    build_joint_acceptance_family_v4,
)
from evaluation.sample_size.joint_design import (
    compose_joint_sample_size_design,
    verify_joint_sample_size_design,
)

COMPONENTS = {
    "joint": "loslab.joint_sample_size_planning.v1",
    "material_assurance": "loslab.bootstrap_assurance_planning.v1",
    "complex_csi_assurance": "loslab.complex_csi_bootstrap_assurance_planning.v1",
    "propagation_assurance": "loslab.propagation_bootstrap_assurance_planning.v1",
    "receiver_assurance": "loslab.receiver_impact_bootstrap_assurance_planning.v1",
}


@pytest.fixture
def family() -> dict[str, Any]:
    return build_joint_acceptance_family_v4(
        family_id="composition-rules-v4",
        material_expected_strata=[("brick", "V", "normal")],
        complex_csi_expected_strata=[("urban", "LOS")],
        propagation_expected_strata=[("urban", "LOS", "total")],
        receiver_impact_expected_strata=[("urban", "LOS")],
        path_equivalence_expected_scenarios=[("urban", "LOS", "episode-1")],
    )


def _plans(family: dict[str, Any]) -> dict[str, dict[str, Any]]:
    identity = {"family_id": family["family_id"], "family_sha256": family["family_sha256"]}
    plans: dict[str, dict[str, Any]] = {}
    for name, schema in COMPONENTS.items():
        plan: dict[str, Any] = {
            "schema": schema,
            "admission_possible": False,
            "qualification_status": "NOT_QUALIFIED",
            "joint_acceptance_family": dict(identity),
        }
        if name != "joint":
            plan["scenario_provenance_verified"] = False
        plans[name] = plan
    return plans


def _compose(family, plans, **overrides):
    arguments = {
        "joint_acceptance_family": family,
        "joint_plan": plans["joint"],
        "material_assurance_plan": plans["material_assurance"],
        "complex_csi_assurance_plan": plans["complex_csi_assurance"],
        "propagation_assurance_plan": plans["propagation_assurance"],
        "receiver_assurance_plan": plans["receiver_assurance"],
    }
    arguments.update(overrides)
    return compose_joint_sample_size_design(**arguments)


def _minimal_joint_plan_inputs(family):
    identity = {"family_id": family["family_id"], "family_sha256": family["family_sha256"]}
    plans = _plans(family)
    plans["joint"].update(
        {
            "single_combined_sample_size_is_reported": False,
            "complete_sample_size": None,
            "material_design": {
                "stratum_count": 1,
                "required_independent_measurement_units_per_stratum_for_supported_claims": None,
                "angular_maximum_claim": {"assumption_supplied": False},
            },
            "complex_csi_design": {
                "stratum_count": 1,
                "required_independent_scene_clusters_per_stratum_for_supported_claims": None,
            },
            "propagation_design": {
                "stratum_count": 1,
                "strata": [["urban", "LOS", "total"]],
                "required_independent_clusters_per_stratum_for_supported_claims": None,
            },
            "receiver_impact_design": {"stratum_count": 1, "strata": [["urban", "LOS"]]},
        }
    )
    plans["material_assurance"].update(
        {
            "stratum_roster_by_id": {"material-1": ["brick", "V", "normal"]},
            "selected_units_per_stratum_among_evaluated_candidates": None,
        }
    )
    plans["complex_csi_assurance"]["selected_clusters_per_stratum_among_evaluated_candidates"] = (
        None
    )
    plans["propagation_assurance"]["selected_clusters_per_stratum_among_evaluated_candidates"] = (
        None
    )
    plans["receiver_assurance"]["selected_clusters_per_stratum_among_evaluated_candidates"] = None
    arguments = {
        "joint_acceptance_family": family,
        "joint_plan": plans["joint"],
        "material_assurance_plan": plans["material_assurance"],
        "complex_csi_assurance_plan": plans["complex_csi_assurance"],
        "propagation_assurance_plan": plans["propagation_assurance"],
        "receiver_assurance_plan": plans["receiver_assurance"],
    }
    assert identity == plans["joint"]["joint_acceptance_family"]
    return arguments


def _select_csi_assurance_candidate(arguments):
    plan = arguments["complex_csi_assurance_plan"]
    plan["selected_clusters_per_stratum_among_evaluated_candidates"] = 5
    plan["target_assurance"] = 0.8
    plan["candidate_designs"] = [
        {
            "candidate_clusters_per_stratum": 5,
            "joint_assurance_lower_bound_union_bound": 0.9,
            "meets_target_assurance": True,
            "stratum_assurance": [
                {
                    "family_id": "urban",
                    "condition": "LOS",
                    "cluster_count": 5,
                    "assurance_one_sided_exact_lower_bound": 0.9,
                }
            ],
        }
    ]
    return plan


def _select_material_assurance_candidate(arguments):
    plan = arguments["material_assurance_plan"]
    plan["selected_units_per_stratum_among_evaluated_candidates"] = 5
    plan["target_assurance"] = 0.8
    plan["candidate_designs"] = [
        {
            "independent_units_per_stratum": 5,
            "joint_assurance_lower_bound_union_bound": 0.9,
            "meets_target_assurance": True,
            "stratum_assurance": {
                "material-1": {
                    "assumed_support_atom_count": 2,
                    "joint_assurance_one_sided_exact_lower_bound": 0.9,
                }
            },
        }
    ]
    return plan


def _select_all_remaining_candidates(arguments):
    propagation = arguments["propagation_assurance_plan"]
    propagation["selected_clusters_per_stratum_among_evaluated_candidates"] = 5
    propagation["target_assurance"] = 0.8
    propagation["candidate_designs"] = [
        {
            "candidate_clusters_per_stratum": 5,
            "joint_assurance_lower_bound_union_bound": 0.9,
            "meets_target_assurance": True,
            "stratum_assurance": [
                {
                    "family_id": "urban",
                    "condition": "LOS",
                    "component": "total",
                    "cluster_count": 5,
                    "assurance_one_sided_exact_lower_bound": 0.9,
                }
            ],
        }
    ]
    receiver = arguments["receiver_assurance_plan"]
    receiver["selected_clusters_per_stratum_among_evaluated_candidates"] = 5
    receiver["target_assurance"] = 0.8
    receiver["candidate_designs"] = [
        {
            "candidate_clusters_per_stratum": 5,
            "joint_assurance_lower_bound_union_bound": 0.9,
            "meets_target_assurance": True,
            "stratum_assurance": [
                {
                    "family_id": "urban",
                    "condition": "LOS",
                    "cluster_count": 5,
                    "assurance_one_sided_exact_lower_bound": 0.9,
                }
            ],
        }
    ]


@pytest.mark.parametrize("bound", [0.0, 1.0, 1.5, float("nan"), True])
def test_target_assurance_must_be_a_probability(family, bound) -> None:
    plans = _plans(family)
    with pytest.raises(ValueError):
        _compose(family, plans, target_joint_assurance_lower_bound=bound)


@pytest.mark.parametrize("name", sorted(COMPONENTS))
def test_each_component_plan_must_have_its_schema(family, name) -> None:
    plans = _plans(family)
    plans[name]["schema"] = "loslab.other.v1"
    with pytest.raises(ValueError, match=f"{name} plan has an unsupported schema"):
        _compose(family, plans)


@pytest.mark.parametrize("name", sorted(COMPONENTS))
def test_each_component_plan_must_not_be_a_non_mapping(family, name) -> None:
    plans = _plans(family)
    plans[name] = "not-a-mapping"
    with pytest.raises(ValueError, match=f"{name} plan has an unsupported schema"):
        _compose(family, plans)


@pytest.mark.parametrize(
    "change",
    [
        {"admission_possible": True},
        {"qualification_status": "QUALIFIED"},
    ],
    ids=["admission", "qualification"],
)
@pytest.mark.parametrize("name", sorted(COMPONENTS))
def test_component_plans_must_stay_non_qualifying(family, name, change) -> None:
    plans = _plans(family)
    plans[name].update(change)
    with pytest.raises(ValueError, match=f"{name} plan must remain explicitly non-qualifying"):
        _compose(family, plans)


@pytest.mark.parametrize("name", [name for name in sorted(COMPONENTS) if name != "joint"])
def test_scenario_provenance_cannot_be_claimed_by_branch_plans(family, name) -> None:
    plans = _plans(family)
    plans[name]["scenario_provenance_verified"] = True
    with pytest.raises(
        ValueError, match=f"{name} plan must not claim verified scenario provenance"
    ):
        _compose(family, plans)


@pytest.mark.parametrize("name", sorted(COMPONENTS))
def test_component_plans_must_name_the_supplied_family(family, name) -> None:
    plans = _plans(family)
    plans[name] = deepcopy(plans[name])
    plans[name]["joint_acceptance_family"] = {"family_id": "other", "family_sha256": "0" * 64}
    with pytest.raises(
        ValueError, match=f"{name} plan does not match the supplied acceptance family"
    ):
        _compose(family, plans)


def test_minimal_v4_plan_composes_with_all_branch_counts_unresolved(family) -> None:
    result = compose_joint_sample_size_design(**_minimal_joint_plan_inputs(family))
    assert result["all_branch_counts_available"] is False
    assert result["component_designs_claim_complete_powered_design"] is False
    assert result["joint_assurance_lower_bound_union_bound"] is None
    assert verify_joint_sample_size_design(result)["complete_precision_powered_design"] is False


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("material_design",), None, "missing material or complex-CSI"),
        (("complex_csi_design",), None, "missing material or complex-CSI"),
        (("single_combined_sample_size_is_reported",), True, "keep sampling units separate"),
        (("complete_sample_size",), 10, "keep sampling units separate"),
        (
            (
                "material_design",
                "required_independent_measurement_units_per_stratum_for_supported_claims",
            ),
            0,
            "must be a positive integer or null",
        ),
        (("propagation_design",), "bad", "propagation_design must be an object or null"),
        (("propagation_design",), None, "include propagation and receiver designs"),
        (("receiver_impact_design",), None, "include propagation and receiver designs"),
        (("material_design", "stratum_count"), 2, "material_calibration stratum count differs"),
        (("complex_csi_design", "stratum_count"), 0, "complex_csi stratum count differs"),
        (("propagation_design", "stratum_count"), 2, "propagation_metrics stratum count differs"),
        (("receiver_impact_design", "stratum_count"), 2, "receiver_impact stratum count differs"),
        (("propagation_design", "strata"), "bad", "propagation_metrics stratum roster is invalid"),
        (
            ("receiver_impact_design", "strata"),
            [["elsewhere", "LOS"]],
            "receiver_impact stratum roster differs",
        ),
    ],
)
def test_composer_checks_joint_plan_structure(family, path, value, message) -> None:
    arguments = _minimal_joint_plan_inputs(family)
    joint_plan = arguments["joint_plan"]
    target = joint_plan
    for part in path[:-1]:
        target = target[part]
    if value is None and len(path) == 1 and path[0] in {"material_design", "complex_csi_design"}:
        joint_plan.pop(path[0])
    else:
        target[path[-1]] = value
    with pytest.raises(ValueError, match=message):
        compose_joint_sample_size_design(**arguments)


def test_composer_preserves_union_bound_blocker_for_low_cross_branch_assurance(family) -> None:
    arguments = deepcopy(_minimal_joint_plan_inputs(family))
    _select_material_assurance_candidate(arguments)
    _select_csi_assurance_candidate(arguments)
    _select_all_remaining_candidates(arguments)
    result = compose_joint_sample_size_design(**arguments)
    assert result["joint_assurance_lower_bound_union_bound"] == pytest.approx(0.6)
    assert any("below the requested joint target" in item for item in result["blockers"])


def test_composer_detects_angular_maximum_conflict_with_selected_units(family) -> None:
    arguments = deepcopy(_minimal_joint_plan_inputs(family))
    joint_plan = arguments["joint_plan"]
    joint_plan["material_design"][
        "required_independent_measurement_units_per_stratum_for_supported_claims"
    ] = 5
    joint_plan["material_design"]["angular_maximum_claim"] = {
        "assumption_supplied": True,
        "compatible_with_material_design": True,
        "maximum_measurement_units_per_stratum": 3,
    }
    _select_material_assurance_candidate(arguments)
    result = compose_joint_sample_size_design(**arguments)
    assert result["material_angular_maximum_assumption"] == {
        "supplied": True,
        "compatible_with_material_design": False,
        "maximum_units_per_stratum": 3,
    }
    assert verify_joint_sample_size_design(result)["complete_precision_powered_design"] is False


def test_composer_keeps_joint_assurance_at_or_above_target(family) -> None:
    arguments = deepcopy(_minimal_joint_plan_inputs(family))
    _select_material_assurance_candidate(arguments)
    _select_csi_assurance_candidate(arguments)
    _select_all_remaining_candidates(arguments)
    plans = (
        arguments["material_assurance_plan"],
        arguments["complex_csi_assurance_plan"],
        arguments["propagation_assurance_plan"],
        arguments["receiver_assurance_plan"],
    )
    for plan in plans:
        candidate = plan["candidate_designs"][0]
        candidate["joint_assurance_lower_bound_union_bound"] = 0.99
        for row in candidate["stratum_assurance"]:
            if isinstance(row, dict):
                row["assurance_one_sided_exact_lower_bound"] = 0.99
                row["joint_assurance_one_sided_exact_lower_bound"] = 0.99
        if isinstance(candidate["stratum_assurance"], dict):
            candidate["stratum_assurance"]["material-1"][
                "joint_assurance_one_sided_exact_lower_bound"
            ] = 0.99
    result = compose_joint_sample_size_design(**arguments)
    assert result["joint_assurance_lower_bound_union_bound"] == pytest.approx(0.96)


def test_composer_records_unresolved_angular_compatibility(family) -> None:
    arguments = deepcopy(_minimal_joint_plan_inputs(family))
    arguments["joint_plan"]["material_design"]["angular_maximum_claim"] = {
        "assumption_supplied": True,
        "compatible_with_material_design": None,
        "maximum_measurement_units_per_stratum": None,
    }
    result = compose_joint_sample_size_design(**arguments)
    assert "compatibility remains unresolved" in " ".join(result["blockers"])


def test_composer_accepts_compatible_angular_assumption_without_a_limit(family) -> None:
    arguments = deepcopy(_minimal_joint_plan_inputs(family))
    arguments["joint_plan"]["material_design"]["angular_maximum_claim"] = {
        "assumption_supplied": True,
        "compatible_with_material_design": True,
        "maximum_measurement_units_per_stratum": None,
    }
    result = compose_joint_sample_size_design(**arguments)
    assert result["material_angular_maximum_assumption"]["compatible_with_material_design"] is True


def test_composer_requires_acceptance_family_v4():
    family = build_joint_acceptance_family_v3(
        family_id="composition-rules-v3",
        material_expected_strata=[("brick", "V", "normal")],
        complex_csi_expected_strata=[("urban", "LOS")],
        propagation_expected_strata=[("urban", "LOS", "total")],
        receiver_impact_expected_strata=[("urban", "LOS")],
    )
    with pytest.raises(ValueError, match="requires acceptance-family v4"):
        compose_joint_sample_size_design(**_minimal_joint_plan_inputs(family))


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (
            ("selected_clusters_per_stratum_among_evaluated_candidates",),
            True,
            "positive integer or null",
        ),
        (
            ("selected_clusters_per_stratum_among_evaluated_candidates",),
            0,
            "positive integer or null",
        ),
        (("target_assurance",), 1.5, r"must be in \(0, 1\)"),
        (("candidate_designs",), "bad", "must be an array"),
        (("candidate_designs",), [], "not a passing candidate design"),
        (
            ("candidate_designs", 0, "meets_target_assurance"),
            False,
            "not a passing candidate design",
        ),
        (
            ("candidate_designs", 0, "joint_assurance_lower_bound_union_bound"),
            0.7,
            "does not meet joint assurance",
        ),
        (
            ("candidate_designs", 0, "joint_assurance_lower_bound_union_bound"),
            True,
            "does not meet joint assurance",
        ),
        (("candidate_designs", 0, "stratum_assurance"), None, "no stratum assurance rows"),
        (("candidate_designs", 0, "stratum_assurance"), [None], "rows must be objects"),
        (
            ("candidate_designs", 0, "stratum_assurance", 0, "condition"),
            "",
            "invalid or duplicate assurance strata",
        ),
        (
            ("candidate_designs", 0, "stratum_assurance", 0, "cluster_count"),
            4,
            "cluster count differs",
        ),
        (
            (
                "candidate_designs",
                0,
                "stratum_assurance",
                0,
                "assurance_one_sided_exact_lower_bound",
            ),
            0.7,
            "bound does not meet target",
        ),
        (
            ("candidate_designs", 0, "stratum_assurance", 0, "condition"),
            "NLOS",
            "assurance strata differ",
        ),
        (
            ("candidate_designs", 0, "joint_assurance_lower_bound_union_bound"),
            0.85,
            "bound is inconsistent",
        ),
    ],
)
def test_selected_assurance_candidate_is_fully_checked(family, path, value, message) -> None:
    arguments = deepcopy(_minimal_joint_plan_inputs(family))
    plan = _select_csi_assurance_candidate(arguments)
    target = plan
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    with pytest.raises(ValueError, match=message):
        compose_joint_sample_size_design(**arguments)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("joint_acceptance_family",), {}, "does not match the supplied acceptance family"),
        (("stratum_roster_by_id",), "bad", "must bind material stratum IDs"),
        (("stratum_roster_by_id", "material-1"), ["brick", "V"], "invalid stratum binding"),
        (("stratum_roster_by_id", "material-1"), ["other", "V", "normal"], "strata differ"),
        (("candidate_designs",), "bad", "candidate_designs must be an array"),
        (("candidate_designs",), [], "not a passing candidate"),
        (("candidate_designs", 0, "meets_target_assurance"), False, "not a passing candidate"),
        (("candidate_designs", 0, "stratum_assurance"), {}, "do not cover the bound stratum IDs"),
        (("candidate_designs", 0, "stratum_assurance", "material-1"), None, "rows must be objects"),
        (
            (
                "candidate_designs",
                0,
                "stratum_assurance",
                "material-1",
                "assumed_support_atom_count",
            ),
            0,
            "bound is invalid",
        ),
        (
            (
                "candidate_designs",
                0,
                "stratum_assurance",
                "material-1",
                "joint_assurance_one_sided_exact_lower_bound",
            ),
            0.7,
            "bound is invalid",
        ),
        (
            ("candidate_designs", 0, "joint_assurance_lower_bound_union_bound"),
            0.85,
            "inconsistent or below target",
        ),
    ],
)
def test_material_assurance_candidate_and_roster_are_fully_checked(
    family, path, value, message
) -> None:
    arguments = deepcopy(_minimal_joint_plan_inputs(family))
    plan = _select_material_assurance_candidate(arguments)
    target = plan
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    with pytest.raises(ValueError, match=message):
        compose_joint_sample_size_design(**arguments)


def test_material_assurance_plan_family_must_match_exactly(family):
    arguments = deepcopy(_minimal_joint_plan_inputs(family))
    plan = _select_material_assurance_candidate(arguments)
    plan["joint_acceptance_family"]["unexpected"] = "extra"
    with pytest.raises(ValueError, match="does not match the supplied acceptance family"):
        compose_joint_sample_size_design(**arguments)


def test_selected_assurance_candidate_rejects_duplicate_stratum_rows(family):
    arguments = deepcopy(_minimal_joint_plan_inputs(family))
    plan = _select_csi_assurance_candidate(arguments)
    row = dict(plan["candidate_designs"][0]["stratum_assurance"][0])
    plan["candidate_designs"][0]["stratum_assurance"].append(row)
    with pytest.raises(ValueError, match="invalid or duplicate assurance strata"):
        compose_joint_sample_size_design(**arguments)


def _rehash_design(design):
    saved = dict(design)
    saved.pop("design_sha256", None)
    design["design_sha256"] = hashlib.sha256(
        json.dumps(
            saved,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("material_angular_maximum_assumption",), None, "declaration is invalid"),
        (("material_angular_maximum_assumption", "supplied"), 1, "supplied flag must be Boolean"),
        (
            ("material_angular_maximum_assumption", "compatible_with_material_design"),
            1,
            "compatibility must be Boolean or null",
        ),
        (
            ("material_angular_maximum_assumption", "maximum_units_per_stratum"),
            True,
            "count must be a positive integer or null",
        ),
        (("component_plans",), None, "does not cover the exact required planning outputs"),
        (("component_plans", "joint"), {}, "component_plans.joint is invalid"),
        (("branch_designs",), None, "does not cover the exact evaluator branches"),
        (("branch_designs", "material_calibration"), None, "must be an object"),
        (("branch_designs", "material_calibration", "unexpected"), True, "fields are invalid"),
        (
            ("branch_designs", "material_calibration", "required_units_by_stratum"),
            [],
            "nonempty stratum roster",
        ),
        (
            ("branch_designs", "material_calibration", "required_units_by_stratum", 0),
            None,
            "invalid stratum row",
        ),
        (
            ("branch_designs", "material_calibration", "required_units_by_stratum", 0, "stratum"),
            [],
            "invalid or duplicate stratum",
        ),
        (
            (
                "branch_designs",
                "material_calibration",
                "required_units_by_stratum",
                0,
                "required_units",
            ),
            0,
            "required_units is invalid",
        ),
        (
            (
                "branch_designs",
                "path_equivalence",
                "required_units_by_stratum",
                0,
                "required_units",
            ),
            2,
            "one-per-episode census",
        ),
        (
            ("branch_designs", "material_calibration", "count_available"),
            True,
            "count_available is inconsistent",
        ),
        (("all_branch_counts_available",), True, "all_branch_counts_available is inconsistent"),
        (("branch_assurance_lower_bounds",), None, "does not cover the evaluator branches"),
        (("branch_assurance_lower_bounds", "complex_csi"), True, "lower bound is invalid"),
        (
            ("joint_assurance_lower_bound_union_bound",),
            0.9,
            "must be null when a branch bound is unresolved",
        ),
        (("target_joint_assurance_lower_bound",), 0.0, r"must be in \(0, 1\)"),
        (("complete_precision_powered_design",), True, "inconsistent with component plans"),
    ],
)
def test_joint_design_verifier_rejects_tampered_nested_fields(family, path, value, message):
    design = compose_joint_sample_size_design(**_minimal_joint_plan_inputs(family))
    target = design
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    _rehash_design(design)
    with pytest.raises(ValueError, match=message):
        verify_joint_sample_size_design(design)


def test_joint_design_verifier_checks_digest_and_exact_schema(family):
    design = compose_joint_sample_size_design(**_minimal_joint_plan_inputs(family))
    with pytest.raises(TypeError, match="must be an object"):
        verify_joint_sample_size_design(None)
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        verify_joint_sample_size_design({**design, "design_sha256": "bad"})
    with pytest.raises(ValueError, match="digest does not match"):
        verify_joint_sample_size_design({**design, "blockers": []})

    extra = dict(design)
    extra["unexpected"] = True
    _rehash_design(extra)
    with pytest.raises(ValueError, match="fields differ from its v1 schema"):
        verify_joint_sample_size_design(extra)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (
            ("material_angular_maximum_assumption", "compatible_with_material_design"),
            True,
            "unsupplied angular-maximum assumptions",
        ),
        (("component_plans", "joint", "sha256"), "ABC", "component_plans.joint is invalid"),
        (
            ("branch_designs", "material_calibration", "required_units_by_stratum", 0, "stratum"),
            [],
            "invalid or duplicate stratum",
        ),
    ],
)
def test_joint_design_verifier_checks_additional_semantic_invariants(family, path, value, message):
    design = compose_joint_sample_size_design(**_minimal_joint_plan_inputs(family))
    target = design
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    _rehash_design(design)
    with pytest.raises(ValueError, match=message):
        verify_joint_sample_size_design(design)


def test_joint_design_verifier_rejects_angular_maximum_conflicting_with_count(family):
    arguments = deepcopy(_minimal_joint_plan_inputs(family))
    joint_plan = arguments["joint_plan"]
    joint_plan["material_design"][
        "required_independent_measurement_units_per_stratum_for_supported_claims"
    ] = 5
    joint_plan["complex_csi_design"][
        "required_independent_scene_clusters_per_stratum_for_supported_claims"
    ] = 5
    joint_plan["propagation_design"][
        "required_independent_clusters_per_stratum_for_supported_claims"
    ] = 5
    _select_material_assurance_candidate(arguments)
    _select_csi_assurance_candidate(arguments)
    _select_all_remaining_candidates(arguments)
    result = compose_joint_sample_size_design(**arguments)
    result["material_angular_maximum_assumption"] = {
        "supplied": True,
        "compatible_with_material_design": True,
        "maximum_units_per_stratum": 3,
    }
    _rehash_design(result)
    with pytest.raises(ValueError, match="conflicts with required counts"):
        verify_joint_sample_size_design(result)


def test_joint_design_verifier_accepts_compatible_angular_assumption_without_a_limit(family):
    result = compose_joint_sample_size_design(**_minimal_joint_plan_inputs(family))
    result["material_angular_maximum_assumption"] = {
        "supplied": True,
        "compatible_with_material_design": True,
        "maximum_units_per_stratum": None,
    }
    _rehash_design(result)
    assert verify_joint_sample_size_design(result)["complete_precision_powered_design"] is False
