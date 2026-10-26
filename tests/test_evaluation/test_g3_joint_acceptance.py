from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from evaluation.g3_joint_acceptance import (
    build_g3_joint_acceptance_family,
    build_g3_joint_acceptance_family_v2,
    build_g3_joint_acceptance_family_v3,
    build_g3_joint_acceptance_family_v4,
    family_alpha_for_claim,
    validate_g3_joint_acceptance_family,
)
from evaluation.g3_sample_size_planning import plan_g3_joint_independent_units


def test_joint_family_is_order_independent_hash_bound_and_explicit():
    first = build_g3_joint_acceptance_family(
        family_id="protocol-1",
        material_expected_strata=[("concrete", "V", "40deg"), ("concrete", "V", "20deg")],
        complex_csi_expected_strata=[("site-b", "NLOS"), ("site-a", "LOS")],
    )
    reordered = build_g3_joint_acceptance_family(
        family_id="protocol-1",
        material_expected_strata=[("concrete", "V", "20deg"), ("concrete", "V", "40deg")],
        complex_csi_expected_strata=[("site-a", "LOS"), ("site-b", "NLOS")],
    )

    assert first == reordered
    assert first["claim_count"] == 14
    assert first["familywise_alpha"] == 0.05
    assert first["familywise_confidence_level"] == 0.95
    assert first["per_claim_alpha"] == pytest.approx(0.05 / 14)
    assert len(first["claim_roster"]) == first["claim_count"]
    assert validate_g3_joint_acceptance_family(first) == first


@pytest.mark.parametrize(
    ("material", "csi", "message"),
    [
        (
            [("concrete", "V", "20deg"), ("concrete", "V", "20deg")],
            [("site", "LOS")],
            "duplicate material",
        ),
        ([("concrete", "X", "20deg")], [("site", "LOS")], "polarization"),
        ([("concrete", "V", "20deg")], [("site", "BOTH")], "condition"),
    ],
)
def test_joint_family_rejects_ambiguous_or_invalid_strata(material, csi, message):
    with pytest.raises(ValueError, match=message):
        build_g3_joint_acceptance_family(
            family_id="protocol-1",
            material_expected_strata=material,
            complex_csi_expected_strata=csi,
        )


@pytest.mark.parametrize(
    ("family_id", "material", "csi", "path_scenario"),
    [
        (" protocol-1", [("concrete", "V", "20deg")], [("site", "LOS")], None),
        ("protocol-1", [(" concrete", "V", "20deg")], [("site", "LOS")], None),
        ("protocol-1", [("concrete", "V", "20deg")], [("site ", "LOS")], None),
        (
            "protocol-1",
            [("concrete", "V", "20deg")],
            [("site", "LOS")],
            ("site", "LOS", " episode-1"),
        ),
    ],
)
def test_joint_family_rejects_padded_identity_labels(
    family_id, material, csi, path_scenario
):
    kwargs: dict[str, Any] = {
        "family_id": family_id,
        "material_expected_strata": material,
        "complex_csi_expected_strata": csi,
    }
    if path_scenario is None:
        with pytest.raises(ValueError, match="canonical string"):
            build_g3_joint_acceptance_family(**kwargs)
    else:
        with pytest.raises(ValueError, match="canonical strings"):
            build_g3_joint_acceptance_family_v4(
                **kwargs,
                propagation_expected_strata=[("site", "LOS", "total")],
                receiver_impact_expected_strata=[("site", "LOS")],
                path_equivalence_expected_scenarios=[path_scenario],
            )


def test_joint_family_validator_rejects_changed_claim_roster():
    family = build_g3_joint_acceptance_family(
        family_id="protocol-1",
        material_expected_strata=[("concrete", "V", "20deg")],
        complex_csi_expected_strata=[("site", "LOS")],
    )
    tampered = deepcopy(family)
    tampered["claim_roster"].pop()

    with pytest.raises(ValueError, match="fields or canonical SHA-256 are inconsistent"):
        validate_g3_joint_acceptance_family(tampered)


def test_v2_joint_family_includes_propagation_claims_and_preserves_shared_alpha():
    family = build_g3_joint_acceptance_family_v2(
        family_id="protocol-v2",
        material_expected_strata=[("concrete", "V", "20deg")],
        complex_csi_expected_strata=[("site-a", "LOS")],
        propagation_expected_strata=[
            ("site-a", "LOS", "total"),
            ("site-a", "LOS", "specular"),
        ],
    )

    assert family["schema"] == "loslab.g3_joint_acceptance_family.v2"
    assert family["claim_count"] == 23
    assert family["claim_count"] == len(family["claim_roster"])
    assert (
        len([item for item in family["claim_roster"] if item["evaluator"] == "propagation_metrics"])
        == 16
    )
    for evaluator, strata in (
        ("material_calibration", family["material_expected_strata"]),
        ("complex_csi", family["complex_csi_expected_strata"]),
        ("propagation_metrics", family["propagation_expected_strata"]),
    ):
        alpha, count = family_alpha_for_claim(family, evaluator=evaluator, expected_strata=strata)
        assert alpha == pytest.approx(0.05 / 23)
        assert count == 23
    assert validate_g3_joint_acceptance_family(family) == family


def test_v2_joint_family_rejects_propagation_roster_tampering():
    family = build_g3_joint_acceptance_family_v2(
        family_id="protocol-v2",
        material_expected_strata=[("concrete", "V", "20deg")],
        complex_csi_expected_strata=[("site-a", "LOS")],
        propagation_expected_strata=[("site-a", "LOS", "total")],
    )
    tampered = deepcopy(family)
    tampered["propagation_expected_strata"].append(["site-b", "NLOS", "diffuse"])

    with pytest.raises(ValueError, match="fields or canonical SHA-256 are inconsistent"):
        validate_g3_joint_acceptance_family(tampered)


def test_v3_joint_family_includes_receiver_intervals_in_one_shared_roster():
    family = build_g3_joint_acceptance_family_v3(
        family_id="protocol-v3",
        material_expected_strata=[("concrete", "V", "20deg")],
        complex_csi_expected_strata=[("site-a", "LOS")],
        propagation_expected_strata=[("site-a", "LOS", "total")],
        receiver_impact_expected_strata=[("site-a", "LOS")],
    )

    assert family["schema"] == "loslab.g3_joint_acceptance_family.v3"
    assert family["claim_count"] == 19
    assert len(family["claim_roster"]) == 19
    receiver_claims = [
        item for item in family["claim_roster"] if item["evaluator"] == "receiver_impact"
    ]
    assert len(receiver_claims) == 4
    assert validate_g3_joint_acceptance_family(family) == family

    alpha, count = family_alpha_for_claim(
        family,
        evaluator="receiver_impact",
        expected_strata=family["receiver_impact_expected_strata"],
    )
    assert alpha == pytest.approx(0.05 / 19)
    assert count == 19


def test_v3_family_rejects_receiver_strata_not_declared_by_the_contract():
    family = build_g3_joint_acceptance_family_v3(
        family_id="protocol-v3",
        material_expected_strata=[("concrete", "V", "20deg")],
        complex_csi_expected_strata=[("site-a", "LOS")],
        propagation_expected_strata=[("site-a", "LOS", "total")],
        receiver_impact_expected_strata=[("site-a", "LOS")],
    )

    with pytest.raises(ValueError, match="expected strata differ"):
        family_alpha_for_claim(
            family,
            evaluator="receiver_impact",
            expected_strata=[("site-b", "NLOS")],
        )


def test_joint_sample_size_planner_uses_all_claims_and_keeps_units_separate():
    family = build_g3_joint_acceptance_family(
        family_id="protocol-1",
        material_expected_strata=[("concrete", "V", "20deg"), ("concrete", "H", "20deg")],
        complex_csi_expected_strata=[("site-a", "LOS"), ("site-a", "NLOS")],
    )
    report = plan_g3_joint_independent_units(
        joint_acceptance_family=family,
        planned_quantile_pass_probabilities={
            "median_absolute_power_error_db": 0.9,
            "p95_absolute_power_error_db": 0.99,
            "median_prediction_interval_width_db": 0.9,
        },
        planned_interval_coverage=0.95,
        planned_complex_csi_p95_pass_probability=0.99,
        planned_complex_csi_failure_probability=0.01,
        planned_angular_unit_pass_probability=0.999999,
    )

    assert family["claim_count"] == 14
    assert report["joint_acceptance_family"]["claim_count"] == 14
    assert report["joint_acceptance_family"]["per_claim_alpha_bonferroni"] == pytest.approx(
        0.05 / 14
    )
    assert report["material_design"]["stratum_count"] == 2
    assert report["complex_csi_design"]["stratum_count"] == 2
    assert report["sampling_units_are_distinct"] is True
    assert report["single_combined_sample_size_is_reported"] is False
    assert report["complete_g3_sample_size"] is None


def test_joint_sample_size_planner_rejects_unfavorable_csi_p95_assumption():
    family = build_g3_joint_acceptance_family(
        family_id="protocol-1",
        material_expected_strata=[("concrete", "V", "20deg")],
        complex_csi_expected_strata=[("site-a", "LOS")],
    )
    with pytest.raises(ValueError, match="pass probability must exceed"):
        plan_g3_joint_independent_units(
            joint_acceptance_family=family,
            planned_quantile_pass_probabilities={
                "median_absolute_power_error_db": 0.9,
                "p95_absolute_power_error_db": 0.99,
                "median_prediction_interval_width_db": 0.9,
            },
            planned_interval_coverage=0.95,
            planned_complex_csi_p95_pass_probability=0.95,
            planned_complex_csi_failure_probability=0.01,
            planned_angular_unit_pass_probability=None,
        )


def test_v4_joint_family_binds_path_roster_without_counting_inferential_claims():
    kwargs = {
        "family_id": "protocol-v4",
        "material_expected_strata": [("concrete", "V", "20deg")],
        "complex_csi_expected_strata": [("site-a", "LOS")],
        "propagation_expected_strata": [("site-a", "LOS", "total")],
        "receiver_impact_expected_strata": [("site-a", "LOS")],
        "path_equivalence_expected_scenarios": [
            ("site-a", "LOS", "episode-2"),
            ("site-a", "LOS", "episode-1"),
        ],
    }
    family = build_g3_joint_acceptance_family_v4(**kwargs)
    reordered = build_g3_joint_acceptance_family_v4(
        **{
            **kwargs,
            "path_equivalence_expected_scenarios": list(
                reversed(kwargs["path_equivalence_expected_scenarios"])
            ),
        }
    )

    assert family == reordered
    assert family["schema"] == "loslab.g3_joint_acceptance_family.v4"
    assert family["claim_count"] == 19
    assert family["per_claim_alpha"] == pytest.approx(0.05 / 19)
    assert family["path_equivalence_bidirectional_coverage_min"] == 0.95
    assert family["path_equivalence_admission_possible"] is False
    assert family["path_equivalence_expected_scenarios"] == [
        ["site-a", "LOS", "episode-1"],
        ["site-a", "LOS", "episode-2"],
    ]
    assert validate_g3_joint_acceptance_family(family) == family


def test_joint_family_validator_rejects_boolean_integer_type_substitution():
    family = build_g3_joint_acceptance_family_v4(
        family_id="protocol-v4",
        material_expected_strata=[("concrete", "V", "20deg")],
        complex_csi_expected_strata=[("site-a", "LOS")],
        propagation_expected_strata=[("site-a", "LOS", "total")],
        receiver_impact_expected_strata=[("site-a", "LOS")],
        path_equivalence_expected_scenarios=[("site-a", "LOS", "episode-1")],
    )
    tampered = deepcopy(family)
    # bool is a subclass of int in Python, so normal dict equality treats
    # this as equal to the declared False while leaving the original digest.
    tampered["path_equivalence_admission_possible"] = 0

    with pytest.raises(ValueError, match="fields or canonical SHA-256 are inconsistent"):
        validate_g3_joint_acceptance_family(tampered)


def test_v4_joint_family_rejects_path_roster_tampering_and_duplicates():
    kwargs = {
        "family_id": "protocol-v4",
        "material_expected_strata": [("concrete", "V", "20deg")],
        "complex_csi_expected_strata": [("site-a", "LOS")],
        "propagation_expected_strata": [("site-a", "LOS", "total")],
        "receiver_impact_expected_strata": [("site-a", "LOS")],
        "path_equivalence_expected_scenarios": [("site-a", "LOS", "episode-1")],
    }
    family = build_g3_joint_acceptance_family_v4(**kwargs)
    tampered = deepcopy(family)
    tampered["path_equivalence_expected_scenarios"].append(
        ["site-a", "NLOS", "episode-2"]
    )
    with pytest.raises(ValueError, match="fields or canonical SHA-256 are inconsistent"):
        validate_g3_joint_acceptance_family(tampered)
    with pytest.raises(ValueError, match="duplicate path-equivalence"):
        build_g3_joint_acceptance_family_v4(
            **{
                **kwargs,
                "path_equivalence_expected_scenarios": [
                    ("site-a", "LOS", "episode-1"),
                    ("site-a", "LOS", "episode-1"),
                ],
            }
        )
