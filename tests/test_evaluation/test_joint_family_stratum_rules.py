"""Joint acceptance family: stratum vocabularies, roster matching and evaluator routing."""

from __future__ import annotations

import pytest
from evaluation.acceptance.joint_acceptance import (
    build_joint_acceptance_family,
    build_joint_acceptance_family_v3,
    build_joint_acceptance_family_v4,
    family_alpha_for_claim,
    validate_joint_acceptance_family,
)


def _v2():
    return build_joint_acceptance_family(
        family_id="stratum-rules",
        material_expected_strata=[("brick", "V", "normal")],
        complex_csi_expected_strata=[("urban", "LOS")],
    )


def _v4(**overrides):
    kwargs = {
        "family_id": "stratum-rules-v4",
        "material_expected_strata": [("brick", "V", "normal")],
        "complex_csi_expected_strata": [("urban", "LOS")],
        "propagation_expected_strata": [("urban", "LOS", "total")],
        "receiver_impact_expected_strata": [("urban", "LOS")],
        "path_equivalence_expected_scenarios": [("urban", "LOS", "episode-1")],
    }
    kwargs.update(overrides)
    return build_joint_acceptance_family_v4(**kwargs)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        pytest.param(
            {"material_expected_strata": []},
            "material expected strata must be a nonempty list",
            id="empty",
        ),
        pytest.param(
            {"material_expected_strata": [("brick", "V")]}, "must contain 3 strings", id="width"
        ),
        pytest.param(
            {"material_expected_strata": [("brick", "X", "normal")]},
            "polarization must be V or H",
            id="polarization",
        ),
        pytest.param(
            {"material_expected_strata": [(" brick", "V", "normal")]},
            "nonempty canonical strings",
            id="padded",
        ),
        pytest.param(
            {"material_expected_strata": [("brick", "V", "normal"), ("brick", "V", "normal")]},
            "duplicate material",
            id="duplicate",
        ),
        pytest.param(
            {"complex_csi_expected_strata": [("urban", "DAY")]},
            "must be LOS or NLOS",
            id="csi-condition",
        ),
        pytest.param(
            {"receiver_impact_expected_strata": [("urban", "DAY")]},
            "receiver-impact stratum condition",
            id="receiver-condition",
        ),
        pytest.param(
            {"propagation_expected_strata": [("urban", "DAY", "total")]},
            "propagation-metric stratum condition",
            id="propagation-condition",
        ),
        pytest.param(
            {"propagation_expected_strata": [("urban", "LOS", "all")]},
            "component must be total, specular, or diffuse",
            id="propagation-component",
        ),
        pytest.param(
            {"path_equivalence_expected_scenarios": []},
            "path-equivalence expected scenarios must be nonempty",
            id="scenarios-empty",
        ),
    ],
)
def test_v4_family_rejects_invalid_stratum_vocabularies(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        _v4(**overrides)


def test_family_must_be_an_object_for_validation() -> None:
    with pytest.raises(ValueError, match="joint_acceptance_family must be an object"):
        validate_joint_acceptance_family(["not", "object"])


def test_family_alpha_routes_each_evaluator_to_its_declared_roster() -> None:
    family = _v2()
    alpha, claims = family_alpha_for_claim(
        family, evaluator="material_calibration", expected_strata=[("brick", "V", "normal")]
    )
    assert alpha > 0 and claims >= 1
    alpha, claims = family_alpha_for_claim(
        family, evaluator="complex_csi", expected_strata=[("urban", "LOS")]
    )
    assert alpha > 0 and claims >= 1
    with pytest.raises(ValueError, match="unsupported G3 evaluator"):
        family_alpha_for_claim(family, evaluator="unknown", expected_strata=[])
    with pytest.raises(ValueError, match="expected strata differ"):
        family_alpha_for_claim(family, evaluator="complex_csi", expected_strata=[("urban", "NLOS")])


def test_v4_family_routes_propagation_and_receiver_rosters() -> None:
    family = _v4()
    alpha, _ = family_alpha_for_claim(
        family, evaluator="propagation_metrics", expected_strata=[("urban", "LOS", "total")]
    )
    assert alpha > 0
    alpha, _ = family_alpha_for_claim(
        family, evaluator="receiver_impact", expected_strata=[("urban", "LOS")]
    )
    assert alpha > 0
    with pytest.raises(ValueError, match="expected strata differ"):
        family_alpha_for_claim(
            family, evaluator="receiver_impact", expected_strata=[("urban", "NLOS")]
        )


def test_older_families_do_not_route_receiver_or_propagation_claims() -> None:
    family = _v2()
    with pytest.raises(ValueError, match="unsupported G3 evaluator"):
        family_alpha_for_claim(
            family, evaluator="receiver_impact", expected_strata=[("urban", "LOS")]
        )
    with pytest.raises(ValueError, match="unsupported G3 evaluator"):
        family_alpha_for_claim(
            family, evaluator="propagation_metrics", expected_strata=[("urban", "LOS", "total")]
        )


def test_v3_family_builds_and_routes_receiver_claims() -> None:
    family = build_joint_acceptance_family_v3(
        family_id="v3-rules",
        material_expected_strata=[("brick", "V", "normal")],
        complex_csi_expected_strata=[("urban", "LOS")],
        propagation_expected_strata=[("urban", "LOS", "total")],
        receiver_impact_expected_strata=[("urban", "LOS")],
    )
    alpha, _ = family_alpha_for_claim(
        family, evaluator="receiver_impact", expected_strata=[("urban", "LOS")]
    )
    assert alpha > 0
