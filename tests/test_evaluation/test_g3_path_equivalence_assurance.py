from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path
from typing import Any

import jsonschema
import pytest
from evaluation.g3_joint_acceptance import build_g3_joint_acceptance_family_v4
from evaluation.g3_path_equivalence_assurance import evaluate_path_equivalence_roster


def _comparison(
    status: str, candidate: float | None, challenger: float | None
) -> dict[str, Any]:
    left = Fraction(str(candidate)).limit_denominator(100) if candidate is not None else None
    right = Fraction(str(challenger)).limit_denominator(100) if challenger is not None else None
    matched = math.lcm(
        left.numerator if left is not None and left.numerator else 1,
        right.numerator if right is not None and right.numerator else 1,
    )
    if left is not None and left.numerator == 0:
        matched = 0
    if right is not None and right.numerator == 0:
        matched = 0
    left_count = (
        max(1, matched * left.denominator // left.numerator)
        if left is not None and left.numerator
        else 1
    )
    right_count = (
        max(1, matched * right.denominator // right.numerator)
        if right is not None and right.numerator
        else 1
    )
    if candidate is None:
        left_count = 0
    if challenger is None:
        right_count = 0
    matched_rows = [
        {
            "candidate_receiver_relevance": {"receiver_relevant": True},
            "challenger_receiver_relevance": {"receiver_relevant": True},
            "geometry_and_channel_deltas": {
                "maximum_interaction_vertex_displacement_m": 0.0,
                "absolute_delay_difference_s": 0.0,
                "maximum_endpoint_direction_difference_rad": 0.0,
                "aggregate_class_cfr_e_h": 0.0,
            },
        }
        for _ in range(matched)
    ]
    candidate_only = [
        {"receiver_relevance": {"receiver_relevant": True}}
        for _ in range(max(0, left_count - matched))
    ]
    challenger_only = [
        {"receiver_relevance": {"receiver_relevant": True}}
        for _ in range(max(0, right_count - matched))
    ]
    return {
        "schema": "loslab.g3_path_equivalence_comparison.v1",
        "status": "PATH_COVERAGE_DIAGNOSTIC_NOT_QUALIFIED",
        "admission_possible": False,
        "class_coverage_status": status,
        "coverage_criterion": 0.95,
        "candidate_relevant_class_count": left_count,
        "challenger_relevant_class_count": right_count,
        "candidate_bidirectional_coverage": candidate,
        "challenger_bidirectional_coverage": challenger,
        "matched_class_pair_count": len(matched_rows),
        "matched_classes": matched_rows,
        "candidate_only_relevant_classes": candidate_only,
        "challenger_only_relevant_classes": challenger_only,
    }


def _record(episode: str, result: dict[str, Any], *, condition: str = "LOS") -> dict[str, Any]:
    return {
        "family_id": "site-family-a",
        "condition": condition,
        "episode_id": episode,
        "comparison": result,
    }


def _raw_path(index: int) -> dict[str, Any]:
    return {
        "valid": True,
        "path_kind": "specular",
        "path_index": index,
        "interaction_type_masks": [1],
        "object_ids": [7],
        "primitive_ids": [2],
        "interaction_vertices_m": [[1.0, 2.0, 3.0]],
        "absolute_delay_s": 1e-7,
        "theta_t_rad": math.pi / 2,
        "phi_t_rad": 0.1,
        "theta_r_rad": math.pi / 2,
        "phi_r_rad": 0.2,
        "coefficient_real": 0.05,
        "coefficient_imag": 0.0,
        "cfr_contribution_real": [0.05, 0.05],
        "cfr_contribution_imag": [0.0, 0.0],
    }


def _raw_record(episode: str = "episode-1") -> dict[str, Any]:
    return {
        "family_id": "site-family-a",
        "condition": "LOS",
        "episode_id": episode,
        "candidate_paths": [_raw_path(0)],
        "challenger_paths": [_raw_path(0)],
        "candidate_full_cfr_real": [0.05, 0.05],
        "candidate_full_cfr_imag": [0.0, 0.0],
        "challenger_full_cfr_real": [0.05, 0.05],
        "challenger_full_cfr_imag": [0.0, 0.0],
    }


def _raw_configuration() -> dict[str, Any]:
    return {
        "candidate_configuration_id": "candidate-a",
        "challenger_configuration_id": "challenger-a",
        "vertex_tolerance_m": 0.01,
        "delay_tolerance_s": 1e-9,
        "direction_tolerance_rad": 0.01,
        "h_floor": 1e-12,
        "transmit_power_per_re_w": 1.0,
        "noise_power_per_re_w": 1e-6,
    }


def _solver_budget() -> dict[str, Any]:
    return {
        "max_rays_per_link": 1000,
        "max_interactions": 8,
        "max_episodes": 20,
        "max_wall_seconds": 60.0,
        "overflow_policy": "record_censored_episode",
    }


def test_raw_roster_recomputes_comparison_and_retains_raw_input_digest() -> None:
    report = evaluate_path_equivalence_roster(
        [_raw_record()],
        expected_scenarios=[("site-family-a", "LOS", "episode-1")],
        comparison_configuration=_raw_configuration(),
    )

    assert report["scenarios"][0]["class_coverage_status"] == "PASS"
    assert len(report["scenarios"][0]["raw_path_input_sha256"]) == 64


def test_raw_roster_applies_frozen_relative_coefficient_tolerance() -> None:
    record = _raw_record()
    record["challenger_paths"][0]["coefficient_real"] = 0.04
    record["challenger_paths"][0]["cfr_contribution_real"] = [0.04, 0.04]
    record["challenger_full_cfr_real"] = [0.04, 0.04]

    report = evaluate_path_equivalence_roster(
        [record],
        expected_scenarios=[("site-family-a", "LOS", "episode-1")],
        comparison_configuration=_raw_configuration(),
        coefficient_relative_tolerance=0.05,
    )

    assert report["scenarios"][0]["class_coverage_status"] == "FAIL"
    assert report["status"] == "ROSTER_PATH_COVERAGE_DIAGNOSTIC_FAIL"


def test_raw_roster_accepts_coefficients_within_frozen_relative_tolerance() -> None:
    report = evaluate_path_equivalence_roster(
        [_raw_record()],
        expected_scenarios=[("site-family-a", "LOS", "episode-1")],
        comparison_configuration=_raw_configuration(),
        coefficient_relative_tolerance=0.0,
    )

    assert report["scenarios"][0]["class_coverage_status"] == "PASS"
    assert report["status"] == "ROSTER_PATH_COVERAGE_DIAGNOSTIC_PASS"


def test_budget_bound_roster_separates_diffuse_distribution_from_path_identity() -> None:
    record = _raw_record()
    for side in ("candidate", "challenger"):
        diffuse = dict(record[f"{side}_paths"][0])
        diffuse.update(
            {
                "path_kind": "diffuse_stochastic",
                "absolute_delay_s": 2e-7,
                "coefficient_real": 0.025,
                "cfr_contribution_real": [0.025, 0.025],
            }
        )
        record[f"{side}_paths"].append(diffuse)
        record[f"{side}_full_cfr_real"] = [0.075, 0.075]

    report = evaluate_path_equivalence_roster(
        [record],
        expected_scenarios=[("site-family-a", "LOS", "episode-1")],
        comparison_configuration=_raw_configuration(),
        solver_budget_provenance=_solver_budget(),
    )

    scenario = report["scenarios"][0]
    diffuse_report = scenario["diffuse_distribution"]
    assert report["schema"] == "loslab.g3_path_equivalence_roster_diagnostic.v2"
    assert report["admission_possible"] is False
    assert report["diffuse_path_diagnostics_are_descriptive_only"] is True
    assert scenario["class_coverage_status"] == "PASS"
    assert diffuse_report["ray_identity_assignment"] is False
    assert diffuse_report["candidate"]["path_count"] == 1
    assert diffuse_report["candidate"]["unweighted_quantiles"]["delay_s"]["q50"] == 2e-7
    assert diffuse_report["finite_solver_caps_are_ground_truth"] is False
    schema_path = (
        Path(__file__).resolve().parents[2]
        / "datasets/schema/g3-path-equivalence-roster-diagnostic.v2.schema.json"
    )
    jsonschema.Draft202012Validator(
        json.loads(schema_path.read_text(encoding="utf-8"))
    ).validate(report)


def test_solver_budget_cannot_be_attached_to_legacy_precomputed_comparisons() -> None:
    with pytest.raises(ValueError, match="require raw paths"):
        evaluate_path_equivalence_roster(
            [_record("episode-1", _comparison("PASS", 1.0, 1.0))],
            expected_scenarios=[("site-family-a", "LOS", "episode-1")],
            solver_budget_provenance=_solver_budget(),
        )


def test_raw_roster_rejects_supplied_comparison_and_unbound_fields() -> None:
    record = _raw_record()
    record["comparison"] = _comparison("PASS", 1.0, 1.0)
    with pytest.raises(ValueError, match="must not include a precomputed comparison"):
        evaluate_path_equivalence_roster(
            [record],
            expected_scenarios=[("site-family-a", "LOS", "episode-1")],
            comparison_configuration=_raw_configuration(),
        )

    record = _raw_record()
    record["unbound_annotation"] = "must not be silently ignored"
    with pytest.raises(ValueError, match="exactly the frozen path and full-CFR fields"):
        evaluate_path_equivalence_roster(
            [record],
            expected_scenarios=[("site-family-a", "LOS", "episode-1")],
            comparison_configuration=_raw_configuration(),
        )


def test_roster_requires_each_independent_episode_to_pass_both_directions() -> None:
    result = evaluate_path_equivalence_roster(
        [
            _record("episode-1", _comparison("PASS", 0.95, 1.0)),
            _record("episode-2", _comparison("PASS", 1.0, 0.96)),
        ],
        expected_scenarios=[
            ("site-family-a", "LOS", "episode-1"),
            ("site-family-a", "LOS", "episode-2"),
        ],
    )

    assert result["all_expected_episodes_pass"] is True
    assert result["admission_possible"] is False
    assert result["strata"] == [
        {
            "family_id": "site-family-a",
            "condition": "LOS",
            "episodes": 2,
            "coverage_pass": 2,
            "coverage_fail": 0,
            "unresolved": 0,
        }
    ]


def test_roster_does_not_pool_coverage_to_hide_one_episode_failure() -> None:
    result = evaluate_path_equivalence_roster(
        [
            _record("episode-1", _comparison("PASS", 1.0, 1.0)),
            _record("episode-2", _comparison("FAIL", 0.0, 0.0)),
        ],
        expected_scenarios=[
            ("site-family-a", "LOS", "episode-1"),
            ("site-family-a", "LOS", "episode-2"),
        ],
    )

    assert result["all_expected_episodes_pass"] is False
    assert result["status"] == "ROSTER_PATH_COVERAGE_DIAGNOSTIC_FAIL"
    assert result["strata"][0]["coverage_fail"] == 1


def test_roster_preserves_no_class_case_as_unresolved_diagnostic() -> None:
    comparison = {
        "schema": "loslab.g3_path_equivalence_comparison.v1",
        "status": "NO_CLASSES_NOT_ASSESSABLE",
        "admission_possible": False,
        "class_coverage_status": "NOT_ASSESSABLE_NO_CLASSES",
        "matched_classes": [],
        "candidate_only_relevant_classes": [],
        "challenger_only_relevant_classes": [],
    }
    result = evaluate_path_equivalence_roster(
        [_record("episode-1", comparison)],
        expected_scenarios=[("site-family-a", "LOS", "episode-1")],
    )

    assert result["status"] == "ROSTER_PATH_COVERAGE_DIAGNOSTIC_UNRESOLVED"
    assert result["all_expected_episodes_pass"] is False
    assert result["scenarios"][0]["class_coverage_status"] == "NOT_ASSESSABLE_NO_CLASSES"


def test_roster_rejects_missing_duplicate_or_misreported_episode() -> None:
    expected = [
        ("site-family-a", "LOS", "episode-1"),
        ("site-family-a", "NLOS", "episode-2"),
    ]
    with pytest.raises(ValueError, match="roster differs"):
        evaluate_path_equivalence_roster(
            [_record("episode-1", _comparison("PASS", 1.0, 1.0))],
            expected_scenarios=expected,
        )
    with pytest.raises(ValueError, match="duplicate path-equivalence episode"):
        evaluate_path_equivalence_roster(
            [_record("episode-1", _comparison("PASS", 1.0, 1.0))] * 2,
            expected_scenarios=[expected[0]],
        )
    with pytest.raises(ValueError, match="claims PASS below"):
        evaluate_path_equivalence_roster(
            [_record("episode-1", _comparison("PASS", 0.94, 1.0))],
            expected_scenarios=[expected[0]],
        )
    with pytest.raises(ValueError, match="scenario record must be an object"):
        evaluate_path_equivalence_roster(
            [None],  # type: ignore[list-item]
            expected_scenarios=[expected[0]],
        )


def test_path_roster_rejects_padded_scenario_and_configuration_identities() -> None:
    with pytest.raises(ValueError, match="expected scenarios"):
        evaluate_path_equivalence_roster(
            [_record("episode-1", _comparison("PASS", 1.0, 1.0))],
            expected_scenarios=[("site-family-a", "LOS", " episode-1")],
        )

    padded_row = _record("episode-1 ", _comparison("PASS", 1.0, 1.0))
    with pytest.raises(ValueError, match="canonical strings"):
        evaluate_path_equivalence_roster(
            [padded_row],
            expected_scenarios=[("site-family-a", "LOS", "episode-1")],
        )

    configuration = _raw_configuration()
    configuration["candidate_configuration_id"] = " candidate-a"
    with pytest.raises(ValueError, match="canonical string"):
        evaluate_path_equivalence_roster(
            [_raw_record()],
            expected_scenarios=[("site-family-a", "LOS", "episode-1")],
            comparison_configuration=configuration,
            solver_budget_provenance=_solver_budget(),
        )


def test_roster_rejects_internally_inconsistent_comparison_diagnostics() -> None:
    comparison = _comparison("PASS", 1.0, 1.0)
    comparison["candidate_bidirectional_coverage"] = 0.99
    with pytest.raises(ValueError, match="disagrees with class counts"):
        evaluate_path_equivalence_roster(
            [_record("episode-1", comparison)],
            expected_scenarios=[("site-family-a", "LOS", "episode-1")],
        )

    comparison = _comparison("PASS", 1.0, 1.0)
    comparison["candidate_relevant_class_count"] += 1
    with pytest.raises(ValueError, match="count disagrees with its roster"):
        evaluate_path_equivalence_roster(
            [_record("episode-1", comparison)],
            expected_scenarios=[("site-family-a", "LOS", "episode-1")],
        )

    comparison = _comparison("PASS", 1.0, 1.0)
    comparison["coverage_criterion"] = 0.90
    with pytest.raises(ValueError, match="different coverage criterion"):
        evaluate_path_equivalence_roster(
            [_record("episode-1", comparison)],
            expected_scenarios=[("site-family-a", "LOS", "episode-1")],
        )


@pytest.mark.parametrize(
    ("field", "value", "remove_field"),
    [
        ("maximum_interaction_vertex_displacement_m", None, True),
        ("absolute_delay_difference_s", -1.0, False),
        ("maximum_endpoint_direction_difference_rad", True, False),
        ("aggregate_class_cfr_e_h", "0.1", False),
        ("absolute_delay_difference_s", float("nan"), False),
        ("maximum_endpoint_direction_difference_rad", float("inf"), False),
    ],
)
def test_roster_rejects_missing_or_malformed_matched_class_deltas(
    field: str, value: Any, remove_field: bool
) -> None:
    comparison = _comparison("PASS", 1.0, 1.0)
    deltas = comparison["matched_classes"][0]["geometry_and_channel_deltas"]
    if remove_field:
        del deltas[field]
    else:
        deltas[field] = value

    with pytest.raises(ValueError, match="matched-class delta"):
        evaluate_path_equivalence_roster(
            [_record("episode-1", comparison)],
            expected_scenarios=[("site-family-a", "LOS", "episode-1")],
        )


def test_roster_verifies_and_reports_joint_family_reference() -> None:
    expected = [("site-family-a", "LOS", "episode-1")]
    family = build_g3_joint_acceptance_family_v4(
        family_id="g3-v4",
        material_expected_strata=[("concrete", "V", "20deg")],
        complex_csi_expected_strata=[("site-family-a", "LOS")],
        propagation_expected_strata=[("site-family-a", "LOS", "total")],
        receiver_impact_expected_strata=[("site-family-a", "LOS")],
        path_equivalence_expected_scenarios=expected,
    )
    result = evaluate_path_equivalence_roster(
        [_record("episode-1", _comparison("PASS", 0.95, 1.0))],
        expected_scenarios=expected,
        joint_acceptance_family=family,
    )

    assert result["joint_acceptance_family"] == {
        "family_id": "g3-v4",
        "family_sha256": family["family_sha256"],
    }


def test_roster_rejects_a_joint_family_with_a_different_path_roster() -> None:
    expected = [("site-family-a", "LOS", "episode-1")]
    family = build_g3_joint_acceptance_family_v4(
        family_id="g3-v4",
        material_expected_strata=[("concrete", "V", "20deg")],
        complex_csi_expected_strata=[("site-family-a", "LOS")],
        propagation_expected_strata=[("site-family-a", "LOS", "total")],
        receiver_impact_expected_strata=[("site-family-a", "LOS")],
        path_equivalence_expected_scenarios=[("site-family-a", "NLOS", "episode-1")],
    )
    with pytest.raises(ValueError, match="differs from joint family v4"):
        evaluate_path_equivalence_roster(
            [_record("episode-1", _comparison("PASS", 1.0, 1.0))],
            expected_scenarios=expected,
            joint_acceptance_family=family,
        )


def test_roster_report_matches_its_versioned_json_schema() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    result = evaluate_path_equivalence_roster(
        [_record("episode-1", _comparison("PASS", 1.0, 1.0))],
        expected_scenarios=[("site-family-a", "LOS", "episode-1")],
    )
    schema_path = (
        Path(__file__).parents[2]
        / "datasets"
        / "schema"
        / "g3-path-equivalence-roster-diagnostic.v2.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)

    jsonschema.Draft202012Validator(schema).validate(result)
