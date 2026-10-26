from __future__ import annotations

from copy import deepcopy
from typing import Any, cast

import numpy as np
import pytest
from evaluation.g3_joint_acceptance import (
    build_g3_joint_acceptance_family_v3,
    build_g3_joint_acceptance_family_v4,
)

from evaluation import g3_joint_evaluation as joint


def _inputs() -> dict[str, object]:
    family = build_g3_joint_acceptance_family_v4(
        family_id="preregistered-g3-family",
        material_expected_strata=[("concrete", "V", "20deg")],
        complex_csi_expected_strata=[("scene-a", "LOS")],
        propagation_expected_strata=[("scene-a", "LOS", "total")],
        receiver_impact_expected_strata=[("scene-a", "LOS")],
        path_equivalence_expected_scenarios=[("scene-a", "LOS", "episode-a")],
    )
    return {
        "schema": "loslab.g3_joint_evaluator_inputs.v1",
        "joint_acceptance_family": family,
        "material_calibration": {
            "payload": {"joint_acceptance_family": family, "scorer_input": "opaque"}
        },
        "complex_csi": {
            "records": [
                {
                    "family_id": "scene-a",
                    "condition": "LOS",
                    "episode_id": "episode-a",
                    "cluster_id": "cluster-a",
                    "evaluation_grid_id": "grid-a",
                    "candidate_configuration_id": "candidate-a",
                    "challenger_configuration_id": "challenger-a",
                    "h_floor": 1e-12,
                    "candidate_cfr": np.asarray([1.0 + 0.0j]),
                    "challenger_cfr": np.asarray([1.0 + 0.0j]),
                }
            ],
            "expected_strata": [["scene-a", "LOS"]],
        },
        "propagation_metrics": {
            "records": [
                {
                    "family_id": "scene-a",
                    "condition": "LOS",
                    "episode_id": "episode-a",
                }
            ],
            "expected_strata": [["scene-a", "LOS", "total"]],
            "effective_bandwidth_hz": 20e6,
        },
        "receiver_impact": {
            "records": [{"block_id": "block-a"}],
            "expected_strata": [["scene-a", "LOS"]],
            "candidate_config_id": "candidate-a",
            "challenger_config_id": "challenger-a",
            "snr_grid_db": [0.0, 5.0],
            "comparison_snr_db": 0.0,
        },
        "path_equivalence": {
            "records": [
                {
                    "family_id": "scene-a",
                    "condition": "LOS",
                    "episode_id": "episode-a",
                }
            ],
            "expected_scenarios": [["scene-a", "LOS", "episode-a"]],
        },
    }


def _inputs_v3() -> dict[str, Any]:
    payload = cast(dict[str, Any], _inputs())
    payload["schema"] = "loslab.g3_joint_evaluator_inputs.v3"
    execution = {
        "candidate": {
            "config_id": "candidate-a",
            "configuration": {"receiver": "ls"},
            "configuration_sha256": joint._sha256({"receiver": "ls"}),
        },
        "challenger": {
            "config_id": "challenger-a",
            "configuration": {"receiver": "dft"},
            "configuration_sha256": joint._sha256({"receiver": "dft"}),
        },
        "waveform": {
            "waveform_id": "pusch-fixture",
            "configuration": {"mu": 1},
            "configuration_sha256": joint._sha256({"mu": 1}),
        },
        "link_budget_noise": {
            "tx_power_dbm": 23.0,
            "tx_gain_dbi": 0.0,
            "rx_gain_dbi": 0.0,
            "noise_figure_db": 7.0,
            "noise_temperature_k": 290.0,
            "noise_bandwidth_hz": 20e6,
        },
    }
    payload["receiver_execution"] = execution
    receiver_record = payload["receiver_impact"]["records"][0]
    receiver_record.update(
        {
            "family_id": "scene-a",
            "condition": "LOS",
            "episode_id": "episode-a",
            "receiver_execution_context_sha256": joint._sha256(execution),
        }
    )
    return payload


def _inputs_v4() -> dict[str, Any]:
    payload = _inputs_v3()
    payload["schema"] = "loslab.g3_joint_evaluator_inputs.v4"
    receiver = payload["receiver_impact"]["records"][0]
    receiver["configuration_id"] = "candidate-a"
    receiver["receiver_configuration_sha256"] = payload["receiver_execution"]["candidate"][
        "configuration_sha256"
    ]
    return payload


def test_joint_bundle_recomputes_and_binds_all_five_v4_branches(monkeypatch) -> None:
    observed: dict[str, object] = {}

    def fake_material(payload):
        observed["material"] = payload["joint_acceptance_family"]["family_sha256"]
        return {"status": "MATERIAL_PILOT_ONLY"}

    def fake_branch(name):
        def run(**kwargs):
            observed[name] = kwargs
            return {"status": f"{name.upper()}_NUMERICAL_ONLY"}

        return run

    monkeypatch.setattr(joint, "evaluate_g3_material_calibration", fake_material)
    monkeypatch.setattr(joint, "evaluate_complex_csi", fake_branch("csi"))
    monkeypatch.setattr(joint, "evaluate_propagation_metrics", fake_branch("propagation"))
    monkeypatch.setattr(joint, "evaluate_receiver_impact", fake_branch("receiver"))
    monkeypatch.setattr(joint, "evaluate_path_equivalence_roster", fake_branch("path"))

    payload = _inputs()
    bundle = joint.evaluate_g3_joint_evidence(payload)

    assert bundle["qualification_status"] == "NOT_QUALIFIED"
    assert set(bundle["branch_inputs_sha256"]) == {
        "material_calibration",
        "complex_csi",
        "propagation_metrics",
        "receiver_impact",
        "path_equivalence",
    }
    assert observed["material"] == payload["joint_acceptance_family"]["family_sha256"]
    for branch in ("csi", "propagation", "receiver", "path"):
        assert (
            observed[branch]["joint_acceptance_family"]["family_sha256"]
            == payload["joint_acceptance_family"]["family_sha256"]
        )
    assert observed["csi"]["expected_strata"] == (("scene-a", "LOS"),)
    assert observed["path"]["expected_scenarios"] == (("scene-a", "LOS", "episode-a"),)
    assert bundle["evaluator_statuses"]["path_equivalence"] == "PATH_NUMERICAL_ONLY"
    assert bundle["schema"] == "loslab.g3_joint_evaluator_bundle.v2"
    assert bundle["numerical_criterion_rollup"]["status"] == "UNRESOLVED_NUMERICAL_CRITERIA"
    assert bundle["numerical_criterion_rollup"]["qualification_status"] == "NOT_QUALIFIED"

    verified = joint.verify_g3_joint_evidence_bundle(payload, bundle)
    assert verified["status"] == "PASS_INPUT_SCOPED_RECOMPUTATION"
    assert verified["qualification_status"] == "NOT_QUALIFIED"


def test_branch_input_rejects_type_substituted_redundant_family() -> None:
    payload = _inputs()
    family = deepcopy(payload["joint_acceptance_family"])
    family["path_equivalence_admission_possible"] = 0
    payload["complex_csi"]["joint_acceptance_family"] = family

    with pytest.raises(ValueError, match="joint family differs"):
        joint._branch_input(payload, "complex_csi")


def test_joint_evaluator_rejects_type_substituted_material_family() -> None:
    payload = _inputs()
    family = deepcopy(payload["joint_acceptance_family"])
    family["path_equivalence_admission_possible"] = 0
    payload["material_calibration"]["payload"]["joint_acceptance_family"] = family

    with pytest.raises(ValueError, match="material calibration input must embed"):
        joint.evaluate_g3_joint_evidence(payload)


def test_joint_numerical_criterion_rollup_uses_only_explicit_existing_outcomes() -> None:
    outputs = {
        "material_calibration": {"strata": [{"pilot_margins_pass": True}]},
        "complex_csi": {"strata": [{"numeric_criteria_pass": True}]},
        "propagation_metrics": {
            "any_metric_assessable": True,
            "strata": [
                {
                    "metrics": [
                        {"criteria_status": "PASS"},
                        {"criteria_status": "NOT_APPLICABLE"},
                    ]
                }
            ],
        },
        "receiver_impact": {
            "strata": [{"bler_criteria_pass": True, "snr_at_10_percent_criteria_pass": True}]
        },
        "path_equivalence": {"scenarios": [{"class_coverage_status": "PASS"}]},
    }

    report = joint._numerical_criterion_rollup(outputs)

    assert report["status"] == "PASS_NUMERICAL_CRITERIA_ONLY"
    assert report["qualification_status"] == "NOT_QUALIFIED"
    assert report["branches"]["propagation_metrics"]["pass_count"] == 2


def test_joint_criterion_rollup_does_not_pass_all_not_applicable_propagation() -> None:
    outputs = {
        "material_calibration": {"strata": [{"pilot_margins_pass": True}]},
        "complex_csi": {"strata": [{"numeric_criteria_pass": True}]},
        "propagation_metrics": {
            "any_metric_assessable": False,
            "strata": [
                {
                    "metrics": [
                        {"criteria_status": "NOT_APPLICABLE"},
                        {"criteria_status": "NOT_APPLICABLE"},
                    ]
                }
            ],
        },
        "receiver_impact": {
            "strata": [{"bler_criteria_pass": True, "snr_at_10_percent_criteria_pass": True}]
        },
        "path_equivalence": {"scenarios": [{"class_coverage_status": "PASS"}]},
    }

    report = joint._numerical_criterion_rollup(outputs)

    assert report["branches"]["propagation_metrics"]["status"] == "UNRESOLVED"
    assert report["branches"]["propagation_metrics"]["pass_count"] == 0
    assert report["branches"]["propagation_metrics"]["unresolved_count"] == 2
    assert report["status"] == "UNRESOLVED_NUMERICAL_CRITERIA"


def test_joint_numerical_criterion_rollup_keeps_failure_and_missing_criteria_visible() -> None:
    outputs = {
        "material_calibration": {"strata": [{"pilot_margins_pass": False}]},
        "complex_csi": {"strata": [{"numeric_criteria_pass": True}, {}]},
        "propagation_metrics": {"strata": [{"metrics": [{"criteria_status": "UNRESOLVED"}]}]},
        "receiver_impact": {"strata": [{"bler_criteria_pass": True}]},
        "path_equivalence": {"scenarios": []},
    }

    report = joint._numerical_criterion_rollup(outputs)

    assert report["status"] == "FAIL_NUMERICAL_CRITERIA"
    assert report["qualification_status"] == "NOT_QUALIFIED"
    assert report["branches"]["complex_csi"]["unresolved_count"] == 1
    assert report["branches"]["receiver_impact"]["unresolved_count"] == 1
    assert report["branches"]["path_equivalence"]["status"] == "UNRESOLVED"


def test_joint_numerical_criterion_rollup_keeps_cross_branch_statuses_unresolved() -> None:
    outputs = {
        "material_calibration": {"strata": [{"pilot_margins_pass": "PASS"}]},
        "complex_csi": {"strata": [{"numeric_criteria_pass": "NOT_APPLICABLE"}]},
        "propagation_metrics": {
            "any_metric_assessable": True,
            "strata": [
                {
                    "metrics": [
                        {"criteria_status": "NOT_APPLICABLE"},
                        {"criteria_status": "PASS"},
                    ]
                }
            ],
        },
        "receiver_impact": {
            "strata": [
                {
                    "bler_criteria_pass": "PASS",
                    "snr_at_10_percent_criteria_pass": "NOT_APPLICABLE",
                }
            ]
        },
        "path_equivalence": {"scenarios": [{"class_coverage_status": "NOT_APPLICABLE"}]},
    }

    report = joint._numerical_criterion_rollup(outputs)

    assert report["branches"]["material_calibration"]["status"] == "UNRESOLVED"
    assert report["branches"]["complex_csi"]["status"] == "UNRESOLVED"
    assert report["branches"]["propagation_metrics"]["status"] == "PASS"
    assert report["branches"]["receiver_impact"]["status"] == "UNRESOLVED"
    assert report["branches"]["path_equivalence"]["status"] == "UNRESOLVED"
    assert report["status"] == "UNRESOLVED_NUMERICAL_CRITERIA"


def test_joint_bundle_v5_routes_raw_paths_to_path_recomputation(monkeypatch) -> None:
    payload = _inputs_v4()
    payload["schema"] = "loslab.g3_joint_evaluator_inputs.v5"
    payload["path_equivalence"]["records"] = [
        {
            "family_id": "scene-a",
            "condition": "LOS",
            "episode_id": "episode-a",
            "candidate_paths": [],
            "challenger_paths": [],
            "candidate_full_cfr_real": [0.0, 0.0],
            "candidate_full_cfr_imag": [0.0, 0.0],
            "challenger_full_cfr_real": [0.0, 0.0],
            "challenger_full_cfr_imag": [0.0, 0.0],
        }
    ]
    payload["path_equivalence"]["comparison_configuration"] = {
        "candidate_configuration_id": "candidate-a",
        "challenger_configuration_id": "challenger-a",
        "vertex_tolerance_m": 0.01,
        "delay_tolerance_s": 1e-9,
        "direction_tolerance_rad": 0.01,
        "h_floor": 1e-12,
        "transmit_power_per_re_w": 1.0,
        "noise_power_per_re_w": 1e-6,
    }
    monkeypatch.setattr(
        joint,
        "evaluate_g3_material_calibration",
        lambda payload: {"status": "MATERIAL_DIAGNOSTIC"},
    )
    monkeypatch.setattr(
        joint, "evaluate_complex_csi", lambda **kwargs: {"status": "CSI_DIAGNOSTIC"}
    )
    monkeypatch.setattr(
        joint,
        "evaluate_propagation_metrics",
        lambda **kwargs: {"status": "PROPAGATION_DIAGNOSTIC"},
    )
    monkeypatch.setattr(
        joint, "evaluate_receiver_impact", lambda **kwargs: {"status": "RECEIVER_DIAGNOSTIC"}
    )

    result = joint._recompute(payload)

    path_report = result["evaluations"]["path_equivalence"]
    assert path_report["scenarios"][0]["class_coverage_status"] == "NOT_ASSESSABLE_NO_CLASSES"
    assert len(path_report["scenarios"][0]["raw_path_input_sha256"]) == 64


def test_joint_bundle_rejects_tampered_scorer_output(monkeypatch) -> None:
    monkeypatch.setattr(
        joint,
        "evaluate_g3_material_calibration",
        lambda payload: {"status": "MATERIAL_PILOT_ONLY"},
    )
    for name in (
        "evaluate_complex_csi",
        "evaluate_propagation_metrics",
        "evaluate_receiver_impact",
        "evaluate_path_equivalence_roster",
    ):
        monkeypatch.setattr(joint, name, lambda **kwargs: {"status": "NUMERICAL_ONLY"})
    payload = _inputs()
    bundle = joint.evaluate_g3_joint_evidence(payload)
    tampered = deepcopy(bundle)
    tampered["evaluations"]["complex_csi"]["status"] = "PASS"

    with pytest.raises(ValueError, match="deterministic recomputation"):
        joint.verify_g3_joint_evidence_bundle(payload, tampered)


def test_joint_bundle_input_digest_changes_with_complex_cfr_bytes(monkeypatch) -> None:
    monkeypatch.setattr(
        joint,
        "evaluate_g3_material_calibration",
        lambda payload: {"status": "MATERIAL_PILOT_ONLY"},
    )
    for name in (
        "evaluate_complex_csi",
        "evaluate_propagation_metrics",
        "evaluate_receiver_impact",
        "evaluate_path_equivalence_roster",
    ):
        monkeypatch.setattr(joint, name, lambda **kwargs: {"status": "NUMERICAL_ONLY"})
    first = _inputs()
    second = _inputs()
    second["complex_csi"]["records"][0]["candidate_cfr"][0] = 2.0 + 0.0j

    first_bundle = joint.evaluate_g3_joint_evidence(first)
    second_bundle = joint.evaluate_g3_joint_evidence(second)

    assert (
        first_bundle["branch_inputs_sha256"]["complex_csi"]
        != second_bundle["branch_inputs_sha256"]["complex_csi"]
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.pop("path_equivalence"), "exactly the v1 fields"),
        (lambda payload: payload["complex_csi"].update(unknown=True), "unsupported fields"),
        (
            lambda payload: payload["receiver_impact"].update(
                joint_acceptance_family={"family_id": "different"}
            ),
            "differs from the bundle family",
        ),
    ],
)
def test_joint_bundle_rejects_malformed_or_mixed_inputs(mutation, message) -> None:
    payload = _inputs()
    mutation(payload)

    with pytest.raises(ValueError, match=message):
        joint.evaluate_g3_joint_evidence(payload)


@pytest.mark.parametrize(
    ("branch", "family_change", "input_change"),
    [
        (
            "complex_csi",
            {"complex_csi_expected_strata": [("scene-b", "LOS")]},
            {"expected_strata": [["scene-b", "LOS"]]},
        ),
        (
            "propagation_metrics",
            {"propagation_expected_strata": [("scene-b", "LOS", "total")]},
            {"expected_strata": [["scene-b", "LOS", "total"]]},
        ),
        (
            "receiver_impact",
            {"receiver_impact_expected_strata": [("scene-b", "LOS")]},
            {"expected_strata": [["scene-b", "LOS"]]},
        ),
        (
            "path_equivalence",
            {"path_equivalence_expected_scenarios": [("scene-b", "LOS", "episode-a")]},
            {"expected_scenarios": [["scene-b", "LOS", "episode-a"]]},
        ),
    ],
)
def test_joint_bundle_rejects_branch_roster_drift_before_scoring(
    branch, family_change, input_change
) -> None:
    payload = _inputs()
    old = payload["joint_acceptance_family"]
    family_arguments = {
        "family_id": old["family_id"],
        "material_expected_strata": [tuple(item) for item in old["material_expected_strata"]],
        "complex_csi_expected_strata": [tuple(item) for item in old["complex_csi_expected_strata"]],
        "propagation_expected_strata": [tuple(item) for item in old["propagation_expected_strata"]],
        "receiver_impact_expected_strata": [
            tuple(item) for item in old["receiver_impact_expected_strata"]
        ],
        "path_equivalence_expected_scenarios": [
            tuple(item) for item in old["path_equivalence_expected_scenarios"]
        ],
    }
    family_arguments.update(family_change)
    family = build_g3_joint_acceptance_family_v4(**family_arguments)
    payload["joint_acceptance_family"] = family
    payload["material_calibration"]["payload"]["joint_acceptance_family"] = family
    payload[branch].update(input_change)

    with pytest.raises(ValueError, match="roster must match across scene-based branches"):
        joint.evaluate_g3_joint_evidence(payload)


@pytest.mark.parametrize(
    "branch", ["complex_csi", "propagation_metrics", "receiver_impact", "path_equivalence"]
)
def test_joint_bundle_rejects_observed_episode_drift_before_scoring(branch) -> None:
    payload = _inputs_v3()
    payload[branch]["records"][0]["episode_id"] = "episode-unregistered"

    with pytest.raises(ValueError, match="observed episode roster differs"):
        joint.evaluate_g3_joint_evidence(payload)


@pytest.mark.parametrize(
    "mutation",
    [
        {"configuration_id": "unknown-config"},
        {"configuration_id": "candidate-a", "receiver_configuration_sha256": "0" * 64},
        {"configuration_id": ["candidate-a"]},
    ],
)
def test_joint_bundle_rejects_unbound_receiver_configuration(mutation) -> None:
    payload = _inputs_v4()
    payload["receiver_impact"]["records"][0].update(mutation)

    with pytest.raises(ValueError, match="not bound to the configuration named"):
        joint.evaluate_g3_joint_evidence(payload)


def test_joint_bundle_rejects_nonfinite_array_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _inputs()
    monkeypatch.setattr(
        joint,
        "evaluate_g3_material_calibration",
        lambda _material_payload: {"status": "MATERIAL_PILOT_ONLY"},
    )
    payload["complex_csi"]["records"][0]["candidate_cfr"] = np.asarray([np.nan + 0.0j])
    with pytest.raises(ValueError, match="finite values"):
        joint.evaluate_g3_joint_evidence(payload)


def test_joint_bundle_rejects_family_v3_even_when_its_digest_is_valid() -> None:
    payload = _inputs()
    family_v3 = build_g3_joint_acceptance_family_v3(
        family_id="legacy-family",
        material_expected_strata=[("concrete", "V", "20deg")],
        complex_csi_expected_strata=[("scene-a", "LOS")],
        propagation_expected_strata=[("scene-a", "LOS", "total")],
        receiver_impact_expected_strata=[("scene-a", "LOS")],
    )
    payload["joint_acceptance_family"] = family_v3
    payload["material_calibration"]["payload"]["joint_acceptance_family"] = family_v3

    with pytest.raises(ValueError, match="requires acceptance family v4"):
        joint.evaluate_g3_joint_evidence(payload)


def test_canonical_digest_preserves_scalar_array_shape_and_rejects_object_arrays() -> None:
    scalar = np.asarray(1.0)
    vector = np.asarray([1.0])
    assert joint._canonical_tree(scalar)["$array"]["shape"] == []
    assert joint._sha256(scalar) != joint._sha256(vector)
    with pytest.raises(TypeError, match="non-object numeric"):
        joint._sha256(np.asarray([object()], dtype=object))
