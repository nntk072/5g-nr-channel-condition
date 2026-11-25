"""Joint evaluation inputs: canonical values, branch fields, receiver execution and version rosters."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pytest
from evaluation.acceptance import joint_evaluation as joint
from tests.test_evaluation.joint_evidence_artifact_support import (
    _payload,
    _payload_v2,
    _payload_v3,
    _payload_v4,
    _payload_v5,
    _payload_v6,
)


def _evaluate(payload: dict[str, Any]) -> dict[str, Any]:
    return joint.evaluate_joint_evidence(payload)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        pytest.param(np.array([1.0, np.nan]), "non-finite values", id="array-nan"),
        pytest.param({1: "x"}, "object keys must be strings", id="key-type"),
        pytest.param({"x": complex(float("inf"), 0.0)}, "non-finite complex", id="complex-inf"),
        pytest.param({"x": float("nan")}, "non-finite numbers", id="float-nan"),
        pytest.param({"x": object()}, "unsupported joint evidence value", id="object"),
    ],
)
def test_canonical_tree_rejects_nonfinite_or_untyped_values(value, message) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        joint._canonical_tree(value)


def test_canonical_tree_accepts_numpy_scalars_and_arrays() -> None:
    tree = joint._canonical_tree({"a": np.array([1.0, 2.0]), "b": np.float64(3.0)})
    assert "$array" in tree["a"]
    assert tree["b"] == 3.0


def test_version_specific_fields_are_gated() -> None:
    payload = copy.deepcopy(_payload_v4())
    payload["solver_budget_provenance"] = {"max_wall_seconds": 1.0}
    with pytest.raises(
        ValueError, match="joint evaluator v4 inputs must contain exactly the v4 fields"
    ):
        _evaluate(payload)


def test_receiver_execution_must_declare_its_four_objects() -> None:
    payload = copy.deepcopy(_payload_v2())
    payload["receiver_execution"].pop("candidate")
    with pytest.raises(ValueError, match="exactly its four declared objects"):
        _evaluate(payload)


def test_receiver_execution_waveform_and_budget_fields_are_checked() -> None:
    payload = copy.deepcopy(_payload_v2())
    payload["receiver_execution"]["waveform"] = {"waveform_id": "", "configuration": {}}
    with pytest.raises(ValueError, match="waveform"):
        _evaluate(payload)


@pytest.mark.parametrize(
    ("branch_value", "message"),
    [
        (None, "must be an object"),
        ({}, "missing required fields"),
        ({"records": [], "expected_strata": [], "unexpected": 1}, "unsupported fields"),
        ({"records": []}, "missing required fields"),
    ],
)
def test_branch_input_rejects_wrong_shapes_and_fields(branch_value, message) -> None:
    payload = _payload()
    payload["complex_csi"] = branch_value
    with pytest.raises(ValueError, match=message):
        joint._branch_input(payload, "complex_csi")


@pytest.mark.parametrize(
    ("payload_factory", "field", "message"),
    [
        (_payload_v4, "solver_budget_provenance", "requires joint evaluator input v6"),
        (_payload_v4, "comparison_configuration", "requires joint evaluator input v5 or v6"),
        (_payload_v5, "solver_budget_provenance", "requires joint evaluator input v6"),
    ],
)
def test_path_branch_rejects_fields_from_newer_input_versions(
    payload_factory, field, message
) -> None:
    payload = payload_factory()
    payload["path_equivalence"][field] = {}
    with pytest.raises(ValueError, match=message):
        joint._branch_input(payload, "path_equivalence")


@pytest.mark.parametrize(
    ("roster", "message"),
    [
        ("not-an-array", "must be an array"),
        (["scene-a"], "entries must be arrays"),
    ],
)
def test_branch_input_requires_array_rosters(roster, message) -> None:
    payload = _payload()
    payload["complex_csi"]["expected_strata"] = roster
    with pytest.raises(ValueError, match=message):
        joint._branch_input(payload, "complex_csi")


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda c: c.update(configuration_sha256="bad"), "hash is inconsistent"),
        (lambda c: c.update(configuration=[]), "configuration must be an object"),
        (lambda c: c.update(config_id="  "), "config_id must be nonempty"),
        (lambda c: c.update(extra="field"), "fields are invalid"),
    ],
)
def test_receiver_context_rejects_invalid_receiver_configurations(edit, message) -> None:
    payload = _payload_v2()
    edit(payload["receiver_execution"]["candidate"])
    with pytest.raises(ValueError, match=message):
        _evaluate(payload)


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda w: w.update(waveform_id=" "), "waveform_id must be nonempty"),
        (lambda w: w.update(configuration=[]), "waveform configuration hash is inconsistent"),
        (
            lambda w: w.update(configuration_sha256="bad"),
            "waveform configuration hash is inconsistent",
        ),
    ],
)
def test_receiver_context_rejects_invalid_waveform_configuration(edit, message) -> None:
    payload = _payload_v2()
    edit(payload["receiver_execution"]["waveform"])
    with pytest.raises(ValueError, match=message):
        _evaluate(payload)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("noise_bandwidth_hz", True, "finite numbers"),
        ("tx_power_dbm", float("inf"), "finite numbers"),
        ("noise_temperature_k", 0.0, "must be positive"),
        ("noise_bandwidth_hz", -1.0, "must be positive"),
    ],
)
def test_receiver_context_rejects_invalid_link_budget_values(key, value, message) -> None:
    payload = _payload_v2()
    payload["receiver_execution"]["link_budget_noise"][key] = value
    with pytest.raises(ValueError, match=message):
        _evaluate(payload)


def test_receiver_context_rejects_link_budget_field_drift_and_config_id_mismatch() -> None:
    payload = _payload_v2()
    payload["receiver_execution"]["link_budget_noise"]["extra"] = 1.0
    with pytest.raises(ValueError, match="link_budget_noise fields are invalid"):
        _evaluate(payload)
    payload = _payload_v2()
    payload["receiver_execution"]["candidate"]["config_id"] = "other"
    with pytest.raises(ValueError, match="configuration IDs differ"):
        _evaluate(payload)


def test_receiver_context_rejects_unbound_records_and_empty_record_rosters() -> None:
    payload = _payload_v2()
    payload["receiver_impact"]["records"][0]["receiver_execution_context_sha256"] = "bad"
    with pytest.raises(ValueError, match="not bound to the declared"):
        _evaluate(payload)
    payload = _payload_v2()
    payload["receiver_impact"]["records"] = []
    with pytest.raises(ValueError, match="nonempty array"):
        _evaluate(payload)


def test_receiver_context_checks_record_configuration_binding_for_v4() -> None:
    payload = _payload_v4()
    payload["receiver_impact"]["records"][0]["receiver_configuration_sha256"] = "bad"
    with pytest.raises(ValueError, match="not bound to the configuration"):
        _evaluate(payload)


def test_receiver_context_checks_every_row_configuration_binding() -> None:
    payload = _payload_v4()
    record = copy.deepcopy(payload["receiver_impact"]["records"][0])
    record["episode_id"] = "episode-b"
    payload["receiver_impact"]["records"].append(record)
    joint._validate_receiver_execution_context(
        payload["receiver_execution"],
        payload["receiver_impact"],
        require_row_configuration_binding=True,
    )
    legacy_payload = _payload_v2()
    legacy_record = copy.deepcopy(legacy_payload["receiver_impact"]["records"][0])
    legacy_payload["receiver_impact"]["records"].append(legacy_record)
    joint._validate_receiver_execution_context(
        legacy_payload["receiver_execution"], legacy_payload["receiver_impact"]
    )


@pytest.mark.parametrize(
    ("payload_factory", "path", "replacement", "message"),
    [
        (
            _payload_v3,
            ("complex_csi", "expected_strata"),
            [["", "LOS"]],
            "expected-stratum roster is malformed",
        ),
        (
            _payload_v3,
            ("path_equivalence", "expected_scenarios"),
            [["scene-a", "DAY", "episode-a"]],
            "expected-scenario roster is malformed",
        ),
        (
            _payload_v3,
            ("path_equivalence", "expected_scenarios"),
            [["scene-b", "LOS", "episode-a"]],
            "episode roster differs from the shared family",
        ),
        (_payload_v3, ("receiver_impact", "records"), [], "nonempty episode roster"),
        (_payload_v3, ("receiver_impact", "records"), [None], "record 0 must be an object"),
        (
            _payload_v3,
            ("receiver_impact", "records"),
            [{"family_id": "scene-a", "condition": "DAY", "episode_id": "episode-a"}],
            "needs family_id",
        ),
    ],
)
def test_shared_roster_rejects_malformed_rows(payload_factory, path, replacement, message) -> None:
    payload = payload_factory()
    payload[path[0]][path[1]] = replacement
    with pytest.raises(ValueError, match=message):
        _evaluate(payload)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "inputs must be an object"),
        ({"schema": "unknown"}, "schema must be one of"),
        ({"schema": "loslab.joint_evaluator_inputs.v1"}, "v1 inputs must contain exactly"),
    ],
)
def test_recompute_rejects_invalid_input_envelopes(payload, message) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        joint._recompute(payload)


def test_recompute_requires_material_calibration_payload_object() -> None:
    payload = _payload()
    payload["material_calibration"]["payload"] = None
    with pytest.raises(ValueError, match="material_calibration.payload must be an object"):
        _evaluate(payload)


def test_bundle_verifier_requires_object_bundle() -> None:
    with pytest.raises(TypeError, match="bundle must be an object"):
        joint.verify_joint_evidence_bundle(_payload(), [])


def test_criterion_report_rejects_unknown_outcome_kind() -> None:
    with pytest.raises(ValueError, match="unsupported criterion outcome kind"):
        joint._criterion_branch_report("branch", [], outcome_kind="unknown")


@pytest.mark.parametrize(
    ("payload_factory", "version"),
    [
        (_payload_v2, "v2"),
        (_payload_v3, "v3"),
        (_payload_v4, "v4"),
        (_payload_v5, "v5"),
        (_payload_v6, "v6"),
    ],
)
def test_recompute_rejects_extra_fields_for_each_version(payload_factory, version) -> None:
    payload = payload_factory()
    payload["extra"] = True
    with pytest.raises(ValueError, match=f"{version} inputs must contain exactly"):
        joint._recompute(payload)
