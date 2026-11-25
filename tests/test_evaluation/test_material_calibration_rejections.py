"""Material calibration evaluator: each rejected split, incidence, prediction and angular input."""

from __future__ import annotations

import copy
import hashlib
import json
import sys

import pytest
from evaluation.acceptance.material_calibration import (
    _angular_joint_cdf_ks,
    _spherical_cell_solid_angles,
    evaluate_material_calibration,
)
from evaluation.acceptance.material_calibration import (
    main as material_calibration_main,
)
from tests.test_evaluation.test_material_calibration import (
    _assessable_payload,
    _payload,
    _refresh_split_manifest_hash,
)


def _mutated(mutate, *, refresh: bool = True):
    payload = copy.deepcopy(_assessable_payload())
    mutate(payload)
    if refresh:
        _refresh_split_manifest_hash(payload)
    return payload


def _rehash_only(payload):
    encoded = json.dumps(payload["full_split_manifest"], sort_keys=True, separators=(",", ":"))
    payload["full_split_manifest_sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
    return payload


_UNHASHED = {"refresh": False}


def _strip_prediction(payload, index=0, **changes):
    payload["test_predictions"][index].update(changes)


@pytest.mark.parametrize(
    ("mutate", "message", "refresh"),
    [
        (
            lambda p: p.update(uncertainty_description="  "),
            "uncertainty_description must document",
            True,
        ),
        (
            lambda p: p.update(incidence_groups=[]),
            "incidence_groups must be a nonempty complete group manifest",
            False,
        ),
        (
            lambda p: p.update(test_predictions=[]),
            "test_predictions must be a nonempty list",
            False,
        ),
        (
            lambda p: p["expected_strata"].append("bogus"),
            "each expected_strata entry must be an object",
            False,
        ),
        (
            lambda p: p["expected_strata"].append(copy.deepcopy(p["expected_strata"][0])),
            "duplicate expected stratum",
            False,
        ),
        (
            lambda p: p["full_split_manifest"].update(schema="other.schema"),
            "full_split_manifest must use schema",
            True,
        ),
        (
            lambda p: p["full_split_manifest"]["groups"].pop(0),
            "outside its declared test group",
            True,
        ),
        (
            lambda p: p["full_split_manifest"]["groups"][0].update(split="validation"),
            "outside its declared test group",
            True,
        ),
        (
            lambda p: p["full_split_manifest"]["groups"][0].update(sample_ids=[]),
            "requires nonempty sample_ids",
            True,
        ),
        (
            lambda p: p["test_predictions"][0].update(polarization="X"),
            "polarization must be V or H",
            True,
        ),
    ],
)
def test_split_and_prediction_rejections(mutate, message, refresh) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate_material_calibration(_mutated(mutate, refresh=refresh))


def test_each_prediction_must_be_an_object_and_unique() -> None:
    with pytest.raises(ValueError, match="each test prediction must be an object"):
        evaluate_material_calibration(
            _mutated(lambda p: p["test_predictions"].append("not-a-row"), refresh=False)
        )
    with pytest.raises(ValueError, match="duplicate sample_id"):
        evaluate_material_calibration(
            _mutated(
                lambda p: p["test_predictions"].append(copy.deepcopy(p["test_predictions"][0])),
                refresh=False,
            )
        )


def test_prediction_interval_must_be_an_ordered_pair() -> None:
    with pytest.raises(ValueError, match="prediction_interval_95_db must be"):
        evaluate_material_calibration(
            _mutated(lambda p: _strip_prediction(p, prediction_interval_95_db=[1.0]))
        )
    with pytest.raises(ValueError, match="lower bound exceeds upper bound"):
        evaluate_material_calibration(
            _mutated(lambda p: _strip_prediction(p, prediction_interval_95_db=[5.0, 1.0]))
        )


def test_angular_semantics_are_fixed_by_the_contract() -> None:
    with pytest.raises(
        ValueError, match="angular_power_semantics must be integrated_power_per_bin"
    ):
        evaluate_material_calibration(
            _mutated(lambda p: _strip_prediction(p, angular_power_semantics="density"))
        )


@pytest.mark.parametrize("value", [True, "3.5", float("inf")])
def test_finite_number_rejects_booleans_strings_and_infinity(value) -> None:
    with pytest.raises(ValueError, match="target_frequency_hz must be a finite number"):
        evaluate_material_calibration(
            _mutated(lambda payload: payload.update(target_frequency_hz=value), refresh=False)
        )


@pytest.mark.parametrize(
    "edges",
    [
        ((1.0, 180.0), (-180.0, 180.0)),
        ((0.0, 180.0), (-170.0, 180.0)),
        ((0.0, 0.0, 180.0), (-180.0, 180.0)),
    ],
)
def test_spherical_grid_requires_full_sphere_and_strictly_increasing_edges(edges) -> None:
    with pytest.raises(ValueError, match="angular grid must cover zenith"):
        _spherical_cell_solid_angles(*edges)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p["full_split_manifest"].update(expected_strata={}), "bind the preregistered"),
        (lambda p: p["full_split_manifest"]["groups"].append(None), "group must be an object"),
        (
            lambda p: p["test_predictions"][0].update(solid_angle_sr=[[1.0]]),
            "shape must match theta/phi angular bins",
        ),
        (
            lambda p: p["test_predictions"][0].update(theta_edges_deg="bad"),
            "theta_edges_deg and phi_edges_deg must be arrays",
        ),
        (
            lambda p: p["test_predictions"][0].update(
                measured_angular_power_per_bin_w=[[0.0] * 4 for _ in range(3)]
            ),
            "must have positive sums",
        ),
    ],
)
def test_split_assignments_and_measurement_rows_reject_binding_drift(mutate, message) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate_material_calibration(_mutated(mutate))


@pytest.mark.parametrize(
    ("mutate", "message", "refresh"),
    [
        (lambda p: p["full_split_manifest"].update(groups=[]), "groups must declare", True),
        (
            lambda p: p["incidence_groups"].append(None),
            "each incidence group must be an object",
            False,
        ),
        (
            lambda p: p["incidence_groups"][1].update(
                measurement_unit_id="measurement-unit-angle-20deg-0"
            ),
            "one measurement_unit_id maps to conflicting provenance IDs",
            True,
        ),
        (
            lambda p: p["incidence_groups"][1].update(
                measurement_unit_provenance_id="source-record-angle-20deg-0"
            ),
            "distinct measurement units cannot reuse one provenance record ID",
            True,
        ),
    ],
)
def test_manifest_rosters_and_independent_unit_identity_are_enforced(
    mutate, message, refresh
) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate_material_calibration(_mutated(mutate, refresh=refresh))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda p: p["full_split_manifest"]["test_sample_assignments"].append(None),
            "each test_sample_assignments entry must be an object",
        ),
        (
            lambda p: p["full_split_manifest"]["test_sample_assignments"].pop(),
            "test_sample_assignments must exactly cover all test sample IDs",
        ),
        (
            lambda p: p["full_split_manifest"]["test_sample_assignments"].append(
                copy.deepcopy(p["full_split_manifest"]["test_sample_assignments"][0])
            ),
            "duplicate test sample assignment",
        ),
        (
            lambda p: p["full_split_manifest"]["test_sample_assignments"].clear(),
            "bind every test sample",
        ),
    ],
)
def test_hashed_test_assignments_reject_bad_rows(mutate, message) -> None:
    payload = _mutated(lambda p: None)
    mutate(payload)
    _rehash_only(payload)
    with pytest.raises(ValueError, match=message):
        evaluate_material_calibration(payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("incidence_stratum_id", "wrong", "disagrees with its incidence-group stratum"),
        ("polarization", "X", "polarization must be V or H"),
        ("material_id", "other", "disagrees with its hashed material/polarization"),
    ],
)
def test_prediction_rows_cannot_drift_from_the_hashed_split(field, value, message) -> None:
    payload = _mutated(lambda p: None)
    _refresh_split_manifest_hash(payload)
    payload["test_predictions"][0][field] = value
    _rehash_only(payload)
    with pytest.raises(ValueError, match=message):
        evaluate_material_calibration(payload)


def test_angular_joint_cdf_requires_nonnegative_nonzero_patterns() -> None:
    with pytest.raises(ValueError, match="same-shape 2D finite"):
        _angular_joint_cdf_ks([[0.0, 0.0]], [[1.0, 0.0]])


def test_angular_power_fields_must_be_nested_lists() -> None:
    with pytest.raises(ValueError, match="must be nested 2D arrays"):
        evaluate_material_calibration(
            _mutated(
                lambda p: p["test_predictions"][0].update(
                    measured_angular_power_per_bin_w={"not": "an array"}
                )
            )
        )


def test_hashed_assignment_must_retain_measurement_unit_provenance_binding() -> None:
    payload = _mutated(lambda p: None)
    payload["full_split_manifest"]["test_sample_assignments"][0][
        "measurement_unit_provenance_id"
    ] = "different-record"
    _rehash_only(payload)
    with pytest.raises(ValueError, match="test sample assignment disagrees with incidence group"):
        evaluate_material_calibration(payload)


def test_cli_rejects_non_object_json_root(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "predictions.json"
    input_path.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["evaluate-material-calibration", str(input_path)])
    with pytest.raises(ValueError, match="input JSON root must be an object"):
        material_calibration_main()


@pytest.mark.parametrize("write_output", [False, True])
def test_cli_evaluates_payload_and_writes_or_prints_report(
    tmp_path, monkeypatch, capsys, write_output
) -> None:
    input_path = tmp_path / "predictions.json"
    input_path.write_text(json.dumps(_payload()), encoding="utf-8")
    output_path = tmp_path / "report.json"
    argv = ["evaluate-material-calibration", str(input_path)]
    if write_output:
        argv.extend(["--output", str(output_path)])
    monkeypatch.setattr(sys, "argv", argv)
    material_calibration_main()
    if write_output:
        report = json.loads(output_path.read_text(encoding="utf-8"))
        assert len(report["prediction_input_sha256"]) == 64
    else:
        assert '"prediction_input_sha256"' in capsys.readouterr().out
