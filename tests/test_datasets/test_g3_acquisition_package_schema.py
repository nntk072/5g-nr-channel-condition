"""Validation for the proposal-only G3.03 acquisition package format."""

import copy
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "datasets/schema/g3.03-acquisition-package.v0.1.schema.json"
FIXTURE_PATH = ROOT / "datasets/schema/examples/g3.03-acquisition-package.v0.1.example.json"


@pytest.fixture(scope="module")
def package():
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def schema_validator():
    jsonschema = pytest.importorskip(
        "jsonschema", reason="jsonschema is not a declared repository test dependency"
    )
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker())


def _assert_references_resolve(package):
    for key, id_key in (
        ("files", "file_id"),
        ("specimens", "specimen_id"),
        ("acquisition_units", "unit_id"),
        ("calibrations", "calibration_id"),
        ("samples", "sample_id"),
        ("fits", "fit_id"),
        ("angular_measurements", "angular_id"),
        ("csi_measurements", "csi_id"),
        ("rights", "rights_id"),
    ):
        ids = [item[id_key] for item in package[key]]
        assert len(ids) == len(set(ids))
    file_ids = {item["file_id"] for item in package["files"]}
    specimens = {item["specimen_id"] for item in package["specimens"]}
    units = {item["unit_id"] for item in package["acquisition_units"]}
    calibrations = {item["calibration_id"] for item in package["calibrations"]}
    samples = {item["sample_id"] for item in package["samples"]}
    attestations = {value["attestation_id"] for value in package["attestations"].values()}
    assert len(attestations) == len(package["attestations"])

    def files(ids):
        assert set(ids) <= file_ids

    for item in package["files"]:
        files(item["dependency_file_ids"])
        assert set(item["sample_ids"]) <= samples
        assert set(item["specimen_ids"]) <= specimens
        assert set(item["calibration_ids"]) <= calibrations
    for item in package["specimens"]:
        files(item["provenance_file_ids"])
    for item in package["acquisition_units"]:
        assert item["specimen_id"] in specimens
        assert item["independence_attestation_id"] in attestations
        files(item["provenance_file_ids"])
    for item in package["calibrations"]:
        assert item["truth_attestation_id"] in attestations
        for key in ("standard_file_ids", "run_file_ids", "processing_file_ids"):
            files(item[key])
    for item in package["samples"]:
        assert item["specimen_id"] in specimens
        assert item["unit_id"] in units
        assert item["calibration_id"] in calibrations
        files(item["file_ids"])
    for item in package["fits"]:
        assert item["specimen_id"] in specimens
        files(item["software_file_ids"])
        for parameter in item["parameters"]:
            if "covariance_file_id" in parameter:
                files([parameter["covariance_file_id"]])
    for item in package["angular_measurements"]:
        assert item["sample_id"] in samples
    for item in package["csi_measurements"]:
        assert item["sample_id"] in samples
        files(item["h_floor"]["noise_input_file_ids"])
        files(item["file_ids"])
    for item in package["rights"]:
        assert item["attestation_id"] in attestations
        files(item["evidence_file_ids"])
    for item in package["attestations"].values():
        files(item["evidence_file_ids"])


def _assert_angular_semantics(package):
    sample_power = {
        item["sample_id"]: item["measured_integrated_power_w"] for item in package["samples"]
    }
    for item in package["angular_measurements"]:
        theta = item["theta_edges_deg"]
        phi = item["phi_edges_deg"]
        assert all(a < b for a, b in zip(theta, theta[1:]))
        assert all(a < b for a, b in zip(phi, phi[1:]))
        n_theta, n_phi = len(theta) - 1, len(phi) - 1
        powers = item["power_per_bin_w"]
        solid_angles = item["solid_angle_sr"]
        assert len(powers) == len(solid_angles) == n_theta
        assert all(len(row) == n_phi for row in powers + solid_angles)
        for i in range(n_theta):
            expected_theta_measure = math.cos(math.radians(theta[i])) - math.cos(
                math.radians(theta[i + 1])
            )
            for j in range(n_phi):
                expected = math.radians(phi[j + 1] - phi[j]) * expected_theta_measure
                assert solid_angles[i][j] == pytest.approx(expected)
        assert sum(map(sum, powers)) == pytest.approx(sample_power[item["sample_id"]])


def _assert_csi_shape(package):
    for item in package["csi_measurements"]:
        expected_shape = [
            len(item["rx_port_ids"]),
            len(item["tx_port_ids"]),
            len(item["subcarrier_indices"]),
            len(item["ofdm_symbol_indices"]),
        ]
        assert item["shape"] == expected_shape
        assert len(item["subcarrier_indices"]) == len(item["subcarrier_frequency_offsets_hz"])
        n_values = math.prod(expected_shape)
        assert len(item["measured_complex"]) == n_values
        assert len(item["predicted_complex"]) == n_values


def test_illustrative_fixture_validates_against_schema(package, schema_validator):
    assert list(schema_validator.iter_errors(package)) == []


def test_illustrative_fixture_is_self_consistent(package):
    _assert_references_resolve(package)
    _assert_angular_semantics(package)
    _assert_csi_shape(package)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p["attestations"].pop("calibration_truth"),
        lambda p: p["files"][0].update(unreviewed_property=True),
        lambda p: p["files"][0].update(sha256="not-a-sha256"),
        lambda p: p.update(schema_version="g3.03-acquisition-package.v1"),
    ],
)
def test_schema_fails_closed_on_missing_or_ambiguous_contract(mutate, package, schema_validator):
    invalid = copy.deepcopy(package)
    mutate(invalid)
    assert list(schema_validator.iter_errors(invalid))


def test_dangling_file_dependency_is_rejected(package):
    invalid = copy.deepcopy(package)
    invalid["files"][0]["dependency_file_ids"] = ["file.missing"]
    with pytest.raises(AssertionError):
        _assert_references_resolve(invalid)


def test_csi_shape_mismatch_is_rejected(package):
    invalid = copy.deepcopy(package)
    invalid["csi_measurements"][0]["shape"] = [1, 1, 2, 1]
    with pytest.raises(AssertionError):
        _assert_csi_shape(invalid)
