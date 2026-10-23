"""Adversarial tests for proposal-only G3.03 package consistency checks."""

from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path
from typing import Any

from evaluation.g3_acquisition_package import verify_g3_acquisition_package

PACKAGE_HELPERS = Path(__file__).parents[1] / "test_datasets/test_g3_acquisition_package_v0_2.py"
_package_factory = runpy.run_path(str(PACKAGE_HELPERS))["_package"]


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _put(root: Path, package: dict[str, Any], file_id: str, data: bytes) -> None:
    entry = next(item for item in package["files"] if item["file_id"] == file_id)
    path = root / entry["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    entry["size_bytes"] = len(data)
    entry["sha256"] = hashlib.sha256(data).hexdigest()
    for binding_name in ("scorer_input", "split_manifest", "evaluator_output"):
        reference = package["evaluator_binding"][binding_name]
        if reference["file_id"] == file_id:
            reference["exact_bytes_size"] = len(data)
            reference["exact_bytes_sha256"] = hashlib.sha256(data).hexdigest()


def _make_package(root: Path) -> Path:
    package = _package_factory()
    stratum = {
        "material_id": "material.example.1",
        "polarization": "V",
        "incidence_stratum_id": "incidence.example.1",
    }
    sample_id = package["samples"][0]["sample_id"]
    group_id = "group.example.1"
    unit_id = package["samples"][0]["unit_id"]
    provenance_id = "unit.provenance.example.1"
    manifest = {
        "schema": "loslab.g3_material_calibration_split_manifest.v1",
        "expected_strata": [stratum],
        "groups": [{"incidence_group_id": group_id, "split": "test", "sample_ids": [sample_id]}],
        "test_sample_assignments": [
            {
                "sample_id": sample_id,
                "incidence_group_id": group_id,
                "incidence_stratum_id": stratum["incidence_stratum_id"],
                "incidence_angle_deg": 20.0,
                "material_id": stratum["material_id"],
                "polarization": stratum["polarization"],
                "measurement_unit_id": unit_id,
                "measurement_unit_provenance_id": provenance_id,
            }
        ],
    }
    prediction = {
        "sample_id": sample_id,
        "incidence_group_id": group_id,
        "incidence_stratum_id": stratum["incidence_stratum_id"],
        "incidence_angle_deg": 20.0,
        "split": "test",
        "material_id": stratum["material_id"],
        "polarization": stratum["polarization"],
        "measured_integrated_power_db": -10.0,
        "predicted_integrated_power_db": -10.1,
        "prediction_interval_95_db": [-11.0, -9.0],
        "measured_angular_power": [1.0, 2.0],
        "predicted_angular_power": [1.1, 1.9],
        "angular_bin_edges_deg": [0.0, 90.0, 180.0],
    }
    scorer_input = {
        "schema": "loslab.g3_material_calibration_predictions.v1",
        "target_frequency_hz": 3.5e9,
        "model_id": "model.example.1",
        "model_sha256": "a" * 64,
        "measurement_calibration_id": "calibration.example.1",
        "material_mapping_id": "mapping.example.1",
        "surface_description": "synthetic proposal fixture",
        "uncertainty_description": "synthetic proposal fixture",
        "measurement_unit_independence_description": "synthetic proposal fixture",
        "cluster_bootstrap_seed": 1,
        "cluster_bootstrap_replicates": 1000,
        "expected_strata": [stratum],
        "full_split_manifest": manifest,
        "full_split_manifest_sha256": _canonical_hash(manifest),
        "split_manifest_external_verification_id": "verification.example.1",
        "incidence_groups": [
            {
                "incidence_group_id": group_id,
                "measurement_unit_id": unit_id,
                "measurement_unit_provenance_id": provenance_id,
                "incidence_stratum_id": stratum["incidence_stratum_id"],
                "incidence_angle_deg": 20.0,
                "split": "test",
                "sample_ids": [sample_id],
            }
        ],
        "test_predictions": [prediction],
    }
    input_bytes = (json.dumps(scorer_input, sort_keys=True, separators=(",", ":")) + "\n").encode()
    output = {
        "schema": "loslab.g3_material_calibration_evaluation.v1",
        "prediction_input_sha256": hashlib.sha256(input_bytes).hexdigest(),
        "strata": [stratum],
    }
    package["protocols"][0]["expected_strata"] = [stratum]
    package["protocols"][0]["split_manifest_canonical_sha256"] = _canonical_hash(manifest)
    package["row_crosswalk"] = [
        {
            "acquisition_sample_id": sample_id,
            "evaluator_sample_id": sample_id,
            "input_row_pointer": "/test_predictions/0",
            "output_stratum_pointer": "/strata/0",
            "incidence_group_id": group_id,
            "split": "test",
            "material_id": stratum["material_id"],
            "polarization": stratum["polarization"],
            "incidence_stratum_id": stratum["incidence_stratum_id"],
            "measurement_unit_id": unit_id,
            "measurement_unit_provenance_id": provenance_id,
            "source_file_ids": ["file.raw.scan1"],
        }
    ]
    files = {item["file_id"]: item for item in package["files"]}
    split_file_id = package["evaluator_binding"]["split_manifest"]["file_id"]
    _put(root, package, split_file_id, (json.dumps(manifest, sort_keys=True) + "\n").encode())
    scorer_file_id = package["evaluator_binding"]["scorer_input"]["file_id"]
    _put(root, package, scorer_file_id, input_bytes)
    output_file_id = package["evaluator_binding"]["evaluator_output"]["file_id"]
    _put(root, package, output_file_id, (json.dumps(output, sort_keys=True) + "\n").encode())
    package["evaluator_binding"]["reported_prediction_input_sha256"] = hashlib.sha256(
        input_bytes
    ).hexdigest()
    for file_id, entry in files.items():
        if file_id not in {split_file_id, scorer_file_id, output_file_id}:
            _put(root, package, file_id, b"synthetic proposal bytes\n")
    package_path = root / "package.json"
    package_path.write_text(json.dumps(package, indent=2), encoding="utf-8")
    return package_path


def _error_text(report: dict[str, Any]) -> str:
    return "\n".join(report["errors"])


def test_consistent_package_reports_proposal_only_and_not_qualified(tmp_path: Path):
    package_path = _make_package(tmp_path)
    package = json.loads(package_path.read_text(encoding="utf-8"))

    report = verify_g3_acquisition_package(package_path)

    for collection in ("samples", "fits", "angular_measurements", "csi_measurements"):
        assert all(
            record["record_status"] == "synthetic_example_only" for record in package[collection]
        )
    assert all(
        parameter["record_status"] == "synthetic_example_only"
        for fit in package["fits"]
        for parameter in fit["parameters"]
    )
    assert all(
        item["attestation_status"] == "planned_not_signed"
        and item["signed_utc"] is None
        and item["signature_value"] is None
        for item in package["attestations"].values()
    )
    assert report["status"] == "consistent_proposal_only", _error_text(report)
    assert report["proposal_only"] is True
    assert report["qualification_status"] == "NOT_QUALIFIED"
    assert any("never measurement evidence" in item for item in report["limitations"])
    assert all(check["status"] == "passed" for check in report["checks"])


def test_inventory_detects_exact_byte_tampering(tmp_path: Path):
    package_path = _make_package(tmp_path)
    package = json.loads(package_path.read_text())
    scorer_path = tmp_path / next(
        row["path"] for row in package["files"] if row["file_id"] == "file.scorer.input"
    )
    scorer_path.write_bytes(scorer_path.read_bytes() + b"tampered")

    report = verify_g3_acquisition_package(package_path)

    assert report["status"] == "inconsistent"
    assert "size_bytes or SHA-256" in _error_text(report)


def test_symlink_component_is_rejected(tmp_path: Path):
    package_path = _make_package(tmp_path)
    package = json.loads(package_path.read_text())
    entry = package["files"][0]
    original = tmp_path / entry["path"]
    external = tmp_path / "outside.bin"
    external.write_bytes(original.read_bytes())
    original.unlink()
    original.symlink_to(external)
    package_path.write_text(json.dumps(package), encoding="utf-8")

    report = verify_g3_acquisition_package(package_path)

    assert report["status"] == "inconsistent"
    assert "Too many levels" in _error_text(report) or "symlink" in _error_text(report).lower()


def test_unsafe_relative_path_is_rejected_by_schema(tmp_path: Path):
    package_path = _make_package(tmp_path)
    package = json.loads(package_path.read_text())
    package["files"][0]["path"] = "../outside.bin"
    package_path.write_text(json.dumps(package), encoding="utf-8")

    report = verify_g3_acquisition_package(package_path)

    assert report["status"] == "inconsistent"
    assert any(
        check["name"] == "json_schema" and check["status"] == "failed" for check in report["checks"]
    )


def test_bad_embedded_canonical_manifest_hash_is_rejected(tmp_path: Path):
    package_path = _make_package(tmp_path)
    package = json.loads(package_path.read_text())
    scorer = next(item for item in package["files"] if item["file_id"] == "file.scorer.input")
    scorer_path = tmp_path / scorer["path"]
    payload = json.loads(scorer_path.read_text())
    payload["full_split_manifest_sha256"] = "f" * 64
    _put(tmp_path, package, "file.scorer.input", json.dumps(payload, sort_keys=True).encode())
    package["evaluator_binding"]["reported_prediction_input_sha256"] = hashlib.sha256(
        scorer_path.read_bytes()
    ).hexdigest()
    output = json.loads(
        (
            tmp_path
            / next(
                item["path"]
                for item in package["files"]
                if item["file_id"] == "file.evaluator.output"
            )
        ).read_text()
    )
    output["prediction_input_sha256"] = hashlib.sha256(scorer_path.read_bytes()).hexdigest()
    _put(
        tmp_path,
        package,
        "file.evaluator.output",
        json.dumps(output, sort_keys=True).encode(),
    )
    package_path.write_text(json.dumps(package), encoding="utf-8")

    report = verify_g3_acquisition_package(package_path)

    assert report["status"] == "inconsistent"
    assert "canonical full_split_manifest hash is incorrect" in _error_text(report)


def test_package_binding_and_evaluator_echo_must_match(tmp_path: Path):
    package_path = _make_package(tmp_path)
    package = json.loads(package_path.read_text())
    package["evaluator_binding"]["reported_prediction_input_sha256"] = "f" * 64
    package_path.write_text(json.dumps(package), encoding="utf-8")

    report = verify_g3_acquisition_package(package_path)

    assert report["status"] == "inconsistent"
    assert "reported_prediction_input_sha256 differs" in _error_text(report)


def test_crosswalk_pointer_and_output_stratum_join_are_checked(tmp_path: Path):
    package_path = _make_package(tmp_path)
    package = json.loads(package_path.read_text())
    package["row_crosswalk"][0]["input_row_pointer"] = "/test_predictions/2"
    package_path.write_text(json.dumps(package), encoding="utf-8")

    report = verify_g3_acquisition_package(package_path)

    assert report["status"] == "inconsistent"
    assert "pointer index is outside its array" in _error_text(report)


def test_missing_per_file_rights_is_rejected(tmp_path: Path):
    package_path = _make_package(tmp_path)
    package = json.loads(package_path.read_text())
    package["per_file_rights"].pop()
    package_path.write_text(json.dumps(package), encoding="utf-8")

    report = verify_g3_acquisition_package(package_path)

    assert report["status"] == "inconsistent"
    assert "per_file_rights must cover every inventoried file exactly once" in _error_text(report)


def test_two_inventory_ids_cannot_alias_the_same_path(tmp_path: Path):
    package_path = _make_package(tmp_path)
    package = json.loads(package_path.read_text())
    package["files"][1]["path"] = package["files"][0]["path"]
    package_path.write_text(json.dumps(package), encoding="utf-8")

    report = verify_g3_acquisition_package(package_path)

    assert report["status"] == "inconsistent"
    assert "path duplicates inventory entry" in _error_text(report)


def test_unreadable_package_still_reports_proposal_only_boundary(tmp_path: Path):
    package_path = tmp_path / "invalid.json"
    package_path.write_text("not JSON", encoding="utf-8")

    report = verify_g3_acquisition_package(package_path)

    assert report["status"] == "inconsistent"
    assert report["proposal_only"] is True
    assert report["qualification_status"] == "NOT_QUALIFIED"


def test_deeply_nested_package_returns_report_instead_of_raising(tmp_path: Path):
    package_path = tmp_path / "deep.json"
    package_path.write_text("[" * 2000 + "0" + "]" * 2000, encoding="utf-8")

    report = verify_g3_acquisition_package(package_path)

    assert report["status"] == "inconsistent"
    assert report["qualification_status"] == "NOT_QUALIFIED"
    assert report["errors"]


def test_package_descriptor_symlink_is_rejected(tmp_path: Path):
    package_path = _make_package(tmp_path)
    alias = tmp_path / "package-alias.json"
    alias.symlink_to(package_path)

    report = verify_g3_acquisition_package(alias)

    assert report["status"] == "inconsistent"
    assert "Too many levels" in _error_text(report) or "symlink" in _error_text(report).lower()


def test_standalone_split_manifest_must_equal_embedded_manifest(tmp_path: Path):
    package_path = _make_package(tmp_path)
    package = json.loads(package_path.read_text(encoding="utf-8"))
    split_id = package["evaluator_binding"]["split_manifest"]["file_id"]
    entry = next(item for item in package["files"] if item["file_id"] == split_id)
    split_path = tmp_path / entry["path"]
    split = json.loads(split_path.read_text(encoding="utf-8"))
    split["groups"][0]["split"] = "train"
    _put(tmp_path, package, split_id, json.dumps(split, sort_keys=True).encode())
    package_path.write_text(json.dumps(package), encoding="utf-8")

    report = verify_g3_acquisition_package(package_path)

    assert report["status"] == "inconsistent"
    assert "split-manifest file content differs" in _error_text(report)


def test_crosswalk_must_cover_all_held_out_assignments(tmp_path: Path):
    package_path = _make_package(tmp_path)
    package = json.loads(package_path.read_text(encoding="utf-8"))
    package["row_crosswalk"][0]["evaluator_sample_id"] = "sample.other"
    package_path.write_text(json.dumps(package), encoding="utf-8")

    report = verify_g3_acquisition_package(package_path)

    assert report["status"] == "inconsistent"
    assert "cover every held-out split assignment" in _error_text(report)


def test_angular_and_csi_cross_field_dimensions_are_checked(tmp_path: Path):
    package_path = _make_package(tmp_path)
    package = json.loads(package_path.read_text(encoding="utf-8"))
    package["angular_measurements"][0]["phi_edges_deg"] = [-180.0, -120.0, -60.0, 0.0]
    package["csi_measurements"][0]["shape"][0] += 1
    package_path.write_text(json.dumps(package), encoding="utf-8")

    report = verify_g3_acquisition_package(package_path)

    assert report["status"] == "inconsistent"
    assert "angular" in _error_text(report) and "dimensions must match" in _error_text(report)
    assert "flattened complex values do not match shape" in _error_text(report)


def test_acquisition_crosswalk_must_join_measurement_unit_and_source_files(tmp_path: Path):
    package_path = _make_package(tmp_path)
    package = json.loads(package_path.read_text(encoding="utf-8"))
    package["row_crosswalk"][0]["measurement_unit_id"] = "unit.other.1"
    package["row_crosswalk"][0]["source_file_ids"] = ["file.not-in-sample"]
    package_path.write_text(json.dumps(package), encoding="utf-8")

    report = verify_g3_acquisition_package(package_path)

    assert report["status"] == "inconsistent"
    assert "differs from acquisition sample unit" in _error_text(report)
    assert "not attached to the acquisition sample" in _error_text(report)
