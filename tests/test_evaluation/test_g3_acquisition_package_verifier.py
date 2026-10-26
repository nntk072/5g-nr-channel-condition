"""Adversarial tests for proposal-only G3.03 package consistency checks."""

from __future__ import annotations

import hashlib
import io
import json
import runpy
from pathlib import Path
from typing import Any

import numpy as np
from evaluation.g3_acquisition_package import verify_g3_acquisition_package
from evaluation.g3_joint_acceptance import (
    build_g3_joint_acceptance_family,
    build_g3_joint_acceptance_family_v3,
)
from evaluation.g3_material_calibration import evaluate_g3_material_calibration

PACKAGE_HELPERS = Path(__file__).parents[1] / "test_datasets/test_g3_acquisition_package_v0_2.py"
_package_factory = runpy.run_path(str(PACKAGE_HELPERS))["_package"]
V03_PLUS = {
    "g3.03-acquisition-package.v0.3",
    "g3.03-acquisition-package.v0.4",
}


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
    for binding_name in (
        "scorer_input",
        "split_manifest",
        "evaluator_output",
        "material_mapping",
    ):
        if binding_name not in package["evaluator_binding"]:
            continue
        reference = package["evaluator_binding"][binding_name]
        if reference["file_id"] == file_id:
            reference["exact_bytes_size"] = len(data)
            reference["exact_bytes_sha256"] = hashlib.sha256(data).hexdigest()


def _make_package(root: Path, schema_version="g3.03-acquisition-package.v0.2") -> Path:
    package = _package_factory(schema_version)
    # Keep the synthetic acquisition record aligned with the held-out evaluator row.
    package["samples"][0]["incidence_zenith_deg"] = 20.0
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
    input_schema = "loslab.g3_material_calibration_predictions.v1"
    output_schema = "loslab.g3_material_calibration_evaluation.v1"
    if schema_version in V03_PLUS:
        prediction.pop("measured_angular_power")
        prediction.pop("predicted_angular_power")
        prediction.pop("angular_bin_edges_deg")
        input_schema = "loslab.g3_material_calibration_predictions.v2"
        output_schema = "loslab.g3_material_calibration_evaluation.v2"
        theta_edges = [0.0, 90.0, 180.0]
        phi_edges = [-180.0, 0.0, 180.0]
        solid_angles = [[3.141592653589793, 3.141592653589793]] * 2
        prediction.update(
            angular_power_semantics="integrated_power_per_bin",
            angular_array_order=["theta_bin", "phi_bin; azimuth varies fastest"],
            angular_coordinate_frame_id="frame.example.1",
            theta_edges_deg=theta_edges,
            phi_edges_deg=phi_edges,
            measured_angular_power_per_bin_w=[
                [2.0e-7, 3.0e-7],
                [1.0e-7, 4.0e-7],
            ],
            predicted_angular_power_per_bin_w=[
                [2.1e-7, 2.9e-7],
                [1.1e-7, 3.9e-7],
            ],
            solid_angle_sr=solid_angles,
            missing_angular_bin_indices=[],
        )
    scorer_input = {
        "schema": input_schema,
        "target_frequency_hz": 3.5e9,
        "model_id": "model.example.1",
        "model_sha256": "a" * 64,
        "measurement_calibration_id": package["samples"][0]["calibration_id"],
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
    if schema_version in V03_PLUS:
        scorer_input["joint_acceptance_family"] = build_g3_joint_acceptance_family(
            family_id="synthetic-package-family",
            material_expected_strata=[
                (stratum["material_id"], stratum["polarization"], stratum["incidence_stratum_id"])
            ],
            complex_csi_expected_strata=[("scene-family-a", "LOS")],
        )
    input_bytes = (json.dumps(scorer_input, sort_keys=True, separators=(",", ":")) + "\n").encode()
    if input_schema == "loslab.g3_material_calibration_predictions.v2":
        output = evaluate_g3_material_calibration(scorer_input)
        output["prediction_input_sha256"] = hashlib.sha256(input_bytes).hexdigest()
    else:
        # v0.2 is a legacy structural fixture; there is no current v1 scorer
        # implementation to recompute its illustrative output.
        output = {
            "schema": output_schema,
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
    if schema_version in V03_PLUS:
        package["row_crosswalk"][0]["angular_measurement_id"] = package[
            "angular_measurements"
        ][0]["angular_id"]
    if schema_version in V03_PLUS:
        mapping = {
            "schema": "loslab.g3_material_mapping.v1",
            "material_mapping_id": scorer_input["material_mapping_id"],
            "mapping_basis": "Synthetic proposal fixture only.",
            "materials": [
                {
                    "material_id": stratum["material_id"],
                    "specimen_ids": [package["samples"][0]["specimen_id"]],
                    "material_claims": [
                        next(
                            item["material_claim"]
                            for item in package["specimens"]
                            if item["specimen_id"] == package["samples"][0]["specimen_id"]
                        )
                    ],
                    "evidence_file_ids": ["file.specimen.record"],
                }
            ],
        }
        package["evaluator_binding"]["material_mapping"] = {
            "file_id": "file.material.mapping",
            "exact_bytes_sha256": "0" * 64,
            "exact_bytes_size": 0,
            "byte_scope": "exact_file_bytes",
        }
        mapping_bytes = (
            json.dumps(mapping, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()
        _put(root, package, "file.material.mapping", mapping_bytes)
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
        if file_id not in {
            split_file_id,
            scorer_file_id,
            output_file_id,
            "file.material.mapping",
        }:
            if (
                schema_version == "g3.03-acquisition-package.v0.4"
                and file_id == "file.noise.input"
            ):
                noise_stream = io.BytesIO()
                noise = np.asarray([3.0 + 4.0j, 0.0j, 0.0j, 0.0j], dtype=np.complex128)
                np.save(noise_stream, noise.reshape((1, 1, 2, 2)))
                _put(root, package, file_id, noise_stream.getvalue())
            else:
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


def test_unit_independence_reference_must_have_the_correct_attestation_type(tmp_path: Path):
    package_path = _make_package(tmp_path, "g3.03-acquisition-package.v0.3")
    package = json.loads(package_path.read_text(encoding="utf-8"))
    package["acquisition_units"][0]["independence_attestation_id"] = "attest.rights.plan"
    package_path.write_text(json.dumps(package), encoding="utf-8")

    report = verify_g3_acquisition_package(package_path)

    assert report["status"] == "inconsistent"
    assert "expected physical_independence" in _error_text(report)


def test_calibration_truth_reference_must_have_a_calibration_attestation_type(tmp_path: Path):
    package_path = _make_package(tmp_path, "g3.03-acquisition-package.v0.3")
    package = json.loads(package_path.read_text(encoding="utf-8"))
    package["calibrations"][0]["truth_attestation_id"] = "attest.rights.plan"
    package_path.write_text(json.dumps(package), encoding="utf-8")

    report = verify_g3_acquisition_package(package_path)

    assert report["status"] == "inconsistent"
    assert "expected calibration_traceability or calibration_truth" in _error_text(report)


def test_attestation_ids_must_be_unique_across_legacy_and_external_records(tmp_path: Path):
    package_path = _make_package(tmp_path, "g3.03-acquisition-package.v0.3")
    package = json.loads(package_path.read_text(encoding="utf-8"))
    package["external_attestations"][0]["attestation_id"] = "attest.independent"
    package_path.write_text(json.dumps(package), encoding="utf-8")

    report = verify_g3_acquisition_package(package_path)

    assert report["status"] == "inconsistent"
    assert "attestation ID 'attest.independent' is not globally unique" in _error_text(report)


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


def test_v2_evaluator_metrics_are_recomputed_from_bound_scorer_input(tmp_path: Path):
    package_path = _make_package(tmp_path, "g3.03-acquisition-package.v0.3")
    package = json.loads(package_path.read_text(encoding="utf-8"))
    output_entry = next(
        item for item in package["files"] if item["file_id"] == "file.evaluator.output"
    )
    output_path = tmp_path / output_entry["path"]
    output = json.loads(output_path.read_text(encoding="utf-8"))
    output["numerical_margins_pass_all_reported_strata"] = not output[
        "numerical_margins_pass_all_reported_strata"
    ]
    _put(tmp_path, package, "file.evaluator.output", json.dumps(output, sort_keys=True).encode())
    package_path.write_text(json.dumps(package), encoding="utf-8")

    report = verify_g3_acquisition_package(package_path)

    assert report["status"] == "inconsistent"
    assert (
        "evaluator output differs from deterministic recomputation from scorer input"
        in _error_text(report)
    )


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


def test_crosswalk_cannot_reuse_one_acquisition_sample(tmp_path: Path):
    package_path = _make_package(tmp_path)
    package = json.loads(package_path.read_text(encoding="utf-8"))
    package["row_crosswalk"].append(dict(package["row_crosswalk"][0]))
    package_path.write_text(json.dumps(package), encoding="utf-8")

    report = verify_g3_acquisition_package(package_path)

    assert report["status"] == "inconsistent"
    assert "row_crosswalk repeats acquisition_sample_id" in _error_text(report)


def test_crosswalk_rejects_acquisition_metadata_mismatches(tmp_path: Path):
    mismatches = (
        ("polarization", "H", "acquisition sample polarization differs"),
        ("incidence_zenith_deg", 25.0, "acquisition sample incidence angle differs"),
        ("frequency_hz", 3.6e9, "acquisition sample frequency differs"),
        ("calibration_id", "calibration.other", "acquisition sample calibration differs"),
    )
    for field, value, expected_error in mismatches:
        case_root = tmp_path / field
        package_path = _make_package(case_root, "g3.03-acquisition-package.v0.3")
        package = json.loads(package_path.read_text(encoding="utf-8"))
        package["samples"][0][field] = value
        package_path.write_text(json.dumps(package, indent=2), encoding="utf-8")

        report = verify_g3_acquisition_package(package_path)

        assert expected_error in _error_text(report), field


def test_sample_specimen_must_match_its_acquisition_unit(tmp_path: Path):
    package_path = _make_package(tmp_path, "g3.03-acquisition-package.v0.3")
    package = json.loads(package_path.read_text(encoding="utf-8"))
    second_specimen = dict(package["specimens"][0], specimen_id="specimen.example.2")
    package["specimens"].append(second_specimen)
    package["acquisition_units"][0]["specimen_id"] = second_specimen["specimen_id"]
    package_path.write_text(json.dumps(package, indent=2), encoding="utf-8")

    report = verify_g3_acquisition_package(package_path)

    assert "specimen_id differs from its acquisition unit" in _error_text(report)


def test_sample_acquisition_must_fall_within_calibration_validity(tmp_path: Path):
    package_path = _make_package(tmp_path, "g3.03-acquisition-package.v0.3")
    package = json.loads(package_path.read_text(encoding="utf-8"))
    package["samples"][0]["acquired_utc"] = "2026-01-01T15:00:01Z"
    package_path.write_text(json.dumps(package, indent=2), encoding="utf-8")

    report = verify_g3_acquisition_package(package_path)

    assert "acquired_utc lies outside calibration validity interval" in _error_text(report)


def test_calibration_validity_interval_must_be_ordered(tmp_path: Path):
    package_path = _make_package(tmp_path, "g3.03-acquisition-package.v0.3")
    package = json.loads(package_path.read_text(encoding="utf-8"))
    package["calibrations"][0]["valid_from_utc"] = "2026-01-01T15:00:00Z"
    package["calibrations"][0]["valid_to_utc"] = "2026-01-01T09:00:00Z"
    package_path.write_text(json.dumps(package, indent=2), encoding="utf-8")

    report = verify_g3_acquisition_package(package_path)

    assert "validity interval starts after it ends" in _error_text(report)


def test_sample_must_reference_its_calibration_run_files(tmp_path: Path):
    package_path = _make_package(tmp_path, "g3.03-acquisition-package.v0.3")
    package = json.loads(package_path.read_text(encoding="utf-8"))
    package["samples"][0]["file_ids"].remove("file.cal.run")
    package_path.write_text(json.dumps(package, indent=2), encoding="utf-8")

    report = verify_g3_acquisition_package(package_path)

    assert "does not reference all calibration run evidence files" in _error_text(report)


def test_sample_must_join_campaign_and_its_instrument(tmp_path: Path):
    package_path = _make_package(tmp_path, "g3.03-acquisition-package.v0.3")
    package = json.loads(package_path.read_text(encoding="utf-8"))
    package["campaigns"][0]["instrument_ids"] = ["instrument.other.1"]
    package_path.write_text(json.dumps(package, indent=2), encoding="utf-8")

    report = verify_g3_acquisition_package(package_path)

    assert "instrument is not assigned to its campaign" in _error_text(report)


def test_sample_frequency_must_match_instrument_configuration(tmp_path: Path):
    package_path = _make_package(tmp_path, "g3.03-acquisition-package.v0.3")
    package = json.loads(package_path.read_text(encoding="utf-8"))
    package["instruments"][0]["configuration"]["carrier_frequency_hz"] = 3.6e9
    package_path.write_text(json.dumps(package, indent=2), encoding="utf-8")

    report = verify_g3_acquisition_package(package_path)

    assert "frequency differs from instrument configuration" in _error_text(report)


def test_sample_acquisition_must_fall_within_campaign_window(tmp_path: Path):
    package_path = _make_package(tmp_path, "g3.03-acquisition-package.v0.3")
    package = json.loads(package_path.read_text(encoding="utf-8"))
    package["campaigns"][0]["end_utc"] = "2026-01-01T12:00:00Z"
    package_path.write_text(json.dumps(package, indent=2), encoding="utf-8")

    report = verify_g3_acquisition_package(package_path)

    assert "acquired_utc lies outside campaign interval" in _error_text(report)


def test_sample_file_must_declare_its_sample_association(tmp_path: Path):
    package_path = _make_package(tmp_path, "g3.03-acquisition-package.v0.3")
    package = json.loads(package_path.read_text(encoding="utf-8"))
    file_id = package["samples"][0]["file_ids"][0]
    next(item for item in package["files"] if item["file_id"] == file_id)["sample_ids"] = []
    package_path.write_text(json.dumps(package, indent=2), encoding="utf-8")

    report = verify_g3_acquisition_package(package_path)

    assert f"file {file_id} does not declare the sample association" in _error_text(report)


def test_crosswalk_angular_measurement_must_match_evaluator_input(tmp_path: Path):
    package_path = _make_package(tmp_path, "g3.03-acquisition-package.v0.3")
    package = json.loads(package_path.read_text(encoding="utf-8"))
    package["angular_measurements"][0]["power_per_bin_w"][0][0] = 7.0
    package_path.write_text(json.dumps(package, indent=2), encoding="utf-8")

    report = verify_g3_acquisition_package(package_path)

    assert (
        "acquisition angular measurement power_per_bin_w differs from scorer input "
        "measured_angular_power_per_bin_w"
    ) in _error_text(report)


def test_crosswalk_material_must_match_hash_bound_specimen_mapping(tmp_path: Path):
    package_path = _make_package(tmp_path, "g3.03-acquisition-package.v0.3")
    package = json.loads(package_path.read_text(encoding="utf-8"))
    mapping_file_id = package["evaluator_binding"]["material_mapping"]["file_id"]
    mapping_path = tmp_path / next(
        row["path"] for row in package["files"] if row["file_id"] == mapping_file_id
    )
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    mapping["materials"][0]["material_id"] = "material.other.1"
    _put(tmp_path, package, mapping_file_id, json.dumps(mapping, sort_keys=True).encode())
    package_path.write_text(json.dumps(package, indent=2), encoding="utf-8")

    report = verify_g3_acquisition_package(package_path)

    assert "crosswalk material_id differs from bound specimen material mapping" in _error_text(
        report
    )


def test_material_mapping_exact_bytes_are_inventory_bound(tmp_path: Path):
    package_path = _make_package(tmp_path, "g3.03-acquisition-package.v0.3")
    package = json.loads(package_path.read_text(encoding="utf-8"))
    mapping_file_id = package["evaluator_binding"]["material_mapping"]["file_id"]
    mapping_path = tmp_path / next(
        row["path"] for row in package["files"] if row["file_id"] == mapping_file_id
    )
    mapping_path.write_bytes(mapping_path.read_bytes() + b"tampered")

    report = verify_g3_acquisition_package(package_path)

    assert f"inventory {mapping_file_id}" in _error_text(report)


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


def test_csi_noise_derived_h_floor_must_be_strictly_positive(tmp_path: Path):
    package_path = _make_package(tmp_path, "g3.03-acquisition-package.v0.3")
    package = json.loads(package_path.read_text(encoding="utf-8"))
    package["csi_measurements"][0]["h_floor"]["amplitude_linear"] = 0.0
    package_path.write_text(json.dumps(package), encoding="utf-8")

    report = verify_g3_acquisition_package(package_path)

    assert report["status"] == "inconsistent"
    assert "is less than or equal to the minimum of 0" in _error_text(report)


def test_v04_h_floor_is_recomputed_from_exact_noise_npy_bytes(tmp_path: Path):
    package_path = _make_package(tmp_path, "g3.03-acquisition-package.v0.4")

    report = verify_g3_acquisition_package(package_path)

    assert report["status"] == "consistent_proposal_only", _error_text(report)
    derivation = next(
        check for check in report["checks"] if check["name"] == "csi_noise_floor:csi.example.1"
    )
    assert derivation["status"] == "passed"
    assert "derived=2.5" in derivation["details"]
    assert report["qualification_status"] == "NOT_QUALIFIED"


def test_v04_declared_h_floor_must_match_noise_samples(tmp_path: Path):
    package_path = _make_package(tmp_path, "g3.03-acquisition-package.v0.4")
    package = json.loads(package_path.read_text(encoding="utf-8"))
    package["csi_measurements"][0]["h_floor"]["amplitude_linear"] = 1.0
    package_path.write_text(json.dumps(package), encoding="utf-8")

    report = verify_g3_acquisition_package(package_path)

    assert report["status"] == "inconsistent"
    assert "does not match derived complex-noise RMS" in _error_text(report)


def test_v04_rejects_noise_sample_bound_to_another_sample(tmp_path: Path):
    package_path = _make_package(tmp_path, "g3.03-acquisition-package.v0.4")
    package = json.loads(package_path.read_text(encoding="utf-8"))
    noise_file = next(item for item in package["files"] if item["file_id"] == "file.noise.input")
    noise_file["sample_ids"] = ["sample.other.1"]
    package_path.write_text(json.dumps(package), encoding="utf-8")

    report = verify_g3_acquisition_package(package_path)

    assert report["status"] == "inconsistent"
    assert "must be bound only to the same sample_id" in _error_text(report)


def test_v04_rejects_real_valued_noise_artifacts(tmp_path: Path):
    package_path = _make_package(tmp_path, "g3.03-acquisition-package.v0.4")
    package = json.loads(package_path.read_text(encoding="utf-8"))
    noise_stream = io.BytesIO()
    np.save(noise_stream, np.asarray([1.0, 2.0], dtype=np.float64))
    _put(tmp_path, package, "file.noise.input", noise_stream.getvalue())
    package_path.write_text(json.dumps(package), encoding="utf-8")

    report = verify_g3_acquisition_package(package_path)

    assert report["status"] == "inconsistent"
    assert "must contain complex64 or complex128 values" in _error_text(report)


def test_v04_noise_estimate_shape_must_match_the_csi_grid(tmp_path: Path):
    package_path = _make_package(tmp_path, "g3.03-acquisition-package.v0.4")
    package = json.loads(package_path.read_text(encoding="utf-8"))
    noise_stream = io.BytesIO()
    np.save(noise_stream, np.ones((4,), dtype=np.complex128))
    _put(tmp_path, package, "file.noise.input", noise_stream.getvalue())
    package_path.write_text(json.dumps(package), encoding="utf-8")

    report = verify_g3_acquisition_package(package_path)

    assert report["status"] == "inconsistent"
    assert "must have the same CFR axis shape as the CSI record" in _error_text(report)


def test_v04_rejects_trailing_bytes_after_npy_payload(tmp_path: Path):
    package_path = _make_package(tmp_path, "g3.03-acquisition-package.v0.4")
    package = json.loads(package_path.read_text(encoding="utf-8"))
    noise_stream = io.BytesIO()
    noise = np.asarray([3.0 + 4.0j, 0.0j, 0.0j, 0.0j], dtype=np.complex128)
    np.save(noise_stream, noise.reshape((1, 1, 2, 2)))
    _put(tmp_path, package, "file.noise.input", noise_stream.getvalue() + b"extra")
    package_path.write_text(json.dumps(package), encoding="utf-8")

    report = verify_g3_acquisition_package(package_path)

    assert report["status"] == "inconsistent"
    assert "payload is truncated or has trailing bytes" in _error_text(report)


def test_csi_h_floor_integer_must_fit_finite_float_range(tmp_path: Path):
    package_path = _make_package(tmp_path, "g3.03-acquisition-package.v0.3")
    package = json.loads(package_path.read_text(encoding="utf-8"))
    package["csi_measurements"][0]["h_floor"]["amplitude_linear"] = 10**1000
    package_path.write_text(json.dumps(package), encoding="utf-8")

    report = verify_g3_acquisition_package(package_path)

    assert report["status"] == "inconsistent"
    assert "h_floor amplitude_linear must be finite and strictly positive" in _error_text(report)


def test_v03_csi_noise_inputs_require_noise_measurement_inventory_role(tmp_path: Path):
    package_path = _make_package(tmp_path, "g3.03-acquisition-package.v0.3")
    package = json.loads(package_path.read_text(encoding="utf-8"))
    noise_file_ids = package["csi_measurements"][0]["h_floor"]["noise_input_file_ids"]
    noise_file = next(item for item in package["files"] if item["file_id"] == noise_file_ids[0])
    noise_file["role"] = "reference"
    package_path.write_text(json.dumps(package), encoding="utf-8")

    report = verify_g3_acquisition_package(package_path)

    assert report["status"] == "inconsistent"
    assert "must have inventory role 'noise_measurement'" in _error_text(report)


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


def test_v03_requires_full_sphere_and_exact_spherical_solid_angles(tmp_path: Path):
    package_path = _make_package(tmp_path, schema_version="g3.03-acquisition-package.v0.3")
    report = verify_g3_acquisition_package(package_path)
    assert report["status"] == "consistent_proposal_only", _error_text(report)
    assert report["qualification_status"] == "NOT_QUALIFIED"

    package = json.loads(package_path.read_text(encoding="utf-8"))
    package["angular_measurements"][0]["solid_angle_sr"][0][0] = 2.0
    package_path.write_text(json.dumps(package), encoding="utf-8")
    report = verify_g3_acquisition_package(package_path)
    assert report["status"] == "inconsistent"
    assert "solid_angle_sr must match the spherical bin geometry" in _error_text(report)


def test_v03_rejects_incomplete_azimuth_coverage(tmp_path: Path):
    package_path = _make_package(tmp_path, schema_version="g3.03-acquisition-package.v0.3")
    package = json.loads(package_path.read_text(encoding="utf-8"))
    package["angular_measurements"][0]["phi_edges_deg"][-1] = 179.0
    package_path.write_text(json.dumps(package), encoding="utf-8")
    report = verify_g3_acquisition_package(package_path)
    assert report["status"] == "inconsistent"
    assert "full 360-degree azimuth" in _error_text(report)


def test_v03_does_not_accept_missing_bins_as_a_complete_pattern(tmp_path: Path):
    package_path = _make_package(tmp_path, schema_version="g3.03-acquisition-package.v0.3")
    package = json.loads(package_path.read_text(encoding="utf-8"))
    package["angular_measurements"][0]["missing_bin_indices"] = [[0, 1]]
    package_path.write_text(json.dumps(package), encoding="utf-8")
    report = verify_g3_acquisition_package(package_path)
    assert report["status"] == "inconsistent"
    assert "missing bins and cannot support a complete pattern" in _error_text(report)


def test_v03_verifier_rechecks_embedded_joint_family_digest(tmp_path: Path):
    package_path = _make_package(tmp_path, schema_version="g3.03-acquisition-package.v0.3")
    package = json.loads(package_path.read_text(encoding="utf-8"))
    scorer_id = package["evaluator_binding"]["scorer_input"]["file_id"]
    entry = next(item for item in package["files"] if item["file_id"] == scorer_id)
    scorer_path = tmp_path / entry["path"]
    scorer = json.loads(scorer_path.read_text(encoding="utf-8"))
    scorer["joint_acceptance_family"]["claim_count"] += 1
    _put(tmp_path, package, scorer_id, json.dumps(scorer, sort_keys=True).encode())
    package_path.write_text(json.dumps(package), encoding="utf-8")

    report = verify_g3_acquisition_package(package_path)

    assert report["status"] == "inconsistent"
    assert "joint G3 family is invalid" in _error_text(report)


def test_v03_verifier_requires_the_shared_joint_family(tmp_path: Path):
    package_path = _make_package(tmp_path, schema_version="g3.03-acquisition-package.v0.3")
    package = json.loads(package_path.read_text(encoding="utf-8"))
    scorer_id = package["evaluator_binding"]["scorer_input"]["file_id"]
    entry = next(item for item in package["files"] if item["file_id"] == scorer_id)
    scorer_path = tmp_path / entry["path"]
    scorer = json.loads(scorer_path.read_text(encoding="utf-8"))
    scorer.pop("joint_acceptance_family")
    _put(tmp_path, package, scorer_id, json.dumps(scorer, sort_keys=True).encode())
    package_path.write_text(json.dumps(package), encoding="utf-8")

    report = verify_g3_acquisition_package(package_path)

    assert report["status"] == "inconsistent"
    assert "v0.3 scorer input requires the shared joint G3 acceptance family" in _error_text(report)


def test_v03_verifier_checks_joint_family_material_roster_against_input(tmp_path: Path):
    package_path = _make_package(tmp_path, schema_version="g3.03-acquisition-package.v0.3")
    package = json.loads(package_path.read_text(encoding="utf-8"))
    scorer_id = package["evaluator_binding"]["scorer_input"]["file_id"]
    entry = next(item for item in package["files"] if item["file_id"] == scorer_id)
    scorer_path = tmp_path / entry["path"]
    scorer = json.loads(scorer_path.read_text(encoding="utf-8"))
    scorer["joint_acceptance_family"] = build_g3_joint_acceptance_family(
        family_id="mismatched-synthetic-family",
        material_expected_strata=[("other-material", "V", "other-incidence")],
        complex_csi_expected_strata=[("scene-family-a", "LOS")],
    )
    _put(tmp_path, package, scorer_id, json.dumps(scorer, sort_keys=True).encode())
    package_path.write_text(json.dumps(package), encoding="utf-8")

    report = verify_g3_acquisition_package(package_path)

    assert report["status"] == "inconsistent"
    assert "expected strata differ from the shared G3 family roster" in _error_text(report)


def test_v03_verifier_accepts_v3_family_binding_receiver_claims(tmp_path: Path):
    package_path = _make_package(tmp_path, schema_version="g3.03-acquisition-package.v0.3")
    package = json.loads(package_path.read_text(encoding="utf-8"))
    scorer_id = package["evaluator_binding"]["scorer_input"]["file_id"]
    entry = next(item for item in package["files"] if item["file_id"] == scorer_id)
    scorer_path = tmp_path / entry["path"]
    scorer = json.loads(scorer_path.read_text(encoding="utf-8"))
    scorer["joint_acceptance_family"] = build_g3_joint_acceptance_family_v3(
        family_id="synthetic-v3-family",
        material_expected_strata=[("material.example.1", "V", "incidence.example.1")],
        complex_csi_expected_strata=[("scene-family-a", "LOS")],
        propagation_expected_strata=[("scene-family-a", "LOS", "total")],
        receiver_impact_expected_strata=[("scene-family-a", "LOS")],
    )
    scorer_bytes = (json.dumps(scorer, sort_keys=True, separators=(",", ":")) + "\n").encode()
    _put(tmp_path, package, scorer_id, scorer_bytes)
    package["evaluator_binding"]["reported_prediction_input_sha256"] = hashlib.sha256(
        scorer_bytes
    ).hexdigest()
    output_id = package["evaluator_binding"]["evaluator_output"]["file_id"]
    output = evaluate_g3_material_calibration(scorer)
    output["prediction_input_sha256"] = hashlib.sha256(scorer_bytes).hexdigest()
    _put(
        tmp_path,
        package,
        output_id,
        (json.dumps(output, sort_keys=True, separators=(",", ":")) + "\n").encode(),
    )
    package_path.write_text(json.dumps(package), encoding="utf-8")

    report = verify_g3_acquisition_package(package_path)

    assert report["status"] == "consistent_proposal_only", _error_text(report)
    assert report["qualification_status"] == "NOT_QUALIFIED"
