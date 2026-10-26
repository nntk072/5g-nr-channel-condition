"""Validate the proposal-only G3.03 acquisition package v0.2 contract."""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "datasets/schema/g3.03-acquisition-package.v0.2.schema.json"
V01_FIXTURE_PATH = ROOT / "datasets/schema/examples/g3.03-acquisition-package.v0.1.example.json"
V03_PLUS = {
    "g3.03-acquisition-package.v0.3",
    "g3.03-acquisition-package.v0.4",
}


def _package(schema_version="g3.03-acquisition-package.v0.2"):
    package = json.loads(V01_FIXTURE_PATH.read_text(encoding="utf-8"))
    package.update(
        schema_version=schema_version,
        proposal_status="proposal_only",
        package_description=(
            "SYNTHETIC TEST FIXTURE ONLY: fictional records; no measurements, evaluator run, "
            "external verification, lab acceptance, or G3 qualification."
        ),
    )
    if schema_version in V03_PLUS:
        noise_file_ids = {
            file_id
            for record in package["csi_measurements"]
            for file_id in record["h_floor"]["noise_input_file_ids"]
        }
        for file_record in package["files"]:
            if file_record["file_id"] in noise_file_ids:
                file_record["role"] = "noise_measurement"
                if schema_version == "g3.03-acquisition-package.v0.4":
                    file_record["path"] = "files/noise-input.npy"
                    file_record["media_type"] = "application/x-npy"
                    file_record["sample_ids"] = [package["csi_measurements"][0]["sample_id"]]
        if schema_version == "g3.03-acquisition-package.v0.4":
            for record in package["csi_measurements"]:
                record["h_floor"]["method"] = "noise_only_cfr_estimate_rms_v1"
                record["h_floor"]["amplitude_linear"] = 2.5
    for collection in ("samples", "fits", "angular_measurements", "csi_measurements"):
        for record in package[collection]:
            record["record_status"] = "synthetic_example_only"
            if collection == "fits":
                for parameter in record["parameters"]:
                    parameter["record_status"] = "synthetic_example_only"
    # Keep the v0.2 historical fixture; v0.3 closes the sphere at +180 degrees.
    if schema_version == "g3.03-acquisition-package.v0.2":
        for record in package["angular_measurements"]:
            record["phi_edges_deg"] = [-180, 0, 179]
    for attestation in package["attestations"].values():
        attestation["attestation_status"] = "planned_not_signed"
        attestation["claim"] = "SYNTHETIC EXAMPLE: proposed wording only; no claim is made."
        attestation["signer_id"] = "signer.example.proposed"
        attestation["signed_utc"] = None
        attestation["signature_value"] = None
    package["files"].extend(
        [
            _file("file.scorer.input", "scorer-input.json", "scorer_input"),
            _file("file.evaluator.output", "evaluator-output.json", "evaluator_output"),
            _file("file.split.manifest", "split-manifest.json", "split_manifest"),
            _file("file.campaign", "campaign.json", "campaign"),
            _file("file.instrument", "instrument.json", "instrument"),
            _file("file.protocol", "protocol.json", "protocol"),
            _file("file.uncertainty", "uncertainty.json", "uncertainty"),
        ]
        + (
            [_file("file.material.mapping", "material-mapping.json", "material_mapping")]
            if schema_version in V03_PLUS
            else []
        )
    )
    package.update(
        campaigns=[
            {
                "campaign_id": "campaign.example.1",
                "study_id": package["study_id"],
                "campaign_status": "proposed",
                "start_utc": "2026-01-01T09:00:00Z",
                "end_utc": "2026-01-01T15:00:00Z",
                "site": "fictional laboratory",
                "operator_role": "proposed operator role",
                "instrument_ids": ["instrument.example.1"],
                "protocol_id": "protocol.example.1",
                "coordinate_frame": "fictional local frame",
                "survey_method": "proposed survey method",
                "survey_uncertainty_m": 0.001,
                "environment": {"temperature_c": 20.0, "humidity_percent": 50.0},
                "record_file_ids": ["file.campaign"],
            }
        ],
        instruments=[
            {
                "instrument_id": "instrument.example.1",
                "make": "fictional make",
                "model": "fictional model",
                "serial_number_reference": "redacted proposal field",
                "firmware": "proposed firmware field",
                "tx_rx_chain": "proposed chain configuration",
                "antenna_ids": ["antenna.example.1"],
                "configuration": {"carrier_frequency_hz": 3.5e9, "if_bandwidth_hz": 1e6},
                "configuration_file_ids": ["file.instrument"],
            }
        ],
        protocols=[
            {
                "protocol_id": "protocol.example.1",
                "protocol_status": "proposed",
                "target_frequency_hz": 3500000000,
                "evaluator_schema": (
                    "loslab.g3_material_calibration_predictions.v2"
                    if schema_version in V03_PLUS
                    else "loslab.g3_material_calibration_predictions.v1"
                ),
                "criteria_reference": "Frozen G3.03 evaluator criteria; no criteria redefined here.",
                "physical_unit_definition": "Proposed independent acquisition unit definition.",
                "expected_strata": [
                    {
                        "material_id": "material.example.1",
                        "polarization": "V",
                        "incidence_stratum_id": "incidence.example.1",
                    }
                ],
                "split_manifest_file_id": "file.split.manifest",
                "split_manifest_canonical_sha256": "1" * 64,
                "external_verification_id": "verification.example.1",
                "bootstrap_seed": 123,
                "bootstrap_replicates": 1000,
                "uncertainty_method": "Proposed uncertainty propagation and cluster method.",
                "protocol_file_ids": ["file.protocol"],
            }
        ],
        evaluator_binding={
            "execution_status": "not_executed_proposal_only",
            "input_contract": (
                "loslab.g3_material_calibration_predictions.v2"
                if schema_version in V03_PLUS
                else "loslab.g3_material_calibration_predictions.v1"
            ),
            "output_contract": (
                "loslab.g3_material_calibration_evaluation.v2"
                if schema_version in V03_PLUS
                else "loslab.g3_material_calibration_evaluation.v1"
            ),
            "scorer_input": _artifact_ref("file.scorer.input"),
            "split_manifest": _artifact_ref("file.split.manifest"),
            "evaluator_output": _artifact_ref("file.evaluator.output"),
            **(
                {"material_mapping": _artifact_ref("file.material.mapping")}
                if schema_version in V03_PLUS
                else {}
            ),
            "reported_prediction_input_sha256": "0" * 64,
            "evaluator_code_sha256": "2" * 64,
            "run_configuration_file_ids": ["file.protocol"],
        },
        row_crosswalk=[
            {
                "acquisition_sample_id": "sample.example.1",
                "evaluator_sample_id": "sample.example.1",
                "input_row_pointer": "/test_predictions/0",
                "output_stratum_pointer": "/strata/0",
                "incidence_group_id": "group.example.1",
                "split": "test",
                "material_id": "material.example.1",
                "polarization": "V",
                "incidence_stratum_id": "incidence.example.1",
                "measurement_unit_id": "unit.example.1",
                "measurement_unit_provenance_id": "unit.provenance.example.1",
                "source_file_ids": ["file.raw.scan1"],
            }
        ],
        uncertainty_records=[
            {
                "uncertainty_id": "uncertainty.example.1",
                "scope": "sample",
                "sample_ids": ["sample.example.1"],
                "components": [
                    {
                        "source": "fictional calibration uncertainty",
                        "standard_uncertainty": 0.01,
                        "unit": "W",
                        "distribution": "normal",
                        "correlation_group": "cal.example.1",
                    }
                ],
                "dependence_model": "Proposed repeat and unit dependence description.",
                "propagation_method": "Proposed propagation method; not executed.",
                "repeatability_file_ids": ["file.raw.scan1"],
                "uncertainty_record_file_ids": ["file.uncertainty"],
            }
        ],
        external_attestations=[
            _planned_attestation("attest.split.plan", "split_preregistration"),
            _planned_attestation("attest.independence.plan", "physical_independence"),
            _planned_attestation("attest.calibration.plan", "calibration_traceability"),
            _planned_attestation("attest.rights.plan", "rights_review"),
            _planned_attestation("attest.review.plan", "scientific_review"),
        ],
    )
    package["per_file_rights"] = [
        {
            "file_id": file["file_id"],
            "rights_holder": "To be established; proposal field only.",
            "license_or_basis": "To be reviewed before acquisition or release.",
            "research_use": "unknown",
            "publication": "unknown",
            "redistribution": "unknown",
            "attribution": "To be determined from verified rights evidence.",
            "restrictions": [],
            "evidence_file_ids": [],
            "attestation_id": "attest.rights.plan",
        }
        for file in package["files"]
    ]
    if schema_version in V03_PLUS:
        package["samples"][0].update(
            campaign_id="campaign.example.1",
            instrument_id="instrument.example.1",
        )
        for angular in package["angular_measurements"]:
            angular["coordinate_frame_id"] = "frame.example.1"
        package["row_crosswalk"][0]["angular_measurement_id"] = package[
            "angular_measurements"
        ][0]["angular_id"]
    return package


def _file(file_id, name, role):
    return {
        "file_id": file_id,
        "path": f"proposal/{name}",
        "size_bytes": 0,
        "media_type": "application/json",
        "sha256": "0" * 64,
        "role": role,
        "dependency_file_ids": [],
        "sample_ids": [],
        "specimen_ids": [],
        "calibration_ids": [],
    }


def _artifact_ref(file_id):
    return {
        "file_id": file_id,
        "exact_bytes_sha256": "0" * 64,
        "exact_bytes_size": 0,
        "byte_scope": "exact_file_bytes",
    }


def _planned_attestation(attestation_id, attestation_type):
    return {
        "attestation_id": attestation_id,
        "attestation_type": attestation_type,
        "attestation_status": "planned_not_signed",
        "claim_template": "Proposed claim wording; no assertion has been made.",
        "subject_file_ids": ["file.split.manifest"],
        "signer_role": "independent custodian role to be appointed",
        "signature_algorithm": "To be selected by the receiving institution.",
        "signature_file_id": None,
        "signature_value": None,
        "signed_utc": None,
        "verification_id": "verification.example.1",
        "verifying_organization": "independent organization to be selected",
        "verification_method": "Proposed external verification procedure; not performed.",
        "verification_record_file_ids": [],
    }


def _assert_cross_record_links(package):
    files = {item["file_id"]: item for item in package["files"]}
    binding = package["evaluator_binding"]
    protocol = package["protocols"][0]
    for name, role in (
        ("scorer_input", "scorer_input"),
        ("split_manifest", "split_manifest"),
        ("evaluator_output", "evaluator_output"),
    ) + (
        (("material_mapping", "material_mapping"),)
        if package["schema_version"] == "g3.03-acquisition-package.v0.3"
        else ()
    ):
        reference = binding[name]
        inventory = files[reference["file_id"]]
        assert inventory["role"] == role
        assert reference["exact_bytes_sha256"] == inventory["sha256"]
        assert reference["exact_bytes_size"] == inventory["size_bytes"]
    assert protocol["split_manifest_file_id"] == binding["split_manifest"]["file_id"]
    assert (
        binding["reported_prediction_input_sha256"] == binding["scorer_input"]["exact_bytes_sha256"]
    )
    verification_ids = {item["verification_id"] for item in package["external_attestations"]}
    assert protocol["external_verification_id"] in verification_ids
    required_attestation_types = {
        "split_preregistration",
        "physical_independence",
        "calibration_traceability",
        "rights_review",
    }
    assert required_attestation_types <= {
        item["attestation_type"] for item in package["external_attestations"]
    }

    sample_ids = {item["sample_id"] for item in package["samples"]}
    attestation_ids = {item["attestation_id"] for item in package["external_attestations"]}
    attestation_ids.update(item["attestation_id"] for item in package["attestations"].values())
    for row in package["row_crosswalk"]:
        assert row["acquisition_sample_id"] in sample_ids
        assert row["evaluator_sample_id"]
        assert row["input_row_pointer"].startswith("/test_predictions/")
        assert set(row["source_file_ids"]) <= files.keys()
    pointers = [item["input_row_pointer"] for item in package["row_crosswalk"]]
    assert len(pointers) == len(set(pointers))
    evaluator_sample_ids = [item["evaluator_sample_id"] for item in package["row_crosswalk"]]
    assert len(evaluator_sample_ids) == len(set(evaluator_sample_ids))
    rights = package["per_file_rights"]
    rights_file_ids = [item["file_id"] for item in rights]
    assert len(rights_file_ids) == len(set(rights_file_ids))
    assert set(rights_file_ids) == files.keys()
    for item in rights:
        assert item["attestation_id"] in attestation_ids
        assert set(item["evidence_file_ids"]) <= files.keys()


@pytest.fixture(scope="module")
def schema_validator():
    jsonschema = pytest.importorskip(
        "jsonschema", reason="jsonschema is a declared repository dev dependency"
    )
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker())


def test_proposal_fixture_validates_and_is_explicitly_unexecuted(schema_validator):
    package = _package()
    assert list(schema_validator.iter_errors(package)) == []
    assert package["proposal_status"] == "proposal_only"
    assert package["evaluator_binding"]["execution_status"] == "not_executed_proposal_only"
    assert all(
        item["attestation_status"] == "planned_not_signed"
        for item in package["external_attestations"]
    )
    assert all(item["signature_value"] is None for item in package["external_attestations"])
    for collection in ("samples", "fits", "angular_measurements", "csi_measurements"):
        assert all(
            item["record_status"] == "synthetic_example_only" for item in package[collection]
        )
    assert all(
        item["attestation_status"] == "planned_not_signed"
        and item["signed_utc"] is None
        and item["signature_value"] is None
        for item in package["attestations"].values()
    )


def test_v04_proposal_fixture_validates_noise_floor_derivation_contract():
    jsonschema = pytest.importorskip("jsonschema")
    package = _package("g3.03-acquisition-package.v0.4")
    schema = json.loads(
        (ROOT / "datasets/schema/g3.03-acquisition-package.v0.4.schema.json").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker())
    assert list(validator.iter_errors(package)) == []


def test_exact_byte_references_and_per_file_rights_join(schema_validator):
    package = _package()
    assert list(schema_validator.iter_errors(package)) == []
    _assert_cross_record_links(package)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.update(proposal_status="accepted"),
        lambda p: p["evaluator_binding"].update(execution_status="executed"),
        lambda p: p["external_attestations"][0].update(attestation_status="signed"),
        lambda p: p["external_attestations"][0].update(signature_value="not-a-signature"),
        lambda p: p["attestations"]["physical_independence"].update(attestation_status="signed"),
        lambda p: p["attestations"]["physical_independence"].update(
            signature_value="not-a-signature"
        ),
        lambda p: p["attestations"]["physical_independence"].update(
            signed_utc="2026-01-01T12:00:00Z"
        ),
        lambda p: p["samples"][0].update(record_status="measured"),
        lambda p: p["fits"][0]["parameters"][0].update(record_status="observed"),
        lambda p: p["protocols"][0].update(bootstrap_replicates=999),
    ],
)
def test_schema_rejects_nonproposal_or_malformed_fields(mutate, schema_validator):
    package = _package()
    mutate(package)
    assert list(schema_validator.iter_errors(package))


def test_semantic_audit_detects_inventory_digest_disagreement():
    package = _package()
    package["evaluator_binding"]["scorer_input"]["exact_bytes_sha256"] = "f" * 64
    with pytest.raises(AssertionError):
        _assert_cross_record_links(package)


def test_semantic_audit_detects_uncovered_inventory_file():
    package = _package()
    package["per_file_rights"].pop()
    with pytest.raises(AssertionError):
        _assert_cross_record_links(package)


def test_semantic_audit_detects_crosswalk_sample_mismatch():
    package = _package()
    package["row_crosswalk"][0]["acquisition_sample_id"] = "sample.unknown.1"
    with pytest.raises(AssertionError):
        _assert_cross_record_links(package)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("phi_edges_deg", [-180, 0, 180]),
        ("phi_edges_deg", [-180, 0, 0]),
        ("theta_edges_deg", [0, 90, 90]),
    ],
)
def test_angular_edges_reject_out_of_domain_or_duplicate_values(field, bad_value, schema_validator):
    package = _package()
    package["angular_measurements"][0][field] = bad_value
    assert list(schema_validator.iter_errors(package))


def test_csi_shape_must_have_exactly_four_positive_dimensions(schema_validator):
    package = _package()
    csi = package["csi_measurements"][0]
    csi["shape"] = [1, 1, 2]
    assert list(schema_validator.iter_errors(package))

    csi["shape"] = [1, 1, 2, 0]
    assert list(schema_validator.iter_errors(package))
