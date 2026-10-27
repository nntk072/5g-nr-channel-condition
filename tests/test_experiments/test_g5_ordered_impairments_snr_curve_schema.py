"""Schema checks for diagnostic independent-channel SNR curves."""

import hashlib
import json
from pathlib import Path
from typing import Any

import experiments.audit_g5_ordered_impairments_result as curve_auditor
import experiments.g5_acceptance as g5_acceptance
import pytest
from experiments.audit_g5_ordered_impairments_result import _curve_report_schema_status


def _curve(version: int = 1) -> dict[str, Any]:
    point_config = {
        "channel_seeds": [7],
        "blocks_per_channel": 1,
        "waveform_awgn_complex_variance": 1.0,
        "nominal_snr_db": 0.0,
        "mean_transmit_waveform_power": 1.0,
        "transmit_waveform_power_by_channel": [1.0],
        "noisy_receiver_input_sha256_by_channel": ["0" * 64],
    }
    return {
        "schema": f"umcd.g5_ordered_impairments_independent_channels_snr_curve.v{version}",
        "scope": "paired diagnostic synthetic curve",
        "inference_scope": "descriptive per-point outcomes",
        "config": {
            "schema_sha256": "0" * 64,
            "snr_db": [0.0],
            "channel_seeds": [7],
            "blocks_per_channel": 1,
            "channel_estimator": "sionna_default_ls",
            "shared_random_streams_across_snr": [
                "channel",
                "payload",
                "TX phase noise",
                "RX phase noise",
                "standardized AWGN",
            ],
            "snr_definition": "unit-reference complex waveform power",
            "transport_blocks_per_snr_point": 1,
            "transport_blocks_across_curve": 1,
            "maximum_transport_blocks_across_curve": 16_384,
            **({"maximum_snr_points": 64} if version >= 2 else {}),
        },
        "acceptance": {
            "criteria_frozen_before_run": {
                "status": "descriptive_only",
                "reason": "single-point criteria are not transferred to the SNR family",
            },
            "pass": False,
        },
        "curve_inference": {
            "status": "available",
            "unit": "independent synthetic channel",
            "cluster_count": 2,
            "bootstrap_seed": 75_013,
            "bootstrap_draws": 4_096,
            "family_size": 4 if version >= 4 else 3,
            "method": "channel-cluster bootstrap",
            "simultaneous_radius": 0.1,
            **(
                {
                    "paired_contrast_definition": "uncorrected macro-BLER minus corrected macro-BLER; positive values favor correction"
                }
                if version >= 4
                else {}
            ),
            "by_condition": {
                condition: [
                    {
                        "snr_db": 0.0,
                        "mean_channel_macro_bler": 0.0,
                        "pointwise_95_percent_interval": [0.0, 0.1],
                        "simultaneous_95_percent_interval": [0.0, 0.1],
                    }
                ]
                for condition in ("clean", "uncorrected", "corrected")
            },
            **({"by_paired_contrast": []} if version >= 4 else {}),
        },
        "points": [
            {
                "snr_db": 0.0,
                "experiment": {
                    "schema": "umcd.g5_ordered_impairments_independent_channels.v15",
                    "config": point_config,
                    "paired_outcomes": {},
                    "acceptance": {},
                },
            }
        ],
    }


def _receiver_module():
    """Load receiver-bound helpers only for checks that require its runtime."""
    return pytest.importorskip("experiments.g5_ordered_impairments_receiver")


def test_independent_channel_snr_curve_schema_accepts_diagnostic_shape() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema_path = (
        Path(__file__).resolve().parents[2]
        / "datasets"
        / "schema"
        / "g5-ordered-impairments-snr-curve.v1.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)

    jsonschema.Draft202012Validator(schema).validate(_curve())


def test_cuda_curve_schemas_are_versioned_and_bind_device_identity() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    root = Path(__file__).resolve().parents[2] / "datasets" / "schema"
    for version, base in {12: 9, 13: 7, 14: 8, 15: 10, 16: 11}.items():
        path = root / f"g5-ordered-impairments-snr-curve.v{version}.schema.json"
        schema = json.loads(path.read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator.check_schema(schema)
        assert schema["properties"]["schema"]["const"].endswith(f".v{version}")
        assert "execution_device" in schema["properties"]["config"]["required"]
        assert schema["properties"]["config"]["properties"]["execution_device"] == {
            "type": "string",
            "pattern": "^cuda:[0-9]+$",
        }
        assert schema["properties"]["points"]["items"]["properties"]["experiment"]["properties"][
            "schema"
        ]["enum"][-4:] == [
            "umcd.g5_ordered_impairments_independent_channels.v21",
            "umcd.g5_ordered_impairments_independent_channels.v22",
            "umcd.g5_ordered_impairments_independent_channels.v23",
            "umcd.g5_ordered_impairments_independent_channels.v24",
        ]
        assert (
            schema["properties"]["acceptance"]
            == json.loads(
                (root / f"g5-ordered-impairments-snr-curve.v{base}.schema.json").read_text(
                    encoding="utf-8"
                )
            )["properties"]["acceptance"]
        )


def test_cuda_diagnostic_curve_schema_requires_explicit_cuda_device() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    path = (
        Path(__file__).resolve().parents[2]
        / "datasets"
        / "schema"
        / "g5-ordered-impairments-snr-curve.v12.schema.json"
    )
    schema = json.loads(path.read_text(encoding="utf-8"))
    curve = _curve(version=12)
    curve["config"]["execution_device"] = "cuda:0"
    curve["points"][0]["experiment"]["schema"] = (
        "umcd.g5_ordered_impairments_independent_channels.v21"
    )
    curve["points"][0]["experiment"]["config"]["execution_device"] = "cuda:0"

    validator = jsonschema.Draft202012Validator(schema)
    assert not list(validator.iter_errors(curve))
    del curve["config"]["execution_device"]
    assert list(validator.iter_errors(curve))


def test_independent_channel_snr_curve_v2_schema_accepts_bounded_grid() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema_path = (
        Path(__file__).resolve().parents[2]
        / "datasets"
        / "schema"
        / "g5-ordered-impairments-snr-curve.v2.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)

    jsonschema.Draft202012Validator(schema).validate(_curve(version=2))


def test_curve_auditor_requires_the_complete_versioned_schema() -> None:
    pytest.importorskip("jsonschema")
    schema_path = (
        Path(__file__).resolve().parents[2]
        / "datasets"
        / "schema"
        / "g5-ordered-impairments-snr-curve.v2.schema.json"
    )
    report = _curve(version=2)

    expected_digest = hashlib.sha256(schema_path.read_bytes()).hexdigest()
    assert _curve_report_schema_status(report, schema_path) == (expected_digest, True)
    invalid = dict(report, unrecognized_claim=True)
    assert _curve_report_schema_status(invalid, schema_path) == (expected_digest, False)


def test_curve_result_hash_verifier_rejects_schema_drift() -> None:
    pytest.importorskip("jsonschema")
    receiver = _receiver_module()
    schema_path = (
        Path(__file__).resolve().parents[2]
        / "datasets"
        / "schema"
        / "g5-ordered-impairments-snr-curve.v2.schema.json"
    )
    report = _curve(version=2)
    report["config"]["schema_sha256"] = hashlib.sha256(schema_path.read_bytes()).hexdigest()
    valid_result = receiver.build_hash_bound_result(report, created_utc="2026-10-26T00:00:00+00:00")
    assert receiver.verify_hash_bound_result(valid_result)

    invalid_report = dict(report, unrecognized_claim=True)
    invalid_result = receiver.build_hash_bound_result(
        invalid_report, created_utc="2026-10-26T00:00:00+00:00"
    )
    assert not receiver.verify_hash_bound_result(invalid_result)


def test_independent_channel_snr_curve_v2_schema_rejects_more_than_64_points() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema_path = (
        Path(__file__).resolve().parents[2]
        / "datasets"
        / "schema"
        / "g5-ordered-impairments-snr-curve.v2.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    curve = _curve(version=2)
    curve["config"]["snr_db"] = [float(value) for value in range(65)]
    curve["config"]["transport_blocks_per_snr_point"] = 1
    curve["config"]["transport_blocks_across_curve"] = 65
    curve["points"] = [
        {
            "snr_db": float(value),
            "experiment": curve["points"][0]["experiment"],
        }
        for value in range(65)
    ]

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(curve)


def test_signed_snr_acceptance_contract_binds_exact_canonical_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    acceptance = g5_acceptance
    criteria = {
        "schema": "umcd.g5_ordered_impairments_snr_acceptance.v1",
        "snr_db": [0.0],
    }
    payload = acceptance._canonical_json_bytes(criteria)
    contract_path = tmp_path / "contract.json"
    signature_path = tmp_path / "contract.sig"
    allowed_signers_path = tmp_path / "allowed_signers"
    contract_path.write_bytes(payload)
    signature_path.write_bytes(b"signature fixture")
    allowed_signers_path.write_bytes(b"allowed signers fixture")
    observed: dict[str, Any] = {}

    def verify(
        observed_payload: bytes,
        observed_signature_path: Path,
        observed_allowed_signers_path: Path,
        *,
        signer_id: str,
        namespace: str,
    ) -> dict[str, str]:
        observed.update(
            {
                "payload": observed_payload,
                "signature_path": observed_signature_path,
                "allowed_signers_path": observed_allowed_signers_path,
                "signer_id": signer_id,
                "namespace": namespace,
            }
        )
        return {
            "signer_id": signer_id,
            "namespace": namespace,
            "signature_sha256": "1" * 64,
            "allowed_signers_sha256": "2" * 64,
        }

    monkeypatch.setattr(acceptance, "verify_openssh_signature_for_bytes", verify)
    loaded, identity = acceptance.load_g5_snr_acceptance_contract(
        contract_path,
        signature_path=signature_path,
        trusted_allowed_signers_path=allowed_signers_path,
        signer_id="reviewer@example.org",
    )

    assert loaded == criteria
    assert identity is not None
    assert observed["payload"] == payload
    assert observed["namespace"] == "umcd-g5-snr-acceptance-v1"
    assert identity["status"] == "verified_before_receiver_run"
    assert identity["contract_payload_sha256"] == hashlib.sha256(payload).hexdigest()
    assert identity["signature_sha256"] == "1" * 64


def test_signed_snr_acceptance_contract_rejects_noncanonical_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    acceptance = g5_acceptance
    criteria = {"schema": "umcd.g5_ordered_impairments_snr_acceptance.v1", "snr_db": [0.0]}
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(criteria, indent=2), encoding="utf-8")
    signature_path = tmp_path / "contract.sig"
    signature_path.write_bytes(b"signature fixture")
    allowed_signers_path = tmp_path / "allowed_signers"
    allowed_signers_path.write_bytes(b"allowed signers fixture")
    monkeypatch.setattr(
        acceptance,
        "verify_openssh_signature_for_bytes",
        lambda *_args, **_kwargs: pytest.fail("noncanonical bytes must fail before verification"),
    )

    with pytest.raises(ValueError, match="canonical sorted compact JSON"):
        acceptance.load_g5_snr_acceptance_contract(
            contract_path,
            signature_path=signature_path,
            trusted_allowed_signers_path=allowed_signers_path,
            signer_id="reviewer@example.org",
        )


def test_v6_curve_schema_requires_signed_preregistration_metadata() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema_path = (
        Path(__file__).resolve().parents[2]
        / "datasets"
        / "schema"
        / "g5-ordered-impairments-snr-curve.v6.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    contract = {
        "schema": "umcd.g5_ordered_impairments_snr_acceptance.v1",
        "snr_db": [0.0],
        "channel_seeds": list(range(20)),
        "channel_estimator": "sionna_default_ls",
        "blocks_per_channel": 1,
        "minimum_independent_channels": 20,
        "maximum_corrected_macro_bler": [0.1],
        "minimum_uncorrected_minus_corrected_macro_bler": [0.05],
    }
    frozen = {
        "status": "contract_supplied_before_run",
        "criteria": contract,
        "criteria_sha256": "0" * 64,
        "confidence_level": 0.95,
        "interval_scope": "simultaneous across all three conditions and paired improvements",
        "decision_unit": "independent synthetic channel; transport blocks are nested",
        "preregistration_signature": {
            "status": "verified_before_receiver_run",
            "namespace": "umcd-g5-snr-acceptance-v1",
            "signer_id": "reviewer@example.org",
            "contract_payload_sha256": "0" * 64,
            "signature_sha256": "1" * 64,
            "trusted_allowed_signers_sha256": "2" * 64,
            "trust_file_custody": "not_established_by_this_verifier",
        },
    }
    acceptance = {
        "criteria_frozen_before_run": frozen,
        "status": "NOT_EVALUABLE",
        "pass": False,
    }

    acceptance_schema = {
        "$schema": schema["$schema"],
        "$defs": schema["$defs"],
        **schema["properties"]["acceptance"],
    }
    validator = jsonschema.Draft202012Validator(acceptance_schema)
    validator.validate(acceptance)
    del frozen["preregistration_signature"]
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(acceptance)


def test_curve_auditor_reverifies_signed_preregistration_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = {
        "schema": "umcd.g5_ordered_impairments_snr_acceptance.v1",
        "snr_db": [0.0],
    }
    payload_digest = hashlib.sha256(curve_auditor._canonical_json_bytes(contract)).hexdigest()
    signature_record = {
        "status": "verified_before_receiver_run",
        "namespace": "umcd-g5-snr-acceptance-v1",
        "signer_id": "reviewer@example.org",
        "contract_payload_sha256": payload_digest,
        "signature_sha256": "1" * 64,
        "trusted_allowed_signers_sha256": "2" * 64,
        "trust_file_custody": "not_established_by_this_verifier",
    }
    observed: dict[str, Any] = {}

    def verify(
        payload: bytes,
        signature_path: Path,
        allowed_signers_path: Path,
        *,
        signer_id: str,
        namespace: str,
    ) -> dict[str, str]:
        observed.update(
            {
                "payload": payload,
                "signature_path": signature_path,
                "allowed_signers_path": allowed_signers_path,
                "signer_id": signer_id,
                "namespace": namespace,
            }
        )
        return {
            "signature_sha256": "1" * 64,
            "allowed_signers_sha256": "2" * 64,
        }

    monkeypatch.setattr(curve_auditor, "verify_openssh_signature_for_bytes", verify)
    signature_path = tmp_path / "contract.sig"
    allowed_signers_path = tmp_path / "allowed_signers"

    assert curve_auditor._verify_g5_preregistration_signature(
        contract,
        signature_record,
        signature_path=signature_path,
        trusted_allowed_signers_path=allowed_signers_path,
    )
    assert observed["payload"] == curve_auditor._canonical_json_bytes(contract)
    assert observed["signer_id"] == "reviewer@example.org"
    assert observed["namespace"] == "umcd-g5-snr-acceptance-v1"

    signature_record["trusted_allowed_signers_sha256"] = "3" * 64
    assert not curve_auditor._verify_g5_preregistration_signature(
        contract,
        signature_record,
        signature_path=signature_path,
        trusted_allowed_signers_path=allowed_signers_path,
    )


def test_independent_channel_snr_curve_v3_schema_accepts_condition_waveforms_by_snr() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema_path = (
        Path(__file__).resolve().parents[2]
        / "datasets"
        / "schema"
        / "g5-ordered-impairments-snr-curve.v3.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    curve = _curve(version=3)
    arrays = [
        "actual_capture",
        "awgn_standard_normal_imag",
        "awgn_standard_normal_real",
        "channel_seeds",
        "clean_receiver_input_by_snr",
        "corrected_receiver_input_by_snr",
        "rx_phase_noise_rad",
        "tx_phase_noise_rad",
        "tx_waveform",
        "uncorrected_receiver_input_by_snr",
    ]
    manifest = {
        "schema": "umcd.g5_ordered_impairments_waveform_evidence.v4",
        "file_name": "curve.waveforms.npz",
        "sha256": "a" * 64,
        "byte_count": 100,
        "channel_count": 1,
        "snr_point_count": 1,
        "snr_db": [0.0],
        "arrays": arrays,
    }
    curve["config"]["waveform_evidence"] = manifest
    curve["points"][0]["experiment"]["config"]["waveform_evidence"] = manifest
    curve["points"][0]["experiment"]["schema"] = (
        "umcd.g5_ordered_impairments_independent_channels.v19"
    )

    jsonschema.Draft202012Validator(schema).validate(curve)


def test_independent_channel_snr_curve_v4_schema_requires_paired_effect_family() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema_path = (
        Path(__file__).resolve().parents[2]
        / "datasets"
        / "schema"
        / "g5-ordered-impairments-snr-curve.v4.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    curve = _curve(version=4)
    curve["curve_inference"]["by_paired_contrast"] = [
        {
            "snr_db": 0.0,
            "mean_channel_macro_bler_improvement": 0.125,
            "pointwise_95_percent_interval": [-0.25, 0.5],
            "simultaneous_95_percent_interval": [-0.5, 0.75],
        }
    ]

    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(curve)


def test_v9_schema_accepts_unavailable_inference_for_single_channel_diagnostics() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema_path = (
        Path(__file__).resolve().parents[2]
        / "datasets"
        / "schema"
        / "g5-ordered-impairments-snr-curve.v9.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    curve = _curve(version=9)
    curve["config"]["channel_seeds"] = [7]
    inference = curve["curve_inference"]
    inference["status"] = "unavailable"
    inference["cluster_count"] = 1
    inference.pop("simultaneous_radius")
    inference["reason"] = "at least two independent channel clusters are required"
    inference["by_condition"] = {
        condition: [] for condition in ("clean", "uncorrected", "corrected")
    }
    inference["by_paired_contrast"] = []

    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(curve)
    curve["config"]["schema_sha256"] = hashlib.sha256(schema_path.read_bytes()).hexdigest()
    assert _curve_report_schema_status(curve, schema_path) == (
        curve["config"]["schema_sha256"],
        True,
    )


def test_v10_acceptance_schema_requires_timestamped_signed_preregistration() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema_path = (
        Path(__file__).resolve().parents[2]
        / "datasets"
        / "schema"
        / "g5-ordered-impairments-snr-curve.v10.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    contract = {
        "schema": "umcd.g5_ordered_impairments_snr_acceptance.v2",
        "preregistered_utc": "2026-10-25T09:00:00Z",
        "snr_db": [0.0],
        "channel_seeds": list(range(20)),
        "channel_estimator": "sionna_default_ls",
        "blocks_per_channel": 1,
        "minimum_independent_channels": 20,
        "maximum_corrected_macro_bler": [0.1],
        "minimum_uncorrected_minus_corrected_macro_bler": [0.05],
    }
    frozen = {
        "status": "contract_supplied_before_run",
        "criteria": contract,
        "criteria_sha256": "0" * 64,
        "confidence_level": 0.95,
        "interval_scope": "simultaneous across all three conditions and paired improvements",
        "decision_unit": "independent synthetic channel; transport blocks are nested",
        "preregistration_signature": {
            "status": "verified_before_receiver_run",
            "namespace": "umcd-g5-snr-acceptance-v1",
            "signer_id": "reviewer@example.org",
            "contract_payload_sha256": "0" * 64,
            "signature_sha256": "1" * 64,
            "trusted_allowed_signers_sha256": "2" * 64,
            "trust_file_custody": "not_established_by_this_verifier",
            "preregistered_utc": contract["preregistered_utc"],
            "receiver_run_started_utc": "2026-10-26T09:00:00Z",
            "declared_temporal_order_status": "preregistration_precedes_receiver_run",
            "timestamp_authority": "not_established_by_this_verifier",
        },
    }
    acceptance = {
        "criteria_frozen_before_run": frozen,
        "status": "NOT_EVALUABLE",
        "pass": False,
    }

    validator = jsonschema.Draft202012Validator(schema["properties"]["acceptance"])
    validator.validate(acceptance)
    del contract["preregistered_utc"]
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(acceptance)


def test_v11_acceptance_schema_requires_independent_reviewer_policy_identity() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema_path = (
        Path(__file__).resolve().parents[2]
        / "datasets"
        / "schema"
        / "g5-ordered-impairments-snr-curve.v11.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    signature = {
        "status": "verified_before_receiver_run",
        "namespace": "umcd-g5-snr-acceptance-v1",
        "signer_id": "reviewer@example.org",
        "contract_payload_sha256": "0" * 64,
        "signature_sha256": "1" * 64,
        "trusted_allowed_signers_sha256": "2" * 64,
        "trust_file_custody": "not_established_by_this_verifier",
        "preregistered_utc": "2026-10-25T09:00:00Z",
        "receiver_run_started_utc": "2026-10-26T09:00:00Z",
        "declared_temporal_order_status": "preregistration_precedes_receiver_run",
        "timestamp_authority": "not_established_by_this_verifier",
        "reviewer_policy": {
            "status": "verified",
            "policy_id": "policy-1",
            "policy_sha256": "3" * 64,
            "reviewer_principal": "reviewer@example.org",
            "reviewer_organization": "Independent Lab",
            "reviewer_roles": "independent_reviewer",
            "purpose": "g5_ordered_impairments_snr_preregistration",
            "custodian_principal": "custodian@example.net",
            "signature_namespace": "umcd-g5-snr-reviewer-policy-v1",
            "signature_sha256": "4" * 64,
            "trusted_allowed_signers_sha256": "5" * 64,
            "trust_file_custody": "not_established_by_this_verifier",
        },
    }
    frozen = {
        "status": "contract_supplied_before_run",
        "criteria": {
            "schema": "umcd.g5_ordered_impairments_snr_acceptance.v2",
            "preregistered_utc": "2026-10-25T09:00:00Z",
            "snr_db": [0.0],
            "channel_seeds": list(range(20)),
            "channel_estimator": "sionna_default_ls",
            "blocks_per_channel": 1,
            "minimum_independent_channels": 20,
            "maximum_corrected_macro_bler": [0.1],
            "minimum_uncorrected_minus_corrected_macro_bler": [0.05],
        },
        "criteria_sha256": "6" * 64,
        "confidence_level": 0.95,
        "interval_scope": "simultaneous across all three conditions and paired improvements",
        "decision_unit": "independent synthetic channel; transport blocks are nested",
        "preregistration_signature": signature,
    }
    acceptance = {
        "criteria_frozen_before_run": frozen,
        "status": "NOT_EVALUABLE",
        "pass": False,
    }
    acceptance_schema = {
        "$schema": schema["$schema"],
        "$defs": schema["$defs"],
        **schema["properties"]["acceptance"],
    }
    validator = jsonschema.Draft202012Validator(acceptance_schema)
    validator.validate(acceptance)
    del signature["reviewer_policy"]
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(acceptance)


def test_independent_channel_snr_curve_v3_schema_rejects_more_than_64_points() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema_path = (
        Path(__file__).resolve().parents[2]
        / "datasets"
        / "schema"
        / "g5-ordered-impairments-snr-curve.v3.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    curve = _curve(version=3)
    curve["config"]["snr_db"] = [float(value) for value in range(65)]
    curve["config"]["transport_blocks_across_curve"] = 65
    curve["points"] = [
        {
            "snr_db": float(value),
            "experiment": curve["points"][0]["experiment"],
        }
        for value in range(65)
    ]

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(curve)


@pytest.mark.parametrize("report_version", (19, 20))
def test_independent_channel_snr_curve_schema_accepts_current_point_report_versions(
    report_version: int,
) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema_path = (
        Path(__file__).resolve().parents[2]
        / "datasets"
        / "schema"
        / "g5-ordered-impairments-snr-curve.v2.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    curve = _curve(version=2)
    curve["points"][0]["experiment"]["schema"] = (
        f"umcd.g5_ordered_impairments_independent_channels.v{report_version}"
    )

    jsonschema.Draft202012Validator(schema).validate(curve)


def test_independent_channel_snr_curve_schema_rejects_qualification_claim() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema_path = (
        Path(__file__).resolve().parents[2]
        / "datasets"
        / "schema"
        / "g5-ordered-impairments-snr-curve.v1.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    invalid = _curve()
    invalid["acceptance"] = {
        "criteria_frozen_before_run": {"status": "qualified", "reason": "invalid"},
        "pass": True,
    }

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(invalid)


def test_independent_channel_snr_curve_schema_accepts_shared_waveform_evidence() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema_path = (
        Path(__file__).resolve().parents[2]
        / "datasets"
        / "schema"
        / "g5-ordered-impairments-snr-curve.v1.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    curve = _curve()
    manifest = {
        "schema": "umcd.g5_ordered_impairments_waveform_evidence.v2",
        "file_name": "curve.waveforms.npz",
        "sha256": "a" * 64,
        "byte_count": 100,
        "channel_count": 1,
        "arrays": [
            "actual_capture",
            "awgn_standard_normal_imag",
            "awgn_standard_normal_real",
            "channel_seeds",
            "rx_phase_noise_rad",
            "tx_phase_noise_rad",
            "tx_waveform",
        ],
    }
    curve["config"]["waveform_evidence"] = manifest
    curve["points"][0]["experiment"]["config"]["waveform_evidence"] = manifest
    curve["points"][0]["experiment"]["schema"] = (
        "umcd.g5_ordered_impairments_independent_channels.v16"
    )

    jsonschema.Draft202012Validator(schema).validate(curve)
