from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest


def _manifest() -> dict[str, Any]:
    return {
        "schema": "umcd.g5_ordered_impairments_waveform_evidence.v1",
        "file_name": "receiver.waveforms.npz",
        "sha256": "a" * 64,
        "byte_count": 4096,
        "channel_count": 64,
        "arrays": [
            "actual_capture",
            "channel_seeds",
            "rx_phase_noise_rad",
            "tx_phase_noise_rad",
            "tx_waveform",
        ],
    }


def _manifest_v2() -> dict[str, Any]:
    return {
        "schema": "umcd.g5_ordered_impairments_waveform_evidence.v2",
        "file_name": "curve.waveforms.npz",
        "sha256": "b" * 64,
        "byte_count": 8192,
        "channel_count": 64,
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


def _manifest_v3() -> dict[str, Any]:
    return {
        "schema": "umcd.g5_ordered_impairments_waveform_evidence.v3",
        "file_name": "receiver.waveforms.npz",
        "sha256": "c" * 64,
        "byte_count": 16384,
        "channel_count": 64,
        "arrays": [
            "actual_capture",
            "awgn_standard_normal_imag",
            "awgn_standard_normal_real",
            "channel_seeds",
            "clean_receiver_input",
            "corrected_receiver_input",
            "rx_phase_noise_rad",
            "tx_phase_noise_rad",
            "tx_waveform",
            "uncorrected_receiver_input",
        ],
    }


def _manifest_v4() -> dict[str, Any]:
    return {
        "schema": "umcd.g5_ordered_impairments_waveform_evidence.v4",
        "file_name": "curve.waveforms.npz",
        "sha256": "d" * 64,
        "byte_count": 32768,
        "channel_count": 64,
        "snr_point_count": 3,
        "snr_db": [-5.0, 0.0, 5.0],
        "arrays": [
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
        ],
    }


def test_waveform_evidence_manifest_matches_versioned_schema() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema_path = (
        Path(__file__).parents[2]
        / "datasets"
        / "schema"
        / "g5-ordered-impairments-waveform-evidence.v1.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)

    jsonschema.Draft202012Validator(schema).validate(_manifest())


def test_waveform_evidence_v2_manifest_matches_versioned_schema() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema_path = (
        Path(__file__).parents[2]
        / "datasets"
        / "schema"
        / "g5-ordered-impairments-waveform-evidence.v2.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)

    jsonschema.Draft202012Validator(schema).validate(_manifest_v2())


def test_waveform_evidence_v3_manifest_matches_versioned_schema() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema_path = (
        Path(__file__).parents[2]
        / "datasets"
        / "schema"
        / "g5-ordered-impairments-waveform-evidence.v3.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)

    jsonschema.Draft202012Validator(schema).validate(_manifest_v3())


def test_waveform_evidence_v4_manifest_matches_versioned_schema() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema_path = (
        Path(__file__).parents[2]
        / "datasets"
        / "schema"
        / "g5-ordered-impairments-waveform-evidence.v4.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)

    jsonschema.Draft202012Validator(schema).validate(_manifest_v4())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("file_name", "../outside.npz"),
        ("sha256", "not-a-digest"),
        ("byte_count", 512 * 1024 * 1024 + 1),
        ("arrays", ["channel_seeds"]),
    ],
)
def test_waveform_evidence_manifest_rejects_contract_violations(
    field: str, value: Any
) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema_path = (
        Path(__file__).parents[2]
        / "datasets"
        / "schema"
        / "g5-ordered-impairments-waveform-evidence.v1.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    manifest = copy.deepcopy(_manifest())
    manifest[field] = value

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(manifest)
