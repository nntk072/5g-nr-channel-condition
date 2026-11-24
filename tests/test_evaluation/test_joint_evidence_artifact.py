"""Joint-evidence artifact binding of exact inputs and recomputed scores."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import jsonschema
import pytest
from evaluation import joint_evidence_artifact as artifact
from evaluation.acceptance import path_equivalence_assurance as path_equivalence_assurance
from tests.test_evaluation.joint_evidence_artifact_support import (
    _install_scorers,
    _payload,
    _payload_v2,
    _payload_v4,
    _payload_v5,
    _payload_v6,
    _write_canonical,
)


def test_joint_artifact_binds_exact_inputs_and_recomputes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_scorers(monkeypatch)
    source = tmp_path / "inputs.json"
    saved = tmp_path / "bundle.json"
    _write_canonical(source, _payload())

    report = artifact.evaluate_joint_evidence_file(source)
    artifact_schema_path = (
        Path(__file__).resolve().parents[2]
        / "datasets/schema/joint-evaluator-artifact.v2.schema.json"
    )
    input_schema_path = (
        Path(__file__).resolve().parents[2] / "datasets/schema/joint-evaluator-input.v1.schema.json"
    )
    input_schema_v4_path = (
        Path(__file__).resolve().parents[2] / "datasets/schema/joint-evaluator-input.v4.schema.json"
    )
    input_schema_v5_path = (
        Path(__file__).resolve().parents[2] / "datasets/schema/joint-evaluator-input.v5.schema.json"
    )
    input_schema_v6_path = (
        Path(__file__).resolve().parents[2] / "datasets/schema/joint-evaluator-input.v6.schema.json"
    )
    jsonschema.Draft202012Validator(
        json.loads(artifact_schema_path.read_text(encoding="utf-8"))
    ).validate(report)
    jsonschema.Draft202012Validator(
        json.loads(input_schema_path.read_text(encoding="utf-8"))
    ).validate(_payload())
    jsonschema.Draft202012Validator(
        json.loads(input_schema_v4_path.read_text(encoding="utf-8"))
    ).validate(_payload_v4())
    jsonschema.Draft202012Validator(
        json.loads(input_schema_v5_path.read_text(encoding="utf-8"))
    ).validate(_payload_v5())
    jsonschema.Draft202012Validator(
        json.loads(input_schema_v6_path.read_text(encoding="utf-8"))
    ).validate(_payload_v6())
    artifact._write_new_artifact(saved, report)
    verified = artifact.verify_joint_evidence_file(source, saved)

    assert report["qualification_status"] == "NOT_QUALIFIED"
    assert verified["status"] == "PASS_INPUT_SCOPED_RECOMPUTATION"
    assert verified["input_bytes_sha256"] == report["input_bytes_sha256"]


def test_joint_artifact_rejects_changed_input_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_scorers(monkeypatch)
    source = tmp_path / "inputs.json"
    saved = tmp_path / "bundle.json"
    payload = _payload_v4()
    _write_canonical(source, payload)
    artifact._write_new_artifact(saved, artifact.evaluate_joint_evidence_file(source))

    changed = deepcopy(payload)
    changed["receiver_impact"]["comparison_snr_db"] = 1.0
    _write_canonical(source, changed)
    with pytest.raises(ValueError, match="exact-byte-bound recomputation"):
        artifact.verify_joint_evidence_file(source, saved)


def test_complex_json_encoding_rejects_nonfinite_components() -> None:
    with pytest.raises(ValueError, match="finite"):
        artifact._decode_json_value({"$complex": [float("inf"), 0.0]})


@pytest.mark.parametrize("role", ["candidate", "challenger"])
def test_joint_input_rejects_execution_configuration_hash_drift(tmp_path: Path, role: str) -> None:
    source = tmp_path / "joint-input.json"
    payload = _payload_v2()
    payload["receiver_execution"][role]["configuration"]["receiver"] = "changed"
    _write_canonical(source, payload)

    with pytest.raises(ValueError, match="configuration hash is inconsistent"):
        artifact.evaluate_joint_evidence_file(source)
