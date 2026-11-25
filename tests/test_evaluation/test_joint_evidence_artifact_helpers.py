"""Strict JSON and file-boundary helpers for joint evidence artifacts."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import numpy as np
import pytest
from evaluation import joint_evidence_artifact as artifact
from evaluation import joint_evidence_run_manifest as run_manifest


@pytest.mark.parametrize(
    "value",
    [
        {"$complex": "1,2"},
        {"$complex": [1]},
        {"$complex": [True, 2]},
        {"$complex": ["1", 2]},
    ],
)
def test_complex_json_decoder_rejects_invalid_component_shapes(value):
    with pytest.raises(ValueError, match="exactly two real numeric components"):
        artifact._decode_json_value(value)


def test_strict_json_parser_rejects_duplicate_keys_constants_and_bad_utf8():
    for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b"\xff"):
        with pytest.raises(ValueError, match="not strict UTF-8 JSON"):
            artifact._parse_json(raw, label="probe")


def test_json_tree_normalizes_numpy_complex_and_mapping_values():
    assert artifact._json_tree(
        {"z": np.complex64(1 + 2j), "a": np.asarray([1, 2]), "nested": (np.int64(3),)}
    ) == {"a": [1, 2], "nested": [3], "z": {"$complex": [1.0, 2.0]}}


@pytest.mark.parametrize(
    "value",
    [
        {1: "non-string key"},
        float("inf"),
        object(),
    ],
)
def test_json_tree_rejects_values_outside_artifact_domain(value):
    with pytest.raises((TypeError, ValueError)):
        artifact._json_tree(value)


def test_json_tree_rejects_nonfinite_complex_components():
    with pytest.raises(ValueError, match="not JSON compliant"):
        artifact._canonical_json_bytes(1 + float("inf") * 1j)


def test_regular_file_reader_requires_no_follow_support(monkeypatch, tmp_path: Path):
    path = tmp_path / "input.json"
    path.write_bytes(b"{}")
    monkeypatch.delattr(artifact.os, "O_NOFOLLOW")
    with pytest.raises(RuntimeError, match="O_NOFOLLOW"):
        artifact._read_regular_file(path, limit=10, label="probe")


def test_regular_file_reader_rejects_hard_links_and_size_growth(monkeypatch, tmp_path: Path):
    path = tmp_path / "input.json"
    path.write_bytes(b"{}")
    alias = tmp_path / "alias.json"
    os.link(path, alias)
    with pytest.raises(ValueError, match="single-link regular file"):
        artifact._read_regular_file(path, limit=10, label="probe")
    alias.unlink()

    monkeypatch.setattr(artifact.os, "read", lambda _descriptor, _limit: b"12345")
    with pytest.raises(ValueError, match="exceeds 4 bytes"):
        artifact._read_regular_file(path, limit=4, label="probe")


def test_regular_file_reader_detects_descriptor_change_while_opening(monkeypatch, tmp_path: Path):
    path = tmp_path / "input.json"
    path.write_bytes(b"{}")
    signature = artifact._stat_signature
    calls = 0

    def changing_signature(value):
        nonlocal calls
        calls += 1
        current = signature(value)
        return current if calls == 1 else (*current, 1)

    monkeypatch.setattr(artifact, "_stat_signature", changing_signature)
    with pytest.raises(ValueError, match="changed while opening"):
        artifact._read_regular_file(path, limit=10, label="probe")


def test_regular_file_reader_rejects_nonregular_open_descriptor(monkeypatch, tmp_path: Path):
    path = tmp_path / "input.json"
    path.write_bytes(b"{}")
    monkeypatch.setattr(artifact, "_stat_signature", lambda _value: ())
    monkeypatch.setattr(
        artifact.os,
        "fstat",
        lambda _descriptor: os.stat_result(
            (stat.S_IFIFO, 1, 1, 1, 0, 0, 0, 0, 0, 0)
        ),
    )
    with pytest.raises(ValueError, match="single-link regular file"):
        artifact._read_regular_file(path, limit=10, label="probe")


def test_regular_file_reader_detects_change_after_read(monkeypatch, tmp_path: Path):
    path = tmp_path / "input.json"
    path.write_bytes(b"{}")
    signature = artifact._stat_signature
    calls = 0

    def changing_signature(value):
        nonlocal calls
        calls += 1
        current = signature(value)
        return (*current, 1) if calls == 3 else current

    monkeypatch.setattr(artifact, "_stat_signature", changing_signature)
    with pytest.raises(ValueError, match="changed while being read"):
        artifact._read_regular_file(path, limit=10, label="probe")


def _run_preflight(tmp_path: Path, source_root: Path | None = None):
    source = tmp_path / "input.json"
    contract = tmp_path / "contract.json"
    source.write_bytes(b"{}")
    contract.write_bytes(b"{}")
    return artifact.run_frozen_qualification_evaluation(
        source,
        contract,
        tmp_path / "artifact.json",
        tmp_path / "receipt.json",
        source_root=source_root,
    )


def test_frozen_preflight_rejects_non_directory_source_root(tmp_path: Path):
    source_root = tmp_path / "source-file"
    source_root.write_text("source", encoding="utf-8")
    with pytest.raises(ValueError, match="roots must be directories"):
        _run_preflight(tmp_path, source_root)


def test_frozen_preflight_rejects_outputs_outside_input_directory(tmp_path: Path):
    source = tmp_path / "input.json"
    contract = tmp_path / "contract.json"
    source.write_bytes(b"{}")
    contract.write_bytes(b"{}")
    with pytest.raises(ValueError, match="written beside"):
        artifact.run_frozen_qualification_evaluation(
            source, contract, tmp_path.parent / "artifact.json", tmp_path / "receipt.json"
        )


@pytest.mark.parametrize("collision", ["outputs", "input"])
def test_frozen_preflight_rejects_colliding_paths(tmp_path: Path, collision: str):
    source = tmp_path / "input.json"
    contract = tmp_path / "contract.json"
    source.write_bytes(b"{}")
    contract.write_bytes(b"{}")
    artifact_path = source if collision == "input" else tmp_path / "artifact.json"
    receipt_path = tmp_path / "artifact.json" if collision == "outputs" else tmp_path / "receipt.json"
    with pytest.raises(ValueError, match="must be distinct"):
        artifact.run_frozen_qualification_evaluation(source, contract, artifact_path, receipt_path)


def test_frozen_preflight_refuses_existing_outputs(tmp_path: Path):
    output = tmp_path / "artifact.json"
    output.write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        _run_preflight(tmp_path)


@pytest.mark.parametrize(
    ("contract_bytes", "exception", "message"),
    [
        (b'{"schema": "loslab.qualification_run_contract.v3"}', ValueError, "canonical compact JSON"),
        (b'{"schema":3}\n', ValueError, "requires contract v3 or v4"),
        (b'{"schema":"unsupported"}\n', ValueError, "requires contract v3 or v4"),
    ],
)
def test_frozen_preflight_requires_canonical_supported_contract(
    tmp_path: Path, contract_bytes: bytes, exception: type[Exception], message: str
):
    with pytest.raises(exception, match=message):
        _run_preflight_with_contract(tmp_path, contract_bytes)


def _run_preflight_with_contract(tmp_path: Path, contract_bytes: bytes):
    source = tmp_path / "input.json"
    contract = tmp_path / "contract.json"
    source.write_bytes(b"{}")
    contract.write_bytes(contract_bytes)
    return artifact.run_frozen_qualification_evaluation(
        source, contract, tmp_path / "artifact.json", tmp_path / "receipt.json"
    )


def test_frozen_preflight_rejects_contract_schema_violations(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        run_manifest,
        "_read_rooted_source_file",
        lambda *_args, **_kwargs: b'{"type":"object","required":["required"]}',
    )
    with pytest.raises(ValueError, match="violates schema"):
        _run_preflight_with_contract(
            tmp_path, b'{"schema":"loslab.qualification_run_contract.v3"}\n'
        )


def test_frozen_preflight_rejects_stale_source_identity(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(run_manifest, "_read_rooted_source_file", lambda *_args, **_kwargs: b"{}")
    monkeypatch.setattr(run_manifest, "verify_qualification_run_contract", lambda _contract: {})
    monkeypatch.setattr(
        run_manifest,
        "evaluator_source_bundle_identity",
        lambda **_kwargs: {"evaluator_source_bundle_sha256": "current"},
    )
    contract = {
        "schema": "loslab.qualification_run_contract.v3",
        "source_identity": {"source_bundle_sha256": "stale"},
    }
    raw = artifact._canonical_json_bytes(contract)
    with pytest.raises(ValueError, match="differs from current evaluator sources"):
        _run_preflight_with_contract(tmp_path, raw)


def test_frozen_preflight_detects_input_digest_drift_while_building_receipt(
    monkeypatch, tmp_path: Path
):
    source = tmp_path / "input.json"
    contract_path = tmp_path / "contract.json"
    source.write_bytes(b'{"schema":"loslab.joint_evaluator_inputs.v4"}\n')
    contract_path.write_bytes(
        b'{"run_id":"run","schema":"loslab.qualification_run_contract.v3",'
        b'"source_identity":{"source_bundle_sha256":"current"}}\n'
    )
    monkeypatch.setattr(run_manifest, "_read_rooted_source_file", lambda *_args, **_kwargs: b"{}")
    monkeypatch.setattr(run_manifest, "verify_qualification_run_contract", lambda _value: {})
    monkeypatch.setattr(
        run_manifest,
        "evaluator_source_bundle_identity",
        lambda **_kwargs: {"evaluator_source_bundle_sha256": "current"},
    )
    monkeypatch.setattr(
        run_manifest,
        "_validate_joint_evaluator_input_schema",
        lambda *_args, **_kwargs: run_manifest.INPUT_SCHEMA_V4_RELATIVE_PATH,
    )
    monkeypatch.setattr(run_manifest, "_verify_qualification_contract_bindings", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run_manifest, "_runtime_identity", lambda: {})
    monkeypatch.setattr(
        artifact,
        "_evaluate_joint_evidence_bytes",
        lambda *_args, **_kwargs: ({}, {"input_bytes_sha256": "expected", "bundle": {}}),
    )
    monkeypatch.setattr(artifact, "_write_new_artifact", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        run_manifest,
        "build_joint_evidence_run_manifest",
        lambda *_args, **_kwargs: {"input": {"sha256": "drifted"}, "artifact": {"sha256": "x"}},
    )
    with pytest.raises(ValueError, match="changed while creating"):
        artifact.run_frozen_qualification_evaluation(
            source, contract_path, tmp_path / "artifact.json", tmp_path / "receipt.json"
        )


@pytest.mark.parametrize("limit_name", ["MAX_INPUT_BYTES", "MAX_ARTIFACT_BYTES"])
def test_artifact_byte_verifier_enforces_each_size_limit(monkeypatch, limit_name: str):
    monkeypatch.setattr(artifact, limit_name, 1)
    with pytest.raises(ValueError, match="exceeds 1 bytes"):
        artifact.verify_joint_evidence_bytes(b"{}", b"{}")


def test_artifact_byte_verifier_rejects_field_roster_and_noncanonical_json():
    with pytest.raises(ValueError, match="fields do not match"):
        artifact.verify_joint_evidence_bytes(b"{}", b"{}")
    artifact_with_roster = (
        b'{ "schema":"", "status":"", "qualification_status":"", '
        b'"input_bytes_sha256":"", "bundle_sha256":"", "bundle":{} }'
    )
    with pytest.raises(ValueError, match="canonical JSON bytes"):
        artifact.verify_joint_evidence_bytes(b"{}", artifact_with_roster)


def test_cli_runs_preregistered_command_and_prints_report(monkeypatch, capsys):
    monkeypatch.setattr(
        artifact,
        "run_frozen_qualification_evaluation",
        lambda *_args, **_kwargs: {"status": "NUMERICAL_ONLY"},
    )
    assert (
        artifact.main(
            [
                "evaluate-preregistered",
                "input.json",
                "artifact.json",
                "--qualification-contract",
                "contract.json",
                "--run-manifest",
                "receipt.json",
            ]
        )
        == 0
    )
    assert '"status": "NUMERICAL_ONLY"' in capsys.readouterr().out
