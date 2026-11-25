"""Runtime inventories, scorer sources and source-roster checks for joint-evidence manifests."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from evaluation import joint_evidence_artifact as artifact
from evaluation import joint_evidence_run_manifest as run_manifest
from evaluation.acceptance import joint_evaluation as joint
from evaluation.acceptance import path_equivalence_assurance as path_equivalence_assurance
from tests.test_evaluation.joint_evidence_artifact_support import (
    _install_scorers,
    _legacy_v1_report,
    _payload,
    _payload_v4,
    _qualification_contract,
    _write_canonical,
    _write_run_manifest,
)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"require_signed_reviewer_admission": "yes"}, "must be a boolean"),
        ({"signature_path": "signature"}, "must be supplied together"),
        ({"trusted_signer_policy_path": "policy"}, "requires a verified manifest signature"),
        (
            {"trusted_signer_policy_signature_path": "signature"},
            "must be supplied together",
        ),
        (
            {
                "trusted_signer_policy_signature_path": "policy.sig",
                "trusted_policy_allowed_signers_path": "policy.allowed",
                "policy_signer_id": "policy-signer",
                "signature_path": "run.sig",
                "trusted_allowed_signers_path": "run.allowed",
                "signer_id": "run-signer",
            },
            "requires both a signer policy and a signed run manifest",
        ),
        (
            {"runtime_inventory_signature_path": "runtime.sig"},
            "must be supplied together",
        ),
        (
            {"require_signed_reviewer_admission": True},
            "requires a signed run manifest",
        ),
    ],
)
def test_run_manifest_verifier_rejects_incomplete_signature_options(
    tmp_path: Path, options: dict[str, Any], message: str
) -> None:
    report = run_manifest.verify_joint_evidence_run_manifest(
        tmp_path / "missing-manifest.json", **options
    )

    assert report["status"] == "INCONSISTENT"
    assert any(message in error for error in report["errors"])


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"[]\n", "root must be an object"),
        (b'{"schema":"unsupported"}\n', "unsupported run manifest schema"),
    ],
)
def test_run_manifest_verifier_rejects_invalid_root_and_schema(
    tmp_path: Path, payload: bytes, message: str
) -> None:
    manifest_path = tmp_path / "run-manifest.json"
    manifest_path.write_bytes(payload)

    report = run_manifest.verify_joint_evidence_run_manifest(manifest_path)

    assert report["status"] == "INCONSISTENT"
    assert any(message in error for error in report["errors"])


def test_run_manifest_hashes_and_replays_exact_inventory_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_scorers(monkeypatch)
    source = tmp_path / "joint-input.json"
    saved = tmp_path / "joint-report.json"
    _write_canonical(source, _payload())
    artifact._write_new_artifact(saved, artifact.evaluate_joint_evidence_file(source))
    manifest_path = tmp_path / "run-manifest.json"
    _write_run_manifest(manifest_path, source, saved)

    report = run_manifest.verify_joint_evidence_run_manifest(manifest_path)

    assert report["status"] == "PASS_INPUT_SCOPED_RECOMPUTATION"
    assert report["qualification_status"] == "NOT_QUALIFIED"
    assert report["branch_count"] == 5
    assert report["evaluator_source_bundle_sha256"]
    assert {"pyproject.toml", "uv.lock"}.issubset(set(run_manifest.EVALUATOR_SOURCE_PATHS))
    assert {
        run_manifest.INPUT_SCHEMA_V4_RELATIVE_PATH,
        run_manifest.INPUT_SCHEMA_V5_RELATIVE_PATH,
        run_manifest.INPUT_SCHEMA_V6_RELATIVE_PATH,
        run_manifest.PATH_DIAGNOSTIC_SCHEMA_V2_RELATIVE_PATH,
        run_manifest.QUALIFICATION_CONTRACT_SCHEMA_V4_RELATIVE_PATH,
        "evaluation/acceptance/path_equivalence.py",
        "evaluation/acceptance/propagation_assurance.py",
        "evaluation/acceptance/receiver_assurance.py",
    }.issubset(set(run_manifest.EVALUATOR_SOURCE_PATHS))
    assert run_manifest.INPUT_SCHEMA_V4_RELATIVE_PATH not in set(
        run_manifest.V1_EVALUATOR_SOURCE_PATHS
    )
    assert report["qualification_contract_bound"] is False


def test_run_manifest_v1_keeps_its_frozen_source_roster(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_scorers(monkeypatch)
    source = tmp_path / "joint-input.json"
    saved = tmp_path / "joint-report.json"
    _write_canonical(source, _payload())
    _write_canonical(saved, _legacy_v1_report(artifact.evaluate_joint_evidence_file(source)))
    manifest = run_manifest.build_joint_evidence_run_manifest(
        "joint-run-001", source, saved, package_root=tmp_path
    )

    source_root = Path(__file__).resolve().parents[2]
    legacy_sources = []
    for relative_path in run_manifest.V1_EVALUATOR_SOURCE_PATHS:
        source_bytes = (source_root / relative_path).read_bytes()
        legacy_sources.append(
            {
                "path": relative_path,
                "size_bytes": len(source_bytes),
                "sha256": hashlib.sha256(source_bytes).hexdigest(),
            }
        )
    manifest.pop("runtime_identity")
    manifest["schema"] = "loslab.joint_evidence_run_manifest.v1"
    manifest["evaluator_sources"] = legacy_sources
    manifest["evaluator_source_bundle_sha256"] = run_manifest._source_bundle_sha256(legacy_sources)
    manifest_path = tmp_path / "legacy-run-manifest.json"
    _write_canonical(manifest_path, manifest)

    report = run_manifest.verify_joint_evidence_run_manifest(manifest_path)

    assert report["status"] == "PASS_INPUT_SCOPED_RECOMPUTATION"
    assert report["qualification_status"] == "NOT_QUALIFIED"


def test_run_manifest_v2_remains_readable_after_runtime_identity_v3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_scorers(monkeypatch)
    source = tmp_path / "joint-input.json"
    saved = tmp_path / "joint-report.json"
    _write_canonical(source, _payload())
    _write_canonical(saved, _legacy_v1_report(artifact.evaluate_joint_evidence_file(source)))
    manifest = run_manifest.build_joint_evidence_run_manifest(
        "joint-run-001", source, saved, package_root=tmp_path
    )
    manifest.pop("runtime_identity")
    manifest["schema"] = "loslab.joint_evidence_run_manifest.v2"
    manifest["evaluator_sources"] = [
        item
        for item in manifest["evaluator_sources"]
        if item["path"]
        not in {
            run_manifest.SCHEMA_V3_RELATIVE_PATH,
            run_manifest.ARTIFACT_SCHEMA_RELATIVE_PATH,
            "artifacts/provenance.py",
        }
    ]
    manifest["evaluator_source_bundle_sha256"] = run_manifest._source_bundle_sha256(
        manifest["evaluator_sources"]
    )
    manifest_path = tmp_path / "legacy-v2-run-manifest.json"
    _write_canonical(manifest_path, manifest)

    report = run_manifest.verify_joint_evidence_run_manifest(manifest_path)

    assert report["status"] == "PASS_INPUT_SCOPED_RECOMPUTATION"
    assert report["runtime_identity_status"] == "not_recorded"


def test_run_manifest_source_roster_sizes_match_versioned_schemas() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    schema_rosters = (
        (
            run_manifest.SCHEMA_RELATIVE_PATH,
            run_manifest.V2_EVALUATOR_SOURCE_PATHS,
        ),
        (
            run_manifest.SCHEMA_V3_RELATIVE_PATH,
            run_manifest.EVALUATOR_SOURCE_PATHS,
        ),
        (
            run_manifest.SCHEMA_V4_RELATIVE_PATH,
            run_manifest.V4_EVALUATOR_SOURCE_PATHS,
        ),
        (
            run_manifest.SCHEMA_V5_RELATIVE_PATH,
            run_manifest.V5_EVALUATOR_SOURCE_PATHS,
        ),
    )

    for relative_schema, source_paths in schema_rosters:
        schema = json.loads((repository_root / relative_schema).read_text(encoding="utf-8"))
        source_count = schema["properties"]["evaluator_sources"]
        assert source_count["minItems"] == len(source_paths)
        assert source_count["maxItems"] == len(source_paths)


def test_run_manifest_builder_rejects_malformed_runtime_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_scorers(monkeypatch)
    source = tmp_path / "joint-input.json"
    saved = tmp_path / "joint-report.json"
    _write_canonical(source, _payload())
    artifact._write_new_artifact(saved, artifact.evaluate_joint_evidence_file(source))

    with pytest.raises(ValueError, match="violates selected schema"):
        run_manifest.build_joint_evidence_run_manifest(
            "joint-run-001",
            source,
            saved,
            package_root=tmp_path,
            runtime_identity={"schema": "invalid"},
        )


def test_run_manifest_binds_contract_to_family_receiver_and_path_roster(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_scorers(monkeypatch)
    source = tmp_path / "joint-input.json"
    saved = tmp_path / "joint-report.json"
    payload = _payload_v4()
    _write_canonical(source, payload)
    artifact._write_new_artifact(saved, artifact.evaluate_joint_evidence_file(source))
    source_identity = run_manifest.evaluator_source_bundle_identity()
    contract = _qualification_contract(payload, source_identity["evaluator_source_bundle_sha256"])
    manifest = run_manifest.build_joint_evidence_run_manifest(
        "run-contract-001", source, saved, qualification_contract=contract
    )
    manifest_path = tmp_path / "run-manifest.json"
    _write_canonical(manifest_path, manifest)

    report = run_manifest.verify_joint_evidence_run_manifest(manifest_path)

    assert report["status"] == "PASS_INPUT_SCOPED_RECOMPUTATION"
    assert report["qualification_status"] == "NOT_QUALIFIED"
    assert report["qualification_contract_bound"] is True
    assert report["qualification_contract_sha256"] == contract["contract_sha256"]


def test_preregistered_run_rejects_contract_before_running_scorers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "joint-input.json"
    contract_path = tmp_path / "qualification-contract.json"
    payload = _payload_v4()
    _write_canonical(source, payload)
    source_identity = run_manifest.evaluator_source_bundle_identity()
    contract = _qualification_contract(
        payload,
        source_identity["evaluator_source_bundle_sha256"],
        candidate_config_id="wrong-candidate",
    )
    _write_canonical(contract_path, contract)

    def unexpected_evaluation(*args: Any, **kwargs: Any) -> dict[str, str]:
        del args, kwargs
        raise AssertionError("an invalid contract must be rejected before evaluator execution")

    monkeypatch.setattr(joint, "evaluate_material_calibration", unexpected_evaluation)

    with pytest.raises(ValueError, match="receiver configurations differ"):
        artifact.run_frozen_qualification_evaluation(
            source,
            contract_path,
            tmp_path / "joint-report.json",
            tmp_path / "run-manifest.json",
        )


def test_run_manifest_rejects_contract_source_bundle_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_scorers(monkeypatch)
    source = tmp_path / "joint-input.json"
    saved = tmp_path / "joint-report.json"
    payload = _payload_v4()
    _write_canonical(source, payload)
    artifact._write_new_artifact(saved, artifact.evaluate_joint_evidence_file(source))
    bad_contract = _qualification_contract(payload, "e" * 64)

    with pytest.raises(ValueError, match="source bundle differs"):
        run_manifest.build_joint_evidence_run_manifest(
            "run-contract-001", source, saved, qualification_contract=bad_contract
        )


def test_run_manifest_rejects_inventory_digest_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_scorers(monkeypatch)
    source = tmp_path / "joint-input.json"
    saved = tmp_path / "joint-report.json"
    _write_canonical(source, _payload())
    artifact._write_new_artifact(saved, artifact.evaluate_joint_evidence_file(source))
    manifest_path = tmp_path / "run-manifest.json"
    _write_run_manifest(manifest_path, source, saved, wrong_input_hash=True)

    report = run_manifest.verify_joint_evidence_run_manifest(manifest_path)

    assert report["status"] == "INCONSISTENT"
    assert any("size or SHA-256" in error for error in report["errors"])


def test_run_manifest_detects_locked_dependency_source_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_scorers(monkeypatch)
    source = tmp_path / "joint-input.json"
    saved = tmp_path / "joint-report.json"
    _write_canonical(source, _payload())
    artifact._write_new_artifact(saved, artifact.evaluate_joint_evidence_file(source))
    manifest_path = tmp_path / "run-manifest.json"
    _write_run_manifest(manifest_path, source, saved)

    repository_root = Path(__file__).resolve().parents[2]
    source_root = tmp_path / "source-snapshot"
    for relative_path in run_manifest.EVALUATOR_SOURCE_PATHS:
        destination = source_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repository_root / relative_path, destination)
    drifted_source = source_root / "uv.lock"
    drifted_source.write_bytes(drifted_source.read_bytes() + b"\n# drift\n")

    report = run_manifest.verify_joint_evidence_run_manifest(manifest_path, source_root=source_root)

    assert report["status"] == "INCONSISTENT"
    assert any("evaluator source bytes differ" in error for error in report["errors"])


def test_run_manifest_rejects_incomplete_scorer_source_roster(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_scorers(monkeypatch)
    source = tmp_path / "joint-input.json"
    saved = tmp_path / "joint-report.json"
    _write_canonical(source, _payload())
    artifact._write_new_artifact(saved, artifact.evaluate_joint_evidence_file(source))
    manifest_path = tmp_path / "run-manifest.json"
    _write_run_manifest(manifest_path, source, saved)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["evaluator_sources"].pop()
    _write_canonical(manifest_path, manifest)

    report = run_manifest.verify_joint_evidence_run_manifest(manifest_path)

    assert report["status"] == "INCONSISTENT"
    assert any("violates schema" in error for error in report["errors"])
