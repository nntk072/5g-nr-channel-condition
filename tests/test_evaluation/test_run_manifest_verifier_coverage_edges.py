"""Run-manifest verifier version and captured-source integrity edges."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from evaluation import joint_evidence_artifact as artifact
from evaluation import joint_evidence_run_manifest as run_manifest
from evaluation.run_manifest import sources
from evaluation.run_manifest import verify as verifier
from tests.test_evaluation.joint_evidence_artifact_support import (
    _install_scorers,
    _payload,
    _payload_v6,
    _qualification_contract,
    _runtime_inventory_fixture,
    _write_canonical,
    _write_run_manifest,
)


def _prepare_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    _install_scorers(monkeypatch)
    input_path = tmp_path / "input.json"
    artifact_path = tmp_path / "artifact.json"
    manifest_path = tmp_path / "manifest.json"
    _write_canonical(input_path, _payload())
    artifact._write_new_artifact(artifact_path, artifact.evaluate_joint_evidence_file(input_path))
    _write_run_manifest(manifest_path, input_path, artifact_path)
    return manifest_path, input_path, artifact_path


def test_verifier_accepts_v4_manifest_source_roster(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_scorers(monkeypatch)
    input_path = tmp_path / "input.json"
    artifact_path = tmp_path / "artifact.json"
    inventory_path = tmp_path / "runtime-inventory.json"
    manifest_path = tmp_path / "manifest.json"
    _write_canonical(input_path, _payload())
    artifact._write_new_artifact(artifact_path, artifact.evaluate_joint_evidence_file(input_path))
    _runtime_inventory_fixture(inventory_path)

    manifest = run_manifest.build_joint_evidence_run_manifest(
        "joint-run-001", input_path, artifact_path, runtime_inventory_path=inventory_path
    )
    source_root = Path(__file__).resolve().parents[2]
    manifest["schema"] = "loslab.joint_evidence_run_manifest.v4"
    manifest["evaluator_sources"] = [
        {
            "path": relative_path,
            "size_bytes": (source_root / relative_path).stat().st_size,
            "sha256": hashlib.sha256((source_root / relative_path).read_bytes()).hexdigest(),
        }
        for relative_path in run_manifest.V4_EVALUATOR_SOURCE_PATHS
    ]
    manifest["evaluator_source_bundle_sha256"] = sources._source_bundle_sha256(
        manifest["evaluator_sources"]
    )
    _write_canonical(manifest_path, manifest)

    report = verifier.verify_joint_evidence_run_manifest(manifest_path)

    assert report["runtime_identity_status"] == "recorded_self_reported_unattested"
    assert report["evaluator_source_bundle_sha256"] == manifest["evaluator_source_bundle_sha256"]


def test_verifier_rejects_package_root_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, input_path, artifact_path = _prepare_run(tmp_path, monkeypatch)
    report = verifier.verify_joint_evidence_run_manifest(manifest_path, package_root=input_path)
    assert report["status"] == "INCONSISTENT"
    assert any("package root must be a directory" in error for error in report["errors"])
    assert artifact_path.is_file()


def test_verifier_rejects_inventory_bytes_that_were_not_captured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, _, _ = _prepare_run(tmp_path, monkeypatch)
    monkeypatch.setattr(verifier, "_read_inventory_file", lambda *_args, **_kwargs: (None, 0, ""))

    report = verifier.verify_joint_evidence_run_manifest(manifest_path)

    assert report["status"] == "INCONSISTENT"
    assert any("bytes were not captured" in error for error in report["errors"])


def test_verifier_enforces_source_total_byte_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, _, _ = _prepare_run(tmp_path, monkeypatch)
    first_source = Path(__file__).resolve().parents[2] / verifier.EVALUATOR_SOURCE_PATHS[0]
    monkeypatch.setattr(verifier, "_MAX_SOURCE_BUNDLE_BYTES", first_source.stat().st_size)

    report = verifier.verify_joint_evidence_run_manifest(manifest_path)

    assert report["status"] == "INCONSISTENT"
    assert any(
        "evaluator source roster exceeds its total byte limit" in error
        for error in report["errors"]
    )


def test_verifier_rejects_source_size_drift_in_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, _, _ = _prepare_run(tmp_path, monkeypatch)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["evaluator_sources"][0]["size_bytes"] += 1
    manifest["evaluator_source_bundle_sha256"] = sources._source_bundle_sha256(
        manifest["evaluator_sources"]
    )
    _write_canonical(manifest_path, manifest)

    report = verifier.verify_joint_evidence_run_manifest(manifest_path)

    assert report["status"] == "INCONSISTENT"
    assert any("evaluator source bytes differ from manifest" in error for error in report["errors"])


@pytest.mark.parametrize(
    ("manifest_run_id", "contract_source_digest", "message"),
    [
        ("joint-run-001", "1" * 64, "run_id differs from run receipt"),
        ("run-contract-001", "0" * 64, "source bundle differs from run receipt sources"),
    ],
)
def test_verifier_binds_qualification_contract_run_and_source_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    manifest_run_id: str,
    contract_source_digest: str,
    message: str,
) -> None:
    _install_scorers(monkeypatch)
    payload = _payload()
    input_path = tmp_path / "input.json"
    artifact_path = tmp_path / "artifact.json"
    manifest_path = tmp_path / "manifest.json"
    _write_canonical(input_path, payload)
    artifact._write_new_artifact(artifact_path, artifact.evaluate_joint_evidence_file(input_path))
    contract = _qualification_contract(payload, contract_source_digest)
    manifest = run_manifest.build_joint_evidence_run_manifest(
        manifest_run_id, input_path, artifact_path
    )
    manifest["qualification_contract"] = contract
    _write_canonical(manifest_path, manifest)

    report = verifier.verify_joint_evidence_run_manifest(manifest_path)

    assert report["status"] == "INCONSISTENT"
    assert any(message in error for error in report["errors"])


def _patch_strict_signatures(monkeypatch: pytest.MonkeyPatch, *, runtime_identity=None) -> None:
    runtime_identity = runtime_identity or {
        "signer_id": "runtime-owner",
        "allowed_signers_sha256": "runtime-keyring",
    }

    def verify_signature(
        _payload, _signature_path, _allowed_path, *, signer_id: str, namespace: str
    ):
        digest_by_label = {
            "manifest-keyring": "1" * 64,
            "policy-keyring": "2" * 64,
            "runtime-keyring": "3" * 64,
        }
        if namespace == verifier.RUNTIME_INVENTORY_SIGNATURE_NAMESPACE:
            result_signer_id = runtime_identity["signer_id"]
            allowed_digest = digest_by_label.get(
                runtime_identity["allowed_signers_sha256"],
                runtime_identity["allowed_signers_sha256"],
            )
        elif namespace == verifier.SIGNER_POLICY_SIGNATURE_NAMESPACE:
            result_signer_id = signer_id
            allowed_digest = digest_by_label["policy-keyring"]
        else:
            result_signer_id = signer_id
            allowed_digest = digest_by_label["manifest-keyring"]
        return {
            "namespace": namespace,
            "signer_id": result_signer_id,
            "signature_sha256": "a" * 64,
            "allowed_signers_sha256": allowed_digest,
        }

    monkeypatch.setattr(verifier, "verify_openssh_signature_for_bytes", verify_signature)
    monkeypatch.setattr(verifier, "_require_independent_signer_policy_custody", lambda *_args: None)
    monkeypatch.setattr(
        verifier,
        "_authorize_run_manifest_signer",
        lambda *_args: {
            "organization": "independent-lab",
            "roles": "independent_reviewer",
            "policy_id": "fixture-policy",
            "policy_sha256": "b" * 64,
        },
    )
    monkeypatch.setattr(verifier, "_verify_signed_runtime_inventory_bindings", lambda *_args: None)


def _strict_options(tmp_path: Path) -> dict[str, object]:
    policy_path = tmp_path / "signer-policy.json"
    policy_path.write_text("{}", encoding="utf-8")
    return {
        "signature_path": tmp_path / "manifest.sig",
        "trusted_allowed_signers_path": tmp_path / "manifest.allowed",
        "signer_id": "manifest-reviewer",
        "trusted_signer_policy_path": policy_path,
        "trusted_signer_policy_signature_path": tmp_path / "policy.sig",
        "trusted_policy_allowed_signers_path": tmp_path / "policy.allowed",
        "policy_signer_id": "policy-custodian",
        "runtime_inventory_signature_path": tmp_path / "runtime.sig",
        "trusted_runtime_inventory_signers_path": tmp_path / "runtime.allowed",
        "runtime_inventory_signer_id": "runtime-owner",
        "require_signed_reviewer_admission": True,
    }


def test_strict_verifier_rejects_pre_v4_manifests_after_signature_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_strict_signatures(monkeypatch)
    manifest_path, _, _ = _prepare_run(tmp_path, monkeypatch)

    report = verifier.verify_joint_evidence_run_manifest(manifest_path, **_strict_options(tmp_path))

    assert report["status"] == "INCONSISTENT"
    assert any(
        "requires v4 signed runtime inventory provenance" in error for error in report["errors"]
    )


def test_strict_verifier_requires_a_frozen_contract_on_v4_plus_manifests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_strict_signatures(monkeypatch)
    _install_scorers(monkeypatch)
    input_path = tmp_path / "input.json"
    artifact_path = tmp_path / "artifact.json"
    inventory_path = tmp_path / "runtime-inventory.json"
    manifest_path = tmp_path / "manifest.json"
    _write_canonical(input_path, _payload())
    artifact._write_new_artifact(artifact_path, artifact.evaluate_joint_evidence_file(input_path))
    _runtime_inventory_fixture(inventory_path)
    _write_run_manifest(
        manifest_path, input_path, artifact_path, runtime_inventory_path=inventory_path
    )

    report = verifier.verify_joint_evidence_run_manifest(manifest_path, **_strict_options(tmp_path))

    assert report["status"] == "INCONSISTENT"
    assert any("requires a frozen contract" in error for error in report["errors"])


def test_runtime_inventory_signature_can_be_verified_without_manifest_signature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_strict_signatures(monkeypatch)
    _install_scorers(monkeypatch)
    input_path = tmp_path / "input.json"
    artifact_path = tmp_path / "artifact.json"
    inventory_path = tmp_path / "runtime-inventory.json"
    manifest_path = tmp_path / "manifest.json"
    _write_canonical(input_path, _payload())
    artifact._write_new_artifact(artifact_path, artifact.evaluate_joint_evidence_file(input_path))
    _runtime_inventory_fixture(inventory_path)
    _write_run_manifest(
        manifest_path, input_path, artifact_path, runtime_inventory_path=inventory_path
    )

    report = verifier.verify_joint_evidence_run_manifest(
        manifest_path,
        runtime_inventory_signature_path=tmp_path / "runtime.sig",
        trusted_runtime_inventory_signers_path=tmp_path / "runtime.allowed",
        runtime_inventory_signer_id="runtime-owner",
    )

    assert report["status"] == "PASS_INPUT_SCOPED_RECOMPUTATION"
    assert report["runtime_inventory_signature_status"] == "verified"
    assert report["manifest_signature_status"] == "not_provided"


@pytest.mark.parametrize(
    ("runtime_signer", "runtime_keyring", "message"),
    [
        ("manifest-reviewer", "runtime-keyring", "signer must differ from run-manifest signer"),
        (
            "runtime-owner",
            "manifest-keyring",
            "trust keyring must differ from run-manifest keyring",
        ),
        ("policy-custodian", "runtime-keyring", "signer must differ from signer-policy custodian"),
        ("runtime-owner", "policy-keyring", "keyring must differ from signer-policy keyring"),
    ],
)
def test_runtime_inventory_signer_and_keyring_must_be_independent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runtime_signer: str,
    runtime_keyring: str,
    message: str,
) -> None:
    runtime_identity = {
        "signer_id": runtime_signer,
        "allowed_signers_sha256": runtime_keyring,
    }
    _patch_strict_signatures(monkeypatch, runtime_identity=runtime_identity)
    _install_scorers(monkeypatch)
    input_path = tmp_path / "input.json"
    artifact_path = tmp_path / "artifact.json"
    inventory_path = tmp_path / "runtime-inventory.json"
    manifest_path = tmp_path / "manifest.json"
    payload = _payload_v6()
    payload["path_equivalence"]["solver_budget_provenance"] = {
        "max_rays_per_link": 100000,
        "max_interactions": 5,
        "max_episodes": 200,
        "max_wall_seconds": 3600.0,
        "overflow_policy": "record_censored_episode",
    }
    _write_canonical(input_path, payload)
    artifact._write_new_artifact(artifact_path, artifact.evaluate_joint_evidence_file(input_path))
    _runtime_inventory_fixture(inventory_path)
    source_identity = run_manifest.evaluator_source_bundle_identity()
    contract = _qualification_contract(
        payload, source_identity["evaluator_source_bundle_sha256"], raw_paths=True
    )
    _write_run_manifest(
        manifest_path,
        input_path,
        artifact_path,
        qualification_contract=contract,
        runtime_inventory_path=inventory_path,
    )

    report = verifier.verify_joint_evidence_run_manifest(manifest_path, **_strict_options(tmp_path))

    assert report["status"] == "INCONSISTENT"
    assert any(message in error for error in report["errors"])
