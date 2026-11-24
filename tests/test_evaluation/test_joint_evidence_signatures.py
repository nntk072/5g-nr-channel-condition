"""Signed run manifests, signer policies and the joint-evidence artifact writer."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from evaluation import joint_evidence_artifact as artifact
from evaluation import joint_evidence_run_manifest as run_manifest
from evaluation import path_equivalence_assurance as path_equivalence_assurance
from tests.test_evaluation.joint_evidence_artifact_support import (
    _install_scorers,
    _payload,
    _payload_v4,
    _qualification_contract,
    _sign_run_manifest,
    _write_canonical,
    _write_run_manifest,
)


@pytest.mark.parametrize(
    "raw",
    [
        b'{"a":1,"a":2}\n',
        b'{"a":NaN}\n',
        b'{ "a": 1 }\n',
    ],
)
def test_joint_input_rejects_ambiguous_or_noncanonical_json(tmp_path: Path, raw: bytes) -> None:
    source = tmp_path / "inputs.json"
    source.write_bytes(raw)
    with pytest.raises(ValueError):
        artifact.evaluate_joint_evidence_file(source)


def test_artifact_writer_refuses_to_overwrite(tmp_path: Path) -> None:
    saved = tmp_path / "bundle.json"
    saved.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError):
        artifact._write_new_artifact(saved, {"schema": artifact.ARTIFACT_SCHEMA})
    assert saved.read_text(encoding="utf-8") == "preserve"


def test_strict_signed_reviewer_admission_rejects_unsigned_run_before_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_scorers(monkeypatch)
    source = tmp_path / "joint-input.json"
    saved = tmp_path / "joint-report.json"
    _write_canonical(source, _payload())
    artifact._write_new_artifact(saved, artifact.evaluate_joint_evidence_file(source))
    manifest_path = tmp_path / "run-manifest.json"
    _write_run_manifest(manifest_path, source, saved)

    report = run_manifest.verify_joint_evidence_run_manifest(
        manifest_path, require_signed_reviewer_admission=True
    )

    assert report["status"] == "INCONSISTENT"
    assert report["qualification_status"] == "NOT_QUALIFIED"
    assert report["signed_reviewer_admission_status"] == "required"
    assert any("strict signed reviewer admission requires" in error for error in report["errors"])


def test_run_manifest_verifies_detached_signature_over_exact_manifest_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_scorers(monkeypatch)
    source = tmp_path / "joint-input.json"
    saved = tmp_path / "joint-report.json"
    manifest_path = tmp_path / "run-manifest.json"
    _write_canonical(source, _payload())
    artifact._write_new_artifact(saved, artifact.evaluate_joint_evidence_file(source))
    _write_run_manifest(manifest_path, source, saved)
    signature_path, allowed_signers, signer_id = _sign_run_manifest(manifest_path, tmp_path)

    report = run_manifest.verify_joint_evidence_run_manifest(
        manifest_path,
        signature_path=signature_path,
        trusted_allowed_signers_path=allowed_signers,
        signer_id=signer_id,
    )

    assert report["status"] == "PASS_INPUT_SCOPED_RECOMPUTATION"
    assert report["qualification_status"] == "NOT_QUALIFIED"
    assert report["manifest_signature_status"] == "verified"
    assert report["manifest_signer_id"] == signer_id
    assert report["trust_file_custody"] == "not_established_by_this_verifier"


def test_run_manifest_cli_publishes_schema_bound_report_without_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_scorers(monkeypatch)
    source = tmp_path / "joint-input.json"
    saved = tmp_path / "joint-report.json"
    manifest_path = tmp_path / "run-manifest.json"
    verification_path = tmp_path / "run-verification.json"
    _write_canonical(source, _payload())
    artifact._write_new_artifact(saved, artifact.evaluate_joint_evidence_file(source))
    _write_run_manifest(manifest_path, source, saved)

    exit_code = run_manifest.main(
        [
            str(manifest_path),
            "--package-root",
            str(tmp_path),
            "--report-output",
            str(verification_path),
        ]
    )

    report = json.loads(verification_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["schema"] == "loslab.g3_joint_evidence_run_verification.v1"
    assert report["qualification_status"] == "NOT_QUALIFIED"
    assert len(report["report_schema_sha256"]) == 64
    assert (
        run_manifest.main(
            [
                str(manifest_path),
                "--package-root",
                str(tmp_path),
                "--report-output",
                str(verification_path),
            ]
        )
        == 2
    )


def test_run_manifest_signature_resolves_external_signer_role_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_scorers(monkeypatch)
    source = tmp_path / "joint-input.json"
    saved = tmp_path / "joint-report.json"
    manifest_path = tmp_path / "run-manifest.json"
    policy_path = tmp_path / "signer-policy.json"
    _write_canonical(source, _payload())
    artifact._write_new_artifact(saved, artifact.evaluate_joint_evidence_file(source))
    _write_run_manifest(manifest_path, source, saved)
    signature_path, allowed_signers, signer_id = _sign_run_manifest(manifest_path, tmp_path)
    _write_canonical(
        policy_path,
        {
            "schema": run_manifest.SIGNER_POLICY_SCHEMA,
            "policy_id": "test-role-policy",
            "signers": [
                {
                    "principal": signer_id,
                    "organization": "independent-lab",
                    "roles": ["independent_reviewer"],
                    "allowed_purposes": [run_manifest.SIGNER_POLICY_PURPOSE],
                }
            ],
        },
    )

    report = run_manifest.verify_joint_evidence_run_manifest(
        manifest_path,
        signature_path=signature_path,
        trusted_allowed_signers_path=allowed_signers,
        trusted_signer_policy_path=policy_path,
        signer_id=signer_id,
    )

    assert report["status"] == "PASS_INPUT_SCOPED_RECOMPUTATION"
    assert report["qualification_status"] == "NOT_QUALIFIED"
    assert report["signer_authorization_status"] == "authorized_for_run_manifest_verification"
    assert report["manifest_signer_roles"] == "independent_reviewer"
    assert report["manifest_signer_organization"] == "independent-lab"
    assert report["trusted_signer_policy_id"] == "test-role-policy"


@pytest.mark.parametrize(
    ("policy_identity", "message"),
    (
        (
            {"signer_id": "reviewer", "allowed_signers_sha256": "b" * 64},
            "custodian must differ",
        ),
        (
            {"signer_id": "custodian", "allowed_signers_sha256": "a" * 64},
            "keyring must differ",
        ),
    ),
)
def test_signer_policy_requires_independent_custodian_and_keyring(
    policy_identity: dict[str, str], message: str
) -> None:
    manifest_identity = {
        "signer_id": "reviewer",
        "allowed_signers_sha256": "a" * 64,
    }

    with pytest.raises(ValueError, match=message):
        run_manifest._require_independent_signer_policy_custody(manifest_identity, policy_identity)


def test_run_manifest_rejects_tampered_signed_signer_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_scorers(monkeypatch)
    source = tmp_path / "joint-input.json"
    saved = tmp_path / "joint-report.json"
    manifest_path = tmp_path / "run-manifest.json"
    policy_path = tmp_path / "signer-policy.json"
    _write_canonical(source, _payload())
    artifact._write_new_artifact(saved, artifact.evaluate_joint_evidence_file(source))
    _write_run_manifest(manifest_path, source, saved)
    manifest_signature, manifest_signers, manifest_signer_id = _sign_run_manifest(
        manifest_path, tmp_path
    )
    _write_canonical(
        policy_path,
        {
            "schema": run_manifest.SIGNER_POLICY_SCHEMA,
            "policy_id": "signed-policy",
            "signers": [
                {
                    "principal": manifest_signer_id,
                    "organization": "independent-lab",
                    "roles": ["independent_reviewer"],
                    "allowed_purposes": [run_manifest.SIGNER_POLICY_PURPOSE],
                }
            ],
        },
    )
    ssh_keygen = shutil.which("ssh-keygen")
    if ssh_keygen is None:
        pytest.skip("OpenSSH is unavailable")
    policy_signer_id = "g3-policy-custodian"
    private_key = tmp_path / "g3-policy-custodian-key"
    subprocess.run(
        [ssh_keygen, "-q", "-t", "ed25519", "-N", "", "-f", str(private_key)],
        check=True,
        capture_output=True,
        text=True,
    )
    public_key = private_key.with_suffix(".pub").read_text(encoding="utf-8").strip()
    policy_signers = tmp_path / "policy-allowed-signers"
    policy_signers.write_text(
        f'{policy_signer_id} namespaces="{run_manifest.SIGNER_POLICY_SIGNATURE_NAMESPACE}" '
        f"{public_key}\n",
        encoding="utf-8",
    )
    subprocess.run(
        [
            ssh_keygen,
            "-Y",
            "sign",
            "-n",
            run_manifest.SIGNER_POLICY_SIGNATURE_NAMESPACE,
            "-f",
            str(private_key),
            str(policy_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    policy_signature = Path(f"{policy_path}.sig")
    _write_canonical(policy_path, {"schema": run_manifest.SIGNER_POLICY_SCHEMA})

    report = run_manifest.verify_joint_evidence_run_manifest(
        manifest_path,
        signature_path=manifest_signature,
        trusted_allowed_signers_path=manifest_signers,
        trusted_signer_policy_path=policy_path,
        signer_id=manifest_signer_id,
        trusted_signer_policy_signature_path=policy_signature,
        trusted_policy_allowed_signers_path=policy_signers,
        policy_signer_id=policy_signer_id,
    )

    assert report["status"] == "INCONSISTENT"
    assert any(
        "signature is invalid or signer is not trusted" in error for error in report["errors"]
    )


def test_run_manifest_requires_all_signer_policy_signature_inputs_together(
    tmp_path: Path,
) -> None:
    report = run_manifest.verify_joint_evidence_run_manifest(
        tmp_path / "missing-manifest.json",
        trusted_signer_policy_signature_path=tmp_path / "policy.json.sig",
    )

    assert report["status"] == "INCONSISTENT"
    assert any("must be supplied together" in error for error in report["errors"])


def test_run_manifest_rejects_role_policy_without_run_manifest_purpose(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_scorers(monkeypatch)
    source = tmp_path / "joint-input.json"
    saved = tmp_path / "joint-report.json"
    manifest_path = tmp_path / "run-manifest.json"
    policy_path = tmp_path / "signer-policy.json"
    _write_canonical(source, _payload())
    artifact._write_new_artifact(saved, artifact.evaluate_joint_evidence_file(source))
    _write_run_manifest(manifest_path, source, saved)
    signature_path, allowed_signers, signer_id = _sign_run_manifest(manifest_path, tmp_path)
    _write_canonical(
        policy_path,
        {
            "schema": run_manifest.SIGNER_POLICY_SCHEMA,
            "policy_id": "wrong-purpose-policy",
            "signers": [
                {
                    "principal": signer_id,
                    "organization": "independent-lab",
                    "roles": ["independent_reviewer"],
                    "allowed_purposes": ["other-purpose"],
                }
            ],
        },
    )

    report = run_manifest.verify_joint_evidence_run_manifest(
        manifest_path,
        signature_path=signature_path,
        trusted_allowed_signers_path=allowed_signers,
        trusted_signer_policy_path=policy_path,
        signer_id=signer_id,
    )

    assert report["status"] == "INCONSISTENT"
    assert any("unsupported run-manifest purpose" in error for error in report["errors"])


def test_run_manifest_signature_rejects_changed_manifest_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_scorers(monkeypatch)
    source = tmp_path / "joint-input.json"
    saved = tmp_path / "joint-report.json"
    manifest_path = tmp_path / "run-manifest.json"
    _write_canonical(source, _payload())
    artifact._write_new_artifact(saved, artifact.evaluate_joint_evidence_file(source))
    _write_run_manifest(manifest_path, source, saved)
    signature_path, allowed_signers, signer_id = _sign_run_manifest(manifest_path, tmp_path)
    manifest_path.write_bytes(manifest_path.read_bytes() + b" ")

    report = run_manifest.verify_joint_evidence_run_manifest(
        manifest_path,
        signature_path=signature_path,
        trusted_allowed_signers_path=allowed_signers,
        signer_id=signer_id,
    )

    assert report["status"] == "INCONSISTENT"
    assert any(
        "signature is invalid or signer is not trusted" in error for error in report["errors"]
    )


def test_run_manifest_signature_options_must_be_supplied_together(tmp_path: Path) -> None:
    report = run_manifest.verify_joint_evidence_run_manifest(
        tmp_path / "manifest.json",
        signature_path=tmp_path / "manifest.json.sig",
    )

    assert report["status"] == "INCONSISTENT"
    assert any("must be supplied together" in error for error in report["errors"])


def test_preregistered_run_preflights_then_writes_artifact_and_bound_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_scorers(monkeypatch)
    source = tmp_path / "joint-input.json"
    contract_path = tmp_path / "qualification-contract.json"
    saved = tmp_path / "joint-report.json"
    manifest_path = tmp_path / "run-manifest.json"
    payload = _payload_v4()
    _write_canonical(source, payload)
    source_identity = run_manifest.evaluator_source_bundle_identity()
    contract = _qualification_contract(payload, source_identity["evaluator_source_bundle_sha256"])
    _write_canonical(contract_path, contract)

    result = artifact.run_frozen_qualification_evaluation(
        source, contract_path, saved, manifest_path
    )

    assert result["status"] == "CONTRACT_BOUND_NUMERICAL_EVALUATION_NOT_QUALIFIED"
    assert result["qualification_status"] == "NOT_QUALIFIED"
    assert saved.is_file()
    assert manifest_path.is_file()
    receipt = run_manifest.verify_joint_evidence_run_manifest(manifest_path)
    assert receipt["status"] == "PASS_INPUT_SCOPED_RECOMPUTATION"
    assert receipt["qualification_contract_bound"] is True
