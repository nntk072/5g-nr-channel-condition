from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from evaluation.g3_joint_evidence_run_manifest import (
    G3_RUN_VERIFICATION_REPORT_SIGNATURE_NAMESPACE,
    verify_g3_joint_evidence_run_report_signature,
)


def _signed_report(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    ssh_keygen = shutil.which("ssh-keygen")
    if ssh_keygen is None:
        pytest.skip("OpenSSH ssh-keygen is required for signature verification tests")
    key_path = tmp_path / "reviewer_key"
    subprocess.run(
        [ssh_keygen, "-q", "-t", "ed25519", "-N", "", "-f", str(key_path)],
        check=True,
        capture_output=True,
    )
    public_key = subprocess.run(
        [ssh_keygen, "-y", "-f", str(key_path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    signer = "reviewer@example.org"
    allowed_signers = tmp_path / "allowed_signers"
    allowed_signers.write_text(
        f'{signer} namespaces="{G3_RUN_VERIFICATION_REPORT_SIGNATURE_NAMESPACE}" {public_key}\n',
        encoding="utf-8",
    )
    report_schema = (
        Path(__file__).resolve().parents[2]
        / "datasets/schema/g3-joint-evidence-run-verification.v1.schema.json"
    )
    report_schema_hash = hashlib.sha256(report_schema.read_bytes()).hexdigest()
    report = {
        "schema": "loslab.g3_joint_evidence_run_verification.v1",
        "status": "PASS_INPUT_SCOPED_RECOMPUTATION",
        "qualification_status": "NOT_QUALIFIED",
        "run_id": "fixture.run",
        "manifest_signature_status": "not_provided",
        "signer_authorization_status": "not_provided",
        "signer_policy_signature_status": "not_provided",
        "signed_reviewer_admission_status": "not_requested",
        "signer_policy_custody": "not_established_by_this_verifier",
        "runtime_identity_status": "not_recorded",
        "report_schema_sha256": report_schema_hash,
        "errors": [],
        "limitations": ["Synthetic test report; no scientific qualification."],
    }
    report_path = tmp_path / "run-report.json"
    report_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    subprocess.run(
        [
            ssh_keygen,
            "-Y",
            "sign",
            "-f",
            str(key_path),
            "-n",
            G3_RUN_VERIFICATION_REPORT_SIGNATURE_NAMESPACE,
            str(report_path),
        ],
        check=True,
        capture_output=True,
    )
    return report_path, Path(f"{report_path}.sig"), allowed_signers, key_path


def test_verifies_exact_report_bytes_and_emits_nonqualifying_receipt(tmp_path: Path) -> None:
    report, signature, allowed_signers, _ = _signed_report(tmp_path)

    receipt = verify_g3_joint_evidence_run_report_signature(
        report, signature, allowed_signers, signer_id="reviewer@example.org"
    )

    assert receipt["status"] == "SIGNATURE_VERIFIED"
    assert receipt["qualification_status"] == "NOT_QUALIFIED"
    assert receipt["run_id"] == "fixture.run"
    assert receipt["signed_report_sha256"] == hashlib.sha256(report.read_bytes()).hexdigest()
    assert receipt["signature_namespace"] == G3_RUN_VERIFICATION_REPORT_SIGNATURE_NAMESPACE


def test_rejects_report_bytes_changed_after_signature(tmp_path: Path) -> None:
    report, signature, allowed_signers, _ = _signed_report(tmp_path)
    report.write_bytes(report.read_bytes() + b" ")

    with pytest.raises(ValueError, match="signature is invalid"):
        verify_g3_joint_evidence_run_report_signature(
            report, signature, allowed_signers, signer_id="reviewer@example.org"
        )


def test_signature_cli_publishes_non_overwriting_receipt(tmp_path: Path) -> None:
    report, signature, allowed_signers, _ = _signed_report(tmp_path)
    receipt_path = tmp_path / "signature-receipt.json"
    command = [
        sys.executable,
        "-m",
        "scripts.verify_g3_run_report_signature",
        str(report),
        "--signature",
        str(signature),
        "--trusted-allowed-signers",
        str(allowed_signers),
        "--signer-id",
        "reviewer@example.org",
        "--output",
        str(receipt_path),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    receipt = json.loads(receipt_path.read_bytes())
    assert receipt["qualification_status"] == "NOT_QUALIFIED"

    failed = subprocess.run(command, capture_output=True, text=True)
    assert failed.returncode != 0
    assert "File exists" in failed.stderr
