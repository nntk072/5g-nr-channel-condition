"""Signed run manifests, signer policies and the joint-evidence artifact writer."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from evaluation import joint_evidence_artifact as artifact
from evaluation import joint_evidence_run_manifest as run_manifest
from evaluation.acceptance import path_equivalence_assurance as path_equivalence_assurance
from evaluation.run_manifest import verify as verify_module
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


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"require_signed_reviewer_admission": 1}, "must be a boolean"),
        (
            {"trusted_signer_policy_path": "policy.json"},
            "requires a verified manifest signature",
        ),
        (
            {"trusted_signer_policy_signature_path": "policy.sig"},
            "signature_path, trusted_policy_allowed_signers_path, and policy_signer_id",
        ),
        (
            {
                "trusted_signer_policy_signature_path": "policy.sig",
                "trusted_policy_allowed_signers_path": "policy.allowed",
                "policy_signer_id": "policy-owner",
            },
            "requires both a signer policy and a signed run manifest",
        ),
        (
            {"runtime_inventory_signature_path": "inventory.sig"},
            "runtime inventory signature, trusted keyring, and signer ID",
        ),
        (
            {"require_signed_reviewer_admission": True},
            "requires a signed run manifest, an externally signed signer policy",
        ),
    ],
)
def test_run_manifest_rejects_incomplete_signature_admission_options(
    tmp_path: Path, options: dict[str, object], message: str
) -> None:
    report = run_manifest.verify_joint_evidence_run_manifest(
        tmp_path / "manifest-not-read.json", **options
    )
    assert report["status"] == "INCONSISTENT"
    assert any(message in error for error in report["errors"])


def test_strict_reviewer_admission_rejects_signer_without_reviewer_role(
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
            "policy_id": "test-non-reviewer-policy",
            "signers": [
                {
                    "principal": signer_id,
                    "organization": "evaluator-lab",
                    "roles": ["evaluator"],
                    "allowed_purposes": [run_manifest.SIGNER_POLICY_PURPOSE],
                }
            ],
        },
    )
    verify_signature = verify_module.verify_openssh_signature_for_bytes

    def verify_with_separate_policy_custodian(
        data, signature, keyring, *, signer_id: str, namespace: str
    ):
        if namespace == run_manifest.SIGNER_POLICY_SIGNATURE_NAMESPACE:
            return {
                "namespace": namespace,
                "signer_id": "policy-owner",
                "signature_sha256": "a" * 64,
                "allowed_signers_sha256": "b" * 64,
            }
        return verify_signature(data, signature, keyring, signer_id=signer_id, namespace=namespace)

    monkeypatch.setattr(
        verify_module, "verify_openssh_signature_for_bytes", verify_with_separate_policy_custodian
    )

    report = run_manifest.verify_joint_evidence_run_manifest(
        manifest_path,
        signature_path=signature_path,
        trusted_allowed_signers_path=allowed_signers,
        trusted_signer_policy_path=policy_path,
        signer_id=signer_id,
        require_signed_reviewer_admission=True,
        trusted_signer_policy_signature_path=tmp_path / "not-read-policy.sig",
        trusted_policy_allowed_signers_path=tmp_path / "not-read-policy.allowed",
        policy_signer_id="policy-owner",
        runtime_inventory_signature_path=tmp_path / "not-read-inventory.sig",
        trusted_runtime_inventory_signers_path=tmp_path / "not-read-inventory.allowed",
        runtime_inventory_signer_id="inventory-owner",
    )

    assert report["status"] == "INCONSISTENT"
    assert any("requires the independent_reviewer role" in error for error in report["errors"])


def test_run_manifest_rejects_inconsistent_retained_package_roster(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_scorers(monkeypatch)
    source = tmp_path / "joint-input.json"
    saved = tmp_path / "joint-report.json"
    manifest_path = tmp_path / "run-manifest.json"
    _write_canonical(source, _payload())
    artifact._write_new_artifact(saved, artifact.evaluate_joint_evidence_file(source))
    _write_run_manifest(manifest_path, source, saved)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    misrostered_sources = [dict(entry) for entry in manifest["evaluator_sources"]]
    misrostered_sources[0]["path"] = "wrong-source.py"

    mutations = [
        (
            {**manifest, "artifact": {**manifest["artifact"], "path": manifest["input"]["path"]}},
            "manifest file paths must be distinct",
        ),
        (
            {**manifest, "input": {**manifest["input"], "sha256": "0" * 64}},
            "input size or SHA-256 differs",
        ),
        (
            {
                **manifest,
                "artifact": {**manifest["artifact"], "role": "joint_evaluator_input"},
            },
            "artifact role must be",
        ),
        (
            {**manifest, "evaluator_sources": misrostered_sources},
            "source roster differs",
        ),
        (
            {**manifest, "evaluator_source_bundle_sha256": "0" * 64},
            "source bundle digest does not match",
        ),
    ]
    for changed, message in mutations:
        _write_canonical(manifest_path, changed)
        report = run_manifest.verify_joint_evidence_run_manifest(manifest_path)
        assert report["status"] == "INCONSISTENT"
        assert any(message in error for error in report["errors"]), report["errors"]

    changed_sources = [dict(entry) for entry in manifest["evaluator_sources"]]
    changed_sources[0]["sha256"] = "0" * 64
    changed = {
        **manifest,
        "evaluator_sources": changed_sources,
        "evaluator_source_bundle_sha256": verify_module._source_bundle_sha256(changed_sources),
    }
    _write_canonical(manifest_path, changed)
    report = run_manifest.verify_joint_evidence_run_manifest(manifest_path)
    assert report["status"] == "INCONSISTENT"
    assert any("evaluator source bytes differ from manifest" in error for error in report["errors"])


def test_run_manifest_rejects_noncanonical_document_and_nonobject_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_scorers(monkeypatch)
    source = tmp_path / "joint-input.json"
    saved = tmp_path / "joint-report.json"
    manifest_path = tmp_path / "run-manifest.json"
    _write_canonical(source, _payload())
    artifact._write_new_artifact(saved, artifact.evaluate_joint_evidence_file(source))
    _write_run_manifest(manifest_path, source, saved)
    canonical_manifest = manifest_path.read_bytes()
    manifest = json.loads(canonical_manifest)

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    report = run_manifest.verify_joint_evidence_run_manifest(manifest_path)
    assert any("canonical compact JSON" in error for error in report["errors"])

    _write_canonical(source, [])
    manifest["input"]["size_bytes"] = source.stat().st_size
    manifest["input"]["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    _write_canonical(manifest_path, manifest)
    report = run_manifest.verify_joint_evidence_run_manifest(manifest_path)
    assert any("joint evaluator input must be an object" in error for error in report["errors"])


@pytest.mark.parametrize(
    ("raw_manifest", "source_root", "message"),
    [
        (b"[]\n", None, "run manifest root must be an object"),
        (b"{}\n", "file", "source_root must be a directory"),
    ],
)
def test_run_manifest_requires_object_document_and_directory_source_root(
    tmp_path: Path, raw_manifest: bytes, source_root: str | None, message: str
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(raw_manifest)
    options: dict[str, Path] = {}
    if source_root == "file":
        source_file = tmp_path / "source-file"
        source_file.write_text("not a directory", encoding="utf-8")
        options["source_root"] = source_file

    report = run_manifest.verify_joint_evidence_run_manifest(manifest_path, **options)
    assert any(message in error for error in report["errors"])


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
    assert report["schema"] == "loslab.joint_evidence_run_verification.v1"
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
    policy_signer_id = "policy-custodian"
    private_key = tmp_path / "policy-custodian-key"
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


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"not-json", "not valid UTF-8 JSON"),
        (b"[]\n", "run verification report root must be an object"),
    ],
)
def test_signed_run_report_verifier_rejects_malformed_report_roots(
    tmp_path: Path, payload: bytes, message: str
) -> None:
    report_path = tmp_path / "verification-report.json"
    report_path.write_bytes(payload)

    with pytest.raises(ValueError, match=message):
        run_manifest.verify_joint_evidence_run_report_signature(
            report_path,
            tmp_path / "report.sig",
            tmp_path / "allowed_signers",
            signer_id="reviewer",
        )


def test_signed_run_report_verifier_builds_a_schema_bound_receipt(
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
    report_path = tmp_path / "verification-report.json"
    _write_canonical(report_path, report)
    monkeypatch.setattr(
        verify_module,
        "verify_openssh_signature_for_bytes",
        lambda *_args, **_kwargs: {
            "namespace": verify_module.RUN_VERIFICATION_REPORT_SIGNATURE_NAMESPACE,
            "signer_id": "reviewer",
            "signature_sha256": "a" * 64,
            "allowed_signers_sha256": "b" * 64,
        },
    )

    receipt = run_manifest.verify_joint_evidence_run_report_signature(
        report_path,
        tmp_path / "report.sig",
        tmp_path / "allowed_signers",
        signer_id="reviewer",
    )

    assert receipt["status"] == "SIGNATURE_VERIFIED"
    assert receipt["qualification_status"] == "NOT_QUALIFIED"
    assert receipt["run_id"] == report["run_id"]
    assert receipt["report_schema_sha256"] == report["report_schema_sha256"]

    original_read = verify_module._read_bounded_regular_file

    def invalid_receipt_schema(path, *, limit: int, description: str) -> bytes:
        if description == "report signature receipt schema":
            return b'{"type":"object","required":["unavailable"]}'
        return original_read(path, limit=limit, description=description)

    monkeypatch.setattr(verify_module, "_read_bounded_regular_file", invalid_receipt_schema)
    with pytest.raises(ValueError, match="receipt violates schema"):
        run_manifest.verify_joint_evidence_run_report_signature(
            report_path,
            tmp_path / "report.sig",
            tmp_path / "allowed_signers",
            signer_id="reviewer",
        )


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
