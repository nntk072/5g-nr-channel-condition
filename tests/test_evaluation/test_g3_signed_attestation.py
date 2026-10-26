from __future__ import annotations

import hashlib
import json
import runpy
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from evaluation.g3_signed_attestation import (
    SCHEMA,
    SIGNATURE_NAMESPACE,
    TRUST_POLICY_SCHEMA,
    verify_g3_signed_attestation,
    verify_g3_signed_attestation_set,
)

_VERIFIER_HELPERS = Path(__file__).with_name("test_g3_acquisition_package_verifier.py")
_make_package = runpy.run_path(str(_VERIFIER_HELPERS))["_make_package"]


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _make_signed_inputs(
    tmp_path: Path, *, schema_version: str = "g3.03-acquisition-package.v0.3"
) -> tuple[Path, Path, Path, Path, Path, Path]:
    ssh_keygen = shutil.which("ssh-keygen")
    if ssh_keygen is None:
        pytest.skip("OpenSSH ssh-keygen is required for signature verification tests")
    key = tmp_path / "reviewer_key"
    subprocess.run(
        [ssh_keygen, "-q", "-t", "ed25519", "-N", "", "-f", str(key)],
        check=True,
        capture_output=True,
    )
    public_key = subprocess.run(
        [ssh_keygen, "-y", "-f", str(key)], check=True, capture_output=True, text=True
    ).stdout.strip()
    allowed_signers = tmp_path / "allowed_signers"
    allowed_signers.write_text(
        f'reviewer@example.org namespaces="{SIGNATURE_NAMESPACE}" {public_key}\n',
        encoding="utf-8",
    )

    package_path = _make_package(tmp_path, schema_version=schema_version)
    package = json.loads(package_path.read_bytes())
    attestation_record = next(
        item
        for item in package["external_attestations"]
        if item["attestation_id"] == "attest.split.plan"
    )
    inventory = {item["file_id"]: item for item in package["files"]}
    envelope = {
        "schema": SCHEMA,
        "package_sha256": hashlib.sha256(package_path.read_bytes()).hexdigest(),
        "attestation_id": "attest.split.plan",
        "attestation_type": attestation_record["attestation_type"],
        "claim_template": attestation_record["claim_template"],
        "subject_files": [
            {
                "file_id": file_id,
                "sha256": inventory[file_id]["sha256"],
                "size_bytes": inventory[file_id]["size_bytes"],
            }
            for file_id in attestation_record["subject_file_ids"]
        ],
        "signer_id": "reviewer@example.org",
        "signer_role": attestation_record["signer_role"],
        "signing_organization": "Example independent institution",
        "signed_utc": "2025-12-31T09:30:00Z",
        "signature_namespace": SIGNATURE_NAMESPACE,
    }
    envelope_path = tmp_path / "attestation.json"
    envelope_path.write_bytes(_canonical_bytes(envelope))
    signer_policy_path = tmp_path / "signer-policy.json"
    signer_policy_path.write_bytes(
        _canonical_bytes(
            {
                "schema": TRUST_POLICY_SCHEMA,
                "policy_id": "synthetic-test-policy",
                "signers": [
                    {
                        "principal": envelope["signer_id"],
                        "organization": envelope["signing_organization"],
                        "allowed_attestation_types": [envelope["attestation_type"]],
                        "allowed_signer_roles": [envelope["signer_role"]],
                    }
                ],
            }
        )
    )
    subprocess.run(
        [ssh_keygen, "-Y", "sign", "-f", str(key), "-n", SIGNATURE_NAMESPACE, str(envelope_path)],
        check=True,
        capture_output=True,
    )
    return (
        package_path,
        envelope_path,
        envelope_path.with_suffix(".json.sig"),
        allowed_signers,
        signer_policy_path,
        key,
    )


def _make_signed_set(
    tmp_path: Path,
    *,
    schema_version: str = "g3.03-acquisition-package.v0.3",
    omit_attestation_type: str | None = None,
) -> tuple[Path, list[Path], Path, Path]:
    ssh_keygen = shutil.which("ssh-keygen")
    if ssh_keygen is None:
        pytest.skip("OpenSSH ssh-keygen is required for signature verification tests")
    key = tmp_path / "reviewer_key"
    subprocess.run(
        [ssh_keygen, "-q", "-t", "ed25519", "-N", "", "-f", str(key)],
        check=True,
        capture_output=True,
    )
    public_key = subprocess.run(
        [ssh_keygen, "-y", "-f", str(key)], check=True, capture_output=True, text=True
    ).stdout.strip()
    allowed_signers = tmp_path / "allowed_signers"
    allowed_signers.write_text(
        f'reviewer@example.org namespaces="{SIGNATURE_NAMESPACE}" {public_key}\n',
        encoding="utf-8",
    )
    package_path = _make_package(tmp_path, schema_version=schema_version)
    package = json.loads(package_path.read_bytes())
    if omit_attestation_type is not None:
        package["external_attestations"] = [
            item
            for item in package["external_attestations"]
            if item["attestation_type"] != omit_attestation_type
        ]
        package_path.write_bytes(_canonical_bytes(package))
    inventory = {item["file_id"]: item for item in package["files"]}
    records = package["external_attestations"]
    policy = {
        "schema": TRUST_POLICY_SCHEMA,
        "policy_id": "synthetic-complete-test-policy",
        "signers": [
            {
                "principal": "reviewer@example.org",
                "organization": "Example independent institution",
                "allowed_attestation_types": sorted({item["attestation_type"] for item in records}),
                "allowed_signer_roles": sorted({item["signer_role"] for item in records}),
            }
        ],
    }
    policy_path = tmp_path / "signer-policy.json"
    policy_path.write_bytes(_canonical_bytes(policy))
    package_sha256 = hashlib.sha256(package_path.read_bytes()).hexdigest()
    envelopes: list[Path] = []
    for index, record in enumerate(records):
        envelope = {
            "schema": SCHEMA,
            "package_sha256": package_sha256,
            "attestation_id": record["attestation_id"],
            "attestation_type": record["attestation_type"],
            "claim_template": record["claim_template"],
            "subject_files": [
                {
                    "file_id": file_id,
                    "sha256": inventory[file_id]["sha256"],
                    "size_bytes": inventory[file_id]["size_bytes"],
                }
                for file_id in sorted(record["subject_file_ids"])
            ],
            "signer_id": "reviewer@example.org",
            "signer_role": record["signer_role"],
            "signing_organization": "Example independent institution",
            "signed_utc": f"2025-12-31T09:{30 + index:02d}:00Z",
            "signature_namespace": SIGNATURE_NAMESPACE,
        }
        envelope_path = tmp_path / f"{record['attestation_id']}.json"
        envelope_path.write_bytes(_canonical_bytes(envelope))
        subprocess.run(
            [
                ssh_keygen,
                "-Y",
                "sign",
                "-f",
                str(key),
                "-n",
                SIGNATURE_NAMESPACE,
                str(envelope_path),
            ],
            check=True,
            capture_output=True,
        )
        envelopes.append(envelope_path)
    return package_path, envelopes, allowed_signers, policy_path


def test_g3_signed_attestation_binds_exact_package_and_subject_file(tmp_path: Path) -> None:
    package, envelope, signature, trusted, policy, _ = _make_signed_inputs(tmp_path)

    report = verify_g3_signed_attestation(package, envelope, signature, trusted, policy)

    split_manifest_entry = next(
        item
        for item in json.loads(package.read_bytes())["files"]
        if item["file_id"] == "file.split.manifest"
    )
    assert report["signature_status"] == "verified"
    assert report["attestation_type"] == "split_preregistration"
    assert report["subject_file_ids"] == ["file.split.manifest"]
    assert report["subject_files"] == [
        {
            "file_id": "file.split.manifest",
            "sha256": split_manifest_entry["sha256"],
            "size_bytes": split_manifest_entry["size_bytes"],
        }
    ]
    assert report["package_consistency_status"] == "consistent_proposal_only"
    assert report["qualification_status"] == "NOT_QUALIFIED"
    assert report["scientific_claim_status"] == "not_assessed"
    assert (
        report["declared_temporal_order_status"]
        == "preregistration_precedes_declared_held_out_acquisitions"
    )
    assert report["held_out_acquisition_sample_count"] == 1
    assert report["latest_bound_acquisition_utc"] == "2026-01-01T12:30:00Z"
    assert len(report["verifier_source_sha256"]) == 64
    assert len(report["trusted_allowed_signers_sha256"]) == 64
    assert len(report["trusted_signer_policy_sha256"]) == 64


def test_split_preregistration_must_precede_declared_held_out_acquisition(
    tmp_path: Path,
) -> None:
    package, envelope, signature, trusted, policy, key = _make_signed_inputs(tmp_path)
    envelope_document = json.loads(envelope.read_bytes())
    envelope_document["signed_utc"] = "2026-01-01T12:30:00Z"
    envelope.write_bytes(_canonical_bytes(envelope_document))
    signature.unlink()
    subprocess.run(
        [
            shutil.which("ssh-keygen") or "ssh-keygen",
            "-Y",
            "sign",
            "-f",
            str(key),
            "-n",
            SIGNATURE_NAMESPACE,
            str(envelope),
        ],
        check=True,
        capture_output=True,
    )

    with pytest.raises(
        ValueError,
        match="signed_utc must precede every declared held-out acquisition",
    ):
        verify_g3_signed_attestation(package, envelope, signature, trusted, policy)


def test_split_preregistration_signature_must_include_bound_split_manifest(
    tmp_path: Path,
) -> None:
    package, envelope, signature, trusted, policy, key = _make_signed_inputs(tmp_path)
    package_document = json.loads(package.read_bytes())
    record = next(
        item
        for item in package_document["external_attestations"]
        if item["attestation_id"] == "attest.split.plan"
    )
    record["subject_file_ids"] = ["file.scorer.input"]
    package_bytes = json.dumps(package_document, indent=2).encode()
    package.write_bytes(package_bytes)

    envelope_document = json.loads(envelope.read_bytes())
    envelope_document["package_sha256"] = hashlib.sha256(package_bytes).hexdigest()
    scorer_entry = next(
        item for item in package_document["files"] if item["file_id"] == "file.scorer.input"
    )
    envelope_document["subject_files"] = [
        {
            "file_id": scorer_entry["file_id"],
            "sha256": scorer_entry["sha256"],
            "size_bytes": scorer_entry["size_bytes"],
        }
    ]
    envelope.write_bytes(_canonical_bytes(envelope_document))
    signature.unlink()
    subprocess.run(
        [
            shutil.which("ssh-keygen") or "ssh-keygen",
            "-Y",
            "sign",
            "-f",
            str(key),
            "-n",
            SIGNATURE_NAMESPACE,
            str(envelope),
        ],
        check=True,
        capture_output=True,
    )

    with pytest.raises(ValueError, match="must sign the evaluator-bound split manifest"):
        verify_g3_signed_attestation(package, envelope, signature, trusted, policy)


def test_g3_signed_attestation_supports_legacy_v02_package(tmp_path: Path) -> None:
    package, envelope, signature, trusted, policy, _ = _make_signed_inputs(
        tmp_path, schema_version="g3.03-acquisition-package.v0.2"
    )

    report = verify_g3_signed_attestation(package, envelope, signature, trusted, policy)

    assert report["signature_status"] == "verified"
    assert report["package_consistency_status"] == "consistent_proposal_only"


def test_g3_signed_attestation_supports_v04_csi_noise_evidence_package(
    tmp_path: Path,
) -> None:
    package, envelope, signature, trusted, policy, _ = _make_signed_inputs(
        tmp_path, schema_version="g3.03-acquisition-package.v0.4"
    )

    report = verify_g3_signed_attestation(package, envelope, signature, trusted, policy)

    assert report["signature_status"] == "verified"
    assert report["package_consistency_status"] == "consistent_proposal_only"
    assert report["qualification_status"] == "NOT_QUALIFIED"


def test_g3_signed_attestation_rejects_package_byte_change(tmp_path: Path) -> None:
    package, envelope, signature, trusted, policy, _ = _make_signed_inputs(tmp_path)
    package.write_bytes(package.read_bytes() + b" ")

    with pytest.raises(ValueError, match="package_sha256 does not match exact package bytes"):
        verify_g3_signed_attestation(package, envelope, signature, trusted, policy)


def test_g3_signed_attestation_rejects_subject_digest_mismatch(tmp_path: Path) -> None:
    package, envelope, signature, trusted, policy, _ = _make_signed_inputs(tmp_path)
    document = json.loads(envelope.read_bytes())
    document["subject_files"][0]["sha256"] = "0" * 64
    envelope.write_bytes(_canonical_bytes(document))

    with pytest.raises(ValueError, match="differs from package inventory"):
        verify_g3_signed_attestation(package, envelope, signature, trusted, policy)


def test_g3_signed_attestation_rejects_tampered_signed_bytes(tmp_path: Path) -> None:
    package, envelope, signature, trusted, policy, _ = _make_signed_inputs(tmp_path)
    document = json.loads(envelope.read_bytes())
    document["signed_utc"] = "2025-12-31T09:31:00Z"
    envelope.write_bytes(_canonical_bytes(document))

    with pytest.raises(ValueError, match="signature is invalid or signer is not trusted"):
        verify_g3_signed_attestation(package, envelope, signature, trusted, policy)


def test_g3_signed_attestation_rejects_untrusted_principal(tmp_path: Path) -> None:
    package, envelope, signature, trusted, policy, _ = _make_signed_inputs(tmp_path)
    public_key = subprocess.run(
        [shutil.which("ssh-keygen") or "ssh-keygen", "-y", "-f", str(tmp_path / "reviewer_key")],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    trusted.write_text(
        f'other@example.org namespaces="{SIGNATURE_NAMESPACE}" {public_key}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="signature is invalid or signer is not trusted"):
        verify_g3_signed_attestation(package, envelope, signature, trusted, policy)


def test_g3_signed_attestation_rejects_unauthorized_role(tmp_path: Path) -> None:
    package, envelope, signature, trusted, policy, _ = _make_signed_inputs(tmp_path)
    trust = json.loads(policy.read_bytes())
    trust["signers"][0]["allowed_signer_roles"] = ["unrelated role"]
    policy.write_bytes(_canonical_bytes(trust))

    with pytest.raises(ValueError, match="signer role is not authorized"):
        verify_g3_signed_attestation(package, envelope, signature, trusted, policy)


def test_g3_signed_attestation_rejects_unauthorized_organization(tmp_path: Path) -> None:
    package, envelope, signature, trusted, policy, _ = _make_signed_inputs(tmp_path)
    trust = json.loads(policy.read_bytes())
    trust["signers"][0]["organization"] = "Different institution"
    policy.write_bytes(_canonical_bytes(trust))

    with pytest.raises(ValueError, match="organization is not authorized"):
        verify_g3_signed_attestation(package, envelope, signature, trusted, policy)


def test_g3_signed_attestation_rejects_unauthorized_attestation_type(tmp_path: Path) -> None:
    package, envelope, signature, trusted, policy, _ = _make_signed_inputs(tmp_path)
    trust = json.loads(policy.read_bytes())
    trust["signers"][0]["allowed_attestation_types"] = ["physical_independence"]
    policy.write_bytes(_canonical_bytes(trust))

    with pytest.raises(ValueError, match="attestation type is not authorized"):
        verify_g3_signed_attestation(package, envelope, signature, trusted, policy)


def test_g3_signed_attestation_rejects_hardlinked_signature(tmp_path: Path) -> None:
    package, envelope, signature, trusted, policy, _ = _make_signed_inputs(tmp_path)
    alias = tmp_path / "signature-copy.sig"
    alias.hardlink_to(signature)

    with pytest.raises(ValueError, match="single-link regular file"):
        verify_g3_signed_attestation(package, envelope, alias, trusted, policy)


def test_g3_signed_attestation_rejects_inconsistent_package_inventory(tmp_path: Path) -> None:
    package, envelope, signature, trusted, policy, _ = _make_signed_inputs(tmp_path)
    document = json.loads(package.read_bytes())
    manifest_entry = next(
        item for item in document["files"] if item["file_id"] == "file.split.manifest"
    )
    (tmp_path / manifest_entry["path"]).write_bytes(b"changed bytes\n")

    with pytest.raises(ValueError, match="fails the proposal-package consistency verifier"):
        verify_g3_signed_attestation(package, envelope, signature, trusted, policy)


def test_g3_signed_attestation_set_verifies_every_package_record(tmp_path: Path) -> None:
    package, envelopes, trusted, policy = _make_signed_set(tmp_path)

    report = verify_g3_signed_attestation_set(package, envelopes, trusted, policy)

    assert report["signature_status"] == "all_declared_attestations_verified"
    assert report["attestation_count"] == 5
    assert report["unique_signer_count"] == 1
    assert report["qualification_status"] == "NOT_QUALIFIED"
    assert "signer independence requires external review" in report["limitations"][1]


def test_g3_signed_attestation_set_supports_v04_package(tmp_path: Path) -> None:
    package, envelopes, trusted, policy = _make_signed_set(
        tmp_path, schema_version="g3.03-acquisition-package.v0.4"
    )

    report = verify_g3_signed_attestation_set(package, envelopes, trusted, policy)

    assert report["signature_status"] == "all_declared_attestations_verified"
    assert report["attestation_count"] == 5
    assert report["package_consistency_status"] == "consistent_proposal_only"
    assert report["qualification_status"] == "NOT_QUALIFIED"


def test_g3_signed_attestation_set_can_require_complete_qualification_types(
    tmp_path: Path,
) -> None:
    package, envelopes, trusted, policy = _make_signed_set(tmp_path)

    report = verify_g3_signed_attestation_set(
        package,
        envelopes,
        trusted,
        policy,
        require_complete_qualification_attestations=True,
    )

    assert report["qualification_attestation_coverage_status"] == "complete_type_roster_verified"
    assert report["missing_qualification_attestation_types"] == []
    assert report["qualification_status"] == "NOT_QUALIFIED"


def test_g3_signed_attestation_set_strict_mode_rejects_missing_scientific_review(
    tmp_path: Path,
) -> None:
    package, envelopes, trusted, policy = _make_signed_set(
        tmp_path,
        schema_version="g3.03-acquisition-package.v0.4",
        omit_attestation_type="scientific_review",
    )

    with pytest.raises(ValueError, match="missing types: scientific_review"):
        verify_g3_signed_attestation_set(
            package,
            envelopes,
            trusted,
            policy,
            require_complete_qualification_attestations=True,
        )

    compatible_report = verify_g3_signed_attestation_set(package, envelopes, trusted, policy)
    assert (
        compatible_report["qualification_attestation_coverage_status"] == "incomplete_type_roster"
    )


def test_g3_signed_attestation_set_strict_mode_requires_strict_boolean(
    tmp_path: Path,
) -> None:
    package, envelopes, trusted, policy = _make_signed_set(tmp_path)

    with pytest.raises(TypeError, match="must be Boolean"):
        verify_g3_signed_attestation_set(
            package,
            envelopes,
            trusted,
            policy,
            require_complete_qualification_attestations=1,
        )


def test_g3_signed_attestation_set_rejects_missing_package_record(tmp_path: Path) -> None:
    package, envelopes, trusted, policy = _make_signed_set(tmp_path)

    with pytest.raises(ValueError, match="does not cover package records"):
        verify_g3_signed_attestation_set(package, envelopes[:-1], trusted, policy)


def test_g3_signed_attestation_set_rejects_duplicate_envelope_path(tmp_path: Path) -> None:
    package, envelopes, trusted, policy = _make_signed_set(tmp_path)

    with pytest.raises(ValueError, match="envelope paths must be unique"):
        verify_g3_signed_attestation_set(package, [*envelopes, envelopes[0]], trusted, policy)
