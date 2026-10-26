"""OpenSSH signature checks for prospective split-policy locks."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from datasets import verify_signed_split_policy_registry
from datasets.split_policy import (
    SplitParentType,
    SplitPolicy,
    SplitSubject,
    freeze_split_policy_registry,
    make_split_policy_registry,
)

pytestmark = pytest.mark.skipif(shutil.which("ssh-keygen") is None, reason="OpenSSH is unavailable")

_SIGNER = "umcd-split-policy-test"
_NAMESPACE = "umcd-split-policy-v2@github.com"


@pytest.fixture
def signed_registry_files(tmp_path: Path) -> tuple[Path, Path, Path, dict]:
    private_key = tmp_path / "test_signing_key"
    subprocess.run(
        [
            shutil.which("ssh-keygen") or "ssh-keygen",
            "-q",
            "-t",
            "ed25519",
            "-N",
            "",
            "-C",
            _SIGNER,
            "-f",
            str(private_key),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    public_key = private_key.with_suffix(".pub").read_text(encoding="utf-8").strip()
    allowed_signers = tmp_path / "trusted_allowed_signers"
    allowed_signers.write_text(
        f'{_SIGNER} namespaces="{_NAMESPACE}" {public_key}\n', encoding="utf-8"
    )

    subjects = [
        SplitSubject(
            subject_id=f"parent-{index:02d}",
            parent_type=SplitParentType.NETWORK_EPISODE,
            source_family="synthetic-test",
        )
        for index in range(20)
    ]
    registry = make_split_policy_registry(
        subjects, policy=SplitPolicy.release_v2(seed="signed-lock-test")
    )
    lock = tmp_path / "split-registry.json"
    freeze_split_policy_registry(lock, registry)
    subprocess.run(
        [
            shutil.which("ssh-keygen") or "ssh-keygen",
            "-Y",
            "sign",
            "-n",
            _NAMESPACE,
            "-f",
            str(private_key),
            str(lock),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    signature = Path(f"{lock}.sig")
    return lock, signature, allowed_signers, registry.to_dict()


def test_signed_split_policy_lock_verifies_against_external_trust_keyring(signed_registry_files):
    lock, signature, allowed_signers, expected = signed_registry_files

    observed = verify_signed_split_policy_registry(
        lock,
        signature,
        trusted_allowed_signers_path=allowed_signers,
        signer_id=_SIGNER,
    )

    assert observed.to_dict() == expected


def test_signed_split_policy_lock_rejects_modified_lock(signed_registry_files):
    lock, signature, allowed_signers, _ = signed_registry_files
    lock.write_bytes(lock.read_bytes() + b" ")

    with pytest.raises(ValueError, match="signature is invalid or signer is not trusted"):
        verify_signed_split_policy_registry(
            lock,
            signature,
            trusted_allowed_signers_path=allowed_signers,
            signer_id=_SIGNER,
        )


def test_signed_split_policy_lock_rejects_wrong_principal(signed_registry_files):
    lock, signature, allowed_signers, _ = signed_registry_files

    with pytest.raises(ValueError, match="signature is invalid or signer is not trusted"):
        verify_signed_split_policy_registry(
            lock,
            signature,
            trusted_allowed_signers_path=allowed_signers,
            signer_id="untrusted-signer",
        )


def test_signed_split_policy_lock_rejects_signature_from_another_namespace(
    signed_registry_files, tmp_path
):
    lock, signature, allowed_signers, _ = signed_registry_files
    signature.unlink()
    subprocess.run(
        [
            shutil.which("ssh-keygen") or "ssh-keygen",
            "-Y",
            "sign",
            "-n",
            "unrelated-artifact-v1@github.com",
            "-f",
            str(tmp_path / "test_signing_key"),
            str(lock),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    with pytest.raises(ValueError, match="signature is invalid or signer is not trusted"):
        verify_signed_split_policy_registry(
            lock,
            signature,
            trusted_allowed_signers_path=allowed_signers,
            signer_id=_SIGNER,
        )


def test_signed_split_policy_lock_rejects_hardlinked_signature(signed_registry_files, tmp_path):
    lock, signature, allowed_signers, _ = signed_registry_files
    signature_alias = tmp_path / "signature-hardlink.sig"
    signature_alias.hardlink_to(signature)

    with pytest.raises(ValueError, match="single-link regular file"):
        verify_signed_split_policy_registry(
            lock,
            signature_alias,
            trusted_allowed_signers_path=allowed_signers,
            signer_id=_SIGNER,
        )
