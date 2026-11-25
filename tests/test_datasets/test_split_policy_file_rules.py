"""Split-policy artifact readers: bounded, single-link, symlink-free and signed-byte checks."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from datasets.split_policy.policy import (
    _read_immutable_lock_bytes,
    verify_openssh_signed_artifact,
)


def test_lock_reader_returns_exact_bytes_of_a_regular_file(tmp_path: Path) -> None:
    target = tmp_path / "lock.json"
    target.write_bytes(b"{}\n")
    assert _read_immutable_lock_bytes(target) == b"{}\n"


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_lock_reader_limit_must_be_a_positive_integer(tmp_path: Path, limit) -> None:
    target = tmp_path / "lock.json"
    target.write_bytes(b"x")
    with pytest.raises(ValueError, match="max_bytes must be a positive integer"):
        _read_immutable_lock_bytes(target, max_bytes=limit)


def test_lock_reader_refuses_files_over_the_limit(tmp_path: Path) -> None:
    target = tmp_path / "lock.json"
    target.write_bytes(b"0123456789")
    with pytest.raises(ValueError, match="exceeds its 4-byte limit"):
        _read_immutable_lock_bytes(target, artifact_label="policy", max_bytes=4)


def test_lock_reader_refuses_symlinks_and_hard_links(tmp_path: Path) -> None:
    target = tmp_path / "lock.json"
    target.write_bytes(b"{}")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="must not be a symlink"):
        _read_immutable_lock_bytes(link)
    hard = tmp_path / "hard.json"
    hard.hardlink_to(target)
    with pytest.raises(ValueError, match="single-link regular file"):
        _read_immutable_lock_bytes(target)


def test_lock_reader_reports_missing_files(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        _read_immutable_lock_bytes(tmp_path / "absent.json")


def _sign(tmp_path: Path, payload: bytes) -> tuple[Path, Path, Path]:
    ssh_keygen = shutil.which("ssh-keygen")
    if ssh_keygen is None:
        pytest.skip("OpenSSH ssh-keygen is required")
    key = tmp_path / "key"
    subprocess.run([ssh_keygen, "-q", "-t", "ed25519", "-N", "", "-f", str(key)], check=True)
    public = subprocess.run(
        [ssh_keygen, "-y", "-f", str(key)], check=True, capture_output=True, text=True
    ).stdout.strip()
    allowed = tmp_path / "allowed_signers"
    allowed.write_text(
        f'signer@example.org namespaces="artifact-test" {public}\n', encoding="utf-8"
    )
    artifact = tmp_path / "artifact.json"
    artifact.write_bytes(payload)
    subprocess.run(
        [ssh_keygen, "-Y", "sign", "-f", str(key), "-n", "artifact-test", str(artifact)],
        check=True,
        capture_output=True,
    )
    return artifact, artifact.with_name("artifact.json.sig"), allowed


def test_signed_artifact_is_verified_against_its_exact_bytes(tmp_path: Path) -> None:
    artifact, signature, allowed = _sign(tmp_path, b'{"a": 1}\n')

    verified = verify_openssh_signed_artifact(
        artifact,
        signature,
        trusted_allowed_signers_path=allowed,
        signer_id="signer@example.org",
        namespace="artifact-test",
    )
    assert verified is not None

    artifact.write_bytes(b'{"a": 2}\n')
    with pytest.raises(ValueError, match="signature is invalid or signer is not trusted"):
        verify_openssh_signed_artifact(
            artifact,
            signature,
            trusted_allowed_signers_path=allowed,
            signer_id="signer@example.org",
            namespace="artifact-test",
        )


@pytest.mark.parametrize(
    ("signer_id", "namespace", "message"),
    [
        pytest.param(
            "bad\nsigner",
            "artifact-test",
            "signer_id must be a single-line value",
            id="newline-signer",
        ),
        pytest.param(
            "signer@example.org",
            "bad\rns",
            "namespace must be a single-line value",
            id="cr-namespace",
        ),
    ],
)
def test_signed_artifact_identity_must_be_single_line(
    tmp_path: Path, signer_id, namespace, message
) -> None:
    artifact, signature, allowed = _sign(tmp_path, b"payload")
    with pytest.raises(ValueError, match=message):
        verify_openssh_signed_artifact(
            artifact,
            signature,
            trusted_allowed_signers_path=allowed,
            signer_id=signer_id,
            namespace=namespace,
        )


def test_signed_artifact_requires_ssh_keygen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact, signature, allowed = _sign(tmp_path, b"payload")
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="OpenSSH ssh-keygen is required"):
        verify_openssh_signed_artifact(
            artifact,
            signature,
            trusted_allowed_signers_path=allowed,
            signer_id="signer@example.org",
            namespace="artifact-test",
        )


def test_signed_artifact_verification_is_time_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact, signature, allowed = _sign(tmp_path, b"payload")

    def expired(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="ssh-keygen", timeout=30)

    monkeypatch.setattr(subprocess, "run", expired)
    with pytest.raises(RuntimeError, match="timed out"):
        verify_openssh_signed_artifact(
            artifact,
            signature,
            trusted_allowed_signers_path=allowed,
            signer_id="signer@example.org",
            namespace="artifact-test",
        )
