"""Provenance identity: experiment sources must be committed, and digests must be exact."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from artifacts import provenance
from artifacts.manifest import mark_failed


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "test@example.invalid")
    _git(tmp_path, "config", "user.name", "Test")
    source = tmp_path / "experiment.py"
    source.write_text("print('committed')\n", encoding="utf-8")
    _git(tmp_path, "add", "experiment.py")
    _git(tmp_path, "commit", "-q", "-m", "fixture")
    return tmp_path


def test_committed_source_matches_head(repo: Path) -> None:
    provenance.require_file_matches_git_head(repo / "experiment.py")


def test_source_outside_a_checkout_is_refused(tmp_path: Path) -> None:
    outside = tmp_path / "loose.py"
    outside.write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Git checkout is required"):
        provenance.require_file_matches_git_head(outside)


def test_untracked_source_is_refused(repo: Path) -> None:
    loose = repo / "untracked.py"
    loose.write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="tracked in the current Git commit"):
        provenance.require_file_matches_git_head(loose)


def test_modified_source_must_be_committed(repo: Path) -> None:
    (repo / "experiment.py").write_text("print('changed')\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="differs from Git HEAD"):
        provenance.require_file_matches_git_head(repo / "experiment.py")


def test_source_outside_the_repository_root_is_refused(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = repo.parent / "elsewhere.py"
    other.write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr(
        provenance.subprocess,
        "check_output",
        lambda *args, **kwargs: str(repo.parent / "nowhere").encode(),
    )
    with pytest.raises(RuntimeError, match="must be inside the Git checkout"):
        provenance.require_file_matches_git_head(other)


def test_package_digest_is_stable_and_canonical_hash_is_key_order_free() -> None:
    digest = provenance.package_source_sha256()
    assert len(digest) == 64
    assert provenance.package_source_sha256() == digest
    assert provenance.canonical_hash({"b": 1, "a": 2}) == provenance.canonical_hash(
        {"a": 2, "b": 1}
    )


def test_versions_report_unavailable_git_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    def failing(*args, **kwargs):
        raise subprocess.CalledProcessError(128, "git")

    monkeypatch.setattr(provenance.subprocess, "check_output", failing)
    assert provenance.versions()["git_sha"] == "unavailable"


def test_failed_manifest_records_its_stage_and_replay_state() -> None:
    manifest = {"state": "running", "run_id": "r1"}
    failed = mark_failed(manifest, error_type="ValueError", error_message="boom", stage="train")
    assert failed["state"] == "failed"
    assert failed["failure"] == {"error_type": "ValueError", "message": "boom", "stage": "train"}
    assert failed["failed_utc"]
    assert "run_replay" not in failed
