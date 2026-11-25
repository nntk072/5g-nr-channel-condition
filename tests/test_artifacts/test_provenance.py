"""Runtime package identity tests for split Sionna PHY/RT distributions."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from pathlib import Path

from artifacts import provenance


def test_versions_records_official_sionna_no_rt_as_phy(monkeypatch) -> None:
    def fake_version(distribution: str) -> str:
        if distribution == "sionna-no-rt":
            return "2.2.0"
        raise PackageNotFoundError(distribution)

    monkeypatch.setattr(provenance.importlib.metadata, "version", fake_version)
    monkeypatch.setattr(provenance.subprocess, "check_output", lambda *args, **kwargs: b"abc123")

    identities = provenance.versions()

    assert identities["sionna-phy"] == "2.2.0"
    assert identities["sionna-rt"] == "2.2.0"


def test_package_source_hash_skips_missing_packages_and_pycache(
    monkeypatch, tmp_path: Path
) -> None:
    artifacts = tmp_path / "artifacts"
    datasets = tmp_path / "datasets"
    cache = datasets / "__pycache__"
    artifacts.mkdir()
    cache.mkdir(parents=True)
    (artifacts / "provenance.py").write_bytes(b"module")
    (datasets / "source.py").write_bytes(b"source")
    cached = cache / "ignored.py"
    cached.write_bytes(b"ignored")
    monkeypatch.setattr(provenance, "__file__", str(artifacts / "provenance.py"))

    digest = provenance.package_source_sha256()
    cached.write_bytes(b"changed ignored cache")

    assert len(digest) == 64
    assert provenance.package_source_sha256() == digest
    (datasets / "new_source.py").write_bytes(b"new source")
    assert provenance.package_source_sha256() != digest


def test_package_source_hash_handles_missing_package_directories(
    monkeypatch, tmp_path: Path
) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "provenance.py").write_bytes(b"module")
    monkeypatch.setattr(provenance, "__file__", str(artifacts / "provenance.py"))

    assert len(provenance.package_source_sha256()) == 64


def test_package_source_hash_skips_pycache_only_package(monkeypatch, tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    cache = tmp_path / "datasets" / "__pycache__"
    artifacts.mkdir()
    cache.mkdir(parents=True)
    (cache / "ignored.py").write_bytes(b"ignored")
    monkeypatch.setattr(provenance, "__file__", str(artifacts / "provenance.py"))

    assert len(provenance.package_source_sha256()) == 64


def test_versions_prefers_combined_sionna_rt_distribution(monkeypatch) -> None:
    calls: list[str] = []

    def fake_version(distribution: str) -> str:
        calls.append(distribution)
        if distribution == "sionna-rt":
            return "2.2.1"
        if distribution == "sionna-no-rt":
            return "2.2.0"
        raise PackageNotFoundError(distribution)

    monkeypatch.setattr(provenance.importlib.metadata, "version", fake_version)
    monkeypatch.setattr(provenance.subprocess, "check_output", lambda *args, **kwargs: b"abc123")

    identities = provenance.versions()

    assert identities["sionna-rt"] == "2.2.1"
    assert identities["sionna-phy"] == "2.2.0"
    assert calls.index("sionna-rt") < calls.index("sionna-no-rt")
