"""Runtime package identity tests for split Sionna PHY/RT distributions."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError

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
