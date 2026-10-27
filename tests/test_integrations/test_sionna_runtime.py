"""Version identity checks for split Sionna PHY/RT installations."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError

import pytest
from integrations import sionna_runtime


def test_sionna_phy_version_prefers_no_rt_distribution(monkeypatch) -> None:
    calls: list[str] = []

    def fake_version(distribution: str) -> str:
        calls.append(distribution)
        if distribution == "sionna-no-rt":
            return "2.2.0"
        raise PackageNotFoundError(distribution)

    monkeypatch.setattr(sionna_runtime, "version", fake_version)

    assert sionna_runtime.sionna_phy_version() == "2.2.0"
    assert calls == ["sionna-no-rt"]


def test_sionna_phy_version_falls_back_to_combined_distribution(monkeypatch) -> None:
    def fake_version(distribution: str) -> str:
        if distribution == "sionna-no-rt":
            raise PackageNotFoundError(distribution)
        if distribution == "sionna":
            return "2.2.0"
        raise AssertionError(distribution)

    monkeypatch.setattr(sionna_runtime, "version", fake_version)

    assert sionna_runtime.sionna_phy_version() == "2.2.0"


def test_sionna_phy_version_requires_a_supported_distribution(monkeypatch) -> None:
    monkeypatch.setattr(
        sionna_runtime,
        "version",
        lambda distribution: (_ for _ in ()).throw(PackageNotFoundError(distribution)),
    )

    with pytest.raises(PackageNotFoundError, match="Sionna PHY"):
        sionna_runtime.sionna_phy_version()
