"""Runtime compatibility: missing distributions, import failures, missing symbols and probe flag handling."""

from __future__ import annotations

import types

import pytest
from integrations import runtime_compatibility as compat
from integrations.runtime_compatibility import build_runtime_compatibility


def _missing_version(name: str) -> str:
    raise compat.metadata.PackageNotFoundError(name)


def test_absent_distributions_are_reported_as_dependency_missing() -> None:
    report = build_runtime_compatibility(version_provider=_missing_version)
    assert report["status"] == "PHY_INCOMPATIBLE"
    assert report["phy"]["status"] == "DEPENDENCY_MISSING"
    assert all(version is None for version in report["phy"]["distributions"].values())


def test_missing_optional_rt_distribution_is_reported_without_importing_rt():
    def version_provider(name: str) -> str:
        if name == "mitsuba":
            raise compat.metadata.PackageNotFoundError(name)
        return {"sionna-no-rt": "2.2.0", "sionna": "2.2.0", "sionna-rt": "2.2.0", "drjit": "1.5.0"}[
            name
        ]

    def importer(name: str):
        if name.startswith("sionna.phy"):
            return types.SimpleNamespace(**{symbol: object() for symbol in compat._PHY_APIS[name]})
        pytest.fail(f"RT API {name} must not be imported when an RT distribution is absent")

    report = build_runtime_compatibility(
        probe_rt=True,
        importer=importer,
        version_provider=version_provider,
    )

    assert report["rt"]["status"] == "DEPENDENCY_MISSING"
    assert report["rt"]["distributions"]["mitsuba"] is None
    assert report["rt"]["modules"] == []


def test_probe_flag_must_be_a_real_boolean() -> None:
    with pytest.raises(TypeError, match="probe_rt must be a Boolean"):
        build_runtime_compatibility(probe_rt=1)


def test_import_failures_are_recorded_per_module_without_raising() -> None:
    def failing_importer(name: str):
        raise ImportError(f"no {name}")

    versions = {name: "2.2.0" for name, _ in compat._DISTRIBUTION_REQUIREMENTS["phy"]}
    versions = dict(versions)
    report = build_runtime_compatibility(
        importer=failing_importer,
        version_provider=lambda name: versions.get(name, "0"),
    )
    phy_modules = report["phy"]["modules"]
    assert phy_modules and all(row["status"] == "IMPORT_ERROR" for row in phy_modules)
    assert all(row["error"].startswith("ImportError:") for row in phy_modules)
    assert report["status"] == "PHY_INCOMPATIBLE"


def test_missing_symbols_are_listed_and_block_compatibility() -> None:
    versions = {name: expected for name, expected in compat._DISTRIBUTION_REQUIREMENTS["phy"]}

    def module_without_symbols(name: str):
        return types.ModuleType(name)

    report = build_runtime_compatibility(
        importer=module_without_symbols,
        version_provider=lambda name: versions.get(name, "0"),
    )
    statuses = {row["status"] for row in report["phy"]["modules"]}
    assert statuses == {"MISSING_SYMBOLS"}
    assert report["phy"]["status"] == "INCOMPATIBLE"
