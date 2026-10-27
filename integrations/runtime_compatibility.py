"""Opt-in runtime checks for the Sionna PHY and RT APIs used by this project."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from importlib import metadata
from typing import Any

_PHY_APIS = {
    "sionna.phy": ("config",),
    "sionna.phy.channel": ("OFDMChannel", "TimeChannel", "time_lag_discrete_time_channel"),
    "sionna.phy.channel.tr38901": (
        "CDL",
        "InF",
        "InH",
        "PanelArray",
        "RMa",
        "TDL",
        "UMa",
        "UMi",
        "models",
    ),
    "sionna.phy.fec.polar": ("Polar5GDecoder", "Polar5GEncoder"),
    "sionna.phy.nr": (
        "PUSCHConfig",
        "PUSCHLSChannelEstimator",
        "PUSCHReceiver",
        "PUSCHTransmitter",
    ),
    "sionna.phy.ofdm": ("OFDMDemodulator", "OFDMModulator"),
}
_RT_APIS = {
    "sionna.rt": ("PathSolver", "PlanarArray", "Receiver", "Transmitter", "load_scene"),
    "sionna.rt.constants": ("InteractionType",),
    "sionna.rt.path_solvers.paths": ("Paths",),
    "sionna.rt.path_solvers.paths_buffer": ("PathsBuffer",),
    "sionna.rt.utils": ("spawn_ray_to",),
    "mitsuba": (
        "Bool",
        "Float",
        "Point3f",
        "RayFlags",
        "TensorXb",
        "UInt",
        "Vector3f",
        "set_variant",
        "variants",
    ),
    "drjit": (
        "arange",
        "compress",
        "eval",
        "full",
        "gather",
        "schedule",
        "scatter",
        "select",
        "shape",
        "width",
        "zeros",
    ),
}
_DISTRIBUTION_REQUIREMENTS = {
    "phy": (("sionna-no-rt", "2.2.0"), ("sionna", "2.2.0")),
    "rt": (
        ("sionna-rt", "2.2.0"),
        ("mitsuba", "3.9.1"),
        ("drjit", "1.5.0"),
    ),
}


def _installed_version(name: str, version_provider: Callable[[str], str]) -> str | None:
    try:
        return version_provider(name)
    except metadata.PackageNotFoundError:
        return None


def _group_report(
    group: str,
    apis: dict[str, tuple[str, ...]],
    *,
    importer: Callable[[str], Any],
    version_provider: Callable[[str], str],
    distribution_requirements: tuple[tuple[str, str], ...],
) -> dict[str, Any]:
    versions = {
        name: _installed_version(name, version_provider)
        for name, _expected in distribution_requirements
    }
    if group == "phy":
        selected = next(
            (name for name in ("sionna-no-rt", "sionna") if versions.get(name) is not None),
            None,
        )
        if selected is None:
            return {
                "status": "DEPENDENCY_MISSING",
                "distributions": versions,
                "modules": [],
            }
        distribution_ok = versions[selected] == "2.2.0"
    else:
        missing = [name for name, version in versions.items() if version is None]
        distribution_ok = not missing and all(
            versions[name] == expected for name, expected in distribution_requirements
        )
        if missing:
            return {
                "status": "DEPENDENCY_MISSING",
                "distributions": versions,
                "modules": [],
            }

    modules: list[dict[str, Any]] = []
    for module_name, required_symbols in apis.items():
        try:
            module = importer(module_name)
        except Exception as error:
            modules.append(
                {
                    "module": module_name,
                    "status": "IMPORT_ERROR",
                    "required_symbols": list(required_symbols),
                    "missing_symbols": list(required_symbols),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            continue
        missing_symbols = [symbol for symbol in required_symbols if not hasattr(module, symbol)]
        modules.append(
            {
                "module": module_name,
                "status": "MISSING_SYMBOLS" if missing_symbols else "AVAILABLE",
                "required_symbols": list(required_symbols),
                "missing_symbols": missing_symbols,
                "error": None,
            }
        )
    api_ok = all(module["status"] == "AVAILABLE" for module in modules)
    return {
        "status": "COMPATIBLE" if distribution_ok and api_ok else "INCOMPATIBLE",
        "distributions": versions,
        "expected_distributions": dict(distribution_requirements),
        "modules": modules,
    }


def build_runtime_compatibility(
    *,
    probe_rt: bool = False,
    importer: Callable[[str], Any] | None = None,
    version_provider: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Check imported APIs against the project-pinned Sionna runtime contract.

    The PHY check is always performed when this function is called. RT module
    imports occur only with ``probe_rt=True``; no propagation scene or solver
    is constructed by either check.
    """
    if type(probe_rt) is not bool:
        raise TypeError("probe_rt must be a Boolean")
    load_module = importer or importlib.import_module
    get_version = version_provider or metadata.version
    phy = _group_report(
        "phy",
        _PHY_APIS,
        importer=load_module,
        version_provider=get_version,
        distribution_requirements=_DISTRIBUTION_REQUIREMENTS["phy"],
    )
    if probe_rt:
        rt = _group_report(
            "rt",
            _RT_APIS,
            importer=load_module,
            version_provider=get_version,
            distribution_requirements=_DISTRIBUTION_REQUIREMENTS["rt"],
        )
    else:
        rt = {"status": "NOT_PROBED", "distributions": {}, "modules": []}
    if not probe_rt:
        status = "PHY_COMPATIBLE" if phy["status"] == "COMPATIBLE" else "PHY_INCOMPATIBLE"
    else:
        status = "COMPATIBLE" if phy["status"] == rt["status"] == "COMPATIBLE" else "INCOMPATIBLE"
    return {"schema": "umcd.runtime_compatibility.v1", "status": status, "phy": phy, "rt": rt}
