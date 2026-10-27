"""Independent first-tap covariance references for scalable ETSI TDL profiles."""

from __future__ import annotations

import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

_REFERENCE_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs/research/G4_SCALABLE_TDL_POWER_DELAY_REFERENCE_2026-10-26.json"
)
_REFERENCE_SHA256 = "38b897029bd4cc6de49eece5096e69c75291eaf6dac45bebe04767ed24a4f0bd"
_SOURCE_SHA256 = "b71460fb4978a59fbffc864652016cf161c9e17cce19f0d0e9b46bea630b1cc4"
_PROFILE_MODELS = frozenset("ABCDE")
_FIXED_REFERENCE_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs/research/G4_FIXED_TDL_TAP_PROFILE_REFERENCE_2026-10-26.json"
)
_FIXED_REFERENCE_SHA256 = "fe22560a71f4e5683ea6780f73e84466155b26f5f97cc4b63e3250c102fafda3"
_FIXED_SOURCE_SHA256 = "b3c6a73c7a40219e7270c08c7a57c84cb46f6757b303a64ec768ba04a6079e81"
_FIXED_PROFILE_MODELS = frozenset({"A10", "A30", "B100", "C60", "C300", "D10", "D30"})


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate TDL reference key: {key}")
        result[key] = value
    return result


@lru_cache(maxsize=1)
def _load_profiles() -> dict[str, tuple[tuple[float, float, str], ...]]:
    payload = _REFERENCE_PATH.read_bytes()
    if hashlib.sha256(payload).hexdigest() != _REFERENCE_SHA256:
        raise ValueError("scalable-TDL reference bytes differ from the pinned digest")
    value = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_object)
    if (
        not isinstance(value, dict)
        or value.get("schema") != "loslab.g4_scalable_tdl_power_delay_reference.v1"
        or value.get("source_sha256") != _SOURCE_SHA256
        or value.get("standard") != "3GPP TR 38.901"
        or value.get("version") != "19.2.0"
    ):
        raise ValueError("scalable-TDL reference has an unexpected source contract")
    rows = value.get("profiles")
    if not isinstance(rows, list) or len(rows) != len(_PROFILE_MODELS):
        raise ValueError("scalable-TDL reference must cover all five profiles")
    result: dict[str, tuple[tuple[float, float, str], ...]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("scalable-TDL profile row must be an object")
        model = row.get("profile_model")
        components = row.get("components")
        if model not in _PROFILE_MODELS or model in result or not isinstance(components, list):
            raise ValueError("scalable-TDL profile roster is invalid")
        parsed: list[tuple[float, float, str]] = []
        for index, component in enumerate(components, start=1):
            if not isinstance(component, dict) or component.get("component_index") != index:
                raise ValueError(f"scalable-TDL {model} component roster is invalid")
            delay = component.get("normalized_delay")
            power_db = component.get("power_db")
            distribution = component.get("fading_distribution")
            if (
                isinstance(delay, bool)
                or not isinstance(delay, (int, float))
                or not math.isfinite(float(delay))
                or delay < 0
                or isinstance(power_db, bool)
                or not isinstance(power_db, (int, float))
                or not math.isfinite(float(power_db))
                or not isinstance(distribution, str)
                or distribution not in {"Rayleigh", "LOS"}
            ):
                raise ValueError(f"scalable-TDL {model} component fields are invalid")
            parsed.append((float(delay), float(power_db), distribution))
        if not parsed or parsed[0][0] != 0.0:
            raise ValueError(f"scalable-TDL {model} must begin at zero normalized delay")
        los_components = [item for item in parsed if item[2] == "LOS"]
        if (model in {"D", "E"}) != (len(los_components) == 1):
            raise ValueError(f"scalable-TDL {model} LOS component roster is inconsistent")
        if los_components and (
            parsed[0][2] != "LOS"
            or len(parsed) < 2
            or parsed[1][2] != "Rayleigh"
            or parsed[1][0] != parsed[0][0]
        ):
            raise ValueError(f"scalable-TDL {model} LOS component must pair at zero delay")
        if any(item[2] != "Rayleigh" for item in parsed[1:] if item[0] > 0.0):
            raise ValueError(f"scalable-TDL {model} has unsupported nonzero-delay fading")
        result[model] = tuple(parsed)
    if set(result) != _PROFILE_MODELS:
        raise ValueError("scalable-TDL reference is missing a profile")
    return result


def scalable_tdl_first_tap_power_components(model: str) -> tuple[float, float]:
    """Return normalized (diffuse, specular) power in a scalable TDL first tap."""
    if not isinstance(model, str) or model not in _PROFILE_MODELS:
        raise ValueError("model must name a scalable TDL profile A, B, C, D, or E")
    components = _load_profiles()[model]
    linear_powers = [10.0 ** (power_db / 10.0) for _, power_db, _ in components]
    total_power = math.fsum(linear_powers)
    first_tap_diffuse = math.fsum(
        power
        for (delay, _, distribution), power in zip(components, linear_powers, strict=True)
        if delay == 0.0 and distribution == "Rayleigh"
    )
    first_tap_specular = math.fsum(
        power
        for (delay, _, distribution), power in zip(components, linear_powers, strict=True)
        if delay == 0.0 and distribution == "LOS"
    )
    return first_tap_diffuse / total_power, first_tap_specular / total_power


def scalable_tdl_rayleigh_tap_indices(model: str) -> tuple[int, ...]:
    """Map ETSI-labeled diffuse components to native Sionna path axes.

    TDL-D/E merge the coincident zero-delay LOS and diffuse terms into one
    native tap, so that aggregate tap is omitted from Rayleigh-only checks.
    """
    if not isinstance(model, str) or model not in _PROFILE_MODELS:
        raise ValueError("model must name a scalable TDL profile A, B, C, D, or E")
    components = _load_profiles()[model]
    has_los = any(distribution == "LOS" for _, _, distribution in components)
    result = tuple(
        index - (2 if has_los else 1)
        for index, (_, _, distribution) in enumerate(components, start=1)
        if distribution == "Rayleigh" and not (has_los and index - 2 == 0)
    )
    if not result or any(index < 0 for index in result):
        raise ValueError(f"scalable TDL-{model} has no valid diffuse native tap roster")
    return result


@lru_cache(maxsize=1)
def _load_fixed_profiles() -> dict[str, tuple[tuple[int, float, str], ...]]:
    payload = _FIXED_REFERENCE_PATH.read_bytes()
    if hashlib.sha256(payload).hexdigest() != _FIXED_REFERENCE_SHA256:
        raise ValueError("fixed-TDL reference bytes differ from the pinned digest")
    value = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_object)
    if (
        not isinstance(value, dict)
        or value.get("schema") != "loslab.g4_fixed_tdl_tap_profile_reference.v1"
        or value.get("source_sha256") != _FIXED_SOURCE_SHA256
        or value.get("standard") != "3GPP TS 38.101-4"
        or value.get("version") != "19.2.2"
    ):
        raise ValueError("fixed-TDL reference has an unexpected source contract")
    rows = value.get("profiles")
    if not isinstance(rows, list) or len(rows) != len(_FIXED_PROFILE_MODELS):
        raise ValueError("fixed-TDL reference must cover all seven profiles")
    result: dict[str, tuple[tuple[int, float, str], ...]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("fixed-TDL profile row must be an object")
        model = row.get("profile_model")
        components = row.get("components")
        if (
            model not in _FIXED_PROFILE_MODELS
            or model in result
            or not isinstance(components, list)
        ):
            raise ValueError("fixed-TDL profile roster is invalid")
        parsed: list[tuple[int, float, str]] = []
        for index, component in enumerate(components, start=1):
            if not isinstance(component, dict) or component.get("component_index") != index:
                raise ValueError(f"fixed-TDL {model} component roster is invalid")
            tap_index = component.get("tap_index")
            power_db = component.get("power_db")
            distribution = component.get("fading_distribution")
            if (
                isinstance(tap_index, bool)
                or not isinstance(tap_index, int)
                or tap_index <= 0
                or isinstance(power_db, bool)
                or not isinstance(power_db, (int, float))
                or not math.isfinite(float(power_db))
                or distribution not in {"Rayleigh", "LOS", "LOS path"}
            ):
                raise ValueError(f"fixed-TDL {model} component fields are invalid")
            parsed.append((tap_index, float(power_db), distribution))
        result[model] = tuple(parsed)
    if set(result) != _FIXED_PROFILE_MODELS:
        raise ValueError("fixed-TDL reference is missing a profile")
    return result


def fixed_tdl_first_tap_power_components(model: str) -> tuple[float, float]:
    """Return normalized (diffuse, specular) power in a fixed TDL first tap."""
    if not isinstance(model, str) or model not in _FIXED_PROFILE_MODELS:
        raise ValueError("model must name a supported fixed TDL profile")
    components = _load_fixed_profiles()[model]
    linear_powers = [10.0 ** (power_db / 10.0) for _, power_db, _ in components]
    total_power = math.fsum(linear_powers)
    first_tap_diffuse = math.fsum(
        power
        for (tap_index, _, distribution), power in zip(components, linear_powers, strict=True)
        if tap_index == 1 and distribution == "Rayleigh"
    )
    first_tap_specular = math.fsum(
        power
        for (tap_index, _, distribution), power in zip(components, linear_powers, strict=True)
        if tap_index == 1 and distribution in {"LOS", "LOS path"}
    )
    return first_tap_diffuse / total_power, first_tap_specular / total_power


def fixed_tdl_rayleigh_tap_indices(model: str) -> tuple[int, ...]:
    """Return zero-based fixed-profile taps made only from ETSI Rayleigh rows.

    A tap containing both an Annex B LOS component and Rayleigh diffuse power
    is a Rician aggregate in the native channel output and is excluded from
    diffuse-only statistical references.
    """
    if not isinstance(model, str) or model not in _FIXED_PROFILE_MODELS:
        raise ValueError("model must name a supported fixed TDL profile")
    by_tap: dict[int, list[str]] = {}
    for tap_index, _, distribution in _load_fixed_profiles()[model]:
        by_tap.setdefault(tap_index, []).append(distribution)
    return tuple(
        tap_index - 1
        for tap_index, distributions in sorted(by_tap.items())
        if all(distribution == "Rayleigh" for distribution in distributions)
    )


def scalable_tdl_first_tap_covariance_reference(
    model: str, rx_correlation: Any, num_tx_ant: int
) -> np.ndarray:
    """Derive first-tap RX×TX covariance from ETSI rows and the declared RX law.

    Diffuse coefficients follow ``P_diff * (R_rx kron I_tx)``. The Sionna
    adapter's correlated-LOS operation applies the receive Cholesky factor to a
    common specular coefficient, giving ``P_los * ((L 1)(L 1)^H kron 11^H)``.
    Profile powers come only from the pinned TR 38.901 fixture.
    """
    if isinstance(num_tx_ant, bool) or not isinstance(num_tx_ant, int) or num_tx_ant <= 0:
        raise ValueError("num_tx_ant must be a positive integer")
    diffuse_power, specular_power = scalable_tdl_first_tap_power_components(model)
    correlation = np.asarray(rx_correlation, dtype=np.complex128)
    if (
        correlation.ndim != 2
        or correlation.shape[0] != correlation.shape[1]
        or correlation.shape[0] == 0
        or not np.isfinite(correlation).all()
        or not np.allclose(correlation, correlation.conj().T, rtol=0.0, atol=1e-12)
    ):
        raise ValueError("rx_correlation must be a finite square Hermitian matrix")
    if not np.allclose(np.diag(correlation), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("rx_correlation must have unit diagonal")
    try:
        receive_cholesky = np.linalg.cholesky(correlation)
    except np.linalg.LinAlgError as error:
        raise ValueError("rx_correlation must be positive definite") from error

    rx_count = correlation.shape[0]
    covariance = diffuse_power * np.kron(correlation, np.eye(num_tx_ant))
    if specular_power:
        los_vector = receive_cholesky @ np.ones(rx_count, dtype=np.complex128)
        los_covariance = np.outer(los_vector, los_vector.conj())
        covariance += specular_power * np.kron(
            los_covariance, np.ones((num_tx_ant, num_tx_ant), dtype=np.complex128)
        )
    return np.asarray(covariance, dtype=np.complex128)
