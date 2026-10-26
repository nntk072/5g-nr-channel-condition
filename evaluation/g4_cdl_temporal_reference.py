"""Independent G4 CDL temporal references derived from pinned ETSI rows."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

_REFERENCE_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs/research/G4_CDL_TAP_ANGLE_REFERENCE_2026-10-26.json"
)
_REFERENCE_SHA256 = "24ee2f8f73f26e27522a296a4a0860bdaa7f284e9138855666e09636bc97b090"
_SOURCE_SHA256 = "b71460fb4978a59fbffc864652016cf161c9e17cce19f0d0e9b46bea630b1cc4"
_RAY_ANGLE_OFFSETS = (
    0.0447,
    -0.0447,
    0.1413,
    -0.1413,
    0.2492,
    -0.2492,
    0.3715,
    -0.3715,
    0.5129,
    -0.5129,
    0.6797,
    -0.6797,
    0.8844,
    -0.8844,
    1.1481,
    -1.1481,
    1.5195,
    -1.5195,
    2.1551,
    -2.1551,
)


def _finite_real(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite real number")
    return result


def _load_pinned_cdl_profile(model: str) -> dict[str, Any]:
    if model not in {"A", "B", "C"}:
        raise ValueError("independent NLOS temporal reference supports CDL-A/B/C only")
    payload = _REFERENCE_PATH.read_bytes()
    if hashlib.sha256(payload).hexdigest() != _REFERENCE_SHA256:
        raise ValueError("CDL temporal reference bytes do not match the frozen digest")
    try:
        reference = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("CDL temporal reference must be valid UTF-8 JSON") from error
    if (
        not isinstance(reference, dict)
        or reference.get("schema") != "loslab.g4_cdl_tap_angle_reference.v1"
        or reference.get("source_id") != "tr-138901-v19.2.0"
        or reference.get("source_sha256") != _SOURCE_SHA256
    ):
        raise ValueError("CDL temporal reference is not bound to ETSI TR 38.901 V19.2.0")
    profiles = reference.get("profiles")
    if not isinstance(profiles, list):
        raise ValueError("CDL temporal reference must contain a profile list")
    matches = [row for row in profiles if isinstance(row, dict) and row.get("profile_model") == model]
    if len(matches) != 1:
        raise ValueError(f"CDL temporal reference must contain one CDL-{model} profile")
    profile = matches[0]
    if profile.get("los") is not False:
        raise ValueError(f"CDL-{model} temporal reference must be NLOS")
    return profile


def cdl_nlos_temporal_correlation_reference(
    model: str,
    *,
    carrier_frequency_hz: float,
    speed_mps: float,
    lag_s: float,
) -> complex:
    """Return the ETSI-row-weighted NLOS narrowband covariance at ``lag_s``.

    The expected value uses the pinned CDL cluster powers, mean AOA/ZOA, ASA,
    and ZSA. The standardized 20-ray angular offset roster is independently
    applied to azimuth and zenith and averaged over its 400 Cartesian pairs;
    the spherical direction cosine supplies the Doppler phase.
    No channel-adapter metadata enters this calculation.
    """
    carrier = _finite_real(carrier_frequency_hz, label="carrier_frequency_hz")
    speed = _finite_real(speed_mps, label="speed_mps")
    lag = _finite_real(lag_s, label="lag_s")
    if carrier <= 0 or speed < 0 or lag < 0:
        raise ValueError("carrier frequency must be positive; speed and lag must be nonnegative")
    profile = _load_pinned_cdl_profile(model)
    spreads = profile.get("cluster_spreads_deg")
    components = profile.get("components")
    if not isinstance(spreads, dict) or not isinstance(components, list) or not components:
        raise ValueError(f"CDL-{model} temporal reference has malformed clusters or spreads")
    asa = _finite_real(spreads.get("asa"), label="cluster ASA")
    zsa = _finite_real(spreads.get("zsa"), label="cluster ZSA")
    cluster_rows: dict[int, tuple[float, float, float]] = {}
    for component in components:
        if not isinstance(component, dict) or component.get("component") != "diffuse":
            raise ValueError(f"CDL-{model} NLOS reference may contain only diffuse components")
        cluster_id = component.get("cluster_index")
        angles = component.get("angles_deg")
        if isinstance(cluster_id, bool) or not isinstance(cluster_id, int) or cluster_id < 1:
            raise ValueError(f"CDL-{model} reference has an invalid cluster index")
        if not isinstance(angles, dict):
            raise ValueError(f"CDL-{model} reference has malformed cluster angles")
        if cluster_id in cluster_rows:
            raise ValueError(f"CDL-{model} reference duplicates cluster {cluster_id}")
        power_db = _finite_real(component.get("power_db"), label="cluster power")
        aoa_deg = _finite_real(angles.get("aoa"), label="cluster AOA")
        zoa_deg = _finite_real(angles.get("zoa"), label="cluster ZOA")
        power = 10.0 ** (power_db / 10.0)
        if not math.isfinite(power) or power <= 0:
            raise ValueError(f"CDL-{model} reference has invalid linear cluster power")
        cluster_rows[cluster_id] = (power, aoa_deg, zoa_deg)
    expected_count = profile.get("cluster_count")
    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or sorted(cluster_rows) != list(range(1, expected_count + 1))
    ):
        raise ValueError(f"CDL-{model} cluster roster differs from its declared count")
    max_doppler_hz = speed * carrier / 299_792_458.0
    weighted = 0j
    total_power = math.fsum(row[0] for row in cluster_rows.values())
    for power, aoa_deg, zoa_deg in cluster_rows.values():
        ray_sum = 0j
        for aoa_offset in _RAY_ANGLE_OFFSETS:
            aoa = math.radians(aoa_deg + asa * aoa_offset)
            for zoa_offset in _RAY_ANGLE_OFFSETS:
                zoa = math.radians(zoa_deg + zsa * zoa_offset)
                direction_cosine = math.sin(zoa) * math.cos(aoa)
                phase = 2.0 * math.pi * max_doppler_hz * lag * direction_cosine
                ray_sum += complex(math.cos(phase), math.sin(phase))
        weighted += power * (ray_sum / len(_RAY_ANGLE_OFFSETS) ** 2)
    result = weighted / total_power
    if not math.isfinite(result.real) or not math.isfinite(result.imag):
        raise ValueError("CDL temporal reference calculation produced a non-finite value")
    return result
