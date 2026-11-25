"""Helpers shared by the CDL covariance modules.

The bodies here were identical in every module that defined them. Each family module keeps its
own helper where the body differs. Reference loading checks the frozen digest of the reference file.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[3]
_REFERENCE_PATH = _ROOT / "config/reference/CDL_TAP_ANGLE_REFERENCE_2026-10-26.json"
_REFERENCE_SHA256 = "24ee2f8f73f26e27522a296a4a0860bdaa7f284e9138855666e09636bc97b090"
_ETSI_SOURCE_SHA256 = "b71460fb4978a59fbffc864652016cf161c9e17cce19f0d0e9b46bea630b1cc4"
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


def _reference_document() -> dict[str, Any]:
    payload = _REFERENCE_PATH.read_bytes()
    if hashlib.sha256(payload).hexdigest() != _REFERENCE_SHA256:
        raise ValueError("CDL reference bytes do not match their frozen digest")
    document = json.loads(payload)
    if (
        not isinstance(document, dict)
        or document.get("schema") != "loslab.g4_cdl_tap_angle_reference.v1"
        or document.get("source_id") != "tr-138901-v19.2.0"
        or document.get("source_sha256") != _ETSI_SOURCE_SHA256
    ):
        raise ValueError("CDL reference source contract is invalid")
    return document


def _finite_real(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _unit_direction(azimuth_deg: float, zenith_deg: float) -> tuple[float, float, float]:
    azimuth = math.radians(azimuth_deg)
    zenith = math.radians(zenith_deg)
    sin_zenith = math.sin(zenith)
    return (
        sin_zenith * math.cos(azimuth),
        sin_zenith * math.sin(azimuth),
        math.cos(zenith),
    )


def _diffuse_directions(
    azimuth_deg: float,
    zenith_deg: float,
    azimuth_spread_deg: float,
    zenith_spread_deg: float,
) -> tuple[tuple[float, float, float], ...]:
    return tuple(
        _unit_direction(
            azimuth_deg + azimuth_spread_deg * az_offset,
            zenith_deg + zenith_spread_deg * zen_offset,
        )
        for az_offset in _RAY_ANGLE_OFFSETS
        for zen_offset in _RAY_ANGLE_OFFSETS
    )
