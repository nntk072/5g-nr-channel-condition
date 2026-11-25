"""Independent ETSI-derived CDL joint space-time covariance reference."""

from __future__ import annotations

import cmath
import math
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

from evaluation.covariance.cdl.shared import (
    _RAY_ANGLE_OFFSETS,
    _finite_real,
    _reference_document,
)

_C0_MPS = 299_792_458.0
_ROOT = Path(__file__).resolve().parents[3]


@lru_cache(maxsize=1)
def _unit_arrival_direction(aoa_deg: float, zoa_deg: float) -> tuple[float, float, float]:
    azimuth = math.radians(aoa_deg)
    zenith = math.radians(zoa_deg)
    sin_zenith = math.sin(zenith)
    return (
        sin_zenith * math.cos(azimuth),
        sin_zenith * math.sin(azimuth),
        math.cos(zenith),
    )


def cdl_spatiotemporal_covariance_reference(
    model: str,
    velocity_mps: Sequence[float],
    *,
    carrier_frequency_hz: float = 3.5e9,
    lag_s: float = 0.7e-3,
    element_count: int = 2,
    element_spacing_wavelengths: float = 0.5,
) -> tuple[tuple[complex, ...], ...]:
    """Return E[h_i(0)^* h_j(tau)] for a zero-oriented local-y UT ULA.

    The expectation is formed only from the byte-pinned ETSI CDL component
    rows and the TR 38.901 ray-offset set. Diffuse azimuth and zenith offsets
    are averaged over their Cartesian product, matching their independent
    random coupling. Single-V omnidirectional ports make the co-polar field
    factor unit magnitude, so XPR does not enter this bounded observable.
    """
    if model not in set("ABCDE"):
        raise ValueError(f"unsupported CDL profile: {model}")
    if len(velocity_mps) != 3:
        raise ValueError("velocity_mps must contain exactly three Cartesian components")
    velocity = tuple(_finite_real(value, label="velocity_mps component") for value in velocity_mps)
    carrier = _finite_real(carrier_frequency_hz, label="carrier_frequency_hz")
    lag = _finite_real(lag_s, label="lag_s")
    spacing = _finite_real(element_spacing_wavelengths, label="element_spacing_wavelengths")
    if carrier <= 0.0 or lag < 0.0 or spacing < 0.0:
        raise ValueError("carrier must be positive and lag/spacing must be nonnegative")
    if isinstance(element_count, bool) or not isinstance(element_count, int) or element_count < 1:
        raise ValueError("element_count must be a positive integer")

    document = _reference_document()
    profile = next(row for row in document["profiles"] if row["profile_model"] == model)
    components = profile["components"]
    raw_powers = [10.0 ** (float(row["power_db"]) / 10.0) for row in components]
    power_sum = math.fsum(raw_powers)
    powers = [power / power_sum for power in raw_powers]
    asa = float(profile["cluster_spreads_deg"]["asa"])
    zsa = float(profile["cluster_spreads_deg"]["zsa"])
    temporal_scale = 2.0 * math.pi * carrier * lag / _C0_MPS
    spatial_scale = 2.0 * math.pi * spacing

    covariance = [[0.0j for _ in range(element_count)] for _ in range(element_count)]
    for component, power in zip(components, powers, strict=True):
        base_aoa = float(component["angles_deg"]["aoa"])
        base_zoa = float(component["angles_deg"]["zoa"])
        directions: tuple[tuple[float, float, float], ...]
        if component["component"] == "specular":
            directions = (_unit_arrival_direction(base_aoa, base_zoa),)
        elif component["component"] == "diffuse":
            directions = tuple(
                _unit_arrival_direction(base_aoa + asa * az_offset, base_zoa + zsa * zen_offset)
                for az_offset in _RAY_ANGLE_OFFSETS
                for zen_offset in _RAY_ANGLE_OFFSETS
            )
        else:
            raise ValueError(f"unsupported CDL component kind: {component['component']}")

        for row in range(element_count):
            for column in range(element_count):
                separation = column - row
                phase_sum = 0.0j
                for direction in directions:
                    temporal_projection = sum(
                        v * u for v, u in zip(velocity, direction, strict=True)
                    )
                    phase = (
                        spatial_scale * separation * direction[1]
                        + temporal_scale * temporal_projection
                    )
                    phase_sum += cmath.exp(1j * phase)
                covariance[row][column] += power * phase_sum / len(directions)

    return tuple(tuple(value for value in row) for row in covariance)
