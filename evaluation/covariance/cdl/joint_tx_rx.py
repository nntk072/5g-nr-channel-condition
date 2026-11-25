"""Independent ETSI CDL joint TX/RX space-time covariance reference."""

from __future__ import annotations

import cmath
import math
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

from evaluation.covariance.cdl.shared import (
    _diffuse_directions,
    _finite_real,
    _reference_document,
    _unit_direction,
)

_C0_MPS = 299_792_458.0
_ROOT = Path(__file__).resolve().parents[3]


@lru_cache(maxsize=1)
def cdl_joint_tx_rx_spatiotemporal_covariance_reference(
    model: str,
    velocity_mps: Sequence[float],
    *,
    carrier_frequency_hz: float = 3.5e9,
    lag_s: float = 0.7e-3,
    tx_count: int = 2,
    rx_count: int = 2,
    element_spacing_wavelengths: float = 0.5,
) -> tuple[tuple[complex, ...], ...]:
    """Return E[h_p(0)^* h_q(tau)] for zero-oriented local-y ULAs.

    Ports are flattened in RX-major/TX-minor order. Diffuse departure and
    arrival ray couplings are independent, so each cluster expectation is the
    product of a receive-plus-Doppler mean and a transmit spatial mean.
    """
    if model not in set("ABCDE"):
        raise ValueError(f"unsupported CDL profile: {model}")
    if len(velocity_mps) != 3:
        raise ValueError("velocity_mps must contain exactly three Cartesian components")
    velocity = tuple(_finite_real(value, label="velocity_mps component") for value in velocity_mps)
    carrier = _finite_real(carrier_frequency_hz, label="carrier_frequency_hz")
    lag = _finite_real(lag_s, label="lag_s")
    spacing = _finite_real(element_spacing_wavelengths, label="element_spacing_wavelengths")
    for name, count in (("tx_count", tx_count), ("rx_count", rx_count)):
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(f"{name} must be a positive integer")
    if carrier <= 0.0 or lag < 0.0 or spacing < 0.0:
        raise ValueError("carrier must be positive and lag/spacing must be nonnegative")

    document = _reference_document()
    profile = next(row for row in document["profiles"] if row["profile_model"] == model)
    components = profile["components"]
    raw_powers = [10.0 ** (float(row["power_db"]) / 10.0) for row in components]
    power_sum = math.fsum(raw_powers)
    powers = [power / power_sum for power in raw_powers]
    asa = float(profile["cluster_spreads_deg"]["asa"])
    zsa = float(profile["cluster_spreads_deg"]["zsa"])
    asd = float(profile["cluster_spreads_deg"]["asd"])
    zsd = float(profile["cluster_spreads_deg"]["zsd"])
    temporal_scale = 2.0 * math.pi * carrier * lag / _C0_MPS
    spatial_scale = 2.0 * math.pi * spacing
    port_count = tx_count * rx_count
    covariance = [[0.0j for _ in range(port_count)] for _ in range(port_count)]

    for component, power in zip(components, powers, strict=True):
        angles = component["angles_deg"]
        receive_directions: tuple[tuple[float, float, float], ...]
        transmit_directions: tuple[tuple[float, float, float], ...]
        if component["component"] == "specular":
            receive_directions = (_unit_direction(float(angles["aoa"]), float(angles["zoa"])),)
            transmit_directions = (_unit_direction(float(angles["aod"]), float(angles["zod"])),)
        elif component["component"] == "diffuse":
            receive_directions = _diffuse_directions(
                float(angles["aoa"]), float(angles["zoa"]), asa, zsa
            )
            transmit_directions = _diffuse_directions(
                float(angles["aod"]), float(angles["zod"]), asd, zsd
            )
        else:
            raise ValueError(f"unsupported CDL component kind: {component['component']}")

        receive_means: dict[int, complex] = {}
        for delta_rx in range(-(rx_count - 1), rx_count):
            phase_sum = 0.0j
            for direction in receive_directions:
                doppler_projection = sum(
                    value * unit for value, unit in zip(velocity, direction, strict=True)
                )
                phase = (
                    spatial_scale * delta_rx * direction[1] + temporal_scale * doppler_projection
                )
                phase_sum += cmath.exp(1j * phase)
            receive_means[delta_rx] = phase_sum / len(receive_directions)

        transmit_means: dict[int, complex] = {}
        for delta_tx in range(-(tx_count - 1), tx_count):
            phase_sum = sum(
                cmath.exp(1j * spatial_scale * delta_tx * direction[1])
                for direction in transmit_directions
            )
            transmit_means[delta_tx] = phase_sum / len(transmit_directions)

        for rx in range(rx_count):
            for tx in range(tx_count):
                row = tx_count * rx + tx
                for rx_prime in range(rx_count):
                    for tx_prime in range(tx_count):
                        column = tx_count * rx_prime + tx_prime
                        covariance[row][column] += (
                            power * receive_means[rx_prime - rx] * transmit_means[tx_prime - tx]
                        )

    return tuple(tuple(value for value in row) for row in covariance)
