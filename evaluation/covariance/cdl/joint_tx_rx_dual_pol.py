"""Independent ETSI CDL joint TX/RX dual-pol space-time covariance reference."""

from __future__ import annotations

import cmath
import math
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

from evaluation.covariance.cdl.joint_tx_rx import (
    cdl_joint_tx_rx_spatiotemporal_covariance_reference,
)
from evaluation.covariance.cdl.shared import (
    _diffuse_directions,
    _finite_real,
    _reference_document,
)

_C0_MPS = 299_792_458.0
_ROOT = Path(__file__).resolve().parents[3]


@lru_cache(maxsize=1)
def _horizontal_diffuse_block(
    model: str,
    velocity_mps: tuple[float, float, float],
    *,
    carrier_frequency_hz: float,
    lag_s: float,
    tx_count: int,
    rx_element_count: int,
    element_spacing_wavelengths: float,
) -> tuple[tuple[complex, ...], ...]:
    """Return the H-receive diffuse block for a single-V transmit array."""
    document = _reference_document()
    profile = next(row for row in document["profiles"] if row["profile_model"] == model)
    components = profile["components"]
    raw_powers = [10.0 ** (float(row["power_db"]) / 10.0) for row in components]
    power_sum = math.fsum(raw_powers)
    xpr_linear = 10.0 ** (float(profile["xpr_db"]) / 10.0)
    asa = float(profile["cluster_spreads_deg"]["asa"])
    zsa = float(profile["cluster_spreads_deg"]["zsa"])
    asd = float(profile["cluster_spreads_deg"]["asd"])
    zsd = float(profile["cluster_spreads_deg"]["zsd"])
    temporal_scale = 2.0 * math.pi * carrier_frequency_hz * lag_s / _C0_MPS
    spatial_scale = 2.0 * math.pi * element_spacing_wavelengths
    port_count = tx_count * rx_element_count
    covariance = [[0.0j for _ in range(port_count)] for _ in range(port_count)]

    for component, raw_power in zip(components, raw_powers, strict=True):
        if component["component"] == "specular":
            continue
        if component["component"] != "diffuse":
            raise ValueError(f"unsupported CDL component kind: {component['component']}")
        power = raw_power / power_sum / xpr_linear
        angles = component["angles_deg"]
        receive_directions = _diffuse_directions(
            float(angles["aoa"]), float(angles["zoa"]), asa, zsa
        )
        transmit_directions = _diffuse_directions(
            float(angles["aod"]), float(angles["zod"]), asd, zsd
        )

        receive_means: dict[int, complex] = {}
        for delta_rx in range(-(rx_element_count - 1), rx_element_count):
            phase_sum = 0.0j
            for direction in receive_directions:
                doppler_projection = sum(
                    value * unit for value, unit in zip(velocity_mps, direction, strict=True)
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

        for rx in range(rx_element_count):
            for tx in range(tx_count):
                row = tx_count * rx + tx
                for rx_prime in range(rx_element_count):
                    for tx_prime in range(tx_count):
                        column = tx_count * rx_prime + tx_prime
                        covariance[row][column] += (
                            power * receive_means[rx_prime - rx] * transmit_means[tx_prime - tx]
                        )

    return tuple(tuple(value for value in row) for row in covariance)


def cdl_joint_tx_rx_dual_pol_spatiotemporal_covariance_reference(
    model: str,
    velocity_mps: Sequence[float],
    *,
    carrier_frequency_hz: float = 3.5e9,
    lag_s: float = 0.7e-3,
    tx_count: int = 2,
    rx_element_count: int = 2,
    element_spacing_wavelengths: float = 0.5,
) -> tuple[tuple[complex, ...], ...]:
    """Return the full dual-VH RX / single-V TX cross-time covariance matrix.

    Channel ports are flattened in receive-port-major/transmit-element-minor
    order. Receive ports follow Sionna PanelArray's frozen dual-VH ordering:
    all V element ports first, followed by all H element ports.
    """
    if model not in set("ABCDE"):
        raise ValueError(f"unsupported CDL profile: {model}")
    if len(velocity_mps) != 3:
        raise ValueError("velocity_mps must contain exactly three Cartesian components")
    velocity = (
        _finite_real(velocity_mps[0], label="velocity_mps component"),
        _finite_real(velocity_mps[1], label="velocity_mps component"),
        _finite_real(velocity_mps[2], label="velocity_mps component"),
    )
    carrier = _finite_real(carrier_frequency_hz, label="carrier_frequency_hz")
    lag = _finite_real(lag_s, label="lag_s")
    spacing = _finite_real(element_spacing_wavelengths, label="element_spacing_wavelengths")
    for name, count in (("tx_count", tx_count), ("rx_element_count", rx_element_count)):
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(f"{name} must be a positive integer")
    if carrier <= 0.0 or lag < 0.0 or spacing < 0.0:
        raise ValueError("carrier must be positive and lag/spacing must be nonnegative")

    vertical = cdl_joint_tx_rx_spatiotemporal_covariance_reference(
        model,
        velocity,
        carrier_frequency_hz=carrier,
        lag_s=lag,
        tx_count=tx_count,
        rx_count=rx_element_count,
        element_spacing_wavelengths=spacing,
    )
    horizontal = _horizontal_diffuse_block(
        model,
        velocity,
        carrier_frequency_hz=carrier,
        lag_s=lag,
        tx_count=tx_count,
        rx_element_count=rx_element_count,
        element_spacing_wavelengths=spacing,
    )
    block_size = tx_count * rx_element_count
    port_count = 2 * block_size
    covariance = [[0.0j for _ in range(port_count)] for _ in range(port_count)]
    for row in range(block_size):
        for column in range(block_size):
            covariance[row][column] = vertical[row][column]
            covariance[block_size + row][block_size + column] = horizontal[row][column]
    return tuple(tuple(value for value in row) for row in covariance)
