"""Independent ETSI CDL dual-TX/dual-RX joint space-time covariance reference."""

from __future__ import annotations

import cmath
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

_C0_MPS = 299_792_458.0
_ROOT = Path(__file__).resolve().parents[1]
_REFERENCE_PATH = _ROOT / "docs/research/G4_CDL_TAP_ANGLE_REFERENCE_2026-10-26.json"
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


@lru_cache(maxsize=1)
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


def _decode_dual_vh_port(port: int, element_count: int) -> tuple[int, int]:
    """Return (polarization, physical element) for one-panel Sionna dual-VH order."""
    if port < element_count:
        return 0, port
    return 1, port - element_count


def _diffuse_polarization_covariance(
    row_pair: tuple[int, int],
    column_pair: tuple[int, int],
    *,
    xpr_linear: float,
) -> float:
    """Return E[J_row* J_column] for the four independent Step-10 phases."""
    if row_pair != column_pair:
        return 0.0
    rx_pol, tx_pol = row_pair
    return 1.0 if rx_pol == tx_pol else 1.0 / xpr_linear


def _specular_polarization_covariance(
    row_pair: tuple[int, int], column_pair: tuple[int, int]
) -> float:
    """Return J_row* J_column for the zero-orientation LOS diag(1,-1) matrix."""
    coefficients = {(0, 0): 1.0, (1, 1): -1.0}
    return coefficients.get(row_pair, 0.0) * coefficients.get(column_pair, 0.0)


def cdl_joint_tx_rx_dual_tx_rx_spatiotemporal_covariance_reference(
    model: str,
    velocity_mps: Sequence[float],
    *,
    carrier_frequency_hz: float = 3.5e9,
    lag_s: float = 0.7e-3,
    tx_element_count: int = 2,
    rx_element_count: int = 2,
    element_spacing_wavelengths: float = 0.5,
) -> tuple[tuple[complex, ...], ...]:
    """Return E[h_p(0)^* h_q(tau)] for dual-VH TX and RX arrays.

    The reference is deliberately restricted to one zero-oriented omni panel at
    each endpoint. Native antenna ports are all V physical elements followed by
    all H physical elements. Channel ports are flattened RX-port-major and
    TX-port-minor.

    For diffuse rays, the four TR 38.901 Jones-matrix phases are independent.
    Therefore covariance survives only between channel ports selecting the same
    Jones entry. Co-polar entries have unit expected power and cross-polar
    entries have expected power 1/XPR. For CDL-D/E specular LOS, the frozen
    zero-orientation field matrix is diag(1,-1), so VV and HH also have a
    deterministic signed cross-covariance while VH and HV remain zero.
    """
    if model not in set("ABCDE"):
        raise ValueError(f"unsupported CDL profile: {model}")
    if len(velocity_mps) != 3:
        raise ValueError("velocity_mps must contain exactly three Cartesian components")
    velocity = tuple(_finite_real(value, label="velocity_mps component") for value in velocity_mps)
    carrier = _finite_real(carrier_frequency_hz, label="carrier_frequency_hz")
    lag = _finite_real(lag_s, label="lag_s")
    spacing = _finite_real(element_spacing_wavelengths, label="element_spacing_wavelengths")
    for name, count in (
        ("tx_element_count", tx_element_count),
        ("rx_element_count", rx_element_count),
    ):
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(f"{name} must be a positive integer")
    if carrier <= 0.0 or lag < 0.0 or spacing < 0.0:
        raise ValueError("carrier must be positive and lag/spacing must be nonnegative")

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
    temporal_scale = 2.0 * math.pi * carrier * lag / _C0_MPS
    spatial_scale = 2.0 * math.pi * spacing

    tx_port_count = 2 * tx_element_count
    rx_port_count = 2 * rx_element_count
    channel_port_count = tx_port_count * rx_port_count
    covariance = [[0.0j for _ in range(channel_port_count)] for _ in range(channel_port_count)]

    for component, raw_power in zip(components, raw_powers, strict=True):
        kind = component["component"]
        angles = component["angles_deg"]
        if kind == "specular":
            receive_directions = (_unit_direction(float(angles["aoa"]), float(angles["zoa"])),)
            transmit_directions = (_unit_direction(float(angles["aod"]), float(angles["zod"])),)
        elif kind == "diffuse":
            receive_directions = _diffuse_directions(
                float(angles["aoa"]), float(angles["zoa"]), asa, zsa
            )
            transmit_directions = _diffuse_directions(
                float(angles["aod"]), float(angles["zod"]), asd, zsd
            )
        else:
            raise ValueError(f"unsupported CDL component kind: {kind}")

        receive_means: dict[int, complex] = {}
        for delta_rx in range(-(rx_element_count - 1), rx_element_count):
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
        for delta_tx in range(-(tx_element_count - 1), tx_element_count):
            phase_sum = sum(
                cmath.exp(1j * spatial_scale * delta_tx * direction[1])
                for direction in transmit_directions
            )
            transmit_means[delta_tx] = phase_sum / len(transmit_directions)

        normalized_power = raw_power / power_sum
        for row in range(channel_port_count):
            row_rx_port, row_tx_port = divmod(row, tx_port_count)
            row_rx_pol, row_rx_element = _decode_dual_vh_port(row_rx_port, rx_element_count)
            row_tx_pol, row_tx_element = _decode_dual_vh_port(row_tx_port, tx_element_count)
            row_pair = (row_rx_pol, row_tx_pol)
            for column in range(channel_port_count):
                column_rx_port, column_tx_port = divmod(column, tx_port_count)
                column_rx_pol, column_rx_element = _decode_dual_vh_port(
                    column_rx_port, rx_element_count
                )
                column_tx_pol, column_tx_element = _decode_dual_vh_port(
                    column_tx_port, tx_element_count
                )
                column_pair = (column_rx_pol, column_tx_pol)
                if kind == "diffuse":
                    polarization_factor = _diffuse_polarization_covariance(
                        row_pair, column_pair, xpr_linear=xpr_linear
                    )
                else:
                    polarization_factor = _specular_polarization_covariance(row_pair, column_pair)
                if polarization_factor == 0.0:
                    continue
                covariance[row][column] += (
                    normalized_power
                    * polarization_factor
                    * receive_means[column_rx_element - row_rx_element]
                    * transmit_means[column_tx_element - row_tx_element]
                )

    return tuple(tuple(value for value in row) for row in covariance)
