"""Independent ETSI CDL rotated dual-TX/dual-RX space-time covariance reference."""

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


def _orientation(value: Sequence[float], *, label: str) -> tuple[float, float, float]:
    if len(value) != 3:
        raise ValueError(f"{label} must contain exactly three radians")
    return tuple(_finite_real(component, label=f"{label} component") for component in value)  # type: ignore[return-value]


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
    # Sionna independently shuffles each CDL angular domain in Step 8. For one
    # endpoint, the ensemble ray direction is therefore the Cartesian product
    # of the 20 azimuth and 20 zenith offsets.
    return tuple(
        _unit_direction(
            azimuth_deg + azimuth_spread_deg * az_offset,
            zenith_deg + zenith_spread_deg * zenith_offset,
        )
        for az_offset in _RAY_ANGLE_OFFSETS
        for zenith_offset in _RAY_ANGLE_OFFSETS
    )


def _angles_from_direction(direction: tuple[float, float, float]) -> tuple[float, float]:
    x, y, z = direction
    return math.acos(max(-1.0, min(1.0, z))), math.atan2(y, x)


def _displacement_angle(
    orientation_rad: tuple[float, float, float],
    direction: tuple[float, float, float],
) -> float:
    """TR 38.901 (7.1-15) LCS-to-GCS field displacement angle."""
    alpha, beta, gamma = orientation_rad
    theta, phi = _angles_from_direction(direction)
    delta = phi - alpha
    real = math.sin(gamma) * math.cos(theta) * math.sin(delta)
    real += math.cos(gamma) * (
        math.cos(beta) * math.sin(theta) - math.sin(beta) * math.cos(theta) * math.cos(delta)
    )
    imag = math.sin(gamma) * math.cos(delta)
    imag += math.sin(beta) * math.cos(gamma) * math.sin(delta)
    return math.atan2(imag, real)


def _vh_basis(
    orientation_rad: tuple[float, float, float],
    direction: tuple[float, float, float],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return GCS (theta, phi) field vectors for local V and H omni ports."""
    psi = _displacement_angle(orientation_rad, direction)
    cos_psi = math.cos(psi)
    sin_psi = math.sin(psi)
    return ((cos_psi, sin_psi), (-sin_psi, cos_psi))


def _local_y_unit_vector(
    orientation_rad: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Return the rotated local-y axis, the horizontal panel-column direction."""
    alpha, beta, gamma = orientation_rad
    return (
        math.cos(alpha) * math.sin(beta) * math.sin(gamma) - math.sin(alpha) * math.cos(gamma),
        math.sin(alpha) * math.sin(beta) * math.sin(gamma) + math.cos(alpha) * math.cos(gamma),
        math.cos(beta) * math.sin(gamma),
    )


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return math.fsum(a * b for a, b in zip(left, right, strict=True))


def _decode_dual_vh_port(port: int, element_count: int) -> tuple[int, int]:
    if port < element_count:
        return 0, port
    return 1, port - element_count


def _endpoint_moments(
    directions: Sequence[tuple[float, float, float]],
    *,
    orientation_rad: tuple[float, float, float],
    element_count: int,
    spatial_scale: float,
    temporal_scale: float = 0.0,
    velocity_mps: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> dict[tuple[int, int, int, int], complex]:
    """Average phase-weighted basis products over independent angle coupling."""
    local_y = _local_y_unit_vector(orientation_rad)
    deltas = range(-(element_count - 1), element_count)
    accumulators = {
        (delta, row_pol, column_pol, field_index): 0.0j
        for delta in deltas
        for row_pol in range(2)
        for column_pol in range(2)
        for field_index in range(2)
    }
    for direction in directions:
        basis = _vh_basis(orientation_rad, direction)
        spatial_projection = _dot(direction, local_y)
        doppler_projection = _dot(velocity_mps, direction)
        for delta in deltas:
            phase = spatial_scale * delta * spatial_projection
            phase += temporal_scale * doppler_projection
            phasor = cmath.exp(1j * phase)
            for row_pol in range(2):
                for column_pol in range(2):
                    for field_index in range(2):
                        accumulators[(delta, row_pol, column_pol, field_index)] += (
                            phasor * basis[row_pol][field_index] * basis[column_pol][field_index]
                        )
    scale = 1.0 / len(directions)
    return {key: value * scale for key, value in accumulators.items()}


def _specular_field(
    rx_pol: int,
    tx_pol: int,
    *,
    rx_orientation_rad: tuple[float, float, float],
    tx_orientation_rad: tuple[float, float, float],
    receive_direction: tuple[float, float, float],
    transmit_direction: tuple[float, float, float],
) -> float:
    rx_basis = _vh_basis(rx_orientation_rad, receive_direction)[rx_pol]
    tx_basis = _vh_basis(tx_orientation_rad, transmit_direction)[tx_pol]
    # Step 11 LOS polarization matrix is diag(1, -1).
    return rx_basis[0] * tx_basis[0] - rx_basis[1] * tx_basis[1]


def cdl_rotated_dual_tx_rx_spatiotemporal_covariance_reference(
    model: str,
    velocity_mps: Sequence[float],
    *,
    tx_orientation_rad: Sequence[float],
    rx_orientation_rad: Sequence[float],
    carrier_frequency_hz: float = 3.5e9,
    lag_s: float = 0.7e-3,
    tx_element_count: int = 2,
    rx_element_count: int = 2,
    element_spacing_wavelengths: float = 0.5,
) -> tuple[tuple[complex, ...], ...]:
    """Return E[h_p(0)^* h_q(tau)] for rotated dual-VH CDL arrays.

    The reference is independent of Sionna runtime metadata. It uses the frozen
    ETSI-derived CDL fixture, TR 38.901 field-coordinate transform, rotated
    panel geometry, the four independent diffuse Jones phases, Sionna's
    Step-8 independent angular coupling semantics, and receive-side Doppler.
    """
    if model not in set("ABCDE"):
        raise ValueError(f"unsupported CDL profile: {model}")
    if len(velocity_mps) != 3:
        raise ValueError("velocity_mps must contain exactly three Cartesian components")
    velocity = tuple(_finite_real(value, label="velocity_mps component") for value in velocity_mps)
    tx_orientation = _orientation(tx_orientation_rad, label="tx_orientation_rad")
    rx_orientation = _orientation(rx_orientation_rad, label="rx_orientation_rad")
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
    jones_variances = ((1.0, 1.0 / xpr_linear), (1.0 / xpr_linear, 1.0))
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

        rx_moments = _endpoint_moments(
            receive_directions,
            orientation_rad=rx_orientation,
            element_count=rx_element_count,
            spatial_scale=spatial_scale,
            temporal_scale=temporal_scale,
            velocity_mps=velocity,
        )
        tx_moments = _endpoint_moments(
            transmit_directions,
            orientation_rad=tx_orientation,
            element_count=tx_element_count,
            spatial_scale=spatial_scale,
        )
        normalized_power = raw_power / power_sum

        for row in range(channel_port_count):
            row_rx_port, row_tx_port = divmod(row, tx_port_count)
            row_rx_pol, row_rx_element = _decode_dual_vh_port(row_rx_port, rx_element_count)
            row_tx_pol, row_tx_element = _decode_dual_vh_port(row_tx_port, tx_element_count)
            for column in range(channel_port_count):
                column_rx_port, column_tx_port = divmod(column, tx_port_count)
                column_rx_pol, column_rx_element = _decode_dual_vh_port(
                    column_rx_port, rx_element_count
                )
                column_tx_pol, column_tx_element = _decode_dual_vh_port(
                    column_tx_port, tx_element_count
                )
                delta_rx = column_rx_element - row_rx_element
                delta_tx = column_tx_element - row_tx_element
                if kind == "diffuse":
                    polarization_spatial_temporal = 0.0j
                    for rx_field_index in range(2):
                        rx_factor = rx_moments[
                            (delta_rx, row_rx_pol, column_rx_pol, rx_field_index)
                        ]
                        for tx_field_index in range(2):
                            tx_factor = tx_moments[
                                (delta_tx, row_tx_pol, column_tx_pol, tx_field_index)
                            ]
                            polarization_spatial_temporal += (
                                jones_variances[rx_field_index][tx_field_index]
                                * rx_factor
                                * tx_factor
                            )
                else:
                    receive_direction = receive_directions[0]
                    transmit_direction = transmit_directions[0]
                    row_field = _specular_field(
                        row_rx_pol,
                        row_tx_pol,
                        rx_orientation_rad=rx_orientation,
                        tx_orientation_rad=tx_orientation,
                        receive_direction=receive_direction,
                        transmit_direction=transmit_direction,
                    )
                    column_field = _specular_field(
                        column_rx_pol,
                        column_tx_pol,
                        rx_orientation_rad=rx_orientation,
                        tx_orientation_rad=tx_orientation,
                        receive_direction=receive_direction,
                        transmit_direction=transmit_direction,
                    )
                    phase = (
                        spatial_scale
                        * delta_rx
                        * _dot(receive_direction, _local_y_unit_vector(rx_orientation))
                        + temporal_scale * _dot(velocity, receive_direction)
                        + spatial_scale
                        * delta_tx
                        * _dot(transmit_direction, _local_y_unit_vector(tx_orientation))
                    )
                    polarization_spatial_temporal = row_field * column_field * cmath.exp(1j * phase)
                covariance[row][column] += normalized_power * polarization_spatial_temporal

    return tuple(tuple(value for value in row) for row in covariance)
