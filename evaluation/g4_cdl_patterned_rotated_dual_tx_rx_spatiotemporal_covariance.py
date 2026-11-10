"""Independent patterned rotated dual-TX/dual-RX CDL space-time covariance reference."""

from __future__ import annotations

import cmath
import math
from typing import Sequence

from evaluation.g4_cdl_rotated_dual_tx_rx_spatiotemporal_covariance import (
    _C0_MPS,
    _decode_dual_vh_port,
    _diffuse_directions,
    _dot,
    _finite_real,
    _local_y_unit_vector,
    _orientation,
    _reference_document,
    _unit_direction,
    _vh_basis,
)

_PATTERN_PARAMETERS = {
    "38.901": (65.0, 65.0, 30.0, 30.0, 8.0),
    "38.901-handheld": (125.0, 125.0, 22.5, 22.5, 5.3),
}
_SUPPORTED_PATTERNS = frozenset(("omni", *_PATTERN_PARAMETERS))


def _local_axes(
    orientation_rad: tuple[float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    """Return the GCS unit vectors of the endpoint's local x/y/z axes."""
    alpha, beta, gamma = orientation_rad
    cos_alpha, sin_alpha = math.cos(alpha), math.sin(alpha)
    cos_beta, sin_beta = math.cos(beta), math.sin(beta)
    cos_gamma, sin_gamma = math.cos(gamma), math.sin(gamma)
    local_x = (cos_alpha * cos_beta, sin_alpha * cos_beta, -sin_beta)
    local_y = (
        cos_alpha * sin_beta * sin_gamma - sin_alpha * cos_gamma,
        sin_alpha * sin_beta * sin_gamma + cos_alpha * cos_gamma,
        cos_beta * sin_gamma,
    )
    local_z = (
        cos_alpha * sin_beta * cos_gamma + sin_alpha * sin_gamma,
        sin_alpha * sin_beta * cos_gamma - cos_alpha * sin_gamma,
        cos_beta * cos_gamma,
    )
    return local_x, local_y, local_z


def _pattern_power(
    pattern: str,
    orientation_rad: tuple[float, float, float],
    direction: tuple[float, float, float],
) -> float:
    """Evaluate the TR 38.901 element power pattern for one global ray direction."""
    if pattern not in _SUPPORTED_PATTERNS:
        raise ValueError(f"unsupported antenna pattern: {pattern}")
    if pattern == "omni":
        return 1.0
    local_x, local_y, local_z = _local_axes(orientation_rad)
    theta = math.acos(max(-1.0, min(1.0, _dot(direction, local_z))))
    phi = math.atan2(_dot(direction, local_y), _dot(direction, local_x))
    theta_3db, phi_3db, sla_v_db, a_max_db, g_max_db = _PATTERN_PARAMETERS[pattern]
    attenuation_vertical = -min(
        12.0 * ((theta - math.pi / 2.0) / math.radians(theta_3db)) ** 2,
        sla_v_db,
    )
    attenuation_horizontal = -min(
        12.0 * (phi / math.radians(phi_3db)) ** 2,
        a_max_db,
    )
    attenuation_db = -min(-(attenuation_vertical + attenuation_horizontal), a_max_db) + g_max_db
    return 10.0 ** (attenuation_db / 10.0)


def _patterned_vh_basis(
    pattern: str,
    orientation_rad: tuple[float, float, float],
    direction: tuple[float, float, float],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return patterned local V/H fields in the GCS theta/phi basis."""
    amplitude = math.sqrt(_pattern_power(pattern, orientation_rad, direction))
    base = _vh_basis(orientation_rad, direction)
    return tuple(
        (amplitude * field[0], amplitude * field[1])
        for field in base
    )  # type: ignore[return-value]


def _endpoint_moments(
    directions: Sequence[tuple[float, float, float]],
    *,
    pattern: str,
    orientation_rad: tuple[float, float, float],
    element_count: int,
    spatial_scale: float,
    temporal_scale: float = 0.0,
    velocity_mps: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> dict[tuple[int, int, int, int], complex]:
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
        basis = _patterned_vh_basis(pattern, orientation_rad, direction)
        spatial_projection = _dot(direction, local_y)
        doppler_projection = _dot(velocity_mps, direction)
        for delta in deltas:
            phase = spatial_scale * delta * spatial_projection + temporal_scale * doppler_projection
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
    rx_pattern: str,
    tx_pattern: str,
    rx_orientation_rad: tuple[float, float, float],
    tx_orientation_rad: tuple[float, float, float],
    receive_direction: tuple[float, float, float],
    transmit_direction: tuple[float, float, float],
) -> float:
    rx_basis = _patterned_vh_basis(rx_pattern, rx_orientation_rad, receive_direction)[rx_pol]
    tx_basis = _patterned_vh_basis(tx_pattern, tx_orientation_rad, transmit_direction)[tx_pol]
    return rx_basis[0] * tx_basis[0] - rx_basis[1] * tx_basis[1]


def cdl_patterned_rotated_dual_tx_rx_spatiotemporal_covariance_reference(
    model: str,
    velocity_mps: Sequence[float],
    *,
    tx_pattern: str,
    rx_pattern: str,
    tx_orientation_rad: Sequence[float],
    rx_orientation_rad: Sequence[float],
    carrier_frequency_hz: float = 3.5e9,
    lag_s: float = 0.7e-3,
    tx_element_count: int = 2,
    rx_element_count: int = 2,
    element_spacing_wavelengths: float = 0.5,
) -> tuple[tuple[complex, ...], ...]:
    """Return E[h_p(0)^* h_q(tau)] for patterned rotated dual-VH CDL arrays."""
    if model not in set("ABCDE"):
        raise ValueError(f"unsupported CDL profile: {model}")
    if tx_pattern not in _SUPPORTED_PATTERNS or rx_pattern not in _SUPPORTED_PATTERNS:
        raise ValueError("tx_pattern and rx_pattern must be supported TR 38.901 element patterns")
    if len(velocity_mps) != 3:
        raise ValueError("velocity_mps must contain exactly three Cartesian components")
    velocity = tuple(_finite_real(value, label="velocity_mps component") for value in velocity_mps)
    tx_orientation = _orientation(tx_orientation_rad, label="tx_orientation_rad")
    rx_orientation = _orientation(rx_orientation_rad, label="rx_orientation_rad")
    carrier = _finite_real(carrier_frequency_hz, label="carrier_frequency_hz")
    lag = _finite_real(lag_s, label="lag_s")
    spacing = _finite_real(element_spacing_wavelengths, label="element_spacing_wavelengths")
    for name, count in (("tx_element_count", tx_element_count), ("rx_element_count", rx_element_count)):
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
    spreads = profile["cluster_spreads_deg"]
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
                float(angles["aoa"]), float(angles["zoa"]), float(spreads["asa"]), float(spreads["zsa"])
            )
            transmit_directions = _diffuse_directions(
                float(angles["aod"]), float(angles["zod"]), float(spreads["asd"]), float(spreads["zsd"])
            )
        else:
            raise ValueError(f"unsupported CDL component kind: {kind}")

        rx_moments = _endpoint_moments(
            receive_directions,
            pattern=rx_pattern,
            orientation_rad=rx_orientation,
            element_count=rx_element_count,
            spatial_scale=spatial_scale,
            temporal_scale=temporal_scale,
            velocity_mps=velocity,
        )
        tx_moments = _endpoint_moments(
            transmit_directions,
            pattern=tx_pattern,
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
                column_rx_pol, column_rx_element = _decode_dual_vh_port(column_rx_port, rx_element_count)
                column_tx_pol, column_tx_element = _decode_dual_vh_port(column_tx_port, tx_element_count)
                delta_rx = column_rx_element - row_rx_element
                delta_tx = column_tx_element - row_tx_element
                if kind == "diffuse":
                    term = 0.0j
                    for rx_field_index in range(2):
                        rx_factor = rx_moments[(delta_rx, row_rx_pol, column_rx_pol, rx_field_index)]
                        for tx_field_index in range(2):
                            tx_factor = tx_moments[(delta_tx, row_tx_pol, column_tx_pol, tx_field_index)]
                            term += jones_variances[rx_field_index][tx_field_index] * rx_factor * tx_factor
                else:
                    receive_direction = receive_directions[0]
                    transmit_direction = transmit_directions[0]
                    row_field = _specular_field(
                        row_rx_pol,
                        row_tx_pol,
                        rx_pattern=rx_pattern,
                        tx_pattern=tx_pattern,
                        rx_orientation_rad=rx_orientation,
                        tx_orientation_rad=tx_orientation,
                        receive_direction=receive_direction,
                        transmit_direction=transmit_direction,
                    )
                    column_field = _specular_field(
                        column_rx_pol,
                        column_tx_pol,
                        rx_pattern=rx_pattern,
                        tx_pattern=tx_pattern,
                        rx_orientation_rad=rx_orientation,
                        tx_orientation_rad=tx_orientation,
                        receive_direction=receive_direction,
                        transmit_direction=transmit_direction,
                    )
                    phase = (
                        spatial_scale * delta_rx * _dot(receive_direction, _local_y_unit_vector(rx_orientation))
                        + temporal_scale * _dot(velocity, receive_direction)
                        + spatial_scale * delta_tx * _dot(transmit_direction, _local_y_unit_vector(tx_orientation))
                    )
                    term = row_field * column_field * cmath.exp(1j * phase)
                covariance[row][column] += normalized_power * term

    return tuple(tuple(value for value in row) for row in covariance)
