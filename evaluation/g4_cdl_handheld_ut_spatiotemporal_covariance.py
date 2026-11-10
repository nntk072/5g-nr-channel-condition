"""Independent physical handheld-UT CDL space-time covariance reference."""

from __future__ import annotations

import cmath
import math
from typing import Sequence

from evaluation.g4_cdl_patterned_rotated_dual_tx_rx_spatiotemporal_covariance import (
    _local_axes,
)
from evaluation.g4_cdl_rotated_dual_tx_rx_spatiotemporal_covariance import (
    _C0_MPS,
    _diffuse_directions,
    _dot,
    _finite_real,
    _orientation,
    _reference_document,
    _unit_direction,
    _vh_basis,
)

_HANDHELD_CANDIDATE_FRACTIONS = {
    1: (-0.5, -0.5),
    2: (0.0, -0.5),
    3: (0.5, -0.5),
    4: (0.5, 0.0),
    5: (0.5, 0.5),
    6: (0.0, 0.5),
    7: (-0.5, 0.5),
    8: (-0.5, 0.0),
}
_HANDHELD_ROSTERS = {
    "tr38901": tuple(range(1, 9)),
    "tr38901-4": (1, 7, 3, 5),
}


def _cross(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _normalize(v: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = math.sqrt(_dot(v, v))
    if norm == 0.0:
        raise ValueError("vector must be nonzero")
    return tuple(component / norm for component in v)  # type: ignore[return-value]


def _linear_combination(
    axes: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ],
    coordinates: tuple[float, float, float],
) -> tuple[float, float, float]:
    return tuple(
        math.fsum(axes[column][row] * coordinates[column] for column in range(3))
        for row in range(3)
    )  # type: ignore[return-value]


def _coordinates_in_axes(
    axes: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    return tuple(_dot(axis, vector) for axis in axes)  # type: ignore[return-value]


def _spherical_basis(
    direction: tuple[float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    x, y, z = _normalize(direction)
    theta = math.acos(max(-1.0, min(1.0, z)))
    phi = math.atan2(y, x)
    e_theta = (
        math.cos(theta) * math.cos(phi),
        math.cos(theta) * math.sin(phi),
        -math.sin(theta),
    )
    e_phi = (-math.sin(phi), math.cos(phi), 0.0)
    return e_theta, e_phi


def _handheld_pattern_power_local(direction: tuple[float, float, float]) -> float:
    """Evaluate TR 38.901 Table 7.3-2 in one antenna port's local frame."""
    x, y, z = _normalize(direction)
    theta = math.acos(max(-1.0, min(1.0, z)))
    phi = math.atan2(y, x)
    attenuation_vertical = -min(
        12.0 * ((theta - math.pi / 2.0) / math.radians(125.0)) ** 2,
        22.5,
    )
    attenuation_horizontal = -min(
        12.0 * (phi / math.radians(125.0)) ** 2,
        22.5,
    )
    attenuation_db = (
        -min(
            -(attenuation_vertical + attenuation_horizontal),
            22.5,
        )
        + 5.3
    )
    return 10.0 ** (attenuation_db / 10.0)


def _handheld_locations(
    antenna_locations: str | Sequence[int],
) -> tuple[int, ...]:
    if isinstance(antenna_locations, str):
        try:
            return _HANDHELD_ROSTERS[antenna_locations]
        except KeyError as exc:
            raise ValueError("unsupported handheld antenna-location roster") from exc
    locations = tuple(antenna_locations)
    if not locations:
        raise ValueError("antenna_locations must not be empty")
    if any(
        isinstance(index, bool)
        or not isinstance(index, int)
        or index not in _HANDHELD_CANDIDATE_FRACTIONS
        for index in locations
    ):
        raise ValueError("handheld candidate indices must be integers from 1 to 8")
    if len(set(locations)) != len(locations):
        raise ValueError("handheld candidate indices must be unique")
    return locations


def handheld_ut_single_v_geometry(
    antenna_locations: str | Sequence[int],
    *,
    device_depth_m: float = 0.15,
    device_width_m: float = 0.07,
) -> tuple[
    tuple[tuple[float, float, float], ...],
    tuple[
        tuple[
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
        ],
        ...,
    ],
]:
    """Return independent single-V handheld positions and per-port bases."""
    depth = _finite_real(device_depth_m, label="device_depth_m")
    width = _finite_real(device_width_m, label="device_width_m")
    if depth <= 0.0 or width <= 0.0:
        raise ValueError("device dimensions must be positive")
    locations = _handheld_locations(antenna_locations)
    positions: list[tuple[float, float, float]] = []
    bases: list[
        tuple[
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
        ]
    ] = []
    device_normal = (0.0, 0.0, 1.0)
    for index in locations:
        x_fraction, y_fraction = _HANDHELD_CANDIDATE_FRACTIONS[index]
        position = (x_fraction * depth, y_fraction * width, 0.0)
        boresight = _normalize(position)
        polarization = _normalize(_cross(boresight, device_normal))
        x_axis = boresight
        z_axis = (-polarization[0], -polarization[1], -polarization[2])
        y_axis = _normalize(_cross(z_axis, x_axis))
        z_axis = _normalize(_cross(x_axis, y_axis))
        positions.append(position)
        bases.append((x_axis, y_axis, z_axis))
    return tuple(positions), tuple(bases)


def _rotate_about_axis(
    vector: tuple[float, float, float],
    axis: tuple[float, float, float],
    angle_rad: float,
) -> tuple[float, float, float]:
    """Rotate a vector around a unit axis with Rodrigues' formula."""
    unit_axis = _normalize(axis)
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    cross_term = _cross(unit_axis, vector)
    projection = _dot(unit_axis, vector)
    return tuple(
        vector[index] * cosine
        + cross_term[index] * sine
        + unit_axis[index] * projection * (1.0 - cosine)
        for index in range(3)
    )  # type: ignore[return-value]


def handheld_ut_dual_geometry(
    antenna_locations: str | Sequence[int],
    *,
    polarization_type: str = "cross",
    device_depth_m: float = 0.15,
    device_width_m: float = 0.07,
) -> tuple[
    tuple[tuple[float, float, float], ...],
    tuple[
        tuple[
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
        ],
        ...,
    ],
    tuple[int, ...],
]:
    """Return dual-polarized handheld positions, bases, and local field components."""
    if polarization_type not in {"VH", "cross"}:
        raise ValueError("polarization_type must be 'VH' or 'cross'")
    depth = _finite_real(device_depth_m, label="device_depth_m")
    width = _finite_real(device_width_m, label="device_width_m")
    if depth <= 0.0 or width <= 0.0:
        raise ValueError("device dimensions must be positive")
    locations = _handheld_locations(antenna_locations)
    positions: list[tuple[float, float, float]] = []
    bases: list[
        tuple[
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
        ]
    ] = []
    device_normal = (0.0, 0.0, 1.0)
    for index in locations:
        x_fraction, y_fraction = _HANDHELD_CANDIDATE_FRACTIONS[index]
        position = (x_fraction * depth, y_fraction * width, 0.0)
        boresight = _normalize(position)
        polarization = _normalize(_cross(boresight, device_normal))
        if polarization_type == "cross":
            polarization = _normalize(_rotate_about_axis(polarization, boresight, math.pi / 4.0))
        x_axis = boresight
        z_axis = (-polarization[0], -polarization[1], -polarization[2])
        y_axis = _normalize(_cross(z_axis, x_axis))
        z_axis = _normalize(_cross(x_axis, y_axis))
        positions.append(position)
        bases.append((x_axis, y_axis, z_axis))
    return (
        tuple(positions + positions),
        tuple(bases + bases),
        tuple([0] * len(positions) + [1] * len(positions)),
    )


def handheld_ut_port_field_gcs(
    direction_gcs: tuple[float, float, float],
    *,
    endpoint_orientation_rad: Sequence[float],
    port_basis_device: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ],
    field_component: int,
) -> tuple[float, float]:
    """Return one physical handheld port field in the GCS theta/phi basis."""
    if field_component not in {0, 1}:
        raise ValueError("field_component must be 0 (theta) or 1 (phi)")
    orientation = _orientation(endpoint_orientation_rad, label="endpoint_orientation_rad")
    endpoint_axes = _local_axes(orientation)
    direction_device = _coordinates_in_axes(endpoint_axes, _normalize(direction_gcs))
    direction_port = _coordinates_in_axes(port_basis_device, direction_device)
    amplitude = math.sqrt(_handheld_pattern_power_local(direction_port))
    e_theta_port, e_phi_port = _spherical_basis(direction_port)
    selected_basis = e_theta_port if field_component == 0 else e_phi_port
    field_device = _linear_combination(
        port_basis_device,
        tuple(amplitude * value for value in selected_basis),  # type: ignore[arg-type]
    )
    field_gcs = _linear_combination(endpoint_axes, field_device)
    e_theta_gcs, e_phi_gcs = _spherical_basis(direction_gcs)
    return _dot(field_gcs, e_theta_gcs), _dot(field_gcs, e_phi_gcs)


def handheld_ut_single_v_field_gcs(
    direction_gcs: tuple[float, float, float],
    *,
    endpoint_orientation_rad: Sequence[float],
    port_basis_device: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ],
) -> tuple[float, float]:
    """Return one physical single-V handheld port field in the GCS theta/phi basis."""
    return handheld_ut_port_field_gcs(
        direction_gcs,
        endpoint_orientation_rad=endpoint_orientation_rad,
        port_basis_device=port_basis_device,
        field_component=0,
    )


def _handheld_positions_gcs(
    positions_device: Sequence[tuple[float, float, float]],
    orientation_rad: tuple[float, float, float],
) -> tuple[tuple[float, float, float], ...]:
    axes = _local_axes(orientation_rad)
    return tuple(_linear_combination(axes, position) for position in positions_device)


def cdl_handheld_ut_spatiotemporal_covariance_reference(
    model: str,
    velocity_mps: Sequence[float],
    *,
    antenna_locations: str | Sequence[int],
    ut_orientation_rad: Sequence[float],
    carrier_frequency_hz: float = 3.5e9,
    lag_s: float = 0.7e-3,
    device_depth_m: float = 0.15,
    device_width_m: float = 0.07,
) -> tuple[tuple[complex, ...], ...]:
    """Return E[h_i(0)^* h_j(tau)] for a physical single-V handheld UT."""
    if model not in set("ABCDE"):
        raise ValueError(f"unsupported CDL profile: {model}")
    velocity = tuple(_finite_real(value, label="velocity_mps component") for value in velocity_mps)
    if len(velocity) != 3:
        raise ValueError("velocity_mps must contain exactly three Cartesian components")
    orientation = _orientation(ut_orientation_rad, label="ut_orientation_rad")
    carrier = _finite_real(carrier_frequency_hz, label="carrier_frequency_hz")
    lag = _finite_real(lag_s, label="lag_s")
    if carrier <= 0.0 or lag < 0.0:
        raise ValueError("carrier must be positive and lag must be nonnegative")

    positions_device, port_bases = handheld_ut_single_v_geometry(
        antenna_locations,
        device_depth_m=device_depth_m,
        device_width_m=device_width_m,
    )
    positions_gcs = _handheld_positions_gcs(positions_device, orientation)
    port_count = len(positions_gcs)
    covariance = [[0.0j for _ in range(port_count)] for _ in range(port_count)]

    document = _reference_document()
    profile = next(row for row in document["profiles"] if row["profile_model"] == model)
    components = profile["components"]
    raw_powers = [10.0 ** (float(row["power_db"]) / 10.0) for row in components]
    power_sum = math.fsum(raw_powers)
    xpr_linear = 10.0 ** (float(profile["xpr_db"]) / 10.0)
    jones_variances = ((1.0, 1.0 / xpr_linear), (1.0 / xpr_linear, 1.0))
    spreads = profile["cluster_spreads_deg"]
    spatial_scale = 2.0 * math.pi * carrier / _C0_MPS
    temporal_scale = spatial_scale * lag

    for component, raw_power in zip(components, raw_powers, strict=True):
        kind = component["component"]
        angles = component["angles_deg"]
        receive_directions: tuple[tuple[float, float, float], ...]
        transmit_directions: tuple[tuple[float, float, float], ...]
        if kind == "specular":
            receive_directions = (_unit_direction(float(angles["aoa"]), float(angles["zoa"])),)
            transmit_directions = (_unit_direction(float(angles["aod"]), float(angles["zod"])),)
        elif kind == "diffuse":
            receive_directions = _diffuse_directions(
                float(angles["aoa"]),
                float(angles["zoa"]),
                float(spreads["asa"]),
                float(spreads["zsa"]),
            )
            transmit_directions = _diffuse_directions(
                float(angles["aod"]),
                float(angles["zod"]),
                float(spreads["asd"]),
                float(spreads["zsd"]),
            )
        else:
            raise ValueError(f"unsupported CDL component kind: {kind}")

        normalized_power = raw_power / power_sum
        if kind == "diffuse":
            tx_field_power = [0.0, 0.0]
            for direction in transmit_directions:
                tx_field = _vh_basis((0.0, 0.0, 0.0), direction)[0]
                for field_index in range(2):
                    tx_field_power[field_index] += tx_field[field_index] ** 2
            tx_field_power = [value / len(transmit_directions) for value in tx_field_power]

            rx_accumulators = {
                (row, column, field_index): 0.0j
                for row in range(port_count)
                for column in range(port_count)
                for field_index in range(2)
            }
            for direction in receive_directions:
                fields = [
                    handheld_ut_single_v_field_gcs(
                        direction,
                        endpoint_orientation_rad=orientation,
                        port_basis_device=basis,
                    )
                    for basis in port_bases
                ]
                doppler_projection = _dot(velocity, direction)
                for row in range(port_count):
                    for column in range(port_count):
                        delta_position = tuple(
                            positions_gcs[column][axis] - positions_gcs[row][axis]
                            for axis in range(3)
                        )
                        phase = (
                            spatial_scale * _dot(direction, delta_position)
                            + temporal_scale * doppler_projection
                        )
                        phasor = cmath.exp(1j * phase)
                        for field_index in range(2):
                            rx_accumulators[(row, column, field_index)] += (
                                phasor * fields[row][field_index] * fields[column][field_index]
                            )
            ray_scale = 1.0 / len(receive_directions)
            for row in range(port_count):
                for column in range(port_count):
                    term = 0.0j
                    for rx_field_index in range(2):
                        rx_factor = rx_accumulators[(row, column, rx_field_index)] * ray_scale
                        for tx_field_index in range(2):
                            term += (
                                jones_variances[rx_field_index][tx_field_index]
                                * rx_factor
                                * tx_field_power[tx_field_index]
                            )
                    covariance[row][column] += normalized_power * term
        else:
            receive_direction = receive_directions[0]
            transmit_direction = transmit_directions[0]
            tx_field = _vh_basis((0.0, 0.0, 0.0), transmit_direction)[0]
            rx_fields = [
                handheld_ut_single_v_field_gcs(
                    receive_direction,
                    endpoint_orientation_rad=orientation,
                    port_basis_device=basis,
                )
                for basis in port_bases
            ]
            scalar_fields = [
                rx_field[0] * tx_field[0] - rx_field[1] * tx_field[1] for rx_field in rx_fields
            ]
            for row in range(port_count):
                for column in range(port_count):
                    delta_position = tuple(
                        positions_gcs[column][axis] - positions_gcs[row][axis] for axis in range(3)
                    )
                    phase = spatial_scale * _dot(
                        receive_direction, delta_position
                    ) + temporal_scale * _dot(velocity, receive_direction)
                    covariance[row][column] += (
                        normalized_power
                        * scalar_fields[row]
                        * scalar_fields[column]
                        * cmath.exp(1j * phase)
                    )

    return tuple(tuple(value for value in row) for row in covariance)


def cdl_handheld_ut_dual_spatiotemporal_covariance_reference(
    model: str,
    velocity_mps: Sequence[float],
    *,
    antenna_locations: str | Sequence[int],
    ut_orientation_rad: Sequence[float],
    polarization_type: str = "cross",
    carrier_frequency_hz: float = 28.0e9,
    lag_s: float = 0.2e-3,
    device_depth_m: float = 0.15,
    device_width_m: float = 0.07,
) -> tuple[tuple[complex, ...], ...]:
    """Return E[h_i(0)^* h_j(tau)] for a physical dual-pol handheld UT."""
    if model not in set("ABCDE"):
        raise ValueError(f"unsupported CDL profile: {model}")
    velocity = tuple(_finite_real(value, label="velocity_mps component") for value in velocity_mps)
    if len(velocity) != 3:
        raise ValueError("velocity_mps must contain exactly three Cartesian components")
    orientation = _orientation(ut_orientation_rad, label="ut_orientation_rad")
    carrier = _finite_real(carrier_frequency_hz, label="carrier_frequency_hz")
    lag = _finite_real(lag_s, label="lag_s")
    if carrier <= 0.0 or lag < 0.0:
        raise ValueError("carrier must be positive and lag must be nonnegative")

    positions_device, port_bases, field_components = handheld_ut_dual_geometry(
        antenna_locations,
        polarization_type=polarization_type,
        device_depth_m=device_depth_m,
        device_width_m=device_width_m,
    )
    positions_gcs = _handheld_positions_gcs(positions_device, orientation)
    port_count = len(positions_gcs)
    covariance = [[0.0j for _ in range(port_count)] for _ in range(port_count)]

    document = _reference_document()
    profile = next(row for row in document["profiles"] if row["profile_model"] == model)
    components = profile["components"]
    raw_powers = [10.0 ** (float(row["power_db"]) / 10.0) for row in components]
    power_sum = math.fsum(raw_powers)
    xpr_linear = 10.0 ** (float(profile["xpr_db"]) / 10.0)
    jones_variances = ((1.0, 1.0 / xpr_linear), (1.0 / xpr_linear, 1.0))
    spreads = profile["cluster_spreads_deg"]
    spatial_scale = 2.0 * math.pi * carrier / _C0_MPS
    temporal_scale = spatial_scale * lag

    for component, raw_power in zip(components, raw_powers, strict=True):
        kind = component["component"]
        angles = component["angles_deg"]
        if kind == "specular":
            receive_directions = (_unit_direction(float(angles["aoa"]), float(angles["zoa"])),)
            transmit_directions = (_unit_direction(float(angles["aod"]), float(angles["zod"])),)
        elif kind == "diffuse":
            receive_directions = _diffuse_directions(
                float(angles["aoa"]),
                float(angles["zoa"]),
                float(spreads["asa"]),
                float(spreads["zsa"]),
            )
            transmit_directions = _diffuse_directions(
                float(angles["aod"]),
                float(angles["zod"]),
                float(spreads["asd"]),
                float(spreads["zsd"]),
            )
        else:
            raise ValueError(f"unsupported CDL component kind: {kind}")

        normalized_power = raw_power / power_sum
        if kind == "diffuse":
            tx_field_power = [0.0, 0.0]
            for direction in transmit_directions:
                tx_field = _vh_basis((0.0, 0.0, 0.0), direction)[0]
                for field_index in range(2):
                    tx_field_power[field_index] += tx_field[field_index] ** 2
            tx_field_power = [value / len(transmit_directions) for value in tx_field_power]

            rx_accumulators = {
                (row, column, field_index): 0.0j
                for row in range(port_count)
                for column in range(port_count)
                for field_index in range(2)
            }
            for direction in receive_directions:
                fields = [
                    handheld_ut_port_field_gcs(
                        direction,
                        endpoint_orientation_rad=orientation,
                        port_basis_device=basis,
                        field_component=field_component,
                    )
                    for basis, field_component in zip(port_bases, field_components, strict=True)
                ]
                doppler_projection = _dot(velocity, direction)
                for row in range(port_count):
                    for column in range(port_count):
                        delta_position = tuple(
                            positions_gcs[column][axis] - positions_gcs[row][axis]
                            for axis in range(3)
                        )
                        phase = (
                            spatial_scale * _dot(direction, delta_position)
                            + temporal_scale * doppler_projection
                        )
                        phasor = cmath.exp(1j * phase)
                        for field_index in range(2):
                            rx_accumulators[(row, column, field_index)] += (
                                phasor * fields[row][field_index] * fields[column][field_index]
                            )
            ray_scale = 1.0 / len(receive_directions)
            for row in range(port_count):
                for column in range(port_count):
                    term = 0.0j
                    for rx_field_index in range(2):
                        rx_factor = rx_accumulators[(row, column, rx_field_index)] * ray_scale
                        for tx_field_index in range(2):
                            term += (
                                jones_variances[rx_field_index][tx_field_index]
                                * rx_factor
                                * tx_field_power[tx_field_index]
                            )
                    covariance[row][column] += normalized_power * term
        else:
            receive_direction = receive_directions[0]
            transmit_direction = transmit_directions[0]
            tx_field = _vh_basis((0.0, 0.0, 0.0), transmit_direction)[0]
            rx_fields = [
                handheld_ut_port_field_gcs(
                    receive_direction,
                    endpoint_orientation_rad=orientation,
                    port_basis_device=basis,
                    field_component=field_component,
                )
                for basis, field_component in zip(port_bases, field_components, strict=True)
            ]
            scalar_fields = [
                rx_field[0] * tx_field[0] - rx_field[1] * tx_field[1] for rx_field in rx_fields
            ]
            for row in range(port_count):
                for column in range(port_count):
                    delta_position = tuple(
                        positions_gcs[column][axis] - positions_gcs[row][axis] for axis in range(3)
                    )
                    phase = spatial_scale * _dot(
                        receive_direction, delta_position
                    ) + temporal_scale * _dot(velocity, receive_direction)
                    covariance[row][column] += (
                        normalized_power
                        * scalar_fields[row]
                        * scalar_fields[column]
                        * cmath.exp(1j * phase)
                    )

    return tuple(tuple(value for value in row) for row in covariance)
