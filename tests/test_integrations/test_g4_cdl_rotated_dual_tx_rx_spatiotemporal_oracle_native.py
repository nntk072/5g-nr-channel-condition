"""Deterministic Step-11 checks for the rotated dual-endpoint CDL oracle."""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any

import pytest
from evaluation.g4_cdl_rotated_dual_tx_rx_spatiotemporal_covariance import (
    _local_y_unit_vector,
    _unit_direction,
    _vh_basis,
)

torch = pytest.importorskip("torch")
pytest.importorskip("sionna.phy")

from integrations.sionna.cdl import CDLArrayConfig  # noqa: E402
from sionna.phy.channel.tr38901 import PanelArray  # noqa: E402
from sionna.phy.channel.tr38901.channel_coefficients import (  # noqa: E402
    ChannelCoefficientsGenerator,
    Topology,
)

_CARRIER_FREQUENCY_HZ = 3.5e9
_ORIENTATION_PAIRS = (
    ((-0.22, 0.31, 0.47), (0.37, 0.28, -0.41)),
    ((0.41, -0.27, 0.0), (0.0, 0.0, 0.65)),
    ((0.0, -0.33, -0.52), (0.28, -0.36, 0.0)),
)
_ANGLES_DEG = (
    {"aoa": 37.0, "zoa": 103.0, "aod": -52.0, "zod": 78.0},
    {"aoa": -121.0, "zoa": 64.0, "aod": 143.0, "zod": 112.0},
)
_JONES = ((1.2 + 0.3j, -0.4 + 0.7j), (0.2 - 0.5j, -1.1 + 0.8j))


def _generator() -> ChannelCoefficientsGenerator:
    config = CDLArrayConfig(
        num_cols_per_panel=2,
        element_horizontal_spacing=0.5,
        polarization="dual",
        polarization_type="VH",
        antenna_pattern="omni",
    )
    tx_array = PanelArray(
        **asdict(config), carrier_frequency=_CARRIER_FREQUENCY_HZ, precision="double", device="cpu"
    )
    rx_array = PanelArray(
        **asdict(config), carrier_frequency=_CARRIER_FREQUENCY_HZ, precision="double", device="cpu"
    )
    return ChannelCoefficientsGenerator(
        carrier_frequency=_CARRIER_FREQUENCY_HZ,
        tx_array=tx_array,
        rx_array=rx_array,
        subclustering=False,
        precision="double",
        device="cpu",
    )


def _topology(
    generator: ChannelCoefficientsGenerator,
    tx_orientation: tuple[float, float, float],
    rx_orientation: tuple[float, float, float],
) -> Topology:
    dtype = generator.dtype
    zero_angles = torch.zeros((1, 1, 1), dtype=dtype)
    return Topology(
        velocities=torch.zeros((1, 1, 3), dtype=dtype),
        moving_end="rx",
        los_aoa=zero_angles,
        los_aod=zero_angles,
        los_zoa=zero_angles,
        los_zod=zero_angles,
        los=torch.zeros((1, 1, 1), dtype=torch.bool),
        distance_3d=torch.zeros((1, 1, 1), dtype=dtype),
        tx_orientations=torch.tensor([[tx_orientation]], dtype=dtype),
        rx_orientations=torch.tensor([[rx_orientation]], dtype=dtype),
    )


def _angle_tensor(generator: ChannelCoefficientsGenerator, value_deg: float) -> Any:
    return torch.tensor([[[[[math.radians(value_deg)]]]]], dtype=generator.dtype)


def _field_reference(
    rx_pol: int,
    tx_pol: int,
    *,
    receive_direction: tuple[float, float, float],
    transmit_direction: tuple[float, float, float],
    rx_orientation: tuple[float, float, float],
    tx_orientation: tuple[float, float, float],
) -> complex:
    rx_basis = _vh_basis(rx_orientation, receive_direction)[rx_pol]
    tx_basis = _vh_basis(tx_orientation, transmit_direction)[tx_pol]
    return sum(
        rx_basis[row] * _JONES[row][column] * tx_basis[column]
        for row in range(2)
        for column in range(2)
    )


@pytest.mark.native_sionna
@pytest.mark.parametrize(("tx_orientation", "rx_orientation"), _ORIENTATION_PAIRS)
@pytest.mark.parametrize("angles", _ANGLES_DEG)
def test_rotated_dual_endpoint_field_and_array_offsets_match_sionna_step_11(
    tx_orientation: tuple[float, float, float],
    rx_orientation: tuple[float, float, float],
    angles: dict[str, float],
) -> None:
    generator = _generator()
    topology = _topology(generator, tx_orientation, rx_orientation)
    aoa = _angle_tensor(generator, angles["aoa"])
    aod = _angle_tensor(generator, angles["aod"])
    zoa = _angle_tensor(generator, angles["zoa"])
    zod = _angle_tensor(generator, angles["zod"])
    jones = torch.tensor(_JONES, dtype=torch.complex128).reshape(1, 1, 1, 1, 1, 2, 2)

    field = generator._step_11_field_matrix(topology, aoa, aod, zoa, zod, jones)[0, 0, 0, 0, 0].to(
        torch.complex128
    )
    receive_direction = _unit_direction(angles["aoa"], angles["zoa"])
    transmit_direction = _unit_direction(angles["aod"], angles["zod"])
    for rx_pol in range(2):
        for tx_pol in range(2):
            expected = _field_reference(
                rx_pol,
                tx_pol,
                receive_direction=receive_direction,
                transmit_direction=transmit_direction,
                rx_orientation=rx_orientation,
                tx_orientation=tx_orientation,
            )
            for rx_element in range(2):
                for tx_element in range(2):
                    assert complex(
                        field[rx_pol * 2 + rx_element, tx_pol * 2 + tx_element]
                    ) == pytest.approx(expected, abs=2e-12)

    array = generator._step_11_array_offsets(topology, aoa, aod, zoa, zod)[0, 0, 0, 0, 0].to(
        torch.complex128
    )
    rx_local_y = _local_y_unit_vector(rx_orientation)
    tx_local_y = _local_y_unit_vector(tx_orientation)
    rx_projection = sum(a * b for a, b in zip(receive_direction, rx_local_y, strict=True))
    tx_projection = sum(a * b for a, b in zip(transmit_direction, tx_local_y, strict=True))
    element_coordinates = (-0.25, 0.25, -0.25, 0.25)
    for rx_port, rx_coordinate in enumerate(element_coordinates):
        for tx_port, tx_coordinate in enumerate(element_coordinates):
            phase = 2.0 * math.pi * (rx_coordinate * rx_projection + tx_coordinate * tx_projection)
            expected = complex(math.cos(phase), math.sin(phase))
            assert complex(array[rx_port, tx_port]) == pytest.approx(expected, abs=2e-12)
