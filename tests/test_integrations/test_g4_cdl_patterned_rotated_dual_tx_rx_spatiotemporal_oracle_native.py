"""Deterministic Step-11 checks for the patterned rotated dual-endpoint CDL oracle."""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any

import pytest
from evaluation.g4_cdl_patterned_rotated_dual_tx_rx_spatiotemporal_covariance import (
    _patterned_vh_basis,
)
from evaluation.g4_cdl_rotated_dual_tx_rx_spatiotemporal_covariance import _unit_direction

torch = pytest.importorskip("torch")
pytest.importorskip("sionna.phy")

from integrations.sionna.cdl import CDLArrayConfig  # noqa: E402
from sionna.phy.channel.tr38901 import PanelArray  # noqa: E402
from sionna.phy.channel.tr38901.channel_coefficients import (  # noqa: E402
    ChannelCoefficientsGenerator,
    Topology,
)

_CARRIER_FREQUENCY_HZ = 3.5e9
_TX_ORIENTATION = (-0.7, 0.35, 1.1)
_RX_ORIENTATION = (0.2, -0.1, 0.3)
_PATTERNS = ("38.901", "38.901-handheld")
_ANGLES_DEG = (
    {"aoa": 37.0, "zoa": 103.0, "aod": -52.0, "zod": 78.0},
    {"aoa": -121.0, "zoa": 64.0, "aod": 143.0, "zod": 112.0},
)
_JONES = ((1.2 + 0.3j, -0.4 + 0.7j), (0.2 - 0.5j, -1.1 + 0.8j))


def _generator(tx_pattern: str, rx_pattern: str) -> ChannelCoefficientsGenerator:
    common = dict(
        num_cols_per_panel=2,
        element_horizontal_spacing=0.5,
        polarization="dual",
        polarization_type="VH",
    )
    tx_config = CDLArrayConfig(**common, antenna_pattern=tx_pattern)
    rx_config = CDLArrayConfig(**common, antenna_pattern=rx_pattern)
    tx_array = PanelArray(
        **asdict(tx_config), carrier_frequency=_CARRIER_FREQUENCY_HZ, precision="double", device="cpu"
    )
    rx_array = PanelArray(
        **asdict(rx_config), carrier_frequency=_CARRIER_FREQUENCY_HZ, precision="double", device="cpu"
    )
    return ChannelCoefficientsGenerator(
        carrier_frequency=_CARRIER_FREQUENCY_HZ,
        tx_array=tx_array,
        rx_array=rx_array,
        subclustering=False,
        precision="double",
        device="cpu",
    )


def _topology(generator: ChannelCoefficientsGenerator) -> Topology:
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
        tx_orientations=torch.tensor([[_TX_ORIENTATION]], dtype=dtype),
        rx_orientations=torch.tensor([[_RX_ORIENTATION]], dtype=dtype),
    )


def _angle_tensor(generator: ChannelCoefficientsGenerator, value_deg: float) -> Any:
    return torch.tensor([[[[[math.radians(value_deg)]]]]], dtype=generator.dtype)


def _field_reference(
    rx_pol: int,
    tx_pol: int,
    *,
    rx_pattern: str,
    tx_pattern: str,
    receive_direction: tuple[float, float, float],
    transmit_direction: tuple[float, float, float],
) -> complex:
    rx_basis = _patterned_vh_basis(rx_pattern, _RX_ORIENTATION, receive_direction)[rx_pol]
    tx_basis = _patterned_vh_basis(tx_pattern, _TX_ORIENTATION, transmit_direction)[tx_pol]
    return sum(
        rx_basis[row] * _JONES[row][column] * tx_basis[column]
        for row in range(2)
        for column in range(2)
    )


@pytest.mark.native_sionna
@pytest.mark.parametrize("tx_pattern", _PATTERNS)
@pytest.mark.parametrize("rx_pattern", _PATTERNS)
@pytest.mark.parametrize("angles", _ANGLES_DEG)
def test_patterned_rotated_dual_endpoint_fields_match_sionna_step_11(
    tx_pattern: str,
    rx_pattern: str,
    angles: dict[str, float],
) -> None:
    generator = _generator(tx_pattern, rx_pattern)
    topology = _topology(generator)
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
                rx_pattern=rx_pattern,
                tx_pattern=tx_pattern,
                receive_direction=receive_direction,
                transmit_direction=transmit_direction,
            )
            for rx_element in range(2):
                for tx_element in range(2):
                    assert complex(
                        field[rx_pol * 2 + rx_element, tx_pol * 2 + tx_element]
                    ) == pytest.approx(expected, abs=2e-12)
