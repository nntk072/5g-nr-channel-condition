"""Native deterministic checks for physical HandheldUTArray geometry and fields."""

from __future__ import annotations

import math
from dataclasses import asdict

import pytest
from evaluation.g4_cdl_handheld_ut_spatiotemporal_covariance import (
    handheld_ut_single_v_field_gcs,
    handheld_ut_single_v_geometry,
)
from evaluation.g4_cdl_rotated_dual_tx_rx_spatiotemporal_covariance import (
    _unit_direction,
    _vh_basis,
)

torch = pytest.importorskip("torch")
pytest.importorskip("sionna.phy")

from integrations.sionna.cdl import (  # noqa: E402
    CDLArrayConfig,
    CDLHandheldUTArrayConfig,
    generate_cdl_frequency_response,
)
from integrations.sionna.random import call_with_sionna_seed  # noqa: E402
from sionna.phy.channel.tr38901 import CDL, HandheldUTArray, PanelArray  # noqa: E402
from sionna.phy.channel.tr38901.channel_coefficients import (  # noqa: E402
    ChannelCoefficientsGenerator,
    Topology,
)

_CARRIER_FREQUENCY_HZ = 3.5e9
_UT_ORIENTATION = (0.2, -0.1, 0.3)
_ANGLES_DEG = (
    {"aoa": 37.0, "zoa": 103.0, "aod": -52.0, "zod": 78.0},
    {"aoa": -121.0, "zoa": 64.0, "aod": 143.0, "zod": 112.0},
)
_JONES = ((1.2 + 0.3j, -0.4 + 0.7j), (0.2 - 0.5j, -1.1 + 0.8j))


def _handheld(roster: str) -> HandheldUTArray:
    return HandheldUTArray(
        carrier_frequency=_CARRIER_FREQUENCY_HZ,
        polarization="single",
        polarization_type="V",
        antenna_locations=roster,
        antenna_pattern="38.901-handheld",
        device_depth=0.15,
        device_width=0.07,
        precision="double",
        device="cpu",
    )


def _tx_array() -> PanelArray:
    return PanelArray(
        num_rows_per_panel=1,
        num_cols_per_panel=1,
        polarization="single",
        polarization_type="V",
        antenna_pattern="omni",
        carrier_frequency=_CARRIER_FREQUENCY_HZ,
        precision="double",
        device="cpu",
    )


def _generator(roster: str) -> ChannelCoefficientsGenerator:
    return ChannelCoefficientsGenerator(
        carrier_frequency=_CARRIER_FREQUENCY_HZ,
        tx_array=_tx_array(),
        rx_array=_handheld(roster),
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
        tx_orientations=torch.zeros((1, 1, 3), dtype=dtype),
        rx_orientations=torch.tensor([[_UT_ORIENTATION]], dtype=dtype),
    )


def _angle_tensor(generator: ChannelCoefficientsGenerator, value_deg: float):
    return torch.tensor([[[[[math.radians(value_deg)]]]]], dtype=generator.dtype)


@pytest.mark.native_sionna
@pytest.mark.parametrize("roster", ("tr38901-4", "tr38901"))
def test_handheld_geometry_matches_independent_tr38901_construction(roster: str) -> None:
    array = _handheld(roster)
    positions, bases = handheld_ut_single_v_geometry(roster)
    torch.testing.assert_close(
        array.ant_pos,
        torch.tensor(positions, dtype=array.ant_pos.dtype),
        atol=1e-15,
        rtol=1e-15,
    )
    torch.testing.assert_close(
        array.port_basis,
        torch.tensor(bases, dtype=array.port_basis.dtype).transpose(-1, -2),
        atol=1e-15,
        rtol=1e-15,
    )


@pytest.mark.native_sionna
@pytest.mark.parametrize("roster", ("tr38901-4", "tr38901"))
@pytest.mark.parametrize("angles", _ANGLES_DEG)
def test_handheld_step_11_fields_match_independent_port_geometry(
    roster: str,
    angles: dict[str, float],
) -> None:
    generator = _generator(roster)
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
    tx_field = _vh_basis((0.0, 0.0, 0.0), transmit_direction)[0]
    _, bases = handheld_ut_single_v_geometry(roster)

    for rx_port, basis in enumerate(bases):
        rx_field = handheld_ut_single_v_field_gcs(
            receive_direction,
            endpoint_orientation_rad=_UT_ORIENTATION,
            port_basis_device=basis,
        )
        expected = sum(
            rx_field[row] * _JONES[row][column] * tx_field[column]
            for row in range(2)
            for column in range(2)
        )
        assert complex(field[rx_port, 0]) == pytest.approx(expected, abs=2e-12)


@pytest.mark.native_sionna
@pytest.mark.parametrize(("roster", "port_count"), (("tr38901-4", 4), ("tr38901", 8)))
def test_cdl_adapter_uses_physical_handheld_ut_array(roster: str, port_count: int) -> None:
    ut_config = CDLHandheldUTArrayConfig(
        polarization="single",
        polarization_type="V",
        antenna_locations=roster,
        antenna_pattern="38.901-handheld",
    )
    bs_config = CDLArrayConfig()
    values = {
        "model": "A",
        "seed": 2_818_991,
        "carrier_frequency_hz": _CARRIER_FREQUENCY_HZ,
        "subcarrier_spacing_hz": 30_000.0,
        "num_subcarriers": 8,
        "delay_spread_s": 100e-9,
        "batch_size": 2,
        "num_time_steps": 2,
        "sampling_frequency_hz": 1.0 / 0.7e-3,
        "ut_array": ut_config,
        "bs_array": bs_config,
        "direction": "downlink",
        "ut_orientation_rad": _UT_ORIENTATION,
        "bs_orientation_rad": (0.0, 0.0, 0.0),
        "ut_velocity_mps": (10.0, 20.0, 20.0),
        "spec_version": "19.2",
        "device": "cpu",
    }
    channel = generate_cdl_frequency_response(**values)

    direct_model = CDL(
        model="A",
        delay_spread=100e-9,
        carrier_frequency=_CARRIER_FREQUENCY_HZ,
        ut_array=HandheldUTArray(
            **asdict(ut_config),
            carrier_frequency=_CARRIER_FREQUENCY_HZ,
            device="cpu",
        ),
        bs_array=PanelArray(
            **asdict(bs_config),
            carrier_frequency=_CARRIER_FREQUENCY_HZ,
            device="cpu",
        ),
        direction="downlink",
        ut_orientation=_UT_ORIENTATION,
        bs_orientation=(0.0, 0.0, 0.0),
        ut_velocity=(10.0, 20.0, 20.0),
        min_speed=0.0,
        max_speed=0.0,
        normalize_delays=True,
        spec_version="19.2",
        device="cpu",
    )
    expected_h_path, expected_tau = call_with_sionna_seed(
        "cpu",
        values["seed"],
        lambda: direct_model(
            batch_size=2,
            num_time_steps=2,
            sampling_frequency=1.0 / 0.7e-3,
        ),
    )
    torch.testing.assert_close(channel.h_path, expected_h_path)
    torch.testing.assert_close(channel.tau_s, expected_tau)
    assert channel.h_path.shape[2] == port_count
    assert channel.h_path.shape[4] == 1
    assert channel.metadata["ut_array_kind"] == "handheld_ut"
    assert channel.metadata["bs_array_kind"] == "panel"
    assert channel.metadata["ut_array"]["antenna_locations"] == roster
    assert channel.metadata["rx_antenna_count"] == port_count
