"""Native checks for the reusable Sionna CDL-to-CFR adapter."""

from __future__ import annotations

from dataclasses import asdict

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sionna.phy")

from integrations.sionna.cdl import (  # noqa: E402
    CDLArrayConfig,
    generate_cdl_frequency_response,
)
from integrations.sionna.random import call_with_sionna_seed  # noqa: E402


@pytest.mark.native_sionna
@pytest.mark.parametrize("model", ["A", "D"])
@pytest.mark.parametrize("direction", ["downlink", "uplink"])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"] if torch.cuda.is_available() else ["cpu"])
def test_cdl_adapter_preserves_array_axes_reproducibility_and_cfr(
    model: str, direction: str, device: str
):
    ut_array = CDLArrayConfig(
        num_cols_per_panel=2,
        polarization="dual",
        polarization_type="VH",
    )
    bs_array = CDLArrayConfig(
        num_cols_per_panel=1,
        polarization="dual",
        polarization_type="cross",
    )
    values = {
        "model": model,
        "seed": 51_007,
        "carrier_frequency_hz": 3.5e9,
        "subcarrier_spacing_hz": 30_000.0,
        "num_subcarriers": 24,
        "delay_spread_s": 100e-9,
        "batch_size": 2,
        "num_time_steps": 3,
        "sampling_frequency_hz": 15_000.0,
        "ut_array": ut_array,
        "bs_array": bs_array,
        "direction": direction,
        "ut_velocity_mps": (3.0, 4.0, 0.0),
        "device": device,
    }
    channel = generate_cdl_frequency_response(**values)
    repeated = generate_cdl_frequency_response(**values)

    ut_antennas = 4
    bs_antennas = 2
    rx_antennas, tx_antennas = (
        (ut_antennas, bs_antennas) if direction == "downlink" else (bs_antennas, ut_antennas)
    )
    num_clusters = len(channel.metadata["nominal_cluster_delays_s"])
    assert channel.h_path.shape == (2, 1, rx_antennas, 1, tx_antennas, num_clusters, 3)
    assert channel.tau_s.shape == (2, 1, 1, num_clusters)
    assert channel.channel_frequency_response.shape == (
        2,
        1,
        rx_antennas,
        1,
        tx_antennas,
        3,
        24,
    )
    torch.testing.assert_close(channel.h_path, repeated.h_path)
    torch.testing.assert_close(channel.tau_s, repeated.tau_s)
    torch.testing.assert_close(
        channel.channel_frequency_response, repeated.channel_frequency_response
    )

    path_time = channel.h_path.movedim(-2, -1)
    frequencies = (torch.arange(24, dtype=channel.tau_s.dtype, device=device) - 12) * 30_000.0
    expected_cfr = torch.sum(
        path_time.unsqueeze(-1)
        * torch.exp(
            -2j
            * torch.pi
            * channel.tau_s[:, :, None, :, None, None, :].unsqueeze(-1)
            * frequencies.reshape(1, 1, 1, 1, 1, 1, 1, 24)
        ),
        dim=-2,
    )
    torch.testing.assert_close(channel.channel_frequency_response, expected_cfr)
    torch.testing.assert_close(
        channel.tau_s[0, 0, 0],
        torch.as_tensor(
            channel.metadata["nominal_cluster_delays_s"],
            dtype=channel.tau_s.dtype,
            device=device,
        ),
    )

    assert channel.metadata["direction"] == direction
    assert channel.metadata["geometric_los_label"] is None
    assert channel.metadata["geometric_los_label_provided"] is False
    assert channel.metadata["profile_los_component"] is (model in {"D", "E"})
    if model == "A":
        assert channel.metadata["profile_k_factor_linear"] is None
    else:
        assert channel.metadata["profile_k_factor_linear"] > 0.0
    assert channel.metadata["maximum_doppler_hz"] == pytest.approx(5.0 * 3.5e9 / 299_792_458.0)
    assert len(channel.metadata["parameter_table_sha256"]) == 64
    assert sum(channel.metadata["normalized_cluster_powers"]) == pytest.approx(1.0)


@pytest.mark.native_sionna
@pytest.mark.parametrize("model", ["A", "D"])
@pytest.mark.parametrize("direction", ["downlink", "uplink"])
@pytest.mark.parametrize("rotated_endpoint", ["ut", "bs"])
def test_cdl_adapter_applies_defined_array_orientations_on_cpu(
    model: str, direction: str, rotated_endpoint: str
):
    from sionna.phy.channel.tr38901 import CDL, PanelArray

    carrier_frequency_hz = 3.5e9
    ut_array = CDLArrayConfig(num_cols_per_panel=2)
    bs_array = CDLArrayConfig(num_cols_per_panel=4)
    ut_orientation_rad = (0.2, -0.1, 0.3)
    bs_orientation_rad = (-0.4, 0.25, 0.6)
    values = {
        "model": model,
        "seed": 71203,
        "carrier_frequency_hz": carrier_frequency_hz,
        "subcarrier_spacing_hz": 30_000.0,
        "num_subcarriers": 8,
        "delay_spread_s": 100e-9,
        "batch_size": 2,
        "num_time_steps": 1,
        "sampling_frequency_hz": 30_000.0,
        "ut_array": ut_array,
        "bs_array": bs_array,
        "direction": direction,
        "ut_orientation_rad": ut_orientation_rad,
        "bs_orientation_rad": bs_orientation_rad,
        "device": "cpu",
        "spec_version": "19.2",
    }
    channel = generate_cdl_frequency_response(**values)

    ut_array_object = PanelArray(
        **asdict(ut_array), carrier_frequency=carrier_frequency_hz, device="cpu"
    )
    bs_array_object = PanelArray(
        **asdict(bs_array), carrier_frequency=carrier_frequency_hz, device="cpu"
    )
    direct_model = CDL(
        model=model,
        delay_spread=100e-9,
        carrier_frequency=carrier_frequency_hz,
        ut_array=ut_array_object,
        bs_array=bs_array_object,
        direction=direction,
        ut_orientation=ut_orientation_rad,
        bs_orientation=bs_orientation_rad,
        min_speed=0.0,
        max_speed=0.0,
        spec_version="19.2",
        device="cpu",
    )
    expected_h_path, expected_tau_s = call_with_sionna_seed(
        "cpu",
        values["seed"],
        lambda: direct_model(
            batch_size=values["batch_size"],
            num_time_steps=values["num_time_steps"],
            sampling_frequency=values["sampling_frequency_hz"],
        ),
    )
    torch.testing.assert_close(channel.h_path, expected_h_path)
    torch.testing.assert_close(channel.tau_s, expected_tau_s)
    assert channel.metadata["ut_orientation_rad"] == list(ut_orientation_rad)
    assert channel.metadata["bs_orientation_rad"] == list(bs_orientation_rad)

    orientation_key = f"{rotated_endpoint}_orientation_rad"
    orientation = values[orientation_key]
    rotated = generate_cdl_frequency_response(
        **{**values, orientation_key: (orientation[0], orientation[1], 0.0)}
    )
    assert not torch.allclose(channel.h_path, rotated.h_path)
    torch.testing.assert_close(channel.tau_s, rotated.tau_s)
