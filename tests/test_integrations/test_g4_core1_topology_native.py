"""CPU-native topology and port-axis checks for Core-1 TDL/CDL profiles."""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sionna.phy")

from integrations.sionna.cdl import CDLArrayConfig, generate_cdl_frequency_response  # noqa: E402
from integrations.sionna.tdl import generate_tdl_frequency_response  # noqa: E402

_TOPOLOGIES = ((1, 1), (2, 1), (1, 2), (2, 2), (4, 4))
_SUBCARRIER_SPACING_HZ = 30_000.0
_NUM_SUBCARRIERS = 8


@pytest.mark.native_sionna
@pytest.mark.parametrize("family", ("TDL", "CDL"))
@pytest.mark.parametrize("model", tuple("ABCDE"))
@pytest.mark.parametrize(("num_rx_ant", "num_tx_ant"), _TOPOLOGIES)
def test_core1_profile_topology_preserves_port_axes_and_path_cfr(
    family: str, model: str, num_rx_ant: int, num_tx_ant: int
) -> None:
    """Check profile generation and path-to-CFR mapping at Core-1 port counts."""
    batch_size = 2
    common: dict[str, Any] = {
        "model": model,
        "seed": 93_101,
        "carrier_frequency_hz": 3.5e9,
        "subcarrier_spacing_hz": _SUBCARRIER_SPACING_HZ,
        "num_subcarriers": _NUM_SUBCARRIERS,
        "batch_size": batch_size,
        "spec_version": "19.2",
        "device": "cpu",
    }
    channel: Any
    if family == "TDL":
        channel = generate_tdl_frequency_response(
            **common,
            delay_spread_s=100e-9,
            num_rx_ant=num_rx_ant,
            num_tx_ant=num_tx_ant,
        )
    else:
        channel = generate_cdl_frequency_response(
            **common,
            delay_spread_s=100e-9,
            ut_array=CDLArrayConfig(num_cols_per_panel=num_rx_ant),
            bs_array=CDLArrayConfig(num_cols_per_panel=num_tx_ant),
            direction="downlink",
        )

    num_paths = channel.h_path.shape[-2]
    assert channel.h_path.shape == (batch_size, 1, num_rx_ant, 1, num_tx_ant, num_paths, 1)
    assert channel.tau_s.shape == (batch_size, 1, 1, num_paths)
    assert channel.channel_frequency_response.shape == (
        batch_size,
        1,
        num_rx_ant,
        1,
        num_tx_ant,
        1,
        _NUM_SUBCARRIERS,
    )
    assert channel.metadata["rx_antenna_count"] == num_rx_ant
    assert channel.metadata["tx_antenna_count"] == num_tx_ant
    assert channel.metadata["device"] == "cpu"

    frequencies = (
        torch.arange(_NUM_SUBCARRIERS, dtype=channel.tau_s.dtype, device="cpu")
        - _NUM_SUBCARRIERS // 2
    ) * _SUBCARRIER_SPACING_HZ
    path_coefficients = channel.h_path.movedim(-2, -1).unsqueeze(-1)
    path_delays = channel.tau_s[:, :, None, :, None, None, :, None]
    phase = torch.exp(
        -2j * torch.pi * path_delays * frequencies.reshape(1, 1, 1, 1, 1, 1, 1, _NUM_SUBCARRIERS)
    )
    reconstructed = torch.sum(path_coefficients * phase, dim=-2)
    torch.testing.assert_close(channel.channel_frequency_response, reconstructed)
