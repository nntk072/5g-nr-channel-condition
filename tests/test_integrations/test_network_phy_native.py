"""Native Sionna PHY check that cochannel pilot reuse contaminates CSI."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sionna")

from integrations.sionna.network_phy import compose_downlink_resource_grids  # noqa: E402
from integrations.sionna.nr_pusch import SionnaPUSCHLSReceiver  # noqa: E402
from receivers.nr import NRPUSCHConfig  # noqa: E402


@pytest.mark.native_sionna
def test_reused_pusch_dmrs_estimates_sum_of_serving_and_interfering_channels():
    receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(n_size_bwp=6, subcarrier_spacing_khz=30),
        device="cpu",
    )
    one_user_grid, _ = receiver.transmit(1, seed=1604)
    one_user_grid = one_user_grid[:, :1]
    power_w_per_re = float(one_user_grid[:, 0].abs().square().mean().item())
    resource_grid = receiver.resource_grid
    symbols = resource_grid.num_ofdm_symbols
    fft_size = resource_grid.fft_size
    desired_channel = torch.ones((1, 1, 1, 1, 1, symbols, fft_size), dtype=torch.complex64)
    interfering_gain = torch.tensor(0.3 + 0.2j, dtype=torch.complex64)
    cochannel = torch.cat(
        (desired_channel, torch.full_like(desired_channel, interfering_gain)), dim=3
    )
    reused_grid = torch.cat((one_user_grid, one_user_grid), dim=1)

    desired_only = compose_downlink_resource_grids(
        desired_channel,
        one_user_grid,
        (power_w_per_re,),
    )
    both_users = compose_downlink_resource_grids(
        cochannel,
        reused_grid,
        (power_w_per_re, power_w_per_re),
    )
    desired_estimate = receiver.estimate(desired_only.received_grid, 0.0)
    contaminated_estimate = receiver.estimate(both_users.received_grid, 0.0)
    pilot_mask = receiver.pilot_mask[0, 0]
    desired_channel_hat = desired_estimate.h_hat[0, 0, 0, 0, 0]
    contaminated_channel_hat = contaminated_estimate.h_hat[0, 0, 0, 0, 0]

    torch.testing.assert_close(
        desired_channel_hat[pilot_mask],
        torch.ones_like(desired_channel_hat[pilot_mask]),
        rtol=2e-4,
        atol=2e-4,
    )
    torch.testing.assert_close(
        contaminated_channel_hat[pilot_mask],
        torch.full_like(contaminated_channel_hat[pilot_mask], 1.0 + interfering_gain),
        rtol=2e-4,
        atol=2e-4,
    )
    assert not torch.allclose(
        contaminated_channel_hat[pilot_mask],
        desired_channel_hat[pilot_mask],
        rtol=1e-3,
        atol=1e-3,
    )
