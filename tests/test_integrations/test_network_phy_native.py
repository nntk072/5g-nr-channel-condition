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


@pytest.mark.native_sionna
def test_native_multi_node_rt_channel_matches_single_link_and_composes_sources():
    import mitsuba as mi

    mi.set_variant("llvm_ad_mono_polarized")
    rt = pytest.importorskip("sionna.rt")
    from types import SimpleNamespace

    from integrations.sionna.rt_to_nr import paths_to_nr_ofdm_channel

    grid = SimpleNamespace(fft_size=16, num_ofdm_symbols=4, subcarrier_spacing=30e3)
    solver = rt.PathSolver(deterministic=True)

    def make_scene(tx_positions, rx_positions):
        scene = rt.load_scene()
        scene.frequency = 3.5e9
        scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
        scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
        for index, position in enumerate(tx_positions):
            scene.add(rt.Transmitter(f"tx{index}", list(position)))
        for index, position in enumerate(rx_positions):
            scene.add(rt.Receiver(f"rx{index}", list(position)))
        return scene

    tx_positions = ((0.0, 0.0, 1.0), (0.0, 3.0, 1.0))
    rx_positions = ((10.0, 0.0, 1.0), (12.0, 3.0, 1.0))
    multi_scene = make_scene(tx_positions, rx_positions)
    multi_paths = solver(
        multi_scene,
        max_depth=0,
        los=True,
        specular_reflection=False,
        diffuse_reflection=False,
        refraction=False,
        diffraction=False,
        synthetic_array=True,
        seed=1605,
    )
    multi_channel = paths_to_nr_ofdm_channel(multi_paths, grid).channel

    single_scene = make_scene(tx_positions[:1], rx_positions[:1])
    single_paths = solver(
        single_scene,
        max_depth=0,
        los=True,
        specular_reflection=False,
        diffuse_reflection=False,
        refraction=False,
        diffraction=False,
        synthetic_array=True,
        seed=1605,
    )
    single_channel = paths_to_nr_ofdm_channel(single_paths, grid).channel
    torch.testing.assert_close(multi_channel[:, :1, :, :1], single_channel, rtol=2e-5, atol=1e-8)

    from integrations.sionna.network_phy import compose_downlink_resource_grids

    tx_grids = torch.ones((1, 2, 1, grid.num_ofdm_symbols, grid.fft_size), dtype=torch.complex64)
    power = (1.0, 1.0)
    joint = compose_downlink_resource_grids(multi_channel, tx_grids, power)
    individual = [
        compose_downlink_resource_grids(
            multi_channel[:, :, :, tx_index : tx_index + 1],
            tx_grids[:, tx_index : tx_index + 1],
            (power[tx_index],),
        ).received_grid
        for tx_index in range(2)
    ]
    torch.testing.assert_close(joint.received_grid, individual[0] + individual[1])
