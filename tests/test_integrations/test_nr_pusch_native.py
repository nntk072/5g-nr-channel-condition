import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sionna")

from integrations.sionna.nr_pusch import SionnaPUSCHLSReceiver  # noqa: E402
from receivers.nr import NRPUSCHConfig  # noqa: E402


@pytest.mark.native_sionna
def test_pusch_dmrs_estimator_recovers_frequency_selective_noiseless_channel_on_cpu():
    receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(n_size_bwp=6, subcarrier_spacing_khz=30), device="cpu"
    )
    grid = receiver.resource_grid
    transmit_grid, _ = receiver.transmit(2, seed=5302)
    fft_bins = torch.arange(grid.fft_size)
    # Distinct adjacent FFT-bin gains expose a one-bin or DC/guard mapping shift.
    channel_on_fft_bins = torch.where(
        (fft_bins % 2) == 0,
        torch.tensor(0.25 + 0.4j, dtype=torch.complex64),
        torch.tensor(0.8 - 0.15j, dtype=torch.complex64),
    )
    received_grid = transmit_grid[:, 0, 0].unsqueeze(1).unsqueeze(1) * channel_on_fft_bins.reshape(
        1, 1, 1, 1, grid.fft_size
    )

    observation = receiver.observe("cpu-noiseless-frequency-selective", received_grid, 0.0)
    effective_bins = torch.as_tensor(grid.effective_subcarrier_ind, dtype=torch.long)
    expected = channel_on_fft_bins.index_select(0, effective_bins)
    expected = expected.reshape(1, 1, 1, 1, 1, 1, -1).expand_as(observation.channel_estimate.h_hat)
    pilot_mask = observation.channel_estimate.pilot_mask.to(dtype=torch.bool).reshape(
        1, 1, 1, 1, 1, grid.num_ofdm_symbols, grid.num_effective_subcarriers
    )
    pilot_mask = pilot_mask.expand_as(observation.channel_estimate.h_hat)

    assert observation.fft_bin_indices == tuple(int(index) for index in effective_bins)
    assert grid.num_effective_subcarriers == grid.fft_size
    assert observation.fft_bin_indices == tuple(range(grid.fft_size))
    assert bool(pilot_mask.any())
    torch.testing.assert_close(
        observation.channel_estimate.h_hat[pilot_mask], expected[pilot_mask], atol=1e-5, rtol=1e-5
    )


@pytest.mark.native_sionna
def test_pusch_dmrs_estimator_runs_on_cuda_with_named_axes():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(n_size_bwp=2, subcarrier_spacing_khz=30),
        device="cuda:0",
    )
    grid = receiver.resource_grid
    pilot_mask = receiver.pilot_mask[0, 0].bool()
    pilot_symbols = grid.pilot_pattern.pilots[0, 0]
    expected_channel = torch.tensor(0.5 + 0.25j, device="cuda:0")
    received = torch.zeros(
        (2, 1, 2, grid.num_ofdm_symbols, grid.fft_size),
        dtype=torch.complex64,
        device="cuda:0",
    )
    received[..., pilot_mask] = pilot_symbols * expected_channel
    result = receiver.estimate(received, torch.tensor(1e-3, device="cuda:0"))
    observation = receiver.observe("cuda-flat-channel-001", received, torch.tensor(1e-3))

    assert result.h_hat.shape == (
        2,
        1,
        2,
        1,
        1,
        grid.num_ofdm_symbols,
        grid.num_effective_subcarriers,
    )
    assert result.h_hat.is_cuda
    assert result.error_variance.is_cuda
    assert result.pilot_mask.is_cuda
    assert observation.sample_id == "cuda-flat-channel-001"
    assert observation.fft_bin_indices == tuple(
        int(index) for index in grid.effective_subcarrier_ind
    )
    assert observation.model_inputs().received_grid.is_cuda
    assert bool(torch.isfinite(result.h_hat).all())
    assert bool(torch.isfinite(result.error_variance).all())
    assert result.error_variance.shape == (
        1,
        1,
        1,
        1,
        1,
        grid.num_ofdm_symbols,
        grid.num_effective_subcarriers,
    )
    torch.testing.assert_close(
        result.h_hat,
        expected_channel.expand_as(result.h_hat),
        atol=1e-5,
        rtol=1e-5,
    )
    with pytest.raises(ValueError, match="dimensions must be positive"):
        receiver.estimate(received[:, :0], torch.tensor(1e-3, device="cuda:0"))

    torch.manual_seed(7)
    clean_grid = torch.zeros(
        (16, 1, 2, grid.num_ofdm_symbols, grid.fft_size),
        dtype=torch.complex64,
        device="cuda:0",
    )
    clean_grid[..., pilot_mask] = pilot_symbols * expected_channel

    def estimate_nmse(noise_variance: float) -> torch.Tensor:
        variance = torch.tensor(noise_variance, device="cuda:0")
        real_noise = torch.randn(clean_grid.shape, device="cuda:0")
        imag_noise = torch.randn(clean_grid.shape, device="cuda:0")
        noisy_grid = clean_grid + torch.complex(real_noise, imag_noise) * torch.sqrt(variance / 2)
        estimate = receiver.estimate(noisy_grid, variance)
        return torch.mean(torch.abs(estimate.h_hat - expected_channel) ** 2)

    low_noise_nmse = estimate_nmse(1e-3)
    high_noise_nmse = estimate_nmse(1e-1)
    assert low_noise_nmse < high_noise_nmse


@pytest.mark.native_sionna
def test_pusch_transmitter_rayleigh_channel_and_estimator_close_the_loop():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    from sionna.phy.channel import OFDMChannel, RayleighBlockFading

    receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(n_size_bwp=2, subcarrier_spacing_khz=30),
        device="cuda:0",
    )
    resource_grid = receiver.resource_grid
    transmit_grid, _ = receiver._transmitter(8)
    channel = OFDMChannel(
        RayleighBlockFading(num_rx=1, num_rx_ant=2, num_tx=1, num_tx_ant=1),
        resource_grid,
        return_channel=True,
        device="cuda:0",
    )
    received_grid, channel_truth = channel(transmit_grid)
    channel_truth = channel_truth[:, :, :, :, 0, :, :].unsqueeze(4)
    noiseless_estimate = receiver.estimate(received_grid, torch.tensor(0.0, device="cuda:0"))

    assert noiseless_estimate.h_hat.shape == channel_truth.shape
    torch.testing.assert_close(noiseless_estimate.h_hat, channel_truth, atol=1e-5, rtol=1e-5)

    torch.manual_seed(11)

    def estimate_nmse(noise_variance: float) -> torch.Tensor:
        variance = torch.tensor(noise_variance, device="cuda:0")
        real_noise = torch.randn(received_grid.shape, device="cuda:0")
        imag_noise = torch.randn(received_grid.shape, device="cuda:0")
        noisy_grid = received_grid + torch.complex(real_noise, imag_noise) * torch.sqrt(
            variance / 2
        )
        estimate = receiver.estimate(noisy_grid, variance)
        return torch.mean(torch.abs(estimate.h_hat - channel_truth) ** 2) / torch.mean(
            torch.abs(channel_truth) ** 2
        )

    assert estimate_nmse(1e-3) < estimate_nmse(1e-1)
