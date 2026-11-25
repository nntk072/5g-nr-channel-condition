"""Native Sionna PUSCH channel estimation, interpolation and noise checks."""

from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sionna")

from integrations.sionna.noise import proper_complex_awgn  # noqa: E402
from integrations.sionna.nr_pusch import SionnaPUSCHLSReceiver  # noqa: E402
from receivers.nr import NRPUSCHConfig  # noqa: E402
from tests.test_integrations.nr_pusch_native_support import (  # noqa: E402
    _InterpolationConfig,
)


@pytest.mark.native_sionna
@pytest.mark.parametrize("interpolation_type", ("nn", "lin", "lin_time_avg"))
def test_pusch_decoder_uses_the_configured_channel_estimator(interpolation_type: str) -> None:
    receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(
            n_size_bwp=2,
            subcarrier_spacing_khz=30,
            interpolation_type=interpolation_type,
        ),
        device="cpu",
    )
    transmitted_grid, _ = receiver.transmit(batch_size=2, seed=921)

    _, crc_ok = receiver.decode_pusch(transmitted_grid, torch.tensor(1e-6))

    assert bool(crc_ok.all())
    assert receiver._decoder is not None
    assert receiver._decoder._channel_estimator is receiver._estimator


@pytest.mark.native_sionna
def test_type2_four_layer_estimator_and_decoder_recover_known_flat_mimo_channel() -> None:
    """Estimate and decode four layers through a known flat 4x4 MIMO channel."""
    from sionna.phy.nr import PUSCHReceiver

    receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(
            n_size_bwp=4,
            subcarrier_spacing_khz=30,
            num_antenna_ports=4,
            num_layers=4,
            dmrs_config_type=2,
            dmrs_length=1,
            dmrs_additional_position=0,
            num_cdm_groups_without_data=2,
        ),
        device="cpu",
    )
    transmitted_grid, transmitted_bits = receiver.transmit(batch_size=1, seed=7)
    channel = torch.tensor(
        [
            [1.0 + 0.0j, 0.2 + 0.1j, 0.0 - 0.1j, 0.3 + 0.0j],
            [0.0 + 0.1j, 0.8 + 0.0j, 0.2 + 0.0j, 0.1 - 0.1j],
            [0.2 + 0.0j, 0.0 + 0.1j, 0.9 + 0.0j, 0.15 + 0.0j],
            [0.05 + 0.0j, 0.2 + 0.0j, 0.0 + 0.1j, 1.1 + 0.0j],
        ],
        dtype=torch.complex64,
    )
    received_grid = torch.einsum("rl,btlsf->btrsf", channel, transmitted_grid)

    estimate = receiver.estimate(received_grid, noise_variance=0.0)
    estimated_channel = estimate.h_hat[0, 0, :, 0]

    assert tuple(estimated_channel.shape) == (4, 4, 14, 48)
    torch.testing.assert_close(
        estimated_channel,
        channel[:, :, None, None].expand_as(estimated_channel),
        rtol=0,
        atol=1e-6,
    )

    channel_truth = channel.reshape(1, 1, 4, 1, 4, 1, 1).expand(
        1,
        1,
        4,
        1,
        4,
        receiver.resource_grid.num_ofdm_symbols,
        receiver.resource_grid.fft_size,
    )
    decoder = PUSCHReceiver(
        receiver._transmitter,
        channel_estimator="perfect",
        return_tb_crc_status=True,
        input_domain="freq",
        device="cpu",
    )
    decoded_bits, crc_ok = decoder(
        received_grid,
        torch.tensor(1e-7),
        h=channel_truth,
    )
    torch.testing.assert_close(decoded_bits, transmitted_bits)
    assert bool(crc_ok.all())


@pytest.mark.native_sionna
@pytest.mark.parametrize(
    ("subcarrier_spacing_khz", "n_size_bwp"),
    ((15, 2), (30, 6), (60, 12), (120, 24)),
)
def test_pusch_dmrs_estimator_recovers_frequency_selective_noiseless_channel_on_cpu(
    subcarrier_spacing_khz: int, n_size_bwp: int
) -> None:
    receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(
            n_size_bwp=n_size_bwp,
            subcarrier_spacing_khz=subcarrier_spacing_khz,
        ),
        device="cpu",
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
def test_pusch_linear_dmrs_interpolation_tracks_analytical_two_tap_channel_on_all_res():
    """Check linear DM-RS interpolation against known delay-domain channel truth."""
    receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(
            n_size_bwp=4,
            subcarrier_spacing_khz=30,
            interpolation_type="lin",
        ),
        device="cpu",
    )
    grid = receiver.resource_grid
    transmitted_grid, _ = receiver.transmit(batch_size=1, seed=5_302)
    centered_bins = torch.arange(grid.fft_size, dtype=torch.float32) - grid.fft_size // 2
    direct = 0.9 + 0.1j
    echo = 0.25 - 0.18j
    delay_samples = 3
    channel = direct + echo * torch.exp(
        -2j * torch.pi * centered_bins * delay_samples / grid.fft_size
    )
    received_grid = transmitted_grid[:, 0, 0].unsqueeze(1).unsqueeze(1) * channel.reshape(
        1, 1, 1, 1, grid.fft_size
    )

    estimate = receiver.estimate(received_grid, noise_variance=0.0).h_hat
    expected = channel.reshape(1, 1, 1, 1, 1, 1, grid.fft_size).expand_as(estimate)
    normalized_mse = torch.mean(torch.abs(estimate - expected) ** 2) / torch.mean(
        torch.abs(expected) ** 2
    )
    nearest_receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(
            n_size_bwp=4,
            subcarrier_spacing_khz=30,
            interpolation_type="nn",
        ),
        device="cpu",
    )
    nearest_estimate = nearest_receiver.estimate(received_grid, noise_variance=0.0).h_hat
    nearest_nmse = torch.mean(torch.abs(nearest_estimate - expected) ** 2) / torch.mean(
        torch.abs(expected) ** 2
    )

    assert tuple(estimate.shape[-2:]) == (grid.num_ofdm_symbols, grid.fft_size)
    assert float(normalized_mse) < 0.02
    assert float(normalized_mse) < float(nearest_nmse)


@pytest.mark.native_sionna
@pytest.mark.parametrize(
    (
        "dmrs_config_type",
        "dmrs_length",
        "dmrs_additional_position",
        "delay_samples",
    ),
    (
        (1, 1, 0, 3),
        (1, 1, 1, 3),
        (1, 2, 0, 5),
        (2, 1, 0, 5),
        (2, 1, 1, 5),
        (2, 2, 0, 5),
    ),
)
@pytest.mark.parametrize("complex_noise_variance", (0.01, 0.05, 0.2))
def test_pusch_interpolation_bias_variance_tradeoff_matches_monte_carlo(
    dmrs_config_type: int,
    dmrs_length: int,
    dmrs_additional_position: int,
    delay_samples: int,
    complex_noise_variance: float,
) -> None:
    """Compare interpolation bias and reported uncertainty on paired channels."""
    common_config: _InterpolationConfig = {
        "n_size_bwp": 4,
        "subcarrier_spacing_khz": 30,
        "dmrs_config_type": dmrs_config_type,
        "dmrs_length": dmrs_length,
        "dmrs_additional_position": dmrs_additional_position,
        "num_cdm_groups_without_data": 1,
    }
    linear_receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(**common_config, interpolation_type="lin"), device="cpu"
    )
    nearest_receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(**common_config, interpolation_type="nn"), device="cpu"
    )
    assert (
        linear_receiver.metadata["error_variance_semantics"]
        == "sionna_linear_interpolated_marginal_variance_not_exact_covariance"
    )
    assert (
        nearest_receiver.metadata["error_variance_semantics"]
        == "sionna_nearest_neighbor_pilot_error_variance"
    )
    batch_size = 1024
    transmitted_grid, _ = linear_receiver.transmit(batch_size, seed=73491)
    fft_size = linear_receiver.resource_grid.fft_size
    centered_bins = torch.arange(fft_size, dtype=torch.float32) - fft_size // 2
    channel = (
        0.9
        + 0.1j
        + (0.25 - 0.18j) * torch.exp(-2j * torch.pi * centered_bins * delay_samples / fft_size)
    )
    clean_grid = transmitted_grid[:, 0, 0].unsqueeze(1).unsqueeze(1) * channel.reshape(
        1, 1, 1, 1, fft_size
    )
    noise_generator = torch.Generator().manual_seed(23917)
    noise = (
        torch.complex(
            torch.randn(clean_grid.shape, generator=noise_generator),
            torch.randn(clean_grid.shape, generator=noise_generator),
        )
        * (complex_noise_variance / 2.0) ** 0.5
    )

    linear_noisy = linear_receiver.estimate(clean_grid + noise, complex_noise_variance)
    nearest_noisy = nearest_receiver.estimate(clean_grid + noise, complex_noise_variance)
    linear_reference = linear_receiver.estimate(clean_grid[:1], 0.0).h_hat[0, 0, 0, 0, 0]
    nearest_reference = nearest_receiver.estimate(clean_grid[:1], 0.0).h_hat[0, 0, 0, 0, 0]
    true_channel = channel.reshape(1, fft_size).expand_as(linear_reference)
    linear_bias_nmse = torch.mean(torch.abs(linear_reference - true_channel) ** 2) / torch.mean(
        torch.abs(true_channel) ** 2
    )
    nearest_bias_nmse = torch.mean(torch.abs(nearest_reference - true_channel) ** 2) / torch.mean(
        torch.abs(true_channel) ** 2
    )

    ratios = []
    for noisy_estimate in (linear_noisy, nearest_noisy):
        samples = noisy_estimate.h_hat[:, 0, 0, 0, 0]
        empirical_mean = samples.mean(dim=0)
        empirical_variance = (samples - empirical_mean).abs().square().mean(dim=0)
        predicted_variance = noisy_estimate.error_variance[0, 0, 0, 0, 0]
        valid = predicted_variance > 0
        reference = linear_reference if noisy_estimate is linear_noisy else nearest_reference
        mean_bias = (empirical_mean - reference).abs()
        bias_tolerance = 6.0 * torch.sqrt(predicted_variance / batch_size)
        assert bool(valid.any())
        assert bool((mean_bias[valid] < bias_tolerance[valid]).all())
        ratios.append(empirical_variance[valid].sum() / predicted_variance[valid].sum())

    # The linear estimate should reduce two-tap channel bias in this bounded
    # resolution regime; nearest-neighbor variance remains the calibrated mode.
    assert float(linear_bias_nmse) < float(nearest_bias_nmse)
    assert float(ratios[0]) <= 1.05
    assert float(ratios[0]) >= 0.6
    assert float(ratios[1]) == pytest.approx(1.0, abs=0.12)


@pytest.mark.native_sionna
@pytest.mark.parametrize(
    ("doppler_hz", "dmrs_additional_position"),
    ((35.0, 1), (350.0, 1), (35.0, 2), (350.0, 2)),
)
def test_pusch_interpolation_tracks_cp_timed_doppler_with_declared_variance(
    doppler_hz: float, dmrs_additional_position: int
) -> None:
    """Check temporal interpolation on a symbol-static Doppler CFR model."""
    from integrations.sionna.nr_timing import nr_normal_cp_symbol_start_times

    common_config: _InterpolationConfig = {
        "n_size_bwp": 4,
        "subcarrier_spacing_khz": 30,
        "dmrs_additional_position": dmrs_additional_position,
    }
    linear_receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(**common_config, interpolation_type="lin"), device="cpu"
    )
    nearest_receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(**common_config, interpolation_type="nn"), device="cpu"
    )
    batch_size = 1024
    noise_variance = 0.05
    transmitted_grid, _ = linear_receiver.transmit(batch_size, seed=88107)
    fft_size = linear_receiver.resource_grid.fft_size
    centered_bins = torch.arange(fft_size, dtype=torch.float32) - fft_size // 2
    symbol_times = torch.tensor(nr_normal_cp_symbol_start_times(30_000.0, 14), dtype=torch.float32)
    delayed_echo = (0.25 - 0.18j) * torch.exp(-2j * torch.pi * centered_bins * 3 / fft_size)
    channel = (0.9 + 0.1j) + torch.exp(2j * torch.pi * doppler_hz * symbol_times).reshape(
        14, 1
    ) * delayed_echo.reshape(1, fft_size)
    clean_grid = transmitted_grid[:, 0, 0].unsqueeze(1).unsqueeze(1) * channel.reshape(
        1, 1, 1, 14, fft_size
    )
    noise_generator = torch.Generator().manual_seed(19733)
    noise = (
        torch.complex(
            torch.randn(clean_grid.shape, generator=noise_generator),
            torch.randn(clean_grid.shape, generator=noise_generator),
        )
        * (noise_variance / 2.0) ** 0.5
    )

    linear_noisy = linear_receiver.estimate(clean_grid + noise, noise_variance)
    nearest_noisy = nearest_receiver.estimate(clean_grid + noise, noise_variance)
    linear_reference = linear_receiver.estimate(clean_grid[:1], 0.0).h_hat[0, 0, 0, 0, 0]
    nearest_reference = nearest_receiver.estimate(clean_grid[:1], 0.0).h_hat[0, 0, 0, 0, 0]
    linear_nmse = torch.mean(torch.abs(linear_reference - channel) ** 2) / torch.mean(
        torch.abs(channel) ** 2
    )
    nearest_nmse = torch.mean(torch.abs(nearest_reference - channel) ** 2) / torch.mean(
        torch.abs(channel) ** 2
    )

    ratios = []
    for noisy_estimate, reference in (
        (linear_noisy, linear_reference),
        (nearest_noisy, nearest_reference),
    ):
        samples = noisy_estimate.h_hat[:, 0, 0, 0, 0]
        empirical_mean = samples.mean(dim=0)
        empirical_variance = (samples - empirical_mean).abs().square().mean(dim=0)
        predicted_variance = noisy_estimate.error_variance[0, 0, 0, 0, 0]
        valid = predicted_variance > 0
        mean_bias = (empirical_mean - reference).abs()
        bias_tolerance = 6.0 * torch.sqrt(predicted_variance / batch_size)
        assert bool(valid.any())
        assert bool((mean_bias[valid] < bias_tolerance[valid]).all())
        ratios.append(empirical_variance[valid].sum() / predicted_variance[valid].sum())

    assert float(linear_nmse) < float(nearest_nmse)
    assert 0.6 <= float(ratios[0]) <= 1.05
    assert float(ratios[1]) == pytest.approx(1.0, abs=0.12)


@pytest.mark.native_sionna
@pytest.mark.parametrize(
    ("dmrs_config_type", "dmrs_length", "num_cdm_groups_without_data", "n_size_bwp"),
    (
        (1, 1, 1, 2),
        (1, 1, 2, 4),
        (2, 1, 1, 4),
        (2, 1, 2, 6),
        (1, 2, 1, 4),
        (2, 2, 1, 4),
    ),
)
@pytest.mark.parametrize("complex_noise_variance", (0.01, 0.05, 0.2))
def test_pusch_ls_pilot_bias_and_variance_match_analysis_across_dmrs_and_snr(
    dmrs_config_type: int,
    dmrs_length: int,
    num_cdm_groups_without_data: int,
    n_size_bwp: int,
    complex_noise_variance: float,
) -> None:
    """Check all pilot REs against their analytical LS variance across regimes."""
    receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(
            n_size_bwp=n_size_bwp,
            subcarrier_spacing_khz=30,
            dmrs_config_type=dmrs_config_type,
            dmrs_length=dmrs_length,
            num_cdm_groups_without_data=num_cdm_groups_without_data,
            interpolation_type="nn",
        ),
        device="cpu",
    )
    batch_size = 1024
    transmit_grid, _ = receiver.transmit(batch_size, seed=4801)
    channel = torch.tensor(0.6 - 0.2j, dtype=torch.complex64)
    clean_grid = transmit_grid[:, 0, 0].unsqueeze(1).unsqueeze(1) * channel
    noise_generator = torch.Generator().manual_seed(15)
    noise = (
        torch.complex(
            torch.randn(clean_grid.shape, generator=noise_generator),
            torch.randn(clean_grid.shape, generator=noise_generator),
        )
        * (complex_noise_variance / 2.0) ** 0.5
    )
    estimate = receiver.estimate(clean_grid + noise, complex_noise_variance)
    pilot_mask = receiver.pilot_mask[0, 0].bool()
    samples = estimate.h_hat[:, 0, 0, 0, 0][:, pilot_mask]
    predicted_variance = estimate.error_variance[0, 0, 0, 0, 0][pilot_mask]
    empirical_bias = (samples.mean(dim=0) - channel).abs()
    empirical_variance = (samples - samples.mean(dim=0)).abs().square().mean()
    mean_predicted_variance = predicted_variance.mean()
    bias_tolerance = 5.0 * torch.sqrt(predicted_variance / batch_size)

    assert samples.shape[1] == int(pilot_mask.sum())
    assert bool(torch.isfinite(predicted_variance).all())
    assert bool((predicted_variance > 0).all())
    torch.testing.assert_close(
        predicted_variance,
        torch.full_like(
            predicted_variance,
            complex_noise_variance / (2 * dmrs_length * num_cdm_groups_without_data),
        ),
        rtol=1e-6,
        atol=1e-7,
    )
    assert bool((empirical_bias < bias_tolerance).all())
    assert float(empirical_variance / mean_predicted_variance) == pytest.approx(1.0, abs=0.12)


@pytest.mark.native_sionna
def test_pusch_dmrs_uncertainty_tracks_distinct_per_batch_noise_variances():
    """Check batch noise broadcasting against realized pilot-estimation errors."""
    receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(n_size_bwp=2, subcarrier_spacing_khz=30), device="cpu"
    )
    batch_per_level = 256
    noise_levels = torch.tensor((0.01, 0.05, 0.2), dtype=torch.float32)
    noise_variance = noise_levels.repeat_interleave(batch_per_level)
    batch_size = noise_variance.numel()
    transmit_grid, _ = receiver.transmit(batch_size, seed=58_311)
    channel = torch.tensor(0.63 - 0.27j, dtype=torch.complex64)
    clean_grid = transmit_grid[:, 0, 0].unsqueeze(1).unsqueeze(1) * channel
    generator = torch.Generator().manual_seed(91_227)
    white_noise = torch.complex(
        torch.randn(clean_grid.shape, generator=generator),
        torch.randn(clean_grid.shape, generator=generator),
    )
    received_grid = clean_grid + white_noise * torch.sqrt(
        noise_variance.reshape(batch_size, 1, 1, 1, 1) / 2
    )

    estimate = receiver.estimate(received_grid, noise_variance)
    pilot_mask = receiver.pilot_mask[0, 0].bool()
    channel_errors = estimate.h_hat[:, 0, 0, 0, 0][:, pilot_mask] - channel
    predicted_variance = estimate.error_variance[:, 0, 0, 0, 0][:, pilot_mask]

    assert estimate.error_variance.shape[0] == batch_size
    for level_index, expected_noise in enumerate(noise_levels):
        start = level_index * batch_per_level
        stop = start + batch_per_level
        empirical_variance = channel_errors[start:stop].abs().square().mean()
        predicted = predicted_variance[start:stop].mean()
        torch.testing.assert_close(
            predicted,
            expected_noise / 2,
            atol=1e-7,
            rtol=1e-6,
        )
        assert float(empirical_variance / predicted) == pytest.approx(1.0, abs=0.08)

    unsupported_per_re_variance = noise_variance[:1].reshape(1, 1, 1, 1, 1, 1, 1)
    with pytest.raises(ValueError, match=r"\[batch, rx, rx_ant\]"):
        receiver.estimate(received_grid[:1], unsupported_per_re_variance)


@pytest.mark.native_sionna
def test_sionna_lmmse_effective_noise_matches_analytical_correlated_noise_fixture():
    from sionna.phy.mimo import lmmse_equalizer

    num_samples = 32_768
    channel = torch.tensor(
        [[1.0 + 0.2j, 0.1 - 0.3j], [0.4 + 0.1j, 0.9 + 0.2j], [0.2 - 0.1j, -0.3 + 0.8j]],
        dtype=torch.complex64,
    )
    noise_covariance = torch.tensor(
        [
            [0.8 + 0j, 0.12 + 0.08j, 0.0 + 0j],
            [0.12 - 0.08j, 0.9 + 0j, 0.05 - 0.04j],
            [0.0 + 0j, 0.05 + 0.04j, 0.7 + 0j],
        ],
        dtype=torch.complex64,
    )
    cholesky = torch.linalg.cholesky(noise_covariance)
    generator = torch.Generator().manual_seed(6917)
    symbols = proper_complex_awgn(
        torch.zeros((num_samples, 2), dtype=torch.complex64), 1.0, generator=generator
    )
    white_noise = proper_complex_awgn(
        torch.zeros((num_samples, 3), dtype=torch.complex64), 1.0, generator=generator
    )
    noise = white_noise @ cholesky.transpose(-2, -1)
    received = symbols @ channel.transpose(-2, -1) + noise

    equalized, effective_variance = lmmse_equalizer(received, channel, noise_covariance)

    inverse_noise_covariance = torch.linalg.inv(noise_covariance)
    linear_mmse_filter = torch.linalg.solve(
        channel.mH @ inverse_noise_covariance @ channel + torch.eye(2, dtype=torch.complex64),
        channel.mH @ inverse_noise_covariance,
    )
    unbiased_filter = linear_mmse_filter / torch.diagonal(linear_mmse_filter @ channel).unsqueeze(
        -1
    )
    expected_equalized = received @ unbiased_filter.transpose(-2, -1)
    residual_channel = unbiased_filter @ channel - torch.eye(2, dtype=torch.complex64)
    analytical_error_covariance = (
        residual_channel @ residual_channel.mH
        + unbiased_filter @ noise_covariance @ unbiased_filter.mH
    )
    analytical_variance = torch.diagonal(analytical_error_covariance).real
    empirical_variance = (equalized - symbols).abs().square().mean(dim=0)

    torch.testing.assert_close(equalized, expected_equalized, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(effective_variance, analytical_variance, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(
        empirical_variance / analytical_variance,
        torch.ones_like(analytical_variance),
        atol=0.025,
        rtol=0,
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

    def estimate_nmse(noise_variance: float) -> Any:
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

    def estimate_nmse(noise_variance: float) -> Any:
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
