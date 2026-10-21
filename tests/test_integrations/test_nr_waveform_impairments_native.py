"""Native PUSCH waveform synchronization impairment checks."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sionna")

from integrations.sionna.nr_pusch import SionnaPUSCHLSReceiver  # noqa: E402
from integrations.sionna.ofdm_impairments import (  # noqa: E402
    apply_integer_timing_offset,
    apply_residual_cfo,
    estimate_integer_timing_offset_from_cp,
    estimate_residual_cfo_from_cp,
)
from receivers.nr import NRPUSCHConfig  # noqa: E402


@pytest.mark.native_sionna
def test_ofdm_roundtrip_and_residual_cfo_timing_errors_reach_dmrs_estimator():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    from sionna.phy.ofdm import OFDMDemodulator, OFDMModulator

    receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(n_size_bwp=2, subcarrier_spacing_khz=30),
        device="cuda:0",
    )
    grid = receiver.resource_grid
    modulator = OFDMModulator(
        cyclic_prefix_length=grid.cyclic_prefix_length,
        device="cuda:0",
    )
    demodulator = OFDMDemodulator(
        fft_size=grid.fft_size,
        l_min=0,
        cyclic_prefix_length=grid.cyclic_prefix_length,
        device="cuda:0",
    )
    transmit_grid, _ = receiver.transmit(16, seed=821)
    waveform = modulator(transmit_grid)
    recovered_grid = demodulator(waveform)
    torch.testing.assert_close(recovered_grid, transmit_grid, atol=1e-5, rtol=1e-5)

    sample_rate_hz = grid.fft_size * receiver.config.subcarrier_spacing_khz * 1_000
    cfo_waveform = apply_residual_cfo(
        waveform,
        cfo_hz=0.05 * receiver.config.subcarrier_spacing_khz * 1_000,
        sampling_frequency_hz=sample_rate_hz,
    )
    timing_waveform = apply_integer_timing_offset(waveform, offset_samples=1)
    cfo_grid = demodulator(cfo_waveform)
    timing_grid = demodulator(timing_waveform)
    cfo_estimate_hz = estimate_residual_cfo_from_cp(
        cfo_waveform,
        fft_size=grid.fft_size,
        cyclic_prefix_length=grid.cyclic_prefix_length,
        sampling_frequency_hz=sample_rate_hz,
    )
    corrected_waveform = apply_residual_cfo(
        cfo_waveform,
        cfo_hz=-cfo_estimate_hz,
        sampling_frequency_hz=sample_rate_hz,
    )
    corrected_grid = demodulator(corrected_waveform)

    max_timing_offset = min(3, grid.cyclic_prefix_length - 1)
    timing_offset = min(2, max_timing_offset)
    capture = torch.cat((waveform, torch.zeros_like(waveform[..., :max_timing_offset])), dim=-1)
    delayed_capture = apply_integer_timing_offset(capture, offset_samples=timing_offset)
    acquired_timing = estimate_integer_timing_offset_from_cp(
        delayed_capture,
        fft_size=grid.fft_size,
        cyclic_prefix_length=grid.cyclic_prefix_length,
        max_offset_samples=max_timing_offset,
    )
    corrected_capture = apply_integer_timing_offset(
        delayed_capture, offset_samples=-int(acquired_timing[0].item())
    )
    timing_corrected_grid = demodulator(corrected_capture[..., : waveform.shape[-1]])

    def estimated_channel_nmse(observed_grid):
        observation = observed_grid[:, 0, 0].unsqueeze(1).unsqueeze(1)
        estimate = receiver.estimate(observation, torch.tensor(0.0, device="cuda:0")).h_hat
        return torch.mean(torch.abs(estimate - 1.0) ** 2)

    ideal_nmse = estimated_channel_nmse(recovered_grid)
    cfo_nmse = estimated_channel_nmse(cfo_grid)
    timing_nmse = estimated_channel_nmse(timing_grid)
    assert ideal_nmse < 1e-8
    assert cfo_nmse > 1e-5
    assert timing_nmse > 1e-5
    assert torch.mean(torch.abs(cfo_grid - recovered_grid) ** 2) > 1e-4
    assert torch.mean(torch.abs(timing_grid - recovered_grid) ** 2) > 1e-4
    assert torch.max(torch.abs(cfo_estimate_hz - 0.05 * 30_000.0)) < 1.0
    torch.testing.assert_close(corrected_grid, transmit_grid, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(
        acquired_timing,
        torch.full_like(acquired_timing, timing_offset),
    )
    torch.testing.assert_close(timing_corrected_grid, transmit_grid, atol=1e-4, rtol=1e-4)


@pytest.mark.native_sionna
def test_waveform_impairment_helpers_preserve_shape_and_validate_units():
    waveform = torch.ones((2, 32), dtype=torch.complex64)
    delayed = apply_integer_timing_offset(waveform, offset_samples=3)
    advanced = apply_integer_timing_offset(waveform, offset_samples=-3)
    cfo = apply_residual_cfo(waveform, cfo_hz=100.0, sampling_frequency_hz=1_000.0)

    assert delayed.shape == waveform.shape
    assert advanced.shape == waveform.shape
    assert cfo.shape == waveform.shape
    assert torch.count_nonzero(delayed[..., :3]) == 0
    assert torch.count_nonzero(advanced[..., -3:]) == 0
    assert torch.allclose(torch.abs(cfo), torch.ones_like(torch.abs(cfo)))
    with pytest.raises(ValueError, match="positive"):
        apply_residual_cfo(waveform, cfo_hz=100.0, sampling_frequency_hz=0.0)
    with pytest.raises(ValueError, match="smaller"):
        apply_integer_timing_offset(waveform, offset_samples=32)


@pytest.mark.native_sionna
def test_pusch_frequency_grid_decoder_has_correct_high_vs_low_snr_trend():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    from sionna.phy.nr import PUSCHReceiver

    transmitter = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(n_size_bwp=2, subcarrier_spacing_khz=30),
        device="cuda:0",
    )
    tx_grid, tx_bits = transmitter.transmit(32, seed=904)
    decoder = PUSCHReceiver(
        transmitter._transmitter,
        return_tb_crc_status=True,
        input_domain="freq",
        device="cuda:0",
    )
    torch.manual_seed(905)
    noise = torch.complex(
        torch.randn(tx_grid.shape, device="cuda:0"), torch.randn(tx_grid.shape, device="cuda:0")
    )

    def decode_at_variance(noise_variance: float):
        noisy_grid = tx_grid + noise * torch.sqrt(torch.tensor(noise_variance / 2, device="cuda:0"))
        return decoder(noisy_grid, torch.tensor(noise_variance, device="cuda:0"))

    high_snr_bits, high_snr_crc = decode_at_variance(1e-3)
    low_snr_bits, low_snr_crc = decode_at_variance(1.0)

    high_snr_ber = torch.mean((high_snr_bits != tx_bits).float())
    low_snr_ber = torch.mean((low_snr_bits != tx_bits).float())
    high_snr_bler = torch.mean((~high_snr_crc).float())
    low_snr_bler = torch.mean((~low_snr_crc).float())

    assert high_snr_bits.shape == tx_bits.shape
    assert high_snr_ber < low_snr_ber
    assert high_snr_bler < low_snr_bler


@pytest.mark.native_sionna
def test_time_domain_two_tap_pusch_waveform_decodes_and_tracks_awgn_trend():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    from sionna.phy.nr import PUSCHReceiver
    from sionna.phy.ofdm import OFDMDemodulator, OFDMModulator

    tx_receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(n_size_bwp=2, subcarrier_spacing_khz=30),
        device="cuda:0",
    )
    tx_grid, tx_bits = tx_receiver.transmit(96, seed=2906)
    grid = tx_receiver.resource_grid
    modulator = OFDMModulator(
        cyclic_prefix_length=grid.cyclic_prefix_length,
        device="cuda:0",
    )
    demodulator = OFDMDemodulator(
        fft_size=grid.fft_size,
        l_min=0,
        cyclic_prefix_length=grid.cyclic_prefix_length,
        device="cuda:0",
    )
    waveform = modulator(tx_grid)
    delay_samples = 3
    delayed = torch.cat(
        (torch.zeros_like(waveform[..., :delay_samples]), waveform[..., :-delay_samples]), dim=-1
    )
    channel_waveform = (0.85 + 0.0j) * waveform + (0.28 + 0.16j) * delayed
    decoder = PUSCHReceiver(
        tx_receiver._transmitter,
        return_tb_crc_status=True,
        input_domain="freq",
        device="cuda:0",
    )
    torch.manual_seed(2907)
    unit_noise = torch.complex(
        torch.randn(channel_waveform.shape, device="cuda:0"),
        torch.randn(channel_waveform.shape, device="cuda:0"),
    ) / (2**0.5)

    def decode(noise_variance: float):
        received_waveform = channel_waveform + unit_noise * noise_variance**0.5
        received_grid = demodulator(received_waveform)
        decoded_bits, crc_ok = decoder(received_grid, torch.tensor(noise_variance, device="cuda:0"))
        ber = torch.mean((decoded_bits != tx_bits).float())
        bler = torch.mean((~crc_ok).float())
        return ber, bler

    high_snr_ber, high_snr_bler = decode(1e-3)
    low_snr_ber, low_snr_bler = decode(0.5)

    assert high_snr_ber < low_snr_ber
    assert high_snr_bler < low_snr_bler
