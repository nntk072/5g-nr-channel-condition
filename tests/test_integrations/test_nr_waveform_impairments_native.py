"""Native PUSCH waveform synchronization impairment checks."""

import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sionna")

from experiments.cp_margin import run_static_cp_margin_experiment  # noqa: E402
from integrations.sionna.nr_pusch import SionnaPUSCHLSReceiver  # noqa: E402
from integrations.sionna.ofdm_impairments import (  # noqa: E402
    apply_fractional_timing_offset,
    apply_integer_timing_offset,
    apply_iq_imbalance,
    apply_phase_noise,
    apply_residual_cfo,
    estimate_integer_timing_offset_from_cp,
    estimate_residual_cfo_from_cp,
    generate_wiener_phase_noise_rad,
)
from receivers.nr import NRPUSCHConfig  # noqa: E402


@pytest.mark.native_sionna
def test_pusch_diagonal_ofdm_channel_is_valid_inside_cp_and_fails_beyond_cp_on_cpu():
    from sionna.phy.ofdm import OFDMDemodulator, OFDMModulator

    receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(n_size_bwp=6, subcarrier_spacing_khz=30), device="cpu"
    )
    grid = receiver.resource_grid
    transmit_grid, _ = receiver.transmit(2, seed=34_017)
    modulator = OFDMModulator(
        cyclic_prefix_length=grid.cyclic_prefix_length,
        device="cpu",
    )
    demodulator = OFDMDemodulator(
        fft_size=grid.fft_size,
        l_min=0,
        cyclic_prefix_length=grid.cyclic_prefix_length,
        device="cpu",
    )
    waveform = modulator(transmit_grid)
    direct_gain = 0.85 + 0.0j
    echo_gain = 0.48 + 0.12j
    frequencies = torch.arange(grid.fft_size, dtype=torch.float32) - grid.fft_size // 2
    effective_bins = torch.as_tensor(grid.effective_subcarrier_ind, dtype=torch.long)

    def pass_through_two_tap(delay_samples: int):
        delayed = torch.cat(
            (torch.zeros_like(waveform[..., :delay_samples]), waveform[..., :-delay_samples]),
            dim=-1,
        )
        received = direct_gain * waveform + echo_gain * delayed
        received_grid = demodulator(received)
        response = direct_gain + echo_gain * torch.exp(
            -2j * torch.pi * frequencies * delay_samples / grid.fft_size
        )
        expected = transmit_grid * response.reshape(1, 1, 1, 1, grid.fft_size)
        return received_grid.index_select(-1, effective_bins), expected.index_select(
            -1, effective_bins
        )

    inside, inside_frequency_domain = pass_through_two_tap(grid.cyclic_prefix_length - 1)
    beyond, beyond_frequency_domain = pass_through_two_tap(grid.cyclic_prefix_length + 1)
    inside_nmse = torch.mean(torch.abs(inside - inside_frequency_domain) ** 2) / torch.mean(
        torch.abs(inside_frequency_domain) ** 2
    )
    beyond_nmse = torch.mean(torch.abs(beyond - beyond_frequency_domain) ** 2) / torch.mean(
        torch.abs(beyond_frequency_domain) ** 2
    )

    assert grid.num_effective_subcarriers == grid.fft_size
    assert float(inside_nmse) < 1e-6
    assert float(beyond_nmse) > float(inside_nmse) + 1e-3


@pytest.mark.native_sionna
def test_pusch_cpu_waveform_roundtrip_decodes_noiseless_payload():
    from sionna.phy.nr import PUSCHReceiver
    from sionna.phy.ofdm import OFDMDemodulator, OFDMModulator

    transmitter = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(n_size_bwp=6, subcarrier_spacing_khz=30),
        device="cpu",
    )
    grid = transmitter.resource_grid
    tx_grid, tx_bits = transmitter.transmit(8, seed=34_018)
    modulator = OFDMModulator(
        cyclic_prefix_length=grid.cyclic_prefix_length,
        device="cpu",
    )
    demodulator = OFDMDemodulator(
        fft_size=grid.fft_size,
        l_min=0,
        cyclic_prefix_length=grid.cyclic_prefix_length,
        device="cpu",
    )
    received_grid = demodulator(modulator(tx_grid))
    decoder = PUSCHReceiver(
        transmitter._transmitter,
        channel_estimator="perfect",
        return_tb_crc_status=True,
        input_domain="freq",
        device="cpu",
    )

    channel_truth = torch.ones(
        (
            tx_grid.shape[0],
            1,
            1,
            1,
            1,
            grid.num_ofdm_symbols,
            grid.fft_size,
        ),
        dtype=tx_grid.dtype,
    )
    # Keep the ideal waveform noise-free; the small positive detector variance avoids a singular LMMSE solve.
    rx_bits, tb_crc_status = decoder(received_grid, torch.tensor(1e-8), h=channel_truth)

    assert received_grid.shape == tx_grid.shape
    torch.testing.assert_close(received_grid, tx_grid, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(rx_bits, tx_bits)
    assert bool(tb_crc_status.all())


@pytest.mark.native_sionna
def test_composed_cpu_rf_impairments_increase_pusch_ls_channel_error():
    from sionna.phy.ofdm import OFDMDemodulator, OFDMModulator

    receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(n_size_bwp=2, subcarrier_spacing_khz=30),
        device="cpu",
    )
    grid = receiver.resource_grid
    tx_grid, _ = receiver.transmit(8, seed=34_019)
    modulator = OFDMModulator(
        cyclic_prefix_length=grid.cyclic_prefix_length,
        device="cpu",
    )
    demodulator = OFDMDemodulator(
        fft_size=grid.fft_size,
        l_min=0,
        cyclic_prefix_length=grid.cyclic_prefix_length,
        device="cpu",
    )
    sample_rate_hz = grid.fft_size * receiver.config.subcarrier_spacing_khz * 1_000.0
    channel_gain = 0.75 + 0.2j
    waveform = modulator(tx_grid)
    ideal_grid = demodulator(waveform * channel_gain)
    tx_iq = apply_iq_imbalance(
        waveform,
        gain_imbalance_db=0.5,
        phase_imbalance_rad=0.02,
    )
    through_channel = tx_iq * channel_gain
    with_cfo = apply_residual_cfo(
        through_channel,
        cfo_hz=0.05 * receiver.config.subcarrier_spacing_khz * 1_000.0,
        sampling_frequency_hz=sample_rate_hz,
    )
    phase_noise = generate_wiener_phase_noise_rad(
        waveform.shape[-1],
        sampling_frequency_hz=sample_rate_hz,
        linewidth_hz=25.0,
        generator=torch.Generator().manual_seed(34_020),
    )
    with_phase_noise = apply_phase_noise(with_cfo, phase_noise_rad=phase_noise)
    rx_iq = apply_iq_imbalance(
        with_phase_noise,
        gain_imbalance_db=-0.5,
        phase_imbalance_rad=-0.02,
    )
    impaired_waveform = apply_integer_timing_offset(rx_iq, offset_samples=1)
    impaired_grid = demodulator(impaired_waveform)

    def estimate_nmse(received_grid):
        receiver_grid = received_grid[:, 0, 0].unsqueeze(1).unsqueeze(1)
        estimate = receiver.estimate(receiver_grid, torch.tensor(1e-6)).h_hat
        expected = torch.full_like(estimate, channel_gain)
        return torch.mean(torch.abs(estimate - expected) ** 2) / abs(channel_gain) ** 2

    ideal_nmse = estimate_nmse(ideal_grid)
    impaired_nmse = estimate_nmse(impaired_grid)

    assert ideal_nmse < 1e-8
    assert impaired_nmse > ideal_nmse + 1e-2


@pytest.mark.native_sionna
def test_cpu_ofdm_roundtrip_and_residual_cfo_timing_acquisition_and_correction():
    from sionna.phy.ofdm import OFDMDemodulator, OFDMModulator

    receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(n_size_bwp=2, subcarrier_spacing_khz=30),
        device="cpu",
    )
    grid = receiver.resource_grid
    modulator = OFDMModulator(
        cyclic_prefix_length=grid.cyclic_prefix_length,
        device="cpu",
    )
    demodulator = OFDMDemodulator(
        fft_size=grid.fft_size,
        l_min=0,
        cyclic_prefix_length=grid.cyclic_prefix_length,
        device="cpu",
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
    fractional_waveform = apply_fractional_timing_offset(waveform, offset_samples=0.35)
    cfo_grid = demodulator(cfo_waveform)
    timing_grid = demodulator(timing_waveform)
    fractional_grid = demodulator(fractional_waveform)
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
        estimate = receiver.estimate(observation, torch.tensor(0.0, device="cpu")).h_hat
        return torch.mean(torch.abs(estimate - 1.0) ** 2)

    ideal_nmse = estimated_channel_nmse(recovered_grid)
    cfo_nmse = estimated_channel_nmse(cfo_grid)
    timing_nmse = estimated_channel_nmse(timing_grid)
    fractional_nmse = estimated_channel_nmse(fractional_grid)
    assert ideal_nmse < 1e-8
    assert cfo_nmse > 1e-5
    assert timing_nmse > 1e-5
    assert fractional_nmse > 1e-5
    assert torch.mean(torch.abs(cfo_grid - recovered_grid) ** 2) > 1e-4
    assert torch.mean(torch.abs(timing_grid - recovered_grid) ** 2) > 1e-4
    assert torch.mean(torch.abs(fractional_grid - recovered_grid) ** 2) > 1e-4
    assert torch.max(torch.abs(cfo_estimate_hz - 0.05 * 30_000.0)) < 1.0
    torch.testing.assert_close(corrected_grid, transmit_grid, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(
        acquired_timing,
        torch.full_like(acquired_timing, timing_offset),
    )
    torch.testing.assert_close(timing_corrected_grid, transmit_grid, atol=1e-4, rtol=1e-4)


@pytest.mark.native_sionna
def test_cp_cfo_acquisition_handles_known_in_cp_multipath_and_awgn():
    from sionna.phy.ofdm import OFDMModulator

    receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(n_size_bwp=2, subcarrier_spacing_khz=30), device="cpu"
    )
    grid = receiver.resource_grid
    transmit_grid, _ = receiver.transmit(16, seed=821)
    waveform = OFDMModulator(
        cyclic_prefix_length=grid.cyclic_prefix_length,
        device="cpu",
    )(transmit_grid)

    # A causal two-tap channel with its delayed path inside the cyclic prefix.
    path_delay_samples = 2
    delayed = torch.cat(
        (torch.zeros_like(waveform[..., :path_delay_samples]), waveform[..., :-path_delay_samples]),
        dim=-1,
    )
    multipath = waveform + (0.35 + 0.2j) * delayed
    sample_rate_hz = grid.fft_size * receiver.config.subcarrier_spacing_khz * 1_000.0
    cfo_hz = 0.03 * receiver.config.subcarrier_spacing_khz * 1_000.0
    impaired = apply_residual_cfo(
        multipath,
        cfo_hz=cfo_hz,
        sampling_frequency_hz=sample_rate_hz,
    )

    noise_snr_db = 45.0
    signal_power = torch.mean(torch.abs(impaired) ** 2)
    noise_variance = signal_power / (10.0 ** (noise_snr_db / 10.0))
    generator = torch.Generator(device="cpu").manual_seed(821_001)
    noise = torch.complex(
        torch.randn(impaired.shape, generator=generator),
        torch.randn(impaired.shape, generator=generator),
    ) * torch.sqrt(noise_variance / 2.0)
    noisy = impaired + noise

    estimate_hz = estimate_residual_cfo_from_cp(
        noisy,
        fft_size=grid.fft_size,
        cyclic_prefix_length=grid.cyclic_prefix_length,
        sampling_frequency_hz=sample_rate_hz,
        guard_samples=path_delay_samples,
    )
    corrected = apply_residual_cfo(
        noisy,
        cfo_hz=-estimate_hz,
        sampling_frequency_hz=sample_rate_hz,
    )

    def normalized_error(value):
        return torch.mean(torch.abs(value - multipath) ** 2) / torch.mean(torch.abs(multipath) ** 2)

    assert torch.max(torch.abs(estimate_hz - cfo_hz)) < 20.0
    assert normalized_error(corrected) < 0.002
    assert normalized_error(noisy) > 0.1


@pytest.mark.native_sionna
def test_cp_cfo_acquisition_is_robust_across_independent_in_cp_channels():
    from sionna.phy.ofdm import OFDMModulator

    batch_size = 32
    receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(n_size_bwp=6, subcarrier_spacing_khz=30), device="cpu"
    )
    grid = receiver.resource_grid
    transmit_grid, _ = receiver.transmit(batch_size, seed=821_101)
    waveform = OFDMModulator(
        cyclic_prefix_length=grid.cyclic_prefix_length,
        device="cpu",
    )(transmit_grid)

    # Independent static three-tap channels with randomized complex gains.
    # Every echo lies inside the CP guard used by the acquisition estimator.
    channel_generator = torch.Generator(device="cpu").manual_seed(821_102)
    coefficients = torch.complex(
        torch.randn((batch_size, 3), generator=channel_generator),
        torch.randn((batch_size, 3), generator=channel_generator),
    ) / torch.sqrt(torch.tensor(2.0))
    coefficients[:, 0] += 1.5
    channel_output = coefficients[:, 0, None, None, None] * waveform
    for tap_index, delay_samples in enumerate((2, 4), start=1):
        delayed = torch.nn.functional.pad(waveform, (delay_samples, 0))
        delayed = delayed[..., : waveform.shape[-1]]
        channel_output = channel_output + (coefficients[:, tap_index, None, None, None] * delayed)

    sample_rate_hz = grid.fft_size * receiver.config.subcarrier_spacing_khz * 1_000.0
    cfo_hz = 0.08 * receiver.config.subcarrier_spacing_khz * 1_000.0
    impaired = apply_residual_cfo(
        channel_output,
        cfo_hz=cfo_hz,
        sampling_frequency_hz=sample_rate_hz,
    )
    snr_db = 20.0
    signal_power = torch.mean(torch.abs(impaired) ** 2, dim=-1, keepdim=True)
    noise_variance = signal_power / (10.0 ** (snr_db / 10.0))
    noise_generator = torch.Generator(device="cpu").manual_seed(821_103)
    noise = torch.complex(
        torch.randn(impaired.shape, generator=noise_generator),
        torch.randn(impaired.shape, generator=noise_generator),
    ) * torch.sqrt(noise_variance / 2.0)
    noisy = impaired + noise

    estimate_hz = estimate_residual_cfo_from_cp(
        noisy,
        fft_size=grid.fft_size,
        cyclic_prefix_length=grid.cyclic_prefix_length,
        sampling_frequency_hz=sample_rate_hz,
        guard_samples=4,
    )
    assert estimate_hz.shape == (batch_size, 1, 1)
    absolute_error_hz = torch.abs(estimate_hz - cfo_hz)

    # The channel draw is the statistical unit; repeated OFDM symbols within
    # a realization improve its estimator but do not inflate the sample count.
    assert absolute_error_hz.shape == (batch_size, 1, 1)
    # 150 Hz is 0.5% of SCS; the wider max bound accommodates the finite
    # 32-channel batch without hiding a gross acquisition failure.
    assert torch.quantile(absolute_error_hz, 0.95) < 150.0
    assert torch.max(absolute_error_hz) < 250.0


@pytest.mark.native_sionna
def test_cp_timing_acquisition_handles_independent_in_cp_channels_and_awgn():
    from sionna.phy.ofdm import OFDMModulator

    batch_size = 32
    receiver = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(n_size_bwp=6, subcarrier_spacing_khz=30), device="cpu"
    )
    grid = receiver.resource_grid
    transmit_grid, _ = receiver.transmit(batch_size, seed=821_201)
    waveform = OFDMModulator(
        cyclic_prefix_length=grid.cyclic_prefix_length,
        device="cpu",
    )(transmit_grid)

    # Independent static three-tap channels; echo delays are within the CP.
    channel_generator = torch.Generator(device="cpu").manual_seed(821_202)
    coefficients = torch.complex(
        torch.randn((batch_size, 3), generator=channel_generator),
        torch.randn((batch_size, 3), generator=channel_generator),
    ) / torch.sqrt(torch.tensor(2.0))
    coefficients[:, 0] += 1.5
    channel_output = coefficients[:, 0, None, None, None] * waveform
    for tap_index, delay_samples in enumerate((2, 4), start=1):
        delayed = torch.nn.functional.pad(waveform, (delay_samples, 0))
        delayed = delayed[..., : waveform.shape[-1]]
        channel_output = channel_output + (coefficients[:, tap_index, None, None, None] * delayed)

    max_offset_samples = 4
    timing_offset_samples = 2
    # Keep enough trailing zeros that applying and then correcting the delay
    # does not truncate any samples from the original channel response.
    capture = torch.cat(
        (
            channel_output,
            torch.zeros_like(channel_output[..., : max_offset_samples + timing_offset_samples]),
        ),
        dim=-1,
    )
    delayed_capture = apply_integer_timing_offset(capture, offset_samples=timing_offset_samples)

    snr_db = 20.0
    signal_power = torch.mean(torch.abs(delayed_capture) ** 2, dim=-1, keepdim=True)
    noise_variance = signal_power / (10.0 ** (snr_db / 10.0))
    noise_generator = torch.Generator(device="cpu").manual_seed(821_203)
    noise = torch.complex(
        torch.randn(delayed_capture.shape, generator=noise_generator),
        torch.randn(delayed_capture.shape, generator=noise_generator),
    ) * torch.sqrt(noise_variance / 2.0)
    noisy_capture = delayed_capture + noise

    estimate = estimate_integer_timing_offset_from_cp(
        noisy_capture,
        fft_size=grid.fft_size,
        cyclic_prefix_length=grid.cyclic_prefix_length,
        max_offset_samples=max_offset_samples,
        guard_samples=4,
    )
    torch.testing.assert_close(
        estimate,
        torch.full_like(estimate, timing_offset_samples),
    )

    corrected_capture = apply_integer_timing_offset(
        noisy_capture, offset_samples=-timing_offset_samples
    )[..., : waveform.shape[-1]]
    reference = channel_output
    per_draw_nmse = torch.mean(torch.abs(corrected_capture - reference) ** 2, dim=-1) / torch.mean(
        torch.abs(reference) ** 2, dim=-1
    )
    assert per_draw_nmse.shape == (batch_size, 1, 1)
    assert torch.max(per_draw_nmse) < 0.015


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
def test_iq_imbalance_matches_analytic_image_rejection_for_complex_tone():
    sample_count = 256
    tone_bin = 17
    gain_imbalance_db = 1.5
    phase_imbalance_rad = 0.08
    sample_index = torch.arange(sample_count, dtype=torch.float64)
    phase = 2.0 * torch.pi * tone_bin * sample_index / sample_count
    tone = torch.polar(torch.ones_like(phase), phase)

    impaired = apply_iq_imbalance(
        tone,
        gain_imbalance_db=gain_imbalance_db,
        phase_imbalance_rad=phase_imbalance_rad,
    )
    spectrum = torch.fft.fft(impaired) / sample_count
    i_gain = 10.0 ** (gain_imbalance_db / 40.0)
    q_gain = 10.0 ** (-gain_imbalance_db / 40.0)
    direct_gain = 0.5 * (
        i_gain
        + q_gain * math.cos(phase_imbalance_rad)
        + 1j * q_gain * math.sin(phase_imbalance_rad)
    )
    image_gain = 0.5 * (
        i_gain
        - q_gain * math.cos(phase_imbalance_rad)
        - 1j * q_gain * math.sin(phase_imbalance_rad)
    )

    torch.testing.assert_close(
        spectrum[tone_bin],
        torch.tensor(direct_gain, dtype=torch.complex128),
        rtol=0,
        atol=1e-12,
    )
    torch.testing.assert_close(
        spectrum[-tone_bin],
        torch.tensor(image_gain, dtype=torch.complex128),
        rtol=0,
        atol=1e-12,
    )
    image_rejection_db = 20.0 * math.log10(abs(direct_gain / image_gain))
    assert image_rejection_db > 20.0
    other_bins = torch.cat(
        (spectrum[:tone_bin], spectrum[tone_bin + 1 : -tone_bin], spectrum[-tone_bin + 1 :])
    )
    assert float(torch.max(torch.abs(other_bins))) < 1e-12


@pytest.mark.native_sionna
@pytest.mark.parametrize("delay_samples", (-0.35, 0.35))
def test_fractional_timing_offset_matches_bandlimited_tone_reference(delay_samples: float):
    sample_count = 512
    frequencies_cycles_per_sample = torch.tensor((1.0 / 64.0, 0.125, 0.4), dtype=torch.float64)
    sample_index = torch.arange(sample_count, dtype=torch.float64)
    phase = 2.0 * torch.pi * frequencies_cycles_per_sample[:, None] * sample_index
    waveform = torch.polar(torch.ones_like(phase), phase)

    delayed = apply_fractional_timing_offset(
        waveform, offset_samples=delay_samples, kernel_radius=16
    )
    expected = (
        waveform
        * torch.polar(
            torch.ones_like(frequencies_cycles_per_sample),
            -2.0 * torch.pi * frequencies_cycles_per_sample * delay_samples,
        )[:, None]
    )
    interior = slice(16, sample_count - 16)
    relative_rmse = torch.sqrt(
        torch.mean(torch.abs(delayed[:, interior] - expected[:, interior]) ** 2, dim=-1)
        / torch.mean(torch.abs(expected[:, interior]) ** 2, dim=-1)
    )

    assert torch.max(relative_rmse) < 3e-4


@pytest.mark.native_sionna
def test_pusch_frequency_grid_decoder_has_correct_high_vs_low_snr_trend():
    from sionna.phy.nr import PUSCHReceiver

    transmitter = SionnaPUSCHLSReceiver(
        NRPUSCHConfig(n_size_bwp=2, subcarrier_spacing_khz=30),
        device="cpu",
    )
    tx_grid, tx_bits = transmitter.transmit(32, seed=904)
    decoder = PUSCHReceiver(
        transmitter._transmitter,
        return_tb_crc_status=True,
        input_domain="freq",
        device="cpu",
    )
    noise_generator = torch.Generator().manual_seed(905)
    noise = torch.complex(
        torch.randn(tx_grid.shape, generator=noise_generator),
        torch.randn(tx_grid.shape, generator=noise_generator),
    )

    def decode_at_variance(noise_variance: float):
        noisy_grid = tx_grid + noise * torch.sqrt(torch.tensor(noise_variance / 2))
        return decoder(noisy_grid, torch.tensor(noise_variance))

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


@pytest.mark.native_sionna
def test_fixed_payload_pusch_cp_margin_experiment_pairs_same_bits_and_noise():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    result = run_static_cp_margin_experiment(
        frames=128,
        batch_size=32,
        seed=82_417,
        snr_db=30.0,
        n_size_bwp=12,
        subcarrier_spacing_khz=30,
        in_cp_guard_samples=2,
        beyond_cp_samples=3,
        additional_echo_delays_samples=(1, 4, 8, 13, 20),
    )
    config = result["configuration"]
    inside = result["points"]["inside_cp"]
    beyond = result["points"]["beyond_cp"]

    assert config["in_cp_delay_samples"] < config["cyclic_prefix_min_samples"]
    assert config["beyond_cp_delay_samples"] > config["cyclic_prefix_min_samples"]
    assert config["same_pusch_bits_and_awgn_across_conditions"]
    assert len(config["matched_realization_checks"]) == 4
    assert len(inside["frame_error_mask"]) == len(beyond["frame_error_mask"]) == 128
    assert inside["bits"] == beyond["bits"]
    assert len(inside["measured_receive_snr_db_by_batch"]) == 4
    assert len(beyond["measured_receive_snr_db_by_batch"]) == 4
    assert result["points"]["echo_delay_13_samples"]["frames"] == 128
    assert result["points"]["echo_delay_20_samples"]["frames"] == 128
    assert "echo_delay_13_samples" in result["paired_comparisons_vs_inside_cp"]
    assert beyond["bler"] > inside["bler"]
    assert result["paired_comparison"]["mcnemar_exact_two_sided_p_value"] < 0.05
