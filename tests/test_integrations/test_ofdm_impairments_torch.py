"""CPU-level checks for optional PyTorch waveform impairment helpers."""

import math

import pytest

torch = pytest.importorskip("torch")

from integrations.sionna.ofdm_impairments import (  # noqa: E402
    apply_fractional_timing_offset,
    apply_integer_timing_offset,
    apply_iq_imbalance,
    apply_phase_noise,
    apply_residual_cfo,
    estimate_integer_timing_offset_from_cp,
    estimate_residual_cfo_from_cp,
)


def test_phase_noise_and_iq_imbalance_match_samplewise_reference() -> None:
    sample_count = 128
    sample_indices = torch.arange(sample_count, dtype=torch.float64)
    waveform = torch.exp(2j * torch.pi * 0.13 * sample_indices).to(torch.complex128)
    phase_noise_rad = 0.04 * torch.sin(2.0 * torch.pi * sample_indices / 37.0)
    phase_impaired = apply_phase_noise(waveform, phase_noise_rad=phase_noise_rad)
    expected_phase = waveform * torch.exp(1j * phase_noise_rad)
    torch.testing.assert_close(phase_impaired, expected_phase, atol=1e-12, rtol=1e-12)

    gain_imbalance_db = 1.5
    phase_imbalance_rad = 0.08
    imbalanced = apply_iq_imbalance(
        waveform,
        gain_imbalance_db=gain_imbalance_db,
        phase_imbalance_rad=phase_imbalance_rad,
    )
    i_gain = 10.0 ** (gain_imbalance_db / 40.0)
    q_gain = 10.0 ** (-gain_imbalance_db / 40.0)
    expected_iq = torch.complex(
        i_gain * waveform.real - q_gain * math.sin(phase_imbalance_rad) * waveform.imag,
        q_gain * math.cos(phase_imbalance_rad) * waveform.imag,
    )
    torch.testing.assert_close(imbalanced, expected_iq, atol=1e-12, rtol=1e-12)


def test_tx_channel_oscillator_rx_iq_and_timing_order_matches_reference() -> None:
    sample_count = 256
    sample_rate_hz = 4_000.0
    sample_indices = torch.arange(sample_count, dtype=torch.float64)
    waveform = torch.exp(2j * torch.pi * 317.0 * sample_indices / sample_rate_hz).to(
        torch.complex128
    )
    phase_noise_rad = 0.02 * torch.sin(2.0 * torch.pi * sample_indices / 53.0)
    cfo_hz = 21.0
    timing_offset_samples = 3
    tx_gain_db, tx_phase_rad = 1.0, 0.03
    rx_gain_db, rx_phase_rad = -0.7, -0.02
    channel_gain = 0.8 - 0.25j

    tx_output = apply_iq_imbalance(
        waveform,
        gain_imbalance_db=tx_gain_db,
        phase_imbalance_rad=tx_phase_rad,
    )
    after_channel = tx_output * channel_gain
    after_oscillator = apply_residual_cfo(
        after_channel,
        cfo_hz=cfo_hz,
        sampling_frequency_hz=sample_rate_hz,
    )
    after_oscillator = apply_phase_noise(
        after_oscillator,
        phase_noise_rad=phase_noise_rad,
    )
    after_rx_iq = apply_iq_imbalance(
        after_oscillator,
        gain_imbalance_db=rx_gain_db,
        phase_imbalance_rad=rx_phase_rad,
    )
    actual = apply_integer_timing_offset(after_rx_iq, offset_samples=timing_offset_samples)

    def reference_iq(value: torch.Tensor, gain_db: float, phase_rad: float) -> torch.Tensor:
        i_gain = 10.0 ** (gain_db / 40.0)
        q_gain = 10.0 ** (-gain_db / 40.0)
        return torch.complex(
            i_gain * value.real - q_gain * math.sin(phase_rad) * value.imag,
            q_gain * math.cos(phase_rad) * value.imag,
        )

    tx_reference = reference_iq(waveform, tx_gain_db, tx_phase_rad) * channel_gain
    oscillator_phase = 2.0 * math.pi * cfo_hz * sample_indices / sample_rate_hz
    oscillator_reference = tx_reference * torch.exp(1j * (oscillator_phase + phase_noise_rad))
    rx_reference = reference_iq(oscillator_reference, rx_gain_db, rx_phase_rad)
    expected = torch.cat(
        (
            torch.zeros_like(rx_reference[:timing_offset_samples]),
            rx_reference[:-timing_offset_samples],
        )
    )

    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


def test_phase_noise_and_iq_imbalance_reject_invalid_parameters() -> None:
    waveform = torch.ones(32, dtype=torch.complex64)
    with pytest.raises(ValueError, match="finite"):
        apply_phase_noise(waveform, phase_noise_rad=float("nan"))
    with pytest.raises(TypeError, match="real-valued"):
        apply_phase_noise(waveform, phase_noise_rad=1.0 + 0.0j)
    with pytest.raises(ValueError, match="broadcast"):
        apply_phase_noise(waveform, phase_noise_rad=torch.ones(2))
    with pytest.raises(TypeError, match="real number"):
        apply_iq_imbalance(waveform, gain_imbalance_db=True)
    with pytest.raises(ValueError, match="finite"):
        apply_iq_imbalance(waveform, phase_imbalance_rad=float("inf"))


def test_fractional_timing_offset_has_bandlimited_tone_phase_response() -> None:
    sample_count = 4096
    frequency_cycles_per_sample = 0.071
    delay_samples = 0.375
    sample_indices = torch.arange(sample_count, dtype=torch.float64)
    waveform = torch.exp(2j * torch.pi * frequency_cycles_per_sample * sample_indices).to(
        torch.complex128
    )

    delayed = apply_fractional_timing_offset(
        waveform,
        offset_samples=delay_samples,
        kernel_radius=20,
    )
    expected = waveform * torch.exp(
        torch.tensor(
            -2j * torch.pi * frequency_cycles_per_sample * delay_samples,
            dtype=torch.complex128,
        )
    )

    torch.testing.assert_close(delayed[24:-24], expected[24:-24], atol=2e-5, rtol=2e-5)


def test_fractional_timing_offset_validates_delay_and_kernel() -> None:
    waveform = torch.ones(32, dtype=torch.complex64)
    with pytest.raises(TypeError, match="real number"):
        apply_fractional_timing_offset(waveform, offset_samples=True)
    with pytest.raises(ValueError, match="below one sample"):
        apply_fractional_timing_offset(waveform, offset_samples=1.0)
    with pytest.raises(ValueError, match="kernel_radius"):
        apply_fractional_timing_offset(waveform, offset_samples=0.25, kernel_radius=1)


def test_cyclic_prefix_cfo_acquisition_recovers_per_waveform_and_corrects() -> None:
    fft_size = 64
    cp_length = 8
    sampling_frequency_hz = 6_400.0
    generator = torch.Generator().manual_seed(31)
    useful = torch.complex(
        torch.randn((2, 5, fft_size), generator=generator),
        torch.randn((2, 5, fft_size), generator=generator),
    )
    symbols = torch.cat((useful[..., -cp_length:], useful), dim=-1).flatten(-2)
    expected_cfo_hz = torch.tensor([12.5, -25.0])

    impaired = apply_residual_cfo(
        symbols,
        cfo_hz=expected_cfo_hz,
        sampling_frequency_hz=sampling_frequency_hz,
    )
    estimate_hz = estimate_residual_cfo_from_cp(
        impaired,
        fft_size=fft_size,
        cyclic_prefix_length=cp_length,
        sampling_frequency_hz=sampling_frequency_hz,
        guard_samples=1,
    )
    corrected = apply_residual_cfo(
        impaired,
        cfo_hz=-estimate_hz,
        sampling_frequency_hz=sampling_frequency_hz,
    )

    torch.testing.assert_close(estimate_hz, expected_cfo_hz, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(corrected, symbols, atol=2e-6, rtol=2e-6)


def test_cyclic_prefix_cfo_acquisition_rejects_invalid_frame_and_parameters() -> None:
    waveform = torch.ones(144, dtype=torch.complex64)
    with pytest.raises(ValueError, match="whole OFDM symbols"):
        estimate_residual_cfo_from_cp(
            waveform[:-1],
            fft_size=64,
            cyclic_prefix_length=8,
            sampling_frequency_hz=6_400.0,
        )
    with pytest.raises(ValueError, match="guard_samples"):
        estimate_residual_cfo_from_cp(
            waveform,
            fft_size=64,
            cyclic_prefix_length=8,
            sampling_frequency_hz=6_400.0,
            guard_samples=8,
        )
    with pytest.raises(ValueError, match="finite"):
        apply_residual_cfo(
            waveform,
            cfo_hz=float("nan"),
            sampling_frequency_hz=6_400.0,
        )


def test_cyclic_prefix_timing_acquisition_finds_bounded_delay_per_waveform() -> None:
    fft_size = 64
    cp_length = 8
    generator = torch.Generator().manual_seed(87)
    useful = torch.complex(
        torch.randn((2, 6, fft_size), generator=generator),
        torch.randn((2, 6, fft_size), generator=generator),
    )
    symbols = torch.cat((useful[..., -cp_length:], useful), dim=-1).flatten(-2)
    delayed = apply_integer_timing_offset(symbols, offset_samples=5)
    delayed = apply_residual_cfo(
        delayed,
        cfo_hz=12.5,
        sampling_frequency_hz=6_400.0,
    )

    estimate = estimate_integer_timing_offset_from_cp(
        delayed,
        fft_size=fft_size,
        cyclic_prefix_length=cp_length,
        max_offset_samples=7,
    )

    torch.testing.assert_close(estimate, torch.tensor([5, 5]))


def test_cyclic_prefix_timing_acquisition_rejects_unsearchable_capture() -> None:
    waveform = torch.zeros(160, dtype=torch.complex64)
    with pytest.raises(ValueError, match="insufficient signal energy"):
        estimate_integer_timing_offset_from_cp(
            waveform,
            fft_size=64,
            cyclic_prefix_length=8,
            max_offset_samples=8,
        )
    with pytest.raises(ValueError, match="complete symbol"):
        estimate_integer_timing_offset_from_cp(
            torch.ones(79, dtype=torch.complex64),
            fft_size=64,
            cyclic_prefix_length=8,
            max_offset_samples=8,
        )
