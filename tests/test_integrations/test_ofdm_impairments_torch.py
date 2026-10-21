"""CPU-level checks for optional PyTorch waveform impairment helpers."""

import pytest

torch = pytest.importorskip("torch")

from integrations.sionna.ofdm_impairments import (  # noqa: E402
    apply_integer_timing_offset,
    apply_residual_cfo,
    estimate_integer_timing_offset_from_cp,
    estimate_residual_cfo_from_cp,
)


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
