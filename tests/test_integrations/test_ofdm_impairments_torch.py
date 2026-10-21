"""CPU-level checks for optional PyTorch waveform impairment helpers."""

import pytest

torch = pytest.importorskip("torch")

from integrations.sionna.ofdm_impairments import (  # noqa: E402
    apply_residual_cfo,
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
