"""Signal models for ordered impairments receiver."""

from __future__ import annotations

import importlib
import importlib.metadata
import math
from typing import TYPE_CHECKING, Any

from integrations.sionna.ofdm_impairments import (
    apply_integer_timing_offset,
    apply_iq_imbalance,
    apply_phase_noise,
    apply_residual_cfo,
    apply_time_varying_multipath,
    generate_wiener_phase_noise_rad,
)

if TYPE_CHECKING:
    import torch
else:

    class _LazyTorch:
        def __getattr__(self, name: str) -> Any:
            module = importlib.import_module("torch")
            globals()["torch"] = module
            return getattr(module, name)

    torch = _LazyTorch()

from .constants import (
    CFO_HZ,
    DIRECT_COEFFICIENT,
    ECHO_COEFFICIENT,
    ECHO_DELAY_SAMPLES,
    RX_IQ_GAIN_IMBALANCE_DB,
    RX_IQ_PHASE_IMBALANCE_RAD,
    TX_IQ_GAIN_IMBALANCE_DB,
    TX_IQ_PHASE_IMBALANCE_RAD,
)


def _proper_complex_noise(shape: Any, seed: int, variance: float) -> torch.Tensor:
    real, imag = _standard_complex_noise_draws(shape, seed)
    return torch.complex(real, imag) * math.sqrt(variance / 2.0)


def _standard_complex_noise_draws(
    shape: Any, seed: int, *, device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return CPU-seeded unscaled Gaussian draws on the requested device."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    real = torch.randn(shape, generator=generator, dtype=torch.float32)
    imag = torch.randn(shape, generator=generator, dtype=torch.float32)
    return real.to(device=device), imag.to(device=device)


def _phase_noise(
    shape: Any,
    seed: int,
    sample_rate_hz: float,
    linewidth_hz: float,
    *,
    device: str = "cpu",
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    phase = generate_wiener_phase_noise_rad(
        shape[-1],
        sampling_frequency_hz=sample_rate_hz,
        linewidth_hz=linewidth_hz,
        generator=generator,
        leading_shape=tuple(shape[:-1]),
    )
    return phase.to(device=device)


def _known_chain(
    waveform: torch.Tensor,
    *,
    sample_rate_hz: float,
    tx_phase: torch.Tensor,
    rx_phase: torch.Tensor,
    capture_delay: int,
    path_coefficients: tuple[complex, ...] = (DIRECT_COEFFICIENT, ECHO_COEFFICIENT),
    path_delays_samples: tuple[int, ...] = (0, ECHO_DELAY_SAMPLES),
) -> torch.Tensor:
    """Apply the physical impairment sequence declared by this experiment."""
    tx_iq = apply_iq_imbalance(
        waveform,
        gain_imbalance_db=TX_IQ_GAIN_IMBALANCE_DB,
        phase_imbalance_rad=TX_IQ_PHASE_IMBALANCE_RAD,
    )
    tx_noisy = apply_phase_noise(tx_iq, phase_noise_rad=tx_phase)
    propagated = apply_time_varying_multipath(
        tx_noisy,
        path_coefficients=path_coefficients,
        path_delays_samples=path_delays_samples,
        path_dopplers_hz=(0.0,) * len(path_coefficients),
        sampling_frequency_hz=sample_rate_hz,
    )
    cfo = apply_residual_cfo(propagated, cfo_hz=CFO_HZ, sampling_frequency_hz=sample_rate_hz)
    rx_noisy = apply_phase_noise(cfo, phase_noise_rad=rx_phase)
    rx_iq = apply_iq_imbalance(
        rx_noisy,
        gain_imbalance_db=RX_IQ_GAIN_IMBALANCE_DB,
        phase_imbalance_rad=RX_IQ_PHASE_IMBALANCE_RAD,
    )
    return apply_integer_timing_offset(rx_iq, offset_samples=capture_delay)
