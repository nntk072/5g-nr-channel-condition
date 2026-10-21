"""Sample-domain residual synchronization impairments for native PHY checks."""

from __future__ import annotations

import math
from typing import Any


def _validate_waveform(waveform: Any) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install PyTorch to apply OFDM waveform impairments") from exc
    if not torch.is_tensor(waveform) or not torch.is_complex(waveform):
        raise TypeError("waveform must be a complex PyTorch tensor")
    if waveform.ndim < 1 or waveform.shape[-1] == 0:
        raise ValueError("waveform must have a non-empty final time-sample axis")
    if not bool(torch.isfinite(waveform).all()):
        raise ValueError("waveform must contain only finite samples")
    return torch


def apply_residual_cfo(
    waveform: Any,
    *,
    cfo_hz: float,
    sampling_frequency_hz: float,
    initial_phase_rad: float = 0.0,
) -> Any:
    """Apply constant residual CFO and phase rotation along the final sample axis.

    The CFO is an impairment level for simulation metadata only. It must not be
    added to the receiver's model inputs unless an actual estimator exposes it.
    """
    torch = _validate_waveform(waveform)
    if not math.isfinite(cfo_hz) or not math.isfinite(initial_phase_rad):
        raise ValueError("cfo_hz and initial_phase_rad must be finite")
    if not math.isfinite(sampling_frequency_hz) or sampling_frequency_hz <= 0:
        raise ValueError("sampling_frequency_hz must be finite and positive")
    time_s = (
        torch.arange(waveform.shape[-1], dtype=waveform.real.dtype, device=waveform.device)
        / sampling_frequency_hz
    )
    phase = (2.0 * math.pi * cfo_hz) * time_s + initial_phase_rad
    rotation = torch.polar(torch.ones_like(phase), phase)
    return waveform * rotation


def apply_integer_timing_offset(waveform: Any, *, offset_samples: int) -> Any:
    """Shift the receive FFT window using zero-filled frame boundaries.

    A positive offset delays the waveform; a negative offset advances it.
    This deliberately models an integer sample residual only. It does not model
    synchronization acquisition, fractional delay, or a channel impulse response.
    """
    torch = _validate_waveform(waveform)
    if isinstance(offset_samples, bool) or not isinstance(offset_samples, int):
        raise TypeError("offset_samples must be an integer")
    sample_count = waveform.shape[-1]
    if abs(offset_samples) >= sample_count:
        raise ValueError("absolute timing offset must be smaller than the waveform")
    if offset_samples == 0:
        return waveform.clone()
    zeros_shape = (*waveform.shape[:-1], abs(offset_samples))
    zeros = torch.zeros(zeros_shape, dtype=waveform.dtype, device=waveform.device)
    if offset_samples > 0:
        return torch.cat((zeros, waveform[..., :-offset_samples]), dim=-1)
    return torch.cat((waveform[..., -offset_samples:], zeros), dim=-1)
