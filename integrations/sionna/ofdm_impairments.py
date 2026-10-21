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
    cfo_hz: Any,
    sampling_frequency_hz: float,
    initial_phase_rad: float = 0.0,
) -> Any:
    """Apply constant residual CFO and phase rotation along the final sample axis.

    The CFO is an impairment level for simulation metadata only. It must not be
    added to the receiver's model inputs unless an actual estimator exposes it.
    """
    torch = _validate_waveform(waveform)
    if not math.isfinite(initial_phase_rad):
        raise ValueError("initial_phase_rad must be finite")
    if not math.isfinite(sampling_frequency_hz) or sampling_frequency_hz <= 0:
        raise ValueError("sampling_frequency_hz must be finite and positive")
    cfo = torch.as_tensor(cfo_hz, dtype=waveform.real.dtype, device=waveform.device)
    if not bool(torch.isfinite(cfo).all()):
        raise ValueError("cfo_hz must contain only finite values")
    leading_shape = waveform.shape[:-1]
    if cfo.ndim > len(leading_shape):
        raise ValueError("cfo_hz dimensions must broadcast over waveform dimensions")
    cfo_shape = (*cfo.shape, *((1,) * (len(leading_shape) - cfo.ndim)))
    cfo = torch.broadcast_to(cfo.reshape(cfo_shape), leading_shape).unsqueeze(-1)
    time_s = (
        torch.arange(waveform.shape[-1], dtype=waveform.real.dtype, device=waveform.device)
        / sampling_frequency_hz
    )
    phase = (2.0 * math.pi * cfo) * time_s + initial_phase_rad
    rotation = torch.polar(torch.ones_like(phase), phase)
    return waveform * rotation


def estimate_residual_cfo_from_cp(
    waveform: Any,
    *,
    fft_size: int,
    cyclic_prefix_length: int,
    sampling_frequency_hz: float,
    guard_samples: int = 0,
) -> Any:
    """Estimate per-waveform CFO from repeated OFDM cyclic-prefix samples.

    The waveform must begin at an OFDM-symbol boundary and contain an integer
    number of equal-length symbols. The estimate is unambiguous only within
    ``±sampling_frequency_hz/(2*fft_size)``. ``guard_samples`` skips the start
    of each cyclic prefix, which can reduce multipath edge contamination when a
    channel-delay upper bound is known. This is a simple acquisition reference,
    not a standards-complete 5G NR synchronizer.
    """
    torch = _validate_waveform(waveform)
    if isinstance(fft_size, bool) or not isinstance(fft_size, int) or fft_size <= 0:
        raise ValueError("fft_size must be a positive integer")
    if (
        isinstance(cyclic_prefix_length, bool)
        or not isinstance(cyclic_prefix_length, int)
        or not 0 < cyclic_prefix_length <= fft_size
    ):
        raise ValueError("cyclic_prefix_length must be an integer in [1, fft_size]")
    if isinstance(guard_samples, bool) or not isinstance(guard_samples, int):
        raise TypeError("guard_samples must be an integer")
    if not 0 <= guard_samples < cyclic_prefix_length:
        raise ValueError("guard_samples must be in [0, cyclic_prefix_length)")
    if not math.isfinite(sampling_frequency_hz) or sampling_frequency_hz <= 0:
        raise ValueError("sampling_frequency_hz must be finite and positive")

    symbol_length = fft_size + cyclic_prefix_length
    if waveform.shape[-1] < symbol_length or waveform.shape[-1] % symbol_length:
        raise ValueError("waveform must contain whole OFDM symbols with cyclic prefixes")
    symbol_count = waveform.shape[-1] // symbol_length
    framed = waveform.reshape(*waveform.shape[:-1], symbol_count, symbol_length)
    prefix = framed[..., guard_samples:cyclic_prefix_length]
    repeated_tail = framed[..., fft_size + guard_samples : fft_size + cyclic_prefix_length]
    correlation = torch.sum(torch.conj(prefix) * repeated_tail, dim=(-1, -2))
    if bool((torch.abs(correlation) <= torch.finfo(waveform.real.dtype).eps).any()):
        raise ValueError("cyclic-prefix correlation has insufficient energy")
    phase = torch.angle(correlation)
    return phase * sampling_frequency_hz / (2.0 * math.pi * fft_size)


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
