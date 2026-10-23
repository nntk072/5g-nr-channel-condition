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


def apply_phase_noise(waveform: Any, *, phase_noise_rad: Any) -> Any:
    """Apply a caller-supplied oscillator phase trajectory in radians.

    The trajectory is broadcast over waveform axes and must identify its sample
    time axis explicitly through its shape. This function applies a realization;
    it does not generate or claim a particular oscillator PSD.
    """
    torch = _validate_waveform(waveform)
    phase = torch.as_tensor(phase_noise_rad, device=waveform.device)
    if torch.is_complex(phase) or phase.dtype == torch.bool:
        raise TypeError("phase_noise_rad must be real-valued")
    phase = phase.to(dtype=waveform.real.dtype)
    if not bool(torch.isfinite(phase).all()):
        raise ValueError("phase_noise_rad must contain only finite values")
    try:
        phase = torch.broadcast_to(phase, waveform.shape)
    except RuntimeError as exc:
        raise ValueError("phase_noise_rad must broadcast to the waveform shape") from exc
    rotation = torch.polar(torch.ones_like(phase), phase)
    return waveform * rotation


def generate_wiener_phase_noise_rad(
    sample_count: int,
    *,
    sampling_frequency_hz: float,
    linewidth_hz: float,
    generator: Any,
    leading_shape: tuple[int, ...] = (),
    initial_phase_rad: float = 0.0,
    device: str = "cpu",
) -> Any:
    """Draw seeded Wiener phase noise with a Lorentzian linewidth model.

    Independent phase increments have variance ``2*pi*linewidth_hz/Fs``
    radians squared. This is a white-frequency-noise model; it does not model
    PLL shaping, a measured oscillator PSD, or filtering of the phase-noisy
    waveform. A relative Tx/Rx oscillator can use the sum of their linewidths.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install PyTorch to generate oscillator phase noise") from exc

    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 1:
        raise ValueError("sample_count must be a positive integer")
    if (
        isinstance(sampling_frequency_hz, bool)
        or not math.isfinite(sampling_frequency_hz)
        or sampling_frequency_hz <= 0
    ):
        raise ValueError("sampling_frequency_hz must be finite and positive")
    if isinstance(linewidth_hz, bool) or not math.isfinite(linewidth_hz) or linewidth_hz < 0:
        raise ValueError("linewidth_hz must be finite and nonnegative")
    if isinstance(initial_phase_rad, bool) or not math.isfinite(initial_phase_rad):
        raise ValueError("initial_phase_rad must be finite")
    if not isinstance(generator, torch.Generator):
        raise TypeError("generator must be an explicitly seeded torch.Generator")
    if not isinstance(leading_shape, tuple) or any(
        isinstance(size, bool) or not isinstance(size, int) or size < 1 for size in leading_shape
    ):
        raise ValueError("leading_shape must contain positive integer dimensions")
    if not isinstance(device, str) or not device:
        raise ValueError("device must be a non-empty PyTorch device string")

    increment_std = math.sqrt(2.0 * math.pi * linewidth_hz / sampling_frequency_hz)
    increments = (
        torch.randn(
            (*leading_shape, sample_count - 1),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        * increment_std
    )
    initial = torch.full((*leading_shape, 1), initial_phase_rad, dtype=torch.float32, device=device)
    return torch.cat((initial, initial + torch.cumsum(increments, dim=-1)), dim=-1)


def apply_iq_imbalance(
    waveform: Any,
    *,
    gain_imbalance_db: float = 0.0,
    phase_imbalance_rad: float = 0.0,
) -> Any:
    """Apply memoryless I/Q gain and quadrature-phase imbalance.

    ``gain_imbalance_db`` is ``20 log10(g_I/g_Q)`` with reciprocal branch
    gains around unity geometric mean. ``phase_imbalance_rad`` is the Q-branch
    quadrature skew; no DC offset, frequency selectivity, or PA nonlinearity is
    included.
    """
    torch = _validate_waveform(waveform)
    for name, value in (
        ("gain_imbalance_db", gain_imbalance_db),
        ("phase_imbalance_rad", phase_imbalance_rad),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a real number")
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    i_gain = 10.0 ** (gain_imbalance_db / 40.0)
    q_gain = 10.0 ** (-gain_imbalance_db / 40.0)
    phase_cos = math.cos(phase_imbalance_rad)
    phase_sin = math.sin(phase_imbalance_rad)
    real = i_gain * waveform.real - q_gain * phase_sin * waveform.imag
    imag = q_gain * phase_cos * waveform.imag
    return torch.complex(real, imag)


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


def estimate_integer_timing_offset_from_cp(
    waveform: Any,
    *,
    fft_size: int,
    cyclic_prefix_length: int,
    max_offset_samples: int,
    guard_samples: int = 0,
) -> Any:
    """Acquire a bounded nonnegative integer delay from OFDM CP repetition.

    The capture must include the first transmitted symbol, delayed by at most
    ``max_offset_samples`` and zero-filled at the front. The detector scans
    candidate starts for that first symbol and sums normalized CP-to-tail
    correlation over all complete symbols available at the worst-case offset.
    It returns one integer sample offset per leading waveform index.

    This resolves a bounded sample delay when the capture's nominal origin is
    known. It does not detect an NR frame in a continuous stream, resolve
    fractional timing, or replace PSS/SSS/DMRS synchronization.
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
    if isinstance(max_offset_samples, bool) or not isinstance(max_offset_samples, int):
        raise TypeError("max_offset_samples must be an integer")
    if not 0 <= max_offset_samples < waveform.shape[-1]:
        raise ValueError("max_offset_samples must be nonnegative and smaller than waveform length")
    if isinstance(guard_samples, bool) or not isinstance(guard_samples, int):
        raise TypeError("guard_samples must be an integer")
    if not 0 <= guard_samples < cyclic_prefix_length:
        raise ValueError("guard_samples must be in [0, cyclic_prefix_length)")

    symbol_length = fft_size + cyclic_prefix_length
    symbol_count = (waveform.shape[-1] - max_offset_samples) // symbol_length
    if symbol_count < 1:
        raise ValueError("waveform must contain a complete symbol at every candidate offset")

    offsets = torch.arange(max_offset_samples + 1, device=waveform.device)
    symbols = torch.arange(symbol_count, device=waveform.device) * symbol_length
    within_prefix = torch.arange(guard_samples, cyclic_prefix_length, device=waveform.device)
    indices = offsets[:, None, None] + symbols[None, :, None] + within_prefix[None, None, :]
    prefix_indices = indices.reshape(-1)
    tail_indices = prefix_indices + fft_size
    index_shape = (*waveform.shape[:-1], max_offset_samples + 1, symbol_count, -1)
    prefix = waveform.index_select(-1, prefix_indices).reshape(index_shape)
    tail = waveform.index_select(-1, tail_indices).reshape(index_shape)
    correlation = torch.sum(torch.conj(prefix) * tail, dim=(-1, -2))
    prefix_energy = torch.sum(torch.abs(prefix) ** 2, dim=(-1, -2))
    tail_energy = torch.sum(torch.abs(tail) ** 2, dim=(-1, -2))
    energy_product = prefix_energy * tail_energy
    if bool((torch.amax(energy_product, dim=-1) <= torch.finfo(waveform.real.dtype).tiny).any()):
        raise ValueError("cyclic-prefix correlation has insufficient signal energy")
    score = torch.abs(correlation) / torch.sqrt(
        torch.clamp_min(energy_product, torch.finfo(waveform.real.dtype).tiny)
    )
    return torch.argmax(score, dim=-1)


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


def apply_fractional_timing_offset(
    waveform: Any,
    *,
    offset_samples: float,
    kernel_radius: int = 16,
) -> Any:
    """Apply a bandlimited fractional-sample delay on the final time axis.

    This uses a finite Blackman-windowed sinc interpolator and zero-valued
    samples outside the captured waveform. ``offset_samples`` is restricted
    to the fractional interval (-1, 1); integer sample timing is handled by
    :func:`apply_integer_timing_offset`. This models an impairment only: it
    does not acquire frame timing or estimate a receiver correction.
    """
    torch = _validate_waveform(waveform)
    if isinstance(offset_samples, bool) or not isinstance(offset_samples, (int, float)):
        raise TypeError("offset_samples must be a real number")
    delay = float(offset_samples)
    if not math.isfinite(delay) or abs(delay) >= 1.0:
        raise ValueError("fractional offset must be finite and have magnitude below one sample")
    if isinstance(kernel_radius, bool) or not isinstance(kernel_radius, int):
        raise TypeError("kernel_radius must be an integer")
    if not 2 <= kernel_radius <= 128:
        raise ValueError("kernel_radius must be in [2, 128]")
    if delay == 0.0:
        return waveform.clone()

    sample_count = waveform.shape[-1]
    positions = torch.arange(sample_count, dtype=waveform.real.dtype, device=waveform.device)
    source_positions = positions - delay
    centers = torch.floor(source_positions).to(torch.int64)
    taps = torch.arange(-kernel_radius, kernel_radius + 1, device=waveform.device)
    source_indices = centers[:, None] + taps[None, :]
    distance = source_indices.to(waveform.real.dtype) - source_positions[:, None]
    normalized_distance = distance / kernel_radius
    window = (
        0.42
        + 0.5 * torch.cos(math.pi * normalized_distance)
        + 0.08 * torch.cos(2.0 * math.pi * normalized_distance)
    )
    window = torch.where(torch.abs(normalized_distance) <= 1.0, window, 0.0)
    weights = torch.sinc(distance) * window
    weights = weights / torch.sum(weights, dim=-1, keepdim=True)

    valid = (source_indices >= 0) & (source_indices < sample_count)
    safe_indices = source_indices.clamp(0, sample_count - 1).reshape(-1)
    samples = waveform.index_select(-1, safe_indices).reshape(
        *waveform.shape[:-1], sample_count, 2 * kernel_radius + 1
    )
    samples = samples * valid.to(waveform.real.dtype)
    return torch.sum(samples * weights, dim=-1)
