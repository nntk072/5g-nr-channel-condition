"""Unscaled OFDM modulation and demodulation for one NR SS/PBCH block."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral

import numpy as np
from numpy.typing import NDArray

_SSB_SUBCARRIER_COUNT = 240
_SSB_SYMBOL_COUNT = 4


def nr_ssb_ofdm_modulate(
    ssb_grid: NDArray[np.generic],
    *,
    fft_size: int,
    cyclic_prefix_lengths: Sequence[int],
) -> NDArray[np.complex128]:
    """Map a 4×240 SSB grid to a centered, unitary-FFT baseband waveform.

    This applies only subcarrier placement, the inverse DFT, and the supplied
    per-symbol cyclic prefixes. It does not set SCS/sample rate, apply a carrier
    offset or TS 38.213 power scaling, or place the SSB in a larger burst.
    """
    fft_length = _fft_size(fft_size)
    cp_lengths = _cyclic_prefix_lengths(cyclic_prefix_lengths, fft_length)
    source = np.asarray(ssb_grid)
    if source.shape != (_SSB_SYMBOL_COUNT, _SSB_SUBCARRIER_COUNT):
        raise ValueError("ssb_grid must have shape (4, 240)")
    if source.dtype.kind not in "iufc" or source.dtype.kind == "b":
        raise TypeError("ssb_grid must be a real or complex numeric array")
    grid = np.asarray(source, dtype=np.complex128)
    if not np.isfinite(grid.real).all() or not np.isfinite(grid.imag).all():
        raise ValueError("ssb_grid must be finite")

    full_grid = np.zeros((_SSB_SYMBOL_COUNT, fft_length), dtype=np.complex128)
    active_start = fft_length // 2 - _SSB_SUBCARRIER_COUNT // 2
    full_grid[:, active_start : active_start + _SSB_SUBCARRIER_COUNT] = grid
    useful_symbols = np.fft.ifft(np.fft.ifftshift(full_grid, axes=-1), axis=-1, norm="ortho")
    waveform = np.concatenate(
        [
            np.concatenate((symbol[-cp:], symbol))
            for symbol, cp in zip(useful_symbols, cp_lengths, strict=True)
        ]
    )
    waveform.setflags(write=False)
    return waveform


def nr_ssb_ofdm_demodulate(
    waveform: NDArray[np.generic],
    *,
    fft_size: int,
    cyclic_prefix_lengths: Sequence[int],
) -> NDArray[np.complex128]:
    """Remove supplied per-symbol cyclic prefixes and extract the 240 SSB bins."""
    fft_length = _fft_size(fft_size)
    cp_lengths = _cyclic_prefix_lengths(cyclic_prefix_lengths, fft_length)
    samples = np.asarray(waveform)
    expected_length = sum(fft_length + cp for cp in cp_lengths)
    if samples.shape != (expected_length,) or samples.dtype.kind != "c":
        raise ValueError(f"waveform must be a complex vector of shape ({expected_length},)")
    samples = np.asarray(samples, dtype=np.complex128)
    if not np.isfinite(samples).all():
        raise ValueError("waveform must be finite")

    active_start = fft_length // 2 - _SSB_SUBCARRIER_COUNT // 2
    ssb_grid = np.empty((_SSB_SYMBOL_COUNT, _SSB_SUBCARRIER_COUNT), dtype=np.complex128)
    offset = 0
    for symbol, cp in enumerate(cp_lengths):
        useful_start = offset + cp
        useful = samples[useful_start : useful_start + fft_length]
        spectrum = np.fft.fftshift(np.fft.fft(useful, norm="ortho"))
        ssb_grid[symbol] = spectrum[active_start : active_start + _SSB_SUBCARRIER_COUNT]
        offset += cp + fft_length
    ssb_grid.setflags(write=False)
    return ssb_grid


def _fft_size(value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError("fft_size must be an integer")
    result = int(value)
    if result <= _SSB_SUBCARRIER_COUNT or result % 2:
        raise ValueError("fft_size must be even and greater than 240")
    return result


def _cyclic_prefix_lengths(values: Sequence[int], fft_size: int) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or len(values) != _SSB_SYMBOL_COUNT:
        raise ValueError("cyclic_prefix_lengths must contain four lengths")
    result: list[int] = []
    for value in values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise TypeError("cyclic prefix lengths must be integers")
        length = int(value)
        if not 0 < length < fft_size:
            raise ValueError("cyclic prefix lengths must be in [1, fft_size)")
        result.append(length)
    return tuple(result)
