"""Unscaled OFDM modulation and demodulation for one NR SS/PBCH block."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import SupportsFloat

import numpy as np
from numpy.typing import NDArray

_SSB_SUBCARRIER_COUNT = 240
_SSB_SYMBOL_COUNT = 4
_SCS_TO_MU = {
    15_000: 0,
    30_000: 1,
    60_000: 2,
    120_000: 3,
    240_000: 4,
    480_000: 5,
    960_000: 6,
}


@dataclass(frozen=True)
class NRSSBWaveformConfig:
    """Numerology-derived discrete-time configuration for one SS/PBCH block."""

    subcarrier_spacing_hz: int
    numerology: int
    fft_size: int
    sample_rate_hz: int
    candidate_start_symbol: int
    cyclic_prefix_lengths: tuple[int, int, int, int]


def nr_ssb_waveform_config(
    *,
    subcarrier_spacing_hz: SupportsFloat,
    candidate_start_symbol: int,
    fft_size: int,
) -> NRSSBWaveformConfig:
    """Derive normal-CP lengths and sample rate for an SSB candidate.

    CP lengths follow TS 38.211 V19.1.0, clauses 4.2 and 5.3.1, with the
    standard useful-symbol length scaled to ``fft_size``. The candidate symbol
    index is zero-based from the start of a half-frame, as in TS 38.213. The
    returned sample rate is ``fft_size * subcarrier_spacing_hz``. This utility
    derives waveform timing only; it does not select the band-specific SSB
    case, occupied bandwidth, carrier frequency, or RF power.
    """
    if isinstance(subcarrier_spacing_hz, (bool, np.bool_)) or not isinstance(
        subcarrier_spacing_hz, Real
    ):
        raise TypeError("subcarrier_spacing_hz must be a real scalar in hertz")
    try:
        scs_value = float(subcarrier_spacing_hz)
    except OverflowError:
        raise ValueError("subcarrier_spacing_hz must be finite") from None
    if not math.isfinite(scs_value):
        raise ValueError("subcarrier_spacing_hz must be finite")
    if not scs_value.is_integer() or int(scs_value) not in _SCS_TO_MU:
        raise ValueError("subcarrier_spacing_hz must be one of 15 to 960 kHz in powers of two")
    scs_hz = int(scs_value)
    mu = _SCS_TO_MU[scs_hz]

    if isinstance(candidate_start_symbol, (bool, np.bool_)) or not isinstance(
        candidate_start_symbol, Integral
    ):
        raise TypeError("candidate_start_symbol must be an integer")
    start_symbol = int(candidate_start_symbol)
    symbols_per_half_frame = 70 * (2**mu)
    if not 0 <= start_symbol <= symbols_per_half_frame - _SSB_SYMBOL_COUNT:
        raise ValueError("candidate_start_symbol must fit a four-symbol block in one half-frame")

    fft_length = _fft_size(fft_size)
    short_numerator = 144 * fft_length
    long_numerator = (144 + 16 * (2**mu)) * fft_length
    if short_numerator % 2048 or long_numerator % 2048:
        raise ValueError("fft_size cannot represent integer normal-CP lengths for this numerology")
    short_cp = short_numerator // 2048
    long_cp = long_numerator // 2048
    symbols_per_subframe = 14 * (2**mu)
    long_cp_symbol = 7 * (2**mu)
    prefix_lengths = tuple(
        long_cp
        if (start_symbol + relative_symbol) % symbols_per_subframe in (0, long_cp_symbol)
        else short_cp
        for relative_symbol in range(_SSB_SYMBOL_COUNT)
    )
    return NRSSBWaveformConfig(
        subcarrier_spacing_hz=scs_hz,
        numerology=mu,
        fft_size=fft_length,
        sample_rate_hz=fft_length * scs_hz,
        candidate_start_symbol=start_symbol,
        cyclic_prefix_lengths=prefix_lengths,  # type: ignore[arg-type]
    )


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
