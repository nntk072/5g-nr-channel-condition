"""NR SS/PBCH OFDM waveform construction from one block to a radio frame."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import SupportsFloat

import numpy as np
from numpy.typing import NDArray

from .pbch_dmrs import PBCHDMRSResourceMap
from .ssb_grid import NRSSBResourceGrid
from .ssb_timing import ssb_candidate_symbols

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


@dataclass(frozen=True)
class NRSSBHalfFrameWaveform:
    """One sampled 5 ms half-frame with candidate-indexed SSB transmissions."""

    waveform: NDArray[np.complex128]
    subcarrier_spacing_hz: int
    fft_size: int
    sample_rate_hz: int
    half_frame_index: int
    candidate_start_symbols: tuple[int, ...]
    transmitted_candidate_indices: tuple[int, ...]


@dataclass(frozen=True)
class NRSSBRadioFrameWaveform:
    """One 10 ms radio frame containing zero or more SSBs in each half-frame."""

    waveform: NDArray[np.complex128]
    subcarrier_spacing_hz: int
    fft_size: int
    sample_rate_hz: int
    candidate_start_symbols: tuple[int, ...]
    transmitted_candidate_indices_by_half_frame: tuple[tuple[int, ...], tuple[int, ...]]


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


def nr_ssb_ofdm_modulate_half_frame(
    ssb_blocks: Mapping[int, NRSSBResourceGrid],
    *,
    case: str,
    carrier_frequency_hz: SupportsFloat,
    subcarrier_spacing_hz: SupportsFloat,
    frequency_range: str,
    fft_size: int,
    paired_spectrum: bool = True,
) -> NRSSBHalfFrameWaveform:
    """Place candidate-indexed SSB grids in a standards-timed half-frame.

    Only provided candidate indices are transmitted; all other symbols remain
    zero. Candidate positions follow TS 38.213 and each symbol's normal CP is
    derived from its absolute half-frame position under TS 38.211. Every input
    grid must carry matching PBCH-DM-RS candidate, half-frame, and cell
    metadata. The model uses the same centered-frequency placement as the
    single-block OFDM adapter and does not place the SSB within a wider carrier.
    """
    if not isinstance(ssb_blocks, Mapping) or not ssb_blocks:
        raise ValueError("ssb_blocks must be a non-empty candidate-index mapping")
    positions = ssb_candidate_symbols(
        case,
        carrier_frequency_hz,
        subcarrier_spacing_hz=subcarrier_spacing_hz,
        frequency_range=frequency_range,
        paired_spectrum=paired_spectrum,
    )
    fft_length = _fft_size(fft_size)
    if isinstance(subcarrier_spacing_hz, (bool, np.bool_)) or not isinstance(
        subcarrier_spacing_hz, Real
    ):
        raise TypeError("subcarrier_spacing_hz must be a real scalar in hertz")
    scs_hz = int(float(subcarrier_spacing_hz))
    mu = _SCS_TO_MU[scs_hz]
    symbols_per_half_frame = 70 * (2**mu)
    symbols_per_subframe = 14 * (2**mu)
    long_cp_symbol = 7 * (2**mu)
    short_numerator = 144 * fft_length
    long_numerator = (144 + 16 * (2**mu)) * fft_length
    if short_numerator % 2048 or long_numerator % 2048:
        raise ValueError("fft_size cannot represent integer normal-CP lengths for this numerology")
    short_cp = short_numerator // 2048
    long_cp = long_numerator // 2048
    cp_by_symbol = tuple(
        long_cp if symbol % symbols_per_subframe in (0, long_cp_symbol) else short_cp
        for symbol in range(symbols_per_half_frame)
    )
    symbol_lengths = tuple(fft_length + cp for cp in cp_by_symbol)
    sample_offsets = np.empty(symbols_per_half_frame + 1, dtype=np.int64)
    sample_offsets[0] = 0
    np.cumsum(symbol_lengths, out=sample_offsets[1:])
    sample_rate_hz = fft_length * scs_hz
    if sample_offsets[-1] != sample_rate_hz * 0.005:
        raise RuntimeError("normal-CP symbol lengths do not sum to a 5 ms half-frame")

    candidate_count = len(positions)
    indices: list[int] = []
    for index in ssb_blocks:
        if isinstance(index, (bool, np.bool_)) or not isinstance(index, Integral):
            raise TypeError("SSB candidate indices must be integers")
        normalized_index = int(index)
        if not 0 <= normalized_index < candidate_count:
            raise ValueError(f"SSB candidate indices must be in [0, {candidate_count - 1}]")
        indices.append(normalized_index)
    ordered_indices = tuple(sorted(indices))
    if any(not isinstance(block, NRSSBResourceGrid) for block in ssb_blocks.values()):
        raise TypeError("ssb_blocks values must be NRSSBResourceGrid instances")
    first_block = ssb_blocks[ordered_indices[0]]
    if not isinstance(first_block.pbch_dmrs, PBCHDMRSResourceMap):
        raise TypeError("SSB grids must contain PBCH-DM-RS resource maps")
    cell_id = first_block.physical_cell_id
    half_frame_index = first_block.pbch_dmrs.half_frame_index
    if half_frame_index not in (0, 1):
        raise ValueError("PBCH-DM-RS half-frame index must be 0 or 1")
    for candidate_index in ordered_indices:
        block = ssb_blocks[candidate_index]
        if not isinstance(block.pbch_dmrs, PBCHDMRSResourceMap):
            raise TypeError("SSB grids must contain PBCH-DM-RS resource maps")
        if block.physical_cell_id != 3 * block.nid1 + block.nid2:
            raise ValueError("SSB grid identity fields are inconsistent")
        if block.physical_cell_id != cell_id or block.pbch_dmrs.physical_cell_id != cell_id:
            raise ValueError("all SSB blocks must use the same physical cell ID")
        if block.pbch_dmrs.half_frame_index != half_frame_index:
            raise ValueError("all SSB blocks must use the same half-frame index")
        if block.pbch_dmrs.ssb_index_in_half_frame != int(candidate_index):
            raise ValueError("mapping key must match the PBCH-DM-RS candidate index")
        if block.pbch_dmrs.max_candidate_blocks_per_half_frame != candidate_count:
            raise ValueError("PBCH-DM-RS candidate count must match the selected SSB timing case")
        grid = np.asarray(block.grid)
        if grid.shape != (_SSB_SYMBOL_COUNT, _SSB_SUBCARRIER_COUNT):
            raise ValueError("SSB resource grids must have shape (4, 240)")
        if grid.dtype.kind not in "iufc" or grid.dtype.kind == "b":
            raise TypeError("SSB resource grids must be real or complex numeric arrays")
        if not np.isfinite(grid.real).all() or not np.isfinite(grid.imag).all():
            raise ValueError("SSB resource grids must be finite")

    waveform = np.zeros(int(sample_offsets[-1]), dtype=np.complex128)
    for candidate_index in ordered_indices:
        start_symbol = positions[int(candidate_index)]
        cp_lengths = cp_by_symbol[start_symbol : start_symbol + _SSB_SYMBOL_COUNT]
        block_waveform = nr_ssb_ofdm_modulate(
            ssb_blocks[candidate_index].grid,
            fft_size=fft_length,
            cyclic_prefix_lengths=cp_lengths,
        )
        start_sample = int(sample_offsets[start_symbol])
        end_sample = int(sample_offsets[start_symbol + _SSB_SYMBOL_COUNT])
        if block_waveform.shape != (end_sample - start_sample,):
            raise RuntimeError("SSB block samples do not align with the half-frame symbol clock")
        waveform[start_sample:end_sample] = block_waveform
    waveform.setflags(write=False)
    return NRSSBHalfFrameWaveform(
        waveform=waveform,
        subcarrier_spacing_hz=scs_hz,
        fft_size=fft_length,
        sample_rate_hz=sample_rate_hz,
        half_frame_index=half_frame_index,
        candidate_start_symbols=positions,
        transmitted_candidate_indices=tuple(int(index) for index in ordered_indices),
    )


def nr_ssb_ofdm_modulate_radio_frame(
    ssb_blocks_by_half_frame: Mapping[int, Mapping[int, NRSSBResourceGrid]],
    *,
    case: str,
    carrier_frequency_hz: SupportsFloat,
    subcarrier_spacing_hz: SupportsFloat,
    frequency_range: str,
    fft_size: int,
    paired_spectrum: bool = True,
) -> NRSSBRadioFrameWaveform:
    """Build a standards-timed 10 ms frame with candidate-indexed SSB bursts.

    The outer mapping uses absolute half-frame indices 0 and 1. An empty or
    omitted half-frame is represented by zeros. An empty mapping produces a
    silent radio frame. Timing-case selection and OFDM placement are delegated
    to :func:`nr_ssb_ofdm_modulate_half_frame`.
    """
    if not isinstance(ssb_blocks_by_half_frame, Mapping):
        raise TypeError("ssb_blocks_by_half_frame must be a mapping")
    normalized: dict[int, Mapping[int, NRSSBResourceGrid]] = {}
    for key, blocks in ssb_blocks_by_half_frame.items():
        if isinstance(key, (bool, np.bool_)) or not isinstance(key, Integral):
            raise TypeError("half-frame indices must be integers")
        half_frame_index = int(key)
        if half_frame_index not in (0, 1):
            raise ValueError("half-frame indices must be 0 or 1")
        if not isinstance(blocks, Mapping):
            raise TypeError("each half-frame value must be a candidate-index mapping")
        normalized[half_frame_index] = blocks

    positions = ssb_candidate_symbols(
        case,
        carrier_frequency_hz,
        subcarrier_spacing_hz=subcarrier_spacing_hz,
        frequency_range=frequency_range,
        paired_spectrum=paired_spectrum,
    )
    fft_length = _fft_size(fft_size)
    if isinstance(subcarrier_spacing_hz, (bool, np.bool_)) or not isinstance(
        subcarrier_spacing_hz, Real
    ):
        raise TypeError("subcarrier_spacing_hz must be a real scalar in hertz")
    scs_value = float(subcarrier_spacing_hz)
    if not math.isfinite(scs_value) or not scs_value.is_integer():
        raise ValueError("subcarrier_spacing_hz must be a finite integer")
    scs_hz = int(scs_value)
    if scs_hz not in _SCS_TO_MU:
        raise ValueError("unsupported SSB subcarrier spacing")
    sample_rate_hz = fft_length * scs_hz
    if sample_rate_hz % 200:
        raise ValueError("sample rate must represent an exact 5 ms half-frame")
    half_frame_samples = sample_rate_hz // 200

    waveform = np.zeros(2 * half_frame_samples, dtype=np.complex128)
    transmitted: list[tuple[int, ...]] = [(), ()]
    radio_frame_cell_id: int | None = None
    for half_frame_index in (0, 1):
        blocks = normalized.get(half_frame_index, {})
        if not blocks:
            continue
        for block in blocks.values():
            if not isinstance(block, NRSSBResourceGrid):
                raise TypeError("SSB grids must be NRSSBResourceGrid instances")
            if radio_frame_cell_id is None:
                radio_frame_cell_id = block.physical_cell_id
            elif block.physical_cell_id != radio_frame_cell_id:
                raise ValueError("all radio-frame SSB blocks must use the same physical cell ID")
        half_frame = nr_ssb_ofdm_modulate_half_frame(
            blocks,
            case=case,
            carrier_frequency_hz=carrier_frequency_hz,
            subcarrier_spacing_hz=scs_hz,
            frequency_range=frequency_range,
            fft_size=fft_length,
            paired_spectrum=paired_spectrum,
        )
        if half_frame.half_frame_index != half_frame_index:
            raise ValueError("SSB grid half-frame metadata must match its mapping key")
        if (
            half_frame.subcarrier_spacing_hz != scs_hz
            or half_frame.fft_size != fft_length
            or half_frame.sample_rate_hz != sample_rate_hz
            or half_frame.candidate_start_symbols != positions
            or half_frame.waveform.shape != (half_frame_samples,)
        ):
            raise ValueError("half-frame waveform configuration is inconsistent")
        start = half_frame_index * half_frame_samples
        waveform[start : start + half_frame_samples] = half_frame.waveform
        transmitted[half_frame_index] = half_frame.transmitted_candidate_indices

    waveform.setflags(write=False)
    return NRSSBRadioFrameWaveform(
        waveform=waveform,
        subcarrier_spacing_hz=scs_hz,
        fft_size=fft_length,
        sample_rate_hz=sample_rate_hz,
        candidate_start_symbols=positions,
        transmitted_candidate_indices_by_half_frame=(transmitted[0], transmitted[1]),
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
