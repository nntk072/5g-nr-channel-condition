"""Known-occasion PBCH OFDM demodulation and channel-affected decoding."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .pbch_bch import nr_pbch_bch_decode
from .pbch_data import nr_pbch_scramble_bits
from .pbch_dmrs import nr_pbch_dmrs_ssb
from .pbch_mib import PBCHMIB, nr_pbch_mib_parse
from .pbch_payload import PBCHPayloadRecovery, nr_pbch_payload_recover
from .ssb_grid import _pbch_data_coordinates

_SSB_SUBCARRIERS = 240
_SSB_SYMBOLS = 4
_PBCH_DATA_BITS = 864


@dataclass(frozen=True)
class PBCHOFDMReception:
    """Decoded output and intermediate measurements for a known SSB occasion."""

    crc_valid: bool
    mib: PBCHMIB | None
    payload_recovery: PBCHPayloadRecovery | None
    decoded_transport_block: NDArray[np.uint8]
    channel_estimates: NDArray[np.complex128]
    equalized_pbch_symbols: NDArray[np.complex128]
    descrambled_llrs: NDArray[np.float64]


def nr_pbch_receive_ofdm(
    received_samples: NDArray[np.generic],
    *,
    fft_size: int,
    cyclic_prefix_lengths: Sequence[int],
    physical_cell_id: int,
    ssb_index_in_half_frame: int,
    half_frame_index: int,
    max_candidate_blocks_per_half_frame: int,
    noise_variance: float,
    device: str = "cpu",
    list_size: int = 8,
) -> PBCHOFDMReception:
    """Decode one CP-OFDM SSB whose cell, candidate, and timing are known.

    ``noise_variance`` is complex-sample variance (E[|n|^2]); the unitary FFT
    preserves this variance. PBCH-DM-RS is independently interpolated across
    frequency for each data-bearing SSB symbol. The function assumes one
    transmit and receive chain, a static channel over the four symbols, and
    perfect symbol timing/frequency synchronization. It performs neither
    PSS/SSS acquisition nor broad receiver qualification.
    """
    fft_length = _positive_integer(fft_size, "fft_size")
    if fft_length <= _SSB_SUBCARRIERS or fft_length % 2:
        raise ValueError("fft_size must be even and greater than 240")
    cp_lengths = _validate_cp_lengths(cyclic_prefix_lengths, fft_length)
    variance = _positive_real(noise_variance, "noise_variance")
    samples = np.asarray(received_samples)
    if samples.ndim != 1 or samples.dtype.kind != "c" or not np.isfinite(samples).all():
        raise ValueError("received_samples must be a finite one-dimensional complex array")
    samples = np.asarray(samples, dtype=np.complex128)
    expected_samples = sum(fft_length + cp for cp in cp_lengths)
    if samples.shape != (expected_samples,):
        raise ValueError(f"received_samples must have shape ({expected_samples},)")

    received_grid = _demodulate_known_ssb(samples, fft_length, cp_lengths)
    dmrs = nr_pbch_dmrs_ssb(
        physical_cell_id,
        ssb_index_in_half_frame,
        half_frame_index=half_frame_index,
        max_candidate_blocks_per_half_frame=max_candidate_blocks_per_half_frame,
    )
    channel = np.empty((_SSB_SYMBOLS - 1, _SSB_SUBCARRIERS), dtype=np.complex128)
    for output_symbol, ssb_symbol in enumerate((1, 2, 3)):
        pilot_mask = dmrs.ofdm_symbols == ssb_symbol
        pilot_subcarriers = dmrs.subcarriers[pilot_mask].astype(np.intp)
        if pilot_subcarriers.size < 2:
            raise RuntimeError("each PBCH data symbol must contain at least two DM-RS pilots")
        pilot_channel = received_grid[ssb_symbol, pilot_subcarriers] / dmrs.sequence[pilot_mask]
        channel[output_symbol] = _interpolate_channel(pilot_subcarriers, pilot_channel)

    data_subcarriers, data_symbols = _pbch_data_coordinates(dmrs)
    symbol_rows = data_symbols.astype(np.intp) - 1
    data_channel = channel[symbol_rows, data_subcarriers.astype(np.intp)]
    if np.any(np.abs(data_channel) <= np.finfo(np.float64).tiny):
        raise ValueError("channel estimate contains a zero at a PBCH data resource element")
    equalized = received_grid[data_symbols, data_subcarriers] / data_channel
    equalized = np.asarray(equalized, dtype=np.complex128)
    equalized.setflags(write=False)

    # With normalized QPSK, each real component has variance noise_variance/2.
    # LLRs are log(P(bit=1)/P(bit=0)), matching the BCH decoder interface.
    bit_variance = variance / (2.0 * np.square(np.abs(data_channel)))
    llrs = np.empty(_PBCH_DATA_BITS, dtype=np.float64)
    llrs[0::2] = -math.sqrt(2.0) * equalized.real / bit_variance
    llrs[1::2] = -math.sqrt(2.0) * equalized.imag / bit_variance
    scramble_mask = nr_pbch_scramble_bits(
        np.zeros(_PBCH_DATA_BITS, dtype=np.uint8),
        physical_cell_id,
        ssb_index_in_half_frame,
        max_candidate_blocks_per_half_frame=max_candidate_blocks_per_half_frame,
    )
    llrs[scramble_mask.astype(bool)] *= -1.0
    llrs.setflags(write=False)

    decoded, crc_status = nr_pbch_bch_decode(llrs, device=device, list_size=list_size)
    crc_valid = bool(crc_status)
    decoded.setflags(write=False)
    recovery: PBCHPayloadRecovery | None = None
    mib: PBCHMIB | None = None
    if crc_valid:
        recovery = nr_pbch_payload_recover(
            decoded,
            physical_cell_id,
            max_candidate_blocks_per_half_frame=max_candidate_blocks_per_half_frame,
        )
        if int(recovery.higher_layer_bits[0]) == 0:
            mib = nr_pbch_mib_parse(recovery.higher_layer_bits)

    channel.setflags(write=False)
    return PBCHOFDMReception(
        crc_valid=crc_valid,
        mib=mib,
        payload_recovery=recovery,
        decoded_transport_block=decoded,
        channel_estimates=channel,
        equalized_pbch_symbols=equalized,
        descrambled_llrs=llrs,
    )


def _demodulate_known_ssb(
    samples: NDArray[np.generic], fft_size: int, cp_lengths: tuple[int, ...]
) -> NDArray[np.complex128]:
    grid = np.empty((_SSB_SYMBOLS, _SSB_SUBCARRIERS), dtype=np.complex128)
    offset = 0
    lower = fft_size // 2 - _SSB_SUBCARRIERS // 2
    upper = lower + _SSB_SUBCARRIERS
    for symbol, cp_length in enumerate(cp_lengths):
        start = offset + cp_length
        useful = np.asarray(samples[start : start + fft_size], dtype=np.complex128)
        spectrum = np.fft.fftshift(np.fft.fft(useful, norm="ortho"))
        grid[symbol] = spectrum[lower:upper]
        offset += cp_length + fft_size
    return grid


def _interpolate_channel(
    pilot_subcarriers: NDArray[np.intp], pilot_channel: NDArray[np.complex128]
) -> NDArray[np.complex128]:
    order = np.argsort(pilot_subcarriers, kind="stable")
    carriers = pilot_subcarriers[order]
    estimates = pilot_channel[order]
    return np.interp(np.arange(_SSB_SUBCARRIERS), carriers, estimates.real) + 1j * np.interp(
        np.arange(_SSB_SUBCARRIERS), carriers, estimates.imag
    )


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _positive_real(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _validate_cp_lengths(values: Sequence[int], fft_size: int) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or len(values) != _SSB_SYMBOLS:
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
