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
from .pbch_mib import PBCHBCCHMessage, PBCHMIB, nr_bcch_bch_message_parse
from .pbch_payload import PBCHPayloadRecovery, nr_pbch_payload_recover
from .ssb_grid import _pbch_data_coordinates
from .ssb_ofdm import nr_ssb_ofdm_demodulate

_SSB_SUBCARRIERS = 240
_SSB_SYMBOLS = 4
_PBCH_DATA_BITS = 864


@dataclass(frozen=True)
class PBCHOFDMReception:
    """Decoded output and intermediate measurements for a known SSB occasion.

    ``channel_estimates`` has shape ``[symbol, subcarrier]`` for one receive
    chain and ``[receive_chain, symbol, subcarrier]`` for receive diversity.
    """

    crc_valid: bool
    mib: PBCHMIB | None
    bcch_bch_message: PBCHBCCHMessage | None
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
    noise_variance: float | Sequence[float] | NDArray[np.generic],
    device: str = "cpu",
    list_size: int = 8,
) -> PBCHOFDMReception:
    """Decode one CP-OFDM SSB whose cell, candidate, and timing are known.

    ``noise_variance`` is complex-sample variance (E[|n|^2]); the unitary FFT
    preserves this variance. It may be a scalar shared by all receive chains
    or one variance per chain. PBCH-DM-RS is independently interpolated across
    frequency for each data-bearing SSB symbol and chain, then combined with
    noise-whitened maximal-ratio combining. The function assumes one transmit
    chain, independent receiver noise, a static channel over the four symbols,
    and perfect symbol timing/frequency synchronization. It performs neither
    PSS/SSS acquisition nor broad receiver qualification.
    """
    fft_length = _positive_integer(fft_size, "fft_size")
    if fft_length <= _SSB_SUBCARRIERS or fft_length % 2:
        raise ValueError("fft_size must be even and greater than 240")
    cp_lengths = _validate_cp_lengths(cyclic_prefix_lengths, fft_length)
    samples = np.asarray(received_samples)
    if samples.ndim not in (1, 2) or samples.dtype.kind != "c" or not np.isfinite(samples).all():
        raise ValueError("received_samples must be a finite complex vector or receive-chain matrix")
    single_receive_chain = samples.ndim == 1
    samples = np.asarray(samples, dtype=np.complex128)
    if single_receive_chain:
        samples = samples[None, :]
    variances = _noise_variance_by_chain(noise_variance, samples.shape[0])
    expected_samples = sum(fft_length + cp for cp in cp_lengths)
    if samples.ndim != 2 or samples.shape[0] == 0 or samples.shape[1] != expected_samples:
        raise ValueError(
            "received_samples must have shape "
            f"({expected_samples},) or (num_receive_chains, {expected_samples})"
        )

    received_grids = np.stack(
        [
            nr_ssb_ofdm_demodulate(
                chain_samples, fft_size=fft_length, cyclic_prefix_lengths=cp_lengths
            )
            for chain_samples in samples
        ],
        axis=0,
    )
    dmrs = nr_pbch_dmrs_ssb(
        physical_cell_id,
        ssb_index_in_half_frame,
        half_frame_index=half_frame_index,
        max_candidate_blocks_per_half_frame=max_candidate_blocks_per_half_frame,
    )
    channel = np.empty((samples.shape[0], _SSB_SYMBOLS - 1, _SSB_SUBCARRIERS), dtype=np.complex128)
    for receive_chain, received_grid in enumerate(received_grids):
        for output_symbol, ssb_symbol in enumerate((1, 2, 3)):
            pilot_mask = dmrs.ofdm_symbols == ssb_symbol
            pilot_subcarriers = dmrs.subcarriers[pilot_mask].astype(np.intp)
            if pilot_subcarriers.size < 2:
                raise RuntimeError("each PBCH data symbol must contain at least two DM-RS pilots")
            pilot_channel = received_grid[ssb_symbol, pilot_subcarriers] / dmrs.sequence[pilot_mask]
            channel[receive_chain, output_symbol] = _interpolate_channel(
                pilot_subcarriers, pilot_channel
            )

    data_subcarriers, data_symbols = _pbch_data_coordinates(dmrs)
    symbol_rows = data_symbols.astype(np.intp) - 1
    data_channel = channel[:, symbol_rows, data_subcarriers.astype(np.intp)]
    weighted_channel_power = np.square(np.abs(data_channel)) / variances[:, None]
    combined_channel_power = np.sum(weighted_channel_power, axis=0)
    if np.any(combined_channel_power <= np.finfo(np.float64).tiny):
        raise ValueError("combined channel estimate is zero at a PBCH data resource element")
    received_data = received_grids[:, data_symbols, data_subcarriers.astype(np.intp)]
    equalized = (
        np.sum(np.conj(data_channel) * received_data / variances[:, None], axis=0)
        / combined_channel_power
    )
    equalized = np.asarray(equalized, dtype=np.complex128)
    equalized.setflags(write=False)

    # With normalized QPSK, each real component has variance 1/(2*combined gain).
    # LLRs are log(P(bit=1)/P(bit=0)), matching the BCH decoder interface.
    bit_variance = 1.0 / (2.0 * combined_channel_power)
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
    bcch_bch_message: PBCHBCCHMessage | None = None
    if crc_valid:
        recovery = nr_pbch_payload_recover(
            decoded,
            physical_cell_id,
            max_candidate_blocks_per_half_frame=max_candidate_blocks_per_half_frame,
        )
        bcch_bch_message = nr_bcch_bch_message_parse(recovery.higher_layer_bits)
        if isinstance(bcch_bch_message, PBCHMIB):
            mib = bcch_bch_message

    channel_result = channel[0] if single_receive_chain else channel
    channel_result.setflags(write=False)
    return PBCHOFDMReception(
        crc_valid=crc_valid,
        mib=mib,
        bcch_bch_message=bcch_bch_message,
        payload_recovery=recovery,
        decoded_transport_block=decoded,
        channel_estimates=channel_result,
        equalized_pbch_symbols=equalized,
        descrambled_llrs=llrs,
    )


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


def _noise_variance_by_chain(value: object, num_receive_chains: int) -> NDArray[np.float64]:
    """Expand a shared complex-noise variance or validate per-chain values."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("noise_variance must contain real numeric values, not booleans")
    if isinstance(value, Real):
        return np.full(
            num_receive_chains, _positive_real(value, "noise_variance"), dtype=np.float64
        )
    object_values = np.asarray(value, dtype=object)
    if any(isinstance(item, (bool, np.bool_)) for item in object_values.flat):
        raise TypeError("noise_variance must contain real numeric values, not booleans")
    variances = np.asarray(value)
    if variances.shape != (num_receive_chains,):
        raise ValueError("noise_variance must be scalar or contain one value per receive chain")
    if variances.dtype.kind not in "iuf":
        raise TypeError("noise_variance values must be real numbers")
    variances = np.asarray(variances, dtype=np.float64)
    if not np.isfinite(variances).all() or np.any(variances <= 0.0):
        raise ValueError("noise_variance values must be finite and positive")
    return variances


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
