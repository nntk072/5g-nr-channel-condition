"""Blind PSS/SSS acquisition and finite PBCH occasion search."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .nr_sync import NRCellDetection, acquire_nr_cell
from .pbch_ofdm_receiver import PBCHOFDMReception, nr_pbch_receive_ofdm

_SUPPORTED_CANDIDATE_COUNTS = (4, 8, 10, 20, 64)


class PBCHAcquisitionError(RuntimeError):
    """Cell acquisition succeeded, but the supplied PBCH occasion set did not decode uniquely."""


@dataclass(frozen=True)
class PBCHAcquisitionReception:
    """Blind PSS/SSS result and the unique CRC-valid PBCH occasion."""

    cell_detection: NRCellDetection
    ssb_index_in_half_frame: int
    half_frame_index: int
    pbch_reception: PBCHOFDMReception


def nr_pbch_acquire_and_receive(
    capture: NDArray[np.generic],
    *,
    fft_size: int,
    cyclic_prefix_lengths: Sequence[int],
    sampling_frequency_hz: float,
    max_candidate_blocks_per_half_frame: int,
    noise_variance: float | Sequence[float] | NDArray[np.generic],
    cfo_hypotheses_hz: tuple[float, ...] = (0.0,),
    cfo_guard_samples: int = 0,
    minimum_pss_score: float = 0.3,
    minimum_sss_score: float = 0.4,
    device: str = "cpu",
    list_size: int = 8,
) -> PBCHAcquisitionReception:
    """Acquire PCI/timing/CFO, then search DM-RS-visible candidate/half-frame pairs.

    A vector is one receive chain; a matrix is shaped ``[receive_chain, sample]``.
    Matrix input combines PSS/SSS search energy noncoherently across branches
    and uses inverse-noise weighting for PSS/SSS scoring and PBCH MRC.
    ``noise_variance`` may be shared or provide one independent complex-sample
    variance per receive chain. The receiver assumes a
    static channel over one four-symbol SSB, known FFT/CP/numerology, a supplied
    CFO search grid, and a capture that contains the PSS and its later
    SSS/PBCH symbols. A complete 10 ms radio-frame capture is also supported
    when the configured timing and CPs match that frame. For candidate counts above eight, DM-RS identifies only
    the three least-significant index bits; the remaining bits are reconstructed
    from the CRC-valid PBCH payload. Exactly one candidate/half-frame result is
    required. This does not locate an SSB in a continuous radio stream or
    qualify operation across channels, numerologies, or antenna topologies.
    """
    fft_length = _positive_integer(fft_size, "fft_size")
    cp_lengths = _validate_cp_lengths(cyclic_prefix_lengths, fft_length)
    if cp_lengths[0] != cp_lengths[2]:
        raise ValueError("PSS and SSS cyclic-prefix lengths must match for acquisition")
    sample_rate_hz = _positive_real(sampling_frequency_hz, "sampling_frequency_hz")
    list_size = _positive_integer(list_size, "list_size")
    if not isinstance(device, str):
        raise TypeError("device must be a string")
    if not device:
        raise ValueError("device must not be empty")
    if isinstance(max_candidate_blocks_per_half_frame, (bool, np.bool_)) or not isinstance(
        max_candidate_blocks_per_half_frame, Integral
    ):
        raise TypeError("max_candidate_blocks_per_half_frame must be an integer")
    candidate_count = int(max_candidate_blocks_per_half_frame)
    if candidate_count not in _SUPPORTED_CANDIDATE_COUNTS:
        raise ValueError("max_candidate_blocks_per_half_frame must be one of 4, 8, 10, 20, or 64")

    samples = np.asarray(capture)
    if samples.ndim not in (1, 2) or samples.dtype.kind != "c" or not np.isfinite(samples).all():
        raise ValueError("capture must be a finite complex vector or receive-chain matrix")
    single_receive_chain = samples.ndim == 1
    samples = np.asarray(samples, dtype=np.complex128)
    if single_receive_chain:
        samples = samples[None, :]
    detection = acquire_nr_cell(
        samples,
        fft_size=fft_length,
        cyclic_prefix_length=cp_lengths[0],
        sampling_frequency_hz=sample_rate_hz,
        noise_variance=noise_variance,
        cfo_hypotheses_hz=cfo_hypotheses_hz,
        cfo_guard_samples=cfo_guard_samples,
        minimum_pss_score=minimum_pss_score,
        minimum_sss_score=minimum_sss_score,
    )

    frame_samples = sum(fft_length + cp for cp in cp_lengths)
    frame_start = detection.sample_offset
    frame_end = frame_start + frame_samples
    if frame_end > samples.shape[1]:
        raise ValueError("capture ends before the complete SS/PBCH block")
    frame_indexes = np.arange(frame_start, frame_end, dtype=np.float64)
    aligned_ssb = (
        samples[:, frame_start:frame_end]
        * np.exp(-2j * np.pi * detection.cfo_hz * frame_indexes / sample_rate_hz)[None, :]
    )
    if single_receive_chain:
        aligned_ssb = aligned_ssb[0]

    decoded: list[tuple[int, int, PBCHOFDMReception]] = []
    dmrs_candidate_count = min(candidate_count, 8)
    for half_frame in (0, 1):
        for candidate_lsb in range(dmrs_candidate_count):
            reception = nr_pbch_receive_ofdm(
                aligned_ssb,
                fft_size=fft_length,
                cyclic_prefix_lengths=cp_lengths,
                physical_cell_id=detection.physical_cell_id,
                ssb_index_in_half_frame=candidate_lsb,
                half_frame_index=half_frame,
                max_candidate_blocks_per_half_frame=candidate_count,
                noise_variance=noise_variance,
                device=device,
                list_size=list_size,
            )
            if reception.crc_valid:
                recovery = reception.payload_recovery
                if recovery is None or recovery.half_frame_index != half_frame:
                    continue
                candidate = _reconstruct_candidate_index(
                    candidate_lsb, recovery.ssb_index_msb_bits, candidate_count
                )
                if candidate is not None:
                    decoded.append((candidate, half_frame, reception))

    if not decoded:
        raise PBCHAcquisitionError("no candidate SSB index/half-frame hypothesis passed BCH CRC")
    if len(decoded) != 1:
        hypotheses = [(candidate, half) for candidate, half, _ in decoded]
        raise PBCHAcquisitionError(f"PBCH occasion search was ambiguous: {hypotheses}")
    candidate, half_frame, reception = decoded[0]
    return PBCHAcquisitionReception(
        cell_detection=detection,
        ssb_index_in_half_frame=candidate,
        half_frame_index=half_frame,
        pbch_reception=reception,
    )


def _reconstruct_candidate_index(
    candidate_lsb: int, candidate_msb_bits: tuple[int, ...], candidate_count: int
) -> int | None:
    """Join the DM-RS-visible low bits and PBCH-payload high bits."""
    if candidate_count <= 8:
        return candidate_lsb if not candidate_msb_bits else None
    expected_msb_bits = (candidate_count - 1).bit_length() - 3
    if len(candidate_msb_bits) != expected_msb_bits or any(
        bit not in (0, 1) for bit in candidate_msb_bits
    ):
        return None
    candidate_msb = 0
    for bit in candidate_msb_bits:
        candidate_msb = (candidate_msb << 1) | bit
    candidate = (candidate_msb << 3) | candidate_lsb
    return candidate if candidate < candidate_count else None


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
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must be finite and positive") from error
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _validate_cp_lengths(values: Sequence[int], fft_size: int) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or len(values) != 4:
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
