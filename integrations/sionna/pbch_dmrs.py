"""PBCH demodulation-reference sequence and resource map for an SS/PBCH block.

The implementation follows 3GPP TS 38.211 V19.1.0 clauses 5.2.1,
7.4.1.4.1, and 7.4.3.1.3. It returns the unscaled 144-symbol PBCH-DMRS
sequence and its ordered resource elements within a 240-subcarrier,
four-symbol SS/PBCH block. It does not encode PBCH bits, apply TS 38.213
power scaling, or locate a block in a radio frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import NDArray

_CELL_ID_MAX = 1007
_SSB_SUBCARRIER_COUNT = 240
_DMRS_SYMBOL_COUNT = 144
_GOLD_SEQUENCE_OFFSET = 1600
_PBCH_DMRS_SYMBOLS = (1, 2, 3)
_SUPPORTED_CANDIDATE_COUNTS = (4, 8, 10, 20, 64)


@dataclass(frozen=True)
class PBCHDMRSResourceMap:
    """Unscaled PBCH-DMRS sequence and zero-based SS/PBCH-block coordinates.

    ``subcarriers[i]``, ``ofdm_symbols[i]``, and ``sequence[i]`` describe
    one resource element. Entries are ordered by subcarrier first, then OFDM
    symbol, as specified for PBCH-DMRS mapping.
    """

    physical_cell_id: int
    ssb_index_in_half_frame: int
    half_frame_index: int
    max_candidate_blocks_per_half_frame: int
    c_init: int
    subcarriers: NDArray[np.int16]
    ofdm_symbols: NDArray[np.int8]
    sequence: NDArray[np.complex128]


def _integer_in_range(value: object, name: str, lower: int, upper: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    integer = int(value)
    if not lower <= integer <= upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}]")
    return integer


def _gold_bits(c_init: int, count: int, *, sequence_offset: int = 0) -> NDArray[np.uint8]:
    """Generate clause 5.2.1 Gold bits at an offset after the 1600-chip skip."""
    end = _GOLD_SEQUENCE_OFFSET + sequence_offset + count
    x1 = np.zeros(end + 31, dtype=np.uint8)
    x2 = np.zeros(end + 31, dtype=np.uint8)
    x1[0] = 1
    for index in range(31):
        x2[index] = (c_init >> index) & 1

    for index in range(end):
        x1[index + 31] = x1[index + 3] ^ x1[index]
        x2[index + 31] = x2[index + 3] ^ x2[index + 2] ^ x2[index + 1] ^ x2[index]

    start = _GOLD_SEQUENCE_OFFSET + sequence_offset
    return x1[start:end] ^ x2[start:end]


def _pbch_dmrs_coordinates(physical_cell_id: int) -> tuple[NDArray[np.int16], NDArray[np.int8]]:
    """Return the standard's DM-RS REs in increasing (subcarrier, symbol) order."""
    offset = physical_cell_id % 4
    coordinates = [
        (subcarrier, symbol)
        for subcarrier in range(offset, _SSB_SUBCARRIER_COUNT, 4)
        for symbol in _PBCH_DMRS_SYMBOLS
        if symbol != 2 or subcarrier < 48 or subcarrier >= 192
    ]
    if len(coordinates) != _DMRS_SYMBOL_COUNT:
        raise RuntimeError("internal PBCH-DMRS coordinate count is inconsistent")

    subcarriers = np.fromiter((item[0] for item in coordinates), dtype=np.int16)
    symbols = np.fromiter((item[1] for item in coordinates), dtype=np.int8)
    return subcarriers, symbols


def nr_pbch_dmrs_ssb(
    physical_cell_id: int,
    ssb_index_in_half_frame: int,
    *,
    half_frame_index: int,
    max_candidate_blocks_per_half_frame: int,
) -> PBCHDMRSResourceMap:
    """Generate the PBCH-DMRS sequence and map for one SS/PBCH-block occasion.

    Args:
        physical_cell_id: NR physical cell identity, from 0 through 1007.
        ssb_index_in_half_frame: Zero-based candidate block index within the
            selected half-frame. Its allowed range is determined by
            ``max_candidate_blocks_per_half_frame``.
        half_frame_index: 0 for the first half-frame in a frame, 1 for the
            second half-frame.
        max_candidate_blocks_per_half_frame: Supported values are 4, 8, 10,
            20, and 64. Select the value for the carrier, SSB pattern, and
            operation using TS 38.213.

    Returns:
        A frozen record containing the 144 QPSK reference symbols, their
        zero-based SS/PBCH-block coordinates, and the sequence initialization.

    Raises:
        TypeError: If an index is not an integer or is a boolean.
        ValueError: If an index or candidate count is outside the supported
            range.

    This maps only PBCH-DM-RS for one known candidate occasion. It does not
    infer the candidate from PSS/SSS, map PBCH data, or decode the MIB.
    """
    cell_id = _integer_in_range(physical_cell_id, "physical_cell_id", 0, _CELL_ID_MAX)
    if isinstance(max_candidate_blocks_per_half_frame, (bool, np.bool_)) or not isinstance(
        max_candidate_blocks_per_half_frame, Integral
    ):
        raise TypeError("max_candidate_blocks_per_half_frame must be an integer")
    candidate_count = int(max_candidate_blocks_per_half_frame)
    if candidate_count not in _SUPPORTED_CANDIDATE_COUNTS:
        raise ValueError("max_candidate_blocks_per_half_frame must be one of 4, 8, 10, 20, or 64")
    ssb_index = _integer_in_range(
        ssb_index_in_half_frame,
        "ssb_index_in_half_frame",
        0,
        candidate_count - 1,
    )
    half_frame = _integer_in_range(half_frame_index, "half_frame_index", 0, 1)

    # For L_max=4, the candidate index includes the half-frame offset. For
    # L_max>4, the three least-significant candidate-index bits are used.
    i_bar_ssb = ssb_index + 4 * half_frame if candidate_count == 4 else ssb_index % 8
    c_init = (
        (1 << 11) * (i_bar_ssb + 1) * (cell_id // 4 + 1) + (1 << 6) * (i_bar_ssb + 1) + cell_id % 4
    )
    bits = _gold_bits(c_init, 2 * _DMRS_SYMBOL_COUNT)
    sequence = (
        (1.0 - 2.0 * bits[0::2].astype(np.float64))
        + 1j * (1.0 - 2.0 * bits[1::2].astype(np.float64))
    ) / np.sqrt(2.0)
    subcarriers, symbols = _pbch_dmrs_coordinates(cell_id)

    for array in (subcarriers, symbols, sequence):
        array.setflags(write=False)
    return PBCHDMRSResourceMap(
        physical_cell_id=cell_id,
        ssb_index_in_half_frame=ssb_index,
        half_frame_index=half_frame,
        max_candidate_blocks_per_half_frame=candidate_count,
        c_init=c_init,
        subcarriers=subcarriers,
        ofdm_symbols=symbols,
        sequence=sequence,
    )
