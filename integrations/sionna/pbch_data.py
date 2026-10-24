"""PBCH data-bit scrambling and QPSK modulation for one SS/PBCH block."""

from __future__ import annotations

from numbers import Integral

import numpy as np
from numpy.typing import NDArray

from .pbch_dmrs import _gold_bits

_PBCH_BIT_COUNT = 864
_CELL_ID_MAX = 1007


def _binary_vector(bits: NDArray[np.generic], *, name: str) -> NDArray[np.uint8]:
    array = np.asarray(bits)
    if array.shape != (_PBCH_BIT_COUNT,):
        raise ValueError(f"{name} must have shape (864,)")
    if array.dtype.kind not in "biu" or np.any(array < 0) or np.any(array > 1):
        raise ValueError(f"{name} must contain only binary values")
    return np.asarray(array, dtype=np.uint8)


def nr_pbch_scramble_bits(
    bits: NDArray[np.generic],
    physical_cell_id: int,
    ssb_index_in_half_frame: int,
    *,
    max_candidate_blocks_per_half_frame: int,
) -> NDArray[np.uint8]:
    """Scramble the 864 rate-matched PBCH bits for a known SSB candidate.

    Implements TS 38.211 clause 7.3.3.1. The candidate-index sequence offset
    is ``v * M_bit`` (``M_bit=864``), with ``v`` selected from the two or
    three least-significant SSB-index bits for candidate counts four or
    eight, respectively. Half-frame index does not enter this data-scrambler
    initialization or offset.
    """
    source = _binary_vector(bits, name="bits")
    cell_id = _bounded_integer(physical_cell_id, "physical_cell_id", 0, _CELL_ID_MAX)
    if isinstance(max_candidate_blocks_per_half_frame, (bool, np.bool_)) or not isinstance(
        max_candidate_blocks_per_half_frame, Integral
    ):
        raise TypeError("max_candidate_blocks_per_half_frame must be an integer")
    candidate_count = int(max_candidate_blocks_per_half_frame)
    if candidate_count not in (4, 8):
        raise ValueError("max_candidate_blocks_per_half_frame must be 4 or 8")
    ssb_index = _bounded_integer(
        ssb_index_in_half_frame,
        "ssb_index_in_half_frame",
        0,
        candidate_count - 1,
    )

    sequence = _gold_bits(
        cell_id,
        _PBCH_BIT_COUNT,
        sequence_offset=ssb_index * _PBCH_BIT_COUNT,
    )
    scrambled = np.bitwise_xor(source, sequence)
    scrambled.setflags(write=False)
    return scrambled


def nr_pbch_qpsk_modulate(bits: NDArray[np.generic]) -> NDArray[np.complex128]:
    """Map 864 PBCH bits to 432 TS 38.211 clause 5.1.3 QPSK symbols."""
    binary = _binary_vector(bits, name="bits")
    symbols = (
        (1.0 - 2.0 * binary[0::2].astype(np.float64))
        + 1j * (1.0 - 2.0 * binary[1::2].astype(np.float64))
    ) / np.sqrt(2.0)
    symbols.setflags(write=False)
    return symbols


def _bounded_integer(value: object, name: str, lower: int, upper: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if not lower <= result <= upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}]")
    return result
