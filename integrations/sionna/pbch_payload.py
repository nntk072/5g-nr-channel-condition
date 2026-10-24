"""PBCH payload generation, interleaving, scrambling, and recovery.

Implements the 32-bit payload procedures in TS 38.212 clauses 7.1.1 and
7.1.2. The 24 higher-layer bits are accepted as an already serialized bit
sequence; decoding their MIB/RRC field semantics is outside this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import NDArray

from .pbch_dmrs import _gold_bits

_CELL_ID_MAX = 1007
_PAYLOAD_SIZE = 32
_HIGHER_LAYER_SIZE = 24
_INTERLEAVER = np.asarray(
    [
        16,
        23,
        18,
        17,
        8,
        30,
        10,
        6,
        24,
        7,
        0,
        5,
        3,
        2,
        1,
        4,
        9,
        11,
        12,
        13,
        14,
        15,
        19,
        20,
        21,
        22,
        25,
        26,
        27,
        28,
        29,
        31,
    ],
    dtype=np.int8,
)


@dataclass(frozen=True)
class PBCHPayloadRecovery:
    """Recovered BCCH-BCH message bits and PBCH timing/index payload fields."""

    higher_layer_bits: NDArray[np.uint8]
    system_frame_number: int
    half_frame_index: int
    max_candidate_blocks_per_half_frame: int
    k_ssb_msb: int | None
    ssb_index_msb_bits: tuple[int, ...]
    pdcch_repetition: int | None


def nr_pbch_payload_generate(
    higher_layer_bits: NDArray[np.generic],
    physical_cell_id: int,
    system_frame_number: int,
    half_frame_index: int,
    ssb_index_in_half_frame: int,
    *,
    max_candidate_blocks_per_half_frame: int,
    k_ssb_msb: int = 0,
    pdcch_repetition: int = 0,
) -> NDArray[np.uint8]:
    """Generate the scrambled 32-bit PBCH transport-block input.

    ``higher_layer_bits`` must contain the serialized 24-bit BCCH-BCH
    message. For the MIB alternative, its choice bit is zero and the six SFN
    MSBs occupy bits 1 through 6. Timing and candidate-index bits are
    appended, TS 38.212's category-dependent payload interleaver is applied,
    then payload scrambling is applied. The result can be passed to
    :func:`nr_pbch_bch_encode`.

    ``max_candidate_blocks_per_half_frame`` supports 4, 8, 10, 20, or 64.
    The ``k_ssb_msb`` input is used for candidate counts 4, 8, 10, and 20.
    ``pdcch_repetition`` is the Release 19 bit used in place of the former
    reserved final timing bit for candidate counts 4 and 8.
    """
    payload = _binary_vector(higher_layer_bits, _HIGHER_LAYER_SIZE, "higher_layer_bits")
    cell_id = _integer(physical_cell_id, "physical_cell_id", 0, _CELL_ID_MAX)
    sfn = _integer(system_frame_number, "system_frame_number", 0, 1023)
    expected_sfn_msb = np.asarray([(sfn >> bit) & 1 for bit in range(9, 3, -1)], dtype=np.uint8)
    if int(payload[0]) != 0:
        raise ValueError("higher_layer_bits must select the BCCH-BCH MIB alternative")
    if not np.array_equal(payload[1:7], expected_sfn_msb):
        raise ValueError("higher_layer_bits bits 1 through 6 must contain the six SFN MSBs")
    half_frame = _integer(half_frame_index, "half_frame_index", 0, 1)
    candidate_count = _candidate_count(max_candidate_blocks_per_half_frame)
    ssb_index = _integer(ssb_index_in_half_frame, "ssb_index_in_half_frame", 0, candidate_count - 1)
    k_ssb = _integer(k_ssb_msb, "k_ssb_msb", 0, 1)
    repetition = _integer(pdcch_repetition, "pdcch_repetition", 0, 1)

    raw = np.zeros(_PAYLOAD_SIZE, dtype=np.uint8)
    raw[:_HIGHER_LAYER_SIZE] = payload
    raw[24:28] = np.asarray([(sfn >> bit) & 1 for bit in (3, 2, 1, 0)], dtype=np.uint8)
    raw[28] = half_frame

    if candidate_count in (4, 8):
        raw[29] = k_ssb
        raw[31] = repetition
    elif candidate_count in (10, 20):
        raw[29] = k_ssb
        n_index_bits = 1 if candidate_count == 10 else 2
        for offset in range(n_index_bits):
            most_significant_index_bit = 3 if candidate_count == 10 else 4
            raw[32 - n_index_bits + offset] = (
                ssb_index >> (most_significant_index_bit - offset)
            ) & 1
    else:
        raw[29:32] = np.asarray([(ssb_index >> bit) & 1 for bit in (5, 4, 3)], dtype=np.uint8)

    interleaved = _interleave_payload(raw)
    return nr_pbch_payload_scramble(
        interleaved,
        cell_id,
        sfn,
        max_candidate_blocks_per_half_frame=candidate_count,
    )


def nr_pbch_payload_scramble(
    interleaved_payload: NDArray[np.generic],
    physical_cell_id: int,
    system_frame_number: int,
    *,
    max_candidate_blocks_per_half_frame: int,
) -> NDArray[np.uint8]:
    """Apply TS 38.212 payload scrambling to a 32-bit interleaved payload.

    The operation is self-inverse. Bits carrying the half-frame index, the
    candidate SSB index, and SFN's second and third least-significant bits
    are excluded from XOR scrambling. This is distinct from scrambling the
    later 864 rate-matched PBCH bits under TS 38.211.
    """
    source = _binary_vector(interleaved_payload, _PAYLOAD_SIZE, "interleaved_payload")
    cell_id = _integer(physical_cell_id, "physical_cell_id", 0, _CELL_ID_MAX)
    sfn = _integer(system_frame_number, "system_frame_number", 0, 1023)
    candidate_count = _candidate_count(max_candidate_blocks_per_half_frame)

    skip_mask = _payload_scramble_skip_mask(candidate_count)
    sequence_length = _PAYLOAD_SIZE - int(np.count_nonzero(skip_mask))
    v = (((sfn >> 2) & 1) << 1) | ((sfn >> 1) & 1)
    sequence = _gold_bits(cell_id, sequence_length, sequence_offset=v * sequence_length)

    result = np.array(source, copy=True)
    result[~skip_mask] ^= sequence
    result.setflags(write=False)
    return result


def nr_pbch_payload_descramble(
    interleaved_payload: NDArray[np.generic],
    physical_cell_id: int,
    system_frame_number: int,
    *,
    max_candidate_blocks_per_half_frame: int,
) -> NDArray[np.uint8]:
    """Undo payload scrambling; equivalent to applying it a second time."""
    return nr_pbch_payload_scramble(
        interleaved_payload,
        physical_cell_id,
        system_frame_number,
        max_candidate_blocks_per_half_frame=max_candidate_blocks_per_half_frame,
    )


def nr_pbch_payload_recover(
    scrambled_interleaved_payload: NDArray[np.generic],
    physical_cell_id: int,
    *,
    max_candidate_blocks_per_half_frame: int,
) -> PBCHPayloadRecovery:
    """Recover payload bits and timing fields from the BCH decoder output.

    The second- and third-least-significant SFN bits are excluded from payload
    scrambling, so their interleaved positions identify the Gold-sequence
    offset without requiring the full SFN as an input. The function then
    descrambles and deinterleaves the 32 bits. It returns the serialized
    BCCH-BCH message; the fixed MIB choice has a separate strict field parser,
    while general ASN.1 extensions and complete SSB-index inference remain
    outside.
    """
    source = _binary_vector(
        scrambled_interleaved_payload, _PAYLOAD_SIZE, "scrambled_interleaved_payload"
    )
    cell_id = _integer(physical_cell_id, "physical_cell_id", 0, _CELL_ID_MAX)
    candidate_count = _candidate_count(max_candidate_blocks_per_half_frame)
    destinations = _payload_interleaver_destinations()

    third_sfn_lsb = int(source[destinations[25]])
    second_sfn_lsb = int(source[destinations[26]])
    scramble_offset_index = (third_sfn_lsb << 1) | second_sfn_lsb
    payload_sfn = scramble_offset_index << 1
    descrambled = nr_pbch_payload_descramble(
        source,
        cell_id,
        payload_sfn,
        max_candidate_blocks_per_half_frame=candidate_count,
    )
    raw = np.asarray(descrambled[destinations], dtype=np.uint8)

    sfn_msb = sum(int(raw[index + 1]) << (5 - index) for index in range(6))
    sfn_lsb = sum(int(raw[24 + index]) << (3 - index) for index in range(4))
    k_ssb: int | None = None
    pdcch_repetition: int | None = None
    if candidate_count in (4, 8):
        k_ssb = int(raw[29])
        pdcch_repetition = int(raw[31])
        ssb_index_bits: tuple[int, ...] = ()
    elif candidate_count == 10:
        k_ssb = int(raw[29])
        ssb_index_bits = (int(raw[31]),)
    elif candidate_count == 20:
        k_ssb = int(raw[29])
        ssb_index_bits = (int(raw[30]), int(raw[31]))
    else:
        ssb_index_bits = (int(raw[29]), int(raw[30]), int(raw[31]))

    higher_layer_bits = np.array(raw[:_HIGHER_LAYER_SIZE], copy=True)
    higher_layer_bits.setflags(write=False)
    return PBCHPayloadRecovery(
        higher_layer_bits=higher_layer_bits,
        system_frame_number=(sfn_msb << 4) | sfn_lsb,
        half_frame_index=int(raw[28]),
        max_candidate_blocks_per_half_frame=candidate_count,
        k_ssb_msb=k_ssb,
        ssb_index_msb_bits=ssb_index_bits,
        pdcch_repetition=pdcch_repetition,
    )


def _interleave_payload(bits: NDArray[np.uint8]) -> NDArray[np.uint8]:
    output = np.empty(_PAYLOAD_SIZE, dtype=np.uint8)
    output[_payload_interleaver_destinations()] = bits
    output.setflags(write=False)
    return output


def _payload_interleaver_destinations() -> NDArray[np.int8]:
    destinations = np.empty(_PAYLOAD_SIZE, dtype=np.int8)
    groups = {"sfn": 0, "half_frame": 10, "ssb": 11, "other": 14}
    for source_index in range(_PAYLOAD_SIZE):
        if source_index < 6 or 24 <= source_index <= 27:
            group = "sfn"
        elif source_index == 28:
            group = "half_frame"
        elif 29 <= source_index <= 31:
            group = "ssb"
        else:
            group = "other"
        destinations[source_index] = _INTERLEAVER[groups[group]]
        groups[group] += 1
    destinations.setflags(write=False)
    return destinations


def _payload_scramble_skip_mask(candidate_count: int) -> NDArray[np.bool_]:
    skipped_sources = {25, 26, 28}
    if candidate_count == 10:
        skipped_sources.add(31)
    elif candidate_count == 20:
        skipped_sources.update((30, 31))
    elif candidate_count == 64:
        skipped_sources.update((29, 30, 31))

    mask = np.zeros(_PAYLOAD_SIZE, dtype=np.bool_)
    destinations = _payload_interleaver_destinations()
    mask[destinations[sorted(skipped_sources)]] = True
    mask.setflags(write=False)
    return mask


def _candidate_count(value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError("max_candidate_blocks_per_half_frame must be an integer")
    result = int(value)
    if result not in (4, 8, 10, 20, 64):
        raise ValueError("max_candidate_blocks_per_half_frame must be one of 4, 8, 10, 20, or 64")
    return result


def _integer(value: object, name: str, lower: int, upper: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if not lower <= result <= upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}]")
    return result


def _binary_vector(bits: NDArray[np.generic], size: int, name: str) -> NDArray[np.uint8]:
    array = np.asarray(bits)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},)")
    if array.dtype.kind not in "biu" or np.any(array < 0) or np.any(array > 1):
        raise ValueError(f"{name} must contain only binary values")
    return np.asarray(array, dtype=np.uint8)
