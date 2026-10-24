"""Independent TS 38.212 checks for PBCH payload generation and scrambling."""

from __future__ import annotations

import numpy as np
import pytest
from integrations.sionna import (
    nr_pbch_payload_descramble,
    nr_pbch_payload_generate,
    nr_pbch_payload_recover,
    nr_pbch_payload_scramble,
)

_G = (
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
)


def _reference_gold(c_init: int, count: int, offset: int) -> np.ndarray:
    final = 1600 + offset + count
    x1 = [1] + [0] * 30
    x2 = [(c_init >> index) & 1 for index in range(31)]
    for index in range(final):
        x1.append(x1[index + 3] ^ x1[index])
        x2.append(x2[index + 3] ^ x2[index + 2] ^ x2[index + 1] ^ x2[index])
    start = 1600 + offset
    return np.asarray([x1[i] ^ x2[i] for i in range(start, final)], dtype=np.uint8)


def _reference_payload(
    mib: np.ndarray, sfn: int, half_frame: int, ssb_index: int, lmax: int, cell_id: int
) -> np.ndarray:
    raw = np.zeros(32, dtype=np.uint8)
    raw[:24] = mib
    raw[24:28] = [(sfn >> bit) & 1 for bit in (3, 2, 1, 0)]
    raw[28] = half_frame
    if lmax in (4, 8):
        raw[29], raw[31] = 1, 0
    elif lmax == 10:
        raw[29], raw[31] = 1, (ssb_index >> 3) & 1
    elif lmax == 20:
        raw[29], raw[30], raw[31] = 1, (ssb_index >> 4) & 1, (ssb_index >> 3) & 1
    else:
        raw[29:32] = [(ssb_index >> bit) & 1 for bit in (5, 4, 3)]

    shuffled = np.zeros(32, dtype=np.uint8)
    sfn_counter = half_counter = ssb_counter = other_counter = 0
    for index, bit in enumerate(raw):
        if index < 6 or 24 <= index <= 27:
            destination = _G[sfn_counter]
            sfn_counter += 1
        elif index == 28:
            destination = _G[10 + half_counter]
            half_counter += 1
        elif 29 <= index <= 31:
            destination = _G[11 + ssb_counter]
            ssb_counter += 1
        else:
            destination = _G[14 + other_counter]
            other_counter += 1
        shuffled[destination] = bit

    skipped_sources = {25, 26, 28}
    if lmax == 10:
        skipped_sources.add(31)
    elif lmax == 20:
        skipped_sources.update((30, 31))
    elif lmax == 64:
        skipped_sources.update((29, 30, 31))
    skipped_outputs: set[int] = set()
    for index in skipped_sources:
        if index < 6 or 24 <= index <= 27:
            counter = sum(i < index and (i < 6 or 24 <= i <= 27) for i in range(32))
            group_start = 0
        elif index == 28:
            counter, group_start = 0, 10
        else:
            counter = sum(29 <= i < index for i in range(32))
            group_start = 11
        skipped_outputs.add(_G[group_start + counter])

    count = 32 - len(skipped_outputs)
    v = (((sfn >> 2) & 1) << 1) | ((sfn >> 1) & 1)
    sequence = _reference_gold(cell_id, count, v * count)
    sequence_index = 0
    for index in range(32):
        if index not in skipped_outputs:
            shuffled[index] ^= sequence[sequence_index]
            sequence_index += 1
    return shuffled


@pytest.mark.parametrize("lmax", [4, 8, 10, 20, 64])
@pytest.mark.parametrize("sfn", [0, 6, 10, 1023])
def test_payload_generation_matches_independent_scalar_reference(lmax: int, sfn: int) -> None:
    mib = np.arange(24, dtype=np.uint8) % 2
    mib[0] = 0
    mib[1:7] = [(sfn >> bit) & 1 for bit in range(9, 3, -1)]
    index = min(lmax - 1, 19)
    actual = nr_pbch_payload_generate(
        mib,
        321,
        sfn,
        1,
        index,
        max_candidate_blocks_per_half_frame=lmax,
        k_ssb_msb=1,
        pdcch_repetition=0,
    )
    expected = _reference_payload(mib, sfn, 1, index, lmax, 321)
    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == (32,)
    assert not actual.flags.writeable


@pytest.mark.parametrize("lmax", [4, 8, 10, 20, 64])
@pytest.mark.parametrize("sfn", [1, 42, 512, 1023])
def test_payload_recovery_round_trips_without_known_full_sfn(lmax: int, sfn: int) -> None:
    mib = np.arange(24, dtype=np.uint8) % 2
    mib[0] = 0
    mib[1:7] = [(sfn >> bit) & 1 for bit in range(9, 3, -1)]
    ssb_index = lmax - 1
    encoded = nr_pbch_payload_generate(
        mib,
        721,
        sfn,
        1,
        ssb_index,
        max_candidate_blocks_per_half_frame=lmax,
        k_ssb_msb=1,
        pdcch_repetition=1,
    )

    recovered = nr_pbch_payload_recover(encoded, 721, max_candidate_blocks_per_half_frame=lmax)

    np.testing.assert_array_equal(recovered.higher_layer_bits, mib)
    assert recovered.system_frame_number == sfn
    assert recovered.half_frame_index == 1
    assert recovered.max_candidate_blocks_per_half_frame == lmax
    if lmax in (4, 8):
        assert recovered.k_ssb_msb == 1
        assert recovered.pdcch_repetition == 1
        assert recovered.ssb_index_msb_bits == ()
    elif lmax == 10:
        assert recovered.k_ssb_msb == 1
        assert recovered.ssb_index_msb_bits == ((ssb_index >> 3) & 1,)
    elif lmax == 20:
        assert recovered.k_ssb_msb == 1
        assert recovered.ssb_index_msb_bits == (
            (ssb_index >> 4) & 1,
            (ssb_index >> 3) & 1,
        )
    else:
        assert recovered.k_ssb_msb is None
        assert recovered.ssb_index_msb_bits == tuple((ssb_index >> bit) & 1 for bit in (5, 4, 3))
    assert not recovered.higher_layer_bits.flags.writeable


@pytest.mark.parametrize(
    ("lmax", "scrambled_count"), [(4, 29), (8, 29), (10, 28), (20, 27), (64, 26)]
)
def test_payload_scrambling_skips_known_timing_and_candidate_bits(
    lmax: int, scrambled_count: int
) -> None:
    payload = np.arange(32, dtype=np.uint8) % 2
    result = nr_pbch_payload_scramble(
        payload, 1007, 0b1010110011, max_candidate_blocks_per_half_frame=lmax
    )
    round_trip = nr_pbch_payload_descramble(
        result, 1007, 0b1010110011, max_candidate_blocks_per_half_frame=lmax
    )
    np.testing.assert_array_equal(round_trip, payload)
    assert np.count_nonzero(result != payload) <= scrambled_count
    assert not result.flags.writeable


@pytest.mark.parametrize(("lmax", "ssb_index"), [(4, 4), (8, 8), (10, 10), (20, 20), (64, 64)])
def test_payload_generation_rejects_out_of_range_candidate(lmax: int, ssb_index: int) -> None:
    with pytest.raises(ValueError, match="ssb_index"):
        nr_pbch_payload_generate(
            np.zeros(24, dtype=np.uint8),
            0,
            0,
            0,
            ssb_index,
            max_candidate_blocks_per_half_frame=lmax,
        )


def test_payload_generation_checks_sfn_against_mib_bits() -> None:
    with pytest.raises(ValueError, match="bits 1 through 6"):
        nr_pbch_payload_generate(
            np.zeros(24, dtype=np.uint8),
            0,
            64,
            0,
            0,
            max_candidate_blocks_per_half_frame=4,
        )


def test_payload_generation_rejects_bcch_bch_extension_choice() -> None:
    higher_layer_bits = np.zeros(24, dtype=np.uint8)
    higher_layer_bits[0] = 1
    with pytest.raises(ValueError, match="MIB alternative"):
        nr_pbch_payload_generate(
            higher_layer_bits,
            0,
            0,
            0,
            0,
            max_candidate_blocks_per_half_frame=4,
        )


@pytest.mark.parametrize(
    ("field", "value", "error", "message"),
    [
        ("physical_cell_id", True, TypeError, "physical_cell_id"),
        ("system_frame_number", 1024, ValueError, "system_frame_number"),
        ("half_frame_index", 2, ValueError, "half_frame_index"),
        ("max_candidate_blocks_per_half_frame", 12, ValueError, "max_candidate_blocks"),
    ],
)
def test_payload_generation_rejects_invalid_inputs(
    field: str, value: object, error: type[Exception], message: str
) -> None:
    arguments: dict[str, object] = {
        "higher_layer_bits": np.zeros(24, dtype=np.uint8),
        "physical_cell_id": 0,
        "system_frame_number": 0,
        "half_frame_index": 0,
        "ssb_index_in_half_frame": 0,
        "max_candidate_blocks_per_half_frame": 4,
    }
    arguments[field] = value
    with pytest.raises(error, match=message):
        nr_pbch_payload_generate(**arguments)  # type: ignore[arg-type]


def test_interleaver_table_is_a_permutation() -> None:
    assert sorted(_G) == list(range(32))
