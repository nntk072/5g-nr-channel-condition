"""Independent TS 38.211 checks for PBCH-DM-RS sequence and SSB mapping."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
from integrations.sionna import nr_pbch_dmrs_ssb as public_nr_pbch_dmrs_ssb
from integrations.sionna.pbch_dmrs import nr_pbch_dmrs_ssb


def _reference_gold_bits(c_init: int, count: int) -> list[int]:
    """Scalar clause 5.2.1 recurrence, separately written from production code."""
    offset = 1600
    x1 = [1] + [0] * 30
    x2 = [(c_init >> index) & 1 for index in range(31)]
    for index in range(offset + count):
        x1.append(x1[index + 3] ^ x1[index])
        x2.append(x2[index + 3] ^ x2[index + 2] ^ x2[index + 1] ^ x2[index])
    return [x1[index + offset] ^ x2[index + offset] for index in range(count)]


def _reference_coordinates(cell_id: int) -> list[tuple[int, int]]:
    """Expand the TS 38.211 table into coordinates, ordered by k and then l."""
    offset = cell_id % 4
    coordinates: list[tuple[int, int]] = []
    for subcarrier in range(240):
        if (subcarrier - offset) % 4:
            continue
        for symbol in range(4):
            if symbol in (1, 3) or (symbol == 2 and (subcarrier < 48 or subcarrier >= 192)):
                coordinates.append((subcarrier, symbol))
    return coordinates


@pytest.mark.parametrize(
    ("cell_id", "ssb_index", "half_frame", "candidate_count", "expected_c_init"),
    [
        (0, 0, 0, 4, 2112),
        (1, 3, 1, 4, 16897),
        (42, 7, 1, 8, 180738),
        (1007, 4, 0, 8, 2580803),
    ],
)
def test_pbch_dmrs_initialization_and_sequence_match_scalar_reference(
    cell_id: int,
    ssb_index: int,
    half_frame: int,
    candidate_count: int,
    expected_c_init: int,
) -> None:
    result = nr_pbch_dmrs_ssb(
        cell_id,
        ssb_index,
        half_frame_index=half_frame,
        max_candidate_blocks_per_half_frame=candidate_count,
    )
    expected_bits = _reference_gold_bits(expected_c_init, 288)
    expected_sequence = np.asarray(
        [
            ((1 - 2 * expected_bits[2 * index]) + 1j * (1 - 2 * expected_bits[2 * index + 1]))
            / np.sqrt(2.0)
            for index in range(144)
        ],
        dtype=np.complex128,
    )

    assert result.c_init == expected_c_init
    np.testing.assert_array_equal(result.sequence, expected_sequence)
    assert result.sequence.shape == (144,)
    np.testing.assert_allclose(np.abs(result.sequence), 1.0, rtol=0.0, atol=1e-15)
    assert not result.sequence.flags.writeable


@pytest.mark.parametrize("cell_id", [0, 1, 2, 3, 42, 1007])
def test_pbch_dmrs_mapping_matches_ts_38211_table_and_is_read_only(cell_id: int) -> None:
    result = nr_pbch_dmrs_ssb(
        cell_id,
        0,
        half_frame_index=0,
        max_candidate_blocks_per_half_frame=8,
    )
    expected = _reference_coordinates(cell_id)
    actual = list(zip(result.subcarriers.tolist(), result.ofdm_symbols.tolist(), strict=True))

    assert actual == expected
    assert len(actual) == 144
    assert Counter(result.ofdm_symbols.tolist()) == {1: 60, 2: 24, 3: 60}
    assert all(0 <= k < 240 and 0 <= symbol < 4 for k, symbol in actual)
    assert all(
        not array.flags.writeable
        for array in (result.subcarriers, result.ofdm_symbols, result.sequence)
    )


def test_lmax_four_includes_half_frame_in_sequence_index_but_lmax_eight_does_not() -> None:
    first_half_lmax4 = nr_pbch_dmrs_ssb(
        321, 2, half_frame_index=0, max_candidate_blocks_per_half_frame=4
    )
    second_half_lmax4 = nr_pbch_dmrs_ssb(
        321, 2, half_frame_index=1, max_candidate_blocks_per_half_frame=4
    )
    first_half_lmax8 = nr_pbch_dmrs_ssb(
        321, 2, half_frame_index=0, max_candidate_blocks_per_half_frame=8
    )
    second_half_lmax8 = nr_pbch_dmrs_ssb(
        321, 2, half_frame_index=1, max_candidate_blocks_per_half_frame=8
    )

    assert first_half_lmax4.c_init != second_half_lmax4.c_init
    assert first_half_lmax8.c_init == second_half_lmax8.c_init


def test_pbch_dmrs_public_export_is_the_native_implementation() -> None:
    assert public_nr_pbch_dmrs_ssb is nr_pbch_dmrs_ssb


@pytest.mark.parametrize(
    ("args", "kwargs", "error", "message"),
    [
        (
            (True, 0),
            {"half_frame_index": 0, "max_candidate_blocks_per_half_frame": 4},
            TypeError,
            "physical_cell_id",
        ),
        (
            (1008, 0),
            {"half_frame_index": 0, "max_candidate_blocks_per_half_frame": 4},
            ValueError,
            "physical_cell_id",
        ),
        (
            (1, -1),
            {"half_frame_index": 0, "max_candidate_blocks_per_half_frame": 4},
            ValueError,
            "ssb_index_in_half_frame",
        ),
        (
            (1, 4),
            {"half_frame_index": 0, "max_candidate_blocks_per_half_frame": 4},
            ValueError,
            "ssb_index_in_half_frame",
        ),
        (
            (1, 0),
            {"half_frame_index": 2, "max_candidate_blocks_per_half_frame": 4},
            ValueError,
            "half_frame_index",
        ),
        (
            (1, 0),
            {"half_frame_index": 0, "max_candidate_blocks_per_half_frame": 5},
            ValueError,
            "must be 4 or 8",
        ),
        (
            (1, 0),
            {"half_frame_index": 0, "max_candidate_blocks_per_half_frame": True},
            TypeError,
            "must be an integer",
        ),
    ],
)
def test_pbch_dmrs_rejects_invalid_inputs(
    args: tuple[object, ...],
    kwargs: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        nr_pbch_dmrs_ssb(*args, **kwargs)  # type: ignore[arg-type]
