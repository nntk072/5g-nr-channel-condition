"""Independent TS 38.211 checks for PBCH data scrambling and modulation."""

from __future__ import annotations

import numpy as np
import pytest
from integrations.sionna import map_nr_ssb_resource_grid
from integrations.sionna import nr_pbch_qpsk_modulate as public_qpsk_modulate
from integrations.sionna import nr_pbch_scramble_bits as public_scramble_bits
from integrations.sionna.pbch_data import nr_pbch_qpsk_modulate, nr_pbch_scramble_bits


def _gold_reference(c_init: int, count: int, sequence_offset: int) -> np.ndarray:
    """Scalar Gold-sequence recurrence independent of production implementation."""
    skip = 1600
    final_index = skip + sequence_offset + count
    x1 = [1] + [0] * 30
    x2 = [(c_init >> index) & 1 for index in range(31)]
    for index in range(final_index):
        x1.append(x1[index + 3] ^ x1[index])
        x2.append(x2[index + 3] ^ x2[index + 2] ^ x2[index + 1] ^ x2[index])
    start = skip + sequence_offset
    return np.asarray(
        [x1[index] ^ x2[index] for index in range(start, final_index)], dtype=np.uint8
    )


@pytest.mark.parametrize(
    ("cell_id", "ssb_index", "candidate_count"),
    [
        (0, 0, 4),
        (321, 3, 4),
        (42, 7, 8),
        (1007, 4, 8),
        (42, 9, 10),
        (321, 19, 20),
        (1007, 63, 64),
    ],
)
def test_pbch_scrambler_matches_scalar_gold_sequence_and_index_offset(
    cell_id: int,
    ssb_index: int,
    candidate_count: int,
) -> None:
    bits = np.arange(864, dtype=np.uint8) % 2
    actual = nr_pbch_scramble_bits(
        bits,
        cell_id,
        ssb_index,
        max_candidate_blocks_per_half_frame=candidate_count,
    )
    v = ssb_index % (4 if candidate_count == 4 else 8)
    expected = bits ^ _gold_reference(cell_id, 864, v * 864)

    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == (864,)
    assert not actual.flags.writeable


def test_pbch_qpsk_uses_first_bit_for_real_and_second_for_imaginary_sign() -> None:
    bits = np.tile(np.asarray([0, 0, 0, 1, 1, 0, 1, 1], dtype=np.uint8), 108)
    symbols = nr_pbch_qpsk_modulate(bits)
    expected_cycle = np.asarray([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j], dtype=np.complex128) / np.sqrt(
        2.0
    )

    np.testing.assert_array_equal(symbols[:4], expected_cycle)
    np.testing.assert_allclose(np.abs(symbols), 1.0, rtol=0.0, atol=1e-15)
    assert symbols.shape == (432,)
    assert not symbols.flags.writeable


def test_pbch_data_functions_are_exported() -> None:
    assert public_scramble_bits is nr_pbch_scramble_bits
    assert public_qpsk_modulate is nr_pbch_qpsk_modulate


def test_scrambled_pbch_bits_flow_into_the_ssb_grid_mapper() -> None:
    cell_id = 3 * 87 + 1
    bits = np.arange(864, dtype=np.uint8) % 2
    scrambled = nr_pbch_scramble_bits(
        bits,
        cell_id,
        3,
        max_candidate_blocks_per_half_frame=4,
    )
    symbols = nr_pbch_qpsk_modulate(scrambled)
    result = map_nr_ssb_resource_grid(
        87,
        1,
        symbols,
        ssb_index_in_half_frame=3,
        half_frame_index=1,
        max_candidate_blocks_per_half_frame=4,
    )
    data_coordinates = [
        (subcarrier, symbol)
        for subcarrier in range(240)
        for symbol in (1, 2, 3)
        if (symbol in (1, 3) or subcarrier < 48 or subcarrier >= 192)
        and (subcarrier, symbol)
        not in set(
            zip(
                result.pbch_dmrs.subcarriers.tolist(),
                result.pbch_dmrs.ofdm_symbols.tolist(),
                strict=True,
            )
        )
    ]

    assert result.physical_cell_id == cell_id
    np.testing.assert_array_equal(
        result.grid[
            [symbol for _, symbol in data_coordinates],
            [subcarrier for subcarrier, _ in data_coordinates],
        ],
        symbols,
    )


@pytest.mark.parametrize(
    ("bits", "cell_id", "ssb_index", "candidate_count", "error", "message"),
    [
        (np.zeros(863, dtype=np.uint8), 1, 0, 4, ValueError, "shape"),
        (np.zeros(864, dtype=np.float64), 1, 0, 4, ValueError, "binary"),
        (np.full(864, 2, dtype=np.uint8), 1, 0, 4, ValueError, "binary"),
        (np.zeros(864, dtype=np.uint8), True, 0, 4, TypeError, "physical_cell_id"),
        (np.zeros(864, dtype=np.uint8), 1008, 0, 4, ValueError, "physical_cell_id"),
        (np.zeros(864, dtype=np.uint8), 1, 4, 4, ValueError, "ssb_index"),
        (np.zeros(864, dtype=np.uint8), 1, 0, 3, ValueError, "candidate_blocks"),
    ],
)
def test_pbch_scrambler_rejects_invalid_inputs(
    bits: np.ndarray,
    cell_id: int,
    ssb_index: int,
    candidate_count: int,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        nr_pbch_scramble_bits(
            bits,
            cell_id,
            ssb_index,
            max_candidate_blocks_per_half_frame=candidate_count,
        )


@pytest.mark.parametrize(
    ("bits", "message"),
    [(np.zeros(863, dtype=np.uint8), "shape"), (np.full(864, -1, dtype=np.int8), "binary")],
)
def test_pbch_qpsk_rejects_invalid_bit_vectors(bits: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        nr_pbch_qpsk_modulate(bits)
