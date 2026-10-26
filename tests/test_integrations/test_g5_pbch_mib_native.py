"""Independent fixed-field checks for NR BCCH-BCH MIB serialization."""

from __future__ import annotations

import numpy as np
import pytest
from integrations.sionna import (
    PBCHMIB,
    PBCHMessageClassExtension,
    nr_bcch_bch_message_parse,
    nr_bcch_bch_message_serialize,
    nr_pbch_mib_parse,
    nr_pbch_mib_serialize,
)


@pytest.mark.parametrize(
    "mib",
    [
        PBCHMIB(0, "scs15or60", 0, "pos2", 0, 0, "barred", "allowed", 0),
        PBCHMIB(63, "scs30or120", 15, "pos3", 15, 15, "notBarred", "notAllowed", 1),
        PBCHMIB(37, "scs30or120", 9, "pos2", 3, 12, "barred", "notAllowed", 1),
    ],
)
def test_mib_fields_match_independent_24_bit_bcch_bch_layout(mib: PBCHMIB) -> None:
    expected = np.asarray(
        [
            0,
            *[(mib.system_frame_number_msb >> shift) & 1 for shift in range(5, -1, -1)],
            int(mib.subcarrier_spacing_common == "scs30or120"),
            *[(mib.ssb_subcarrier_offset >> shift) & 1 for shift in range(3, -1, -1)],
            int(mib.dmrs_type_a_position == "pos3"),
            *[(mib.control_resource_set_zero >> shift) & 1 for shift in range(3, -1, -1)],
            *[(mib.search_space_zero >> shift) & 1 for shift in range(3, -1, -1)],
            int(mib.cell_barred == "notBarred"),
            int(mib.intra_freq_reselection == "notAllowed"),
            mib.spare,
        ],
        dtype=np.uint8,
    )
    actual = nr_pbch_mib_serialize(mib)

    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == (24,)
    assert not actual.flags.writeable
    assert nr_pbch_mib_parse(expected) == mib


@pytest.mark.parametrize(
    "bits, error",
    [
        (np.zeros(23, dtype=np.uint8), ValueError),
        (np.full(24, 2, dtype=np.uint8), ValueError),
        (np.asarray([1, *([0] * 23)], dtype=np.uint8), ValueError),
        (np.asarray([0.0] * 24), ValueError),
    ],
)
def test_mib_parser_rejects_non_mib_or_malformed_input(
    bits: np.ndarray, error: type[Exception]
) -> None:
    with pytest.raises(error):
        nr_pbch_mib_parse(bits)


@pytest.mark.parametrize(
    "trailing_bits",
    [
        (0,) * 23,
        tuple(index % 2 for index in range(23)),
    ],
)
def test_message_class_extension_choice_round_trips_opaque_trailing_bits(
    trailing_bits: tuple[int, ...],
) -> None:
    message = PBCHMessageClassExtension(trailing_bits)

    encoded = nr_bcch_bch_message_serialize(message)

    assert encoded.shape == (24,)
    assert encoded[0] == 1
    assert not encoded.flags.writeable
    assert nr_bcch_bch_message_parse(encoded) == message
    with pytest.raises(ValueError, match="messageClassExtension"):
        nr_pbch_mib_parse(encoded)


@pytest.mark.parametrize(
    "trailing_bits",
    [
        (0,) * 22,
        (0,) * 22 + (2,),
        (0,) * 22 + (True,),
    ],
)
def test_message_class_extension_rejects_malformed_trailing_bits(
    trailing_bits: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match="trailing_bits"):
        PBCHMessageClassExtension(trailing_bits)


def test_mib_serializer_rejects_out_of_range_field() -> None:
    invalid = PBCHMIB(64, "scs15or60", 0, "pos2", 0, 0, "barred", "allowed", 0)
    with pytest.raises(ValueError, match="system_frame_number_msb"):
        nr_pbch_mib_serialize(invalid)
