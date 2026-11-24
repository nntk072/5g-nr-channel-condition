"""SRS DCI 0_3 request-list selector: payload, offset and reserved-codepoint rules."""

from __future__ import annotations

import numpy as np
import pytest
from integrations.sionna.srs.srs_constants.dci_0_3 import (
    nr_srs_dci_0_3_request_list_index_from_bits,
)


def test_absent_or_single_entry_lists_need_no_selector_bits() -> None:
    assert (
        nr_srs_dci_0_3_request_list_index_from_bits(
            [1, 0], field_offset=0, request_list_entry_count=0
        )
        is None
    )
    assert (
        nr_srs_dci_0_3_request_list_index_from_bits(
            [1, 0], field_offset=1, request_list_entry_count=1
        )
        == 0
    )


def test_selector_is_read_msb_first_from_the_declared_offset() -> None:
    bits = [1, 0, 1, 1, 0]
    # Bits 1..2 are [0, 1], read most-significant first.
    assert (
        nr_srs_dci_0_3_request_list_index_from_bits(
            bits, field_offset=1, request_list_entry_count=4
        )
        == 1
    )
    assert (
        nr_srs_dci_0_3_request_list_index_from_bits(
            [True, False, True], field_offset=0, request_list_entry_count=4
        )
        == 2
    )


@pytest.mark.parametrize(
    ("bits", "offset", "count", "error", "message"),
    [
        pytest.param("1010", 0, 2, TypeError, "sequence of decoded bits", id="string-payload"),
        pytest.param(
            [1, 0], True, 2, TypeError, "field_offset must be an integer", id="offset-bool"
        ),
        pytest.param(
            [1, 0], -1, 2, ValueError, "field_offset must be nonnegative", id="offset-negative"
        ),
        pytest.param(
            [1, 0],
            0,
            True,
            TypeError,
            "request_list_entry_count must be an integer",
            id="count-bool",
        ),
        pytest.param([1, 0], 0, 17, ValueError, "must be in \\[0, 16\\]", id="count-large"),
        pytest.param([1, 2], 0, 2, ValueError, "must be a Boolean or integer bit", id="bit-value"),
        pytest.param([1, 0], 3, 2, ValueError, "lies beyond the DCI payload", id="offset-beyond"),
        pytest.param([1], 0, 4, ValueError, "complete SRS request selector", id="payload-short"),
        pytest.param([1, 1], 0, 3, ValueError, "reserved codepoint", id="reserved"),
    ],
)
def test_selector_rejects_malformed_payloads_and_reserved_codepoints(
    bits, offset, count, error, message
) -> None:
    with pytest.raises(error, match=message):
        nr_srs_dci_0_3_request_list_index_from_bits(
            bits, field_offset=offset, request_list_entry_count=count
        )
