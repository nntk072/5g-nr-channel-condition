"""SRS DCI 0_3 argument rules: cell combos, indicator selection and available slot offsets."""

from __future__ import annotations

import pytest
from integrations.sionna.srs.srs_constants.dci_0_3 import (
    nr_srs_dci_0_3_request_list_index_from_bits,
    nr_srs_dci_0_3_scheduled_cell_ids,
)

CELL_SETS = [
    {"set_id": 0, "scheduled_cell_ids": ["a", "b"], "scheduled_cell_combos": [[0], [0, 1]]},
    {"set_id": 1, "scheduled_cell_ids": ["c", "d"]},
]


@pytest.mark.parametrize(
    ("combos", "error", "message"),
    [
        pytest.param("0", TypeError, "must be an ordered sequence", id="string"),
        pytest.param([], ValueError, "must contain 1..16 rows", id="no-rows"),
        pytest.param(
            [[0], "01"],
            TypeError,
            "each scheduled-cell combo must be an ordered index sequence",
            id="combo-type",
        ),
        pytest.param([[]], ValueError, "must select 1..N configured cells", id="empty-combo"),
        pytest.param([[0, 1, 0]], ValueError, "must select 1..N configured cells", id="too-many"),
        pytest.param([[True]], TypeError, "must be integers", id="bool-index"),
    ],
)
def test_scheduled_cell_combo_rows_are_refused_when_malformed(combos, error, message) -> None:
    configs = [{"set_id": 0, "scheduled_cell_ids": ["a", "b"], "scheduled_cell_combos": combos}]
    with pytest.raises(error, match=message):
        nr_srs_dci_0_3_scheduled_cell_ids(
            configs, scheduled_cell_set_indicator=0, scheduled_cells_indicator=0
        )


def test_combo_indicator_must_be_an_integer_within_the_configured_rows() -> None:
    with pytest.raises(TypeError, match="must select a configured combo row"):
        nr_srs_dci_0_3_scheduled_cell_ids(
            CELL_SETS, scheduled_cell_set_indicator=0, scheduled_cells_indicator="0"
        )
    with pytest.raises(ValueError, match="selects no configured combo row"):
        nr_srs_dci_0_3_scheduled_cell_ids(
            CELL_SETS, scheduled_cell_set_indicator=0, scheduled_cells_indicator=5
        )
    assert nr_srs_dci_0_3_scheduled_cell_ids(
        CELL_SETS, scheduled_cell_set_indicator=0, scheduled_cells_indicator=1
    ) == ("a", "b")


def test_request_selector_reads_msb_first_bits_at_the_declared_offset() -> None:
    bits = [1, 0, 1, 1, 0, 0, 1]
    assert (
        nr_srs_dci_0_3_request_list_index_from_bits(
            bits, field_offset=2, request_list_entry_count=4
        )
        == 0b11
    )
    # A one-entry list has a zero-bit selector and resolves to entry zero.
    assert (
        nr_srs_dci_0_3_request_list_index_from_bits(
            bits, field_offset=2, request_list_entry_count=1
        )
        == 0
    )
    # With no list the field is absent.
    assert (
        nr_srs_dci_0_3_request_list_index_from_bits(
            bits, field_offset=2, request_list_entry_count=0
        )
        is None
    )
