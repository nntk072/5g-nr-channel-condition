"""SRS DCI 0_3 decoded-field rules: aperiodic slot targets, cell sets, combos and bit fields."""

from __future__ import annotations

import pytest
from integrations.sionna.srs.srs_constants.dci_0_3 import (
    nr_srs_aperiodic_target_slot_index,
    nr_srs_dci_0_3_scheduled_cell_ids,
)


@pytest.mark.parametrize(
    ("slot", "kwargs", "error", "message"),
    [
        pytest.param(
            "3",
            {"slot_offset": 1},
            TypeError,
            "triggering_dci_slot must be an integer",
            id="slot-type",
        ),
        pytest.param(
            True,
            {"slot_offset": 1},
            TypeError,
            "triggering_dci_slot must be an integer",
            id="slot-bool",
        ),
        pytest.param(
            -1,
            {"slot_offset": 1},
            ValueError,
            "triggering_dci_slot must be nonnegative",
            id="slot-negative",
        ),
        pytest.param(
            3, {"slot_offset": "1"}, TypeError, "slot_offset must be an integer", id="offset-type"
        ),
        pytest.param(
            3,
            {"available_slot_offsets": [0, 9]},
            ValueError,
            "offsets must be in \\[0, 7\\]",
            id="offset-range",
        ),
    ],
)
def test_aperiodic_target_slot_rejects_malformed_inputs(slot, kwargs, error, message) -> None:
    with pytest.raises(error, match=message):
        nr_srs_aperiodic_target_slot_index(slot, **kwargs)


def test_aperiodic_target_slot_adds_the_selected_offset() -> None:
    assert nr_srs_aperiodic_target_slot_index(3, slot_offset=1) == 4
    # The selected available offset is measured from slot 3 + slot_offset 1 = 4; index 1 is 2.
    assert (
        nr_srs_aperiodic_target_slot_index(
            3,
            slot_offset=1,
            available_slot_offsets=[0, 2, 4],
            selected_available_offset_index=1,
        )
        == 6
    )


def _set(set_id: int, cells: list[str], combos=None) -> dict[str, object]:
    config: dict[str, object] = {"set_id": set_id, "scheduled_cell_ids": cells}
    if combos is not None:
        config["scheduled_cell_combos"] = combos
    return config


@pytest.mark.parametrize(
    ("configs", "kwargs", "message"),
    [
        pytest.param("not-a-sequence", {}, "must be an ordered sequence", id="configs-type"),
        pytest.param([], {}, "1..4 configured sets", id="configs-empty"),
        pytest.param(
            ["not-a-mapping"], {}, "each cell-set config must be a mapping", id="config-type"
        ),
        pytest.param(
            [{"set_id": "0", "scheduled_cell_ids": ["a", "b"]}],
            {},
            "each set_id must be an integer",
            id="set-id-type",
        ),
        pytest.param(
            [_set(0, ["a", "b"]), _set(0, ["c", "d"])],
            {},
            "set_id values must be unique",
            id="set-id-duplicate",
        ),
        pytest.param([_set(0, "ab")], {}, "must be an ordered sequence", id="cells-type"),
        pytest.param([_set(0, ["a"])], {}, "2..4 nonempty IDs", id="cells-short"),
        pytest.param([_set(0, ["a", "a"])], {}, "unique within each set", id="cells-repeat"),
        pytest.param(
            [_set(0, ["a", "b"]), _set(1, ["b", "c"])],
            {},
            "more than one set",
            id="cells-cross-set",
        ),
        pytest.param(
            [_set(0, ["a", "b"]), _set(1, ["c", "d"])],
            {"default_set_id": "0"},
            "default_set_id must be an integer",
            id="default-type",
        ),
        pytest.param(
            [_set(0, ["a", "b"]), _set(1, ["c", "d"])],
            {"default_set_id": 3},
            "does not identify a configured set",
            id="default-unknown",
        ),
        pytest.param(
            [_set(0, ["a", "b"]), _set(1, ["c", "d"])],
            {"scheduled_cell_set_indicator": "1"},
            "must be an integer",
            id="indicator-type",
        ),
        pytest.param(
            [_set(0, ["a", "b"]), _set(1, ["c", "d"])],
            {"scheduled_cell_set_indicator": 1, "default_set_id": 0},
            "must be omitted",
            id="indicator-and-default",
        ),
        pytest.param(
            [_set(0, ["a", "b"]), _set(1, ["c", "d"])],
            {"scheduled_cell_set_indicator": 5},
            "selects no configured set",
            id="indicator-unknown",
        ),
    ],
)
def test_scheduled_cell_resolution_rejects_invalid_configurations(configs, kwargs, message) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        nr_srs_dci_0_3_scheduled_cell_ids(configs, **kwargs)


def test_scheduled_cell_resolution_selects_the_declared_set() -> None:
    configs = [_set(0, ["a", "b"]), _set(1, ["c", "d"])]
    assert nr_srs_dci_0_3_scheduled_cell_ids(configs, scheduled_cell_set_indicator=1) == ("c", "d")
    assert nr_srs_dci_0_3_scheduled_cell_ids(configs, default_set_id=0) == ("a", "b")
    assert nr_srs_dci_0_3_scheduled_cell_ids([_set(0, ["a", "b"])]) == ("a", "b")
