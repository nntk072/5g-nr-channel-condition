"""SRS configuration resolution: transmission comb, port shifts, hopping offsets and request selectors."""

from __future__ import annotations

from typing import Any

import pytest
from integrations.sionna.srs.srs_configuration import (
    nr_srs_cyclic_shift_hopping_offset,
    nr_srs_cyclic_shift_hopping_offsets,
    nr_srs_port_cyclic_shift_indices,
    nr_srs_transmission_comb_from_rrc,
)


@pytest.mark.parametrize(
    ("choice", "positioning", "message"),
    [
        pytest.param("comb2", False, "single-choice decoded mapping", id="not-mapping"),
        pytest.param(
            {"comb2": {}, "comb4": {}}, False, "single-choice decoded mapping", id="two-choices"
        ),
        pytest.param(
            {"comb2": {}}, "yes", "positioning_resource must be a boolean", id="flag-type"
        ),
        pytest.param(
            {"comb2": "x"}, False, "transmissionComb choice must be n2, n4, or n8", id="value-type"
        ),
    ],
)
def test_transmission_comb_rejects_malformed_choices(choice, positioning, message) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        nr_srs_transmission_comb_from_rrc(choice, positioning_resource=positioning)


@pytest.mark.parametrize(
    ("choice", "positioning", "message"),
    [
        ({"n2": None}, False, "choice value must be a mapping"),
        ({"n2": {"combOffset-n2": 0}}, False, "missing or unsupported fields"),
        ({"n2-r16": {"combOffset-n2-r16": 0, "cyclicShift-n2-r16": 0}}, False, "choice must be"),
    ],
)
def test_transmission_comb_rejects_invalid_decoded_choice_fields(
    choice, positioning, message
) -> None:
    with pytest.raises(ValueError, match=message):
        nr_srs_transmission_comb_from_rrc(choice, positioning_resource=positioning)


def test_transmission_comb_decodes_standard_and_positioning_choices() -> None:
    standard = nr_srs_transmission_comb_from_rrc({"n4": {"combOffset-n4": 2, "cyclicShift-n4": 7}})
    positioning = nr_srs_transmission_comb_from_rrc(
        {"n8-r16": {"combOffset-n8-r16": 5, "cyclicShift-n8-r16": 5}},
        positioning_resource=True,
    )

    assert (standard.transmission_comb, standard.comb_offset, standard.base_cyclic_shift) == (
        4,
        2,
        7,
    )
    assert (
        positioning.transmission_comb,
        positioning.comb_offset,
        positioning.base_cyclic_shift,
    ) == (8, 5, 5)


@pytest.mark.parametrize(
    ("ports", "comb", "expected"),
    [
        ((1000,), 2, (3,)),
        ((1000, 1001), 2, (3, 7)),
        ((1000, 1001, 1002, 1003), 2, (3, 5, 7, 1)),
        (tuple(range(1000, 1008)), 2, (3, 4, 5, 6, 7, 0, 1, 2)),
    ],
)
def test_srs_port_cyclic_shift_indices_follow_comb_specific_port_spacing(
    ports, comb, expected
) -> None:
    assert (
        nr_srs_port_cyclic_shift_indices(
            ports,
            transmission_comb=comb,
            base_cyclic_shift=3,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("ports", "comb", "ports8_tdm", "expected"),
    [
        pytest.param(
            tuple(range(1000, 1008)), 8, False, (0, 0, 0, 0, 3, 3, 3, 3), id="eight-port-comb8"
        ),
        pytest.param((1000, 1001, 1002, 1003), 8, False, (0, 0, 3, 3), id="four-port-comb8"),
        pytest.param(
            tuple(range(1000, 1008)),
            4,
            False,
            (0, 0, 3, 3, 6, 6, 9, 9),
            id="eight-port-comb4",
        ),
        pytest.param(
            tuple(range(1000, 1008)),
            8,
            True,
            (0, 0, 0, 0, 3, 3, 3, 3),
            id="eight-port-tdm-comb8",
        ),
    ],
)
def test_srs_port_cyclic_shifts_cover_six_and_twelve_shift_mappings(
    ports, comb, ports8_tdm, expected
) -> None:
    assert (
        nr_srs_port_cyclic_shift_indices(
            ports,
            transmission_comb=comb,
            base_cyclic_shift=0,
            ports8_tdm=ports8_tdm,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        pytest.param(
            {"antenna_ports": ()}, TypeError, "nonempty tuple of integers", id="ports-empty"
        ),
        pytest.param(
            {"antenna_ports": [0]}, TypeError, "nonempty tuple of integers", id="ports-list"
        ),
        pytest.param(
            {"ports8_tdm": "yes"}, TypeError, "ports8_tdm must be a boolean", id="tdm-type"
        ),
        pytest.param(
            {"transmission_comb": "2"},
            TypeError,
            "transmission_comb must be an integer",
            id="comb-type",
        ),
        pytest.param(
            {"base_cyclic_shift": 99}, ValueError, "base_cyclic_shift must be in", id="shift-range"
        ),
        pytest.param(
            {"antenna_ports": (0, 1, 2)}, ValueError, "must be 1, 2, 4, or 8", id="port-count"
        ),
        pytest.param(
            {"antenna_ports": ("a",)},
            TypeError,
            "antenna_ports must contain integers",
            id="port-type",
        ),
    ],
)
def test_port_cyclic_shifts_reject_malformed_inputs(kwargs, error, message) -> None:
    arguments: dict[str, Any] = {
        "antenna_ports": (0,),
        "transmission_comb": 2,
        "base_cyclic_shift": 0,
    }
    arguments.update(kwargs)
    with pytest.raises(error, match=message):
        nr_srs_port_cyclic_shift_indices(**arguments)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        pytest.param(
            {"cyclic_shift_hopping_id": 70000}, "cyclic_shift_hopping_id must be in", id="id"
        ),
        pytest.param(
            {"slots_per_frame": 7}, "radio frame and frame-local slot indices", id="slots"
        ),
        pytest.param({"slot_symbol_count": 13}, "slot_symbol_count must be 12 or 14", id="symbols"),
        pytest.param(
            {"hopping_finer_granularity": "yes"},
            "hopping_finer_granularity must be a boolean",
            id="granularity",
        ),
        pytest.param({"slot_index": True}, "slot_index must be an integer", id="slot-type"),
    ],
)
def test_cyclic_shift_hopping_offset_rejects_invalid_slot_or_identity(kwargs, message) -> None:
    arguments: dict[str, Any] = {
        "transmission_comb": 2,
        "cyclic_shift_hopping_id": 0,
        "radio_frame_index": 0,
        "slot_index": 0,
        "slots_per_frame": 10,
        "slot_symbol_count": 14,
        "start_symbol_index": 0,
        "resource_symbol_index": 0,
    }
    arguments.update(kwargs)
    with pytest.raises((TypeError, ValueError), match=message):
        nr_srs_cyclic_shift_hopping_offset(**arguments)


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"hopping_subset": [True] * 8}, ValueError, "bitmap matching"),
        ({"hopping_subset": (True,) * 7}, ValueError, "bitmap matching"),
        (
            {"hopping_subset": (True, 1, False, False, False, False, False, False)},
            TypeError,
            "boolean bitmap entries",
        ),
        (
            {"hopping_subset": (True, False, False, False, False, False, False, False)},
            ValueError,
            "enable between 2",
        ),
        (
            {
                "hopping_subset": (True, True, True, True, True, True, True, True),
                "hopping_finer_granularity": True,
            },
            ValueError,
            "cannot both be configured",
        ),
    ],
)
def test_cyclic_shift_hopping_offset_validates_subset_configuration(kwargs, error, message) -> None:
    arguments = {
        "transmission_comb": 2,
        "cyclic_shift_hopping_id": 42,
        "radio_frame_index": 0,
        "slot_index": 0,
        "slots_per_frame": 10,
        "slot_symbol_count": 14,
        "start_symbol_index": 10,
        "resource_symbol_index": 0,
    }
    arguments.update(kwargs)
    with pytest.raises(error, match=message):
        nr_srs_cyclic_shift_hopping_offset(**arguments)


def test_cyclic_shift_hopping_offsets_cover_symbol_count_and_slot_extent_guards() -> None:
    arguments = {
        "transmission_comb": 2,
        "cyclic_shift_hopping_id": 42,
        "radio_frame_index": 0,
        "slot_index": 0,
        "slots_per_frame": 10,
        "slot_symbol_count": 14,
        "start_symbol_index": 12,
        "srs_symbol_count": 2,
    }
    offsets = nr_srs_cyclic_shift_hopping_offsets(**arguments)
    assert len(offsets) == 2
    with pytest.raises(ValueError, match="one of the supported SRS lengths"):
        nr_srs_cyclic_shift_hopping_offsets(**{**arguments, "srs_symbol_count": 3})
    with pytest.raises(ValueError, match="extend past the slot"):
        nr_srs_cyclic_shift_hopping_offsets(**{**arguments, "srs_symbol_count": 4})
    with pytest.raises(TypeError, match="start_symbol_index must be an integer"):
        nr_srs_cyclic_shift_hopping_offsets(**{**arguments, "start_symbol_index": True})
    with pytest.raises(ValueError, match="slot symbol coordinates are invalid"):
        nr_srs_cyclic_shift_hopping_offsets(**{**arguments, "slot_symbol_count": 13})
