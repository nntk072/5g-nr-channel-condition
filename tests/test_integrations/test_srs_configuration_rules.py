"""SRS configuration resolution: transmission comb, port shifts, hopping offsets and request selectors."""

from __future__ import annotations

from typing import Any

import pytest
from integrations.sionna.srs.srs_configuration import (
    nr_srs_cyclic_shift_hopping_offset,
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
