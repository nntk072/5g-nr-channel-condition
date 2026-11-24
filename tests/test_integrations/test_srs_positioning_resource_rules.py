"""SRS positioning resource mapping: integer types, symbol window, comb and sequence rules."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from integrations.sionna.srs.srs_configuration import (
    map_nr_srs_positioning_multiport_resource,
    map_nr_srs_positioning_resource,
)

BASE: dict[str, Any] = {
    "slot_symbol_count": 14,
    "subcarrier_count": 24,
    "start_position_offset": 4,
    "srs_symbol_count": 1,
    "comb_size": 2,
    "comb_offset": 0,
    "frequency_start_subcarrier": 0,
    "allocated_bandwidth_subcarriers": 12,
}


def _sequence(length: int = 6) -> np.ndarray:
    return np.ones(length, dtype=np.complex64)


def test_valid_positioning_resource_maps() -> None:
    assert map_nr_srs_positioning_resource(_sequence(), **BASE) is not None


@pytest.mark.parametrize(
    ("change", "error", "message"),
    [
        pytest.param({"comb_size": "2"}, TypeError, "comb_size must be an integer", id="type"),
        pytest.param(
            {"slot_symbol_count": 13}, ValueError, "slot_symbol_count must be 12 or 14", id="slot"
        ),
        pytest.param(
            {"subcarrier_count": 0},
            ValueError,
            "subcarrier_count must be positive",
            id="subcarriers",
        ),
        pytest.param(
            {"srs_symbol_count": 15}, ValueError, "symbol count is unsupported", id="symbols"
        ),
        pytest.param(
            {"start_position_offset": 14}, ValueError, "must fit the SRS symbols", id="offset-high"
        ),
        pytest.param(
            {"comb_size": 3}, ValueError, "transmission_comb must be 2, 4, or 8", id="comb"
        ),
        pytest.param({"comb_offset": 2}, ValueError, "comb_offset must be in", id="comb-offset"),
        pytest.param(
            {"frequency_start_subcarrier": -1},
            ValueError,
            "frequency_start_subcarrier must be nonnegative",
            id="freq",
        ),
    ],
)
def test_positioning_resource_rejects_invalid_geometry(change, error, message) -> None:
    arguments = {**BASE, **change}
    with pytest.raises(error, match=message):
        map_nr_srs_positioning_resource(_sequence(), **arguments)


def test_positioning_resource_rejects_invalid_sequences() -> None:
    with pytest.raises(TypeError, match="sequence must be complex-valued"):
        map_nr_srs_positioning_resource(np.ones(6), **BASE)
    with pytest.raises(ValueError, match="sequence length does not match"):
        map_nr_srs_positioning_resource(_sequence(12), **BASE)
    bad = _sequence()
    bad[0] = np.nan
    with pytest.raises(ValueError, match="only finite values"):
        map_nr_srs_positioning_resource(bad, **BASE)


def test_multiport_positioning_requires_complex_sequences_and_port_rules() -> None:
    with pytest.raises(TypeError, match="sequences must be complex-valued"):
        map_nr_srs_positioning_multiport_resource(np.ones((1, 1, 6)), **BASE)
    with pytest.raises(TypeError, match="ports8_tdm must be a boolean"):
        map_nr_srs_positioning_multiport_resource(
            np.ones((1, 1, 6), dtype=np.complex64), ports8_tdm="yes", **BASE
        )


def test_multiport_positioning_checks_shape_rows_and_port_counts() -> None:
    sequences = np.ones((2, 1, 6), dtype=np.complex64)
    with pytest.raises(ValueError, match="sequences must have shape"):
        map_nr_srs_positioning_multiport_resource(np.ones((6,), dtype=np.complex64), **BASE)
    with pytest.raises(ValueError, match="each port must have exactly srs_symbol_count rows"):
        map_nr_srs_positioning_multiport_resource(np.ones((2, 2, 6), dtype=np.complex64), **BASE)
    with pytest.raises(ValueError, match="ports8_tdm requires eight ports"):
        map_nr_srs_positioning_multiport_resource(sequences, ports8_tdm=True, **BASE)
    assert map_nr_srs_positioning_multiport_resource(sequences, **BASE) is not None
