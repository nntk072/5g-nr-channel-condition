"""SRS lookups: port roster, symbol activity, frequency positions and periodicity contracts."""

from __future__ import annotations

import pytest
from integrations.sionna.srs.srs_lookups import (
    nr_srs_frequency_position_indices,
    nr_srs_periodicity_and_offset_slots,
    nr_srs_port_configuration_from_rrc,
    nr_srs_port_symbol_activity,
    nr_srs_positioning_symbol_offset,
    srs_cyclic_shift_count,
)


@pytest.mark.parametrize(
    ("value", "error", "message"),
    [
        pytest.param("2", TypeError, "transmission_comb must be an integer", id="type"),
        pytest.param(3, ValueError, "transmission_comb must be 2, 4, or 8", id="value"),
    ],
)
def test_cyclic_shift_count_validates_the_comb(value, error, message) -> None:
    with pytest.raises(error, match=message):
        srs_cyclic_shift_count(value)


@pytest.mark.parametrize(
    ("ports", "symbols", "ports8", "error", "message"),
    [
        pytest.param((), 1, False, TypeError, "nonempty tuple of integers", id="empty"),
        pytest.param([1000], 1, False, TypeError, "nonempty tuple of integers", id="list"),
        pytest.param(
            (1000,), "1", False, TypeError, "srs_symbol_count must be an integer", id="symbols-type"
        ),
        pytest.param(
            (1000,), 3, False, ValueError, "srs_symbol_count is unsupported", id="symbols-value"
        ),
        pytest.param((1000,), 1, "yes", TypeError, "ports8_tdm must be a boolean", id="tdm-type"),
        pytest.param(
            ("1000",), 1, False, TypeError, "antenna_ports must contain integers", id="port-type"
        ),
        pytest.param(
            (1000, 1000), 1, False, ValueError, "must be unique SRS ports", id="port-duplicate"
        ),
        pytest.param(
            (1000, 1008), 1, False, ValueError, "must be unique SRS ports", id="port-range"
        ),
        pytest.param(
            (1000, 1001), 1, True, ValueError, "complete ordered port roster", id="tdm-roster"
        ),
    ],
)
def test_symbol_activity_rejects_invalid_port_rosters(
    ports, symbols, ports8, error, message
) -> None:
    with pytest.raises(error, match=message):
        nr_srs_port_symbol_activity(ports, srs_symbol_count=symbols, ports8_tdm=ports8)


def test_standard_port_symbol_activity_covers_every_resource_symbol() -> None:
    assert nr_srs_port_symbol_activity((1000, 1001), srs_symbol_count=4) == (
        (True, True, True, True),
        (True, True, True, True),
    )


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        pytest.param({"c_srs": 64}, ValueError, "c_srs must be in", id="c-range"),
        pytest.param({"b_srs": 4}, ValueError, "b_srs must be in", id="b-range"),
        pytest.param({"b_hop": 4}, ValueError, "b_hop must be in", id="hop-range"),
        pytest.param(
            {"frequency_domain_position": -1},
            ValueError,
            "frequency_domain_position must be nonnegative",
            id="position",
        ),
        pytest.param({"n_srs": "1"}, TypeError, "n_srs must be an integer", id="n-type"),
        pytest.param({"n_srs": -1}, ValueError, "n_srs must be nonnegative", id="n-range"),
    ],
)
def test_frequency_positions_reject_out_of_range_parameters(kwargs, error, message) -> None:
    arguments = {"c_srs": 0, "b_srs": 0, "b_hop": 0}
    arguments.update(kwargs)
    with pytest.raises(error, match=message):
        nr_srs_frequency_position_indices(**arguments)


def test_frequency_position_indices_reject_boolean_fields() -> None:
    with pytest.raises(TypeError, match="c_srs must be an integer"):
        nr_srs_frequency_position_indices(True, b_srs=0, b_hop=0)


def test_positioning_symbol_offset_requires_an_integer_comb_and_known_symbols() -> None:
    with pytest.raises(TypeError, match="transmission_comb must be an integer|must be an integer"):
        nr_srs_positioning_symbol_offset("2", srs_symbol_count=1, resource_symbol_index=0)
    with pytest.raises(ValueError, match="symbol count is unsupported"):
        nr_srs_positioning_symbol_offset(2, srs_symbol_count=3, resource_symbol_index=0)
    with pytest.raises(ValueError, match="transmission_comb must be 2, 4, or 8"):
        nr_srs_positioning_symbol_offset(3, srs_symbol_count=1, resource_symbol_index=0)


@pytest.mark.parametrize(
    ("periodicity", "message"),
    [
        pytest.param({"sl20": 7}, None, id="valid"),
        pytest.param({1: 7}, "must be a string", id="key-type"),
    ],
)
def test_periodicity_decoding_checks_the_choice_key(periodicity, message) -> None:
    if message is None:
        slots, offset = nr_srs_periodicity_and_offset_slots(periodicity)
        assert slots == 20 and offset == 7
    else:
        with pytest.raises(TypeError, match=message):
            nr_srs_periodicity_and_offset_slots(periodicity)


@pytest.mark.parametrize(
    ("ports", "kwargs", "error", "message"),
    [
        pytest.param(
            "port1",
            {"cyclic_shift_hopping_enabled": 1},
            TypeError,
            "flags must be booleans",
            id="flag-type",
        ),
        pytest.param(
            None, {"nrof_srs_ports_n8": 8}, TypeError, "must be a string enum", id="ports8-type"
        ),
        pytest.param(
            "ports4",
            {"nrof_srs_ports_n8": "ports16"},
            ValueError,
            "must be ports8 or ports8tdm",
            id="ports8-value",
        ),
        pytest.param(4, {}, TypeError, "nrofSRS-Ports must be a string enum", id="ports-type"),
        pytest.param(
            "ports8", {}, ValueError, "must be port1, ports2, or ports4", id="ports-value"
        ),
    ],
)
def test_rrc_srs_port_configuration_rejects_mistyped_or_unsupported_enums(
    ports, kwargs, error, message
) -> None:
    with pytest.raises(error, match=message):
        nr_srs_port_configuration_from_rrc(ports, **kwargs)
