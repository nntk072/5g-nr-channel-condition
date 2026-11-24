"""SRS port and bandwidth-part records: complete rosters, indicator widths, aligned BWP outcomes and bounds."""

from __future__ import annotations

import pytest
from integrations.sionna.srs.srs_constants.ports_and_bwp import (
    NRSRSBWPFrequencyBounds,
    NRSRSBWPSelection,
    NRSRSPortConfiguration,
)


@pytest.mark.parametrize(
    ("ports", "tdm", "error", "message"),
    [
        pytest.param([1000], False, TypeError, "nonempty tuple", id="list-not-tuple"),
        pytest.param((), False, TypeError, "nonempty tuple", id="empty"),
        pytest.param(("a",), False, TypeError, "must contain integers", id="strings"),
        pytest.param((1000,), 1, TypeError, "must be a boolean", id="tdm-type"),
        pytest.param((1000, 1002), False, ValueError, "complete ordered SRS port roster", id="gap"),
        pytest.param((1000, 1001), True, ValueError, "complete eight-port roster", id="tdm-short"),
    ],
)
def test_port_configuration_rejects_incomplete_or_mistyped_rosters(ports, tdm, error, message):
    with pytest.raises(error, match=message):
        NRSRSPortConfiguration(antenna_ports=ports, ports8_tdm=tdm)


def test_complete_port_roster_is_accepted() -> None:
    config = NRSRSPortConfiguration(antenna_ports=(1000, 1001), ports8_tdm=False)
    assert config.antenna_ports == (1000, 1001)


def test_indicator_width_and_bwp_rows_are_validated() -> None:
    with pytest.raises(ValueError, match=r"indicator_bit_width must be in \[0, 2\]"):
        NRSRSBWPSelection(
            indicator_bit_width=3,
            selected_bwp_id_by_cell=(("cell-a", 0),),
            outcome_by_cell=(("cell-a", "single_bwp"),),
        )
    with pytest.raises(ValueError, match="nonempty and aligned"):
        NRSRSBWPSelection(
            indicator_bit_width=1,
            selected_bwp_id_by_cell=(),
            outcome_by_cell=(("cell-a", "single_bwp"),),
        )
    with pytest.raises(ValueError, match="identical cell order"):
        NRSRSBWPSelection(
            indicator_bit_width=1,
            selected_bwp_id_by_cell=(("cell-a", 0),),
            outcome_by_cell=(("cell-b", "single_bwp"),),
        )
    with pytest.raises(ValueError, match="cell IDs and BWP IDs in"):
        NRSRSBWPSelection(
            indicator_bit_width=1,
            selected_bwp_id_by_cell=(("cell-a", 9),),
            outcome_by_cell=(("cell-a", "single_bwp"),),
        )
    with pytest.raises(ValueError, match="invalid status"):
        NRSRSBWPSelection(
            indicator_bit_width=1,
            selected_bwp_id_by_cell=(("cell-a", 0),),
            outcome_by_cell=(("cell-a", "guessed"),),
        )


@pytest.mark.parametrize(
    ("start", "size", "error", "message"),
    [
        pytest.param(0.5, 10, TypeError, "start_subcarrier must be an integer", id="float"),
        pytest.param(True, 10, TypeError, "start_subcarrier must be an integer", id="bool"),
        pytest.param(-1, 10, ValueError, "start must be nonnegative", id="negative-start"),
        pytest.param(0, 0, ValueError, "size must be positive", id="empty-size"),
    ],
)
def test_bwp_frequency_bounds_are_integer_and_bounded(start, size, error, message) -> None:
    with pytest.raises(error, match=message):
        NRSRSBWPFrequencyBounds(start_subcarrier=start, size_subcarriers=size)
