"""SRS BWP schedule dataclasses: cell, BWP index, frequency bounds and set-schedule typing."""

from __future__ import annotations

import pytest
from integrations.sionna.srs import (
    NRSRSBWPFrequencyBounds,
    NRSRSResourceSetAperiodicSchedule,
    nr_srs_fixed_frequency_allocation,
)
from integrations.sionna.srs.srs_bwp_schedules import NRSRSBWPCellResourceSetSchedules


@pytest.fixture
def bounds():
    return NRSRSBWPFrequencyBounds(start_subcarrier=0, size_subcarriers=48)


@pytest.fixture
def resource_set():
    allocation = nr_srs_fixed_frequency_allocation(0, transmission_comb=2, b_srs=0, b_hop=0)
    return NRSRSResourceSetAperiodicSchedule(0, 3, (allocation,))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        pytest.param({"cell_id": ""}, "cell_id must be a nonempty string", id="cell"),
        pytest.param({"ul_bwp_id": 5}, "ul_bwp_id must be in", id="bwp-range"),
        pytest.param({"ul_bwp_id": True}, "ul_bwp_id must be in", id="bwp-bool"),
        pytest.param(
            {"frequency_bounds": "x"},
            "frequency_bounds must be NRSRSBWPFrequencyBounds",
            id="bounds-type",
        ),
        pytest.param(
            {"resource_set_schedules": ("x",)},
            "resource_set_schedules must contain SRS set schedules",
            id="sets-type",
        ),
    ],
)
def test_aperiodic_bwp_schedules_validate_identity(bounds, resource_set, kwargs, message) -> None:
    arguments = {
        "cell_id": "cell-0",
        "ul_bwp_id": 0,
        "frequency_bounds": bounds,
        "resource_set_schedules": (resource_set,),
    }
    arguments.update(kwargs)
    with pytest.raises((TypeError, ValueError), match=message):
        NRSRSBWPCellResourceSetSchedules(**arguments)
