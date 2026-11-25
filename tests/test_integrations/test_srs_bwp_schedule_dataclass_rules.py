"""SRS BWP schedule dataclasses: cell, BWP index, frequency bounds and set-schedule typing."""

from __future__ import annotations

import pytest
from integrations.sionna.srs import (
    NRSRSBWPFrequencyBounds,
    NRSRSResourceSetAperiodicSchedule,
    nr_srs_fixed_frequency_allocation,
    nr_srs_periodic_resource_set_schedule,
)
from integrations.sionna.srs.srs_bwp_schedules import (
    NRSRSBWPCellPeriodicResourceSetSchedules,
    NRSRSBWPCellResourceSetSchedules,
    NRSRSBWPPeriodicResourceSetSchedule,
)


@pytest.fixture
def bounds():
    return NRSRSBWPFrequencyBounds(start_subcarrier=0, size_subcarriers=120)


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


@pytest.fixture
def periodic_resource_set():
    return nr_srs_periodic_resource_set_schedule(
        0,
        7,
        (
            {
                "resource_id": 3,
                "allocation": {
                    "c_srs": 0,
                    "transmission_comb": 2,
                    "b_srs": 0,
                    "b_hop": 0,
                    "srs_symbol_count": 1,
                },
            },
        ),
        periodicity_slots=1,
        slot_offset=0,
    )


@pytest.fixture
def periodic_bwp_schedule(bounds, periodic_resource_set):
    return NRSRSBWPPeriodicResourceSetSchedule("cell-0", 0, bounds, periodic_resource_set)


def test_periodic_bwp_schedule_binds_active_resource_set(
    periodic_resource_set, periodic_bwp_schedule
) -> None:
    assert periodic_bwp_schedule.resource_set_schedule is periodic_resource_set
    assert periodic_bwp_schedule.cell_id == "cell-0"


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"cell_id": "  "}, ValueError, "cell_id must be a nonempty string"),
        ({"ul_bwp_id": True}, ValueError, "ul_bwp_id must be in"),
        ({"frequency_bounds": None}, TypeError, "frequency_bounds must be"),
        ({"resource_set_schedule": None}, TypeError, "resource_set_schedule must be"),
    ],
)
def test_periodic_bwp_schedule_rejects_invalid_identity_and_types(
    bounds, periodic_resource_set, overrides, error, message
) -> None:
    arguments = {
        "cell_id": "cell-0",
        "ul_bwp_id": 0,
        "frequency_bounds": bounds,
        "resource_set_schedule": periodic_resource_set,
    }
    arguments.update(overrides)
    with pytest.raises(error, match=message):
        NRSRSBWPPeriodicResourceSetSchedule(**arguments)


def test_periodic_bwp_schedule_rejects_resource_outside_selected_bwp(periodic_resource_set) -> None:
    narrow_bounds = NRSRSBWPFrequencyBounds(start_subcarrier=100, size_subcarriers=100)
    with pytest.raises(ValueError, match="lies outside selected UL BWP"):
        NRSRSBWPPeriodicResourceSetSchedule("cell-0", 0, narrow_bounds, periodic_resource_set)


def test_periodic_cell_schedules_require_consistent_unique_bwp_bindings(
    bounds, periodic_bwp_schedule
) -> None:
    valid = NRSRSBWPCellPeriodicResourceSetSchedules(
        " cell-0 ", 0, bounds, (periodic_bwp_schedule,)
    )
    assert valid.cell_id == "cell-0"
    assert valid.resource_set_schedules == (periodic_bwp_schedule,)
    with pytest.raises(ValueError, match="share the cell's active-BWP binding"):
        NRSRSBWPCellPeriodicResourceSetSchedules("cell-1", 0, bounds, (periodic_bwp_schedule,))
    with pytest.raises(ValueError, match="IDs must be unique"):
        NRSRSBWPCellPeriodicResourceSetSchedules(
            "cell-0", 0, bounds, (periodic_bwp_schedule, periodic_bwp_schedule)
        )


@pytest.mark.parametrize(
    ("cell_id", "bwp_id", "frequency_bounds", "schedules", "error", "message"),
    [
        (" ", 0, None, (), ValueError, "cell_id must be a nonempty string"),
        ("cell-a", True, None, (), ValueError, "ul_bwp_id must be in"),
        ("cell-a", 0, None, (), TypeError, "frequency_bounds must be"),
        (
            "cell-a",
            0,
            NRSRSBWPFrequencyBounds(start_subcarrier=0, size_subcarriers=100),
            ("invalid",),
            TypeError,
            "BWP-validated periodic sets",
        ),
    ],
)
def test_periodic_cell_schedule_rejects_invalid_context_fields(
    cell_id, bwp_id, frequency_bounds, schedules, error, message
) -> None:
    with pytest.raises(error, match=message):
        NRSRSBWPCellPeriodicResourceSetSchedules(cell_id, bwp_id, frequency_bounds, schedules)


@pytest.mark.parametrize(
    ("slot", "resources", "error", "message"),
    [
        (True, (), TypeError, "absolute_slot_index must be an integer"),
        (-1, (), ValueError, "absolute_slot_index must be nonnegative"),
        (0, "bad", TypeError, "resources must be an ordered sequence"),
        (0, (), ValueError, "resources must contain at least one"),
        (0, ("bad",), TypeError, "resources\\[0\\] must be a mapping"),
        (0, ({"resource_id": True},), TypeError, "resource_id must be an integer"),
        (0, ({"resource_id": -1},), ValueError, "unique nonnegative integers"),
        (
            0,
            ({"resource_id": 1}, {"resource_id": 1}),
            ValueError,
            "unique nonnegative integers",
        ),
        (
            0,
            ({"resource_id": 1, "unexpected": True},),
            ValueError,
            "unsupported fields",
        ),
        (
            0,
            ({"resource_id": 1, "allocation": "bad"},),
            ValueError,
            "allocation must be a mapping",
        ),
        (
            0,
            ({"resource_id": 1, "allocation": {"c_srs": 0}},),
            ValueError,
            "allocation fields are invalid",
        ),
        (
            0,
            ({"resource_id": 1, "allocation": {1: 0}},),
            ValueError,
            "allocation keys must be strings",
        ),
        (
            0,
            (
                {
                    "resource_id": 1,
                    "allocation": {
                        "c_srs": 0,
                        "transmission_comb": 2,
                        "b_srs": 0,
                        "b_hop": 0,
                        "srs_symbol_count": 1,
                    },
                    "timing": "bad",
                },
            ),
            ValueError,
            "timing must be a mapping",
        ),
        (
            0,
            (
                {
                    "resource_id": 1,
                    "allocation": {
                        "c_srs": 0,
                        "transmission_comb": 2,
                        "b_srs": 0,
                        "b_hop": 0,
                        "srs_symbol_count": 1,
                    },
                    "timing": {1: 2},
                },
            ),
            ValueError,
            "timing keys must be strings",
        ),
        (
            0,
            (
                {
                    "resource_id": 1,
                    "allocation": {
                        "c_srs": 0,
                        "transmission_comb": 2,
                        "b_srs": 0,
                        "b_hop": 0,
                        "srs_symbol_count": 1,
                    },
                    "timing": {"unexpected": 2},
                },
            ),
            ValueError,
            "unsupported fields",
        ),
    ],
)
def test_periodic_resource_set_rejects_malformed_decoded_resources(
    slot, resources, error, message
) -> None:
    with pytest.raises(error, match=message):
        nr_srs_periodic_resource_set_schedule(
            slot,
            0,
            resources,
            periodicity_slots=1,
            slot_offset=0,
        )
