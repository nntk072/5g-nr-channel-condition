"""SRS resource schedule dataclasses: slot, identity, allocation and uniqueness rules."""

from __future__ import annotations

from dataclasses import replace

import pytest
from integrations.sionna.srs import (
    NRSRSAperiodicResourceSchedule,
    NRSRSPeriodicResourceSetSchedule,
    NRSRSResourceSetAperiodicSchedule,
    nr_srs_fixed_frequency_allocation,
    srs_configuration,
)
from integrations.sionna.srs.srs_resource_schedules import NRSRSScheduledResourceOccurrence


@pytest.fixture
def allocation():
    return nr_srs_fixed_frequency_allocation(0, transmission_comb=2, b_srs=0, b_hop=0)


@pytest.mark.parametrize("slot", [-1, True, "3"])
def test_aperiodic_schedule_slot_must_be_a_nonnegative_integer(allocation, slot) -> None:
    with pytest.raises(ValueError, match="target_slot_index must be a nonnegative integer"):
        NRSRSAperiodicResourceSchedule(slot, (allocation,))


def test_aperiodic_schedule_requires_fixed_allocations(allocation) -> None:
    with pytest.raises(
        ValueError, match="allocations must contain fixed SRS frequency allocations"
    ):
        NRSRSAperiodicResourceSchedule(3, ())
    with pytest.raises(
        ValueError, match="allocations must contain fixed SRS frequency allocations"
    ):
        NRSRSAperiodicResourceSchedule(3, ("not-an-allocation",))


def test_periodic_set_schedule_validates_its_identities(allocation) -> None:
    schedule = NRSRSAperiodicResourceSchedule(3, (allocation,))
    with pytest.raises(ValueError, match="resource_set_id must be a nonnegative integer"):
        NRSRSPeriodicResourceSetSchedule(-1, 3, ((0, schedule),))
    with pytest.raises(ValueError, match="target_slot_index must be a nonnegative integer"):
        NRSRSPeriodicResourceSetSchedule(0, -3, ((0, schedule),))
    with pytest.raises(ValueError, match="resources must contain at least one"):
        NRSRSPeriodicResourceSetSchedule(0, 3, ())


def test_periodic_set_schedule_checks_each_resource_entry(allocation) -> None:
    schedule = NRSRSAperiodicResourceSchedule(3, (allocation,))
    with pytest.raises(TypeError, match="each resource entry must be a"):
        NRSRSPeriodicResourceSetSchedule(0, 3, (schedule,))
    with pytest.raises(ValueError, match="resource IDs must be nonnegative"):
        NRSRSPeriodicResourceSetSchedule(0, 3, ((-1, schedule),))
    with pytest.raises(TypeError, match="each resource schedule must be a periodic SRS schedule"):
        NRSRSPeriodicResourceSetSchedule(0, 3, ((0, allocation),))
    other_slot = NRSRSAperiodicResourceSchedule(4, (allocation,))
    with pytest.raises(ValueError, match="must target the resource-set slot"):
        NRSRSPeriodicResourceSetSchedule(0, 3, ((0, other_slot),))
    with pytest.raises(ValueError, match="resource IDs must be unique"):
        NRSRSPeriodicResourceSetSchedule(0, 3, ((0, schedule), (0, schedule)))


def test_fixed_frequency_allocation_rejects_an_invalid_optional_counter(
    allocation, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        srs_configuration,
        "nr_srs_frequency_position_indices",
        lambda *_args, **_kwargs: allocation.position_indices,
    )
    with pytest.raises(ValueError, match="n_srs must be a nonnegative integer"):
        replace(allocation, n_srs=True)


def test_resource_set_aperiodic_schedule_identity_and_allocations(allocation) -> None:
    with pytest.raises(ValueError, match="resource_set_id must be a nonnegative integer"):
        NRSRSResourceSetAperiodicSchedule(-1, 3, (allocation,))
    with pytest.raises(ValueError, match="target_slot_index must be a nonnegative integer"):
        NRSRSResourceSetAperiodicSchedule(0, -3, (allocation,))
    with pytest.raises(
        ValueError, match="allocations must contain fixed SRS frequency allocations"
    ):
        NRSRSResourceSetAperiodicSchedule(0, 3, ())
    assert NRSRSResourceSetAperiodicSchedule(0, 3, (allocation,)).target_slot_index == 3


@pytest.fixture
def scheduled_occurrence(allocation):
    return NRSRSScheduledResourceOccurrence(
        cell_id="cell-0",
        resource_set_id=3,
        resource_id=4,
        grid_id="grid-a",
        carrier_hz=3.5e9,
        numerology=1,
        subcarrier_spacing_hz=30_000.0,
        target_slot_index=7,
        slot_symbol_count=14,
        start_symbol_index=10,
        srs_symbol_count=2,
        allocations=(allocation,),
    )


def test_scheduled_occurrence_normalizes_shared_grid_context(scheduled_occurrence) -> None:
    occurrence = replace(scheduled_occurrence, cell_id=" cell-0 ", grid_id=" grid-a ")
    assert occurrence.cell_id == "cell-0"
    assert occurrence.grid_id == "grid-a"
    assert occurrence.subcarrier_spacing_hz == 30_000.0
    assert occurrence.allocations == scheduled_occurrence.allocations


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"cell_id": " "}, ValueError, "cell_id must be a nonempty string"),
        ({"grid_id": " "}, ValueError, "grid_id must be a nonempty string"),
        ({"resource_set_id": True}, TypeError, "resource_set_id must be an integer"),
        ({"resource_id": -1}, ValueError, "IDs must be nonnegative"),
        ({"numerology": 7}, ValueError, "numerology must be between 0 and 6"),
        ({"target_slot_index": -1}, ValueError, "target_slot_index must be nonnegative"),
        ({"slot_symbol_count": 13}, ValueError, "slot_symbol_count must be 12 or 14"),
        ({"srs_symbol_count": 3}, ValueError, "unsupported for the slot"),
        ({"allocations": ()}, ValueError, "allocations must contain fixed"),
        ({"allocations": (None,)}, ValueError, "allocations must contain fixed"),
        ({"allocations": (scheduled_occurrence,)}, ValueError, "allocations must contain fixed"),
        ({"start_symbol_index": 14}, ValueError, "must lie inside the slot"),
        ({"start_symbol_index": 13}, ValueError, "extend past the slot"),
        ({"carrier_hz": True}, TypeError, "carrier_hz must be a finite positive number"),
        ({"carrier_hz": float("nan")}, ValueError, "carrier_hz must be a finite positive number"),
        ({"carrier_hz": 0.0}, ValueError, "carrier_hz must be a finite positive number"),
        ({"subcarrier_spacing_hz": False}, TypeError, "subcarrier_spacing_hz must be"),
        ({"subcarrier_spacing_hz": 15_000.0}, ValueError, "disagrees with numerology"),
    ],
)
def test_scheduled_occurrence_rejects_invalid_grid_context(
    scheduled_occurrence, overrides, error, message
) -> None:
    with pytest.raises(error, match=message):
        replace(scheduled_occurrence, **overrides)
