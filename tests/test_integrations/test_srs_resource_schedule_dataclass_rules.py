"""SRS resource schedule dataclasses: slot, identity, allocation and uniqueness rules."""

from __future__ import annotations

import pytest
from integrations.sionna.srs import (
    NRSRSAperiodicResourceSchedule,
    NRSRSPeriodicResourceSetSchedule,
    NRSRSResourceSetAperiodicSchedule,
    nr_srs_fixed_frequency_allocation,
)


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
