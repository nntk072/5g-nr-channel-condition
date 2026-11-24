"""SRS periodic and aperiodic scheduling, frequency hopping and resource-set timing."""

import numpy as np
import pytest
from integrations.sionna.srs import (
    NRSRSAperiodicResourceSchedule,
    NRSRSBWPCellPeriodicResourceSetSchedules,
    NRSRSBWPFrequencyBounds,
    NRSRSBWPPeriodicResourceSetSchedule,
    NRSRSFixedFrequencyAllocation,
    NRSRSHoppedResourceGrid,
    NRSRSPeriodicResourceSetSchedule,
    map_nr_srs_hopped_resource,
    map_nr_srs_multiport_resource,
    nr_srs_aperiodic_resource_schedule,
    nr_srs_aperiodic_resource_set_schedules,
    nr_srs_aperiodic_target_slot_index,
    nr_srs_cyclic_shift_hopping_offset,
    nr_srs_cyclic_shift_hopping_offsets,
    nr_srs_fixed_frequency_allocation,
    nr_srs_frequency_position_indices,
    nr_srs_group_hopping_index,
    nr_srs_hopped_sequence_type1,
    nr_srs_multicell_aperiodic_resource_set_schedules,
    nr_srs_multicell_periodic_resource_set_schedules_by_bwp,
    nr_srs_periodic_resource_schedule,
    nr_srs_periodic_resource_set_schedule,
    nr_srs_periodic_resource_set_schedule_by_bwp,
    nr_srs_periodicity_and_offset_slots,
    nr_srs_port_configuration_from_rrc,
    nr_srs_port_sequences_type1,
    nr_srs_port_symbol_activity,
    nr_srs_resource_frequency_allocations,
    nr_srs_resource_sequences_type1,
    nr_srs_scheduled_port_sequences_type1,
    nr_srs_sequence_hopping_index,
    nr_srs_sequence_type1,
    nr_srs_transmission_counter,
)
from tests.test_integrations.srs_resource_mapping_support import (
    _scalar_gold_bits,
)


def test_srs_fixed_frequency_allocation_resolves_table_tree() -> None:
    allocation = nr_srs_fixed_frequency_allocation(
        55,
        transmission_comb=4,
        b_srs=3,
        b_hop=3,
        frequency_domain_position=7,
        frequency_domain_shift_rb=2,
        reference_subcarrier=5,
        comb_offset=2,
    )

    assert isinstance(allocation, NRSRSFixedFrequencyAllocation)
    assert allocation.position_indices == (0, 0, 1, 1)
    assert allocation.bandwidth.bandwidth_rb == 8
    assert allocation.bandwidth.subband_count == 2
    assert allocation.start_subcarrier == 317
    assert allocation.comb_offset == 2
    assert allocation.bandwidth.allocated_bandwidth_subcarriers == 96


def test_srs_frequency_hopping_tree_indices_match_counter_sequence() -> None:
    expected = (
        (0, 0, 1, 1),
        (0, 1, 1, 1),
        (0, 0, 0, 1),
        (0, 1, 0, 1),
        (0, 0, 1, 0),
        (0, 1, 1, 0),
        (0, 0, 0, 0),
        (0, 1, 0, 0),
    )
    actual = tuple(
        nr_srs_frequency_position_indices(
            9,
            b_srs=3,
            b_hop=0,
            frequency_domain_position=3,
            n_srs=counter,
        )
        for counter in range(len(expected))
    )

    assert actual == expected


def test_srs_frequency_hopping_handles_odd_and_even_tree_radices() -> None:
    expected = (
        (0, 0, 0, 0),
        (0, 0, 1, 0),
        (0, 0, 2, 0),
        (0, 0, 0, 1),
        (0, 0, 1, 1),
        (0, 0, 2, 1),
    )
    actual = tuple(
        nr_srs_frequency_position_indices(55, b_srs=3, b_hop=1, n_srs=counter)
        for counter in range(len(expected))
    )

    assert actual == expected


def test_srs_frequency_hopping_requires_explicit_transmission_counter() -> None:
    with pytest.raises(ValueError, match="n_srs is required"):
        nr_srs_frequency_position_indices(9, b_srs=3, b_hop=1)

    allocation = nr_srs_fixed_frequency_allocation(
        9,
        transmission_comb=4,
        b_srs=3,
        b_hop=0,
        frequency_domain_position=3,
        n_srs=1,
    )
    assert allocation.position_indices == (0, 1, 1, 1)
    assert allocation.n_srs == 1
    assert allocation.b_hop == 0


def test_srs_resource_frequency_allocations_join_periodic_counter_and_hops() -> None:
    def resolve(slot: int):
        return nr_srs_resource_frequency_allocations(
            9,
            transmission_comb=4,
            b_srs=3,
            b_hop=0,
            srs_symbol_count=2,
            resource_type="periodic",
            frequency_domain_position=3,
            repetition_factor=1,
            absolute_slot_index=slot,
            periodicity_slots=4,
            slot_offset=1,
        )

    first_occasion = resolve(1)
    assert first_occasion is not None
    assert tuple(item.n_srs for item in first_occasion) == (0, 1)
    assert tuple(item.position_indices for item in first_occasion) == (
        (0, 0, 1, 1),
        (0, 1, 1, 1),
    )
    next_occasion = resolve(5)
    assert next_occasion is not None
    assert tuple(item.n_srs for item in next_occasion) == (2, 3)
    assert tuple(item.position_indices for item in next_occasion) == (
        (0, 0, 0, 1),
        (0, 1, 0, 1),
    )
    assert resolve(4) is None


@pytest.mark.parametrize(
    ("choice", "period"),
    [
        ("sl1", 1),
        ("sl2", 2),
        ("sl5", 5),
        ("sl16", 16),
        ("sl640", 640),
        ("sl1280", 1280),
        ("sl2560", 2560),
        ("sl5120", 5120),
        ("sl10240", 10240),
        ("sl40960", 40960),
        ("sl81920", 81920),
    ],
)
def test_srs_periodicity_choice_decodes_slot_period_and_offset(choice: str, period: int) -> None:
    encoded_offset = None if period == 1 else period - 1
    assert nr_srs_periodicity_and_offset_slots({choice: encoded_offset}) == (
        period,
        0 if period == 1 else period - 1,
    )


@pytest.mark.parametrize(
    ("choice", "period"),
    [("sl128", 128), ("sl256", 256), ("sl512", 512), ("sl20480", 20480)],
)
def test_srs_periodicity_extension_choice_decodes_and_overrides_base(
    choice: str, period: int
) -> None:
    assert nr_srs_periodicity_and_offset_slots(
        {"sl20": 7}, periodicity_and_offset_ext={choice: period - 1}
    ) == (period, period - 1)


@pytest.mark.parametrize(
    "choice",
    [
        {},
        {"sl4": 1, "sl8": 2},
        {"sl4": 4},
        {"sl4": True},
        {"sl128": 1},
        {"sl1": 0},
    ],
)
def test_srs_periodicity_choice_rejects_malformed_or_wrong_extension(
    choice: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        nr_srs_periodicity_and_offset_slots(choice)


def test_srs_allocation_resolves_rrc_periodicity_choice() -> None:
    allocations = nr_srs_resource_frequency_allocations(
        9,
        transmission_comb=4,
        b_srs=3,
        b_hop=0,
        srs_symbol_count=2,
        repetition_factor=1,
        resource_type="periodic",
        frequency_domain_position=3,
        absolute_slot_index=5,
        periodicity_and_offset={"sl4": 1},
    )
    assert allocations is not None
    assert tuple(item.n_srs for item in allocations) == (2, 3)


def test_srs_allocation_rejects_conflicting_periodicity_sources() -> None:
    with pytest.raises(ValueError, match="supply resolved periodicity or the RRC CHOICE"):
        nr_srs_resource_frequency_allocations(
            9,
            transmission_comb=4,
            b_srs=3,
            b_hop=0,
            srs_symbol_count=2,
            repetition_factor=1,
            resource_type="periodic",
            frequency_domain_position=3,
            periodicity_slots=4,
            periodicity_and_offset={"sl4": 1},
        )


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        (
            {
                "resource_type": "aperiodic",
                "periodicity_and_offset": {"sl4": 1},
                "periodicity_slots": None,
            },
            ValueError,
            "aperiodic SRS",
        ),
        ({"srs_symbol_count": True}, TypeError, "srs_symbol_count must be an integer"),
        ({"srs_symbol_count": 3}, ValueError, "srs_symbol_count is unsupported"),
    ],
)
def test_srs_frequency_allocation_rejects_invalid_schedule_inputs(kwargs, error, message) -> None:
    arguments = {
        "c_srs": 9,
        "transmission_comb": 4,
        "b_srs": 3,
        "b_hop": 0,
        "srs_symbol_count": 2,
        "repetition_factor": 1,
        "resource_type": "periodic",
        "frequency_domain_position": 3,
        "absolute_slot_index": 1,
        "periodicity_slots": 1,
    }
    arguments.update(kwargs)
    with pytest.raises(error, match=message):
        nr_srs_resource_frequency_allocations(**arguments)


def test_periodic_srs_resource_set_composes_resources_with_individual_occasions() -> None:
    allocation = {
        "c_srs": 9,
        "transmission_comb": 4,
        "b_srs": 3,
        "b_hop": 0,
        "srs_symbol_count": 2,
    }
    schedule = nr_srs_periodic_resource_set_schedule(
        4,
        12,
        (
            {"resource_id": 101, "allocation": allocation},
            {
                "resource_id": 102,
                "allocation": {**allocation, "frequency_domain_position": 2},
                "timing": {"periodicity_slots": 5, "slot_offset": 1},
            },
        ),
        periodicity_slots=4,
        slot_offset=0,
    )

    assert isinstance(schedule, NRSRSPeriodicResourceSetSchedule)
    assert schedule.resource_set_id == 12
    assert schedule.target_slot_index == 4
    assert tuple(resource_id for resource_id, _ in schedule.resources) == (101,)
    assert schedule.resources[0][1].target_slot_index == 4
    assert len(schedule.resources[0][1].allocations) == 2


def test_periodic_srs_resource_set_is_inactive_when_no_resource_has_an_occasion() -> None:
    schedule = nr_srs_periodic_resource_set_schedule(
        2,
        12,
        (
            {
                "resource_id": 101,
                "allocation": {
                    "c_srs": 9,
                    "transmission_comb": 4,
                    "b_srs": 3,
                    "b_hop": 0,
                    "srs_symbol_count": 2,
                },
            },
        ),
        periodicity_slots=4,
        slot_offset=0,
    )

    assert schedule is None


def test_periodic_srs_resource_set_rejects_duplicate_ids_and_unknown_fields() -> None:
    resource = {
        "resource_id": 101,
        "allocation": {
            "c_srs": 9,
            "transmission_comb": 4,
            "b_srs": 3,
            "b_hop": 0,
            "srs_symbol_count": 2,
        },
    }
    with pytest.raises(ValueError, match="unique nonnegative"):
        nr_srs_periodic_resource_set_schedule(0, 12, (resource, resource))
    with pytest.raises(ValueError, match="unsupported fields"):
        nr_srs_periodic_resource_set_schedule(
            0,
            12,
            ({**resource, "periodicity_slots": 4},),
        )
    with pytest.raises(ValueError, match="unsupported fields"):
        nr_srs_periodic_resource_set_schedule(
            0,
            12,
            ({**resource, "timing": {"unknown": 4}},),
        )


def test_periodic_srs_resource_set_checks_selected_bwp_frequency_bounds() -> None:
    resource = {
        "resource_id": 101,
        "allocation": {
            "c_srs": 9,
            "transmission_comb": 4,
            "b_srs": 3,
            "b_hop": 0,
            "srs_symbol_count": 2,
        },
    }
    scheduled = nr_srs_periodic_resource_set_schedule_by_bwp(
        0,
        12,
        (resource,),
        cell_id="cell-1",
        ul_bwp_id=2,
        frequency_bounds=NRSRSBWPFrequencyBounds(0, 100_000),
        periodicity_slots=4,
        slot_offset=0,
    )

    assert isinstance(scheduled, NRSRSBWPPeriodicResourceSetSchedule)
    assert scheduled.cell_id == "cell-1"
    assert scheduled.ul_bwp_id == 2
    assert scheduled.resource_set_schedule.resource_set_id == 12
    with pytest.raises(ValueError, match="outside selected UL BWP"):
        nr_srs_periodic_resource_set_schedule_by_bwp(
            0,
            12,
            (resource,),
            cell_id="cell-1",
            ul_bwp_id=2,
            frequency_bounds=NRSRSBWPFrequencyBounds(0, 1),
            periodicity_slots=4,
            slot_offset=0,
        )


def test_multicell_periodic_srs_schedule_preserves_cell_order_and_active_bwp() -> None:
    resource_set = {
        "resource_set_id": 12,
        "resource_type": "periodic",
        "timing": {"periodicity_slots": 4, "slot_offset": 0},
        "resources": (
            {
                "resource_id": 101,
                "allocation": {
                    "c_srs": 9,
                    "transmission_comb": 4,
                    "b_srs": 3,
                    "b_hop": 0,
                    "srs_symbol_count": 2,
                },
            },
        ),
    }
    schedules = nr_srs_multicell_periodic_resource_set_schedules_by_bwp(
        ("cell-1", "cell-2"),
        {"cell-1": (resource_set,), "cell-2": (resource_set,)},
        {"cell-1": 4, "cell-2": 2},
        {"cell-1": 1, "cell-2": 2},
        {
            "cell-1": NRSRSBWPFrequencyBounds(0, 100_000),
            "cell-2": NRSRSBWPFrequencyBounds(0, 100_000),
        },
    )

    assert tuple(item.cell_id for item in schedules) == ("cell-1", "cell-2")
    assert tuple(item.ul_bwp_id for item in schedules) == (1, 2)
    assert isinstance(schedules[0], NRSRSBWPCellPeriodicResourceSetSchedules)
    assert len(schedules[0].resource_set_schedules) == 1
    assert schedules[0].resource_set_schedules[0].resource_set_schedule.target_slot_index == 4
    assert schedules[1].resource_set_schedules == ()


def test_multicell_periodic_srs_requires_exact_cell_rosters() -> None:
    with pytest.raises(ValueError, match="keys must exactly match"):
        nr_srs_multicell_periodic_resource_set_schedules_by_bwp(
            ("cell-1", "cell-2"),
            {"cell-1": ()},
            {"cell-1": 0, "cell-2": 0},
            {"cell-1": 0, "cell-2": 0},
            {
                "cell-1": NRSRSBWPFrequencyBounds(0, 100),
                "cell-2": NRSRSBWPFrequencyBounds(0, 100),
            },
        )


def test_srs_aperiodic_target_slot_uses_dci_offset_selection() -> None:
    assert nr_srs_aperiodic_target_slot_index(100) == 100
    assert nr_srs_aperiodic_target_slot_index(100, slot_offset=2) == 102
    assert (
        nr_srs_aperiodic_target_slot_index(
            100,
            slot_offset=2,
            available_slot_offsets=[4, 7],
            selected_available_offset_index=1,
        )
        == 109
    )


def test_aperiodic_srs_schedule_binds_target_slot_to_symbol_hops() -> None:
    schedule = nr_srs_aperiodic_resource_schedule(
        100,
        c_srs=9,
        transmission_comb=4,
        b_srs=3,
        b_hop=0,
        srs_symbol_count=2,
        slot_offset=2,
        available_slot_offsets=[4, 7],
        selected_available_offset_index=1,
        frequency_domain_position=3,
        repetition_factor=1,
    )

    assert isinstance(schedule, NRSRSAperiodicResourceSchedule)
    assert schedule.target_slot_index == 109
    assert tuple(item.n_srs for item in schedule.allocations) == (0, 1)
    assert tuple(item.position_indices for item in schedule.allocations) == (
        (0, 0, 1, 1),
        (0, 1, 1, 1),
    )


def test_aperiodic_srs_schedule_requires_offset_selection_for_configured_list() -> None:
    with pytest.raises(TypeError, match="selected_available_offset_index"):
        nr_srs_aperiodic_resource_schedule(
            100,
            c_srs=9,
            transmission_comb=4,
            b_srs=3,
            b_hop=0,
            srs_symbol_count=2,
            available_slot_offsets=[0, 4],
        )


def test_srs_aperiodic_target_slot_rejects_invalid_offsets() -> None:
    with pytest.raises((TypeError, ValueError)):
        nr_srs_aperiodic_target_slot_index(100, slot_offset=-1)
    with pytest.raises(ValueError, match="slot_offset"):
        nr_srs_aperiodic_target_slot_index(100, slot_offset=33)
    with pytest.raises(ValueError, match="slot_offset"):
        nr_srs_aperiodic_target_slot_index(100, slot_offset=0)
    with pytest.raises((TypeError, ValueError)):
        nr_srs_aperiodic_target_slot_index(
            100, available_slot_offsets=[], selected_available_offset_index=0
        )
    with pytest.raises((TypeError, ValueError)):
        nr_srs_aperiodic_target_slot_index(
            100, available_slot_offsets=[2, 2], selected_available_offset_index=0
        )
    with pytest.raises(ValueError, match="available slot offsets"):
        nr_srs_aperiodic_target_slot_index(
            100, available_slot_offsets=[8], selected_available_offset_index=0
        )
    with pytest.raises((TypeError, ValueError)):
        nr_srs_aperiodic_target_slot_index(
            100, available_slot_offsets=[2], selected_available_offset_index=1
        )
    with pytest.raises((TypeError, ValueError)):
        nr_srs_aperiodic_target_slot_index(100, selected_available_offset_index=0)


@pytest.mark.parametrize(
    ("offset_combo_index", "offset_combos", "offset_cells", "message"),
    [
        (0, None, (), "requires offset_combos"),
        (2, ((0,),), ("cell-3",), "does not select"),
        (0, ((0, 1),), ("cell-3",), "match offset_combo_cell_ids"),
    ],
)
def test_srs_multicell_schedule_rejects_invalid_offset_combos(
    offset_combo_index: int,
    offset_combos: tuple[tuple[int, ...], ...] | None,
    offset_cells: tuple[str, ...],
    message: str,
) -> None:
    sets = {
        cell: [
            {
                "resource_set_id": index,
                "resource_type": "aperiodic",
                "aperiodic_trigger": 1,
                "allocation": {
                    "c_srs": 9,
                    "transmission_comb": 4,
                    "b_srs": 3,
                    "b_hop": 0,
                    "srs_symbol_count": 1,
                    "available_slot_offsets": [0, 1],
                },
            }
        ]
        for cell, index in (("cell-3", 31), ("cell-9", 91))
    }
    with pytest.raises((TypeError, ValueError), match=message):
        nr_srs_multicell_aperiodic_resource_set_schedules(
            0,
            ((1, 1),),
            ("cell-3", "cell-9"),
            sets,
            {"cell-3": 20, "cell-9": 200},
            scheduled_cell_ids=("cell-3", "cell-9"),
            offset_combo_index=offset_combo_index,
            offset_combos=offset_combos,
            offset_combo_cell_ids=offset_cells,
        )


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        (
            {"scheduled_cell_ids": "cell-a"},
            TypeError,
            "scheduled_cell_ids must be an ordered sequence",
        ),
        ({"scheduled_cell_ids": ()}, ValueError, "nonempty cell IDs"),
        ({"scheduled_cell_ids": ("unknown",)}, ValueError, "unique subset of serving_cell_ids"),
        (
            {
                "scheduled_cell_ids": ("cell-b", "cell-a"),
                "serving_cell_ids": ("cell-a", "cell-b"),
                "request_list": ((1, 0),),
                "resource_sets_by_cell": {
                    "cell-a": (),
                    "cell-b": (),
                },
                "triggering_dci_slot_by_cell": {"cell-a": 12, "cell-b": 12},
            },
            ValueError,
            "serving-cell order",
        ),
        ({"triggering_dci_slot_by_cell": "bad"}, TypeError, "must be a mapping"),
        ({"triggering_dci_slot_by_cell": {"other": 1}}, ValueError, "keys must exactly match"),
        ({"triggering_dci_slot_by_cell": {"cell-a": True}}, ValueError, "must be nonnegative"),
        ({"offset_combo_cell_ids": "cell-a"}, TypeError, "must be an ordered sequence"),
        ({"offset_combo_cell_ids": ("unknown",)}, ValueError, "unique subset of serving_cell_ids"),
        (
            {"offset_combo_cell_ids": ("cell-a", "cell-a")},
            ValueError,
            "unique subset of serving_cell_ids",
        ),
        ({"offset_combos": "bad"}, TypeError, "offset_combos must be an ordered sequence"),
        ({"offset_combos": ()}, ValueError, "needs 1..8 rows"),
        ({"offset_combo_index": True}, TypeError, "must select a configured offset-combo row"),
        ({"offset_combo_index": 1}, ValueError, "does not select a configured row"),
        ({"offset_combos": ("bad",)}, TypeError, "selected offset-combo row must be a sequence"),
        ({"offset_combos": ((4,),)}, ValueError, "indices in \\[0, 3\\]"),
    ],
)
def test_multicell_aperiodic_schedule_rejects_invalid_cell_and_offset_inputs(
    changes, error, message
) -> None:
    arguments = {
        "request_list_index": 0,
        "request_list": ((1,),),
        "serving_cell_ids": ("cell-a",),
        "resource_sets_by_cell": {
            "cell-a": [
                {
                    "resource_set_id": 10,
                    "resource_type": "aperiodic",
                    "aperiodic_trigger": 1,
                    "allocation": {
                        "c_srs": 9,
                        "transmission_comb": 4,
                        "b_srs": 3,
                        "b_hop": 0,
                        "srs_symbol_count": 1,
                        "available_slot_offsets": [0, 1],
                    },
                }
            ]
        },
        "triggering_dci_slot_by_cell": {"cell-a": 12},
        "scheduled_cell_ids": ("cell-a",),
        "offset_combo_index": 0,
        "offset_combos": ((0,),),
        "offset_combo_cell_ids": ("cell-a",),
    }
    arguments.update(changes)
    with pytest.raises(error, match=message):
        nr_srs_multicell_aperiodic_resource_set_schedules(**arguments)


def test_srs_hopped_mapper_uses_each_symbols_frequency_allocation() -> None:
    allocations = tuple(
        nr_srs_fixed_frequency_allocation(
            9,
            transmission_comb=4,
            b_srs=3,
            b_hop=0,
            frequency_domain_position=3,
            n_srs=counter,
        )
        for counter in range(2)
    )
    sequences = np.stack(
        [
            np.full(allocations[0].bandwidth.sequence_length, 1.0 + 1.0j),
            np.full(allocations[0].bandwidth.sequence_length, 2.0 - 1.0j),
        ]
    )

    mapped = map_nr_srs_hopped_resource(
        sequences,
        frequency_allocations=allocations,
        slot_symbol_count=14,
        subcarrier_count=512,
        start_position_offset=1,
        srs_symbol_count=2,
    )

    assert isinstance(mapped, NRSRSHoppedResourceGrid)
    assert mapped.symbol_indices == (12, 13)
    assert mapped.n_srs_by_symbol == (0, 1)
    assert mapped.position_indices_by_symbol == ((0, 0, 1, 1), (0, 1, 1, 1))
    assert mapped.subcarrier_indices_by_symbol[0] == tuple(range(144, 192, 4))
    assert mapped.subcarrier_indices_by_symbol[1] == tuple(range(336, 384, 4))
    for row, symbol in enumerate(mapped.symbol_indices):
        np.testing.assert_array_equal(
            mapped.grid[symbol, list(mapped.subcarrier_indices_by_symbol[row])],
            sequences[row],
        )
    assert not mapped.grid.flags.writeable
    assert not mapped.mask.flags.writeable


@pytest.mark.parametrize("symbol_index", range(4))
def test_srs_aperiodic_transmission_counter_uses_repetition_factor(symbol_index) -> None:
    assert (
        nr_srs_transmission_counter(
            resource_type="aperiodic",
            srs_symbol_count=4,
            resource_symbol_index=symbol_index,
            repetition_factor=2,
        )
        == symbol_index // 2
    )


def test_srs_aperiodic_counter_accounts_for_eight_port_tdm() -> None:
    counters = tuple(
        nr_srs_transmission_counter(
            resource_type="aperiodic",
            srs_symbol_count=8,
            resource_symbol_index=index,
            repetition_factor=2,
            ports8_tdm=True,
        )
        for index in range(8)
    )

    assert counters == (0, 0, 0, 0, 1, 1, 1, 1)


def test_srs_periodic_counter_advances_by_occasion_and_returns_none_off_schedule() -> None:
    def resolve(slot: int, symbol: int) -> int | None:
        return nr_srs_transmission_counter(
            resource_type="periodic",
            srs_symbol_count=4,
            resource_symbol_index=symbol,
            repetition_factor=2,
            absolute_slot_index=slot,
            periodicity_slots=10,
            slot_offset=3,
        )

    assert tuple(resolve(13, symbol) for symbol in range(4)) == (2, 2, 3, 3)
    assert resolve(8, 0) is None


def test_srs_periodic_counter_rejects_offset_outside_period() -> None:
    with pytest.raises(ValueError, match="offset inside its period"):
        nr_srs_transmission_counter(
            resource_type="semi-persistent",
            srs_symbol_count=2,
            resource_symbol_index=0,
            absolute_slot_index=20,
            periodicity_slots=10,
            slot_offset=10,
        )


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        pytest.param(
            {"resource_type": "invalid"}, ValueError, "resource_type must be", id="resource-type"
        ),
        pytest.param(
            {"srs_symbol_count": True},
            TypeError,
            "srs_symbol_count must be an integer",
            id="symbols-bool",
        ),
        pytest.param(
            {"resource_symbol_index": False},
            TypeError,
            "resource_symbol_index must be an integer",
            id="symbol-index-bool",
        ),
        pytest.param(
            {"srs_symbol_count": 3},
            ValueError,
            "srs_symbol_count is unsupported",
            id="symbols-unsupported",
        ),
        pytest.param(
            {"resource_symbol_index": 2},
            ValueError,
            "inside the SRS resource",
            id="symbol-index-range",
        ),
        pytest.param({"ports8_tdm": 1}, TypeError, "ports8_tdm must be a boolean", id="tdm-type"),
        pytest.param(
            {"positioning_resource": 1},
            TypeError,
            "positioning_resource must be a boolean",
            id="positioning-type",
        ),
        pytest.param(
            {"srs_symbol_count": 1, "ports8_tdm": True},
            ValueError,
            "divisible by the port factor",
            id="tdm-symbol-divisibility",
        ),
        pytest.param(
            {"repetition_factor": True},
            TypeError,
            "repetition_factor must be an integer",
            id="repetition-bool",
        ),
        pytest.param(
            {"repetition_factor": 0},
            ValueError,
            "repetition_factor must divide",
            id="repetition-zero",
        ),
        pytest.param(
            {"repetition_factor": 3},
            ValueError,
            "repetition_factor must divide",
            id="repetition-nondivisor",
        ),
        pytest.param(
            {"absolute_slot_index": 0},
            ValueError,
            "does not use periodic slot parameters",
            id="aperiodic-slot-parameters",
        ),
        pytest.param(
            {"hyper_sfn_index": 0},
            ValueError,
            "does not use Hyper-SFN parameters",
            id="aperiodic-hyper-sfn",
        ),
        pytest.param(
            {
                "resource_type": "periodic",
                "absolute_slot_index": True,
                "periodicity_slots": 4,
                "slot_offset": 0,
            },
            TypeError,
            "absolute_slot_index must be an integer",
            id="periodic-slot-type",
        ),
        pytest.param(
            {
                "resource_type": "periodic",
                "absolute_slot_index": 0,
                "periodicity_slots": 4,
                "slot_offset": 0,
                "numerology": 0,
            },
            ValueError,
            "require hyper_sfn_index",
            id="hyper-sfn-missing-index",
        ),
    ],
)
def test_srs_transmission_counter_enforces_request_types_and_timing_contracts(
    changes, error, message
) -> None:
    arguments = {
        "resource_type": "aperiodic",
        "srs_symbol_count": 2,
        "resource_symbol_index": 0,
    }
    arguments.update(changes)

    with pytest.raises(error, match=message):
        nr_srs_transmission_counter(**arguments)


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"b_srs": 2, "b_hop": 1}, ValueError, "n_srs is required"),
        ({"b_srs": 2, "b_hop": 4}, ValueError, "b_hop"),
        ({"frequency_domain_position": -1}, ValueError, "frequency_domain_position"),
        ({"frequency_domain_shift_rb": True}, TypeError, "frequency_domain_shift_rb"),
        ({"comb_offset": 4}, ValueError, "comb_offset"),
    ],
)
def test_srs_fixed_frequency_allocation_rejects_out_of_contract_inputs(
    kwargs, error, message
) -> None:
    with pytest.raises(error, match=message):
        nr_srs_fixed_frequency_allocation(55, transmission_comb=4, **kwargs)


def test_srs_cyclic_shift_hopping_offset_is_frame_periodic_and_subset_bounded() -> None:
    common = {
        "transmission_comb": 2,
        "cyclic_shift_hopping_id": 913,
        "slot_index": 7,
        "slots_per_frame": 20,
        "slot_symbol_count": 14,
        "start_symbol_index": 9,
        "resource_symbol_index": 1,
    }
    first = nr_srs_cyclic_shift_hopping_offset(radio_frame_index=4, **common)
    assert nr_srs_cyclic_shift_hopping_offset(radio_frame_index=132, **common) == first
    assert 0 <= first < 8
    subset = (False, True, False, True, False, False, True, False)
    selected = nr_srs_cyclic_shift_hopping_offset(
        radio_frame_index=4, hopping_subset=subset, **common
    )
    assert selected in {1, 3, 6}


def test_srs_cyclic_shift_hopping_rejects_invalid_subset_and_symbol_coordinates() -> None:
    common = {
        "transmission_comb": 4,
        "cyclic_shift_hopping_id": 1,
        "radio_frame_index": 0,
        "slot_index": 0,
        "slots_per_frame": 10,
        "slot_symbol_count": 14,
        "start_symbol_index": 12,
        "resource_symbol_index": 1,
    }
    with pytest.raises(ValueError, match="between 2"):
        nr_srs_cyclic_shift_hopping_offset(**common, hopping_subset=(True,) + (False,) * 11)
    with pytest.raises(ValueError, match="cannot both"):
        nr_srs_cyclic_shift_hopping_offset(
            **common,
            hopping_subset=(True, True) + (False,) * 10,
            hopping_finer_granularity=True,
        )
    with pytest.raises(ValueError, match="inside the slot"):
        nr_srs_cyclic_shift_hopping_offset(**{**common, "resource_symbol_index": 2})


def test_srs_rrc_ports8tdm_rejects_release19_hopping_combinations() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        nr_srs_port_configuration_from_rrc(
            None,
            nrof_srs_ports_n8="ports8tdm",
            cyclic_shift_hopping_enabled=True,
        )
    with pytest.raises(ValueError, match="cannot be combined"):
        nr_srs_port_configuration_from_rrc(
            None,
            nrof_srs_ports_n8="ports8tdm",
            comb_offset_hopping_enabled=True,
        )
    with pytest.raises(ValueError, match="must be port1"):
        nr_srs_port_configuration_from_rrc("ports8")
    with pytest.raises(ValueError, match="ports8 or ports8tdm"):
        nr_srs_port_configuration_from_rrc(None, nrof_srs_ports_n8="ports16")


def test_srs_scheduled_port_sequence_and_eight_port_tdm_grid() -> None:
    ports = tuple(range(1000, 1008))
    sequences = nr_srs_scheduled_port_sequences_type1(
        72,
        antenna_ports=ports,
        base_cyclic_shift=0,
        srs_symbol_count=2,
        sequence_id=17,
        slot_index=2,
        start_symbol_index=12,
        hopping_mode="neither",
        transmission_comb=2,
        ports8_tdm=True,
    )
    assert sequences.shape == (8, 2, 72)
    result = map_nr_srs_multiport_resource(
        sequences,
        slot_symbol_count=14,
        subcarrier_count=144,
        start_position_offset=1,
        srs_symbol_count=2,
        comb_size=2,
        comb_offset=0,
        ports8_tdm=True,
    )
    assert result.active_port_symbols == nr_srs_port_symbol_activity(
        ports, srs_symbol_count=2, ports8_tdm=True
    )
    assert result.mask[:, 12, 0::2].tolist() == [
        [True] * 72,
        [True] * 72,
        [False] * 72,
        [False] * 72,
        [True] * 72,
        [True] * 72,
        [False] * 72,
        [False] * 72,
    ]
    assert result.mask[:, 13, 0::2].tolist() == [
        [False] * 72,
        [False] * 72,
        [True] * 72,
        [True] * 72,
        [False] * 72,
        [False] * 72,
        [True] * 72,
        [True] * 72,
    ]
    assert np.all(result.grid[~result.mask] == 0)
    assert not result.grid.flags.writeable


@pytest.mark.parametrize(
    ("shifts", "error", "message"),
    [
        ((), TypeError, "nonempty tuple"),
        ((0, 0), ValueError, "unique"),
        ((8,), ValueError, r"in \[0, 8\)"),
        ((True,), TypeError, "integers"),
    ],
)
def test_srs_multiport_sequence_rejects_invalid_cyclic_shift_schedule(
    shifts, error, message
) -> None:
    with pytest.raises(error, match=message):
        nr_srs_port_sequences_type1(
            72,
            cyclic_shift_indices=shifts,
            srs_symbol_count=1,
            sequence_id=0,
            slot_index=0,
            start_symbol_index=0,
            hopping_mode="neither",
            transmission_comb=2,
        )


@pytest.mark.parametrize(
    (
        "sequence_id",
        "slot_index",
        "start_symbol_index",
        "resource_symbol_index",
        "slot_symbol_count",
    ),
    [(0, 0, 0, 0, 14), (42, 11, 10, 1, 14), (65535, 127, 8, 3, 12)],
)
def test_srs_group_hopping_index_matches_scalar_gold_recurrence(
    sequence_id, slot_index, start_symbol_index, resource_symbol_index, slot_symbol_count
) -> None:
    offset = 8 * (slot_symbol_count * slot_index + start_symbol_index + resource_symbol_index)
    bits = _scalar_gold_bits(sequence_id, offset + 8)
    hopping_value = sum(bits[offset + bit] << bit for bit in range(8))
    assert (
        nr_srs_group_hopping_index(
            sequence_id,
            slot_index,
            start_symbol_index,
            resource_symbol_index,
            slot_symbol_count=slot_symbol_count,
        )
        == (hopping_value + sequence_id) % 30
    )


@pytest.mark.parametrize(
    (
        "sequence_id",
        "slot_index",
        "start_symbol_index",
        "resource_symbol_index",
        "slot_symbol_count",
        "error",
    ),
    [
        (-1, 0, 0, 0, 14, ValueError),
        (65536, 0, 0, 0, 14, ValueError),
        (0, -1, 0, 0, 14, ValueError),
        (0, 0, 14, 0, 14, ValueError),
        (0, 0, 13, 1, 14, ValueError),
        (0, 0, 0, 0, 13, ValueError),
        (True, 0, 0, 0, 14, TypeError),
    ],
)
def test_srs_group_hopping_index_rejects_invalid_positions(
    sequence_id,
    slot_index,
    start_symbol_index,
    resource_symbol_index,
    slot_symbol_count,
    error,
) -> None:
    with pytest.raises(error):
        nr_srs_group_hopping_index(
            sequence_id,
            slot_index,
            start_symbol_index,
            resource_symbol_index,
            slot_symbol_count=slot_symbol_count,
        )


@pytest.mark.parametrize(
    (
        "sequence_id",
        "slot_index",
        "start_symbol_index",
        "resource_symbol_index",
        "slot_symbol_count",
    ),
    [(0, 0, 0, 0, 14), (42, 3, 5, 7, 14), (65535, 9, 6, 4, 12)],
)
def test_srs_sequence_hopping_index_matches_scalar_gold_bit(
    sequence_id, slot_index, start_symbol_index, resource_symbol_index, slot_symbol_count
) -> None:
    bit_index = slot_symbol_count * slot_index + start_symbol_index + resource_symbol_index
    expected = _scalar_gold_bits(sequence_id, bit_index + 1)[bit_index]
    assert (
        nr_srs_sequence_hopping_index(
            sequence_id,
            slot_index,
            start_symbol_index,
            resource_symbol_index,
            72,
            slot_symbol_count=slot_symbol_count,
        )
        == expected
    )


@pytest.mark.parametrize("sequence_length", [6, 12, 18, 24, 30, 36, 71])
def test_srs_sequence_hopping_is_disabled_below_72(sequence_length) -> None:
    assert nr_srs_sequence_hopping_index(101, 7, 0, 2, sequence_length) == 0


def test_srs_hopped_sequence_resolves_sequence_hopping_base_sequence() -> None:
    sequence_id = 77
    slot_index = 3
    start_symbol_index = 8
    resource_symbol_index = 4
    base_sequence_index = nr_srs_sequence_hopping_index(
        sequence_id, slot_index, start_symbol_index, resource_symbol_index, 72
    )
    actual = nr_srs_hopped_sequence_type1(
        72,
        sequence_id=sequence_id,
        slot_index=slot_index,
        start_symbol_index=start_symbol_index,
        resource_symbol_index=resource_symbol_index,
        hopping_mode="sequenceHopping",
        transmission_comb=2,
    )
    expected = nr_srs_sequence_type1(
        72,
        group_index=sequence_id % 30,
        base_sequence_index=base_sequence_index,
        transmission_comb=2,
    )
    np.testing.assert_array_equal(actual, expected)


def test_srs_hopped_sequence_rejects_unknown_hopping_mode() -> None:
    with pytest.raises(ValueError, match="hopping_mode"):
        nr_srs_hopped_sequence_type1(
            72,
            sequence_id=0,
            slot_index=0,
            start_symbol_index=0,
            resource_symbol_index=0,
            hopping_mode="combined",
            transmission_comb=2,
        )


@pytest.mark.parametrize("finer_granularity", [False, True])
def test_srs_resource_sequence_rows_derive_cyclic_shift_hopping_per_symbol(
    finer_granularity: bool,
) -> None:
    arguments = {
        "transmission_comb": 2,
        "cyclic_shift_hopping_id": 913,
        "radio_frame_index": 4,
        "slot_index": 7,
        "slots_per_frame": 20,
        "slot_symbol_count": 14,
        "start_symbol_index": 9,
        "srs_symbol_count": 4,
        "hopping_finer_granularity": finer_granularity,
    }
    offsets = nr_srs_cyclic_shift_hopping_offsets(**arguments)
    expected_offsets = []
    for resource_symbol in range(arguments["srs_symbol_count"]):
        symbol_offset = (
            (
                (arguments["radio_frame_index"] % 128) * arguments["slots_per_frame"]
                + arguments["slot_index"]
            )
            * arguments["slot_symbol_count"]
            + arguments["start_symbol_index"]
            + resource_symbol
        )
        bits = _scalar_gold_bits(arguments["cyclic_shift_hopping_id"], 8 * symbol_offset + 8)[
            8 * symbol_offset :
        ]
        hop_word = sum(bit << bit_index for bit_index, bit in enumerate(bits))
        expected_offsets.append(hop_word % (16 if finer_granularity else 8))
    assert offsets == tuple(expected_offsets)

    rows = nr_srs_resource_sequences_type1(
        72,
        sequence_id=91,
        hopping_mode="groupHopping",
        cyclic_shift_index=3,
        cyclic_shift_hopping_id=arguments["cyclic_shift_hopping_id"],
        radio_frame_index=arguments["radio_frame_index"],
        slot_index=arguments["slot_index"],
        slots_per_frame=arguments["slots_per_frame"],
        start_symbol_index=arguments["start_symbol_index"],
        srs_symbol_count=arguments["srs_symbol_count"],
        slot_symbol_count=arguments["slot_symbol_count"],
        transmission_comb=arguments["transmission_comb"],
        hopping_finer_granularity=finer_granularity,
    )
    expected_rows = np.stack(
        [
            nr_srs_hopped_sequence_type1(
                72,
                sequence_id=91,
                slot_index=arguments["slot_index"],
                start_symbol_index=arguments["start_symbol_index"],
                resource_symbol_index=index,
                hopping_mode="groupHopping",
                transmission_comb=2,
                cyclic_shift_index=3,
                cyclic_shift_hopping_offset=offsets[index],
                hopping_finer_granularity=finer_granularity,
            )
            for index in range(arguments["srs_symbol_count"])
        ]
    )
    np.testing.assert_array_equal(rows, expected_rows)

    multiport = nr_srs_port_sequences_type1(
        72,
        cyclic_shift_indices=(3, 4),
        srs_symbol_count=arguments["srs_symbol_count"],
        sequence_id=91,
        slot_index=arguments["slot_index"],
        start_symbol_index=arguments["start_symbol_index"],
        hopping_mode="groupHopping",
        transmission_comb=2,
        cyclic_shift_hopping_id=arguments["cyclic_shift_hopping_id"],
        radio_frame_index=arguments["radio_frame_index"],
        slots_per_frame=arguments["slots_per_frame"],
        slot_symbol_count=arguments["slot_symbol_count"],
        hopping_finer_granularity=finer_granularity,
    )
    np.testing.assert_array_equal(multiport[0], rows)
    for index, offset in enumerate(offsets):
        expected_second_port = nr_srs_hopped_sequence_type1(
            72,
            sequence_id=91,
            slot_index=arguments["slot_index"],
            start_symbol_index=arguments["start_symbol_index"],
            resource_symbol_index=index,
            hopping_mode="groupHopping",
            transmission_comb=2,
            cyclic_shift_index=4,
            cyclic_shift_hopping_offset=offset,
            hopping_finer_granularity=finer_granularity,
        )
        np.testing.assert_array_equal(multiport[1, index], expected_second_port)


def test_srs_automatic_cyclic_shift_hopping_rejects_ambiguous_coordinates() -> None:
    common = {
        "sequence_length": 24,
        "srs_symbol_count": 2,
        "sequence_id": 12,
        "slot_index": 1,
        "start_symbol_index": 10,
        "hopping_mode": "neither",
        "transmission_comb": 2,
        "cyclic_shift_hopping_id": 50,
    }
    with pytest.raises(ValueError, match="radio_frame_index and slots_per_frame"):
        nr_srs_resource_sequences_type1(**common)
    with pytest.raises(ValueError, match="do not combine derived and explicit"):
        nr_srs_resource_sequences_type1(
            **common,
            radio_frame_index=0,
            slots_per_frame=20,
            cyclic_shift_hopping_offsets=(0, 1),
        )
    with pytest.raises(ValueError, match="exceed the configured granularity"):
        explicit_offsets = {
            key: value for key, value in common.items() if key != "cyclic_shift_hopping_id"
        }
        nr_srs_resource_sequences_type1(
            **explicit_offsets,
            cyclic_shift_hopping_offsets=(0, 8),
        )


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        ({"srs_symbol_count": 3}, ValueError, "unsupported SRS symbol count"),
        ({"srs_symbol_count": True}, TypeError, "srs_symbol_count must be an integer"),
        ({"slot_symbol_count": 13}, ValueError, "slot_symbol_count must be 12 or 14"),
        ({"cyclic_shift_hopping_offsets": [0, 1]}, TypeError, "must be a tuple"),
        ({"cyclic_shift_hopping_offsets": (0,)}, ValueError, "must match the SRS symbol count"),
        ({"cyclic_shift_hopping_offsets": (0, True)}, TypeError, "must contain integers"),
    ],
)
def test_srs_resource_sequences_reject_invalid_symbol_and_hopping_inputs(
    changes, error, message
) -> None:
    arguments = {
        "sequence_length": 24,
        "srs_symbol_count": 2,
        "sequence_id": 12,
        "slot_index": 1,
        "start_symbol_index": 10,
        "hopping_mode": "neither",
        "transmission_comb": 2,
    }
    arguments.update(changes)
    with pytest.raises(error, match=message):
        nr_srs_resource_sequences_type1(**arguments)


def test_srs_resource_sequences_require_identity_for_hopping_coordinates() -> None:
    with pytest.raises(ValueError, match="coordinates require cyclic_shift_hopping_id"):
        nr_srs_resource_sequences_type1(
            24,
            srs_symbol_count=2,
            sequence_id=12,
            slot_index=1,
            start_symbol_index=10,
            hopping_mode="neither",
            transmission_comb=2,
            radio_frame_index=0,
            slots_per_frame=20,
        )


def test_srs_resource_sequences_require_boolean_finer_granularity_flag() -> None:
    with pytest.raises(TypeError, match="hopping_finer_granularity must be a boolean"):
        nr_srs_resource_sequences_type1(
            24,
            srs_symbol_count=2,
            sequence_id=12,
            slot_index=1,
            start_symbol_index=10,
            hopping_mode="neither",
            transmission_comb=2,
            hopping_finer_granularity=1,
        )


def test_srs_scheduled_port_sequences_validate_duplicate_policy_type() -> None:
    with pytest.raises(TypeError, match="allow_duplicate_cyclic_shifts must be a boolean"):
        nr_srs_port_sequences_type1(
            24,
            cyclic_shift_indices=(0,),
            srs_symbol_count=1,
            sequence_id=12,
            slot_index=0,
            start_symbol_index=13,
            hopping_mode="neither",
            transmission_comb=2,
            allow_duplicate_cyclic_shifts=1,
        )


@pytest.mark.parametrize(
    ("slot", "resource_type", "error", "message"),
    [
        (True, "periodic", TypeError, "absolute_slot_index must be an integer"),
        (-1, "periodic", ValueError, "absolute_slot_index must be nonnegative"),
        (0, "aperiodic", ValueError, "requires periodic or semi-persistent"),
    ],
)
def test_periodic_srs_resource_schedule_rejects_invalid_slot_or_type(
    slot, resource_type, error, message
) -> None:
    with pytest.raises(error, match=message):
        nr_srs_periodic_resource_schedule(
            slot,
            c_srs=0,
            transmission_comb=2,
            b_srs=0,
            b_hop=0,
            srs_symbol_count=1,
            resource_type=resource_type,
            periodicity_slots=1,
            slot_offset=0,
        )


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        ({"absolute_slot_index": True}, TypeError, "absolute_slot_index must be an integer"),
        ({"absolute_slot_index": -1}, ValueError, "absolute_slot_index must be nonnegative"),
        ({"resources": "bad"}, TypeError, "resources must be an ordered sequence"),
        ({"resources": ()}, ValueError, "resources must contain at least one"),
        ({"resources": ("bad",)}, TypeError, "resources\\[0\\] must be a mapping"),
        ({"resources": ({"resource_id": True},)}, TypeError, "resource_id must be an integer"),
        ({"resources": ({"resource_id": -1},)}, ValueError, "unique nonnegative integers"),
        (
            {"resources": ({"resource_id": 1}, {"resource_id": 1})},
            ValueError,
            "unique nonnegative integers",
        ),
        ({"resource_set_id": True}, TypeError, "resource_set_id must be an integer"),
        ({"resource_set_id": -1}, ValueError, "resource_set_id must be nonnegative"),
        ({"resource_type": "aperiodic"}, ValueError, "must be periodic or semi-persistent"),
        ({"positioning_resource": 1}, TypeError, "positioning_resource must be a boolean"),
        ({"resources": ({"resource_id": 1, "extra": 0},)}, ValueError, "unsupported fields"),
        (
            {"resources": ({"resource_id": 1, "allocation": {"c_srs": 0}},)},
            ValueError,
            "allocation fields are invalid",
        ),
    ],
)
def test_periodic_srs_resource_set_rejects_invalid_decoded_fields(changes, error, message) -> None:
    arguments = {
        "absolute_slot_index": 0,
        "resource_set_id": 1,
        "resources": (
            {
                "resource_id": 1,
                "allocation": {
                    "c_srs": 0,
                    "transmission_comb": 2,
                    "b_srs": 0,
                    "b_hop": 0,
                    "srs_symbol_count": 1,
                },
            },
        ),
        "periodicity_slots": 1,
        "slot_offset": 0,
    }
    arguments.update(changes)
    with pytest.raises(error, match=message):
        nr_srs_periodic_resource_set_schedule(**arguments)


@pytest.mark.parametrize(
    ("allocation", "error", "message"),
    [
        ("bad", ValueError, "requires an allocation mapping"),
        ({"c_srs": 0}, ValueError, "allocation fields are invalid"),
        (
            {
                "c_srs": 0,
                "transmission_comb": 2,
                "b_srs": 0,
                "b_hop": 0,
                "srs_symbol_count": 1,
                "unsupported": True,
            },
            ValueError,
            "allocation fields are invalid",
        ),
        (
            {
                "c_srs": 0,
                "transmission_comb": 2,
                "b_srs": 0,
                "b_hop": 0,
                "srs_symbol_count": 1,
                "available_slot_offsets": "bad",
            },
            ValueError,
            "available_slot_offsets must contain one to four values",
        ),
    ],
)
def test_aperiodic_resource_set_rejects_invalid_selected_allocation(
    allocation, error, message
) -> None:
    resource_set = {
        "resource_set_id": 1,
        "resource_type": "aperiodic",
        "aperiodic_trigger": 1,
        "allocation": allocation,
    }
    with pytest.raises(error, match=message):
        nr_srs_aperiodic_resource_set_schedules(
            12,
            1,
            (resource_set,),
            dci_format="0_1",
        )


@pytest.mark.parametrize(
    ("allocation_offsets", "combo", "message"),
    [
        ([0, 1], None, "needs an SRS offset-combo entry"),
        ([0, 1], (3,), "outside the available offset list"),
    ],
)
def test_multicell_aperiodic_schedule_requires_valid_selected_offset(
    allocation_offsets, combo, message
) -> None:
    arguments = {
        "request_list_index": 0,
        "request_list": ((1,),),
        "serving_cell_ids": ("cell-a",),
        "resource_sets_by_cell": {
            "cell-a": [
                {
                    "resource_set_id": 1,
                    "resource_type": "aperiodic",
                    "aperiodic_trigger": 1,
                    "allocation": {
                        "c_srs": 0,
                        "transmission_comb": 2,
                        "b_srs": 0,
                        "b_hop": 0,
                        "srs_symbol_count": 1,
                        "available_slot_offsets": allocation_offsets,
                    },
                }
            ]
        },
        "triggering_dci_slot_by_cell": {"cell-a": 12},
        "scheduled_cell_ids": ("cell-a",),
    }
    if combo is not None:
        arguments.update(
            offset_combo_index=0,
            offset_combos=(combo,),
            offset_combo_cell_ids=("cell-a",),
        )

    with pytest.raises(ValueError, match=message):
        nr_srs_multicell_aperiodic_resource_set_schedules(**arguments)


def test_multicell_aperiodic_combo_entry_without_offset_list_keeps_base_offset() -> None:
    result = nr_srs_multicell_aperiodic_resource_set_schedules(
        0,
        ((1,),),
        ("cell-a",),
        {
            "cell-a": [
                {
                    "resource_set_id": 1,
                    "resource_type": "aperiodic",
                    "aperiodic_trigger": 1,
                    "allocation": {
                        "c_srs": 0,
                        "transmission_comb": 2,
                        "b_srs": 0,
                        "b_hop": 0,
                        "srs_symbol_count": 1,
                        "slot_offset": 2,
                    },
                }
            ]
        },
        {"cell-a": 20},
        scheduled_cell_ids=("cell-a",),
        offset_combo_index=0,
        offset_combos=((3,),),
        offset_combo_cell_ids=("cell-a",),
    )

    assert result[0][1][0].target_slot_index == 22


@pytest.mark.parametrize(
    ("available_offsets", "error", "message"),
    [
        ("bad", TypeError, "available_slot_offsets must be an ordered sequence"),
        ((), ValueError, "available_slot_offsets must not be empty"),
    ],
)
def test_multicell_aperiodic_schedule_validates_available_offset_list(
    available_offsets, error, message
) -> None:
    resource_sets = {
        "cell-a": [
            {
                "resource_set_id": 1,
                "resource_type": "aperiodic",
                "aperiodic_trigger": 1,
                "allocation": {
                    "c_srs": 0,
                    "transmission_comb": 2,
                    "b_srs": 0,
                    "b_hop": 0,
                    "srs_symbol_count": 1,
                    "available_slot_offsets": available_offsets,
                },
            }
        ]
    }

    with pytest.raises(error, match=message):
        nr_srs_multicell_aperiodic_resource_set_schedules(
            0,
            ((1,),),
            ("cell-a",),
            resource_sets,
            {"cell-a": 10},
            scheduled_cell_ids=("cell-a",),
        )


def test_multicell_aperiodic_schedule_uses_single_available_offset_without_combo() -> None:
    result = nr_srs_multicell_aperiodic_resource_set_schedules(
        0,
        ((1,),),
        ("cell-a",),
        {
            "cell-a": [
                {
                    "resource_set_id": 1,
                    "resource_type": "aperiodic",
                    "aperiodic_trigger": 1,
                    "allocation": {
                        "c_srs": 0,
                        "transmission_comb": 2,
                        "b_srs": 0,
                        "b_hop": 0,
                        "srs_symbol_count": 1,
                        "available_slot_offsets": (2,),
                    },
                }
            ]
        },
        {"cell-a": 10},
        scheduled_cell_ids=("cell-a",),
    )

    assert result[0][1][0].target_slot_index == 12
