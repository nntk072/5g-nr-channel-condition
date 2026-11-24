"""SRS request selection carried by DCI 0_3, with cell and BWP indicators."""

import pytest
from integrations.sionna.srs import (
    NRSRSBWPCellResourceSetSchedules,
    NRSRSBWPFrequencyBounds,
    NRSRSBWPSelection,
    NRSRSResourceSetAperiodicSchedule,
    nr_srs_aperiodic_resource_set_schedules,
    nr_srs_dci_0_3_multicell_aperiodic_resource_set_schedules,
    nr_srs_dci_0_3_multicell_aperiodic_resource_set_schedules_by_bwp,
    nr_srs_dci_0_3_multicell_aperiodic_resource_set_schedules_from_bits,
    nr_srs_dci_0_3_request_list_index_from_bits,
    nr_srs_dci_0_3_request_list_index_from_layout,
    nr_srs_dci_0_3_scheduled_cell_ids,
    nr_srs_dci_0_3_ul_bwp_selection,
    nr_srs_multicell_aperiodic_resource_set_schedules,
    nr_srs_multicell_request_resource_set_ids,
    nr_srs_triggered_resource_set_ids,
)


def test_dci_0_3_srs_request_selector_decodes_bounded_msb_first_field() -> None:
    payload = (1, 0, 1, 1, 0, 1)
    assert (
        nr_srs_dci_0_3_request_list_index_from_bits(
            payload, field_offset=1, request_list_entry_count=3
        )
        == 1
    )
    assert (
        nr_srs_dci_0_3_request_list_index_from_bits(
            payload, field_offset=2, request_list_entry_count=4
        )
        == 3
    )
    assert (
        nr_srs_dci_0_3_request_list_index_from_bits((), field_offset=0, request_list_entry_count=1)
        == 0
    )
    assert (
        nr_srs_dci_0_3_request_list_index_from_bits(
            payload, field_offset=0, request_list_entry_count=0
        )
        is None
    )


def test_dci_0_3_srs_selector_offset_comes_from_full_ordered_field_width_roster() -> None:
    field_names = (
        "identifier_for_dci_formats",
        "scheduled_cell_set_indicator",
        "scheduled_cells_indicator",
        "bandwidth_part_indicator",
        "frequency_domain_resource_assignment",
        "time_domain_resource_assignment",
        "frequency_hopping_flag",
        "modulation_and_coding_scheme",
        "new_data_indicator",
        "redundancy_version",
        "harq_process_number",
        "first_downlink_assignment_index",
        "second_downlink_assignment_index",
        "tpc_command_for_scheduled_pusch",
        "srs_resource_indicator",
        "precoding_information_and_number_of_layers",
        "antenna_ports",
        "srs_request",
        "srs_offset_indicator",
        "csi_request",
        "ptrs_dmrs_association",
        "beta_offset_indicator",
        "dmrs_sequence_initialization",
        "ul_sch_indicator",
        "channel_access_cpext_capc",
        "open_loop_power_control_parameter_set_indication",
        "priority_indicator",
        "minimum_applicable_scheduling_offset_indicator",
        "scell_dormancy_indication",
        "pdcch_monitoring_adaptation_indication",
        "measurement_gap_cancellation",
    )
    widths = dict.fromkeys(field_names, 0)
    widths.update(
        identifier_for_dci_formats=1,
        scheduled_cell_set_indicator=1,
        scheduled_cells_indicator=2,
        frequency_domain_resource_assignment=3,
        srs_resource_indicator=2,
        antenna_ports=4,
        srs_request=2,
        ul_sch_indicator=1,
    )
    expected_offset = sum(widths[name] for name in field_names[: field_names.index("srs_request")])
    payload = [0] * sum(widths.values())
    payload[expected_offset : expected_offset + 2] = [1, 0]

    assert (
        nr_srs_dci_0_3_request_list_index_from_layout(
            payload, field_widths_bits=widths, request_list_entry_count=3
        )
        == 2
    )


@pytest.mark.parametrize(
    ("field_widths_change", "entry_count", "payload_tail", "error"),
    [
        ({"unexpected": 0}, 2, (), "roster mismatch"),
        ({"srs_request": 1}, 4, (), "does not match"),
        ({}, 3, (1,), "nonzero trailing bits"),
    ],
)
def test_dci_0_3_layout_selector_rejects_malformed_widths_and_padding(
    field_widths_change, entry_count, payload_tail, error
) -> None:
    names = (
        "identifier_for_dci_formats",
        "scheduled_cell_set_indicator",
        "scheduled_cells_indicator",
        "bandwidth_part_indicator",
        "frequency_domain_resource_assignment",
        "time_domain_resource_assignment",
        "frequency_hopping_flag",
        "modulation_and_coding_scheme",
        "new_data_indicator",
        "redundancy_version",
        "harq_process_number",
        "first_downlink_assignment_index",
        "second_downlink_assignment_index",
        "tpc_command_for_scheduled_pusch",
        "srs_resource_indicator",
        "precoding_information_and_number_of_layers",
        "antenna_ports",
        "srs_request",
        "srs_offset_indicator",
        "csi_request",
        "ptrs_dmrs_association",
        "beta_offset_indicator",
        "dmrs_sequence_initialization",
        "ul_sch_indicator",
        "channel_access_cpext_capc",
        "open_loop_power_control_parameter_set_indication",
        "priority_indicator",
        "minimum_applicable_scheduling_offset_indicator",
        "scell_dormancy_indication",
        "pdcch_monitoring_adaptation_indication",
        "measurement_gap_cancellation",
    )
    widths = dict.fromkeys(names, 0)
    widths["srs_request"] = 1 if entry_count == 2 else 2
    widths.update(field_widths_change)
    payload = [0] * sum(widths.values()) + list(payload_tail)

    with pytest.raises((TypeError, ValueError), match=error):
        nr_srs_dci_0_3_request_list_index_from_layout(
            payload, field_widths_bits=widths, request_list_entry_count=entry_count
        )


@pytest.mark.parametrize(
    ("payload", "offset", "entry_count", "error"),
    [
        ((1,), 0, 3, "complete SRS request selector"),
        ((1, 1), 0, 3, "reserved codepoint"),
        ((0,), -1, 2, "nonnegative"),
        ((0,), 2, 1, "beyond the DCI payload"),
        ((0, 2), 0, 2, r"payload_bits\[1\]"),
        ((0,), 0, 17, "entry_count"),
    ],
)
def test_dci_0_3_srs_request_selector_rejects_invalid_boundaries(
    payload, offset, entry_count, error
) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        nr_srs_dci_0_3_request_list_index_from_bits(
            payload, field_offset=offset, request_list_entry_count=entry_count
        )


def test_dci_0_3_cell_set_and_combo_indicators_resolve_rrc_indexes() -> None:
    configs = (
        {
            "set_id": 3,
            "scheduled_cell_ids": ("cell-12", "cell-18", "cell-24"),
            "scheduled_cell_combos": ((0, 2), (1,), (0, 1, 2)),
        },
        {
            "set_id": 1,
            "scheduled_cell_ids": ("cell-3", "cell-9"),
            "scheduled_cell_combos": ((1,), (0,)),
        },
    )

    # The cell-set indicator indexes configurations in ascending set ID order,
    # not their input order; combo indexes refer to each set's ordered UL list.
    assert nr_srs_dci_0_3_scheduled_cell_ids(
        configs, scheduled_cell_set_indicator=0, scheduled_cells_indicator=1
    ) == ("cell-3",)
    assert nr_srs_dci_0_3_scheduled_cell_ids(
        configs, scheduled_cell_set_indicator=1, scheduled_cells_indicator=0
    ) == ("cell-12", "cell-24")


def test_dci_0_3_bwp_indicator_maps_ascending_ids_per_scheduled_cell() -> None:
    result = nr_srs_dci_0_3_ul_bwp_selection(
        ("cell-3", "cell-9"),
        {"cell-3": (0, 4, 2), "cell-9": (0, 4)},
        {"cell-3": 4, "cell-9": 0},
        bwp_indicator=1,
    )

    assert isinstance(result, NRSRSBWPSelection)
    assert result.indicator_bit_width == 2
    assert result.selected_bwp_id_by_cell == (("cell-3", 2), ("cell-9", 4))
    assert result.outcome_by_cell == (
        ("cell-3", "selected_by_indicator"),
        ("cell-9", "selected_by_indicator"),
    )


def test_dci_0_3_bwp_indicator_table_selects_id_1_through_4() -> None:
    result = nr_srs_dci_0_3_ul_bwp_selection(
        ("cell-3", "cell-9"),
        {"cell-3": (0, 1, 2, 3, 4), "cell-9": (0, 2, 4)},
        {"cell-3": 0, "cell-9": 2},
        bwp_indicator=3,
    )

    assert result.indicator_bit_width == 2
    assert result.selected_bwp_id_by_cell == (("cell-3", 4), ("cell-9", 4))


def test_dci_0_3_bwp_indicator_falls_back_per_cell_when_unavailable_or_unsupported() -> None:
    unavailable = nr_srs_dci_0_3_ul_bwp_selection(
        ("cell-3", "cell-9"),
        {"cell-3": (0, 1, 2), "cell-9": (0, 1, 2)},
        {"cell-3": 2, "cell-9": 1},
        bwp_indicator=3,
    )
    unsupported = nr_srs_dci_0_3_ul_bwp_selection(
        ("cell-3", "cell-9"),
        {"cell-3": (0, 1, 2), "cell-9": (0, 1, 2)},
        {"cell-3": 2, "cell-9": 1},
        bwp_indicator=1,
        supports_active_bwp_change=False,
    )

    assert unavailable.selected_bwp_id_by_cell == (("cell-3", 2), ("cell-9", 1))
    assert unavailable.outcome_by_cell == (
        ("cell-3", "codepoint_not_configured_for_cell"),
        ("cell-9", "codepoint_not_configured_for_cell"),
    )
    assert unsupported.selected_bwp_id_by_cell == (("cell-3", 2), ("cell-9", 1))
    assert {status for _, status in unsupported.outcome_by_cell} == {
        "active_bwp_change_unsupported"
    }


def test_dci_0_3_bwp_field_is_absent_when_only_initial_bwps_are_configured() -> None:
    result = nr_srs_dci_0_3_ul_bwp_selection(
        ("cell-3", "cell-9"),
        {"cell-3": (0,), "cell-9": (0,)},
        {"cell-3": 0, "cell-9": 0},
        bwp_indicator=None,
    )

    assert result.indicator_bit_width == 0
    assert result.selected_bwp_id_by_cell == (("cell-3", 0), ("cell-9", 0))


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        pytest.param(
            {"scheduled_cell_ids": "cell-3"}, TypeError, "ordered sequence", id="cells-type"
        ),
        pytest.param({"scheduled_cell_ids": ()}, ValueError, "nonempty cell IDs", id="cells-empty"),
        pytest.param(
            {"scheduled_cell_ids": ("cell-3", "")},
            ValueError,
            "nonempty cell IDs",
            id="cell-empty-id",
        ),
        pytest.param(
            {"scheduled_cell_ids": ("cell-3", "cell-3")},
            ValueError,
            "must be unique",
            id="cell-duplicate",
        ),
        pytest.param(
            {"configured_ul_bwp_ids_by_cell": ()},
            TypeError,
            "must be a mapping",
            id="configured-map-type",
        ),
        pytest.param(
            {"configured_ul_bwp_ids_by_cell": {"other": (0, 1)}},
            ValueError,
            "keys must exactly match",
            id="configured-keys",
        ),
        pytest.param(
            {"active_ul_bwp_id_by_cell": {"other": 0}},
            ValueError,
            "keys must exactly match",
            id="active-keys",
        ),
        pytest.param(
            {"supports_active_bwp_change": 1}, TypeError, "must be a boolean", id="supports-type"
        ),
        pytest.param(
            {"initial_bwp_id": True},
            TypeError,
            "initial_bwp_id must be an integer",
            id="initial-bool",
        ),
        pytest.param(
            {"initial_bwp_id": 5}, ValueError, "initial_bwp_id must be in", id="initial-range"
        ),
        pytest.param(
            {"configured_ul_bwp_ids_by_cell": {"cell-3": "01"}},
            TypeError,
            "ordered sequence",
            id="bwp-sequence",
        ),
        pytest.param(
            {"configured_ul_bwp_ids_by_cell": {"cell-3": ()}},
            ValueError,
            "must lie in",
            id="bwp-empty",
        ),
        pytest.param(
            {"configured_ul_bwp_ids_by_cell": {"cell-3": (True,)}},
            ValueError,
            "must lie in",
            id="bwp-bool",
        ),
        pytest.param(
            {"configured_ul_bwp_ids_by_cell": {"cell-3": (0, 0)}},
            ValueError,
            "must be unique",
            id="bwp-duplicate",
        ),
        pytest.param(
            {"configured_ul_bwp_ids_by_cell": {"cell-3": (1, 2)}},
            ValueError,
            "include initial BWP",
            id="initial-missing",
        ),
        pytest.param(
            {"active_ul_bwp_id_by_cell": {"cell-3": True}},
            ValueError,
            "active UL BWP",
            id="active-bool",
        ),
        pytest.param(
            {"bwp_indicator": None}, TypeError, "decoded integer codepoint", id="indicator-missing"
        ),
        pytest.param(
            {"bwp_indicator": True}, TypeError, "decoded integer codepoint", id="indicator-bool"
        ),
        pytest.param({"bwp_indicator": 4}, ValueError, "does not fit", id="indicator-range"),
    ],
)
def test_dci_0_3_bwp_selection_rejects_invalid_configurations(changes, error, message) -> None:
    arguments = {
        "scheduled_cell_ids": ("cell-3",),
        "configured_ul_bwp_ids_by_cell": {"cell-3": (0, 1)},
        "active_ul_bwp_id_by_cell": {"cell-3": 0},
        "bwp_indicator": 0,
    }
    arguments.update(changes)

    with pytest.raises(error, match=message):
        nr_srs_dci_0_3_ul_bwp_selection(**arguments)


def test_dci_0_3_mixed_single_and_multi_bwp_cells_use_per_cell_outcomes() -> None:
    result = nr_srs_dci_0_3_ul_bwp_selection(
        ("cell-a", "cell-b"),
        {"cell-a": (0,), "cell-b": (0, 1)},
        {"cell-a": 0, "cell-b": 0},
        bwp_indicator=0,
    )

    assert result.selected_bwp_id_by_cell == (("cell-a", 0), ("cell-b", 0))
    assert result.outcome_by_cell == (("cell-a", "single_bwp"), ("cell-b", "selected_by_indicator"))


def test_dci_0_3_bwp_selection_drives_resource_sets_and_offset_roster() -> None:
    def resource_set(
        resource_set_id: int,
        trigger: int,
        available_offsets: tuple[int, ...] | None = None,
    ) -> dict[str, object]:
        allocation: dict[str, object] = {
            "c_srs": 9,
            "transmission_comb": 4,
            "b_srs": 0,
            "b_hop": 0,
            "srs_symbol_count": 1,
            "slot_offset": 2,
            "frequency_domain_position": 0,
            "repetition_factor": 1,
        }
        if available_offsets is not None:
            allocation["available_slot_offsets"] = available_offsets
        return {
            "resource_set_id": resource_set_id,
            "resource_type": "aperiodic",
            "aperiodic_trigger": trigger,
            "allocation": allocation,
        }

    result = nr_srs_dci_0_3_multicell_aperiodic_resource_set_schedules_by_bwp(
        0,
        ((0, 1),),
        ("cell-3", "cell-9"),
        {
            "cell-3": {
                0: (resource_set(30, 1, (0, 4)),),
                1: (),
            },
            "cell-9": {
                0: (),
                2: (resource_set(92, 1, (1, 3)),),
            },
        },
        {"cell-9": 40},
        cell_set_configs=(
            {
                "set_id": 0,
                "scheduled_cell_ids": ("cell-3", "cell-9"),
                "scheduled_cell_combos": ((1,),),
            },
        ),
        configured_ul_bwp_ids_by_cell={"cell-3": (0, 1), "cell-9": (0, 2)},
        active_ul_bwp_id_by_cell={"cell-3": 0, "cell-9": 0},
        ul_bwp_frequency_bounds_by_cell_and_bwp={
            "cell-3": {
                0: NRSRSBWPFrequencyBounds(0, 4096),
                1: NRSRSBWPFrequencyBounds(0, 4096),
            },
            "cell-9": {
                0: NRSRSBWPFrequencyBounds(0, 4096),
                2: NRSRSBWPFrequencyBounds(0, 4096),
            },
        },
        bwp_indicator=1,
        offset_combo_index=0,
        offset_combos=((1, 0),),
    )

    assert len(result) == 1
    cell_schedule = result[0]
    assert isinstance(cell_schedule, NRSRSBWPCellResourceSetSchedules)
    assert (cell_schedule.cell_id, cell_schedule.ul_bwp_id) == ("cell-9", 2)
    assert cell_schedule.frequency_bounds == NRSRSBWPFrequencyBounds(0, 4096)
    assert cell_schedule.resource_set_schedules[0].resource_set_id == 92
    assert cell_schedule.resource_set_schedules[0].target_slot_index == 43
    with pytest.raises(ValueError, match="outside selected UL BWP"):
        NRSRSBWPCellResourceSetSchedules(
            cell_id=cell_schedule.cell_id,
            ul_bwp_id=cell_schedule.ul_bwp_id,
            frequency_bounds=NRSRSBWPFrequencyBounds(1, 4096),
            resource_set_schedules=cell_schedule.resource_set_schedules,
        )
    with pytest.raises(ValueError, match="outside selected UL BWP"):
        NRSRSBWPCellResourceSetSchedules(
            cell_id=cell_schedule.cell_id,
            ul_bwp_id=cell_schedule.ul_bwp_id,
            frequency_bounds=NRSRSBWPFrequencyBounds(0, 1),
            resource_set_schedules=cell_schedule.resource_set_schedules,
        )


def test_dci_0_3_cell_selection_uses_implicit_default_and_single_combo() -> None:
    configs = (
        {
            "set_id": 2,
            "scheduled_cell_ids": ("cell-4", "cell-8"),
            "scheduled_cell_combos": ((0, 1),),
        },
    )

    assert nr_srs_dci_0_3_scheduled_cell_ids(configs) == ("cell-4", "cell-8")
    assert nr_srs_dci_0_3_scheduled_cell_ids(configs, default_set_id=2) == ("cell-4", "cell-8")


@pytest.mark.parametrize(
    ("configs", "kwargs", "error", "message"),
    (
        (
            ({"set_id": 0, "scheduled_cell_ids": ("a", "b")},),
            {"scheduled_cells_indicator": 0},
            ValueError,
            "no combo list",
        ),
        (
            ({"set_id": 0, "scheduled_cell_ids": ("a", "b"), "scheduled_cell_combos": ((0, 1),)},),
            {"scheduled_cells_indicator": 0},
            ValueError,
            "single combo",
        ),
        (
            (
                {
                    "set_id": 0,
                    "scheduled_cell_ids": ("a", "b"),
                    "scheduled_cell_combos": ((0,), (1,)),
                },
            ),
            {"scheduled_cells_indicator": 2},
            ValueError,
            "no configured combo",
        ),
        (
            (
                {
                    "set_id": 0,
                    "scheduled_cell_ids": ("a", "b"),
                    "scheduled_cell_combos": ((1, 0), (0,)),
                },
            ),
            {"scheduled_cells_indicator": 0},
            ValueError,
            "ascending",
        ),
        (
            (
                {
                    "set_id": 0,
                    "scheduled_cell_ids": ("a", "b"),
                    "scheduled_cell_combos": ((0, 2), (1,)),
                },
            ),
            {"scheduled_cells_indicator": 0},
            ValueError,
            "outside scheduled_cell_ids",
        ),
        (
            (
                {"set_id": 0, "scheduled_cell_ids": ("a", "b")},
                {"set_id": 1, "scheduled_cell_ids": ("c", "d")},
            ),
            {},
            ValueError,
            "multiple cell sets",
        ),
    ),
)
def test_dci_0_3_cell_selection_rejects_invalid_configurations(
    configs: tuple[dict[str, object], ...],
    kwargs: dict[str, int],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        nr_srs_dci_0_3_scheduled_cell_ids(configs, **kwargs)


def test_dci_0_3_cell_selection_composes_with_aperiodic_srs_scheduler() -> None:
    resource_sets = {
        "cell-3": [
            {
                "resource_set_id": 31,
                "resource_type": "aperiodic",
                "aperiodic_trigger": 1,
                "allocation": {
                    "c_srs": 9,
                    "transmission_comb": 4,
                    "b_srs": 0,
                    "b_hop": 0,
                    "srs_symbol_count": 1,
                    "slot_offset": 1,
                    "frequency_domain_position": 0,
                    "repetition_factor": 1,
                },
            }
        ],
        "cell-9": [
            {
                "resource_set_id": 91,
                "resource_type": "aperiodic",
                "aperiodic_trigger": 1,
                "allocation": {
                    "c_srs": 9,
                    "transmission_comb": 4,
                    "b_srs": 0,
                    "b_hop": 0,
                    "srs_symbol_count": 1,
                    "slot_offset": 2,
                    "frequency_domain_position": 0,
                    "repetition_factor": 1,
                },
            }
        ],
    }
    result = nr_srs_dci_0_3_multicell_aperiodic_resource_set_schedules(
        0,
        ((1, 1),),
        ("cell-3", "cell-9"),
        resource_sets,
        {"cell-9": 20},
        cell_set_configs=(
            {
                "set_id": 0,
                "scheduled_cell_ids": ("cell-3", "cell-9"),
                "scheduled_cell_combos": ((1,),),
            },
        ),
    )

    assert tuple(cell for cell, _ in result) == ("cell-9",)
    assert result[0][1][0].resource_set_id == 91
    assert result[0][1][0].target_slot_index == 22


def test_dci_0_3_selector_bits_compose_directly_with_multicell_srs_scheduler() -> None:
    resource_sets = {
        "cell-9": [
            {
                "resource_set_id": 91,
                "resource_type": "aperiodic",
                "aperiodic_trigger": 1,
                "allocation": {
                    "c_srs": 9,
                    "transmission_comb": 4,
                    "b_srs": 0,
                    "b_hop": 0,
                    "srs_symbol_count": 1,
                    "slot_offset": 2,
                    "frequency_domain_position": 0,
                    "repetition_factor": 1,
                },
            }
        ],
        "cell-10": [],
    }
    result = nr_srs_dci_0_3_multicell_aperiodic_resource_set_schedules_from_bits(
        (1,),
        field_offset=0,
        request_list=((0, 0), (1, 0)),
        serving_cell_ids=("cell-9", "cell-10"),
        resource_sets_by_cell=resource_sets,
        triggering_dci_slot_by_cell={"cell-9": 20},
        cell_set_configs=(
            {
                "set_id": 0,
                "scheduled_cell_ids": ("cell-9", "cell-10"),
                "scheduled_cell_combos": ((0,),),
            },
        ),
    )

    assert result is not None
    assert tuple(cell for cell, _ in result) == ("cell-9",)
    assert result[0][1][0].resource_set_id == 91
    assert result[0][1][0].target_slot_index == 22


def test_dci_0_3_selector_schedule_returns_none_without_request_list() -> None:
    assert (
        nr_srs_dci_0_3_multicell_aperiodic_resource_set_schedules_from_bits(
            (),
            field_offset=0,
            request_list=(),
            serving_cell_ids=(),
            resource_sets_by_cell={},
            triggering_dci_slot_by_cell={},
            cell_set_configs=(),
        )
        is None
    )


def test_dci_0_3_request_combo_is_scoped_to_selected_cell_set() -> None:
    resource_sets = {
        cell: [
            {
                "resource_set_id": resource_set_id,
                "resource_type": "aperiodic",
                "aperiodic_trigger": 1,
                "allocation": {
                    "c_srs": 9,
                    "transmission_comb": 4,
                    "b_srs": 0,
                    "b_hop": 0,
                    "srs_symbol_count": 1,
                    "slot_offset": 1,
                    "frequency_domain_position": 0,
                    "repetition_factor": 1,
                },
            }
        ]
        for cell, resource_set_id in (
            ("cell-3", 31),
            ("cell-9", 91),
            ("cell-12", 121),
            ("cell-15", 151),
        )
    }
    result = nr_srs_dci_0_3_multicell_aperiodic_resource_set_schedules(
        0,
        ((1, 1),),  # one request codepoint for each cell in selected set 1
        ("cell-3", "cell-9", "cell-12", "cell-15"),
        resource_sets,
        {"cell-15": 40},
        cell_set_configs=(
            {
                "set_id": 0,
                "scheduled_cell_ids": ("cell-3", "cell-9"),
                "scheduled_cell_combos": ((0, 1),),
            },
            {
                "set_id": 2,
                "scheduled_cell_ids": ("cell-12", "cell-15"),
                "scheduled_cell_combos": ((1,),),
            },
        ),
        scheduled_cell_set_indicator=1,
    )

    assert tuple(cell for cell, _ in result) == ("cell-15",)
    assert result[0][1][0].resource_set_id == 151
    assert result[0][1][0].target_slot_index == 41


def test_srs_request_selects_all_configured_matching_aperiodic_sets() -> None:
    resource_sets = [
        {
            "resource_set_id": 4,
            "resource_type": "aperiodic",
            "aperiodic_trigger": 1,
            "aperiodic_trigger_list": [3],
        },
        {
            "resource_set_id": 8,
            "resource_type": "periodic",
            "aperiodic_trigger": 2,
        },
        {
            "resource_set_id": 12,
            "resource_type": "aperiodic",
            "aperiodic_trigger": 3,
            "aperiodic_trigger_list": [2],
        },
    ]

    assert nr_srs_triggered_resource_set_ids(2, resource_sets, dci_format="0_1") == (12,)
    assert nr_srs_triggered_resource_set_ids(3, resource_sets, dci_format="0_1") == (4, 12)
    assert nr_srs_triggered_resource_set_ids(0, resource_sets, dci_format="0_1") == ()


@pytest.mark.parametrize("dci_format", ["0_2", "1_2"])
def test_srs_request_uses_one_bit_for_dci_format_0_2_and_1_2(dci_format: str) -> None:
    resource_sets = [
        {
            "resource_set_id": 4,
            "resource_type": "aperiodic",
            "aperiodic_trigger": 1,
        },
        {
            "resource_set_id": 8,
            "resource_type": "aperiodic",
            "aperiodic_trigger": 2,
        },
    ]

    assert (
        nr_srs_triggered_resource_set_ids(
            0, resource_sets, dci_format=dci_format, srs_request_mode=1
        )
        == ()
    )
    assert nr_srs_triggered_resource_set_ids(
        1, resource_sets, dci_format=dci_format, srs_request_mode=1
    ) == (4,)
    assert nr_srs_triggered_resource_set_ids(
        2, resource_sets, dci_format=dci_format, srs_request_mode=2
    ) == (8,)
    with pytest.raises(ValueError, match="1-bit.*for DCI format"):
        nr_srs_triggered_resource_set_ids(
            2, resource_sets, dci_format=dci_format, srs_request_mode=1
        )


def test_srs_request_requires_dci_0_2_1_2_configuration_mode() -> None:
    with pytest.raises(ValueError, match="requires configured srs_request_mode"):
        nr_srs_triggered_resource_set_ids(1, [], dci_format="0_2")


def test_srs_request_type_two_and_supplementary_uplink_decode_carrier() -> None:
    resource_sets = [
        {
            "resource_set_id": 4,
            "resource_type": "aperiodic",
            "aperiodic_trigger": 1,
            "ul_carrier": "non_sul",
        },
        {
            "resource_set_id": 8,
            "resource_type": "aperiodic",
            "aperiodic_trigger": 2,
            "ul_carrier": "sul",
        },
    ]

    assert nr_srs_triggered_resource_set_ids(
        1,
        resource_sets,
        dci_format="1_2",
        srs_request_mode=2,
        supplementary_uplink=True,
    ) == (4,)
    assert nr_srs_triggered_resource_set_ids(
        6,
        resource_sets,
        dci_format="1_2",
        srs_request_mode=2,
        supplementary_uplink=True,
    ) == (8,)


def test_srs_multicell_request_list_selects_per_cell_sets_in_serving_order() -> None:
    result = nr_srs_multicell_request_resource_set_ids(
        1,
        ((0, 0), (1, 2)),
        ("cell-3", "cell-9"),
        {
            "cell-3": [
                {"resource_set_id": 31, "resource_type": "aperiodic", "aperiodic_trigger": 1}
            ],
            "cell-9": [
                {"resource_set_id": 91, "resource_type": "aperiodic", "aperiodic_trigger": 2}
            ],
        },
    )

    assert result == (("cell-3", (31,)), ("cell-9", (91,)))


def test_srs_multicell_request_list_applies_sul_selector_per_cell() -> None:
    result = nr_srs_multicell_request_resource_set_ids(
        0,
        ((1, 6),),
        ("cell-3", "cell-9"),
        {
            "cell-3": [
                {
                    "resource_set_id": 31,
                    "resource_type": "aperiodic",
                    "aperiodic_trigger": 1,
                    "ul_carrier": "non_sul",
                },
                {
                    "resource_set_id": 32,
                    "resource_type": "aperiodic",
                    "aperiodic_trigger": 1,
                    "ul_carrier": "sul",
                },
            ],
            "cell-9": [
                {
                    "resource_set_id": 91,
                    "resource_type": "aperiodic",
                    "aperiodic_trigger": 2,
                    "ul_carrier": "sul",
                }
            ],
        },
        supplementary_uplink_cells=("cell-9",),
    )

    assert result == (("cell-3", (31,)), ("cell-9", (91,)))


def test_srs_multicell_request_and_offset_combo_compose_per_cell_schedules() -> None:
    resource_sets = {
        "cell-3": [
            {
                "resource_set_id": 31,
                "resource_type": "aperiodic",
                "aperiodic_trigger": 1,
                "allocation": {
                    "c_srs": 9,
                    "transmission_comb": 4,
                    "b_srs": 3,
                    "b_hop": 0,
                    "srs_symbol_count": 2,
                    "slot_offset": 2,
                    "available_slot_offsets": [1, 4],
                    "frequency_domain_position": 3,
                    "repetition_factor": 1,
                },
            }
        ],
        "cell-9": [
            {
                "resource_set_id": 91,
                "resource_type": "aperiodic",
                "aperiodic_trigger": 1,
                "allocation": {
                    "c_srs": 9,
                    "transmission_comb": 4,
                    "b_srs": 3,
                    "b_hop": 0,
                    "srs_symbol_count": 2,
                    "slot_offset": 1,
                    "available_slot_offsets": [0, 3],
                    "frequency_domain_position": 4,
                    "repetition_factor": 1,
                },
            }
        ],
        "cell-12": [
            {
                "resource_set_id": 12,
                "resource_type": "aperiodic",
                "aperiodic_trigger": 1,
                "allocation": {
                    "available_slot_offsets": [0, 1],
                },
            }
        ],
    }

    result = nr_srs_multicell_aperiodic_resource_set_schedules(
        0,
        ((1, 1, 0),),
        ("cell-3", "cell-9", "cell-12"),
        resource_sets,
        {"cell-3": 20, "cell-9": 200},
        scheduled_cell_ids=("cell-3", "cell-9"),
        offset_combo_index=1,
        offset_combos=((0, 1, 2), (1, 0, 3)),
        offset_combo_cell_ids=("cell-3", "cell-9", "cell-12"),
    )

    assert tuple(
        (cell, tuple(item.resource_set_id for item in schedules)) for cell, schedules in result
    ) == (("cell-3", (31,)), ("cell-9", (91,)))
    assert tuple(schedules[0].target_slot_index for _, schedules in result) == (26, 201)
    assert tuple(
        tuple(item.n_srs for item in schedules[0].allocations) for _, schedules in result
    ) == ((0, 1), (0, 1))


@pytest.mark.parametrize(
    ("request_index", "request_list", "cell_ids", "sets_by_cell", "message"),
    [
        (1, ((0,),), ("c1",), {"c1": []}, "does not select"),
        (0, ((0, 1),), ("c1",), {"c1": []}, "one codepoint per cell"),
        (0, ((0,),), ("c1",), {"c2": []}, "keys must exactly match"),
    ],
)
def test_srs_multicell_request_list_rejects_invalid_decoded_rosters(
    request_index, request_list, cell_ids, sets_by_cell, message
) -> None:
    with pytest.raises(ValueError, match=message):
        nr_srs_multicell_request_resource_set_ids(
            request_index, request_list, cell_ids, sets_by_cell
        )


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"dci_format": "0_3"}, ValueError, "unsupported DCI format"),
        ({"dci_format": "0_1", "srs_tpc_pdcch_group": "typeA"}, ValueError, "typeB"),
    ],
)
def test_srs_request_rejects_unsupported_dci_context(kwargs, error, message) -> None:
    with pytest.raises(error, match=message):
        nr_srs_triggered_resource_set_ids(1, [], **kwargs)


def test_srs_request_rejects_duplicate_resource_set_ids() -> None:
    with pytest.raises(ValueError, match="unique nonnegative"):
        nr_srs_triggered_resource_set_ids(
            1,
            [
                {"resource_set_id": 2, "resource_type": "aperiodic", "aperiodic_trigger": 1},
                {"resource_set_id": 2, "resource_type": "aperiodic", "aperiodic_trigger": 2},
            ],
            dci_format="1_1",
        )


@pytest.mark.parametrize(
    ("codepoint", "resource_sets", "kwargs", "error", "message"),
    [
        pytest.param(
            1, [], {"supplementary_uplink": 1}, TypeError, "supplementary_uplink", id="sul-flag"
        ),
        pytest.param(
            1,
            [],
            {"dci_format": "0_2"},
            ValueError,
            "requires configured srs_request_mode",
            id="mode-missing",
        ),
        pytest.param(
            1,
            [],
            {"dci_format": "0_2", "srs_request_mode": True},
            ValueError,
            "requires configured srs_request_mode",
            id="mode-bool",
        ),
        pytest.param(
            1,
            [],
            {"dci_format": "0_2", "srs_request_mode": 3},
            ValueError,
            "srs_request_mode must be 1 or 2",
            id="mode-out-of-range",
        ),
        pytest.param(
            1,
            [],
            {"srs_request_mode": 1},
            ValueError,
            "applies only to DCI 0_2/1_2",
            id="mode-on-0_1",
        ),
        pytest.param(True, [], {}, TypeError, "codepoint must be an integer", id="codepoint-bool"),
        pytest.param(4, [], {}, ValueError, "2-bit SRS request field", id="codepoint-out-of-range"),
        pytest.param(
            1, "sets", {}, TypeError, "resource_sets must be a sequence", id="sets-string"
        ),
        pytest.param(
            1, [None], {}, TypeError, r"resource_sets\[0\] must be a mapping", id="set-mapping"
        ),
        pytest.param(
            1,
            [{"resource_set_id": True}],
            {},
            TypeError,
            "resource_set_id must be an integer",
            id="set-id-bool",
        ),
        pytest.param(
            1,
            [{"resource_set_id": 1, "resource_type": "other"}],
            {},
            ValueError,
            "resource_type is unsupported",
            id="resource-type",
        ),
        pytest.param(
            1,
            [{"resource_set_id": 1, "resource_type": "aperiodic", "aperiodic_trigger": 1}],
            {"supplementary_uplink": True},
            ValueError,
            "ul_carrier must identify",
            id="sul-carrier-required",
        ),
        pytest.param(
            1,
            [
                {
                    "resource_set_id": 1,
                    "resource_type": "aperiodic",
                    "aperiodic_trigger": 1,
                    "ul_carrier": "other",
                }
            ],
            {},
            ValueError,
            "ul_carrier is invalid",
            id="carrier-invalid",
        ),
        pytest.param(
            1,
            [{"resource_set_id": 1, "resource_type": "aperiodic", "aperiodic_trigger": True}],
            {},
            TypeError,
            "aperiodic_trigger must be an integer",
            id="trigger-bool",
        ),
        pytest.param(
            1,
            [{"resource_set_id": 1, "resource_type": "aperiodic", "aperiodic_trigger": 4}],
            {},
            ValueError,
            "aperiodic_trigger must be in",
            id="trigger-range",
        ),
        pytest.param(
            1,
            [
                {
                    "resource_set_id": 1,
                    "resource_type": "aperiodic",
                    "aperiodic_trigger": 1,
                    "aperiodic_trigger_list": "2",
                }
            ],
            {},
            TypeError,
            "trigger_list must be a sequence",
            id="trigger-list-string",
        ),
        pytest.param(
            1,
            [
                {
                    "resource_set_id": 1,
                    "resource_type": "aperiodic",
                    "aperiodic_trigger": 1,
                    "aperiodic_trigger_list": [True],
                }
            ],
            {},
            TypeError,
            "trigger-list entries must be integers",
            id="trigger-list-bool",
        ),
        pytest.param(
            1,
            [
                {
                    "resource_set_id": 1,
                    "resource_type": "aperiodic",
                    "aperiodic_trigger": 1,
                    "aperiodic_trigger_list": [0],
                }
            ],
            {},
            ValueError,
            "trigger-list entries must be in",
            id="trigger-list-range",
        ),
    ],
)
def test_srs_request_rejects_invalid_codepoints_and_resource_set_contracts(
    codepoint, resource_sets, kwargs, error, message
) -> None:
    with pytest.raises(error, match=message):
        options = {"dci_format": "0_1", **kwargs}
        nr_srs_triggered_resource_set_ids(codepoint, resource_sets, **options)


def test_srs_request_composes_selection_and_resource_set_schedules() -> None:
    result = nr_srs_aperiodic_resource_set_schedules(
        50,
        1,
        [
            {
                "resource_set_id": 4,
                "resource_type": "aperiodic",
                "aperiodic_trigger": 1,
                "allocation": {
                    "c_srs": 9,
                    "transmission_comb": 4,
                    "b_srs": 3,
                    "b_hop": 0,
                    "srs_symbol_count": 2,
                    "slot_offset": 2,
                    "available_slot_offsets": [1, 3],
                    "selected_available_offset_index": 0,
                    "frequency_domain_position": 3,
                    "repetition_factor": 1,
                },
            },
            {
                "resource_set_id": 7,
                "resource_type": "aperiodic",
                "aperiodic_trigger": 2,
            },
        ],
        dci_format="0_1",
    )

    assert len(result) == 1
    assert isinstance(result[0], NRSRSResourceSetAperiodicSchedule)
    assert result[0].resource_set_id == 4
    assert result[0].target_slot_index == 53
    assert tuple(item.n_srs for item in result[0].allocations) == (0, 1)


def test_srs_request_requires_allocation_only_for_triggered_resource_sets() -> None:
    resource_sets = [{"resource_set_id": 4, "resource_type": "aperiodic", "aperiodic_trigger": 1}]
    assert nr_srs_aperiodic_resource_set_schedules(50, 0, resource_sets, dci_format="0_1") == ()
    with pytest.raises(ValueError, match="requires an allocation mapping"):
        nr_srs_aperiodic_resource_set_schedules(50, 1, resource_sets, dci_format="0_1")
