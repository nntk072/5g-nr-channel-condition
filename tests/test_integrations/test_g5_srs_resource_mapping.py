"""Regression definitions for bounded TS 38.211 SRS resource mapping."""

import json
from pathlib import Path

import numpy as np
import pytest
from integrations.sionna.srs import (
    NRSRSAperiodicResourceSchedule,
    NRSRSBandwidthConfiguration,
    NRSRSBWPCellPeriodicResourceSetSchedules,
    NRSRSBWPCellResourceSetSchedules,
    NRSRSBWPFrequencyBounds,
    NRSRSBWPPeriodicResourceSetSchedule,
    NRSRSBWPSelection,
    NRSRSFixedFrequencyAllocation,
    NRSRSHoppedMultiportResourceGrid,
    NRSRSHoppedResourceGrid,
    NRSRSMultiportResourceGrid,
    NRSRSPeriodicResourceSetSchedule,
    NRSRSPortConfiguration,
    NRSRSPositioningMultiportResourceGrid,
    NRSRSPositioningResourceGrid,
    NRSRSResourceGrid,
    NRSRSResourceSetAperiodicSchedule,
    NRSRSScheduledResourceOccurrence,
    NRSRSTransmissionCombConfiguration,
    audit_nr_srs_intercell_re_overlap,
    map_nr_srs_hopped_multiport_resource,
    map_nr_srs_hopped_resource,
    map_nr_srs_multiport_resource,
    map_nr_srs_positioning_multiport_resource,
    map_nr_srs_positioning_resource,
    map_nr_srs_resource,
    nr_dci_0_3_field_values_from_layout,
    nr_low_papr_sequence_type1,
    nr_srs_aperiodic_resource_schedule,
    nr_srs_aperiodic_resource_set_schedules,
    nr_srs_aperiodic_target_slot_index,
    nr_srs_cyclic_shift_hopping_offset,
    nr_srs_cyclic_shift_hopping_offsets,
    nr_srs_dci_0_3_field_widths_from_configuration,
    nr_srs_dci_0_3_frequency_assignment_widths,
    nr_srs_dci_0_3_multicell_aperiodic_resource_set_schedules,
    nr_srs_dci_0_3_multicell_aperiodic_resource_set_schedules_by_bwp,
    nr_srs_dci_0_3_multicell_aperiodic_resource_set_schedules_from_bits,
    nr_srs_dci_0_3_request_list_index_from_bits,
    nr_srs_dci_0_3_request_list_index_from_layout,
    nr_srs_dci_0_3_scheduled_cell_ids,
    nr_srs_dci_0_3_ul_bwp_selection,
    nr_srs_fixed_frequency_allocation,
    nr_srs_frequency_position_indices,
    nr_srs_group_hopping_index,
    nr_srs_hopped_sequence_type1,
    nr_srs_multicell_aperiodic_resource_set_schedules,
    nr_srs_multicell_periodic_resource_set_schedules_by_bwp,
    nr_srs_multicell_request_resource_set_ids,
    nr_srs_multiport_ls_channel_estimate,
    nr_srs_periodic_resource_schedule,
    nr_srs_periodic_resource_set_schedule,
    nr_srs_periodic_resource_set_schedule_by_bwp,
    nr_srs_periodicity_and_offset_slots,
    nr_srs_port_configuration_from_rrc,
    nr_srs_port_cyclic_shift_indices,
    nr_srs_port_sequences_type1,
    nr_srs_port_symbol_activity,
    nr_srs_positioning_symbol_offset,
    nr_srs_resource_frequency_allocations,
    nr_srs_resource_sequences_type1,
    nr_srs_scheduled_port_sequences_type1,
    nr_srs_sequence_hopping_index,
    nr_srs_sequence_type1,
    nr_srs_single_port_ls_channel_estimate,
    nr_srs_transmission_comb_from_rrc,
    nr_srs_transmission_counter,
    nr_srs_triggered_resource_set_ids,
    nr_srs_type1_bandwidth_configuration,
    srs_cyclic_shift_count,
)


def test_dci_0_3_srs_request_selector_decodes_bounded_msb_first_field() -> None:
    payload = (1, 0, 1, 1, 0, 1)
    assert nr_srs_dci_0_3_request_list_index_from_bits(
        payload, field_offset=1, request_list_entry_count=3
    ) == 1
    assert nr_srs_dci_0_3_request_list_index_from_bits(
        payload, field_offset=2, request_list_entry_count=4
    ) == 3
    assert nr_srs_dci_0_3_request_list_index_from_bits(
        (), field_offset=0, request_list_entry_count=1
    ) == 0
    assert nr_srs_dci_0_3_request_list_index_from_bits(
        payload, field_offset=0, request_list_entry_count=0
    ) is None


def test_dci_0_3_layout_parser_decodes_complete_field_roster_msb_first() -> None:
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
        identifier_for_dci_formats=2,
        scheduled_cells_indicator=2,
        modulation_and_coding_scheme=3,
        srs_request=2,
        ul_sch_indicator=1,
    )
    field_values = {
        "identifier_for_dci_formats": 2,
        "scheduled_cells_indicator": 1,
        "modulation_and_coding_scheme": 5,
        "srs_request": 3,
        "ul_sch_indicator": 1,
    }
    bits = [
        (field_values.get(name, 0) >> shift) & 1
        for name in field_names
        for shift in range(widths[name] - 1, -1, -1)
    ]

    decoded = nr_dci_0_3_field_values_from_layout(
        bits + [False, False], field_widths_bits=widths
    )

    assert decoded["identifier_for_dci_formats"] == 2
    assert decoded["scheduled_cells_indicator"] == 1
    assert decoded["modulation_and_coding_scheme"] == 5
    assert decoded["srs_request"] == 3
    assert decoded["ul_sch_indicator"] == 1
    assert decoded["bandwidth_part_indicator"] is None


@pytest.mark.parametrize(
    ("bits", "widths", "message"),
    [
        ([], None, "field_widths_bits must be a mapping"),
        ([0], {"wrong": 1}, "field-width roster mismatch"),
    ],
)
def test_dci_0_3_layout_parser_rejects_malformed_contracts(
    bits: list[int], widths: object, message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        nr_dci_0_3_field_values_from_layout(bits, field_widths_bits=widths)  # type: ignore[arg-type]


def test_dci_0_3_layout_parser_enforces_width_and_payload_bounds() -> None:
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
    widths["identifier_for_dci_formats"] = 1

    with pytest.raises(ValueError, match="shorter than the declared"):
        nr_dci_0_3_field_values_from_layout([], field_widths_bits=widths)
    with pytest.raises(ValueError, match="nonzero trailing bits"):
        nr_dci_0_3_field_values_from_layout([0, 1], field_widths_bits=widths)
    with pytest.raises(ValueError, match=r"payload_bits\[0\]"):
        nr_dci_0_3_field_values_from_layout([2], field_widths_bits=widths)

    widths["identifier_for_dci_formats"] = 4097
    with pytest.raises(ValueError, match=r"width for identifier_for_dci_formats.*\[0, 512\]"):
        nr_dci_0_3_field_values_from_layout([], field_widths_bits=widths)


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
    expected_offset = sum(
        widths[name] for name in field_names[: field_names.index("srs_request")]
    )
    payload = [0] * sum(widths.values())
    payload[expected_offset : expected_offset + 2] = [1, 0]

    assert nr_srs_dci_0_3_request_list_index_from_layout(
        payload, field_widths_bits=widths, request_list_entry_count=3
    ) == 2


def test_dci_0_3_frequency_assignment_widths_follow_active_bwp_configuration() -> None:
    cells = ("type0", "type1", "switch", "hopping", "interlace")
    configured = {cell: (0,) for cell in cells}
    active = {cell: 0 for cell in cells}
    configs = {
        "type0": {
            0: {
                "size_prb": 37,
                "start_prb": 1,
                "resource_allocation": "type0",
                "rbg_size_config": "config1",
            }
        },
        "type1": {
            0: {
                "size_prb": 20,
                "start_prb": 1,
                "resource_allocation": "type1",
                "type1_granularity_prb": 4,
            }
        },
        "switch": {
            0: {
                "size_prb": 10,
                "start_prb": 0,
                "resource_allocation": "dynamicSwitch",
                "rbg_size_config": 1,
                "type1_granularity_prb": 2,
            }
        },
        "hopping": {
            0: {
                "size_prb": 20,
                "start_prb": 0,
                "resource_allocation": "type1",
                "type1_granularity_prb": 4,
                "frequency_hopping": True,
                "frequency_hopping_offset_count": 4,
            }
        },
        "interlace": {
            0: {
                "size_prb": 50,
                "start_prb": 0,
                "use_interlace": True,
                "subcarrier_spacing_khz": 30,
                "rb_set_count": 4,
            }
        },
    }

    widths = nr_srs_dci_0_3_frequency_assignment_widths(
        cells, configured, active, configs
    )

    assert widths == {
        "type0": 10,
        "type1": 5,
        "switch": 6,
        "hopping": 6,
        "interlace": 7,
    }


@pytest.mark.parametrize(
    ("configuration", "error"),
    [
        ({"size_prb": 10, "start_prb": 266, "resource_allocation": "type0"}, "extent"),
        ({"size_prb": 10, "start_prb": 0, "resource_allocation": "unsupported"}, "mode"),
    ],
)
def test_dci_0_3_frequency_assignment_rejects_invalid_active_bwp_configuration(
    configuration, error
) -> None:
    with pytest.raises(ValueError, match=error):
        nr_srs_dci_0_3_frequency_assignment_widths(
            ("cell",),
            {"cell": (0,)},
            {"cell": 0},
            {"cell": {0: configuration}},
        )


def test_dci_0_3_known_widths_derive_from_decoded_cell_bwp_and_srs_configuration() -> None:
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
    derived_names = {
        "identifier_for_dci_formats",
        "scheduled_cell_set_indicator",
        "scheduled_cells_indicator",
        "bandwidth_part_indicator",
        "frequency_domain_resource_assignment",
        "time_domain_resource_assignment",
        "frequency_hopping_flag",
        "dmrs_sequence_initialization",
        "ul_sch_indicator",
        "priority_indicator",
        "minimum_applicable_scheduling_offset_indicator",
        "beta_offset_indicator",
        "channel_access_cpext_capc",
        "tpc_command_for_scheduled_pusch",
        "harq_process_number",
        "modulation_and_coding_scheme",
        "new_data_indicator",
        "redundancy_version",
        "first_downlink_assignment_index",
        "second_downlink_assignment_index",
        "srs_resource_indicator",
        "precoding_information_and_number_of_layers",
        "antenna_ports",
        "srs_request",
        "srs_offset_indicator",
        "csi_request",
        "ptrs_dmrs_association",
        "open_loop_power_control_parameter_set_indication",
        "scell_dormancy_indication",
        "pdcch_monitoring_adaptation_indication",
        "measurement_gap_cancellation",
    }
    assert set(derived_names) == set(field_names)
    cell_set_configs = (
        {
            "set_id": 0,
            "scheduled_cell_ids": ("cell-1", "cell-2"),
            "scheduled_cell_combos": ((0, 1),),
        },
        {
            "set_id": 2,
            "scheduled_cell_ids": ("cell-3", "cell-4"),
            "scheduled_cell_combos": ((0,), (1,), (0, 1)),
        },
    )

    resolver_inputs = dict(
        cell_set_configs=cell_set_configs,
        selected_cell_set_id=2,
        configured_ul_bwp_ids_by_cell={"cell-3": (0, 1, 2), "cell-4": (0, 1, 2)},
        active_ul_bwp_id_by_cell={"cell-3": 0, "cell-4": 0},
        ul_bwp_frequency_allocation_by_cell={
            "cell-3": {
                0: {"size_prb": 10, "start_prb": 0, "resource_allocation": "type0"},
                1: {"size_prb": 25, "start_prb": 0, "resource_allocation": "type1"},
                2: {"size_prb": 50, "start_prb": 0, "resource_allocation": "dynamicSwitch"},
            },
            "cell-4": {
                0: {"size_prb": 10, "start_prb": 0, "resource_allocation": "type0"},
                1: {"size_prb": 25, "start_prb": 0, "resource_allocation": "type1"},
                2: {"size_prb": 50, "start_prb": 0, "resource_allocation": "dynamicSwitch"},
            },
        },
        request_list_entry_count=3,
        srs_offset_list=((0, 0), (1, 0), (2, 1), (3, 2), (4, 3)),
        report_trigger_size=3,
        tdra_field_index_list_entry_count=5,
        frequency_hopping_enabled_by_cell={"cell-3": True, "cell-4": False},
        transform_precoding_enabled_by_cell={"cell-3": True, "cell-4": False},
        beta_offsets_semi_static_by_cell={"cell-3": True, "cell-4": False},
        channel_access_config_entry_count_by_cell={"cell-3": 4, "cell-4": 0},
        harq_process_number_size_bits_by_cell={"cell-3": 3, "cell-4": 5},
        max_schedulable_pusch_count_by_cell={"cell-3": 2, "cell-4": 3},
        redundancy_version_bits_by_cell={"cell-3": 2, "cell-4": 1},
        sri_mode="type2",
        tpmi_mode="type2",
        antenna_ports_mode="type2",
        tx_config_by_cell={"cell-3": "nonCodebook", "cell-4": "codebook"},
        srs_resource_count_by_cell={"cell-3": 2, "cell-4": 4},
        max_noncodebook_layers_by_cell={"cell-3": 2, "cell-4": None},
        ue_supports_max_mimo_layers_by_cell={"cell-3": True, "cell-4": True},
        dmrs_port_configuration_by_cell={
            "cell-3": (
                {"mapping_type": "A", "dmrs_type": 1, "dmrs_type_enhanced": False, "max_length": 1},
            ),
            "cell-4": (
                {"mapping_type": "B", "dmrs_type": 2, "dmrs_type_enhanced": True, "max_length": 2},
                {"mapping_type": "A", "dmrs_type": 1, "dmrs_type_enhanced": False, "max_length": 1},
            ),
        },
        codebook_srs_antenna_port_counts_by_cell={"cell-3": (), "cell-4": (4,)},
        codebook_subset_by_cell={"cell-3": None, "cell-4": "nonCoherent"},
        max_rank_for_codebook_by_cell={"cell-3": None, "cell-4": 3},
        ul_full_power_transmission_by_cell={"cell-3": None, "cell-4": None},
        ptrs_uplink_configured_by_cell={"cell-3": True, "cell-4": True},
        max_pusch_rank_by_cell={"cell-3": 4, "cell-4": 3},
        p0_pusch_set_list_configured_by_cell={"cell-3": True, "cell-4": True},
        sri_field_present_by_cell={"cell-3": True, "cell-4": False},
        olpc_parameter_set_dci_0_1_bits_by_cell={"cell-3": 2, "cell-4": 2},
        harq_ack_codebook_types=("semi_static", "dynamic"),
        dynamic_harq_ack_two_sub_codebooks_by_codebook=(False, True),
        priority_indicator_configured=True,
        minimum_scheduling_offset_configured=False,
    )
    widths = nr_srs_dci_0_3_field_widths_from_configuration(
        **resolver_inputs, scheduled_cells_indicator_value=2
    )
    assert widths["scell_dormancy_indication"] == 0
    assert widths["ptrs_dmrs_association"] == 2
    assert widths["open_loop_power_control_parameter_set_indication"] == 2
    no_p0_widths = nr_srs_dci_0_3_field_widths_from_configuration(
        **{
            **resolver_inputs,
            "p0_pusch_set_list_configured_by_cell": {
                "cell-3": False,
                "cell-4": False,
            },
        },
        scheduled_cells_indicator_value=2,
    )
    assert no_p0_widths["open_loop_power_control_parameter_set_indication"] == 0
    with pytest.raises(TypeError, match="all open-loop power-control configuration maps"):
        nr_srs_dci_0_3_field_widths_from_configuration(
            **{
                key: value
                for key, value in resolver_inputs.items()
                if key != "olpc_parameter_set_dci_0_1_bits_by_cell"
            },
            scheduled_cells_indicator_value=2,
        )
    with pytest.raises(ValueError, match="one or two bits"):
        nr_srs_dci_0_3_field_widths_from_configuration(
            **{
                **resolver_inputs,
                "olpc_parameter_set_dci_0_1_bits_by_cell": {
                    "cell-3": 3,
                    "cell-4": 2,
                },
            },
            scheduled_cells_indicator_value=2,
        )
    assert widths["pdcch_monitoring_adaptation_indication"] == 0
    assert widths["measurement_gap_cancellation"] == 0
    assert nr_srs_dci_0_3_field_widths_from_configuration(
        **resolver_inputs,
        scheduled_cells_indicator_value=2,
        measurement_gap_cancellation_configured=True,
    )["measurement_gap_cancellation"] == 1
    with pytest.raises(TypeError, match="must be a boolean"):
        nr_srs_dci_0_3_field_widths_from_configuration(
            **resolver_inputs,
            scheduled_cells_indicator_value=2,
            measurement_gap_cancellation_configured=1,
        )
    assert nr_srs_dci_0_3_field_widths_from_configuration(
        **resolver_inputs,
        scheduled_cells_indicator_value=2,
        pdcch_monitoring_adaptation_enabled=True,
        pdcch_skipping_duration_count=1,
    )["pdcch_monitoring_adaptation_indication"] == 1
    assert nr_srs_dci_0_3_field_widths_from_configuration(
        **resolver_inputs,
        scheduled_cells_indicator_value=2,
        pdcch_monitoring_adaptation_enabled=True,
        pdcch_skipping_duration_count=3,
    )["pdcch_monitoring_adaptation_indication"] == 2
    assert nr_srs_dci_0_3_field_widths_from_configuration(
        **resolver_inputs,
        scheduled_cells_indicator_value=2,
        pdcch_monitoring_adaptation_enabled=True,
        search_space_group_ids=(0, 1),
    )["pdcch_monitoring_adaptation_indication"] == 1
    assert nr_srs_dci_0_3_field_widths_from_configuration(
        **resolver_inputs,
        scheduled_cells_indicator_value=2,
        pdcch_monitoring_adaptation_enabled=True,
        search_space_group_ids=(0, 1, 2),
    )["pdcch_monitoring_adaptation_indication"] == 2
    assert nr_srs_dci_0_3_field_widths_from_configuration(
        **resolver_inputs,
        scheduled_cells_indicator_value=2,
        pdcch_monitoring_adaptation_enabled=True,
        pdcch_skipping_duration_count=1,
        search_space_group_ids=(0, 1),
    )["pdcch_monitoring_adaptation_indication"] == 2
    with pytest.raises(ValueError, match="requires a skipping-duration list"):
        nr_srs_dci_0_3_field_widths_from_configuration(
            **resolver_inputs,
            scheduled_cells_indicator_value=2,
            pdcch_monitoring_adaptation_enabled=True,
        )
    with pytest.raises(ValueError, match="must not contain duplicates"):
        nr_srs_dci_0_3_field_widths_from_configuration(
            **resolver_inputs,
            scheduled_cells_indicator_value=2,
            pdcch_monitoring_adaptation_enabled=True,
            search_space_group_ids=(0, 1, 1),
        )
    for group_ids in ((4,), (2, 4), (1, 3, 5), (0, 2, 4, 6)):
        assert nr_srs_dci_0_3_field_widths_from_configuration(
            **resolver_inputs,
            scheduled_cells_indicator_value=2,
            dormancy_configured=True,
            dormancy_group_ids=group_ids,
            carried_on_primary_cell_within_drx_active_time=True,
            scell_has_multiple_dl_bwps=True,
        )["scell_dormancy_indication"] == len(group_ids)
    assert nr_srs_dci_0_3_field_widths_from_configuration(
        **resolver_inputs,
        scheduled_cells_indicator_value=2,
        dormancy_configured=True,
        dormancy_group_ids=(0, 1),
        carried_on_primary_cell_within_drx_active_time=False,
        scell_has_multiple_dl_bwps=True,
    )["scell_dormancy_indication"] == 0
    with pytest.raises(ValueError, match="requires configured dormancy group IDs"):
        nr_srs_dci_0_3_field_widths_from_configuration(
            **resolver_inputs,
            scheduled_cells_indicator_value=2,
            dormancy_configured=True,
            carried_on_primary_cell_within_drx_active_time=True,
            scell_has_multiple_dl_bwps=True,
        )

    assert widths["identifier_for_dci_formats"] == 1
    assert widths["scheduled_cell_set_indicator"] == 1
    assert widths["scheduled_cells_indicator"] == 2
    assert widths["bandwidth_part_indicator"] == 2
    assert widths["frequency_domain_resource_assignment"] == 10
    assert widths["time_domain_resource_assignment"] == 3
    assert widths["frequency_hopping_flag"] == 1
    assert widths["dmrs_sequence_initialization"] == 1
    assert widths["ul_sch_indicator"] == 1
    assert widths["priority_indicator"] == 1
    assert widths["minimum_applicable_scheduling_offset_indicator"] == 0
    assert widths["beta_offset_indicator"] == 2
    assert widths["channel_access_cpext_capc"] == 2
    assert widths["tpc_command_for_scheduled_pusch"] == 4
    assert widths["harq_process_number"] == 8
    assert widths["modulation_and_coding_scheme"] == 10
    assert widths["new_data_indicator"] == 5
    assert widths["redundancy_version"] == 7
    assert widths["srs_resource_indicator"] == 4
    assert widths["precoding_information_and_number_of_layers"] == 4
    assert widths["antenna_ports"] == 8
    type1a_sri_widths = nr_srs_dci_0_3_field_widths_from_configuration(
        **{
            **resolver_inputs,
            "scheduled_cells_indicator_value": 2,
            "sri_mode": "type1a",
        }
    )
    assert type1a_sri_widths["srs_resource_indicator"] == 2
    type1a_antenna_widths = nr_srs_dci_0_3_field_widths_from_configuration(
        **{
            **resolver_inputs,
            "scheduled_cells_indicator_value": 2,
            "antenna_ports_mode": "type1a",
        }
    )
    assert type1a_antenna_widths["antenna_ports"] == 6
    type1a_precoding_widths = nr_srs_dci_0_3_field_widths_from_configuration(
        **{
            **resolver_inputs,
            "scheduled_cells_indicator_value": 2,
            "tpmi_mode": "type1a",
        }
    )
    assert type1a_precoding_widths["precoding_information_and_number_of_layers"] == 4
    full_power_mode1_inputs = {
        **resolver_inputs,
        "codebook_subset_by_cell": {
            "cell-3": None,
            "cell-4": "partialAndNonCoherent",
        },
        "max_rank_for_codebook_by_cell": {"cell-3": None, "cell-4": 2},
        "ul_full_power_transmission_by_cell": {
            "cell-3": None,
            "cell-4": "fullpowerMode1",
        },
    }
    full_power_mode1_widths = nr_srs_dci_0_3_field_widths_from_configuration(
        **full_power_mode1_inputs, scheduled_cells_indicator_value=2
    )
    assert full_power_mode1_widths["precoding_information_and_number_of_layers"] == 5
    two_port_full_power_inputs = {
        **full_power_mode1_inputs,
        "codebook_srs_antenna_port_counts_by_cell": {
            "cell-3": (),
            "cell-4": (2,),
        },
        "codebook_subset_by_cell": {"cell-3": None, "cell-4": "nonCoherent"},
    }
    two_port_full_power_widths = nr_srs_dci_0_3_field_widths_from_configuration(
        **two_port_full_power_inputs, scheduled_cells_indicator_value=2
    )
    assert two_port_full_power_widths["precoding_information_and_number_of_layers"] == 2
    with pytest.raises(ValueError, match="mapping types must be unique"):
        nr_srs_dci_0_3_field_widths_from_configuration(
            **{
                **resolver_inputs,
                "dmrs_port_configuration_by_cell": {
                    **resolver_inputs["dmrs_port_configuration_by_cell"],
                    "cell-4": (
                        {
                            "mapping_type": "B",
                            "dmrs_type": 2,
                            "dmrs_type_enhanced": True,
                            "max_length": 2,
                        },
                        {
                            "mapping_type": "B",
                            "dmrs_type": 1,
                            "dmrs_type_enhanced": False,
                            "max_length": 1,
                        },
                    ),
                },
            },
            scheduled_cells_indicator_value=2,
        )
    capability_fallback_widths = nr_srs_dci_0_3_field_widths_from_configuration(
        **{
            **resolver_inputs,
            "max_noncodebook_layers_by_cell": {"cell-3": None, "cell-4": None},
            "ue_max_noncodebook_layers_by_cell": {"cell-3": 1, "cell-4": None},
        },
        scheduled_cells_indicator_value=2,
    )
    assert capability_fallback_widths["srs_resource_indicator"] == 3
    unsupported_config_widths = nr_srs_dci_0_3_field_widths_from_configuration(
        **{
            **resolver_inputs,
            "ue_supports_max_mimo_layers_by_cell": {
                "cell-3": False,
                "cell-4": True,
            },
            "ue_max_noncodebook_layers_by_cell": {"cell-3": 1, "cell-4": None},
        },
        scheduled_cells_indicator_value=2,
    )
    assert unsupported_config_widths["srs_resource_indicator"] == 3
    with pytest.raises(ValueError, match="UE non-codebook layer capability is required"):
        nr_srs_dci_0_3_field_widths_from_configuration(
            **{
                **resolver_inputs,
                "max_noncodebook_layers_by_cell": {
                    "cell-3": None,
                    "cell-4": None,
                },
            },
            scheduled_cells_indicator_value=2,
        )
    with pytest.raises(ValueError, match="sri_mode"):
        nr_srs_dci_0_3_field_widths_from_configuration(
            **{
                **resolver_inputs,
                "scheduled_cells_indicator_value": 2,
                "sri_mode": "type3",
            }
        )
    with pytest.raises(ValueError, match="cover the full scheduled cell list"):
        nr_srs_dci_0_3_field_widths_from_configuration(
            **{
                **resolver_inputs,
                "srs_resource_count_by_cell": {"cell-3": 2},
            },
            scheduled_cells_indicator_value=2,
        )
    assert widths["first_downlink_assignment_index"] == 2
    assert widths["second_downlink_assignment_index"] == 2
    assert widths["srs_request"] == 2
    assert widths["srs_offset_indicator"] == 3
    assert widths["csi_request"] == 3
    srs_offset = sum(
        widths[name] for name in field_names[: field_names.index("srs_request")]
    )
    payload = [0] * sum(widths.values())
    payload[srs_offset : srs_offset + 2] = [1, 0]
    assert nr_srs_dci_0_3_request_list_index_from_layout(
        payload, field_widths_bits=widths, request_list_entry_count=3
    ) == 2

    single_cell_widths = nr_srs_dci_0_3_field_widths_from_configuration(
        **resolver_inputs, scheduled_cells_indicator_value=0
    )
    assert single_cell_widths["tpc_command_for_scheduled_pusch"] == 2
    assert single_cell_widths["harq_process_number"] == 3
    assert single_cell_widths["modulation_and_coding_scheme"] == 5
    assert single_cell_widths["new_data_indicator"] == 2
    assert single_cell_widths["redundancy_version"] == 4
    assert single_cell_widths["frequency_domain_resource_assignment"] == 5
    assert single_cell_widths["srs_resource_indicator"] == 2
    assert single_cell_widths["ptrs_dmrs_association"] == 0
    cell_four_widths = nr_srs_dci_0_3_field_widths_from_configuration(
        **resolver_inputs, scheduled_cells_indicator_value=1
    )
    assert cell_four_widths["ptrs_dmrs_association"] == 2
    assert cell_four_widths["open_loop_power_control_parameter_set_indication"] == 2
    assert single_cell_widths["open_loop_power_control_parameter_set_indication"] == 1
    rank_one_widths = nr_srs_dci_0_3_field_widths_from_configuration(
        **{
            **resolver_inputs,
            "max_pusch_rank_by_cell": {"cell-3": 1, "cell-4": 1},
        },
        scheduled_cells_indicator_value=2,
    )
    assert rank_one_widths["ptrs_dmrs_association"] == 0
    with pytest.raises(TypeError, match="must both be supplied"):
        nr_srs_dci_0_3_field_widths_from_configuration(
            **{
                key: value
                for key, value in resolver_inputs.items()
                if key != "max_pusch_rank_by_cell"
            },
            scheduled_cells_indicator_value=2,
        )


def test_dci_0_3_field_widths_derive_absent_frequency_hopping_and_single_tdra_entry() -> None:
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
    derived_fields = {
        "identifier_for_dci_formats",
        "scheduled_cell_set_indicator",
        "scheduled_cells_indicator",
        "bandwidth_part_indicator",
        "frequency_domain_resource_assignment",
        "time_domain_resource_assignment",
        "frequency_hopping_flag",
        "dmrs_sequence_initialization",
        "ul_sch_indicator",
        "priority_indicator",
        "minimum_applicable_scheduling_offset_indicator",
        "beta_offset_indicator",
        "channel_access_cpext_capc",
        "tpc_command_for_scheduled_pusch",
        "harq_process_number",
        "modulation_and_coding_scheme",
        "new_data_indicator",
        "redundancy_version",
        "first_downlink_assignment_index",
        "second_downlink_assignment_index",
        "srs_resource_indicator",
        "precoding_information_and_number_of_layers",
        "antenna_ports",
        "srs_request",
        "srs_offset_indicator",
        "csi_request",
        "ptrs_dmrs_association",
        "open_loop_power_control_parameter_set_indication",
        "scell_dormancy_indication",
        "pdcch_monitoring_adaptation_indication",
        "measurement_gap_cancellation",
    }
    assert set(derived_fields) == set(field_names)
    widths = nr_srs_dci_0_3_field_widths_from_configuration(
        cell_set_configs=(
            {
                "set_id": 0,
                "scheduled_cell_ids": ("cell-1", "cell-2"),
                "scheduled_cell_combos": ((0, 1),),
            },
        ),
        configured_ul_bwp_ids_by_cell={"cell-1": (0,), "cell-2": (0,)},
        active_ul_bwp_id_by_cell={"cell-1": 0, "cell-2": 0},
        ul_bwp_frequency_allocation_by_cell={
            "cell-1": {
                0: {"size_prb": 10, "start_prb": 0, "resource_allocation": "type0"}
            },
            "cell-2": {
                0: {"size_prb": 10, "start_prb": 0, "resource_allocation": "type0"}
            },
        },
        request_list_entry_count=0,
        srs_offset_list=None,
        report_trigger_size=0,
        tdra_field_index_list_entry_count=1,
        frequency_hopping_enabled_by_cell={"cell-1": False, "cell-2": False},
        transform_precoding_enabled_by_cell={"cell-1": True, "cell-2": True},
        beta_offsets_semi_static_by_cell={"cell-1": True, "cell-2": True},
        channel_access_config_entry_count_by_cell={"cell-1": 0, "cell-2": 0},
        harq_process_number_size_bits_by_cell={"cell-1": 0, "cell-2": 0},
        max_schedulable_pusch_count_by_cell={"cell-1": 1, "cell-2": 1},
        redundancy_version_bits_by_cell={"cell-1": 0, "cell-2": 0},
        sri_mode="type1a",
        tpmi_mode="type1a",
        antenna_ports_mode="type1a",
        tx_config_by_cell={"cell-1": "codebook", "cell-2": "nonCodebook"},
        srs_resource_count_by_cell={"cell-1": 1, "cell-2": 3},
        max_noncodebook_layers_by_cell={"cell-1": None, "cell-2": 1},
        ue_supports_max_mimo_layers_by_cell={"cell-1": True, "cell-2": True},
        dmrs_port_configuration_by_cell={
            "cell-1": (
                {"mapping_type": "A", "dmrs_type": 1, "dmrs_type_enhanced": False, "max_length": 1},
            ),
            "cell-2": (
                {"mapping_type": "B", "dmrs_type": 1, "dmrs_type_enhanced": False, "max_length": 1},
            ),
        },
        codebook_srs_antenna_port_counts_by_cell={"cell-1": (1,), "cell-2": ()},
        codebook_subset_by_cell={"cell-1": None, "cell-2": None},
        max_rank_for_codebook_by_cell={"cell-1": None, "cell-2": None},
        ul_full_power_transmission_by_cell={"cell-1": None, "cell-2": None},
        harq_ack_codebook_types=("semi_static",),
        dynamic_harq_ack_two_sub_codebooks_by_codebook=(False,),
        priority_indicator_configured=False,
        minimum_scheduling_offset_configured=True,
    )
    assert widths["time_domain_resource_assignment"] == 0
    assert widths["frequency_hopping_flag"] == 0
    assert widths["dmrs_sequence_initialization"] == 0
    assert widths["ul_sch_indicator"] == 1
    assert widths["priority_indicator"] == 0
    assert widths["minimum_applicable_scheduling_offset_indicator"] == 1
    assert widths["beta_offset_indicator"] == 0
    assert widths["channel_access_cpext_capc"] == 0
    assert widths["tpc_command_for_scheduled_pusch"] == 4
    assert widths["harq_process_number"] == 0
    assert widths["modulation_and_coding_scheme"] == 10
    assert widths["new_data_indicator"] == 2
    assert widths["redundancy_version"] == 0
    assert widths["srs_resource_indicator"] == 2
    assert widths["first_downlink_assignment_index"] == 1
    assert widths["second_downlink_assignment_index"] == 0


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
    assert nr_srs_dci_0_3_scheduled_cell_ids(
        configs, default_set_id=2
    ) == ("cell-4", "cell-8")


@pytest.mark.parametrize(
    ("configs", "kwargs", "error", "message"),
    (
        (({"set_id": 0, "scheduled_cell_ids": ("a", "b")},),
         {"scheduled_cells_indicator": 0}, ValueError, "no combo list"),
        (({"set_id": 0, "scheduled_cell_ids": ("a", "b"), "scheduled_cell_combos": ((0, 1),)},),
         {"scheduled_cells_indicator": 0}, ValueError, "single combo"),
        (({"set_id": 0, "scheduled_cell_ids": ("a", "b"), "scheduled_cell_combos": ((0,), (1,))},),
         {"scheduled_cells_indicator": 2}, ValueError, "no configured combo"),
        (({"set_id": 0, "scheduled_cell_ids": ("a", "b"), "scheduled_cell_combos": ((1, 0), (0,))},),
         {"scheduled_cells_indicator": 0}, ValueError, "ascending"),
        (({"set_id": 0, "scheduled_cell_ids": ("a", "b"), "scheduled_cell_combos": ((0, 2), (1,))},),
         {"scheduled_cells_indicator": 0}, ValueError, "outside scheduled_cell_ids"),
        (({"set_id": 0, "scheduled_cell_ids": ("a", "b")}, {"set_id": 1, "scheduled_cell_ids": ("c", "d")}),
         {}, ValueError, "multiple cell sets"),
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
    assert nr_srs_dci_0_3_multicell_aperiodic_resource_set_schedules_from_bits(
        (),
        field_offset=0,
        request_list=(),
        serving_cell_ids=(),
        resource_sets_by_cell={},
        triggering_dci_slot_by_cell={},
        cell_set_configs=(),
    ) is None


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
        nr_srs_frequency_position_indices(
            55, b_srs=3, b_hop=1, n_srs=counter
        )
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
def test_srs_periodicity_choice_decodes_slot_period_and_offset(
    choice: str, period: int
) -> None:
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


def test_positioning_srs_frequency_allocation_applies_hyper_sfn_gate() -> None:
    common = {
        "c_srs": 9,
        "transmission_comb": 4,
        "b_srs": 3,
        "b_hop": 0,
        "srs_symbol_count": 2,
        "resource_type": "periodic",
        "absolute_slot_index": 0,
        "periodicity_slots": 20480,
        "slot_offset": 0,
        "positioning_resource": True,
        "numerology": 1,
    }

    eligible = nr_srs_resource_frequency_allocations(
        **common, hyper_frame_number=3, hyper_sfn_index=1
    )
    ineligible = nr_srs_resource_frequency_allocations(
        **common, hyper_frame_number=3, hyper_sfn_index=0
    )

    assert eligible is not None
    assert ineligible is None


def test_positioning_srs_periodic_schedule_applies_hyper_sfn_gate() -> None:
    common = {
        "absolute_slot_index": 0,
        "c_srs": 9,
        "transmission_comb": 4,
        "b_srs": 3,
        "b_hop": 0,
        "srs_symbol_count": 2,
        "resource_type": "periodic",
        "periodicity_slots": 20480,
        "slot_offset": 0,
        "positioning_resource": True,
        "numerology": 1,
        "hyper_frame_number": 3,
    }

    active = nr_srs_periodic_resource_schedule(**common, hyper_sfn_index=1)
    gated = nr_srs_periodic_resource_schedule(**common, hyper_sfn_index=0)

    assert active is not None
    assert active.target_slot_index == 0
    assert len(active.allocations) == 2
    assert gated is None


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
    assert nr_srs_aperiodic_target_slot_index(
        100,
        slot_offset=2,
        available_slot_offsets=[4, 7],
        selected_available_offset_index=1,
    ) == 109


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

    assert nr_srs_triggered_resource_set_ids(
        2, resource_sets, dci_format="0_1"
    ) == (12,)
    assert nr_srs_triggered_resource_set_ids(
        3, resource_sets, dci_format="0_1"
    ) == (4, 12)
    assert nr_srs_triggered_resource_set_ids(
        0, resource_sets, dci_format="0_1"
    ) == ()


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

    assert nr_srs_triggered_resource_set_ids(
        0, resource_sets, dci_format=dci_format, srs_request_mode=1
    ) == ()
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
        (cell, tuple(item.resource_set_id for item in schedules))
        for cell, schedules in result
    ) == (("cell-3", (31,)), ("cell-9", (91,)))
    assert tuple(schedules[0].target_slot_index for _, schedules in result) == (26, 201)
    assert tuple(
        tuple(item.n_srs for item in schedules[0].allocations) for _, schedules in result
    ) == ((0, 1), (0, 1))


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
    resource_sets = [
        {"resource_set_id": 4, "resource_type": "aperiodic", "aperiodic_trigger": 1}
    ]
    assert nr_srs_aperiodic_resource_set_schedules(
        50, 0, resource_sets, dci_format="0_1"
    ) == ()
    with pytest.raises(ValueError, match="requires an allocation mapping"):
        nr_srs_aperiodic_resource_set_schedules(
            50, 1, resource_sets, dci_format="0_1"
        )


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


def test_srs_hopped_mapper_combines_hopping_and_positioning_offsets() -> None:
    allocations = tuple(
        nr_srs_fixed_frequency_allocation(
            55,
            transmission_comb=4,
            b_srs=3,
            b_hop=1,
            n_srs=counter,
        )
        for counter in range(4)
    )
    sequence = np.ones((4, allocations[0].bandwidth.sequence_length), dtype=np.complex128)

    mapped = map_nr_srs_hopped_resource(
        sequence,
        frequency_allocations=allocations,
        slot_symbol_count=14,
        subcarrier_count=1024,
        start_position_offset=3,
        srs_symbol_count=4,
        positioning_resource=True,
    )

    assert mapped.positioning_offsets == (0, 2, 1, 3)
    assert mapped.n_srs_by_symbol == (0, 1, 2, 3)
    expected_starts = (0, 194, 385, 99)
    for symbol, indices, first in zip(
        mapped.symbol_indices,
        mapped.subcarrier_indices_by_symbol,
        expected_starts,
        strict=True,
    ):
        assert indices == tuple(range(first, first + 96, 4))
        assert mapped.mask[symbol, list(indices)].all()


def test_srs_hopped_multiport_mapper_preserves_per_port_symbols_and_offsets() -> None:
    allocations = tuple(
        nr_srs_fixed_frequency_allocation(
            55,
            transmission_comb=4,
            b_srs=3,
            b_hop=1,
            n_srs=counter,
        )
        for counter in range(4)
    )
    sequence_length = allocations[0].bandwidth.sequence_length
    sequences = np.stack(
        [
            np.full((4, sequence_length), 1.0 + 1.0j),
            np.full((4, sequence_length), 2.0 - 1.0j),
        ]
    )

    mapped = map_nr_srs_hopped_multiport_resource(
        sequences,
        frequency_allocations=allocations,
        slot_symbol_count=14,
        subcarrier_count=1024,
        start_position_offset=3,
        srs_symbol_count=4,
        positioning_resource=True,
    )

    assert isinstance(mapped, NRSRSHoppedMultiportResourceGrid)
    assert mapped.grid.shape == (2, 14, 1024)
    assert mapped.positioning_offsets == (0, 2, 1, 3)
    assert mapped.n_srs_by_symbol == (0, 1, 2, 3)
    expected_starts = (0, 194, 385, 99)
    for symbol, indices, first in zip(
        mapped.symbol_indices,
        mapped.subcarrier_indices_by_symbol,
        expected_starts,
        strict=True,
    ):
        assert indices == tuple(range(first, first + 96, 4))
        for port, value in enumerate((1.0 + 1.0j, 2.0 - 1.0j)):
            np.testing.assert_array_equal(
                mapped.grid[port, symbol, list(indices)],
                np.full(len(indices), value / np.sqrt(2), dtype=np.complex128),
            )
            assert mapped.mask[port, symbol, list(indices)].all()
    assert not mapped.grid.flags.writeable
    assert not mapped.mask.flags.writeable


@pytest.mark.parametrize("symbol_index", range(4))
def test_srs_aperiodic_transmission_counter_uses_repetition_factor(symbol_index) -> None:
    assert nr_srs_transmission_counter(
        resource_type="aperiodic",
        srs_symbol_count=4,
        resource_symbol_index=symbol_index,
        repetition_factor=2,
    ) == symbol_index // 2


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


def test_positioning_srs_long_period_applies_hyper_sfn_gate() -> None:
    common = {
        "resource_type": "periodic",
        "srs_symbol_count": 2,
        "resource_symbol_index": 0,
        "absolute_slot_index": 0,
        "periodicity_slots": 20480,
        "slot_offset": 0,
        "positioning_resource": True,
        "numerology": 1,
    }

    assert nr_srs_transmission_counter(
        **common, hyper_frame_number=3, hyper_sfn_index=1
    ) == 0
    assert nr_srs_transmission_counter(
        **common, hyper_frame_number=3, hyper_sfn_index=0
    ) is None


def test_positioning_srs_hyper_sfn_configuration_is_validated_off_occasion() -> None:
    with pytest.raises(ValueError, match=r"at least 2\^mu \* 10240 slots"):
        nr_srs_transmission_counter(
            resource_type="periodic",
            srs_symbol_count=2,
            resource_symbol_index=0,
            absolute_slot_index=1,
            periodicity_slots=10240,
            slot_offset=0,
            positioning_resource=True,
            numerology=1,
            hyper_frame_number=0,
            hyper_sfn_index=0,
        )


def test_hyper_sfn_gate_is_positioning_only() -> None:
    with pytest.raises(ValueError, match="only defined for positioning SRS"):
        nr_srs_transmission_counter(
            resource_type="periodic",
            srs_symbol_count=2,
            resource_symbol_index=0,
            absolute_slot_index=0,
            periodicity_slots=10240,
            slot_offset=0,
            numerology=0,
            hyper_frame_number=0,
            hyper_sfn_index=0,
        )


@pytest.mark.parametrize("hyper_frame_number", [-1, 1024])
def test_positioning_srs_hyper_frame_number_uses_nr_range(hyper_frame_number) -> None:
    with pytest.raises(ValueError, match=r"hyper_frame_number must be in \[0, 1023\]"):
        nr_srs_transmission_counter(
            resource_type="periodic",
            srs_symbol_count=2,
            resource_symbol_index=0,
            absolute_slot_index=0,
            periodicity_slots=10240,
            slot_offset=0,
            positioning_resource=True,
            numerology=0,
            hyper_frame_number=hyper_frame_number,
            hyper_sfn_index=0,
        )


@pytest.mark.parametrize(
    ("comb", "symbol_count", "expected"),
    [
        (2, 1, (0,)),
        (2, 2, (0, 1)),
        (2, 4, (0, 1, 0, 1)),
        (4, 2, (0, 2)),
        (4, 4, (0, 2, 1, 3)),
        (4, 8, (0, 2, 1, 3, 0, 2, 1, 3)),
        (4, 12, (0, 2, 1, 3, 0, 2, 1, 3, 0, 2, 1, 3)),
        (8, 4, (0, 4, 2, 6)),
        (8, 8, (0, 4, 2, 6, 1, 5, 3, 7)),
        (8, 12, (0, 4, 2, 6, 1, 5, 3, 7, 0, 4, 2, 6)),
    ],
)
def test_srs_positioning_symbol_offsets_match_all_supported_table_rows(
    comb, symbol_count, expected
) -> None:
    actual = tuple(
        nr_srs_positioning_symbol_offset(
            comb,
            srs_symbol_count=symbol_count,
            resource_symbol_index=index,
        )
        for index in range(symbol_count)
    )
    assert actual == expected


@pytest.mark.parametrize(
    ("comb", "symbol_count", "symbol_index", "message"),
    [
        (2, 8, 0, "unsupported"),
        (4, 1, 0, "unsupported"),
        (8, 2, 0, "unsupported"),
        (4, 4, 4, "inside"),
    ],
)
def test_srs_positioning_symbol_offset_rejects_unsupported_table_cases(
    comb, symbol_count, symbol_index, message
) -> None:
    with pytest.raises(ValueError, match=message):
        nr_srs_positioning_symbol_offset(
            comb,
            srs_symbol_count=symbol_count,
            resource_symbol_index=symbol_index,
        )


def test_srs_positioning_mapper_applies_distinct_comb_offsets_by_symbol() -> None:
    allocation = nr_srs_fixed_frequency_allocation(
        55,
        transmission_comb=4,
        b_srs=3,
        b_hop=3,
        frequency_domain_position=7,
    )
    sequence = (
        np.arange(4 * allocation.bandwidth.sequence_length, dtype=np.float64)
        .reshape(4, allocation.bandwidth.sequence_length)
        .astype(np.complex128)
        + 1j
    )

    mapped = map_nr_srs_positioning_resource(
        sequence,
        slot_symbol_count=14,
        subcarrier_count=512,
        start_position_offset=3,
        srs_symbol_count=4,
        comb_size=4,
        comb_offset=2,
        frequency_start_subcarrier=allocation.start_subcarrier,
        allocated_bandwidth_subcarriers=allocation.bandwidth.allocated_bandwidth_subcarriers,
    )

    assert isinstance(mapped, NRSRSPositioningResourceGrid)
    assert mapped.symbol_indices == (10, 11, 12, 13)
    assert mapped.positioning_offsets == (0, 2, 1, 3)
    expected_first_subcarriers = (290, 288, 291, 289)
    for row, (symbol, indices, first) in enumerate(
        zip(
            mapped.symbol_indices,
            mapped.subcarrier_indices_by_symbol,
            expected_first_subcarriers,
            strict=True,
        )
    ):
        assert indices == tuple(range(first, first + 96, 4))
        np.testing.assert_array_equal(mapped.grid[symbol, list(indices)], sequence[row])
        assert mapped.mask[symbol, list(indices)].all()
    assert int(np.count_nonzero(mapped.mask)) == 4 * allocation.bandwidth.sequence_length
    assert not mapped.grid.flags.writeable
    assert not mapped.mask.flags.writeable


def test_srs_positioning_mapper_broadcasts_one_sequence_row_and_applies_beta() -> None:
    allocation = nr_srs_fixed_frequency_allocation(
        55, transmission_comb=4, b_srs=3, b_hop=3
    )
    sequence = np.full(allocation.bandwidth.sequence_length, 1.0 + 2.0j)
    mapped = map_nr_srs_positioning_resource(
        sequence,
        slot_symbol_count=14,
        subcarrier_count=512,
        start_position_offset=3,
        srs_symbol_count=4,
        comb_size=4,
        comb_offset=0,
        frequency_start_subcarrier=allocation.start_subcarrier,
        allocated_bandwidth_subcarriers=allocation.bandwidth.allocated_bandwidth_subcarriers,
        beta_srs=0.5,
    )

    for symbol, indices in zip(
        mapped.symbol_indices, mapped.subcarrier_indices_by_symbol, strict=True
    ):
        np.testing.assert_array_equal(mapped.grid[symbol, list(indices)], 0.5 + 1.0j)
    assert int(np.count_nonzero(mapped.mask)) == 4 * allocation.bandwidth.sequence_length


def test_srs_positioning_multiport_mapper_keeps_port_and_symbol_rosters() -> None:
    allocation = nr_srs_fixed_frequency_allocation(
        55, transmission_comb=4, b_srs=3, b_hop=3
    )
    sequences = np.stack(
        [
            np.full((4, allocation.bandwidth.sequence_length), 1.0 + 1.0j),
            np.full((4, allocation.bandwidth.sequence_length), 2.0 - 1.0j),
        ]
    )

    mapped = map_nr_srs_positioning_multiport_resource(
        sequences,
        slot_symbol_count=14,
        subcarrier_count=512,
        start_position_offset=3,
        srs_symbol_count=4,
        comb_size=4,
        comb_offset=0,
        frequency_start_subcarrier=allocation.start_subcarrier,
        allocated_bandwidth_subcarriers=allocation.bandwidth.allocated_bandwidth_subcarriers,
    )

    assert isinstance(mapped, NRSRSPositioningMultiportResourceGrid)
    assert mapped.grid.shape == (2, 14, 512)
    assert mapped.symbol_indices == (10, 11, 12, 13)
    assert mapped.positioning_offsets == (0, 2, 1, 3)
    for port, sequence_rows in enumerate(sequences):
        for row, symbol in enumerate(mapped.symbol_indices):
            indices = mapped.subcarrier_indices_by_symbol[row]
            np.testing.assert_array_equal(
                mapped.grid[port, symbol, list(indices)], sequence_rows[row] / np.sqrt(2)
            )
            assert mapped.mask[port, symbol, list(indices)].all()
    assert not mapped.grid.flags.writeable
    assert not mapped.mask.flags.writeable


def test_srs_positioning_multiport_mapper_applies_eight_port_tdm() -> None:
    sequences = np.ones((8, 2, 36), dtype=np.complex128)
    mapped = map_nr_srs_positioning_multiport_resource(
        sequences,
        slot_symbol_count=14,
        subcarrier_count=72,
        start_position_offset=1,
        srs_symbol_count=2,
        comb_size=2,
        comb_offset=0,
        frequency_start_subcarrier=0,
        allocated_bandwidth_subcarriers=72,
        ports8_tdm=True,
    )

    assert mapped.symbol_indices == (12, 13)
    assert mapped.active_port_symbols == nr_srs_port_symbol_activity(
        tuple(range(1000, 1008)), srs_symbol_count=2, ports8_tdm=True
    )
    assert mapped.mask[:, 12, :].sum(axis=1).tolist() == [36, 36, 0, 0, 36, 36, 0, 0]
    assert mapped.mask[:, 13, :].sum(axis=1).tolist() == [0, 0, 36, 36, 0, 0, 36, 36]
    assert np.all(mapped.grid[~mapped.mask] == 0)
    assert not mapped.grid.flags.writeable


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"srs_symbol_count": 10}, ValueError, "symbol count is unsupported"),
        (
            {"comb_size": 8, "srs_symbol_count": 2},
            ValueError,
            "unsupported for this SRS-PosResource comb",
        ),
        ({"allocated_bandwidth_subcarriers": 95}, ValueError, "comb-aligned"),
        ({"start_position_offset": 2}, ValueError, "start_position_offset"),
    ],
)
def test_srs_positioning_mapper_rejects_invalid_configuration(overrides, error, message) -> None:
    allocation = nr_srs_fixed_frequency_allocation(
        55, transmission_comb=4, b_srs=3, b_hop=3
    )
    config = {
        "slot_symbol_count": 14,
        "subcarrier_count": 512,
        "start_position_offset": 3,
        "srs_symbol_count": 4,
        "comb_size": 4,
        "comb_offset": 0,
        "frequency_start_subcarrier": allocation.start_subcarrier,
        "allocated_bandwidth_subcarriers": allocation.bandwidth.allocated_bandwidth_subcarriers,
    }
    config.update(overrides)
    with pytest.raises(error, match=message):
        map_nr_srs_positioning_resource(
            np.ones((config["srs_symbol_count"], 24), dtype=np.complex128), **config
        )


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


@pytest.mark.parametrize("transmission_comb", [2, 4, 8])
def test_srs_type1_bandwidth_table_b_srs_zero_matches_all_64_c_srs_rows(
    transmission_comb,
) -> None:
    # TS 38.211 V19.2.0 Table 6.4.1.4.3-1, m_SRS,0 (b_SRS=0).
    expected_bandwidth_rb = (
        4, 8, 12, 16, 16, 20, 24, 24, 28, 32, 36, 40, 48, 48, 52, 56,
        60, 64, 72, 72, 76, 80, 88, 96, 96, 104, 112, 120, 120, 120, 128, 128,
        128, 132, 136, 144, 144, 144, 144, 152, 160, 160, 160, 168, 176, 184, 192, 192,
        192, 192, 208, 216, 224, 240, 240, 240, 240, 256, 256, 256, 264, 272, 272, 272,
    )
    for c_srs, bandwidth_rb in enumerate(expected_bandwidth_rb):
        config = nr_srs_type1_bandwidth_configuration(
            c_srs, transmission_comb=transmission_comb
        )
        assert isinstance(config, NRSRSBandwidthConfiguration)
        assert config.c_srs == c_srs
        assert config.b_srs == 0
        assert config.transmission_comb == transmission_comb
        assert config.bandwidth_rb == bandwidth_rb
        assert config.subband_count == 1
        assert config.allocated_bandwidth_subcarriers == 12 * bandwidth_rb
        assert config.sequence_length == 12 * bandwidth_rb // transmission_comb


@pytest.mark.parametrize(
    ("c_srs", "b_srs", "bandwidth_rb", "subband_count"),
    [
        (0, 0, 4, 1), (0, 1, 4, 1), (0, 2, 4, 1), (0, 3, 4, 1),
        (9, 1, 16, 2), (9, 2, 8, 2), (9, 3, 4, 2),
        (13, 1, 24, 2), (13, 2, 12, 2), (13, 3, 4, 3),
        (36, 1, 48, 3), (36, 2, 24, 2), (36, 3, 12, 2),
        (49, 1, 24, 8), (49, 2, 8, 3), (49, 3, 4, 2),
        (55, 1, 48, 5), (55, 2, 16, 3), (55, 3, 8, 2),
        (61, 1, 136, 2), (61, 2, 68, 2), (61, 3, 4, 17),
        (62, 1, 68, 4), (62, 2, 4, 17), (62, 3, 4, 1),
        (63, 1, 16, 17), (63, 2, 8, 2), (63, 3, 4, 2),
    ],
)
def test_srs_type1_bandwidth_table_resolves_all_b_srs_levels(
    c_srs, b_srs, bandwidth_rb, subband_count
) -> None:
    config = nr_srs_type1_bandwidth_configuration(
        c_srs, b_srs=b_srs, transmission_comb=4
    )

    assert config.bandwidth_rb == bandwidth_rb
    assert config.subband_count == subband_count
    assert config.allocated_bandwidth_subcarriers == 12 * bandwidth_rb
    assert config.sequence_length == 3 * bandwidth_rb


@pytest.mark.parametrize(
    ("c_srs", "comb", "error", "message"),
    [
        (True, 2, TypeError, "c_srs"),
        (-1, 2, ValueError, "c_srs"),
        (64, 2, ValueError, "c_srs"),
        (0, 3, ValueError, "transmission_comb"),
    ],
)
def test_srs_type1_bandwidth_configuration_rejects_invalid_inputs(
    c_srs, comb, error, message
) -> None:
    with pytest.raises(error, match=message):
        nr_srs_type1_bandwidth_configuration(c_srs, transmission_comb=comb)


@pytest.mark.parametrize(
    ("b_srs", "error"),
    [(True, TypeError), (1.0, TypeError), (-1, ValueError), (4, ValueError)],
)
def test_srs_type1_bandwidth_configuration_rejects_invalid_b_srs(b_srs, error) -> None:
    with pytest.raises(error, match="b_srs"):
        nr_srs_type1_bandwidth_configuration(0, b_srs=b_srs, transmission_comb=2)


def test_srs_type1_bandwidth_configuration_rejects_inconsistent_public_value() -> None:
    with pytest.raises(ValueError, match="standardized c_srs row"):
        NRSRSBandwidthConfiguration(
            c_srs=0,
            b_srs=0,
            transmission_comb=2,
            bandwidth_rb=8,
            subband_count=1,
            sequence_length=48,
            allocated_bandwidth_subcarriers=96,
        )


def test_srs_mapper_respects_allocated_bandwidth_inside_wider_grid() -> None:
    sequence = np.exp(1j * np.arange(108, dtype=np.float64) * 0.1)
    result = map_nr_srs_resource(
        sequence,
        slot_symbol_count=14,
        subcarrier_count=600,
        start_position_offset=0,
        srs_symbol_count=1,
        comb_size=4,
        comb_offset=1,
        frequency_start_subcarrier=24,
        allocated_bandwidth_subcarriers=432,
    )

    assert result.subcarrier_indices == tuple(range(25, 456, 4))
    assert len(result.subcarrier_indices) == len(sequence)
    assert not result.mask[:, :25].any()
    assert not result.mask[:, 456:].any()
    np.testing.assert_array_equal(result.grid[13, 25:456:4], sequence)


def test_srs_fixed_bandwidth_sequences_map_to_exactly_the_resolved_extent() -> None:
    bandwidth = nr_srs_type1_bandwidth_configuration(9, transmission_comb=4)
    sequences = nr_srs_port_sequences_type1(
        bandwidth.sequence_length,
        cyclic_shift_indices=(0, 1),
        srs_symbol_count=2,
        sequence_id=21,
        slot_index=3,
        start_symbol_index=10,
        hopping_mode="neither",
        transmission_comb=4,
    )
    mapped = map_nr_srs_multiport_resource(
        sequences,
        slot_symbol_count=14,
        subcarrier_count=600,
        start_position_offset=1,
        srs_symbol_count=2,
        comb_size=bandwidth.transmission_comb,
        comb_offset=2,
        frequency_start_subcarrier=48,
        allocated_bandwidth_subcarriers=bandwidth.allocated_bandwidth_subcarriers,
    )

    assert sequences.shape == (2, 2, 96)
    assert mapped.grid.shape == (2, 14, 600)
    assert mapped.subcarrier_indices == tuple(range(50, 434, 4))
    assert len(mapped.subcarrier_indices) == bandwidth.sequence_length
    assert not mapped.mask[:, :, :50].any()
    assert not mapped.mask[:, :, 434:].any()


def test_srs_mapper_rejects_allocated_bandwidth_outside_grid() -> None:
    with pytest.raises(ValueError, match="allocated bandwidth"):
        map_nr_srs_resource(
            np.ones(24, dtype=np.complex128),
            slot_symbol_count=14,
            subcarrier_count=48,
            start_position_offset=1,
            srs_symbol_count=2,
            comb_size=2,
            comb_offset=0,
            frequency_start_subcarrier=24,
            allocated_bandwidth_subcarriers=25,
        )


def test_srs_multiport_sequences_use_explicit_cyclic_shifts_and_are_orthogonal() -> None:
    sequences = nr_srs_port_sequences_type1(
        72,
        cyclic_shift_indices=(0, 1, 5),
        srs_symbol_count=2,
        sequence_id=37,
        slot_index=4,
        start_symbol_index=10,
        hopping_mode="groupHopping",
        transmission_comb=4,
    )

    assert sequences.shape == (3, 2, 72)
    for symbol_index in range(2):
        for left_port in range(3):
            for right_port in range(left_port + 1, 3):
                cross_correlation = np.vdot(
                    sequences[left_port, symbol_index],
                    sequences[right_port, symbol_index],
                )
                assert abs(cross_correlation) < 1e-10


@pytest.mark.parametrize(
    ("ports", "comb", "base", "ports8_tdm", "expected"),
    [
        ((1000, 1001), 2, 3, False, (3, 7)),
        ((1000, 1001, 1002, 1003), 4, 2, False, (2, 5, 8, 11)),
        ((1000, 1001, 1002, 1003), 8, 1, False, (1, 1, 4, 4)),
        (
            (1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007),
            2,
            0,
            True,
            (0, 2, 0, 2, 4, 6, 4, 6),
        ),
    ],
)
def test_srs_port_cyclic_shift_roster_matches_standard_mapping(
    ports, comb, base, ports8_tdm, expected
) -> None:
    assert nr_srs_port_cyclic_shift_indices(
        ports,
        transmission_comb=comb,
        base_cyclic_shift=base,
        ports8_tdm=ports8_tdm,
    ) == expected


def test_srs_port_cyclic_shift_roster_rejects_partial_or_invalid_ports() -> None:
    with pytest.raises(ValueError, match="complete configured"):
        nr_srs_port_cyclic_shift_indices(
            (1000, 1002), transmission_comb=2, base_cyclic_shift=0
        )


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
    assert (
        nr_srs_cyclic_shift_hopping_offset(radio_frame_index=132, **common)
        == first
    )
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
        nr_srs_cyclic_shift_hopping_offset(
            **common, hopping_subset=(True,) + (False,) * 11
        )
    with pytest.raises(ValueError, match="cannot both"):
        nr_srs_cyclic_shift_hopping_offset(
            **common,
            hopping_subset=(True, True) + (False,) * 10,
            hopping_finer_granularity=True,
        )
    with pytest.raises(ValueError, match="inside the slot"):
        nr_srs_cyclic_shift_hopping_offset(
            **{**common, "resource_symbol_index": 2}
        )


def test_srs_cyclic_shift_finer_granularity_adds_half_shift() -> None:
    ordinary = nr_srs_sequence_type1(
        72, group_index=5, transmission_comb=2
    )
    half_shifted = nr_srs_sequence_type1(
        72,
        group_index=5,
        transmission_comb=2,
        cyclic_shift_hopping_offset=1,
        hopping_finer_granularity=True,
    )
    n = np.arange(72)
    np.testing.assert_allclose(half_shifted, ordinary * np.exp(2j * np.pi * n / 16))
    with pytest.raises(ValueError, match="complete configured"):
        nr_srs_port_cyclic_shift_indices(
            (1000, 1001),
            transmission_comb=2,
            base_cyclic_shift=0,
            ports8_tdm=True,
        )


def test_srs_multiport_resource_mapping_preserves_port_axis_and_mask() -> None:
    sequences = nr_srs_port_sequences_type1(
        24,
        cyclic_shift_indices=(0, 1),
        srs_symbol_count=2,
        sequence_id=12,
        slot_index=0,
        start_symbol_index=12,
        hopping_mode="neither",
        transmission_comb=2,
    )
    result = map_nr_srs_multiport_resource(
        sequences,
        slot_symbol_count=14,
        subcarrier_count=48,
        start_position_offset=1,
        srs_symbol_count=2,
        comb_size=2,
        comb_offset=0,
        beta_srs=0.5,
    )

    assert isinstance(result, NRSRSMultiportResourceGrid)
    assert result.grid.shape == (2, 14, 48)
    assert result.mask.shape == result.grid.shape
    assert result.symbol_indices == (12, 13)
    assert int(np.count_nonzero(result.mask)) == 2 * 2 * 24
    np.testing.assert_array_equal(
        result.grid[0, 12, 0::2], 0.5 / np.sqrt(2) * sequences[0, 0]
    )
    np.testing.assert_array_equal(
        result.grid[1, 13, 0::2], 0.5 / np.sqrt(2) * sequences[1, 1]
    )
    assert not result.grid.flags.writeable
    assert not result.mask.flags.writeable


def test_srs_multiport_mapping_normalizes_four_port_three_tx_power() -> None:
    sequences = np.ones((4, 1, 24), dtype=np.complex128)
    common = {
        "slot_symbol_count": 14,
        "subcarrier_count": 48,
        "start_position_offset": 0,
        "srs_symbol_count": 1,
        "comb_size": 2,
        "comb_offset": 0,
    }

    result = map_nr_srs_multiport_resource(
        sequences, four_port_srs_3tx=True, **common
    )
    np.testing.assert_array_equal(
        result.grid[:, 13, 0::2],
        np.full((4, 24), 1 / np.sqrt(3), dtype=np.complex128),
    )
    with pytest.raises(ValueError, match="requires exactly four"):
        map_nr_srs_multiport_resource(sequences[:3], four_port_srs_3tx=True, **common)


def test_srs_eight_port_tdm_activity_matches_resource_local_symbol_pattern() -> None:
    ports = tuple(range(1000, 1008))
    activity = nr_srs_port_symbol_activity(
        ports, srs_symbol_count=4, ports8_tdm=True
    )
    assert activity == (
        (True, False, True, False),
        (True, False, True, False),
        (False, True, False, True),
        (False, True, False, True),
        (True, False, True, False),
        (True, False, True, False),
        (False, True, False, True),
        (False, True, False, True),
    )


@pytest.mark.parametrize(
    ("choice", "expected"),
    [
        (
            {"n2": {"combOffset-n2": 1, "cyclicShift-n2": 7}},
            NRSRSTransmissionCombConfiguration(2, 1, 7),
        ),
        (
            {"n4": {"combOffset-n4": 3, "cyclicShift-n4": 11}},
            NRSRSTransmissionCombConfiguration(4, 3, 11),
        ),
        (
            {"n8": {"combOffset-n8": 7, "cyclicShift-n8": 5}},
            NRSRSTransmissionCombConfiguration(8, 7, 5),
        ),
    ],
)
def test_srs_transmission_comb_adapter_resolves_rrc_choices(choice, expected) -> None:
    assert nr_srs_transmission_comb_from_rrc(choice) == expected


def test_srs_positioning_transmission_comb_adapter_handles_r16_names() -> None:
    assert nr_srs_transmission_comb_from_rrc(
        {"n8-r16": {"combOffset-n8-r16": 6, "cyclicShift-n8-r16": 4}},
        positioning_resource=True,
    ) == NRSRSTransmissionCombConfiguration(8, 6, 4)
    with pytest.raises(ValueError, match="choice must be n2, n4, or n8"):
        nr_srs_transmission_comb_from_rrc(
            {"n8": {"combOffset-n8": 0, "cyclicShift-n8": 0}},
            positioning_resource=True,
        )


@pytest.mark.parametrize(
    "choice",
    [
        {},
        {"n2": {"combOffset-n2": 0}},
        {"n2": {"combOffset-n2": 0, "cyclicShift-n2": 0, "extra": 1}},
        {"n4": {"combOffset-n4": True, "cyclicShift-n4": 0}},
        {"n8": {"combOffset-n8": 8, "cyclicShift-n8": 0}},
    ],
)
def test_srs_transmission_comb_adapter_rejects_malformed_decoded_choices(choice) -> None:
    with pytest.raises((TypeError, ValueError)):
        nr_srs_transmission_comb_from_rrc(choice)


@pytest.mark.parametrize(
    ("port_count", "expected"),
    [
        (None, NRSRSPortConfiguration((1000,), False)),
        ("port1", NRSRSPortConfiguration((1000,), False)),
        ("ports2", NRSRSPortConfiguration((1000, 1001), False)),
        ("ports4", NRSRSPortConfiguration((1000, 1001, 1002, 1003), False)),
    ],
)
def test_srs_rrc_port_configuration_resolves_standard_enums(port_count, expected) -> None:
    assert nr_srs_port_configuration_from_rrc(port_count) == expected


@pytest.mark.parametrize("port_mode", ["ports8", "ports8tdm"])
def test_srs_rrc_eight_port_configuration(port_mode) -> None:
    assert nr_srs_port_configuration_from_rrc(
        "ports4", nrof_srs_ports_n8=port_mode
    ) == NRSRSPortConfiguration(tuple(range(1000, 1008)), port_mode == "ports8tdm")


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


def test_srs_multiport_grid_rejects_inconsistent_mask() -> None:
    with pytest.raises(ValueError, match="mask disagrees"):
        NRSRSMultiportResourceGrid(
            grid=np.zeros((2, 14, 48), dtype=np.complex128),
            mask=np.zeros((2, 14, 48), dtype=np.bool_),
            symbol_indices=(13,),
            subcarrier_indices=(0,),
        )


def test_srs_multiport_mapper_rejects_wrong_rank_and_symbol_count() -> None:
    common = {
        "slot_symbol_count": 14,
        "subcarrier_count": 48,
        "start_position_offset": 1,
        "srs_symbol_count": 2,
        "comb_size": 2,
        "comb_offset": 0,
    }
    with pytest.raises(ValueError, match=r"shape \[ports, SRS symbols"):
        map_nr_srs_multiport_resource(np.ones((2, 24), dtype=np.complex128), **common)
    with pytest.raises(ValueError, match="exactly srs_symbol_count"):
        map_nr_srs_multiport_resource(
            np.ones((2, 1, 24), dtype=np.complex128), **common
        )


@pytest.mark.parametrize(
    (
        "sequence_length",
        "group_index",
        "base_sequence_index",
        "cyclic_shift_index",
        "transmission_comb",
    ),
    [(30, 17, 0, 6, 2), (36, 0, 0, 0, 2), (48, 3, 0, 7, 2), (72, 29, 1, 5, 8)],
)
def test_srs_low_papr_sequence_matches_independent_scalar_reference(
    sequence_length, group_index, base_sequence_index, cyclic_shift_index, transmission_comb
) -> None:
    shift_count = srs_cyclic_shift_count(transmission_comb)
    sequence = nr_low_papr_sequence_type1(
        sequence_length,
        group_index=group_index,
        base_sequence_index=base_sequence_index,
        cyclic_shift_index=cyclic_shift_index,
        cyclic_shift_count=shift_count,
    )
    expected = np.asarray(
        [
            np.exp(
                -1j
                * np.pi
                * (
                    group_index * (n + 1) * (n + 2) / 31
                    if sequence_length == 30
                    else _reference_low_papr_phase(
                        sequence_length, group_index, base_sequence_index, n
                    )
                )
                + 2j * np.pi * cyclic_shift_index * n / shift_count
            )
            for n in range(sequence_length)
        ],
        dtype=np.complex128,
    )

    np.testing.assert_allclose(sequence, expected, atol=5e-12, rtol=5e-12)
    np.testing.assert_allclose(np.abs(sequence), 1.0, atol=1e-12, rtol=0.0)


def _reference_low_papr_phase(sequence_length, group_index, base_sequence_index, n):
    zc_length = max(
        candidate
        for candidate in range(sequence_length - 1, 1, -1)
        if all(candidate % divisor for divisor in range(2, int(candidate**0.5) + 1))
    )
    q_numerator = zc_length * (group_index + 1)
    q_bar = (2 * q_numerator + 31) // 62
    q_sign = (-1) ** ((2 * q_numerator) // 31)
    q = q_bar + base_sequence_index * q_sign
    return q * (n % zc_length) * (n % zc_length + 1) / zc_length


@pytest.mark.parametrize(
    ("sequence_length", "group_index", "phase_row"),
    [
        (6, 0, (-3, -1, 3, 3, -1, -3)),
        (12, 29, (-3, 3, -3, 3, -3, -3, 3, -1, -1, 1, 3, -3)),
        (18, 0, (-1, 3, -1, -3, 3, 1, -3, -1, 3, -3, -1, -1, 1, 1, 1, -1, -1, -1)),
        (
            24,
            29,
            (3, -3, -1, 1, 3, -1, -1, -3, -1, 3, -1, -3, -1, -3, 3, -1, 3, 1, 1, -3, 3, -3, -3, -3),
        ),
    ],
)
def test_srs_short_type1_sequences_match_ts_phase_tables(
    sequence_length, group_index, phase_row
) -> None:
    cyclic_shift_index = 3
    cyclic_shift_count = 8
    sequence = nr_low_papr_sequence_type1(
        sequence_length,
        group_index=group_index,
        cyclic_shift_index=cyclic_shift_index,
        cyclic_shift_count=cyclic_shift_count,
    )
    indices = np.arange(sequence_length)
    expected = np.exp(
        1j * np.pi * np.asarray(phase_row) / 4
        + 2j * np.pi * cyclic_shift_index * indices / cyclic_shift_count
    )

    np.testing.assert_allclose(sequence, expected, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(np.abs(sequence), 1.0, atol=1e-12, rtol=0.0)


@pytest.mark.parametrize(
    ("args", "error", "message"),
    [
        ((32,), ValueError, "supported type-1 sequence lengths"),
        ((36,), TypeError, "group_index"),
        ((36,), ValueError, "group_index"),
        ((36,), ValueError, "cyclic_shift_index"),
        ((48,), ValueError, "sequence lengths below 72"),
    ],
)
def test_srs_low_papr_sequence_rejects_unsupported_parameters(args, error, message) -> None:
    kwargs = {"group_index": 0}
    if message == "group_index" and error is TypeError:
        kwargs["group_index"] = True
    elif message == "group_index":
        kwargs["group_index"] = 30
    elif message == "cyclic_shift_index":
        kwargs.update(cyclic_shift_index=8, cyclic_shift_count=8)
    elif message == "sequence lengths below 72":
        kwargs["base_sequence_index"] = 1
    with pytest.raises(error, match=message):
        nr_low_papr_sequence_type1(*args, **kwargs)


@pytest.mark.parametrize(("comb", "expected"), [(2, 8), (4, 12), (8, 6)])
def test_srs_cyclic_shift_count_matches_transmission_comb(comb, expected) -> None:
    assert srs_cyclic_shift_count(comb) == expected


@pytest.mark.parametrize(("comb", "error"), [(3, ValueError), (True, TypeError)])
def test_srs_cyclic_shift_count_rejects_invalid_comb(comb, error) -> None:
    with pytest.raises(error, match="transmission_comb"):
        srs_cyclic_shift_count(comb)


def test_srs_sequence_binds_cyclic_shift_count_to_comb() -> None:
    sequence = nr_srs_sequence_type1(
        48,
        group_index=3,
        base_sequence_index=0,
        cyclic_shift_index=7,
        transmission_comb=2,
    )
    explicit = nr_low_papr_sequence_type1(
        48,
        group_index=3,
        base_sequence_index=0,
        cyclic_shift_index=7,
        cyclic_shift_count=8,
    )
    np.testing.assert_array_equal(sequence, explicit)


def test_srs_sequence_rejects_shift_index_outside_comb_range() -> None:
    with pytest.raises(ValueError, match="cyclic_shift_index"):
        nr_srs_sequence_type1(
            48,
            group_index=3,
            cyclic_shift_index=8,
            transmission_comb=2,
        )


def _scalar_gold_bits(c_init: int, count: int) -> list[int]:
    x1 = [1] + [0] * (1600 + count + 30)
    x2 = [(c_init >> bit) & 1 for bit in range(31)] + [0] * (1600 + count)
    for index in range(1600 + count):
        x1[index + 31] = x1[index + 3] ^ x1[index]
        x2[index + 31] = x2[index + 3] ^ x2[index + 2] ^ x2[index + 1] ^ x2[index]
    return [x1[1600 + index] ^ x2[1600 + index] for index in range(count)]


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
    offset = 8 * (
        slot_symbol_count * slot_index + start_symbol_index + resource_symbol_index
    )
    bits = _scalar_gold_bits(sequence_id, offset + 8)
    hopping_value = sum(bits[offset + bit] << bit for bit in range(8))
    assert nr_srs_group_hopping_index(
        sequence_id,
        slot_index,
        start_symbol_index,
        resource_symbol_index,
        slot_symbol_count=slot_symbol_count,
    ) == (
        hopping_value + sequence_id
    ) % 30


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
    assert nr_srs_sequence_hopping_index(
        sequence_id,
        slot_index,
        start_symbol_index,
        resource_symbol_index,
        72,
        slot_symbol_count=slot_symbol_count,
    ) == expected


@pytest.mark.parametrize("sequence_length", [6, 12, 18, 24, 30, 36, 71])
def test_srs_sequence_hopping_is_disabled_below_72(sequence_length) -> None:
    assert (
        nr_srs_sequence_hopping_index(101, 7, 0, 2, sequence_length) == 0
    )


@pytest.mark.parametrize("hopping_mode", ["neither", "groupHopping"])
def test_srs_hopped_sequence_resolves_group_before_sequence_generation(hopping_mode) -> None:
    sequence_id = 71
    slot_index = 2
    start_symbol_index = 8
    resource_symbol_index = 3
    group_index = (
        sequence_id % 30
        if hopping_mode == "neither"
        else nr_srs_group_hopping_index(
            sequence_id, slot_index, start_symbol_index, resource_symbol_index
        )
    )
    sequence = nr_srs_hopped_sequence_type1(
        72,
        sequence_id=sequence_id,
        slot_index=slot_index,
        start_symbol_index=start_symbol_index,
        resource_symbol_index=resource_symbol_index,
        hopping_mode=hopping_mode,
        transmission_comb=4,
        cyclic_shift_index=4,
    )
    expected = nr_srs_sequence_type1(
        72,
        group_index=group_index,
        base_sequence_index=0,
        cyclic_shift_index=4,
        transmission_comb=4,
    )
    np.testing.assert_array_equal(sequence, expected)


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


@pytest.mark.parametrize(
    ("hopping_mode", "sequence_length", "slot_symbol_count"),
    [("neither", 24, 14), ("groupHopping", 48, 14), ("sequenceHopping", 96, 12)],
)
def test_srs_resource_sequence_rows_follow_each_resource_symbol_hop(
    hopping_mode, sequence_length, slot_symbol_count
) -> None:
    start_symbol_index = 10 if slot_symbol_count == 14 else 8
    rows = nr_srs_resource_sequences_type1(
        sequence_length,
        srs_symbol_count=4,
        sequence_id=91,
        slot_index=7,
        start_symbol_index=start_symbol_index,
        hopping_mode=hopping_mode,
        transmission_comb=2,
        cyclic_shift_index=3,
        slot_symbol_count=slot_symbol_count,
    )
    expected = np.stack(
        [
            nr_srs_hopped_sequence_type1(
                sequence_length,
                sequence_id=91,
                slot_index=7,
                start_symbol_index=start_symbol_index,
                resource_symbol_index=index,
                hopping_mode=hopping_mode,
                transmission_comb=2,
                cyclic_shift_index=3,
                slot_symbol_count=slot_symbol_count,
            )
            for index in range(4)
        ]
    )

    np.testing.assert_array_equal(rows, expected)


def test_srs_resource_sequence_rows_map_directly_to_each_resource_symbol() -> None:
    rows = nr_srs_resource_sequences_type1(
        6,
        srs_symbol_count=2,
        sequence_id=12,
        slot_index=0,
        start_symbol_index=12,
        hopping_mode="neither",
        transmission_comb=2,
    )
    mapped = map_nr_srs_resource(
        rows,
        slot_symbol_count=14,
        subcarrier_count=12,
        start_position_offset=1,
        srs_symbol_count=2,
        comb_size=2,
        comb_offset=0,
    )

    assert mapped.symbol_indices == (12, 13)
    np.testing.assert_array_equal(
        mapped.grid[
            np.asarray(mapped.symbol_indices)[:, None],
            list(mapped.subcarrier_indices),
        ],
        rows,
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
            ((arguments["radio_frame_index"] % 128) * arguments["slots_per_frame"]
             + arguments["slot_index"])
            * arguments["slot_symbol_count"]
            + arguments["start_symbol_index"]
            + resource_symbol
        )
        bits = _scalar_gold_bits(
            arguments["cyclic_shift_hopping_id"], 8 * symbol_offset + 8
        )[8 * symbol_offset :]
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


def test_srs_resource_sequences_apply_configured_cyclic_shift_subset() -> None:
    subset = (False, True, False, True, False, False, True, False)
    coordinates = {
        "transmission_comb": 2,
        "cyclic_shift_hopping_id": 913,
        "radio_frame_index": 4,
        "slot_index": 7,
        "slots_per_frame": 20,
        "slot_symbol_count": 14,
        "start_symbol_index": 9,
        "srs_symbol_count": 4,
        "hopping_subset": subset,
    }
    offsets = nr_srs_cyclic_shift_hopping_offsets(**coordinates)
    assert set(offsets) <= {1, 3, 6}
    automatic = nr_srs_resource_sequences_type1(
        72,
        sequence_id=91,
        slot_index=coordinates["slot_index"],
        start_symbol_index=coordinates["start_symbol_index"],
        srs_symbol_count=coordinates["srs_symbol_count"],
        hopping_mode="neither",
        transmission_comb=2,
        cyclic_shift_hopping_id=coordinates["cyclic_shift_hopping_id"],
        radio_frame_index=coordinates["radio_frame_index"],
        slots_per_frame=coordinates["slots_per_frame"],
        hopping_subset=subset,
    )
    explicit = nr_srs_resource_sequences_type1(
        72,
        sequence_id=91,
        slot_index=coordinates["slot_index"],
        start_symbol_index=coordinates["start_symbol_index"],
        srs_symbol_count=coordinates["srs_symbol_count"],
        hopping_mode="neither",
        transmission_comb=2,
        cyclic_shift_hopping_offsets=offsets,
    )
    np.testing.assert_array_equal(automatic, explicit)


def test_srs_mapper_places_one_sequence_on_end_relative_symbols_and_comb() -> None:
    sequence = np.exp(1j * np.arange(24) * 0.17)
    result = map_nr_srs_resource(
        sequence,
        slot_symbol_count=14,
        subcarrier_count=48,
        start_position_offset=3,
        srs_symbol_count=4,
        comb_size=2,
        comb_offset=1,
        beta_srs=0.5,
    )

    assert result.symbol_indices == (10, 11, 12, 13)
    assert result.subcarrier_indices == tuple(range(1, 48, 2))
    assert int(np.count_nonzero(result.mask)) == 4 * 24
    np.testing.assert_array_equal(result.grid[10, 1::2], 0.5 * sequence)
    np.testing.assert_array_equal(result.grid[13, 1::2], 0.5 * sequence)
    assert not result.mask[:, 0::2].any()
    assert not result.grid[0:10].any()
    assert not result.grid.flags.writeable
    assert not result.mask.flags.writeable


def test_srs_mapper_accepts_per_symbol_sequences_and_frequency_origin() -> None:
    sequence = np.arange(30, dtype=np.float64).reshape(2, 15).astype(np.complex128) + 1j
    result = map_nr_srs_resource(
        sequence,
        slot_symbol_count=12,
        subcarrier_count=64,
        start_position_offset=1,
        srs_symbol_count=2,
        comb_size=4,
        comb_offset=2,
        frequency_start_subcarrier=5,
    )

    assert result.symbol_indices == (10, 11)
    # The configured offset counts backward from the slot end, then the rows
    # map consecutively from the derived first SRS symbol.
    assert result.subcarrier_indices == tuple(range(7, 64, 4))
    np.testing.assert_array_equal(result.grid[10, list(result.subcarrier_indices)], sequence[0])
    np.testing.assert_array_equal(result.grid[11, list(result.subcarrier_indices)], sequence[1])


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"slot_symbol_count": 13}, ValueError, "slot_symbol_count"),
        ({"srs_symbol_count": 3}, ValueError, "SRS symbol count"),
        ({"start_position_offset": 1}, ValueError, "start_position_offset"),
        ({"comb_size": 3}, ValueError, "comb_size"),
        ({"comb_offset": 2}, ValueError, "comb_offset"),
        ({"frequency_start_subcarrier": 48}, ValueError, "no resource elements"),
        ({"allocated_bandwidth_subcarriers": 49}, ValueError, "allocated bandwidth"),
        ({"allocated_bandwidth_subcarriers": True}, TypeError, "must be an integer"),
        ({"beta_srs": float("nan")}, ValueError, "beta_srs"),
    ],
)
def test_srs_mapper_rejects_invalid_resource_configuration(overrides, error, message) -> None:
    config = {
        "slot_symbol_count": 14,
        "subcarrier_count": 48,
        "start_position_offset": 3,
        "srs_symbol_count": 4,
        "comb_size": 2,
        "comb_offset": 1,
    }
    config.update(overrides)
    with pytest.raises(error, match=message):
        map_nr_srs_resource(np.ones(24, dtype=np.complex128), **config)


def test_srs_mapper_rejects_sequence_shape_and_type_mismatch() -> None:
    common = {
        "slot_symbol_count": 14,
        "subcarrier_count": 48,
        "start_position_offset": 3,
        "srs_symbol_count": 4,
        "comb_size": 2,
        "comb_offset": 1,
    }
    with pytest.raises(TypeError, match="complex-valued"):
        map_nr_srs_resource(np.ones(24), **common)
    with pytest.raises(ValueError, match="sequence length"):
        map_nr_srs_resource(np.ones(23, dtype=np.complex128), **common)
    with pytest.raises(ValueError, match="exactly srs_symbol_count"):
        map_nr_srs_resource(np.ones((3, 24), dtype=np.complex128), **common)


def test_srs_grid_rejects_mask_that_does_not_match_declared_resource() -> None:
    grid = np.zeros((14, 48), dtype=np.complex128)
    mask = np.zeros((14, 48), dtype=np.bool_)
    with pytest.raises(ValueError, match="mask disagrees"):
        NRSRSResourceGrid(
            grid=grid,
            mask=mask,
            symbol_indices=(10,),
            subcarrier_indices=(1,),
        )


def test_srs_single_port_ls_estimator_recovers_per_re_channel() -> None:
    sequence = np.exp(1j * np.arange(24, dtype=np.float64) * 0.13)
    mapped = map_nr_srs_resource(
        sequence,
        slot_symbol_count=14,
        subcarrier_count=48,
        start_position_offset=2,
        srs_symbol_count=2,
        comb_size=2,
        comb_offset=1,
        beta_srs=0.7,
    )
    channel = np.exp(1j * np.arange(14 * 48).reshape(14, 48) * 0.003)
    result = nr_srs_single_port_ls_channel_estimate(mapped, mapped.grid * channel)

    np.testing.assert_allclose(result.channel_by_re[mapped.mask], channel[mapped.mask])
    assert result.symbol_indices == mapped.symbol_indices
    assert np.array_equal(result.mask, mapped.mask)
    assert not result.channel_by_re.flags.writeable
    assert not result.mask.flags.writeable
    assert np.count_nonzero(result.channel_by_re[~result.mask]) == 0


def test_srs_single_port_ls_estimator_recovers_hopped_resource_channel() -> None:
    allocation = nr_srs_fixed_frequency_allocation(
        0, transmission_comb=2, b_srs=0, b_hop=0, reference_subcarrier=0
    )
    mapped = map_nr_srs_hopped_resource(
        np.ones((2, allocation.bandwidth.sequence_length), dtype=np.complex128),
        frequency_allocations=(allocation,),
        slot_symbol_count=14,
        subcarrier_count=allocation.bandwidth.allocated_bandwidth_subcarriers,
        start_position_offset=1,
        srs_symbol_count=2,
    )
    channel = np.full(mapped.grid.shape, 0.4 - 0.2j, dtype=np.complex128)
    result = nr_srs_single_port_ls_channel_estimate(mapped, mapped.grid * channel)

    np.testing.assert_allclose(result.channel_by_re[result.mask], 0.4 - 0.2j)


def test_srs_multiport_ls_estimator_separates_ports_and_reports_noise_covariance() -> None:
    sequences = np.ones((2, 2, 24), dtype=np.complex128)
    sequences[1, 1] *= -1.0
    mapped = map_nr_srs_multiport_resource(
        sequences,
        slot_symbol_count=14,
        subcarrier_count=48,
        start_position_offset=2,
        srs_symbol_count=2,
        comb_size=2,
        comb_offset=0,
    )
    channel = np.empty((2, 2, 48), dtype=np.complex128)
    channel[:, 0, :] = 0.4 + 0.2j
    channel[:, 1, :] = -0.1 + 0.3j
    received = np.einsum("rpk,psk->rsk", channel, mapped.grid)
    noise = np.array([0.01, 0.04], dtype=np.float64)

    result = nr_srs_multiport_ls_channel_estimate(
        mapped, received, noise_variance_by_rx_ant=noise
    )

    active_subcarriers = np.any(mapped.mask, axis=(0, 1))
    np.testing.assert_array_equal(result.valid_subcarrier_mask, active_subcarriers)
    np.testing.assert_allclose(
        result.channel_by_rx_port_subcarrier[:, :, active_subcarriers],
        channel[:, :, active_subcarriers],
    )
    design = mapped.grid[:, mapped.symbol_indices[0] : mapped.symbol_indices[-1] + 1, 0].T
    expected_base_covariance = np.linalg.inv(design.conj().T @ design)
    np.testing.assert_allclose(
        result.noise_covariance_by_rx_subcarrier[:, active_subcarriers],
        np.broadcast_to(
            noise[:, None, None, None] * expected_base_covariance[None, None, :, :],
            result.noise_covariance_by_rx_subcarrier[:, active_subcarriers].shape,
        ),
        atol=1e-18,
    )
    assert set(result.reason_codes) <= {None, "not_occupied"}
    assert not result.channel_by_rx_port_subcarrier.flags.writeable
    assert not result.noise_covariance_by_rx_subcarrier.flags.writeable


def test_srs_multiport_weighted_ls_uses_per_symbol_noise_variance() -> None:
    sequences = np.ones((2, 4, 24), dtype=np.complex128)
    sequences[1, 1::2] *= -1.0
    mapped = map_nr_srs_multiport_resource(
        sequences,
        slot_symbol_count=14,
        subcarrier_count=48,
        start_position_offset=3,
        srs_symbol_count=4,
        comb_size=2,
        comb_offset=0,
    )
    channel = np.empty((2, 2, 48), dtype=np.complex128)
    channel[:, 0, :] = 0.4 + 0.2j
    channel[:, 1, :] = -0.1 + 0.3j
    received = np.einsum("rpk,psk->rsk", channel, mapped.grid)
    residuals = np.asarray(
        [
            [0.1 + 0.2j, -0.3 + 0.05j],
            [-0.2 + 0.1j, 0.05 + 0.25j],
            [0.3 - 0.1j, 0.2 - 0.15j],
            [-0.05 + 0.3j, -0.25 - 0.1j],
        ]
    )
    for symbol_index, residual in zip(mapped.symbol_indices, residuals, strict=True):
        received[:, symbol_index, :] += residual[:, None]
    symbol_noise = np.ones((2, 14), dtype=np.float64)
    symbol_noise[0, list(mapped.symbol_indices)] = [0.01, 5.0, 0.02, 3.0]
    symbol_noise[1, list(mapped.symbol_indices)] = [5.0, 0.01, 3.0, 0.02]

    result = nr_srs_multiport_ls_channel_estimate(
        mapped,
        received,
        noise_variance_by_rx_symbol=symbol_noise,
    )

    active_subcarrier = int(np.flatnonzero(np.any(mapped.mask, axis=(0, 1)))[0])
    symbols = np.asarray(mapped.symbol_indices)
    design = mapped.grid[:, symbols, active_subcarrier].T
    occupied = np.any(mapped.mask[:, symbols, active_subcarrier], axis=0)
    design = design[occupied]
    observed = received[:, symbols[occupied], active_subcarrier].T
    expected = np.empty((2, 2), dtype=np.complex128)
    expected_covariance = np.empty((2, 2, 2), dtype=np.complex128)
    for rx_index in range(2):
        weights = 1.0 / symbol_noise[rx_index, symbols[occupied]]
        gram = design.conj().T @ (weights[:, None] * design)
        expected_covariance[rx_index] = np.linalg.inv(gram)
        expected[rx_index] = expected_covariance[rx_index] @ (
            design.conj().T @ (weights * observed[:, rx_index])
        )
    ordinary_least_squares = np.linalg.lstsq(design, observed, rcond=None)[0].T
    assert not np.allclose(expected, ordinary_least_squares)

    np.testing.assert_allclose(
        result.channel_by_rx_port_subcarrier[:, :, active_subcarrier],
        expected,
    )
    for rx_index in range(2):
        np.testing.assert_allclose(
            result.noise_covariance_by_rx_subcarrier[
                rx_index, active_subcarrier
            ],
            expected_covariance[rx_index],
        )
    np.testing.assert_array_equal(result.noise_variance_by_rx_symbol, symbol_noise)
    assert not result.noise_variance_by_rx_symbol.flags.writeable


def test_srs_multiport_ls_estimator_marks_rank_deficient_port_signatures() -> None:
    mapped = map_nr_srs_multiport_resource(
        np.ones((2, 2, 24), dtype=np.complex128),
        slot_symbol_count=14,
        subcarrier_count=48,
        start_position_offset=2,
        srs_symbol_count=2,
        comb_size=2,
        comb_offset=0,
    )
    received = np.ones((1, 14, 48), dtype=np.complex128)

    result = nr_srs_multiport_ls_channel_estimate(
        mapped, received, noise_variance_by_rx_ant=0.01
    )

    active_subcarriers = np.any(mapped.mask, axis=(0, 1))
    assert not result.valid_subcarrier_mask[active_subcarriers].any()
    assert set(result.reason_codes[i] for i in np.flatnonzero(active_subcarriers)) == {
        "rank_deficient_port_signature"
    }
    assert np.count_nonzero(result.channel_by_rx_port_subcarrier) == 0


def test_srs_multiport_ls_estimator_rejects_noise_and_receive_shape_errors() -> None:
    mapped = map_nr_srs_multiport_resource(
        np.ones((2, 2, 24), dtype=np.complex128),
        slot_symbol_count=14,
        subcarrier_count=48,
        start_position_offset=2,
        srs_symbol_count=2,
        comb_size=2,
        comb_offset=0,
    )
    received = np.ones((1, 14, 48), dtype=np.complex128)
    with pytest.raises(ValueError, match="symbol/subcarrier axes"):
        nr_srs_multiport_ls_channel_estimate(
            mapped, received[:, :-1], noise_variance_by_rx_ant=0.01
        )
    with pytest.raises(ValueError, match="finite and positive"):
        nr_srs_multiport_ls_channel_estimate(
            mapped, received, noise_variance_by_rx_ant=0.0
        )
    with pytest.raises(TypeError, match="real-valued"):
        nr_srs_multiport_ls_channel_estimate(
            mapped, received, noise_variance_by_rx_ant=1.0 + 0.0j
        )
    symbol_noise = np.ones((1, 14), dtype=np.float64)
    with pytest.raises(TypeError, match="real-valued"):
        nr_srs_multiport_ls_channel_estimate(
            mapped, received, noise_variance_by_rx_symbol=symbol_noise.astype(np.complex128)
        )
    with pytest.raises(ValueError, match="shape"):
        nr_srs_multiport_ls_channel_estimate(
            mapped, received, noise_variance_by_rx_symbol=np.ones((1, 13))
        )
    with pytest.raises(ValueError, match="finite and positive"):
        nr_srs_multiport_ls_channel_estimate(
            mapped, received, noise_variance_by_rx_symbol=np.zeros((1, 14))
        )
    with pytest.raises(ValueError, match="reciprocals must be representable"):
        nr_srs_multiport_ls_channel_estimate(
            mapped,
            received,
            noise_variance_by_rx_symbol=np.full((1, 14), np.nextafter(0.0, 1.0)),
        )
    with pytest.raises(ValueError, match="not both"):
        nr_srs_multiport_ls_channel_estimate(
            mapped,
            received,
            noise_variance_by_rx_ant=0.01,
            noise_variance_by_rx_symbol=np.ones((1, 14)),
        )


@pytest.mark.parametrize(
    ("received", "message"),
    [
        (np.ones((14, 48)), "complex"),
        (np.ones((12, 48), dtype=np.complex128), "shape"),
        (np.full((14, 48), np.nan + 0j), "finite"),
    ],
)
def test_srs_single_port_ls_estimator_rejects_invalid_receive_grid(
    received, message
) -> None:
    mapped = map_nr_srs_resource(
        np.ones(24, dtype=np.complex128),
        slot_symbol_count=14,
        subcarrier_count=48,
        start_position_offset=1,
        srs_symbol_count=1,
        comb_size=2,
        comb_offset=0,
    )
    with pytest.raises(ValueError, match=message):
        nr_srs_single_port_ls_channel_estimate(mapped, received)


def test_srs_intercell_overlap_audit_counts_shared_resource_elements_only() -> None:
    common = nr_srs_fixed_frequency_allocation(
        0,
        transmission_comb=2,
        b_srs=0,
        b_hop=0,
        comb_offset=0,
        reference_subcarrier=24,
    )
    disjoint_comb = nr_srs_fixed_frequency_allocation(
        0,
        transmission_comb=2,
        b_srs=0,
        b_hop=0,
        comb_offset=1,
        reference_subcarrier=24,
    )

    def occurrence(
        cell_id: str,
        allocation: NRSRSFixedFrequencyAllocation,
        *,
        slot: int = 4,
        symbol: int = 12,
    ) -> NRSRSScheduledResourceOccurrence:
        return NRSRSScheduledResourceOccurrence(
            cell_id=cell_id,
            resource_set_id=0,
            resource_id=0,
            grid_id="carrier-a-grid",
            carrier_hz=3.5e9,
            numerology=1,
            subcarrier_spacing_hz=30e3,
            target_slot_index=slot,
            slot_symbol_count=14,
            start_symbol_index=symbol,
            srs_symbol_count=1,
            allocations=(allocation,),
        )

    report = audit_nr_srs_intercell_re_overlap(
        [occurrence("cell-a", common), occurrence("cell-b", common)]
    )
    assert report["status"] == "INTERCELL_RE_OVERLAP_DETECTED"
    assert report["shared_intercell_resource_element_count"] == common.bandwidth.sequence_length
    assert report["pairwise_resource_overlaps"][0]["shared_resource_element_count"] == (
        common.bandwidth.sequence_length
    )
    assert "not proof of interference" in report["limitations"][0]
    jsonschema = pytest.importorskip("jsonschema")
    schema_path = (
        Path(__file__).resolve().parents[2]
        / "datasets/schema/nr-srs-intercell-re-overlap.v1.schema.json"
    )
    jsonschema.Draft202012Validator(
        json.loads(schema_path.read_text(encoding="utf-8"))
    ).validate(report)

    no_overlap = audit_nr_srs_intercell_re_overlap(
        [
            occurrence("cell-a", common),
            occurrence("cell-b", disjoint_comb),
            occurrence("cell-c", common, slot=5),
        ]
    )
    assert no_overlap["status"] == "NO_INTERCELL_RE_OVERLAP"
    assert no_overlap["collision_coordinates"] == []


def test_srs_intercell_overlap_audit_rejects_incomparable_grid_contexts() -> None:
    allocation = nr_srs_fixed_frequency_allocation(
        0, transmission_comb=2, b_srs=0, b_hop=0
    )
    base = {
        "resource_set_id": 0,
        "resource_id": 0,
        "grid_id": "carrier-a-grid",
        "carrier_hz": 3.5e9,
        "numerology": 1,
        "subcarrier_spacing_hz": 30e3,
        "target_slot_index": 4,
        "slot_symbol_count": 14,
        "start_symbol_index": 12,
        "srs_symbol_count": 1,
        "allocations": (allocation,),
    }
    first = NRSRSScheduledResourceOccurrence(cell_id="cell-a", **base)
    second = NRSRSScheduledResourceOccurrence(
        cell_id="cell-b", **{**base, "grid_id": "different-grid"}
    )
    with pytest.raises(ValueError, match="single common grid"):
        audit_nr_srs_intercell_re_overlap([first, second])
