# integrations.sionna.srs

NR SRS configuration and resource scheduling, split by concept.

## Purpose

This package is one module split into concept modules. It keeps the public names of the original module, so callers import from `integrations.sionna.srs` unchanged.

## Modules

| Module | Contents |
| --- | --- |
| `srs_bwp_schedules.py` | NR SRS bandwidth-part and cell-level schedules |
| `srs_configuration.py` | NR SRS configuration: comb, cyclic shift, bandwidth, and frequency allocation |
| `srs_dci_multicell.py` | NR SRS DCI 0_3 multicell aperiodic scheduling |
| `srs_lookups.py` | Lookups that derive SRS parameters from constants |
| `srs_resource_schedules.py` | NR SRS resource-set schedules and intercell resource audits |

## Public API (95 names)

Re-exported from `__init__.py` through `__all__`:

`NRSRSBWPFrequencyBounds`, `NRSRSBWPSelection`, `NRSRSHoppedMultiportResourceGrid`, `NRSRSHoppedResourceGrid`, `NRSRSMultiportChannelEstimate`, `NRSRSMultiportResourceGrid`, `NRSRSPortConfiguration`, `NRSRSPositioningMultiportResourceGrid`, `NRSRSPositioningResourceGrid`, `NRSRSResourceGrid`, `NRSRSSinglePortChannelEstimate`, `_DCI_0_3_CONFIG_DERIVED_FIELDS`, `_DCI_0_3_FIELD_ORDER`, `_LOW_PAPR_SHORT_PHASES`, `_MAX_SRS_INTERCELL_AUDIT_COORDINATES`, `_MAX_SRS_INTERCELL_AUDIT_RESOURCE_ELEMENTS`, `_SRS_BANDWIDTH_TABLE`, `_SRS_COMB_SIZES`, `_SRS_CYCLIC_SHIFT_COUNTS`, `_SRS_PERIODICITY_EXT_OFFSETS`, `_SRS_PERIODICITY_OFFSETS`, `_SRS_POSITIONING_SYMBOL_OFFSETS`, `_SRS_REQUEST_DCI_FORMATS`, `_SRS_SYMBOL_COUNTS`, `_largest_prime_below`, `_normalized_multiport_beta`, `_nr_gold_bits`, `_nr_srs_dci_0_3_antenna_ports_width`, `_nr_srs_dci_0_3_pdcch_monitoring_adaptation_width`, `_nr_srs_dci_0_3_precoding_width`, `_nr_srs_dci_0_3_scell_dormancy_width`, `_nr_srs_dci_0_3_sri_width`, `_srs_gold_bit_offset`, `_validate_srs_hopping_position`, `nr_srs_aperiodic_target_slot_index`, `nr_srs_dci_0_3_frequency_assignment_widths`, `nr_srs_dci_0_3_request_list_index_from_bits`, `nr_srs_dci_0_3_scheduled_cell_ids`, `map_nr_srs_resource`, `nr_dci_0_3_field_values_from_layout`, `nr_low_papr_sequence_type1`, `nr_srs_dci_0_3_ul_bwp_selection`, `nr_srs_frequency_position_indices`, `nr_srs_group_hopping_index`, `nr_srs_multiport_ls_channel_estimate`, `nr_srs_periodicity_and_offset_slots`, `nr_srs_port_configuration_from_rrc`, `nr_srs_port_symbol_activity`, `nr_srs_positioning_symbol_offset`, `nr_srs_sequence_hopping_index`, `nr_srs_single_port_ls_channel_estimate`, `nr_srs_transmission_counter`, `nr_srs_triggered_resource_set_ids`, `srs_cyclic_shift_count`, `NRSRSBandwidthConfiguration`, `NRSRSFixedFrequencyAllocation`, `NRSRSTransmissionCombConfiguration`, `map_nr_srs_multiport_resource`, `map_nr_srs_positioning_multiport_resource`, `map_nr_srs_positioning_resource`, `nr_srs_cyclic_shift_hopping_offset`, `nr_srs_cyclic_shift_hopping_offsets`, `nr_srs_dci_0_3_field_widths_from_configuration`, `nr_srs_dci_0_3_request_list_index_from_layout`, `nr_srs_hopped_sequence_type1`, `nr_srs_multicell_request_resource_set_ids`, `nr_srs_port_cyclic_shift_indices`, `nr_srs_sequence_type1`, `nr_srs_transmission_comb_from_rrc`, `nr_srs_type1_bandwidth_configuration`, `NRSRSAperiodicResourceSchedule`, `NRSRSPeriodicResourceSetSchedule`, `NRSRSResourceSetAperiodicSchedule`, `NRSRSScheduledResourceOccurrence`, `audit_nr_srs_intercell_re_overlap`, `map_nr_srs_hopped_multiport_resource`, `map_nr_srs_hopped_resource`, `nr_srs_fixed_frequency_allocation`, `nr_srs_port_sequences_type1`, `nr_srs_resource_frequency_allocations`, `nr_srs_resource_sequences_type1`, `NRSRSBWPCellPeriodicResourceSetSchedules`, `NRSRSBWPCellResourceSetSchedules`, `NRSRSBWPPeriodicResourceSetSchedule`, `nr_srs_aperiodic_resource_schedule`, `nr_srs_aperiodic_resource_set_schedules`, `nr_srs_multicell_aperiodic_resource_set_schedules`, `nr_srs_periodic_resource_schedule`, `nr_srs_periodic_resource_set_schedule`, `nr_srs_scheduled_port_sequences_type1`, `nr_srs_dci_0_3_multicell_aperiodic_resource_set_schedules`, `nr_srs_dci_0_3_multicell_aperiodic_resource_set_schedules_by_bwp`, `nr_srs_dci_0_3_multicell_aperiodic_resource_set_schedules_from_bits`, `nr_srs_multicell_periodic_resource_set_schedules_by_bwp`, `nr_srs_periodic_resource_set_schedule_by_bwp`

## Invariants

- Public names and behaviour are unchanged by the split; tests import the package, not the modules.
- Module-level imports point down the layers. Any cycle is a defect: `tests/architecture` checks this.
- Files that retained evidence records by path are listed in `docs/reference/digest-rebindings.json`.

