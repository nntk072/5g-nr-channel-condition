# integrations.sionna.srs.srs_constants

Public SRS constants and resource-grid types.

## Purpose

This package is one module split into concept modules. It keeps the public names of the original module, so callers import from `integrations.sionna.srs.srs_constants` unchanged.

## Modules

| Module | Contents |
| --- | --- |
| `channel_estimates.py` | SRS channel-estimate records for single and multiport measurements |
| `constants.py` | SRS configuration constants and symbol-count tables |
| `dci_0_3.py` | DCI format 0_3 SRS request fields, widths, and scheduled cells |
| `ports_and_bwp.py` | SRS port configuration and bandwidth-part selection records |
| `resource_grids.py` | SRS resource-grid types for single, hopped, and positioning allocations |
| `sequences.py` | SRS Gold-sequence, hopping, and prime helpers |

## Public API (38 names)

Re-exported from `__init__.py` through `__all__`:

`_MAX_SRS_INTERCELL_AUDIT_COORDINATES`, `_MAX_SRS_INTERCELL_AUDIT_RESOURCE_ELEMENTS`, `_SRS_BANDWIDTH_TABLE`, `_SRS_COMB_SIZES`, `_SRS_CYCLIC_SHIFT_COUNTS`, `_SRS_PERIODICITY_EXT_OFFSETS`, `_SRS_PERIODICITY_OFFSETS`, `_SRS_POSITIONING_SYMBOL_OFFSETS`, `_SRS_REQUEST_DCI_FORMATS`, `_SRS_SYMBOL_COUNTS`, `_DCI_0_3_CONFIG_DERIVED_FIELDS`, `_DCI_0_3_FIELD_ORDER`, `_nr_srs_dci_0_3_antenna_ports_width`, `_nr_srs_dci_0_3_pdcch_monitoring_adaptation_width`, `_nr_srs_dci_0_3_precoding_width`, `_nr_srs_dci_0_3_scell_dormancy_width`, `_nr_srs_dci_0_3_sri_width`, `nr_srs_aperiodic_target_slot_index`, `nr_srs_dci_0_3_frequency_assignment_widths`, `nr_srs_dci_0_3_request_list_index_from_bits`, `nr_srs_dci_0_3_scheduled_cell_ids`, `NRSRSBWPFrequencyBounds`, `NRSRSBWPSelection`, `NRSRSPortConfiguration`, `_LOW_PAPR_SHORT_PHASES`, `_largest_prime_below`, `_normalized_multiport_beta`, `_nr_gold_bits`, `_srs_gold_bit_offset`, `_validate_srs_hopping_position`, `NRSRSHoppedMultiportResourceGrid`, `NRSRSHoppedResourceGrid`, `NRSRSMultiportResourceGrid`, `NRSRSPositioningMultiportResourceGrid`, `NRSRSPositioningResourceGrid`, `NRSRSResourceGrid`, `NRSRSMultiportChannelEstimate`, `NRSRSSinglePortChannelEstimate`

## Invariants

- Public names and behaviour are unchanged by the split; tests import the package, not the modules.
- Module-level imports point down the layers. Any cycle is a defect: `tests/architecture` checks this.

