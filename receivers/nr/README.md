# receivers.nr

Standards-oriented receiver contracts.

## Purpose

Framework-independent contracts for NR PUSCH receiver observations. They cover the intra-slot frequency-hopping DM-RS plan and grids, and the composition and de-hopping of hopped slots.

## Modules

| Module | Contents |
| --- | --- |
| `pusch.py` | Framework-independent contracts for NR PUSCH receiver observations |

## Public API (19 names)

Re-exported from `__init__.py` through `__all__`:

`NRPUSCHConfig`, `NRChannelEstimate`, `NRPUSCHDehoppedGrid`, `NRPUSCHFrequencyHopGrid`, `NRPUSCHIntraSlotHopDMRSPlan`, `NRPUSCHIntraSlotHopDMRSGrid`, `NRPUSCHHoppedAllocationGrid`, `NRPUSCHHoppedEstimateDemap`, `compose_pusch_intra_slot_hops`, `build_pusch_intra_slot_hop_dmrs_grid`, `map_pusch_intra_slot_hop_data_symbols`, `verify_pusch_intra_slot_hop_allocation_grid`, `derive_pusch_intra_slot_hop_dmrs_plan`, `estimate_demap_pusch_intra_slot_hop`, `dehop_pusch_intra_slot_received_grid`, `verify_pusch_intra_slot_hops`, `verify_pusch_intra_slot_dehopped_grid`, `verify_pusch_intra_slot_hop_dmrs_plan`, `verify_pusch_intra_slot_hop_dmrs_grid`

## Invariants

- Public names and behaviour are unchanged by the split; tests import the package, not the modules.
- Module-level imports point down the layers. Any cycle is a defect: `tests/architecture` checks this.

