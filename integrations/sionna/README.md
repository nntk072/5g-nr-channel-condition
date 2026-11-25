# integrations.sionna

Lazy Sionna PHY and ray-tracing adapters.

## Purpose

This package holds the adapters between the project and Sionna PHY and RT 2.2. Each adapter is gated by the pinned Sionna version, and the native checks run only on hosts that provide the backend.

## Modules

| Module | Contents |
| --- | --- |
| `array_layouts.py` | Extract static Sionna RT scene devices and arrays into UMCD contracts |
| `backend.py` | Runtime selection for Sionna RT's Mitsuba/Dr.Jit backends |
| `candidate_audit.py` | Candidate-buffer accounting for the pinned Sionna RT PathSolver |
| `cdl.py` | Pinned Sionna TR 38.901 CDL-to-NR frequency-response adapter |
| `cp_margin.py` | Fixed-payload PUSCH checks for multipath delay relative to the cyclic prefix |
| `d1_generation_gate.py` | Fail-closed workflow gate for bounded pilot D1 RT generation |
| `los_oracle.py` | Scene-geometry LOS labels independent of Sionna's path search |
| `network_receiver.py` | Bounded sample-domain network waveform handoff to the NR PUSCH receiver |
| `network_sinr.py` | Bounded, replay-identifiable resource-grid network SINR workflow |
| `noise.py` | Shared proper-complex receiver-noise generation for Sionna integrations |
| `nr_prs.py` | NR downlink positioning reference-signal sequence, mapping, and timing view |
| `nr_pusch.py` | Optional Sionna PHY adapter for NR PUSCH DMRS channel estimation |
| `nr_sync.py` | Small NR PSS/SSS acquisition reference for controlled receiver tests |
| `nr_timing.py` | 3GPP NR normal-cyclic-prefix OFDM symbol timing |
| `ofdm_impairments.py` | Sample-domain residual synchronization impairments for native PHY checks |
| `path_lineage.py` | Opt-in, read-only stage snapshots for a pinned Sionna RT PathSolver call |
| `path_oracle.py` | Conversion from native Sionna RT Paths to the unified path-truth table |
| `path_power_threshold.py` | Explicit post-solver relative path-power threshold masks |
| `pbch_acquisition.py` | Blind PSS/SSS acquisition and finite PBCH occasion search |
| `pbch_bch.py` | PBCH BCH CRC, Polar coding, and 864-bit rate matching |
| `pbch_data.py` | PBCH data-bit scrambling and QPSK modulation for one SS/PBCH block |
| `pbch_dmrs.py` | PBCH demodulation-reference sequence and resource map for an SS/PBCH block |
| `pbch_mib.py` | NR BCCH-BCH MIB and empty message-class alternative bit parsing |
| `pbch_ofdm_receiver.py` | Known-occasion PBCH OFDM demodulation and channel-affected decoding |
| `pbch_payload.py` | PBCH payload generation, interleaving, scrambling, and recovery |
| `random.py` | Small helpers for reproducible calls into Sionna's shared RNG registry |
| `rt_cache.py` | Immutable, integrity-checked cache for complete static Sionna RT datasets |
| `rt_dmrs.py` | Build NR DMRS observations from frequency-domain Sionna RT channels |
| `rt_identity.py` | Complete identities for reproducible Sionna RT generation runs |
| `rt_network.py` | Map whole Sionna RT solve outputs onto UMCD links and episode frames |
| `rt_to_nr.py` | Map native Sionna RT paths onto a Sionna NR resource grid |
| `rt_trajectory.py` | Build timestamped NR receiver observations from retraced Sionna RT paths |
| `scenario_gbsm.py` | Explicit-topology adapters for 3GPP TR 38.901 scenario GBSMs |
| `scene_assets.py` | Content identities for Mitsuba XML scenes and their referenced assets |
| `srs_measurements.py` | SRS positioning measurement primitives and strict report contracts |
| `ssb_grid.py` | Resource-grid assembly for one known NR SS/PBCH-block occasion |
| `ssb_ofdm.py` | NR SS/PBCH OFDM waveform construction from one block to a radio frame |
| `ssb_timing.py` | NR SSB candidate symbol positions from 3GPP TS 38.213 clause 4.1 |
| `standardized_pusch.py` | PUSCH/DMRS observations over immutable standardized TDL/CDL channel draws |
| `tdl.py` | Pinned Sionna TR 38.901 TDL-to-NR frequency-response adapter |
| `tdl_pusch.py` | Controlled Sionna TDL → NR PUSCH frequency-domain link generation |

## Public API (195 names)

Re-exported from `__init__.py` through `__all__`:

`GridConfig`, `CDLArrayConfig`, `NRGridChannel`, `NRPRSResourceConfiguration`, `NRPRSDLRSTDMeasurement`, `NRPRSDelayCFOProfile`, `NRPRSTDReport`, `NRPRSTDReportElement`, `NRPRSResourceGrid`, `NRPRSResourceOccasion`, `NRPRSResourceSetGrid`, `NRPRSResourceSetTiming`, `NRPRSTimingQuality`, `NRPRSTimingProfile`, `build_nr_dl_tdoa_measurement_report`, `RayScene`, `RTBackendSelection`, `RaytraceGenerationError`, `RTCacheConflictError`, `SionnaPUSCHLSReceiver`, `NRWaveformImpairments`, `NRSSBResourceGrid`, `NRSSBHalfFrameWaveform`, `NRSSBRadioFrameWaveform`, `NRSSBWaveformConfig`, `NRSRSResourceGrid`, `NetworkGridComposition`, `NetworkLinkBudget`, `NetworkMIMOChannelPropagation`, `NetworkResourceSINR`, `NetworkSINR`, `NetworkTransmitterWaveform`, `NetworkWaveformComposition`, `NetworkPUSCHDecode`, `NoisyNetworkGrid`, `NoisyNetworkWaveform`, `NRSRSMultiportChannelEstimate`, `NRSRSBandwidthConfiguration`, `NRSRSFixedFrequencyAllocation`, `NRSRSPortConfiguration`, `NRSRSTransmissionCombConfiguration`, `NRSRSAperiodicResourceSchedule`, `NRSRSPeriodicResourceSetSchedule`, `NRSRSResourceSetAperiodicSchedule`, `NRSRSBWPSelection`, `NRSRSBWPCellResourceSetSchedules`, `NRSRSBWPPeriodicResourceSetSchedule`, `NRSRSBWPCellPeriodicResourceSetSchedules`, `NRSRSBWPFrequencyBounds`, `NRSRSHoppedMultiportResourceGrid`, `NRSRSHoppedResourceGrid`, `NRSRSMultiportResourceGrid`, `NRSRSPositioningMultiportResourceGrid`, `NRSRSPositioningResourceGrid`, `NRSRSSinglePortChannelEstimate`, `PBCHDMRSResourceMap`, `PBCHPayloadRecovery`, `PBCHMIB`, `PBCHBCCHMessage`, `PBCHMessageClassExtension`, `PBCHOFDMReception`, `PBCHAcquisitionError`, `PBCHAcquisitionReception`, `SionnaCDLChannel`, `SionnaScenarioGBSMChannel`, `SionnaTDLChannel`, `SionnaTDLPUSCHBatch`, `apply_integer_timing_offset`, `apply_network_link_budget`, `apply_residual_cfo`, `apply_time_varying_multipath`, `activate_rt_backend`, `independent_mesh_segment_visibility`, `add_independent_receiver_awgn`, `add_independent_receiver_waveform_awgn`, `apply_network_transmitter_iq_imbalance`, `compose_downlink_resource_grids`, `compose_received_waveforms`, `network_resource_sinr`, `generate_raytraced`, `generate_raytraced_cached`, `generate_standardized`, `generate_cdl_frequency_response`, `generate_scenario_gbsm_frequency_response`, `load_rt_dataset`, `run_static_cp_margin_experiment`, `select_rt_backend`, `generate_tdl_frequency_response`, `generate_tdl_pusch_batch`, `tdl_pusch_records`, `store_rt_dataset`, `paths_to_nr_ofdm_channel`, `map_nr_srs_hopped_multiport_resource`, `network_sinr`, `decode_noisy_network_pusch`, `NetworkPUSCHWorkflowResult`, `NetworkTransmitterPUSCHWorkflowResult`, `run_network_pusch_receiver`, `run_network_pusch_from_transmitters`, `propagate_network_transmit_waveforms`, `NetworkGridSINRWorkflowResult`, `run_network_grid_sinr`, `map_nr_dl_prs_resource`, `map_nr_dl_prs_resource_set`, `measure_nr_dl_rstd`, `map_nr_srs_hopped_resource`, `map_nr_srs_multiport_resource`, `map_nr_srs_resource`, `measure_ul_srs_rsrp`, `measure_ul_srs_rsrp_from_grid`, `measure_ul_srs_rsrp_from_noisy_network_grid`, `report_gnb_rx_tx_time_difference`, `report_ul_aoa`, `report_ul_rtoa`, `report_ul_srs_rsrpp`, `report_ul_srs_tdct_tdcp`, `map_nr_srs_positioning_multiport_resource`, `map_nr_srs_positioning_resource`, `nr_low_papr_sequence_type1`, `nr_srs_group_hopping_index`, `nr_srs_single_port_ls_channel_estimate`, `nr_srs_multiport_ls_channel_estimate`, `nr_srs_fixed_frequency_allocation`, `nr_srs_frequency_position_indices`, `nr_srs_hopped_sequence_type1`, `nr_srs_aperiodic_target_slot_index`, `nr_srs_aperiodic_resource_schedule`, `nr_srs_aperiodic_resource_set_schedules`, `nr_srs_multicell_request_resource_set_ids`, `nr_srs_multicell_aperiodic_resource_set_schedules`, `nr_srs_dci_0_3_scheduled_cell_ids`, `nr_srs_dci_0_3_request_list_index_from_bits`, `nr_srs_dci_0_3_request_list_index_from_layout`, `nr_dci_0_3_field_values_from_layout`, `nr_srs_dci_0_3_frequency_assignment_widths`, `nr_srs_dci_0_3_field_widths_from_configuration`, `nr_srs_dci_0_3_multicell_aperiodic_resource_set_schedules`, `nr_srs_dci_0_3_multicell_aperiodic_resource_set_schedules_from_bits`, `nr_srs_dci_0_3_ul_bwp_selection`, `nr_srs_dci_0_3_multicell_aperiodic_resource_set_schedules_by_bwp`, `NRSRSScheduledResourceOccurrence`, `audit_nr_srs_intercell_re_overlap`, `nr_srs_triggered_resource_set_ids`, `nr_srs_cyclic_shift_hopping_offset`, `nr_srs_cyclic_shift_hopping_offsets`, `nr_srs_periodicity_and_offset_slots`, `nr_srs_periodic_resource_schedule`, `nr_srs_periodic_resource_set_schedule`, `nr_srs_periodic_resource_set_schedule_by_bwp`, `nr_srs_multicell_periodic_resource_set_schedules_by_bwp`, `nr_srs_resource_frequency_allocations`, `nr_srs_positioning_symbol_offset`, `nr_srs_port_cyclic_shift_indices`, `nr_srs_port_configuration_from_rrc`, `nr_srs_port_sequences_type1`, `nr_srs_port_symbol_activity`, `nr_srs_resource_sequences_type1`, `nr_srs_scheduled_port_sequences_type1`, `nr_srs_sequence_hopping_index`, `nr_srs_sequence_type1`, `nr_srs_type1_bandwidth_configuration`, `nr_srs_transmission_counter`, `nr_srs_transmission_comb_from_rrc`, `srs_cyclic_shift_count`, `nr_pbch_dmrs_ssb`, `nr_dl_prs_c_init`, `nr_dl_prs_delay_cfo_profile`, `nr_dl_prs_matched_delay_profile`, `nr_dl_prs_resource_occasion`, `nr_dl_prs_sequence`, `nr_pbch_payload_descramble`, `nr_pbch_payload_generate`, `nr_pbch_payload_recover`, `nr_pbch_payload_scramble`, `nr_pbch_mib_parse`, `nr_pbch_mib_serialize`, `nr_bcch_bch_message_parse`, `nr_bcch_bch_message_serialize`, `nr_pbch_receive_ofdm`, `nr_pbch_acquire_and_receive`, `nr_pbch_bch_encode`, `nr_pbch_bch_decode`, `nr_pbch_qpsk_modulate`, `nr_pbch_scramble_bits`, `map_nr_ssb_resource_grid`, `scale_nr_ssb_resource_grid_epre`, `scale_nr_ssb_resource_grid_power`, `nr_ssb_ofdm_demodulate`, `nr_ssb_ofdm_modulate_half_frame`, `nr_ssb_ofdm_modulate_radio_frame`, `nr_ssb_ofdm_modulate`, `nr_ssb_waveform_config`, `sionna_paths_to_oracle_table`, `ssb_case_a_candidate_symbols`, `ssb_candidate_symbols`

## Invariants

- Public names and behaviour are unchanged by the split; tests import the package, not the modules.
- Module-level imports point down the layers. Any cycle is a defect: `tests/architecture` checks this.

