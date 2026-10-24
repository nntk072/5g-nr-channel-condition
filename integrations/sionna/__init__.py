"""Lazy Sionna PHY and ray-tracing adapters."""

from .adapter import (
    GridConfig,
    RayScene,
    RaytraceGenerationError,
    generate_raytraced,
    generate_raytraced_cached,
    generate_standardized,
)
from .backend import RTBackendSelection, activate_rt_backend, select_rt_backend
from .cdl import CDLArrayConfig, SionnaCDLChannel, generate_cdl_frequency_response
from .cp_margin import run_static_cp_margin_experiment
from .nr_pusch import SionnaPUSCHLSReceiver
from .ofdm_impairments import (
    NRWaveformImpairments,
    apply_integer_timing_offset,
    apply_residual_cfo,
    apply_time_varying_multipath,
)
from .path_oracle import sionna_paths_to_oracle_table
from .pbch_acquisition import (
    PBCHAcquisitionError,
    PBCHAcquisitionReception,
    nr_pbch_acquire_and_receive,
)
from .pbch_bch import nr_pbch_bch_decode, nr_pbch_bch_encode
from .pbch_data import nr_pbch_qpsk_modulate, nr_pbch_scramble_bits
from .pbch_dmrs import PBCHDMRSResourceMap, nr_pbch_dmrs_ssb
from .pbch_mib import PBCHMIB, nr_pbch_mib_parse, nr_pbch_mib_serialize
from .pbch_ofdm_receiver import PBCHOFDMReception, nr_pbch_receive_ofdm
from .pbch_payload import (
    PBCHPayloadRecovery,
    nr_pbch_payload_descramble,
    nr_pbch_payload_generate,
    nr_pbch_payload_recover,
    nr_pbch_payload_scramble,
)
from .rt_cache import RTCacheConflictError, load_rt_dataset, store_rt_dataset
from .rt_to_nr import NRGridChannel, paths_to_nr_ofdm_channel
from .ssb_grid import (
    NRSSBResourceGrid,
    map_nr_ssb_resource_grid,
    scale_nr_ssb_resource_grid_epre,
    scale_nr_ssb_resource_grid_power,
)
from .ssb_ofdm import nr_ssb_ofdm_demodulate, nr_ssb_ofdm_modulate
from .ssb_timing import ssb_case_a_candidate_symbols
from .tdl import SionnaTDLChannel, generate_tdl_frequency_response
from .tdl_pusch import SionnaTDLPUSCHBatch, generate_tdl_pusch_batch, tdl_pusch_records

__all__ = [
    "GridConfig",
    "CDLArrayConfig",
    "NRGridChannel",
    "RayScene",
    "RTBackendSelection",
    "RaytraceGenerationError",
    "RTCacheConflictError",
    "SionnaPUSCHLSReceiver",
    "NRWaveformImpairments",
    "NRSSBResourceGrid",
    "PBCHDMRSResourceMap",
    "PBCHPayloadRecovery",
    "PBCHMIB",
    "PBCHOFDMReception",
    "PBCHAcquisitionError",
    "PBCHAcquisitionReception",
    "SionnaCDLChannel",
    "SionnaTDLChannel",
    "SionnaTDLPUSCHBatch",
    "apply_integer_timing_offset",
    "apply_residual_cfo",
    "apply_time_varying_multipath",
    "activate_rt_backend",
    "generate_raytraced",
    "generate_raytraced_cached",
    "generate_standardized",
    "generate_cdl_frequency_response",
    "load_rt_dataset",
    "run_static_cp_margin_experiment",
    "select_rt_backend",
    "generate_tdl_frequency_response",
    "generate_tdl_pusch_batch",
    "tdl_pusch_records",
    "store_rt_dataset",
    "paths_to_nr_ofdm_channel",
    "nr_pbch_dmrs_ssb",
    "nr_pbch_payload_descramble",
    "nr_pbch_payload_generate",
    "nr_pbch_payload_recover",
    "nr_pbch_payload_scramble",
    "nr_pbch_mib_parse",
    "nr_pbch_mib_serialize",
    "nr_pbch_receive_ofdm",
    "nr_pbch_acquire_and_receive",
    "nr_pbch_bch_encode",
    "nr_pbch_bch_decode",
    "nr_pbch_qpsk_modulate",
    "nr_pbch_scramble_bits",
    "map_nr_ssb_resource_grid",
    "scale_nr_ssb_resource_grid_epre",
    "scale_nr_ssb_resource_grid_power",
    "nr_ssb_ofdm_demodulate",
    "nr_ssb_ofdm_modulate",
    "sionna_paths_to_oracle_table",
    "ssb_case_a_candidate_symbols",
]
