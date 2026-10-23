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
from .pbch_dmrs import PBCHDMRSResourceMap, nr_pbch_dmrs_ssb
from .rt_cache import RTCacheConflictError, load_rt_dataset, store_rt_dataset
from .rt_to_nr import NRGridChannel, paths_to_nr_ofdm_channel
from .ssb_grid import NRSSBResourceGrid, map_nr_ssb_resource_grid
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
    "map_nr_ssb_resource_grid",
    "sionna_paths_to_oracle_table",
    "ssb_case_a_candidate_symbols",
]
