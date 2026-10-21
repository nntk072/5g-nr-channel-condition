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
from .cp_margin import run_static_cp_margin_experiment
from .nr_pusch import SionnaPUSCHLSReceiver
from .ofdm_impairments import apply_integer_timing_offset, apply_residual_cfo
from .rt_cache import RTCacheConflictError, load_rt_dataset, store_rt_dataset
from .rt_to_nr import NRGridChannel, paths_to_nr_ofdm_channel
from .tdl import SionnaTDLChannel, generate_tdl_frequency_response
from .tdl_pusch import SionnaTDLPUSCHBatch, generate_tdl_pusch_batch, tdl_pusch_records

__all__ = [
    "GridConfig",
    "NRGridChannel",
    "RayScene",
    "RTBackendSelection",
    "RaytraceGenerationError",
    "RTCacheConflictError",
    "SionnaPUSCHLSReceiver",
    "SionnaTDLChannel",
    "SionnaTDLPUSCHBatch",
    "apply_integer_timing_offset",
    "apply_residual_cfo",
    "activate_rt_backend",
    "generate_raytraced",
    "generate_raytraced_cached",
    "generate_standardized",
    "load_rt_dataset",
    "run_static_cp_margin_experiment",
    "select_rt_backend",
    "generate_tdl_frequency_response",
    "generate_tdl_pusch_batch",
    "tdl_pusch_records",
    "store_rt_dataset",
    "paths_to_nr_ofdm_channel",
]
