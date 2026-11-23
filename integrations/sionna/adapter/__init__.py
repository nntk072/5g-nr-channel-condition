"""Public Sionna adapter: ray-traced channel generation entry points."""
# ruff: noqa: F401, E402

import hashlib
import json
import math
import sys
from copy import copy
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Protocol, Sequence, cast

import numpy as np
from artifacts.provenance import canonical_hash, package_source_sha256, versions
from datasets.channel import ChannelDataset
from datasets.network import LinkDirection, LinkRole, NodeRole
from datasets.observations import (
    FrequencyDomainModelInputs,
    FrequencyDomainObservation,
    LinkedFrequencyDomainObservation,
)
from datasets.rt_split_identity import rt_spatial_split_subject_id as _rt_spatial_split_subject_id
from propagation.physics import add_estimation_noise, frequency_response, subcarrier_frequencies
from propagation.scene_visibility import SceneTriangleInventory, extract_scene_triangle_inventory

from integrations.sionna.candidate_audit import instrument_path_solver_candidate_counter
from integrations.sionna.d1_generation_gate import D1GenerationAuthorization
from integrations.sionna.los_oracle import (
    SegmentVisibilityState,
    independent_mesh_segment_visibility,
    segment_visibility,
)
from integrations.sionna.random import call_with_sionna_seed
from integrations.sionna.rt_identity import (
    rt_run_identity,
    rt_runtime_fingerprint,
    sionna_rt_siso_antenna_config,
    static_rt_solver_config,
)

from .generate import (
    generate_raytraced,
    generate_raytraced_cached,
)
from .raytrace import (
    _RT_EXECUTION_PROVENANCE_SCHEMA_PATH,
    GridConfig,
    RayScene,
    RaytraceGenerationError,
    RTPaths,
    _attach_rt_execution_provenance,
    _mask_path_coefficients,
    _phy_cir_to_cfr,
    _publish_static_rt_sample_umcd,
    _scene_family_run_binding,
    authorize_rt_data_generation,
    build_rt_data_generation_request,
    count_valid_paths,
    direct_indirect_diagnostics,
    direct_path_mask,
    generate_standardized,
    has_valid_path,
    rt_cfr,
)

__all__ = [
    "GridConfig",
    "RTPaths",
    "RayScene",
    "RaytraceGenerationError",
    "_RT_EXECUTION_PROVENANCE_SCHEMA_PATH",
    "_attach_rt_execution_provenance",
    "_mask_path_coefficients",
    "_phy_cir_to_cfr",
    "_publish_static_rt_sample_umcd",
    "_scene_family_run_binding",
    "authorize_rt_data_generation",
    "build_rt_data_generation_request",
    "count_valid_paths",
    "direct_indirect_diagnostics",
    "direct_path_mask",
    "generate_standardized",
    "has_valid_path",
    "rt_cfr",
    "generate_raytraced",
    "generate_raytraced_cached",
]
