"""Bounded semantic consistency checks for proposal-only packages."""

from __future__ import annotations

from evaluation.acquisition.canonical import (
    _canonical_json,
    _canonical_sha256,
    _json_bytes,
    _parse_utc_timestamp,
    _reject_json_constant,
    _strata_key_set,
    _unique_object,
)
from evaluation.acquisition.constants import (
    _SCHEMA_ROOT,
    CHUNK_BYTES,
    EVALUATOR_INPUT_SCHEMA_PATHS,
    LIMITATIONS,
    MATERIAL_MAPPING_SCHEMA_PATH,
    MAX_FILE_BYTES,
    MAX_INVENTORY_FILES,
    MAX_JSON_ARTIFACT_BYTES,
    MAX_PACKAGE_BYTES,
    MAX_SCHEMA_BYTES,
    MAX_TOTAL_FILE_BYTES,
    SCHEMA_PATHS,
    V03_PLUS_SCHEMA_VERSIONS,
)
from evaluation.acquisition.files import (
    _read_bounded_regular_file,
    _read_inventory_file,
    _safe_relative_path,
)
from evaluation.acquisition.index import (
    _pointer_item,
    _unique_index,
)
from evaluation.acquisition.noise import (
    _noise_rms_summary,
)
from evaluation.acquisition.verify import (
    verify_acquisition_package,
)

__all__ = [
    "CHUNK_BYTES",
    "EVALUATOR_INPUT_SCHEMA_PATHS",
    "LIMITATIONS",
    "MATERIAL_MAPPING_SCHEMA_PATH",
    "MAX_FILE_BYTES",
    "MAX_INVENTORY_FILES",
    "MAX_JSON_ARTIFACT_BYTES",
    "MAX_PACKAGE_BYTES",
    "MAX_SCHEMA_BYTES",
    "MAX_TOTAL_FILE_BYTES",
    "SCHEMA_PATHS",
    "V03_PLUS_SCHEMA_VERSIONS",
    "_SCHEMA_ROOT",
    "_canonical_json",
    "_canonical_sha256",
    "_json_bytes",
    "_noise_rms_summary",
    "_parse_utc_timestamp",
    "_pointer_item",
    "_read_bounded_regular_file",
    "_read_inventory_file",
    "_reject_json_constant",
    "_safe_relative_path",
    "_strata_key_set",
    "_unique_index",
    "_unique_object",
    "verify_acquisition_package",
]
