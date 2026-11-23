"""Channel-split lineage audit: child artifacts, split rosters, and channel audits."""
# ruff: noqa: F401, E402

import argparse
import errno
import hashlib
import io
import json
import math
import os
import re
import stat
import zipfile
import zlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import jsonschema
import numpy as np
from datasets.channel import ChannelDataset
from datasets.legacy_rt_split import upgrade_legacy_rt_split_metadata
from datasets.split_lineage import derive_channel_dataset_split_evidence
from datasets.split_policy import (
    SPLIT_POLICY_SIGNATURE_NAMESPACE,
    SplitChildBinding,
    VerifiedOpenSSHArtifact,
    audit_split_policy_child_roster,
    audit_split_policy_parent_roster,
    verify_openssh_signed_artifact,
    verify_registry_spatiotemporal_audit,
    verify_signed_split_policy_registry_attestation,
)

from .channel_audit import (
    audit_channel_datasets,
    main,
)
from .child_artifacts import (
    _CHILD_ARTIFACT_ROSTER_SIGNATURE_NAMESPACE,
    _GIT_SHA_RE,
    _INVENTORY_SCHEMA,
    _INVENTORY_SIGNATURE_NAMESPACE,
    _LEGACY_ASSET_RECONSTRUCTION_SCHEMA,
    _MAX_CHILD_ARRAY_HEADER_BYTES,
    _MAX_CHILD_ARTIFACT_TREE_DEPTH,
    _MAX_CHILD_ARTIFACT_TREE_FILES,
    _MAX_CHILD_METADATA_NPY_BYTES,
    _MAX_CHILD_ROSTER_BYTES,
    _REPOSITORY_ROOT,
    _SHA256_RE,
    _SPLIT_AUDIT_SCHEMA_PATH,
    _V3_SEMANTIC_CHILD_ARTIFACT_TYPES,
    _V4_SEMANTIC_CHILD_ARTIFACT_TYPES,
    _V6_MULTI_OWNER_ARTIFACT_TYPES,
    _V6_SEMANTIC_CHILD_ARTIFACT_TYPES,
    _attach_reconstructed_scene_assets,
    _enumerate_regular_files_beneath,
    _hash_inventory_relative_file,
    _load_frozen_inventory,
    _load_legacy_asset_reconstruction,
    _load_split_child_roster,
    _parse_dataset,
    _parse_split_child_roster,
    _read_channel_dataset_child_rows,
    _read_inventory_relative_file,
    _read_stable_regular_file,
    _require_exact_child_object_owners,
    _require_shared_content_owners_share_split,
    _stat_identity,
    _unique_json_object,
    _validate_split_audit_report,
)
from .split_artifacts import (
    _load_split_child_artifact_roster,
)

__all__ = [
    "_CHILD_ARTIFACT_ROSTER_SIGNATURE_NAMESPACE",
    "_GIT_SHA_RE",
    "_INVENTORY_SCHEMA",
    "_INVENTORY_SIGNATURE_NAMESPACE",
    "_LEGACY_ASSET_RECONSTRUCTION_SCHEMA",
    "_MAX_CHILD_ARRAY_HEADER_BYTES",
    "_MAX_CHILD_ARTIFACT_TREE_DEPTH",
    "_MAX_CHILD_ARTIFACT_TREE_FILES",
    "_MAX_CHILD_METADATA_NPY_BYTES",
    "_MAX_CHILD_ROSTER_BYTES",
    "_REPOSITORY_ROOT",
    "_SHA256_RE",
    "_SPLIT_AUDIT_SCHEMA_PATH",
    "_V3_SEMANTIC_CHILD_ARTIFACT_TYPES",
    "_V4_SEMANTIC_CHILD_ARTIFACT_TYPES",
    "_V6_MULTI_OWNER_ARTIFACT_TYPES",
    "_V6_SEMANTIC_CHILD_ARTIFACT_TYPES",
    "_attach_reconstructed_scene_assets",
    "_enumerate_regular_files_beneath",
    "_hash_inventory_relative_file",
    "_load_frozen_inventory",
    "_load_legacy_asset_reconstruction",
    "_load_split_child_roster",
    "_parse_dataset",
    "_parse_split_child_roster",
    "_read_channel_dataset_child_rows",
    "_read_inventory_relative_file",
    "_read_stable_regular_file",
    "_require_exact_child_object_owners",
    "_require_shared_content_owners_share_split",
    "_stat_identity",
    "_unique_json_object",
    "_validate_split_audit_report",
    "_load_split_child_artifact_roster",
    "audit_channel_datasets",
    "main",
]
