# ruff: noqa: F401  (re-exports for the public runner API)
"""Experiment runner: feature construction, label and validation logic, model-input contracts, and the run entry point."""

import hashlib
import inspect
import io
import json
import os
import platform
import stat
import sys
import zipfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from artifacts.build_dag import BuildNodeSpec, resolve_build_artifact, run_build_node
from artifacts.layout import RunLayout
from artifacts.manifest import mark_complete, mark_failed, run_manifest
from artifacts.provenance import (
    atomic_json,
    canonical_hash,
    package_source_sha256,
    sha256_file,
    versions,
)
from artifacts.run_replay import build_run_replay_provenance
from configuration.loader import resolve_config
from datasets.channel import ChannelDataset
from datasets.split_policy import (
    SplitPolicyRegistry,
    _reject_duplicate_json_keys,
    verify_openssh_signed_artifact,
    verify_signed_split_policy_registry_attestation,
)
from evaluation.benchmark_partitions import (
    BENCHMARK_COHORTS,
    TEST_COHORTS,
    verify_benchmark_manifest_replay,
)
from evaluation.scoring.class_support import audit_class_support
from evaluation.scoring.low_confidence import build_low_confidence_audit
from evaluation.scoring.metrics import (
    binary_metrics,
    block_bootstrap_auc,
    group_split,
    holdout_domain,
    in_domain_split,
    threshold_from_validation,
)
from evaluation.scoring.prior_shift import (
    adjust_probabilities_between_priors,
    adjust_probabilities_for_prior_shift,
    validate_prior_shift_spec,
)
from evaluation.scoring.probability_calibration import (
    apply_platt_calibrator,
    fit_group_balanced_platt_calibrator,
    invert_platt_calibrator,
)
from evaluation.scoring.statistics import paired_seed_delta
from learning import fit_complex_cnn, fit_coral_mlp, fit_real_cnn, fit_sklearn
from models.capacity import capacity_spec, matched_capacity_names
from models.multipath import fit_sparse_multipath
from propagation.physics import delay_k_proxy_features, observable_features
from receivers.pilots import comb_ls_estimate

from .contracts import (
    _feature_transform_implementation_sha256,
    build_model_input_artifact_contract,
    verify_declared_partition_support,
    verify_model_input_artifact_contract,
    verify_validation_threshold_selection_record,
)
from .experiment_runner import (
    _run_experiment_impl,
    run_experiment,
)
from .features import (
    _CHANNEL_DATASET_BUILD_SCHEMA,
    _MODEL_FEATURE_ROUTES,
    _MODEL_METHODS,
    _VALIDATION_THRESHOLD_POLICY,
    _array_sha256,
    _flatten_real_imag_cfr,
    _identity_complex_cfr,
    _kish_effective_group_count,
    _label_semantics_by_source,
    _learned_parameter_count,
    _log_power_per_link_subcarrier,
    _normalized_label_semantics,
    _precomputed_observation_semantics,
    _run_replay_rng_streams,
    _sparse_multipath_features,
    _stable_input_bytes,
    _stable_single_link_sha256,
    _sync_torch_device,
    _unsupported_benchmark_test_cohorts,
    _validate_evaluation_payload,
    _validate_json_schema,
    _validate_npz_snapshot,
)
from .validation import (
    _declared_partition_support,
    _locked_split_indices,
    _materialize_channel_dataset_build_node,
    _metadata_only_features,
    _oracle_label_leak_features,
    _permute_training_feature_rows,
    _permute_training_labels,
    _phase_randomized_real_imag_cfr,
    _resolve_benchmark_test_cohort,
    _resolved_label_semantics,
    _single_metadata_features,
    _source_only_features,
    _validation_threshold_selection_record,
    _write_benchmark_preflight_failure_receipt,
)

__all__ = [
    "_CHANNEL_DATASET_BUILD_SCHEMA",
    "_MODEL_FEATURE_ROUTES",
    "_MODEL_METHODS",
    "_VALIDATION_THRESHOLD_POLICY",
    "_array_sha256",
    "_flatten_real_imag_cfr",
    "_identity_complex_cfr",
    "_kish_effective_group_count",
    "_label_semantics_by_source",
    "_learned_parameter_count",
    "_log_power_per_link_subcarrier",
    "_normalized_label_semantics",
    "_precomputed_observation_semantics",
    "_run_replay_rng_streams",
    "_sparse_multipath_features",
    "_stable_input_bytes",
    "_stable_single_link_sha256",
    "_sync_torch_device",
    "_unsupported_benchmark_test_cohorts",
    "_validate_evaluation_payload",
    "_validate_json_schema",
    "_validate_npz_snapshot",
    "_declared_partition_support",
    "_locked_split_indices",
    "_materialize_channel_dataset_build_node",
    "_metadata_only_features",
    "_oracle_label_leak_features",
    "_permute_training_feature_rows",
    "_permute_training_labels",
    "_phase_randomized_real_imag_cfr",
    "_resolve_benchmark_test_cohort",
    "_resolved_label_semantics",
    "_single_metadata_features",
    "_source_only_features",
    "_validation_threshold_selection_record",
    "_write_benchmark_preflight_failure_receipt",
    "_feature_transform_implementation_sha256",
    "build_model_input_artifact_contract",
    "verify_declared_partition_support",
    "verify_model_input_artifact_contract",
    "verify_validation_threshold_selection_record",
    "_run_experiment_impl",
    "run_experiment",
]
