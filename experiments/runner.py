"""End-to-end leakage-protected experiment with exact dataset and run provenance."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import stat
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np
from artifacts.build_dag import BuildNodeSpec, resolve_build_artifact, run_build_node
from artifacts.layout import RunLayout
from artifacts.manifest import mark_complete, run_manifest
from artifacts.provenance import (
    atomic_json,
    canonical_hash,
    package_source_sha256,
    sha256_file,
    versions,
)
from configuration.loader import resolve_config
from datasets.channel import ChannelDataset
from datasets.split_policy import SplitPolicyRegistry, verify_signed_split_policy_registry
from evaluation.class_support import audit_class_support
from evaluation.low_confidence import build_low_confidence_audit
from evaluation.metrics import (
    binary_metrics,
    block_bootstrap_auc,
    group_split,
    holdout_domain,
    in_domain_split,
    threshold_from_validation,
)
from evaluation.prior_shift import (
    adjust_probabilities_between_priors,
    adjust_probabilities_for_prior_shift,
    validate_prior_shift_spec,
)
from evaluation.probability_calibration import (
    apply_platt_calibrator,
    fit_group_balanced_platt_calibrator,
    invert_platt_calibrator,
)
from evaluation.statistics import paired_seed_delta
from learning import fit_complex_cnn, fit_coral_mlp, fit_real_cnn, fit_sklearn
from models.capacity import capacity_spec
from propagation.physics import observable_features
from receivers.pilots import comb_ls_estimate

_VALIDATION_THRESHOLD_POLICY = {
    "schema": "loslab.validation_threshold_selection.v1",
    "criterion": "Youden J = sensitivity - false-positive rate",
    "selection_partition": "validation",
    "positive_label": 1,
    "weighting": "unweighted validation rows",
    "tie_break": "first maximum in roc_curve threshold order",
    "test_labels_used_for_selection": False,
}

_MODEL_FEATURE_ROUTES = {
    "cnn": ("observed_cfr", "complex_cfr_grid_v1"),
    "cvnn": ("observed_cfr", "complex_cfr_grid_v1"),
    "raw_mlp": ("observed_cfr", "flattened_real_imag_cfr_v1"),
    "label_permuted": ("observed_cfr", "flattened_real_imag_cfr_v1"),
    "row_shuffled": ("observed_cfr", "flattened_real_imag_cfr_v1"),
    "source_only": ("source", "train_vocabulary_source_one_hot_v1"),
    "metadata_only": (
        "group/domain/source",
        "train_vocabulary_group_domain_source_one_hot_v1",
    ),
    "power_only": ("observed_cfr", "log_power_per_link_subcarrier_v1"),
    "phase_randomized": ("observed_cfr", "iid_phase_randomized_flattened_real_imag_cfr_v1"),
    "observable_features": ("observed_cfr", "logmag_pdp_moments_v1"),
}
_MODEL_METHODS = frozenset(
    {
        "rf",
        "histgb",
        "xgb",
        "mlp",
        "raw_mlp",
        "label_permuted",
        "row_shuffled",
        "source_only",
        "metadata_only",
        "power_only",
        "phase_randomized",
        "logreg",
        "cnn",
        "cvnn",
        "coral",
        "coral0",
    }
)


def _validate_evaluation_payload(payload: dict[str, Any], schema_filename: str) -> None:
    """Validate persisted calibration/prior metadata against its versioned contract."""
    import jsonschema

    schema_path = Path(__file__).resolve().parents[1] / "evaluation/schema" / schema_filename
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    errors = list(jsonschema.Draft202012Validator(schema).iter_errors(payload))
    if errors:
        raise ValueError(
            f"evaluation payload violates {schema_filename}: "
            + "; ".join(error.message for error in errors[:10])
        )


_CHANNEL_DATASET_BUILD_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["operation", "input_order"],
    "properties": {
        "operation": {"const": "ordered_channel_dataset_concatenation_v1"},
        "input_order": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["path", "sha256"],
                "properties": {
                    "path": {"type": "string", "minLength": 1},
                    "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                },
                "additionalProperties": False,
            },
        },
    },
    "additionalProperties": False,
}


def _materialize_channel_dataset_build_node(
    paths: Sequence[str | Path], cache_root: str | Path
) -> tuple[ChannelDataset, dict[str, Any]]:
    source_paths = tuple(Path(path).expanduser().resolve() for path in paths)
    ordered_inputs = [{"path": str(path), "sha256": sha256_file(path)} for path in source_paths]
    input_files = {row["path"]: row["sha256"] for row in ordered_inputs}
    source_revision = versions().get("git_sha") or "unknown"
    source_bundle_sha256 = package_source_sha256()
    spec = BuildNodeSpec(
        node_id="channel_dataset.ordered_concatenation",
        node_version="1.0.0",
        configuration={
            "operation": "ordered_channel_dataset_concatenation_v1",
            "input_order": ordered_inputs,
        },
        configuration_schema=_CHANNEL_DATASET_BUILD_SCHEMA,
        source_revision=source_revision,
        source_bundle_sha256=source_bundle_sha256,
        input_files=input_files,
        dependencies={},
        outputs=("channel-dataset.npz",),
    )

    def build(staging: Path) -> None:
        combined = ChannelDataset.concatenate([ChannelDataset.load(path) for path in source_paths])
        combined.save(staging / "channel-dataset.npz", exclusive=True)

    result = run_build_node(cache_root, spec, build)
    artifact_path = resolve_build_artifact(cache_root, result, "channel-dataset.npz")
    data = ChannelDataset.load(artifact_path)
    manifest_digest = sha256_file(result.manifest_path)
    return data, {
        "schema": "loslab.channel_dataset_materialization.v1",
        "node_key": result.node_key,
        "manifest_sha256": manifest_digest,
        "artifact_sha256": result.artifacts[0]["sha256"],
        "artifact_size_bytes": result.artifacts[0]["size_bytes"],
        "cache_reused": result.reused,
        "input_order": ordered_inputs,
    }


def _identity_complex_cfr(observed_cfr: np.ndarray) -> np.ndarray:
    """Keep the audited complex CFR grid as the model's input matrix."""
    return observed_cfr


def _flatten_real_imag_cfr(observed_cfr: np.ndarray) -> np.ndarray:
    """Flatten each CFR sample in C order, real values before imaginary values."""
    return np.concatenate((observed_cfr.real, observed_cfr.imag), axis=1).reshape(
        len(observed_cfr), -1
    )


def _log_power_per_link_subcarrier(observed_cfr: np.ndarray) -> np.ndarray:
    """Expose only per-link/per-subcarrier channel power in a fixed dB scale."""
    observed = np.asarray(observed_cfr)
    if observed.ndim != 3 or not np.iscomplexobj(observed):
        raise ValueError("power-only features require complex [samples, links, subcarriers] CFR")
    if not np.isfinite(observed).all():
        raise ValueError("power-only CFR input must be finite")
    with np.errstate(over="ignore", invalid="ignore"):
        power = (
            np.asarray(observed, dtype=np.complex128).real ** 2
            + np.asarray(observed, dtype=np.complex128).imag ** 2
        )
    if not np.isfinite(power).all():
        raise ValueError("power-only CFR power is not representable in float64")
    log_power_db = 10.0 * np.log10(np.maximum(power, 1e-12))
    return log_power_db.reshape(len(observed), -1)


def _phase_randomized_real_imag_cfr(observed_cfr: np.ndarray, *, seed: int) -> np.ndarray:
    """Apply independent seeded phase per sample/link/tone, preserving CFR magnitude."""
    observed = np.asarray(observed_cfr)
    if observed.ndim != 3 or not np.iscomplexobj(observed):
        raise ValueError(
            "phase-randomized features require complex [samples, links, subcarriers] CFR"
        )
    if not np.isfinite(observed).all():
        raise ValueError("phase-randomized CFR input must be finite")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**63:
        raise ValueError("phase_randomization_seed must be an integer in [0, 2**63)")
    rng = np.random.default_rng(int(seed))
    phase = rng.uniform(-np.pi, np.pi, size=observed.shape)
    randomized = np.asarray(observed, dtype=np.complex128) * np.exp(1j * phase)
    return _flatten_real_imag_cfr(randomized)


def _permute_training_labels(
    labels: np.ndarray, *, run_seed: int, base_seed: int
) -> tuple[np.ndarray, dict[str, Any]]:
    """Deterministically permute only the training labels, preserving their counts."""
    values = np.asarray(labels)
    if (
        values.ndim != 1
        or not np.issubdtype(values.dtype, np.integer)
        or np.issubdtype(values.dtype, np.bool_)
        or len(values) < 2
        or not np.isin(values, (0, 1)).all()
    ):
        raise ValueError("label permutation requires at least two binary integer training labels")
    for name, seed in (("run_seed", run_seed), ("base_seed", base_seed)):
        if (
            isinstance(seed, bool)
            or not isinstance(seed, (int, np.integer))
            or not 0 <= seed < 2**63
        ):
            raise ValueError(f"{name} must be an integer in [0, 2**63)")
    rng = np.random.default_rng(np.random.SeedSequence([int(run_seed), int(base_seed)]))
    permutation = rng.permutation(len(values)).astype(np.int64, copy=False)
    permuted = np.array(values[permutation], copy=True)
    contract = {
        "schema": "loslab.training_label_permutation.v1",
        "algorithm": "numpy_seedsequence_default_rng_permutation_v1",
        "run_seed": int(run_seed),
        "base_seed": int(base_seed),
        "row_count": int(len(values)),
        "source_targets_sha256": _array_sha256(values),
        "permutation_indices_sha256": _array_sha256(permutation),
        "permuted_targets_sha256": _array_sha256(permuted),
        "positive_count": int(np.count_nonzero(permuted == 1)),
    }
    return permuted, contract


def _permute_training_feature_rows(
    features: np.ndarray, *, run_seed: int, base_seed: int
) -> tuple[np.ndarray, dict[str, Any]]:
    """Deterministically reorder training observations without changing labels."""
    values = np.asarray(features)
    if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("feature-row permutation requires a finite 2D training matrix")
    for name, seed in (("run_seed", run_seed), ("base_seed", base_seed)):
        if (
            isinstance(seed, bool)
            or not isinstance(seed, (int, np.integer))
            or not 0 <= seed < 2**63
        ):
            raise ValueError(f"{name} must be an integer in [0, 2**63)")
    rng = np.random.default_rng(np.random.SeedSequence([int(run_seed), int(base_seed)]))
    permutation = rng.permutation(len(values)).astype(np.int64, copy=False)
    permuted = np.array(values[permutation], copy=True)
    contract = {
        "schema": "loslab.training_feature_row_permutation.v1",
        "algorithm": "numpy_seedsequence_default_rng_permutation_v1",
        "implementation_sha256": hashlib.sha256(
            inspect.getsource(_permute_training_feature_rows).encode("utf-8")
        ).hexdigest(),
        "run_seed": int(run_seed),
        "base_seed": int(base_seed),
        "row_count": int(len(values)),
        "source_features_sha256": _array_sha256(values),
        "permutation_indices_sha256": _array_sha256(permutation),
        "permuted_features_sha256": _array_sha256(permuted),
    }
    return permuted, contract


def _source_only_features(
    source_values: np.ndarray, *, train_indices: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a train-vocabulary-only one-hot control from source identifiers."""
    values = np.asarray(source_values)
    if values.ndim != 1 or not len(values):
        raise ValueError("source-only control requires a nonempty 1D source roster")
    if any(
        not isinstance(value, (str, np.str_)) or not str(value) or str(value) != str(value).strip()
        for value in values
    ):
        raise ValueError("source-only control requires nonempty canonical string identifiers")
    if (
        not isinstance(train_indices, np.ndarray)
        or train_indices.ndim != 1
        or not np.issubdtype(train_indices.dtype, np.integer)
        or np.issubdtype(train_indices.dtype, np.bool_)
        or len(train_indices) < 1
        or np.any(train_indices < 0)
        or np.any(train_indices >= len(values))
        or len(np.unique(train_indices)) != len(train_indices)
    ):
        raise ValueError("source-only training indices are invalid")
    vocabulary = tuple(sorted({str(value) for value in values[train_indices]}))
    if not vocabulary:
        raise ValueError("source-only training vocabulary cannot be empty")
    column_by_value = {value: index for index, value in enumerate(vocabulary)}
    features = np.zeros((len(values), len(vocabulary)), dtype=np.float32)
    for row, value in enumerate(values):
        column = column_by_value.get(str(value))
        if column is not None:
            features[row, column] = 1.0
    transform = {
        "schema": "loslab.source_only_vocabulary.v1",
        "algorithm": "sorted_training_source_identifiers_one_hot_unknown_zero_v1",
        "implementation_sha256": hashlib.sha256(
            inspect.getsource(_source_only_features).encode("utf-8")
        ).hexdigest(),
        "vocabulary": list(vocabulary),
        "vocabulary_sha256": canonical_hash(list(vocabulary)),
        "source_values_sha256": _array_sha256(np.asarray(values, dtype=str)),
        "training_indices_sha256": _array_sha256(train_indices),
        "unknown_category_behavior": "all_zero_vector",
    }
    return features, transform


def _metadata_only_features(
    *,
    group_values: np.ndarray,
    domain_values: np.ndarray,
    source_values: np.ndarray,
    train_indices: np.ndarray,
    max_feature_cells: int = 16_000_000,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a bounded metadata-leakage control from group/domain/source IDs.

    This diagnostic intentionally tests whether non-signal identity fields can
    predict labels. The categorical vocabulary is fitted on training rows only;
    unseen validation/test values map to zero. It is not a receiver feature
    route and is excluded from deployment model recommendations.
    """
    fields = {
        "domain": np.asarray(domain_values),
        "group": np.asarray(group_values),
        "source": np.asarray(source_values),
    }
    sizes = {values.shape for values in fields.values()}
    if len(sizes) != 1 or next(iter(sizes))[0] < 1 or next(iter(sizes))[1:] != ():
        raise ValueError("metadata-only control requires equal nonempty 1D identifier rosters")
    row_count = next(iter(sizes))[0]
    for name, values in fields.items():
        if any(
            not isinstance(value, (str, np.str_))
            or not str(value)
            or str(value) != str(value).strip()
            or "\x00" in str(value)
            for value in values
        ):
            raise ValueError(f"metadata-only {name} IDs must be canonical nonempty strings")
    if (
        not isinstance(train_indices, np.ndarray)
        or train_indices.ndim != 1
        or not np.issubdtype(train_indices.dtype, np.integer)
        or np.issubdtype(train_indices.dtype, np.bool_)
        or len(train_indices) < 1
        or np.any(train_indices < 0)
        or np.any(train_indices >= row_count)
        or len(np.unique(train_indices)) != len(train_indices)
    ):
        raise ValueError("metadata-only training indices are invalid")
    if (
        isinstance(max_feature_cells, bool)
        or not isinstance(max_feature_cells, int)
        or max_feature_cells < 1
    ):
        raise ValueError("max_feature_cells must be a positive integer")
    vocabulary = tuple(
        sorted({(name, str(fields[name][index])) for name in fields for index in train_indices})
    )
    if row_count * len(vocabulary) > max_feature_cells:
        raise ValueError(
            "metadata-only dense feature matrix exceeds max_feature_cells; "
            "reduce the cohort or raise the explicit diagnostic ceiling"
        )
    column_by_value = {value: index for index, value in enumerate(vocabulary)}
    features = np.zeros((row_count, len(vocabulary)), dtype=np.float32)
    for row in range(row_count):
        for name, values in fields.items():
            column = column_by_value.get((name, str(values[row])))
            if column is not None:
                features[row, column] = 1.0
    transform = {
        "schema": "loslab.metadata_only_vocabulary.v1",
        "algorithm": "sorted_training_group_domain_source_one_hot_unknown_zero_v1",
        "implementation_sha256": hashlib.sha256(
            inspect.getsource(_metadata_only_features).encode("utf-8")
        ).hexdigest(),
        "fields": ["domain", "group", "source"],
        "vocabulary": [[name, value] for name, value in vocabulary],
        "vocabulary_sha256": canonical_hash([[name, value] for name, value in vocabulary]),
        "group_values_sha256": _array_sha256(np.asarray(fields["group"], dtype=str)),
        "domain_values_sha256": _array_sha256(np.asarray(fields["domain"], dtype=str)),
        "source_values_sha256": _array_sha256(np.asarray(fields["source"], dtype=str)),
        "training_indices_sha256": _array_sha256(train_indices),
        "max_feature_cells": max_feature_cells,
        "unknown_category_behavior": "all_zero_vector_for_unknown_field_value",
    }
    return features, transform


def _array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    header = json.dumps(
        {"dtype": value.dtype.str, "shape": list(value.shape)},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _feature_transform_implementation_sha256(method: str) -> str:
    if method in ("cnn", "cvnn"):
        transform = _identity_complex_cfr
    elif method == "source_only":
        transform = _source_only_features
    elif method == "metadata_only":
        transform = _metadata_only_features
    elif method in ("raw_mlp", "label_permuted", "row_shuffled"):
        transform = _flatten_real_imag_cfr
    elif method == "power_only":
        transform = _log_power_per_link_subcarrier
    elif method == "phase_randomized":
        transform = _phase_randomized_real_imag_cfr
    else:
        transform = observable_features
    return hashlib.sha256(inspect.getsource(transform).encode("utf-8")).hexdigest()


def _stable_single_link_sha256(path: str | Path) -> str:
    """Hash one stable regular-file snapshot without following file aliases."""
    file_path = Path(path)
    if not hasattr(os, "O_NOFOLLOW"):
        raise ValueError("platform lacks safe single-link model artifact opening")
    path_before = os.lstat(file_path)
    if not stat.S_ISREG(path_before.st_mode) or path_before.st_nlink != 1:
        raise ValueError("model artifact must be a single-link regular file")
    descriptor = os.open(file_path, os.O_RDONLY | os.O_NOFOLLOW)
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError("model artifact must be a single-link regular file")
        if (path_before.st_dev, path_before.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError("model artifact path changed before opening")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        after = os.fstat(descriptor)
        path_after = os.lstat(file_path)
    finally:
        os.close(descriptor)

    def identity(value: os.stat_result) -> tuple[int, ...]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_nlink,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    if identity(before) != identity(after) or identity(before) != identity(path_after):
        raise ValueError("model artifact changed during hashing")
    return digest.hexdigest()


def build_model_input_artifact_contract(
    *,
    method: str,
    method_input: np.ndarray,
    include_phase: bool,
    config_sha256: str,
    source_revision: str | None,
    input_file_digests: list[dict[str, Any]],
    split: dict[str, Any],
    model_file_sha256: str,
    partition_indices: dict[str, np.ndarray],
    target_values: np.ndarray | None = None,
    label_permutation_seed: int | None = None,
    label_permutation_run_seed: int | None = None,
    source_method_input: np.ndarray | None = None,
    feature_permutation_seed: int | None = None,
    feature_permutation_run_seed: int | None = None,
    source_values: np.ndarray | None = None,
    metadata_values: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Bind a saved model to its audited observation-only input route."""
    if method not in _MODEL_METHODS:
        raise ValueError(f"unsupported model-input artifact method: {method!r}")
    if not isinstance(method_input, np.ndarray) or method_input.ndim < 2:
        raise ValueError("model input must be a batched feature array")
    if not np.issubdtype(method_input.dtype, np.number) or not np.isfinite(method_input).all():
        raise ValueError("model input must contain finite numeric values")
    required_partitions = {"train", "validation", "test"}
    if not required_partitions <= set(partition_indices) or set(partition_indices) - (
        required_partitions | {"calibration"}
    ):
        raise ValueError("model input partitions must include train, validation, and test")
    checked_indices: dict[str, np.ndarray] = {}
    for name, indices in partition_indices.items():
        if (
            not isinstance(indices, np.ndarray)
            or indices.ndim != 1
            or not np.issubdtype(indices.dtype, np.integer)
            or np.issubdtype(indices.dtype, np.bool_)
            or np.any(indices < 0)
            or np.any(indices >= len(method_input))
        ):
            raise ValueError(f"model input {name} partition indices are invalid")
        checked_indices[name] = indices
    all_indices = np.concatenate(tuple(checked_indices.values()))
    if len(np.unique(all_indices)) != len(all_indices):
        raise ValueError("model input partitions must be mutually disjoint")
    source_feature, transform = _MODEL_FEATURE_ROUTES.get(
        method, ("observed_cfr", "logmag_pdp_moments_v1")
    )
    input_contract_version = (
        "v8"
        if method == "metadata_only"
        else "v7"
        if method == "source_only"
        else "v6"
        if method == "row_shuffled"
        else "v5"
        if method == "label_permuted"
        else "v4"
        if method == "phase_randomized"
        else "v3"
        if method == "power_only"
        else "v2"
    )
    feature_contract = {
        "schema": f"loslab.model_input_feature_contract.{input_contract_version}",
        "method": method,
        "source_dataclass": (
            "datasets.channel.ChannelDataset"
            if method in {"source_only", "metadata_only"}
            else "datasets.channel.ChannelModelInputs"
        ),
        "allowed_source_fields": (
            ["group", "domain", "source"] if method == "metadata_only" else [source_feature]
        ),
        "excluded_truth_and_metadata_fields": (
            [
                "data.y",
                "data.metadata",
                "data.group",
                "data.domain",
                "observed_cfr",
                "PathOracleTable",
                "target_values",
            ]
            if method == "source_only"
            else [
                "data.y",
                "data.metadata",
                "observed_cfr",
                "PathOracleTable",
                "target_values",
            ]
            if method == "metadata_only"
            else [
                "data.y",
                "data.metadata",
                "data.group",
                "data.domain",
                "PathOracleTable",
                "target_values",
            ]
        ),
        "target_field_used_for_supervision_only": "data.y",
        "feature_transform": transform,
        "feature_transform_implementation_sha256": _feature_transform_implementation_sha256(method),
        "include_phase": bool(include_phase) if transform == "logmag_pdp_moments_v1" else None,
        "feature_dtype": str(method_input.dtype),
        "feature_shape_per_sample": [int(value) for value in method_input.shape[1:]],
        "feature_matrix": {
            "sample_count": int(method_input.shape[0]),
            "dtype": method_input.dtype.str,
            "shape_per_sample": [int(value) for value in method_input.shape[1:]],
            "sha256": _array_sha256(method_input),
        },
        "partition_feature_matrices": {
            name: {
                "sample_count": int(len(indices)),
                "sha256": _array_sha256(method_input[indices]),
            }
            for name, indices in partition_indices.items()
        },
        "source_revision": source_revision,
        "run_config_sha256": config_sha256,
    }
    if method == "label_permuted":
        if (
            target_values is None
            or not isinstance(target_values, np.ndarray)
            or target_values.ndim != 1
            or len(target_values) != len(method_input)
            or label_permutation_seed is None
            or label_permutation_run_seed is None
        ):
            raise ValueError(
                "label-permuted contracts require full target values and a permutation seed"
            )
        train_targets = target_values[checked_indices["train"]]
        _, feature_contract["supervision_transform"] = _permute_training_labels(
            train_targets,
            run_seed=label_permutation_run_seed,
            base_seed=label_permutation_seed,
        )
    elif (
        target_values is not None
        or label_permutation_seed is not None
        or label_permutation_run_seed is not None
    ):
        raise ValueError("target permutation inputs are only valid for the label_permuted method")
    if method == "row_shuffled":
        if (
            not isinstance(source_method_input, np.ndarray)
            or source_method_input.ndim != 2
            or source_method_input.shape != method_input.shape
            or source_method_input.dtype != method_input.dtype
            or feature_permutation_seed is None
            or feature_permutation_run_seed is None
        ):
            raise ValueError("row-shuffled contracts require source features and permutation seeds")
        shuffled, transform_contract = _permute_training_feature_rows(
            source_method_input[checked_indices["train"]],
            run_seed=feature_permutation_run_seed,
            base_seed=feature_permutation_seed,
        )
        expected_input = np.array(source_method_input, copy=True)
        expected_input[checked_indices["train"]] = shuffled
        if not np.array_equal(expected_input, method_input):
            raise ValueError("row-shuffled model input differs from its declared permutation")
        feature_contract["feature_row_permutation"] = transform_contract
    elif (
        source_method_input is not None
        or feature_permutation_seed is not None
        or feature_permutation_run_seed is not None
    ):
        raise ValueError("feature-row permutation inputs are only valid for row_shuffled")
    if method == "source_only":
        if not isinstance(source_values, np.ndarray) or len(source_values) != len(method_input):
            raise ValueError("source-only contracts require the complete source identifier roster")
        expected_features, source_transform = _source_only_features(
            source_values, train_indices=checked_indices["train"]
        )
        if not np.array_equal(expected_features, method_input):
            raise ValueError("source-only features differ from the training vocabulary transform")
        feature_contract["source_vocabulary_transform"] = source_transform
    elif source_values is not None:
        raise ValueError("source identifiers are only valid for the source_only method")
    if method == "metadata_only":
        expected_names = {"group", "domain", "source"}
        if not isinstance(metadata_values, dict) or set(metadata_values) != expected_names:
            raise ValueError("metadata-only contracts require group, domain, and source rosters")
        expected_features, metadata_transform = _metadata_only_features(
            group_values=metadata_values["group"],
            domain_values=metadata_values["domain"],
            source_values=metadata_values["source"],
            train_indices=checked_indices["train"],
        )
        if not np.array_equal(expected_features, method_input):
            raise ValueError("metadata-only features differ from the training vocabulary transform")
        feature_contract["metadata_vocabulary_transform"] = metadata_transform
    elif metadata_values is not None:
        raise ValueError("metadata identifiers are only valid for the metadata_only method")
    feature_contract_sha256 = canonical_hash(feature_contract)
    return {
        "schema": f"loslab.model_input_artifact_contract.{input_contract_version}",
        "model_sha256": model_file_sha256,
        "feature_contract": feature_contract,
        "feature_contract_sha256": feature_contract_sha256,
        "training_input_files": input_file_digests,
        "split": split,
    }


def verify_model_input_artifact_contract(
    model_info_path: str | Path,
    model_path: str | Path,
    *,
    method_input: np.ndarray | None = None,
    partition_indices: dict[str, np.ndarray] | None = None,
    target_values: np.ndarray | None = None,
    source_method_input: np.ndarray | None = None,
    source_values: np.ndarray | None = None,
    metadata_values: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Verify that a saved model is bound to a canonical observation-only input contract."""
    sidecar_path = Path(model_info_path)
    if not hasattr(os, "O_NOFOLLOW"):
        raise ValueError("platform lacks safe model input artifact sidecar opening")
    before_path = os.lstat(sidecar_path)
    if not stat.S_ISREG(before_path.st_mode) or before_path.st_nlink != 1:
        raise ValueError("model input artifact sidecar must be a single-link regular file")
    descriptor = os.open(sidecar_path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > 1_048_576:
            raise ValueError("model input artifact sidecar exceeds its safe file contract")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            sidecar_bytes = stream.read(1_048_577)
        after = os.fstat(descriptor)
        after_path = os.lstat(sidecar_path)
    finally:
        os.close(descriptor)

    def file_identity(value: os.stat_result) -> tuple[int, ...]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_nlink,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    if (
        len(sidecar_bytes) > 1_048_576
        or len(sidecar_bytes) != before.st_size
        or file_identity(before) != file_identity(after)
        or file_identity(before) != file_identity(after_path)
    ):
        raise ValueError("model input artifact sidecar changed during bounded read")
    try:
        sidecar = json.loads(sidecar_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("model input artifact sidecar must be readable UTF-8 JSON") from error
    if not isinstance(sidecar, dict) or not isinstance(
        sidecar.get("model_input_artifact_contract"), dict
    ):
        raise ValueError("model input artifact sidecar lacks its contract")
    contract = sidecar["model_input_artifact_contract"]
    import jsonschema

    contract_schema = contract.get("schema")
    schema_version = {
        "loslab.model_input_artifact_contract.v1": "v1",
        "loslab.model_input_artifact_contract.v2": "v2",
        "loslab.model_input_artifact_contract.v3": "v3",
        "loslab.model_input_artifact_contract.v4": "v4",
        "loslab.model_input_artifact_contract.v5": "v5",
        "loslab.model_input_artifact_contract.v6": "v6",
        "loslab.model_input_artifact_contract.v7": "v7",
        "loslab.model_input_artifact_contract.v8": "v8",
    }.get(contract_schema)
    if schema_version is None:
        raise ValueError("model input artifact contract schema is unsupported")
    schema_path = Path(__file__).resolve().parents[1] / (
        f"datasets/schema/model-input-artifact-contract.{schema_version}.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    schema_errors = list(jsonschema.Draft202012Validator(schema).iter_errors(contract))
    if schema_errors:
        raise ValueError(
            "model input artifact contract violates schema: "
            + "; ".join(error.message for error in schema_errors[:10])
        )
    if (
        set(contract)
        != {
            "schema",
            "model_sha256",
            "feature_contract",
            "feature_contract_sha256",
            "training_input_files",
            "split",
        }
        or contract.get("schema") != contract_schema
    ):
        raise ValueError("model input artifact contract fields or schema are invalid")
    feature = contract.get("feature_contract")
    expected_feature_fields = {
        "schema",
        "method",
        "source_dataclass",
        "allowed_source_fields",
        "excluded_truth_and_metadata_fields",
        "target_field_used_for_supervision_only",
        "feature_transform",
        "include_phase",
        "feature_dtype",
        "feature_shape_per_sample",
        "source_revision",
        "run_config_sha256",
    }
    if contract_schema in {
        "loslab.model_input_artifact_contract.v2",
        "loslab.model_input_artifact_contract.v3",
        "loslab.model_input_artifact_contract.v4",
        "loslab.model_input_artifact_contract.v5",
        "loslab.model_input_artifact_contract.v6",
        "loslab.model_input_artifact_contract.v7",
        "loslab.model_input_artifact_contract.v8",
    }:
        expected_feature_fields.update(
            {
                "feature_transform_implementation_sha256",
                "feature_matrix",
                "partition_feature_matrices",
            }
        )
    if contract_schema == "loslab.model_input_artifact_contract.v5":
        expected_feature_fields.add("supervision_transform")
    if contract_schema == "loslab.model_input_artifact_contract.v6":
        expected_feature_fields.add("feature_row_permutation")
    if contract_schema == "loslab.model_input_artifact_contract.v7":
        expected_feature_fields.add("source_vocabulary_transform")
    if contract_schema == "loslab.model_input_artifact_contract.v8":
        expected_feature_fields.add("metadata_vocabulary_transform")
    if not isinstance(feature, dict) or set(feature) != expected_feature_fields:
        raise ValueError("model feature contract fields are invalid")
    method = feature.get("method")
    expected_format = (
        "loslab.sklearn.v1"
        if method
        in {
            "rf",
            "histgb",
            "xgb",
            "mlp",
            "raw_mlp",
            "power_only",
            "phase_randomized",
            "label_permuted",
            "row_shuffled",
            "source_only",
            "metadata_only",
            "logreg",
        }
        else "loslab.model_input_artifact.v1"
    )
    include_phase = sidecar.get("include_phase")
    threshold = sidecar.get("threshold")
    if (
        sidecar.get("format") != expected_format
        or sidecar.get("method") != method
        or not isinstance(include_phase, bool)
        or isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not np.isfinite(float(threshold))
        or not 0.0 <= float(threshold) <= 1.0
        or feature.get("include_phase")
        != (include_phase if feature.get("feature_transform") == "logmag_pdp_moments_v1" else None)
    ):
        raise ValueError("model info wrapper disagrees with the feature contract")
    expected_route = _MODEL_FEATURE_ROUTES.get(method, ("observed_cfr", "logmag_pdp_moments_v1"))
    expected_feature_schema = f"loslab.model_input_feature_contract.{schema_version}"
    if (
        feature.get("schema") != expected_feature_schema
        or method not in _MODEL_METHODS
        or feature.get("source_dataclass")
        != (
            "datasets.channel.ChannelDataset"
            if method in {"source_only", "metadata_only"}
            else "datasets.channel.ChannelModelInputs"
        )
        or feature.get("allowed_source_fields")
        != (["group", "domain", "source"] if method == "metadata_only" else [expected_route[0]])
        or feature.get("excluded_truth_and_metadata_fields")
        != (
            [
                "data.y",
                "data.metadata",
                "data.group",
                "data.domain",
                "observed_cfr",
                "PathOracleTable",
                "target_values",
            ]
            if method == "source_only"
            else [
                "data.y",
                "data.metadata",
                "observed_cfr",
                "PathOracleTable",
                "target_values",
            ]
            if method == "metadata_only"
            else [
                "data.y",
                "data.metadata",
                "data.group",
                "data.domain",
                "PathOracleTable",
                "target_values",
            ]
        )
        or feature.get("feature_transform") != expected_route[1]
        or feature.get("target_field_used_for_supervision_only") != "data.y"
        or "data.y" not in feature.get("excluded_truth_and_metadata_fields", [])
        or not isinstance(feature.get("run_config_sha256"), str)
        or len(feature["run_config_sha256"]) != 64
        or any(char not in "0123456789abcdef" for char in feature["run_config_sha256"])
    ):
        raise ValueError("model feature contract does not match an audited observation-only route")
    if canonical_hash(feature) != contract.get("feature_contract_sha256"):
        raise ValueError("model feature contract digest does not match its contents")
    if _stable_single_link_sha256(model_path) != contract.get("model_sha256"):
        raise ValueError("model bytes do not match the input artifact contract")
    feature_matrix_verified = False
    if contract_schema in {
        "loslab.model_input_artifact_contract.v2",
        "loslab.model_input_artifact_contract.v3",
        "loslab.model_input_artifact_contract.v4",
        "loslab.model_input_artifact_contract.v5",
        "loslab.model_input_artifact_contract.v6",
        "loslab.model_input_artifact_contract.v7",
        "loslab.model_input_artifact_contract.v8",
    }:
        if method_input is not None:
            matrix = feature["feature_matrix"]
            if (
                not isinstance(method_input, np.ndarray)
                or method_input.shape[0] != matrix["sample_count"]
                or list(method_input.shape[1:]) != matrix["shape_per_sample"]
                or method_input.dtype.str != matrix["dtype"]
                or _array_sha256(method_input) != matrix["sha256"]
            ):
                raise ValueError("model feature matrix differs from its frozen contract")
            if (
                partition_indices is None
                or not {"train", "validation", "test"} <= set(partition_indices)
                or set(partition_indices) - {"train", "validation", "test", "calibration"}
            ):
                raise ValueError("model feature matrix verification requires all split indices")
            index_arrays = tuple(partition_indices.values())
            if any(
                not isinstance(indices, np.ndarray) or indices.ndim != 1 for indices in index_arrays
            ):
                raise ValueError("model feature matrix verification indices are malformed")
            all_indices = np.concatenate(index_arrays)
            if len(np.unique(all_indices)) != len(all_indices):
                raise ValueError("model feature matrix verification partitions overlap")
            for name, indices in partition_indices.items():
                if (
                    not isinstance(indices, np.ndarray)
                    or indices.ndim != 1
                    or not np.issubdtype(indices.dtype, np.integer)
                    or np.issubdtype(indices.dtype, np.bool_)
                    or np.any(indices < 0)
                    or np.any(indices >= len(method_input))
                ):
                    raise ValueError(f"{name} feature matrix verification indices are invalid")
                expected = feature["partition_feature_matrices"][name]
                if (
                    len(indices) != expected["sample_count"]
                    or _array_sha256(method_input[indices]) != expected["sha256"]
                ):
                    raise ValueError(f"{name} feature matrix differs from its frozen contract")
            feature_matrix_verified = True
    if contract_schema == "loslab.model_input_artifact_contract.v5":
        if method != "label_permuted" or target_values is None:
            raise ValueError("label-permuted contract verification requires its target values")
        if (
            method_input is None
            or partition_indices is None
            or not isinstance(target_values, np.ndarray)
            or target_values.ndim != 1
            or len(target_values) != len(method_input)
        ):
            raise ValueError("label-permuted verification inputs are incomplete")
        training_targets = target_values[partition_indices["train"]]
        permutation = feature.get("supervision_transform")
        if not isinstance(permutation, dict):
            raise ValueError("label-permuted feature contract lacks its supervision transform")
        _, replayed_permutation = _permute_training_labels(
            training_targets,
            run_seed=permutation.get("run_seed"),
            base_seed=permutation.get("base_seed"),
        )
        if replayed_permutation != permutation:
            raise ValueError("label permutation does not replay from the retained training labels")
    if contract_schema == "loslab.model_input_artifact_contract.v6":
        if method != "row_shuffled" or source_method_input is None:
            raise ValueError("row-shuffled contract verification requires source features")
        if (
            method_input is None
            or partition_indices is None
            or not isinstance(source_method_input, np.ndarray)
            or source_method_input.shape != method_input.shape
            or source_method_input.dtype != method_input.dtype
        ):
            raise ValueError("row-shuffled verification inputs are incomplete")
        permutation = feature.get("feature_row_permutation")
        if not isinstance(permutation, dict):
            raise ValueError("row-shuffled feature contract lacks its permutation")
        shuffled, replayed_permutation = _permute_training_feature_rows(
            source_method_input[partition_indices["train"]],
            run_seed=permutation.get("run_seed"),
            base_seed=permutation.get("base_seed"),
        )
        replayed_input = np.array(source_method_input, copy=True)
        replayed_input[partition_indices["train"]] = shuffled
        if replayed_permutation != permutation or not np.array_equal(replayed_input, method_input):
            raise ValueError("feature-row permutation does not replay from source observations")
    if contract_schema == "loslab.model_input_artifact_contract.v7":
        if method != "source_only" or source_values is None:
            raise ValueError("source-only contract verification requires source identifiers")
        if (
            method_input is None
            or partition_indices is None
            or not isinstance(source_values, np.ndarray)
            or source_values.ndim != 1
            or len(source_values) != len(method_input)
        ):
            raise ValueError("source-only verification inputs are incomplete")
        features, replayed_transform = _source_only_features(
            source_values, train_indices=partition_indices["train"]
        )
        if replayed_transform != feature.get("source_vocabulary_transform") or not np.array_equal(
            features, method_input
        ):
            raise ValueError("source-only feature vocabulary does not replay from training IDs")
    if contract_schema == "loslab.model_input_artifact_contract.v8":
        if method != "metadata_only" or metadata_values is None:
            raise ValueError("metadata-only contract verification requires metadata ID rosters")
        if (
            method_input is None
            or partition_indices is None
            or not isinstance(metadata_values, dict)
            or set(metadata_values) != {"group", "domain", "source"}
        ):
            raise ValueError("metadata-only verification inputs are incomplete")
        features, replayed_transform = _metadata_only_features(
            group_values=metadata_values["group"],
            domain_values=metadata_values["domain"],
            source_values=metadata_values["source"],
            train_indices=partition_indices["train"],
        )
        if replayed_transform != feature.get("metadata_vocabulary_transform") or not np.array_equal(
            features, method_input
        ):
            raise ValueError("metadata-only feature vocabulary does not replay from training IDs")
    files = contract.get("training_input_files")
    if (
        not isinstance(files, list)
        or not files
        or any(
            not isinstance(row, dict)
            or set(row) != {"path", "sha256"}
            or not isinstance(row["path"], str)
            or not isinstance(row["sha256"], str)
            or len(row["sha256"]) != 64
            or any(char not in "0123456789abcdef" for char in row["sha256"])
            for row in files
        )
    ):
        raise ValueError("model input contract has an invalid training file roster")
    split = contract.get("split")
    required_split_fields = {
        "mode",
        "registry_sha256",
        "indices_sha256",
        "train_rows",
        "validation_rows",
        "test_rows",
    }
    if (
        not isinstance(split, dict)
        or not required_split_fields <= set(split)
        or set(split) - (required_split_fields | {"calibration_rows"})
    ):
        raise ValueError("model input contract split binding is invalid")
    for count_key in ("train_rows", "validation_rows", "test_rows"):
        count = split[count_key]
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("model input contract split counts must be positive integers")
    if "calibration_rows" in split and (
        isinstance(split["calibration_rows"], bool)
        or not isinstance(split["calibration_rows"], int)
        or split["calibration_rows"] <= 0
    ):
        raise ValueError("model input contract calibration split count must be positive")
    for digest_key in ("indices_sha256",):
        digest = split[digest_key]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError("model input contract split digest is invalid")
    registry_digest = split["registry_sha256"]
    if registry_digest is not None and (
        not isinstance(registry_digest, str)
        or len(registry_digest) != 64
        or any(char not in "0123456789abcdef" for char in registry_digest)
    ):
        raise ValueError("model input contract registry digest is invalid")
    return {
        "status": "PASS_MODEL_INPUT_ARTIFACT_BINDING",
        "model_sha256": contract["model_sha256"],
        "feature_contract_sha256": contract["feature_contract_sha256"],
        "feature_matrix_verified": feature_matrix_verified,
    }


def _validation_threshold_selection_record(
    labels: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
    row_indices: np.ndarray,
) -> dict[str, Any]:
    """Select a validation-only operating point and bind its exact evidence."""
    if (
        labels.ndim != 1
        or scores.shape != labels.shape
        or groups.shape != labels.shape
        or row_indices.shape != labels.shape
        or len(labels) == 0
    ):
        raise ValueError("validation threshold evidence arrays must be nonempty and aligned")
    threshold = threshold_from_validation(labels, scores)
    evidence_sha256 = canonical_hash(
        {
            "row_indices": row_indices.tolist(),
            "group_ids": groups.astype(str).tolist(),
            "labels": labels.astype(int).tolist(),
            "positive_label_scores": scores.astype(float).tolist(),
        }
    )
    return {
        **_VALIDATION_THRESHOLD_POLICY,
        "threshold": threshold,
        "validation_row_count": int(len(labels)),
        "validation_group_count": int(len(np.unique(groups))),
        "validation_evidence_sha256": evidence_sha256,
    }


def verify_validation_threshold_selection_record(
    record: Any,
    labels: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
    row_indices: np.ndarray,
) -> dict[str, Any]:
    """Recompute the threshold decision and bind it to exact validation evidence."""
    if not isinstance(record, dict):
        raise TypeError("threshold selection record must be an object")
    expected = _validation_threshold_selection_record(labels, scores, groups, row_indices)
    if canonical_hash(record) != canonical_hash(expected):
        raise ValueError("threshold selection record differs from validation-only recomputation")
    return expected


def _label_semantics_by_source(data: ChannelDataset) -> str:
    semantics: dict[str, set[str]] = {}
    for source, metadata in zip(data.source, data.metadata, strict=True):
        source_name = str(source)
        label_semantic = metadata.get("geometry_label") or metadata.get("label_semantics")
        if label_semantic is None:
            label_semantic = (
                "SYNTHETIC_DEBUG_CONSTRUCTED"
                if source_name == "synthetic_debug"
                else f"UNSPECIFIED({source_name})"
            )
        semantics.setdefault(source_name, set()).add(str(label_semantic))
    if len(semantics) == 1:
        values = next(iter(semantics.values()))
        if len(values) == 1:
            return next(iter(values))
    return "; ".join(
        f"{source}={'|'.join(sorted(values))}" for source, values in sorted(semantics.items())
    )


def _normalized_label_semantics(value: str) -> str:
    normalized = value.strip().upper().replace("-", "_").replace(" ", "_")
    aliases = {
        "GEOMETRIC_DIRECT_PATH_ORACLE": "geometric_direct_path_oracle",
        "PROFILE_PROXY": "profile_proxy",
        "SYNTHETIC_DEBUG_CONSTRUCTED": "synthetic_debug_direct_component",
        "SYNTHETIC_DEBUG_DIRECT_COMPONENT": "synthetic_debug_direct_component",
        "UNSPECIFIED": "unspecified",
        "UNKNOWN": "unspecified",
        "UNKNOWN_LEGACY": "unspecified",
    }
    return aliases.get(normalized, value.strip().lower().replace(" ", "_"))


def _resolved_label_semantics(data: ChannelDataset) -> str:
    """Resolve one target meaning and refuse to score mixed or unknown legacy labels."""
    row_semantics: set[str] = set()
    has_unresolved_row = False
    for source, metadata in zip(data.source, data.metadata, strict=True):
        raw = metadata.get("geometry_label") or metadata.get("label_semantics")
        if raw is None and str(source) == "synthetic_debug":
            raw = "synthetic_debug_direct_component"
        if raw is None:
            has_unresolved_row = True
        else:
            normalized = _normalized_label_semantics(str(raw))
            if normalized == "unspecified":
                has_unresolved_row = True
            else:
                row_semantics.add(normalized)

    declared = data.label_semantics
    if declared != "unspecified_legacy":
        if row_semantics and row_semantics != {declared}:
            raise ValueError(
                "dataset label_semantics conflicts with per-sample label metadata: "
                f"declared={declared!r}, observed={sorted(row_semantics)}"
            )
        return declared
    if has_unresolved_row or len(row_semantics) != 1:
        raise ValueError(
            "cannot evaluate unspecified legacy labels without one explicit, consistent "
            "label meaning; regenerate the dataset or provide per-sample label metadata"
        )
    return next(iter(row_semantics))


def _precomputed_observation_semantics(data: ChannelDataset) -> str:
    semantics = sorted(
        {
            str(metadata.get("receiver_observation"))
            for metadata in data.metadata
            if metadata.get("receiver_observation")
        }
    )
    return "; ".join(semantics) if semantics else "precomputed channel dataset observations"


def _sync_torch_device(device: str, enabled: bool) -> None:
    if enabled and device.startswith("cuda"):
        import torch

        torch.cuda.synchronize(device)


def _learned_parameter_count(model: Any, training: dict[str, Any]) -> int | None:
    reported = training.get("trainable_parameters")
    if isinstance(reported, int) and not isinstance(reported, bool):
        return reported
    estimator = model.steps[-1][1] if hasattr(model, "steps") else model
    coefficients = getattr(estimator, "coefs_", None)
    intercepts = getattr(estimator, "intercepts_", None)
    if coefficients is not None and intercepts is not None:
        return int(sum(array.size for array in (*coefficients, *intercepts)))
    return None


def _locked_split_indices(
    data: ChannelDataset, registry: SplitPolicyRegistry
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Resolve every channel row through its frozen split parent or child ID."""
    split_for_row: list[str] = []
    represented_parents: set[str] = set()
    group_split: dict[str, str] = {}
    for group, metadata in zip(data.group, data.metadata, strict=True):
        subject_id = metadata.get("split_subject_id")
        if not isinstance(subject_id, str) or not subject_id or subject_id != subject_id.strip():
            raise ValueError(
                "signed split evaluation requires a nonempty canonical split_subject_id per row"
            )
        if subject_id in registry.subjects:
            parent_id = subject_id
            partition = registry.split_for_parent(subject_id)
        elif subject_id in registry.child_bindings:
            parent_id = registry.child_bindings[subject_id].parent_subject_id
            partition = registry.split_for_child(subject_id)
        else:
            raise ValueError(f"dataset split subject is absent from signed lock: {subject_id}")
        represented_parents.add(parent_id)
        if partition not in {"train", "validation", "test", "calibration"}:
            raise ValueError(f"unsupported locked split partition: {partition}")
        if not isinstance(group, str) or not group or group != group.strip():
            raise ValueError("dataset group IDs must be nonempty canonical strings")
        group_key = group
        prior = group_split.setdefault(group_key, partition)
        if prior != partition:
            raise ValueError(f"dataset group crosses signed split partitions: {group_key}")
        split_for_row.append(partition)

    if represented_parents != set(registry.subjects):
        missing = sorted(set(registry.subjects) - represented_parents)
        raise ValueError(
            f"dataset does not cover every signed split parent ({len(missing)} missing)"
        )
    labels = np.asarray(split_for_row)
    train = np.flatnonzero(labels == "train")
    validation = np.flatnonzero(labels == "validation")
    test = np.flatnonzero(labels == "test")
    calibration = np.flatnonzero(labels == "calibration")
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError(
            "signed split lock must resolve non-empty train, validation, and test rows"
        )
    report = {
        "mode": "signed_locked",
        "policy_sha256": registry.policy.to_dict()["policy_sha256"],
        "registry_sha256": registry.to_dict()["registry_sha256"],
        "parent_counts": {
            split: sum(registry.split_for_parent(parent) == split for parent in represented_parents)
            for split in registry.policy.split_fractions
        },
        "calibration_rows_reserved": int(np.sum(labels == "calibration")),
    }
    return train, validation, test, calibration, report


def run_experiment(
    paths: Sequence[str | Path],
    output: str | Path,
    *,
    holdout: str | None = None,
    methods: Sequence[str] = ("rf", "histgb", "mlp", "raw_mlp", "coral", "coral0"),
    seeds: Sequence[int] = (11, 23, 47, 59, 71, 83, 97, 101, 109, 127),
    bootstrap: int = 200,
    include_phase: bool = False,
    observation: str = "ideal_cfr",
    pilot_spacing: int = 4,
    pilot_snr_db: float = 12.0,
    observation_seed: int = 2026,
    phase_randomization_seed: int = 2027,
    label_permutation_seed: int = 2028,
    row_shuffle_seed: int = 2029,
    device: str = "cpu",
    coral_penalty: float = 0.1,
    capacity: str = "medium",
    split_lock_path: str | Path | None = None,
    split_signature_path: str | Path | None = None,
    trusted_split_signers_path: str | Path | None = None,
    split_signer_id: str | None = None,
    build_cache_root: str | Path | None = None,
    prior_shift: dict[str, Any] | None = None,
    low_confidence_margin: float | None = None,
) -> dict:
    if not paths:
        raise ValueError("No datasets supplied")
    if (
        isinstance(phase_randomization_seed, bool)
        or not isinstance(phase_randomization_seed, (int, np.integer))
        or not 0 <= phase_randomization_seed < 2**63
    ):
        raise ValueError("phase_randomization_seed must be an integer in [0, 2**63)")
    if (
        isinstance(label_permutation_seed, bool)
        or not isinstance(label_permutation_seed, (int, np.integer))
        or not 0 <= label_permutation_seed < 2**63
    ):
        raise ValueError("label_permutation_seed must be an integer in [0, 2**63)")
    if (
        isinstance(row_shuffle_seed, bool)
        or not isinstance(row_shuffle_seed, (int, np.integer))
        or not 0 <= row_shuffle_seed < 2**63
    ):
        raise ValueError("row_shuffle_seed must be an integer in [0, 2**63)")
    resolved_prior_shift = None if prior_shift is None else validate_prior_shift_spec(prior_shift)
    if resolved_prior_shift is not None:
        _validate_evaluation_payload(resolved_prior_shift, "prior-shift.v1.schema.json")
    if low_confidence_margin is not None and (
        isinstance(low_confidence_margin, bool)
        or not isinstance(low_confidence_margin, (int, float))
        or not np.isfinite(low_confidence_margin)
        or not 0 <= low_confidence_margin < 0.5
    ):
        raise ValueError("low_confidence_margin must be finite and in [0, 0.5)")
    capacity_spec(capacity)
    dataset_build_node = None
    if build_cache_root is None:
        data = ChannelDataset.concatenate([ChannelDataset.load(p) for p in paths])
    else:
        data, dataset_build_node = _materialize_channel_dataset_build_node(paths, build_cache_root)
    signed_split_args = (
        split_lock_path,
        split_signature_path,
        trusted_split_signers_path,
        split_signer_id,
    )
    if any(value is not None for value in signed_split_args) and not all(
        value is not None for value in signed_split_args
    ):
        raise ValueError(
            "signed split mode requires lock, signature, trusted keyring, and signer ID"
        )
    locked_indices = None
    split_provenance: dict[str, Any] = {"mode": "unlocked_pilot"}
    if all(value is not None for value in signed_split_args):
        if holdout is not None:
            raise ValueError("holdout cannot be combined with a signed split lock")
        lock_path = Path(split_lock_path)  # type: ignore[arg-type]
        signature_path = Path(split_signature_path)  # type: ignore[arg-type]
        signers_path = Path(trusted_split_signers_path)  # type: ignore[arg-type]
        registry = verify_signed_split_policy_registry(
            lock_path,
            signature_path,
            trusted_allowed_signers_path=signers_path,
            signer_id=str(split_signer_id),
        )
        locked_indices = _locked_split_indices(data, registry)
        split_provenance = {
            **locked_indices[4],
            "signer_id": split_signer_id,
            "lock_file_sha256": sha256_file(lock_path),
            "signature_file_sha256": sha256_file(signature_path),
            "trusted_keyring_sha256": sha256_file(signers_path),
        }
    resolved_label_semantics = _resolved_label_semantics(data)
    model_inputs = data.model_inputs()
    if observation == "ideal_cfr":
        observable = model_inputs.observed_cfr
    elif observation == "comb_ls":
        observable = comb_ls_estimate(
            model_inputs.observed_cfr,
            spacing=pilot_spacing,
            snr_db=pilot_snr_db,
            seed=observation_seed,
        )
    elif observation == "precomputed":
        observable = model_inputs.observed_cfr
    else:
        raise ValueError("observation must be ideal_cfr, comb_ls, or precomputed")
    x = observable_features(observable, include_phase=include_phase)
    raw_real_imag = _flatten_real_imag_cfr(observable)
    power_only = _log_power_per_link_subcarrier(observable)
    phase_randomized = _phase_randomized_real_imag_cfr(
        observable, seed=int(phase_randomization_seed)
    )
    layout = RunLayout.at(output)
    layout.ensure()
    output = layout.root
    config = {
        "holdout": holdout,
        "methods": list(methods),
        "seeds": list(seeds),
        "bootstrap": bootstrap,
        "include_phase": include_phase,
        "device": device,
        "capacity": capacity,
        "coral_penalty": coral_penalty,
        "observation": observation,
        "pilot_spacing": pilot_spacing,
        "pilot_snr_db": pilot_snr_db,
        "observation_seed": observation_seed,
        "phase_randomization_seed": int(phase_randomization_seed),
        "label_permutation_seed": int(label_permutation_seed),
        "row_shuffle_seed": int(row_shuffle_seed),
        "feature_transform_by_method": {
            method: _MODEL_FEATURE_ROUTES.get(method, ("observed_cfr", "logmag_pdp_moments_v1"))[1]
            for method in methods
        },
        "raw_mlp_input_representation": "flattened real/imaginary CFR channels",
        "power_only_input_representation": (
            "per-link/per-subcarrier channel power in dB with a -120 dB numerical floor"
        ),
        "phase_randomized_input_representation": (
            "flattened real/imaginary CFR after independent seeded phase per sample, link, and subcarrier"
        ),
        "label_permuted_supervision": (
            "training labels only are permuted with a deterministic seed derived from the run and label-permutation seeds; "
            "validation, calibration, and test labels remain unchanged"
        ),
        "row_shuffled_training_features": (
            "training observed-CFR feature rows are deterministically permuted while their labels and all evaluation partitions remain unchanged"
        ),
        "source_only_control": (
            "one-hot source identifiers with a vocabulary fit only on training rows; unseen evaluation identifiers map to all zeros"
        ),
        "metadata_only_control": (
            "bounded training-vocabulary one-hot group/domain/source identifiers; this is an intentional metadata-leakage diagnostic, not a receiver feature"
        ),
        "cnn_input_representation": "per-link real/imaginary channels over ordered subcarriers",
        "cvnn_input_representation": (
            "complex convolution, phase-equivariant encoding, magnitude-invariant classifier"
        ),
        "split_mode": split_provenance["mode"],
        "threshold_selection_policy": _VALIDATION_THRESHOLD_POLICY,
        "prior_shift": resolved_prior_shift,
        "low_confidence_margin": low_confidence_margin,
        "dataset_materialization": (
            "content_addressed_build_node_v1"
            if dataset_build_node is not None
            else "direct_ordered_concatenation"
        ),
    }
    dependency_versions = versions()
    manifest = run_manifest(
        output.name,
        config,
        legacy_schema="loslab.run.v1",
        config_sha256=canonical_hash(config),
        inputs=[{"path": str(p), "sha256": sha256_file(p)} for p in paths],
        dependencies=dependency_versions,
        source_revision=dependency_versions.get("git_sha"),
        package_source_sha256=package_source_sha256(),
        n_samples=len(data.y),
        source_counts={str(s): int(np.sum(data.source == s)) for s in np.unique(data.source)},
        domain_counts={str(s): int(np.sum(data.domain == s)) for s in np.unique(data.domain)},
        label_semantics=resolved_label_semantics,
        label_semantics_by_source=_label_semantics_by_source(data),
        observation_semantics={
            "ideal_cfr": "ideal channel, diagnostic upper bound",
            "comb_ls": "generic noisy comb pilot estimate, NOT 3GPP DMRS",
            "precomputed": _precomputed_observation_semantics(data),
        }[observation],
        study_status=(
            "DEBUG_ONLY" if "synthetic_debug" in data.source else "SIMULATION_ONLY_UNVALIDATED"
        ),
        split_policy=split_provenance,
        dataset_build_node=dataset_build_node,
    )
    atomic_json(layout.config / "resolved.json", resolve_config(config))
    atomic_json(output / "manifest.json", manifest)
    results: list[dict[str, Any]] = []
    for seed in seeds:
        if locked_indices is not None:
            train, val, test = locked_indices[:3]
            calibration = locked_indices[3]
        else:
            train, val, test = (
                holdout_domain(data, holdout, seed=seed)
                if holdout
                else in_domain_split(data, seed=seed)
            )
            calibration = np.empty(0, dtype=np.int64)
        if resolved_prior_shift is not None:
            if len(calibration) == 0:
                train_pool = train
                fit_rows, calibration_rows = group_split(
                    data.y[train_pool],
                    data.group[train_pool],
                    test_fraction=0.2,
                    seed=seed + 907,
                )
                train = train_pool[fit_rows]
                calibration = train_pool[calibration_rows]
        else:
            calibration = np.empty(0, dtype=np.int64)
        assert not (set(data.group[train]) & set(data.group[val]))
        assert not (set(data.group[test]) & (set(data.group[train]) | set(data.group[val])))
        assert not (set(data.group[calibration]) & (set(data.group[train]) | set(data.group[val])))
        assert not (set(data.group[calibration]) & set(data.group[test]))
        for method in methods:
            method_input = (
                _identity_complex_cfr(observable)
                if method in ("cnn", "cvnn")
                else raw_real_imag
                if method in ("raw_mlp", "label_permuted", "row_shuffled")
                else power_only
                if method == "power_only"
                else phase_randomized
                if method == "phase_randomized"
                else _metadata_only_features(
                    group_values=data.group,
                    domain_values=data.domain,
                    source_values=data.source,
                    train_indices=train,
                )[0]
                if method == "metadata_only"
                else x
            )
            if method == "label_permuted":
                training_targets, _ = _permute_training_labels(
                    data.y[train], run_seed=int(seed), base_seed=int(label_permutation_seed)
                )
            else:
                training_targets = data.y[train]
            if method == "row_shuffled":
                shuffled_features, _ = _permute_training_feature_rows(
                    method_input[train], run_seed=int(seed), base_seed=int(row_shuffle_seed)
                )
                method_input = np.array(method_input, copy=True)
                method_input[train] = shuffled_features
            if method == "source_only":
                method_input, _ = _source_only_features(data.source, train_indices=train)
            uses_torch = method in ("cnn", "cvnn", "coral", "coral0")
            _sync_torch_device(device, uses_torch)
            if uses_torch and device.startswith("cuda"):
                import torch

                torch.cuda.reset_peak_memory_stats(device)
            fit_started = perf_counter()
            if method in (
                "rf",
                "histgb",
                "xgb",
                "mlp",
                "raw_mlp",
                "label_permuted",
                "row_shuffled",
                "source_only",
                "metadata_only",
                "power_only",
                "phase_randomized",
                "logreg",
            ):
                clf = fit_sklearn(
                    method_input[train],
                    training_targets,
                    seed=seed,
                    method="mlp"
                    if method == "raw_mlp"
                    else "mlp"
                    if method == "label_permuted"
                    else "mlp"
                    if method == "row_shuffled"
                    else "logreg"
                    if method in ("power_only", "source_only", "metadata_only")
                    else "mlp"
                    if method == "phase_randomized"
                    else method,
                    capacity=capacity,
                )
                extra: dict[str, Any] = {}
            elif method == "cnn":
                clf, extra = fit_real_cnn(
                    observable[train],
                    training_targets,
                    observable[val],
                    data.y[val],
                    seed=seed,
                    device=device,
                    capacity=capacity,
                )
            elif method == "cvnn":
                clf, extra = fit_complex_cnn(
                    observable[train],
                    data.y[train],
                    observable[val],
                    data.y[val],
                    seed=seed,
                    device=device,
                    capacity=capacity,
                )
            elif method in ("coral", "coral0"):
                clf, extra = fit_coral_mlp(
                    x[train],
                    data.y[train],
                    data.domain[train],
                    x[val],
                    data.y[val],
                    seed=seed,
                    penalty=coral_penalty if method == "coral" else 0.0,
                    device=device,
                    capacity=capacity,
                )
            else:
                raise ValueError(f"Unknown method {method}")
            _sync_torch_device(device, uses_torch)
            training_seconds = perf_counter() - fit_started
            parameter_count = _learned_parameter_count(clf, extra)
            calibration_scores_raw = np.empty(0, dtype=np.float64)
            calibration_scores_calibrated = np.empty(0, dtype=np.float64)
            calibrator_report = None
            validation_started = perf_counter()
            val_scores_raw = clf.predict_proba(method_input[val])[:, 1]
            _sync_torch_device(device, uses_torch)
            validation_prediction_seconds = perf_counter() - validation_started
            prior_shift_report = None
            if resolved_prior_shift is None:
                val_scores = val_scores_raw
                val_model_calibrated = val_scores_raw
            else:
                calibration_scores_raw = clf.predict_proba(method_input[calibration])[:, 1]
                calibration_scores_calibrated, calibrator_report = (
                    fit_group_balanced_platt_calibrator(
                        calibration_scores_raw,
                        data.y[calibration],
                        data.group[calibration],
                    )
                )
                _validate_evaluation_payload(
                    calibrator_report, "group-platt-calibrator.v1.schema.json"
                )
                val_calibrated = apply_platt_calibrator(val_scores_raw, calibrator_report)
                val_model_calibrated = val_calibrated
                val_scores, prior_shift_report = adjust_probabilities_for_prior_shift(
                    val_calibrated,
                    data.y[calibration],
                    data.group[calibration],
                    resolved_prior_shift,
                )
                _validate_evaluation_payload(
                    prior_shift_report, "prior-shift-adjustment.v1.schema.json"
                )
            threshold_selection = _validation_threshold_selection_record(
                data.y[val],
                val_scores,
                data.group[val],
                val,
            )
            threshold = threshold_selection["threshold"]
            if prior_shift_report is None:
                model_output_threshold = threshold
            else:
                model_output_threshold = float(
                    invert_platt_calibrator(
                        adjust_probabilities_between_priors(
                            np.asarray([threshold]),
                            source_positive_prevalence=(
                                resolved_prior_shift["target_positive_prevalence"]
                            ),
                            target_positive_prevalence=(
                                prior_shift_report["calibration_group_weighted_positive_prevalence"]
                            ),
                        ),
                        calibrator_report,
                    )[0]
                )
            checkpoint = (
                layout.models / f"{method}_seed{seed}.pt"
                if method.startswith("coral") or method in ("cnn", "cvnn")
                else layout.models / f"{method}_seed{seed}.joblib"
            )
            if method.startswith("coral"):
                clf.save(checkpoint, threshold=model_output_threshold, include_phase=include_phase)
            elif method == "cnn":
                clf.save(checkpoint, threshold=model_output_threshold, include_phase=True)
            elif method == "cvnn":
                clf.save(checkpoint, threshold=model_output_threshold)
            else:
                import joblib

                joblib.dump(clf, checkpoint)
            split_contract = {
                "mode": split_provenance["mode"],
                "registry_sha256": split_provenance.get("registry_sha256"),
                "train_rows": int(len(train)),
                "validation_rows": int(len(val)),
                "test_rows": int(len(test)),
            }
            partition_indices = {"train": train, "validation": val, "test": test}
            indices_payload = {
                "train_idx": train.tolist(),
                "validation_idx": val.tolist(),
                "test_idx": test.tolist(),
            }
            if len(calibration):
                partition_indices["calibration"] = calibration
                split_contract["calibration_rows"] = int(len(calibration))
                indices_payload["calibration_idx"] = calibration.tolist()
            split_contract["indices_sha256"] = canonical_hash(indices_payload)
            model_input_contract = build_model_input_artifact_contract(
                method=method,
                method_input=method_input,
                include_phase=include_phase,
                config_sha256=canonical_hash(config),
                source_revision=dependency_versions.get("git_sha"),
                input_file_digests=manifest["inputs"],
                split=split_contract,
                model_file_sha256=_stable_single_link_sha256(checkpoint),
                partition_indices=partition_indices,
                target_values=data.y if method == "label_permuted" else None,
                label_permutation_seed=(
                    int(label_permutation_seed) if method == "label_permuted" else None
                ),
                label_permutation_run_seed=(int(seed) if method == "label_permuted" else None),
                source_method_input=(raw_real_imag if method == "row_shuffled" else None),
                feature_permutation_seed=(
                    int(row_shuffle_seed) if method == "row_shuffled" else None
                ),
                feature_permutation_run_seed=(int(seed) if method == "row_shuffled" else None),
                source_values=(data.source if method == "source_only" else None),
                metadata_values=(
                    {"group": data.group, "domain": data.domain, "source": data.source}
                    if method == "metadata_only"
                    else None
                ),
            )
            model_info_path = layout.models / f"{method}_seed{seed}_modelinfo.json"
            atomic_json(
                model_info_path,
                {
                    "format": (
                        "loslab.sklearn.v1"
                        if method
                        in {
                            "rf",
                            "histgb",
                            "xgb",
                            "mlp",
                            "raw_mlp",
                            "label_permuted",
                            "row_shuffled",
                            "source_only",
                            "metadata_only",
                            "power_only",
                            "phase_randomized",
                            "logreg",
                        }
                        else "loslab.model_input_artifact.v1"
                    ),
                    "threshold": model_output_threshold,
                    "threshold_probability_space": "raw_model_output",
                    "calibrated_threshold": threshold,
                    "probability_calibration": calibrator_report,
                    "prior_shift": prior_shift_report,
                    "include_phase": include_phase,
                    "method": method,
                    "model_input_artifact_contract": model_input_contract,
                },
            )
            model_input_contract_verification = verify_model_input_artifact_contract(
                model_info_path,
                checkpoint,
                method_input=method_input,
                partition_indices=partition_indices,
                target_values=data.y if method == "label_permuted" else None,
                source_method_input=raw_real_imag if method == "row_shuffled" else None,
                source_values=data.source if method == "source_only" else None,
                metadata_values=(
                    {"group": data.group, "domain": data.domain, "source": data.source}
                    if method == "metadata_only"
                    else None
                ),
            )
            test_prediction_started = perf_counter()
            test_scores_raw = clf.predict_proba(method_input[test])[:, 1]
            _sync_torch_device(device, uses_torch)
            test_prediction_seconds = perf_counter() - test_prediction_started
            test_calibrated = test_scores_raw
            if resolved_prior_shift is None:
                prob = test_scores_raw
            else:
                test_calibrated = apply_platt_calibrator(test_scores_raw, calibrator_report)
                prob, test_prior_shift_report = adjust_probabilities_for_prior_shift(
                    test_calibrated,
                    data.y[calibration],
                    data.group[calibration],
                    resolved_prior_shift,
                )
                if test_prior_shift_report != prior_shift_report:
                    raise RuntimeError("train-only prior-shift report changed across partitions")
            metrics = binary_metrics(data.y[test], prob, threshold=threshold)
            source_families = np.asarray(data.source, dtype=str)
            test_families = source_families[test]
            low_confidence_mask = np.zeros(len(test), dtype=bool)
            low_confidence_report = None
            if low_confidence_margin is not None:
                low_confidence_mask, low_confidence_report = build_low_confidence_audit(
                    prob,
                    data.group[test],
                    test_families,
                    data.domain[test],
                    threshold=threshold,
                    margin=float(low_confidence_margin),
                )
                _validate_evaluation_payload(
                    low_confidence_report, "low-confidence-audit.v1.schema.json"
                )
            expected_test_strata = tuple(
                ("test", family_id) for family_id in sorted(set(source_families.tolist()))
            )
            class_support = audit_class_support(
                labels=data.y[test],
                independent_group_ids=data.group[test],
                split_ids=np.full(len(test), "test", dtype=object),
                family_ids=test_families,
                expected_strata=expected_test_strata,
                classes=(0, 1),
            )
            class_support_payload = {
                "schema": "loslab.class_support.v1",
                "independence_identity_source": "ChannelDataset.group",
                "cells": [
                    {
                        "split_id": cell.split_id,
                        "family_id": cell.family_id,
                        "class_label": cell.class_label,
                        "record_count": cell.record_count,
                        "independent_group_count": cell.independent_group_count,
                        "status": cell.status,
                    }
                    for cell in class_support.cells
                ],
            }
            import jsonschema

            class_support_schema_path = (
                Path(__file__).resolve().parents[1]
                / "evaluation/schema/class_support.v1.schema.json"
            )
            class_support_schema = json.loads(class_support_schema_path.read_text(encoding="utf-8"))
            jsonschema.Draft202012Validator.check_schema(class_support_schema)
            class_support_errors = list(
                jsonschema.Draft202012Validator(class_support_schema).iter_errors(
                    class_support_payload
                )
            )
            if class_support_errors:
                raise ValueError(
                    "class support report violates schema: "
                    + "; ".join(error.message for error in class_support_errors[:10])
                )
            ci = block_bootstrap_auc(
                data.y[test], prob, data.group[test], seed=seed, draws=bootstrap
            )
            result = {
                "seed": int(seed),
                "method": method,
                "input_representation": (
                    "complex observation split into real/imaginary channels"
                    if method == "cnn"
                    else "complex convolution with global-phase-invariant magnitude head"
                    if method == "cvnn"
                    else "flattened raw real/imaginary channel observation"
                    if method == "raw_mlp"
                    else "per-link/per-subcarrier channel power in dB"
                    if method == "power_only"
                    else "phase-randomized flattened real/imaginary CFR"
                    if method == "phase_randomized"
                    else "raw flattened real/imaginary CFR trained on deterministically permuted training labels"
                    if method == "label_permuted"
                    else "row-shuffled flattened real/imaginary CFR"
                    if method == "row_shuffled"
                    else "training-vocabulary one-hot source identifiers"
                    if method == "source_only"
                    else "training-vocabulary one-hot group/domain/source metadata leakage control"
                    if method == "metadata_only"
                    else "engineered observable features"
                ),
                "metrics": metrics,
                "prior_shift": prior_shift_report,
                "probability_calibration": calibrator_report,
                "low_confidence_audit": low_confidence_report,
                "model_output_threshold": model_output_threshold,
                "class_support": class_support_payload,
                "confidence": ci,
                "threshold_selection": threshold_selection,
                "split": {
                    "mode": split_provenance["mode"],
                    "registry_sha256": split_provenance.get("registry_sha256"),
                    "indices_sha256": canonical_hash(
                        {
                            "train_idx": train.tolist(),
                            "validation_idx": val.tolist(),
                            "test_idx": test.tolist(),
                            **(
                                {"calibration_idx": calibration.tolist()}
                                if len(calibration)
                                else {}
                            ),
                        }
                    ),
                    "train_size": len(train),
                    "calibration_size": len(calibration),
                    "val_size": len(val),
                    "test_size": len(test),
                    "train_groups": len(set(data.group[train])),
                    "calibration_groups": len(set(data.group[calibration])),
                    "validation_groups": len(set(data.group[val])),
                    "test_groups": len(set(data.group[test])),
                    "train_domains": sorted(set(data.domain[train].tolist())),
                    "test_domains": sorted(set(data.domain[test].tolist())),
                },
                "training": extra,
                "compute": {
                    "training_seconds": float(training_seconds),
                    "validation_prediction_seconds": float(validation_prediction_seconds),
                    "test_prediction_seconds": float(test_prediction_seconds),
                    "validation_prediction_samples": int(len(val)),
                    "test_prediction_samples": int(len(test)),
                    "learned_parameter_count": parameter_count,
                    "peak_cuda_memory_bytes": (
                        int(torch.cuda.max_memory_allocated(device))
                        if uses_torch and device.startswith("cuda")
                        else None
                    ),
                },
                "checkpoint_path": str(checkpoint),
                "model_sha256": model_input_contract["model_sha256"],
                "model_input_artifact_contract_sha256": canonical_hash(model_input_contract),
                "model_input_artifact_contract_status": model_input_contract_verification["status"],
                "model_input_feature_matrix_verified": model_input_contract_verification[
                    "feature_matrix_verified"
                ],
            }
            results.append(result)
            # Keep split indices and probabilities for independently reproducing every metric.
            np.savez_compressed(
                layout.predictions / f"{method}_seed{seed}.npz",
                train_idx=train,
                calibration_idx=calibration,
                val_idx=val,
                test_idx=test,
                y_test=data.y[test],
                y_val=data.y[val],
                test_group=np.asarray(data.group[test], dtype=str),
                test_domain=np.asarray(data.domain[test], dtype=str),
                test_source=test_families,
                p_test=prob,
                p_val=val_scores,
                p_test_raw=test_scores_raw,
                p_val_raw=val_scores_raw,
                p_test_calibrated=test_calibrated,
                p_val_calibrated=val_model_calibrated,
                p_calibration_raw=calibration_scores_raw,
                p_calibration_calibrated=calibration_scores_calibrated,
                test_low_confidence=low_confidence_mask,
                threshold=np.float64(threshold),
            )
        atomic_json(output / "results.json", results)
        atomic_json(layout.evaluation / "results.json", results)
    aggregated = {}
    for method in methods:
        all_scores = [row["metrics"]["auroc"] for row in results if row["method"] == method]
        seq = [score for score in all_scores if score is not None]
        aggregated[method] = {
            "seeds": len(all_scores),
            "auroc_defined_seeds": len(seq),
            "mean_test_auroc": float(np.mean(seq)) if seq else None,
            "std_test_auroc": float(np.std(seq, ddof=1)) if len(seq) > 1 else None,
        }
    comparisons = {}
    method_pairs = (
        ("rf", "mlp"),
        ("histgb", "mlp"),
        ("raw_mlp", "mlp"),
        ("raw_mlp", "rf"),
        ("raw_mlp", "histgb"),
        ("power_only", "mlp"),
        ("power_only", "rf"),
        ("power_only", "raw_mlp"),
        ("raw_mlp", "phase_randomized"),
        ("raw_mlp", "label_permuted"),
        ("raw_mlp", "row_shuffled"),
        ("raw_mlp", "source_only"),
        ("raw_mlp", "metadata_only"),
        ("cnn", "raw_mlp"),
        ("cvnn", "raw_mlp"),
        ("xgb", "mlp"),
        ("xgb", "rf"),
        ("cnn", "mlp"),
        ("cvnn", "mlp"),
        ("cvnn", "cnn"),
        ("rf", "cvnn"),
        ("histgb", "cvnn"),
        ("rf", "cnn"),
        ("histgb", "cnn"),
        ("logreg", "mlp"),
        ("mlp", "coral"),
        ("coral0", "coral"),
        ("mlp", "coral0"),
    )
    for control, treatment in method_pairs:
        if control in methods and treatment in methods:
            comparisons[f"{treatment}_versus_{control}"] = paired_seed_delta(
                results, control, treatment, draws=5000
            )
    summary = {
        "schema": "loslab.summary.v1",
        "evidence_grade": manifest["study_status"],
        "paired_method_comparisons": comparisons,
        "aggregate_across_fixed_splits": aggregated,
        "warning": "Do not report synthetic-debug results or profile-proxy labels as real LOS measurements.",
    }
    atomic_json(output / "summary.json", summary)
    atomic_json(layout.evaluation / "summary.json", summary)
    complete_manifest = mark_complete(manifest)
    atomic_json(layout.manifest, complete_manifest)
    atomic_json(
        layout.complete,
        {
            "run_id": output.name,
            "manifest_schema": complete_manifest["schema"],
            "config_identity": complete_manifest["config_identity"],
        },
    )
    return summary
