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
from evaluation.metrics import (
    binary_metrics,
    block_bootstrap_auc,
    holdout_domain,
    in_domain_split,
    threshold_from_validation,
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
    "observable_features": ("observed_cfr", "logmag_pdp_moments_v1"),
}
_MODEL_METHODS = frozenset(
    {"rf", "histgb", "xgb", "mlp", "raw_mlp", "logreg", "cnn", "cvnn", "coral", "coral0"}
)


def _identity_complex_cfr(observed_cfr: np.ndarray) -> np.ndarray:
    """Keep the audited complex CFR grid as the model's input matrix."""
    return observed_cfr


def _flatten_real_imag_cfr(observed_cfr: np.ndarray) -> np.ndarray:
    """Flatten each CFR sample in C order, real values before imaginary values."""
    return np.concatenate((observed_cfr.real, observed_cfr.imag), axis=1).reshape(
        len(observed_cfr), -1
    )


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
    elif method == "raw_mlp":
        transform = _flatten_real_imag_cfr
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
) -> dict[str, Any]:
    """Bind a saved model to its audited observation-only input route."""
    if method not in _MODEL_METHODS:
        raise ValueError(f"unsupported model-input artifact method: {method!r}")
    if not isinstance(method_input, np.ndarray) or method_input.ndim < 2:
        raise ValueError("model input must be a batched feature array")
    if not np.issubdtype(method_input.dtype, np.number) or not np.isfinite(method_input).all():
        raise ValueError("model input must contain finite numeric values")
    if set(partition_indices) != {"train", "validation", "test"}:
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
    feature_contract = {
        "schema": "loslab.model_input_feature_contract.v2",
        "method": method,
        "source_dataclass": "datasets.channel.ChannelModelInputs",
        "allowed_source_fields": [source_feature],
        "excluded_truth_and_metadata_fields": [
            "data.y",
            "data.metadata",
            "data.group",
            "data.domain",
            "PathOracleTable",
            "target_values",
        ],
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
    feature_contract_sha256 = canonical_hash(feature_contract)
    return {
        "schema": "loslab.model_input_artifact_contract.v2",
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
    if contract_schema == "loslab.model_input_artifact_contract.v2":
        expected_feature_fields.update(
            {
                "feature_transform_implementation_sha256",
                "feature_matrix",
                "partition_feature_matrices",
            }
        )
    if not isinstance(feature, dict) or set(feature) != expected_feature_fields:
        raise ValueError("model feature contract fields are invalid")
    method = feature.get("method")
    expected_format = (
        "loslab.sklearn.v1"
        if method in {"rf", "histgb", "xgb", "mlp", "raw_mlp", "logreg"}
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
    expected_feature_schema = (
        "loslab.model_input_feature_contract.v2"
        if contract_schema == "loslab.model_input_artifact_contract.v2"
        else "loslab.model_input_feature_contract.v1"
    )
    if (
        feature.get("schema") != expected_feature_schema
        or method not in _MODEL_METHODS
        or feature.get("source_dataclass") != "datasets.channel.ChannelModelInputs"
        or feature.get("allowed_source_fields") != [expected_route[0]]
        or feature.get("excluded_truth_and_metadata_fields")
        != [
            "data.y",
            "data.metadata",
            "data.group",
            "data.domain",
            "PathOracleTable",
            "target_values",
        ]
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
    if contract_schema == "loslab.model_input_artifact_contract.v2":
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
            if partition_indices is None or set(partition_indices) != {
                "train",
                "validation",
                "test",
            }:
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
    if not isinstance(split, dict) or set(split) != {
        "mode",
        "registry_sha256",
        "indices_sha256",
        "train_rows",
        "validation_rows",
        "test_rows",
    }:
        raise ValueError("model input contract split binding is invalid")
    for count_key in ("train_rows", "validation_rows", "test_rows"):
        count = split[count_key]
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("model input contract split counts must be positive integers")
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
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
    return train, validation, test, report


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
    device: str = "cpu",
    coral_penalty: float = 0.1,
    capacity: str = "medium",
    split_lock_path: str | Path | None = None,
    split_signature_path: str | Path | None = None,
    trusted_split_signers_path: str | Path | None = None,
    split_signer_id: str | None = None,
) -> dict:
    if not paths:
        raise ValueError("No datasets supplied")
    capacity_spec(capacity)
    data = ChannelDataset.concatenate([ChannelDataset.load(p) for p in paths])
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
            **locked_indices[3],
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
        "feature_version": "logmag+pdp+moments-v1",
        "raw_mlp_input_representation": "flattened real/imaginary CFR channels",
        "cnn_input_representation": "per-link real/imaginary channels over ordered subcarriers",
        "cvnn_input_representation": (
            "complex convolution, phase-equivariant encoding, magnitude-invariant classifier"
        ),
        "split_mode": split_provenance["mode"],
        "threshold_selection_policy": _VALIDATION_THRESHOLD_POLICY,
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
    )
    atomic_json(layout.config / "resolved.json", resolve_config(config))
    atomic_json(output / "manifest.json", manifest)
    results: list[dict[str, Any]] = []
    for seed in seeds:
        train, val, test = (
            locked_indices[:3]
            if locked_indices is not None
            else holdout_domain(data, holdout, seed=seed)
            if holdout
            else in_domain_split(data, seed=seed)
        )
        assert not (set(data.group[train]) & set(data.group[val]))
        assert not (set(data.group[test]) & (set(data.group[train]) | set(data.group[val])))
        for method in methods:
            method_input = (
                _identity_complex_cfr(observable)
                if method in ("cnn", "cvnn")
                else raw_real_imag
                if method == "raw_mlp"
                else x
            )
            uses_torch = method in ("cnn", "cvnn", "coral", "coral0")
            _sync_torch_device(device, uses_torch)
            if uses_torch and device.startswith("cuda"):
                import torch

                torch.cuda.reset_peak_memory_stats(device)
            fit_started = perf_counter()
            if method in ("rf", "histgb", "xgb", "mlp", "raw_mlp", "logreg"):
                clf = fit_sklearn(
                    method_input[train],
                    data.y[train],
                    seed=seed,
                    method="mlp" if method == "raw_mlp" else method,
                    capacity=capacity,
                )
                extra: dict[str, Any] = {}
            elif method == "cnn":
                clf, extra = fit_real_cnn(
                    observable[train],
                    data.y[train],
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
            validation_started = perf_counter()
            val_scores = clf.predict_proba(method_input[val])[:, 1]
            _sync_torch_device(device, uses_torch)
            validation_prediction_seconds = perf_counter() - validation_started
            threshold_selection = _validation_threshold_selection_record(
                data.y[val],
                val_scores,
                data.group[val],
                val,
            )
            threshold = threshold_selection["threshold"]
            checkpoint = (
                layout.models / f"{method}_seed{seed}.pt"
                if method.startswith("coral") or method in ("cnn", "cvnn")
                else layout.models / f"{method}_seed{seed}.joblib"
            )
            if method.startswith("coral"):
                clf.save(checkpoint, threshold=threshold, include_phase=include_phase)
            elif method == "cnn":
                clf.save(checkpoint, threshold=threshold, include_phase=True)
            elif method == "cvnn":
                clf.save(checkpoint, threshold=threshold)
            else:
                import joblib

                joblib.dump(clf, checkpoint)
            split_contract = {
                "mode": split_provenance["mode"],
                "registry_sha256": split_provenance.get("registry_sha256"),
                "indices_sha256": canonical_hash(
                    {
                        "train_idx": train.tolist(),
                        "validation_idx": val.tolist(),
                        "test_idx": test.tolist(),
                    }
                ),
                "train_rows": int(len(train)),
                "validation_rows": int(len(val)),
                "test_rows": int(len(test)),
            }
            model_input_contract = build_model_input_artifact_contract(
                method=method,
                method_input=method_input,
                include_phase=include_phase,
                config_sha256=canonical_hash(config),
                source_revision=dependency_versions.get("git_sha"),
                input_file_digests=manifest["inputs"],
                split=split_contract,
                model_file_sha256=_stable_single_link_sha256(checkpoint),
                partition_indices={"train": train, "validation": val, "test": test},
            )
            model_info_path = layout.models / f"{method}_seed{seed}_modelinfo.json"
            atomic_json(
                model_info_path,
                {
                    "format": (
                        "loslab.sklearn.v1"
                        if method in {"rf", "histgb", "xgb", "mlp", "raw_mlp", "logreg"}
                        else "loslab.model_input_artifact.v1"
                    ),
                    "threshold": threshold,
                    "include_phase": include_phase,
                    "method": method,
                    "model_input_artifact_contract": model_input_contract,
                },
            )
            model_input_contract_verification = verify_model_input_artifact_contract(
                model_info_path,
                checkpoint,
                method_input=method_input,
                partition_indices={"train": train, "validation": val, "test": test},
            )
            test_prediction_started = perf_counter()
            prob = clf.predict_proba(method_input[test])[:, 1]
            _sync_torch_device(device, uses_torch)
            test_prediction_seconds = perf_counter() - test_prediction_started
            metrics = binary_metrics(data.y[test], prob, threshold=threshold)
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
                    else "engineered observable features"
                ),
                "metrics": metrics,
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
                        }
                    ),
                    "train_size": len(train),
                    "val_size": len(val),
                    "test_size": len(test),
                    "train_groups": len(set(data.group[train])),
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
                val_idx=val,
                test_idx=test,
                y_test=data.y[test],
                y_val=data.y[val],
                test_group=np.asarray(data.group[test], dtype=str),
                test_domain=np.asarray(data.domain[test], dtype=str),
                p_test=prob,
                p_val=val_scores,
                threshold=np.float64(threshold),
            )
        atomic_json(output / "results.json", results)
        atomic_json(layout.evaluation / "results.json", results)
    aggregated = {}
    for method in methods:
        seq = [row["metrics"]["auroc"] for row in results if row["method"] == method]
        aggregated[method] = {
            "seeds": len(seq),
            "mean_test_auroc": float(np.mean(seq)),
            "std_test_auroc": float(np.std(seq, ddof=1)) if len(seq) > 1 else None,
        }
    comparisons = {}
    method_pairs = (
        ("rf", "mlp"),
        ("histgb", "mlp"),
        ("raw_mlp", "mlp"),
        ("raw_mlp", "rf"),
        ("raw_mlp", "histgb"),
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
