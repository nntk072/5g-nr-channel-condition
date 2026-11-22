"""Verification and build of model-input and validation-threshold contracts."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import stat
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from artifacts.provenance import (
    canonical_hash,
)
from datasets.channel import ChannelDataset
from models.multipath import fit_sparse_multipath
from propagation.physics import delay_k_proxy_features, observable_features

from .features import (
    _MODEL_FEATURE_ROUTES,
    _MODEL_METHODS,
    _array_sha256,
    _flatten_real_imag_cfr,
    _identity_complex_cfr,
    _log_power_per_link_subcarrier,
    _sparse_multipath_features,
    _stable_single_link_sha256,
)
from .validation import (
    _declared_partition_support,
    _metadata_only_features,
    _oracle_label_leak_features,
    _permute_training_feature_rows,
    _permute_training_labels,
    _phase_randomized_real_imag_cfr,
    _single_metadata_features,
    _source_only_features,
    _validation_threshold_selection_record,
)


def _feature_transform_implementation_sha256(method: str) -> str:
    transform: Callable[..., Any]
    if method in ("cnn", "cvnn"):
        transform = _identity_complex_cfr
    elif method == "source_only":
        transform = _source_only_features
    elif method == "metadata_only":
        transform = _metadata_only_features
    elif method in {"group_only", "domain_only"}:
        transform = _single_metadata_features
    elif method in ("raw_mlp", "label_permuted", "row_shuffled"):
        transform = _flatten_real_imag_cfr
    elif method == "power_only":
        transform = _log_power_per_link_subcarrier
    elif method == "phase_randomized":
        transform = _phase_randomized_real_imag_cfr
    elif method == "delay_k_proxy":
        transform = delay_k_proxy_features
    elif method == "sparse_multipath":
        estimator_source = Path(inspect.getfile(fit_sparse_multipath)).read_bytes()
        payload = b"\0".join(
            (
                inspect.getsource(_sparse_multipath_features).encode("utf-8"),
                estimator_source,
                b"maximum_paths=3;maximum_delay_normalized=0.5;delay_grid_size=129",
            )
        )
        return hashlib.sha256(payload).hexdigest()
    elif method == "oracle_label_leak":
        transform = _oracle_label_leak_features
    else:
        transform = observable_features
    return hashlib.sha256(inspect.getsource(transform).encode("utf-8")).hexdigest()


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
        "v13"
        if method in {"delay_k_proxy", "sparse_multipath"}
        else "v10"
        if method == "observable_features"
        else "v12"
        if method == "domain_only"
        else "v11"
        if method == "group_only"
        else "v9"
        if method == "oracle_label_leak"
        else "v8"
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
            if method
            in {"source_only", "metadata_only", "group_only", "domain_only", "oracle_label_leak"}
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
                "data.domain" if method == "group_only" else "data.group",
                "data.source",
                "observed_cfr",
                "PathOracleTable",
                "target_values",
            ]
            if method in {"group_only", "domain_only"}
            else [
                "data.metadata",
                "data.group",
                "data.domain",
                "data.source",
                "observed_cfr",
                "PathOracleTable",
                "target_values",
            ]
            if method == "oracle_label_leak"
            else [
                "data.y",
                "data.metadata",
                "data.group",
                "data.domain",
                "PathOracleTable",
                "target_values",
            ]
        ),
        "target_field_used_for_supervision_only": (
            "data.y (also intentionally leaked as a feature)"
            if method == "oracle_label_leak"
            else "data.y"
        ),
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
    elif method == "oracle_label_leak":
        if (
            target_values is None
            or not isinstance(target_values, np.ndarray)
            or target_values.ndim != 1
            or len(target_values) != len(method_input)
            or label_permutation_seed is not None
            or label_permutation_run_seed is not None
        ):
            raise ValueError("oracle-label leakage contract requires the exact full target roster")
        expected_features, leakage_contract = _oracle_label_leak_features(target_values)
        if not np.array_equal(expected_features, method_input):
            raise ValueError(
                "oracle-label input differs from the declared target leakage transform"
            )
        feature_contract["intentional_label_leakage"] = leakage_contract
    elif (
        target_values is not None
        or label_permutation_seed is not None
        or label_permutation_run_seed is not None
    ):
        raise ValueError(
            "target values or permutation seeds are invalid for this model-input method"
        )
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
    elif metadata_values is not None and method not in {"group_only", "domain_only"}:
        raise ValueError("metadata identifiers are only valid for metadata leakage methods")
    if method in {"group_only", "domain_only"}:
        field = "group" if method == "group_only" else "domain"
        if not isinstance(metadata_values, dict) or set(metadata_values) != {field}:
            raise ValueError(f"{method} contracts require only the {field} identifier roster")
        expected_features, metadata_transform = _single_metadata_features(
            metadata_values[field], field=field, train_indices=checked_indices["train"]
        )
        if not np.array_equal(expected_features, method_input):
            raise ValueError(f"{method} features differ from the training vocabulary transform")
        feature_contract["metadata_vocabulary_transform"] = metadata_transform
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
    benchmark_cohort: dict[str, Any] | None = None,
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
        "loslab.model_input_artifact_contract.v9": "v9",
        "loslab.model_input_artifact_contract.v10": "v10",
        "loslab.model_input_artifact_contract.v11": "v11",
        "loslab.model_input_artifact_contract.v12": "v12",
        "loslab.model_input_artifact_contract.v13": "v13",
    }.get(contract_schema)
    if schema_version is None:
        raise ValueError("model input artifact contract schema is unsupported")
    schema_path = Path(__file__).resolve().parents[2] / (
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
        "loslab.model_input_artifact_contract.v9",
        "loslab.model_input_artifact_contract.v10",
        "loslab.model_input_artifact_contract.v11",
        "loslab.model_input_artifact_contract.v12",
        "loslab.model_input_artifact_contract.v13",
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
    if contract_schema in {
        "loslab.model_input_artifact_contract.v8",
        "loslab.model_input_artifact_contract.v11",
        "loslab.model_input_artifact_contract.v12",
    }:
        expected_feature_fields.add("metadata_vocabulary_transform")
    if contract_schema == "loslab.model_input_artifact_contract.v9":
        expected_feature_fields.add("intentional_label_leakage")
    if not isinstance(feature, dict) or set(feature) != expected_feature_fields:
        raise ValueError("model feature contract fields are invalid")
    method = feature.get("method")
    if not isinstance(method, str):
        raise ValueError("model feature contract method must be a string")
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
            "group_only",
            "domain_only",
            "oracle_label_leak",
            "logreg",
            "delay_k_proxy",
            "sparse_multipath",
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
            if method
            in {"source_only", "metadata_only", "group_only", "domain_only", "oracle_label_leak"}
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
                "data.domain" if method == "group_only" else "data.group",
                "data.source",
                "observed_cfr",
                "PathOracleTable",
                "target_values",
            ]
            if method in {"group_only", "domain_only"}
            else [
                "data.metadata",
                "data.group",
                "data.domain",
                "data.source",
                "observed_cfr",
                "PathOracleTable",
                "target_values",
            ]
            if method == "oracle_label_leak"
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
        or feature.get("target_field_used_for_supervision_only")
        != (
            "data.y (also intentionally leaked as a feature)"
            if method == "oracle_label_leak"
            else "data.y"
        )
        or (
            method != "oracle_label_leak"
            and "data.y" not in feature.get("excluded_truth_and_metadata_fields", [])
        )
        or not isinstance(feature.get("run_config_sha256"), str)
        or len(feature["run_config_sha256"]) != 64
        or any(char not in "0123456789abcdef" for char in feature["run_config_sha256"])
    ):
        raise ValueError("model feature contract does not match an audited feature route")
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
        "loslab.model_input_artifact_contract.v9",
        "loslab.model_input_artifact_contract.v10",
        "loslab.model_input_artifact_contract.v11",
        "loslab.model_input_artifact_contract.v12",
        "loslab.model_input_artifact_contract.v13",
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
        permutation_run_seed = permutation.get("run_seed")
        permutation_base_seed = permutation.get("base_seed")
        if (
            isinstance(permutation_run_seed, bool)
            or not isinstance(permutation_run_seed, int)
            or isinstance(permutation_base_seed, bool)
            or not isinstance(permutation_base_seed, int)
        ):
            raise ValueError("label-permuted feature contract seeds must be integers")
        _, replayed_permutation = _permute_training_labels(
            training_targets,
            run_seed=permutation_run_seed,
            base_seed=permutation_base_seed,
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
        permutation_run_seed = permutation.get("run_seed")
        permutation_base_seed = permutation.get("base_seed")
        if (
            isinstance(permutation_run_seed, bool)
            or not isinstance(permutation_run_seed, int)
            or isinstance(permutation_base_seed, bool)
            or not isinstance(permutation_base_seed, int)
        ):
            raise ValueError("row-shuffled feature contract seeds must be integers")
        shuffled, replayed_permutation = _permute_training_feature_rows(
            source_method_input[partition_indices["train"]],
            run_seed=permutation_run_seed,
            base_seed=permutation_base_seed,
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
    if contract_schema in {
        "loslab.model_input_artifact_contract.v11",
        "loslab.model_input_artifact_contract.v12",
    }:
        expected_method = "group_only" if contract_schema.endswith("v11") else "domain_only"
        field_name = "group" if expected_method == "group_only" else "domain"
        if (
            method != expected_method
            or metadata_values is None
            or set(metadata_values) != {field_name}
        ):
            raise ValueError(f"{expected_method} verification requires only its ID roster")
        if method_input is None or partition_indices is None:
            raise ValueError(f"{expected_method} verification inputs are incomplete")
        features, replayed_transform = _single_metadata_features(
            metadata_values[field_name], field=field_name, train_indices=partition_indices["train"]
        )
        if replayed_transform != feature.get("metadata_vocabulary_transform") or not np.array_equal(
            features, method_input
        ):
            raise ValueError(
                f"{expected_method} feature vocabulary does not replay from training IDs"
            )
    if contract_schema == "loslab.model_input_artifact_contract.v9":
        if method != "oracle_label_leak" or target_values is None:
            raise ValueError("oracle-label contract verification requires the exact target roster")
        if (
            method_input is None
            or partition_indices is None
            or not isinstance(target_values, np.ndarray)
            or target_values.ndim != 1
            or len(target_values) != len(method_input)
        ):
            raise ValueError("oracle-label verification inputs are incomplete")
        replayed_features, replayed_leakage = _oracle_label_leak_features(target_values)
        if replayed_leakage != feature.get("intentional_label_leakage") or not np.array_equal(
            replayed_features, method_input
        ):
            raise ValueError("intentional label leakage does not replay from source targets")
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
    sidecar_benchmark_cohort = sidecar.get("benchmark_cohort")
    benchmark_verification = None
    if benchmark_cohort is None:
        if sidecar_benchmark_cohort is not None:
            raise ValueError("unexpected benchmark cohort provenance in model info sidecar")
    else:
        if (
            not isinstance(benchmark_cohort, dict)
            or sidecar_benchmark_cohort != benchmark_cohort
            or partition_indices is None
            or "test" not in partition_indices
        ):
            raise ValueError("model info benchmark cohort provenance does not match replay")
        test_indices = np.asarray(partition_indices["test"])
        if test_indices.ndim != 1 or test_indices.dtype.kind not in "iu":
            raise ValueError("benchmark test indices must be a one-dimensional integer array")
        selected_digest = canonical_hash(test_indices.astype(np.int64, copy=False).tolist())
        if selected_digest != benchmark_cohort.get("selected_test_indices_sha256"):
            raise ValueError("benchmark cohort digest does not bind the model test indices")
        if benchmark_cohort.get("registry_sha256") != registry_digest:
            raise ValueError("benchmark cohort registry does not match model split contract")
        benchmark_verification = {
            "selected_test_cohort": benchmark_cohort.get("selected_test_cohort"),
            "benchmark_manifest_sha256": benchmark_cohort.get("manifest_sha256"),
            "selected_test_indices_sha256": selected_digest,
        }
    verification = {
        "status": "PASS_MODEL_INPUT_ARTIFACT_BINDING",
        "model_sha256": contract["model_sha256"],
        "feature_contract_sha256": contract["feature_contract_sha256"],
        "feature_matrix_verified": feature_matrix_verified,
    }
    if benchmark_verification is not None:
        verification["benchmark_cohort"] = benchmark_verification
    return verification


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


def verify_declared_partition_support(
    data: ChannelDataset,
    partition_indices: Mapping[str, Sequence[int] | np.ndarray],
    report: Mapping[str, Any],
    expected_sha256: str,
) -> dict[str, Any]:
    """Replay a saved per-run split-support report from its source dataset."""
    expected_partitions = {"train", "validation", "calibration", "test"}
    if not isinstance(partition_indices, Mapping) or set(partition_indices) != expected_partitions:
        raise ValueError("partition indices must name train, validation, calibration, and test")
    normalized: dict[str, np.ndarray] = {}
    seen: set[int] = set()
    for partition in ("train", "validation", "calibration", "test"):
        indices = np.asarray(partition_indices[partition])
        if indices.ndim != 1 or indices.dtype.kind not in "iu":
            raise ValueError(f"{partition} indices must be a one-dimensional integer array")
        values = [int(index) for index in indices]
        if (
            len(values) != len(set(values))
            or any(index < 0 or index >= len(data.y) for index in values)
            or seen.intersection(values)
        ):
            raise ValueError("partition indices must be unique, in range, and disjoint")
        seen.update(values)
        normalized[partition] = indices.astype(np.int64, copy=False)
    if seen != set(range(len(data.y))):
        raise ValueError("partition indices must cover every source dataset row exactly once")
    actual_report = {
        partition: _declared_partition_support(data, normalized[partition])
        for partition in ("train", "validation", "calibration", "test")
    }
    actual_sha256 = canonical_hash(actual_report)
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(char not in "0123456789abcdef" for char in expected_sha256)
        or actual_sha256 != expected_sha256
        or dict(report) != actual_report
    ):
        raise ValueError("partition support report does not replay from source data and indices")
    return {
        "schema": "loslab.model_partition_support_verification.v1",
        "status": "PASS_INPUT_SCOPED_REPLAY",
        "qualification_status": "NOT_QUALIFIED",
        "physical_independence_status": "not_established_by_runner",
        "report_sha256": actual_sha256,
    }
