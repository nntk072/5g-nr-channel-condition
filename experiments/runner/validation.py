"""Label routing, permutation controls, and validation helpers for the experiment runner."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from artifacts.build_dag import BuildNodeSpec, resolve_build_artifact, run_build_node
from artifacts.provenance import (
    canonical_hash,
    package_source_sha256,
    sha256_file,
    versions,
)
from datasets.channel import ChannelDataset
from datasets.split_policy import (
    SplitPolicyRegistry,
    _reject_duplicate_json_keys,
    verify_openssh_signed_artifact,
)
from evaluation.benchmark_partitions import (
    BENCHMARK_COHORTS,
    TEST_COHORTS,
    verify_benchmark_manifest_replay,
)
from evaluation.metrics import (
    threshold_from_validation,
)

from .features import (
    _CHANNEL_DATASET_BUILD_SCHEMA,
    _VALIDATION_THRESHOLD_POLICY,
    _array_sha256,
    _flatten_real_imag_cfr,
    _kish_effective_group_count,
    _normalized_label_semantics,
    _stable_input_bytes,
    _unsupported_benchmark_test_cohorts,
    _validate_evaluation_payload,
    _validate_json_schema,
)


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


def _single_metadata_features(
    values: np.ndarray,
    *,
    field: str,
    train_indices: np.ndarray,
    max_feature_cells: int = 16_000_000,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Encode exactly one dataset identity field for leakage attribution."""
    if field not in {"group", "domain"}:
        raise ValueError("single metadata control field must be group or domain")
    values = np.asarray(values)
    if (
        values.ndim != 1
        or len(values) < 1
        or any(
            not isinstance(value, (str, np.str_))
            or not str(value)
            or str(value) != str(value).strip()
            or "\x00" in str(value)
            for value in values
        )
    ):
        raise ValueError(f"{field}-only control requires canonical nonempty string IDs")
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
        raise ValueError("single metadata training indices are invalid")
    if (
        isinstance(max_feature_cells, bool)
        or not isinstance(max_feature_cells, int)
        or max_feature_cells < 1
    ):
        raise ValueError("max_feature_cells must be a positive integer")
    vocabulary = tuple(sorted({str(values[index]) for index in train_indices}))
    if len(values) * len(vocabulary) > max_feature_cells:
        raise ValueError("single metadata dense feature matrix exceeds max_feature_cells")
    columns = {value: index for index, value in enumerate(vocabulary)}
    features = np.zeros((len(values), len(vocabulary)), dtype=np.float32)
    for row, value in enumerate(values):
        column = columns.get(str(value))
        if column is not None:
            features[row, column] = 1.0
    transform = {
        "schema": f"loslab.{field}_only_vocabulary.v1",
        "algorithm": f"sorted_training_{field}_one_hot_unknown_zero_v1",
        "implementation_sha256": hashlib.sha256(
            inspect.getsource(_single_metadata_features).encode("utf-8")
        ).hexdigest(),
        "field": field,
        "vocabulary": list(vocabulary),
        "vocabulary_sha256": canonical_hash(list(vocabulary)),
        "values_sha256": _array_sha256(np.asarray(values, dtype=str)),
        "training_indices_sha256": _array_sha256(train_indices),
        "max_feature_cells": max_feature_cells,
        "unknown_category_behavior": "all_zero_vector",
    }
    return features, transform


def _oracle_label_leak_features(
    target_values: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Expose the binary target as an input for an explicitly leaking control."""
    labels = np.asarray(target_values)
    if (
        labels.ndim != 1
        or not len(labels)
        or not np.issubdtype(labels.dtype, np.integer)
        or np.issubdtype(labels.dtype, np.bool_)
        or not np.isin(labels, (0, 1)).all()
    ):
        raise ValueError("oracle-label leakage requires a nonempty binary integer target vector")
    features = np.eye(2, dtype=np.float32)[labels.astype(np.int64, copy=False)]
    transform = {
        "schema": "loslab.intentional_label_leakage.v1",
        "status": "INTENTIONAL",
        "source_field": "data.y",
        "purpose": "evaluator_ceiling_and_leakage_detection_only",
        "deployment_eligible": False,
        "source_targets_sha256": _array_sha256(labels),
        "leaked_features_sha256": _array_sha256(features),
    }
    return features, transform


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


def _declared_partition_support(data: ChannelDataset, indices: np.ndarray) -> dict[str, Any]:
    """Summarize observed row/class/group support without asserting independence."""
    selected = [int(index) for index in indices]
    group_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    domain_counts: dict[str, int] = {}
    for index in selected:
        group_id = str(data.group[index])
        source_id = str(data.source[index])
        domain_id = str(data.domain[index])
        if any(not value or value != value.strip() for value in (group_id, source_id, domain_id)):
            raise ValueError("partition support requires canonical group/source/domain IDs")
        group_counts[group_id] = group_counts.get(group_id, 0) + 1
        source_counts[source_id] = source_counts.get(source_id, 0) + 1
        domain_counts[domain_id] = domain_counts.get(domain_id, 0) + 1
    groups = sorted(group_counts)
    class_counts = {str(label): int(np.count_nonzero(data.y[indices] == label)) for label in (0, 1)}
    assignments = [
        {"row_index": index, "declared_group_id": str(data.group[index])} for index in selected
    ]
    report = {
        "row_count": len(selected),
        "class_counts": class_counts,
        "class_availability": {label: count > 0 for label, count in class_counts.items()},
        "declared_group_count": len(groups),
        "declared_group_roster_sha256": canonical_hash(groups),
        "effective_declared_group_count_kish": _kish_effective_group_count(
            list(group_counts.values())
        ),
        "source_row_counts": dict(sorted(source_counts.items())),
        "domain_row_counts": dict(sorted(domain_counts.items())),
        "assignment_sha256": canonical_hash(assignments),
        "independence_identity_source": "ChannelDataset.group",
        "physical_independence_status": "not_established_by_runner",
    }
    _validate_evaluation_payload(report, "model-partition-support.v1.schema.json")
    return report


def _resolve_benchmark_test_cohort(
    *,
    paths: Sequence[str | Path],
    dataset_names: Sequence[str],
    datasets: Sequence[tuple[str, str, ChannelDataset]],
    registry: SplitPolicyRegistry,
    registry_attestation: Any,
    manifest_path: str | Path,
    declaration_path: str | Path,
    declaration_signature_path: str | Path,
    declaration_allowed_signers_path: str | Path,
    declaration_signer_id: str,
    test_cohort: str,
    locked_indices: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]],
) -> tuple[np.ndarray, dict[str, Any]]:
    if test_cohort not in TEST_COHORTS:
        raise ValueError(f"unsupported benchmark test cohort: {test_cohort}")
    if len(paths) != len(dataset_names) or len(set(dataset_names)) != len(dataset_names):
        raise ValueError("benchmark dataset names must uniquely match the ordered dataset paths")
    if set(dataset_names) != {name for name, _digest, _dataset in datasets}:
        raise ValueError("benchmark dataset names do not match the loaded dataset roster")

    declaration_artifact = verify_openssh_signed_artifact(
        declaration_path,
        declaration_signature_path,
        trusted_allowed_signers_path=declaration_allowed_signers_path,
        signer_id=declaration_signer_id,
        namespace="umcd-benchmark-cohort-v1@github.com",
    )
    declaration = json.loads(
        declaration_artifact.payload.decode("utf-8"),
        object_pairs_hook=_reject_duplicate_json_keys,
    )
    schema_root = Path(__file__).resolve().parents[2] / "datasets" / "schema"
    declaration_schema = json.loads(
        (schema_root / "benchmark-cohort-declaration.v1.schema.json").read_text(encoding="utf-8")
    )
    _validate_json_schema(declaration, declaration_schema, "signed benchmark cohort declaration")

    manifest_bytes = _stable_input_bytes(manifest_path, max_bytes=64 * 1024**2)
    try:
        benchmark_manifest = json.loads(
            manifest_bytes.decode("utf-8"), object_pairs_hook=_reject_duplicate_json_keys
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("benchmark partition manifest is not valid UTF-8 JSON") from error
    manifest_schema = json.loads(
        (schema_root / "benchmark-partition-manifest.v1.schema.json").read_text(encoding="utf-8")
    )
    _validate_json_schema(benchmark_manifest, manifest_schema, "benchmark partition manifest")
    registry_attestation_dict = {
        "artifact_sha256": registry_attestation.artifact_sha256,
        "signature_sha256": registry_attestation.signature_sha256,
        "allowed_signers_sha256": registry_attestation.allowed_signers_sha256,
        "signer_id": registry_attestation.signer_id,
        "namespace": registry_attestation.namespace,
    }
    declaration_attestation_dict = {
        "artifact_sha256": declaration_artifact.artifact_sha256,
        "signature_sha256": declaration_artifact.signature_sha256,
        "allowed_signers_sha256": declaration_artifact.allowed_signers_sha256,
        "signer_id": declaration_artifact.signer_id,
        "namespace": declaration_artifact.namespace,
    }
    replay = verify_benchmark_manifest_replay(
        benchmark_manifest,
        registry,
        declaration,
        datasets,
        registry_attestation=registry_attestation_dict,
        declaration_attestation=declaration_attestation_dict,
    )

    offsets: dict[str, int] = {}
    cursor = 0
    dataset_by_name = {name: dataset for name, _digest, dataset in datasets}
    for name in dataset_names:
        offsets[name] = cursor
        cursor += len(dataset_by_name[name].y)
    cohort_rows = replay["rows_by_dataset"]
    indices_by_cohort: dict[str, np.ndarray] = {}
    for cohort in BENCHMARK_COHORTS:
        indices_by_cohort[cohort] = np.asarray(
            sorted(
                offsets[name] + local_index
                for name in dataset_names
                for local_index in cohort_rows[name][cohort]
            ),
            dtype=np.int64,
        )
    locked_train, locked_validation, locked_test, locked_calibration = locked_indices[:4]
    for cohort_rows_indices, locked_rows, label in (
        (indices_by_cohort["train"], locked_train, "train"),
        (indices_by_cohort["validation"], locked_validation, "validation"),
        (indices_by_cohort["calibration"], locked_calibration, "calibration"),
    ):
        if not np.array_equal(cohort_rows_indices, np.sort(locked_rows)):
            raise ValueError(f"benchmark {label} rows disagree with the signed split lock")
    if not set(indices_by_cohort[test_cohort]).issubset(set(locked_test.tolist())):
        raise ValueError(f"benchmark {test_cohort} rows escape the signed test partition")
    if len(indices_by_cohort[test_cohort]) == 0:
        raise ValueError(f"benchmark {test_cohort} cohort is unsupported: it contains no rows")

    unsupported = _unsupported_benchmark_test_cohorts(benchmark_manifest["cohorts"])
    return indices_by_cohort[test_cohort], {
        "status": "PASS_INPUT_SCOPED_REPLAY",
        "qualification_status": "NOT_QUALIFIED",
        "selected_test_cohort": test_cohort,
        "manifest_sha256": replay["manifest_sha256"],
        "manifest_file_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "declaration_artifact_sha256": declaration_artifact.artifact_sha256,
        "declaration_signature_sha256": declaration_artifact.signature_sha256,
        "declaration_allowed_signers_sha256": declaration_artifact.allowed_signers_sha256,
        "declaration_signer_id": declaration_artifact.signer_id,
        "declaration_namespace": declaration_artifact.namespace,
        "cohort_summary": benchmark_manifest["cohorts"],
        "unsupported_test_cohorts": unsupported,
        "selected_test_indices_sha256": canonical_hash(indices_by_cohort[test_cohort].tolist()),
        "registry_sha256": replay["registry_sha256"],
    }


def _locked_split_indices(
    data: ChannelDataset, registry: SplitPolicyRegistry
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Resolve every channel row through its frozen split parent or child ID."""
    split_for_row: list[str] = []
    represented_parents: set[str] = set()
    group_split: dict[str, str] = {}
    parent_for_row: list[str] = []
    component_for_row: list[str] = []
    subject_for_row: list[str] = []
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
        parent_for_row.append(parent_id)
        component_for_row.append(registry.subject_to_component[parent_id])
        subject_for_row.append(subject_id)
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
    partition_indices = {
        "train": train,
        "validation": validation,
        "calibration": calibration,
        "test": test,
    }
    partition_support: dict[str, Any] = {}
    for partition, indices in partition_indices.items():
        selected = [int(index) for index in indices]
        parent_ids = sorted({parent_for_row[index] for index in selected})
        component_ids = sorted({component_for_row[index] for index in selected})
        group_ids = sorted({str(data.group[index]) for index in selected})
        group_row_counts: dict[str, int] = {}
        for index in selected:
            group_id = str(data.group[index])
            group_row_counts[group_id] = group_row_counts.get(group_id, 0) + 1
        effective_declared_group_count = _kish_effective_group_count(
            list(group_row_counts.values())
        )
        source_counts: dict[str, int] = {}
        domain_counts: dict[str, int] = {}
        for index in selected:
            source = data.source[index]
            domain = data.domain[index]
            if (
                not isinstance(source, (str, np.str_))
                or not str(source)
                or str(source) != str(source).strip()
                or not isinstance(domain, (str, np.str_))
                or not str(domain)
                or str(domain) != str(domain).strip()
            ):
                raise ValueError("signed benchmark split requires canonical source and domain IDs")
            source_counts[str(source)] = source_counts.get(str(source), 0) + 1
            domain_counts[str(domain)] = domain_counts.get(str(domain), 0) + 1
        class_counts = {
            str(label): int(np.count_nonzero(data.y[indices] == label)) for label in (0, 1)
        }
        assignment_rows = [
            {
                "row_index": index,
                "split_subject_id": subject_for_row[index],
                "parent_subject_id": parent_for_row[index],
                "component_id": component_for_row[index],
                "declared_group_id": str(data.group[index]),
            }
            for index in selected
        ]
        partition_support[partition] = {
            "row_count": len(selected),
            "parent_count": len(parent_ids),
            "parent_roster_sha256": canonical_hash(parent_ids),
            "leakage_component_count": len(component_ids),
            "leakage_component_roster_sha256": canonical_hash(component_ids),
            "declared_group_count": len(group_ids),
            "declared_group_roster_sha256": canonical_hash(group_ids),
            "effective_declared_group_count_kish": effective_declared_group_count,
            "class_counts": class_counts,
            "class_availability": {label: count > 0 for label, count in class_counts.items()},
            "source_row_counts": dict(sorted(source_counts.items())),
            "domain_row_counts": dict(sorted(domain_counts.items())),
            "assignment_sha256": canonical_hash(assignment_rows),
        }
    support_report: dict[str, Any] = {
        "schema": "loslab.locked_benchmark_split_support.v2",
        "status": "PASS_INPUT_SCOPED",
        "qualification_status": "NOT_QUALIFIED",
        "physical_independence_status": "not_established_by_runner",
        "registry_sha256": registry.to_dict()["registry_sha256"],
        "partition_support": partition_support,
    }
    _validate_evaluation_payload(support_report, "locked-benchmark-split-support.v2.schema.json")
    report = {
        "mode": "signed_locked",
        "policy_sha256": registry.policy.to_dict()["policy_sha256"],
        "registry_sha256": registry.to_dict()["registry_sha256"],
        "parent_counts": {
            split: sum(registry.split_for_parent(parent) == split for parent in represented_parents)
            for split in registry.policy.split_fractions
        },
        "calibration_rows_reserved": int(np.sum(labels == "calibration")),
        "partition_support": support_report,
        "partition_support_sha256": canonical_hash(support_report),
    }
    return train, validation, test, calibration, report


def _write_benchmark_preflight_failure_receipt(
    output: str | Path,
    error: Exception,
    *,
    selected_test_cohort: Any,
) -> None:
    """Create an exclusive failure receipt when benchmark preflight has no run manifest."""
    output_root = Path(output).expanduser().absolute()
    if output_root.is_symlink():
        return
    try:
        output_root.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    receipt_path = output_root / "benchmark-cohort-preflight.json"
    status = (
        "UNSUPPORTED_COHORT" if " is unsupported:" in str(error) else "FAILED_BEFORE_RUN_MANIFEST"
    )
    receipt = {
        "schema": "loslab.benchmark_cohort_preflight.v1",
        "status": status,
        "qualification_status": "NOT_QUALIFIED",
        "selected_test_cohort": selected_test_cohort
        if isinstance(selected_test_cohort, str)
        else None,
        "error_type": type(error).__name__,
        "error_message": str(error),
    }
    schema_path = Path(__file__).resolve().parents[2] / (
        "evaluation/schema/benchmark-cohort-preflight.v1.schema.json"
    )
    if schema_path.is_file():
        _validate_json_schema(
            receipt,
            json.loads(schema_path.read_text(encoding="utf-8")),
            "benchmark preflight receipt",
        )
    payload = (json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    try:
        descriptor = os.open(
            receipt_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except FileExistsError:
        return
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
