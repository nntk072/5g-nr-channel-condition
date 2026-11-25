"""Boundary checks for deterministic runner validation helpers."""

from __future__ import annotations

import json

import experiments.runner.validation as validation
import numpy as np
import pytest
from datasets.channel import ChannelDataset
from experiments.runner.validation import (
    _declared_partition_support,
    _metadata_only_features,
    _permute_training_feature_rows,
    _resolve_benchmark_test_cohort,
    _resolved_label_semantics,
    _source_only_features,
    _validation_threshold_selection_record,
    _write_benchmark_preflight_failure_receipt,
)


@pytest.mark.parametrize(
    ("features", "run_seed", "message"),
    [
        (np.asarray([1.0, 2.0]), 1, "finite 2D training matrix"),
        (np.ones((2, 2)), True, "run_seed must be an integer"),
    ],
)
def test_feature_row_permutation_rejects_invalid_inputs(features, run_seed, message):
    with pytest.raises(ValueError, match=message):
        _permute_training_feature_rows(features, run_seed=run_seed, base_seed=7)


def test_source_only_vocabulary_rejects_empty_training_indices():
    with pytest.raises(ValueError, match="training indices are invalid"):
        _source_only_features(np.asarray(["rt", "tdl"]), train_indices=np.asarray([], dtype=int))


def test_metadata_only_vocabulary_rejects_invalid_training_indices():
    values = {field: np.asarray(["a", "b"]) for field in ("group", "domain", "source")}

    with pytest.raises(ValueError, match="training indices are invalid"):
        _metadata_only_features(
            group_values=values["group"],
            domain_values=values["domain"],
            source_values=values["source"],
            train_indices=np.asarray([0, 0]),
        )


def test_threshold_selection_rejects_misaligned_evidence_arrays():
    with pytest.raises(ValueError, match="arrays must be nonempty and aligned"):
        _validation_threshold_selection_record(
            np.asarray([0, 1]),
            np.asarray([0.2]),
            np.asarray(["a", "b"]),
            np.asarray([3, 4]),
        )


def test_label_semantics_rejects_conflicting_dataset_declaration():
    data = ChannelDataset(
        h=np.ones((2, 1, 8), dtype=np.complex64),
        y=np.asarray([0, 1]),
        group=np.asarray(["a", "b"]),
        domain=np.asarray(["scene", "scene"]),
        source=np.asarray(["rt", "rt"]),
        metadata=(
            {"label_semantics": "profile_proxy"},
            {"label_semantics": "profile_proxy"},
        ),
        label_semantics="geometric_direct_path_oracle",
    )

    with pytest.raises(ValueError, match="conflicts with per-sample label metadata"):
        _resolved_label_semantics(data)


def test_label_semantics_rejects_missing_per_row_meaning():
    data = ChannelDataset(
        h=np.ones((2, 1, 8), dtype=np.complex64),
        y=np.asarray([0, 1]),
        group=np.asarray(["a", "b"]),
        domain=np.asarray(["scene", "scene"]),
        source=np.asarray(["legacy", "legacy"]),
        metadata=({}, {}),
    )

    with pytest.raises(ValueError, match="one explicit, consistent label meaning"):
        _resolved_label_semantics(data)


def test_partition_support_rejects_noncanonical_source_identifiers():
    data = ChannelDataset(
        h=np.ones((2, 1, 8), dtype=np.complex64),
        y=np.asarray([0, 1]),
        group=np.asarray(["a", "b"]),
        domain=np.asarray(["scene", "scene"]),
        source=np.asarray([" padded ", "legacy"]),
        metadata=({}, {}),
    )

    with pytest.raises(ValueError, match="canonical group/source/domain IDs"):
        _declared_partition_support(data, np.asarray([0, 1]))


@pytest.mark.parametrize(
    ("paths", "dataset_names", "datasets", "test_cohort", "message"),
    [
        ((), (), (), "unsupported", "unsupported benchmark test cohort"),
        (("a.npz",), (), (), "site_held_out", "dataset names must uniquely match"),
        (
            ("a.npz",),
            ("run",),
            (),
            "site_held_out",
            "names do not match the loaded dataset roster",
        ),
    ],
)
def test_benchmark_cohort_resolver_checks_roster_before_signature_io(
    paths, dataset_names, datasets, test_cohort, message
):
    arguments = {
        "paths": paths,
        "dataset_names": dataset_names,
        "datasets": datasets,
        "registry": None,
        "registry_attestation": None,
        "manifest_path": "unused-manifest.json",
        "declaration_path": "unused-declaration.json",
        "declaration_signature_path": "unused-declaration.sig",
        "declaration_allowed_signers_path": "unused-signers",
        "declaration_signer_id": "reviewer",
        "test_cohort": test_cohort,
        "locked_indices": (),
    }

    with pytest.raises(ValueError, match=message):
        _resolve_benchmark_test_cohort(**arguments)


def test_benchmark_failure_receipt_ignores_symlink_and_non_directory_outputs(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    output_link = tmp_path / "output-link"
    output_link.symlink_to(target, target_is_directory=True)
    _write_benchmark_preflight_failure_receipt(
        output_link, ValueError("unsupported"), selected_test_cohort="fixture"
    )
    assert not (target / "benchmark-cohort-preflight.json").exists()

    output_file = tmp_path / "output-file"
    output_file.write_text("occupied", encoding="utf-8")
    _write_benchmark_preflight_failure_receipt(
        output_file, ValueError("unsupported"), selected_test_cohort="fixture"
    )
    assert output_file.read_text(encoding="utf-8") == "occupied"


def _stub_benchmark_cohort_resolver(
    monkeypatch, rows_by_cohort, *, manifest_bytes=b'{"cohorts":{}}'
):
    from types import SimpleNamespace

    rows = {
        "train": [0],
        "validation": [1],
        "calibration": [],
        "site_held_out": [2],
        "site_array_held_out": [],
        "real_external": [],
    }
    rows.update(rows_by_cohort)
    artifact = SimpleNamespace(
        payload=b"{}",
        artifact_sha256="a" * 64,
        signature_sha256="b" * 64,
        allowed_signers_sha256="c" * 64,
        signer_id="reviewer",
        namespace="umcd-benchmark-cohort-v1@github.com",
    )
    monkeypatch.setattr(
        validation, "verify_openssh_signed_artifact", lambda *args, **kwargs: artifact
    )
    monkeypatch.setattr(validation, "_stable_input_bytes", lambda *args, **kwargs: manifest_bytes)
    monkeypatch.setattr(validation, "_validate_json_schema", lambda *args, **kwargs: None)
    monkeypatch.setattr(validation, "_unsupported_benchmark_test_cohorts", lambda _: [])
    monkeypatch.setattr(
        validation,
        "verify_benchmark_manifest_replay",
        lambda *args, **kwargs: {
            "rows_by_dataset": {"run": rows},
            "manifest_sha256": "d" * 64,
            "registry_sha256": "e" * 64,
        },
    )
    data = ChannelDataset(
        h=np.ones((3, 1, 8), dtype=np.complex64),
        y=np.asarray([0, 1, 0]),
        group=np.asarray(["a", "b", "c"]),
        domain=np.asarray(["scene"] * 3),
        source=np.asarray(["fixture"] * 3),
        metadata=({}, {}, {}),
    )
    arguments = {
        "paths": ("run.npz",),
        "dataset_names": ("run",),
        "datasets": (("run", "f" * 64, data),),
        "registry": object(),
        "registry_attestation": artifact,
        "manifest_path": "manifest.json",
        "declaration_path": "declaration.json",
        "declaration_signature_path": "declaration.sig",
        "declaration_allowed_signers_path": "allowed-signers",
        "declaration_signer_id": "reviewer",
        "test_cohort": "site_held_out",
        "locked_indices": (
            np.asarray([0]),
            np.asarray([1]),
            np.asarray([2]),
            np.asarray([], dtype=np.int64),
            {},
        ),
    }
    return arguments, rows


@pytest.mark.parametrize(
    ("locked_position", "message"),
    [
        (0, "train rows disagree"),
        (1, "validation rows disagree"),
        (3, "calibration rows disagree"),
    ],
)
def test_benchmark_cohort_resolver_enforces_locked_non_test_rows(
    monkeypatch, locked_position, message
):
    arguments, rows = _stub_benchmark_cohort_resolver(monkeypatch, {})
    indices = list(arguments["locked_indices"])
    indices[locked_position] = np.asarray([2])
    arguments["locked_indices"] = tuple(indices)

    with pytest.raises(ValueError, match=message):
        validation._resolve_benchmark_test_cohort(**arguments)


@pytest.mark.parametrize(
    ("test_rows", "message"),
    [([3], "escape the signed test partition"), ([], "contains no rows")],
)
def test_benchmark_cohort_resolver_rejects_unsupported_test_rows(monkeypatch, test_rows, message):
    arguments, _ = _stub_benchmark_cohort_resolver(monkeypatch, {"site_held_out": test_rows})

    with pytest.raises(ValueError, match=message):
        validation._resolve_benchmark_test_cohort(**arguments)


def test_benchmark_cohort_resolver_rejects_invalid_manifest_json(monkeypatch):
    arguments, _ = _stub_benchmark_cohort_resolver(monkeypatch, {}, manifest_bytes=b"\xff")

    with pytest.raises(ValueError, match="not valid UTF-8 JSON"):
        validation._resolve_benchmark_test_cohort(**arguments)


def test_benchmark_failure_receipt_skips_optional_schema_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(validation.Path, "is_file", lambda _path: False)
    _write_benchmark_preflight_failure_receipt(
        tmp_path / "output", ValueError("unsupported"), selected_test_cohort="fixture"
    )

    receipt = tmp_path / "output" / "benchmark-cohort-preflight.json"
    assert json.loads(receipt.read_text(encoding="utf-8"))["status"] == "FAILED_BEFORE_RUN_MANIFEST"
