from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pytest
from scripts.audit_channel_split_lineage import (
    _parse_dataset,
    _read_stable_regular_file,
    _validate_split_audit_report,
    audit_channel_datasets,
    main,
)
from tests.test_scripts.audit_channel_split_support import (
    _write_dataset,
    _write_inventory,
)


def test_split_audit_report_schema_rejects_qualified_claims_and_unknown_fields() -> None:
    report = {
        "schema": "loslab.channel_split_lineage_audit_bundle.v10",
        "status": "NOT_QUALIFIED",
        "spatial_buffer_m": 0,
        "identity_derivation": {
            "mode": "disabled",
            "datasets_upgraded_in_memory": [],
            "rows_upgraded_in_memory": 0,
            "input_hashes_cover_original_bytes": True,
        },
        "input_completeness": {"status": "unverified_input_scope", "authority": "none"},
        "inputs": [
            {"name": "sample", "path": "sample.npz", "sha256": "a" * 64, "bytes": 0, "rows": 0}
        ],
        "lineage": {},
        "subjects": [],
        "interpretation": "Diagnostic only.",
    }
    _validate_split_audit_report(report)
    with pytest.raises(ValueError, match="violates schema"):
        _validate_split_audit_report({**report, "status": "QUALIFIED"})
    with pytest.raises(ValueError, match="violates schema"):
        _validate_split_audit_report({**report, "unreviewed_claim": True})


def test_split_audit_report_schema_checks_signed_registry_summary_counts() -> None:
    report = {
        "schema": "loslab.channel_split_lineage_audit_bundle.v10",
        "status": "NOT_QUALIFIED",
        "spatial_buffer_m": 0,
        "identity_derivation": {
            "mode": "disabled",
            "datasets_upgraded_in_memory": [],
            "rows_upgraded_in_memory": 0,
            "input_hashes_cover_original_bytes": True,
        },
        "input_completeness": {"status": "unverified_input_scope", "authority": "none"},
        "inputs": [
            {"name": "sample", "path": "sample.npz", "sha256": "a" * 64, "bytes": 0, "rows": 0}
        ],
        "lineage": {},
        "subjects": [],
        "interpretation": "Diagnostic only.",
        "split_registry_verification": {
            "status": "PASS_SIGNED_PARENT_AND_FOOTPRINT_AUDIT",
            "schema": "umcd.split_registry.v3",
            "registry_sha256": "a" * 64,
            "signer_id": "reviewer",
            "signature_namespace": "split-registry-v1",
            "registry_file_sha256": "a" * 64,
            "signature_sha256": "a" * 64,
            "allowed_signers_sha256": "a" * 64,
            "parent_roster_audit": {},
            "spatiotemporal_audit": {},
            "children_audited": False,
            "child_bindings_verified": False,
            "child_artifact_existence_verified": False,
            "child_artifact_bytes_verified": False,
            "child_artifact_semantic_rows_verified": False,
            "child_artifact_semantic_count": 0,
            "child_artifact_byte_only_count": 0,
            "child_artifact_trajectory_sample_count": 0,
            "child_artifact_trajectory_count": 0,
            "child_artifact_episode_count": 0,
            "child_artifact_scenario_gbsm_run_count": 0,
            "child_artifact_scenario_gbsm_parent_count": 0,
            "complete_split_integrity_audit": False,
            "child_roster_audits": [],
            "child_artifact_roster_audits": [],
        },
    }
    _validate_split_audit_report(report)
    verification = dict(report["split_registry_verification"])
    verification["child_artifact_episode_count"] = -1
    with pytest.raises(ValueError, match="child_artifact_episode_count"):
        _validate_split_audit_report({**report, "split_registry_verification": verification})


def test_dataset_cli_parser_rejects_padded_names(tmp_path: Path) -> None:
    parsed = _parse_dataset(f"dataset={tmp_path / 'run.npz'}")
    assert parsed == ("dataset", tmp_path / "run.npz")
    for malformed in (" dataset=run.npz", "dataset =run.npz"):
        with pytest.raises(argparse.ArgumentTypeError, match="canonical"):
            _parse_dataset(malformed)


def test_complete_split_audit_requires_signed_corpus_inventory_and_registry(tmp_path) -> None:
    dataset = tmp_path / "not-read.npz"
    with pytest.raises(ValueError, match="requires a signed frozen corpus inventory"):
        audit_channel_datasets(
            [("dataset", dataset)],
            spatial_buffer_m=0.0,
            require_complete_split_audit=True,
        )

    with pytest.raises(ValueError, match="requires a signed split registry"):
        audit_channel_datasets(
            [("dataset", dataset)],
            spatial_buffer_m=0.0,
            inventory_manifest=tmp_path / "inventory.json",
            inventory_signature=tmp_path / "inventory.json.sig",
            require_complete_split_audit=True,
        )


def test_complete_split_audit_flag_requires_strict_boolean(tmp_path) -> None:
    with pytest.raises(TypeError, match="must be a boolean"):
        audit_channel_datasets(
            [("dataset", tmp_path / "not-read.npz")],
            spatial_buffer_m=0.0,
            require_complete_split_audit=1,
        )


def test_audit_rejects_symlinked_inventory_manifest(tmp_path) -> None:
    dataset_path = tmp_path / "run.npz"
    inventory_path = tmp_path / "inventory.json"
    inventory_alias = tmp_path / "inventory-alias.json"
    _write_dataset(dataset_path, "parent-a", 0.0, "a" * 64)
    _write_inventory(inventory_path, dataset_path)
    inventory_alias.symlink_to(inventory_path)

    with pytest.raises(ValueError, match="inventory manifest must not be a symlink"):
        audit_channel_datasets(
            [("pilot", dataset_path)],
            spatial_buffer_m=0.0,
            inventory_manifest=inventory_alias,
        )


def test_stable_regular_file_reader_enforces_byte_limit(tmp_path) -> None:
    path = tmp_path / "bounded.json"
    path.write_bytes(b"12345")

    with pytest.raises(ValueError, match="manifest exceeds 4 bytes"):
        _read_stable_regular_file(path, label="manifest", max_bytes=4)

    assert _read_stable_regular_file(path, label="manifest", max_bytes=5) == b"12345"


@pytest.mark.parametrize("max_bytes", [-1, True, 1.5])
def test_stable_regular_file_reader_rejects_invalid_byte_limit(tmp_path, max_bytes) -> None:
    path = tmp_path / "bounded.json"
    path.write_bytes(b"1")

    with pytest.raises(ValueError, match="max_bytes must be a nonnegative integer"):
        _read_stable_regular_file(path, label="manifest", max_bytes=max_bytes)


def test_audit_rejects_symlink_alias_for_frozen_dataset_input(tmp_path) -> None:
    dataset_path = tmp_path / "run.npz"
    dataset_alias = tmp_path / "run-alias.npz"
    inventory_path = tmp_path / "inventory.json"
    _write_dataset(dataset_path, "parent-a", 0.0, "a" * 64)
    _write_inventory(inventory_path, dataset_path)
    dataset_alias.symlink_to(dataset_path)

    with pytest.raises(ValueError, match="dataset path differs from frozen inventory"):
        audit_channel_datasets(
            [("pilot", dataset_alias)],
            spatial_buffer_m=0.0,
            inventory_manifest=inventory_path,
        )


def test_frozen_inventory_read_rejects_symlinked_parent_directory(tmp_path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_dataset = outside / "run.npz"
    _write_dataset(outside_dataset, "parent-a", 0.0, "a" * 64)
    dataset_root = tmp_path / "dataset-root"
    dataset_root.mkdir()
    (dataset_root / "nested").symlink_to(outside, target_is_directory=True)
    inventory_path = dataset_root / "inventory.json"
    payload = outside_dataset.read_bytes()
    inventory_path.write_text(
        json.dumps(
            {
                "schema": "umcd.channel_dataset_inventory.v1",
                "frozen": True,
                "inventory_id": "symlink-parent-test",
                "datasets": [
                    {
                        "name": "pilot",
                        "path": "nested/run.npz",
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "bytes": len(payload),
                        "rows": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="symlink or non-directory component"):
        audit_channel_datasets(
            [("pilot", dataset_root / "nested" / "run.npz")],
            spatial_buffer_m=0.0,
            inventory_manifest=inventory_path,
        )


def test_cli_writes_report_once_and_refuses_overwrite(tmp_path, monkeypatch, capsys) -> None:
    dataset_path = tmp_path / "run.npz"
    inventory_path = tmp_path / "inventory.json"
    output_path = tmp_path / "report.json"
    _write_dataset(dataset_path, "parent-a", 0.0, "a" * 64)
    _write_inventory(inventory_path, dataset_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_channel_split_lineage.py",
            "--dataset",
            f"pilot={dataset_path}",
            "--spatial-buffer-m",
            "0",
            "--inventory-manifest",
            str(inventory_path),
            "--output",
            str(output_path),
        ],
    )

    main()

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["status"] == "NOT_QUALIFIED"
    assert report["input_completeness"]["status"] == "matched_frozen_inventory"
    assert json.loads(capsys.readouterr().out)["unique_split_subjects"] == 1
    with pytest.raises(SystemExit, match="refusing to overwrite existing output"):
        main()
