from __future__ import annotations

import hashlib
import json

import pytest
from datasets.channel import ChannelDataset
from scripts.audit_channel_split_lineage import (
    audit_channel_datasets,
)
from tests.test_scripts.audit_channel_split_support import (
    _write_legacy_rt_dataset,
    _write_reconstructed_asset_manifest,
    _write_unmanifested_legacy_rt_dataset,
)


def test_opt_in_legacy_rt_migration_merges_cross_transmitter_spatial_parents(
    tmp_path,
) -> None:
    scene_hash = "b" * 64
    center_path = tmp_path / "center.npz"
    east_path = tmp_path / "east.npz"
    west_path = tmp_path / "west.npz"
    _write_legacy_rt_dataset(center_path, tx_name="center", rx_x=5.0, scene_hash=scene_hash)
    _write_legacy_rt_dataset(east_path, tx_name="east", rx_x=15.0, scene_hash=scene_hash)
    _write_legacy_rt_dataset(west_path, tx_name="west", rx_x=65.0, scene_hash=scene_hash)
    original_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (center_path, east_path, west_path)
    }

    with pytest.raises(ValueError, match="split_subject_id must be a canonical nonempty string"):
        audit_channel_datasets([("center", center_path)], spatial_buffer_m=30.0)

    report = audit_channel_datasets(
        [("center", center_path), ("east", east_path), ("west", west_path)],
        spatial_buffer_m=30.0,
        derive_legacy_rt_split_identities=True,
    )

    assert report["status"] == "NOT_QUALIFIED"
    assert report["identity_derivation"] == {
        "mode": "opt_in_complete_rt_legacy_metadata",
        "datasets_upgraded_in_memory": ["center", "east", "west"],
        "rows_upgraded_in_memory": 3,
        "input_hashes_cover_original_bytes": True,
    }
    assert report["lineage"]["unique_split_subjects"] == 2
    assert report["lineage"]["subjects_reused_across_inputs"] == 1
    assert [item["sha256"] for item in report["inputs"]] == [
        original_hashes["center.npz"],
        original_hashes["east.npz"],
        original_hashes["west.npz"],
    ]
    assert {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (center_path, east_path, west_path)
    } == original_hashes


def test_opt_in_legacy_rt_migration_rejects_inconsistent_scene_configuration(tmp_path) -> None:
    path = tmp_path / "legacy.npz"
    _write_legacy_rt_dataset(path, tx_name="center", rx_x=5.0, scene_hash="c" * 64)
    data = ChannelDataset.load(path)
    metadata = dict(data.metadata[0])
    metadata["scene_config"] = {"name": "legacy-city43", "block_size_m": 10.0}
    ChannelDataset(
        h=data.h,
        y=data.y,
        group=data.group,
        domain=data.domain,
        source=data.source,
        metadata=(metadata,),
        generation_report=data.generation_report,
        label_semantics=data.label_semantics,
    ).save(path)

    with pytest.raises(ValueError, match="disagrees with the saved complete RT scene identity"):
        audit_channel_datasets(
            [("legacy", path)],
            spatial_buffer_m=30.0,
            derive_legacy_rt_split_identities=True,
        )


def test_opt_in_legacy_rt_migration_rejects_asset_manifest_digest_mismatch(tmp_path) -> None:
    path = tmp_path / "legacy.npz"
    _write_legacy_rt_dataset(path, tx_name="center", rx_x=5.0, scene_hash="d" * 64)
    data = ChannelDataset.load(path)
    report = dict(data.generation_report)
    asset_manifest = dict(report["scene_asset_manifest"])
    asset_manifest["asset_set_sha256"] = "e" * 64
    report["scene_asset_manifest"] = asset_manifest
    ChannelDataset(
        h=data.h,
        y=data.y,
        group=data.group,
        domain=data.domain,
        source=data.source,
        metadata=data.metadata,
        generation_report=report,
        label_semantics=data.label_semantics,
    ).save(path)

    with pytest.raises(ValueError, match="scene asset digest disagrees with its saved manifest"):
        audit_channel_datasets(
            [("legacy", path)],
            spatial_buffer_m=30.0,
            derive_legacy_rt_split_identities=True,
        )


def test_historical_asset_reconstruction_migrates_only_in_memory_and_binds_original_bytes(
    tmp_path,
) -> None:
    dataset_path = tmp_path / "legacy.npz"
    reconstruction_path = tmp_path / "reconstruction.json"
    source_revision, assets = _write_unmanifested_legacy_rt_dataset(dataset_path, rx_x=5.0)
    _write_reconstructed_asset_manifest(
        reconstruction_path, dataset_path, source_revision=source_revision, assets=assets
    )
    original = dataset_path.read_bytes()
    original_hash = hashlib.sha256(original).hexdigest()

    report = audit_channel_datasets(
        [("historical", dataset_path)],
        spatial_buffer_m=30.0,
        derive_legacy_rt_split_identities=True,
        legacy_scene_assets_manifest=reconstruction_path,
    )

    assert report["status"] == "NOT_QUALIFIED"
    assert (
        report["identity_derivation"]["mode"] == "opt_in_historical_rt_scene_asset_reconstruction"
    )
    assert report["identity_derivation"]["rows_upgraded_in_memory"] == 1
    assert report["inputs"][0]["sha256"] == original_hash
    assert report["identity_derivation"]["scene_asset_reconstruction"]["authority"].startswith(
        "unsigned retrospective"
    )
    assert dataset_path.read_bytes() == original


def test_historical_asset_reconstruction_rejects_changed_input_hash(tmp_path) -> None:
    dataset_path = tmp_path / "legacy.npz"
    reconstruction_path = tmp_path / "reconstruction.json"
    source_revision, assets = _write_unmanifested_legacy_rt_dataset(dataset_path, rx_x=5.0)
    _write_reconstructed_asset_manifest(
        reconstruction_path, dataset_path, source_revision=source_revision, assets=assets
    )
    manifest = json.loads(reconstruction_path.read_text(encoding="utf-8"))
    manifest["datasets"][0]["input_sha256"] = "f" * 64
    reconstruction_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="bytes or row count disagree"):
        audit_channel_datasets(
            [("historical", dataset_path)],
            spatial_buffer_m=30.0,
            derive_legacy_rt_split_identities=True,
            legacy_scene_assets_manifest=reconstruction_path,
        )
