from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import jsonschema
import numpy as np
import pytest
from datasets.channel import ChannelDataset
from datasets.rt_split_identity import rt_spatial_split_subject_id
from datasets.split_policy import SplitPolicy, read_split_policy_registry
from scripts.build_channel_split_registry import build_candidate_split_registry


def _write_dataset(path: Path, *, index: int) -> None:
    scene_hash = f"{index + 1:064x}"
    position = (float(index * 10), 0.0, 1.5)
    subject_id, complete = rt_spatial_split_subject_id(
        scene_asset_set_sha256=scene_hash,
        scene_asset_manifest_complete=True,
        scene_name=f"scene-{index}",
        rx_position_m=position,
        block_size_m=1.0,
    )
    ChannelDataset(
        h=np.ones((1, 1, 8), dtype=np.complex64),
        y=np.asarray([index % 2], dtype=np.uint8),
        group=np.asarray([subject_id]),
        domain=np.asarray([f"scene-{index}"]),
        source=np.asarray(["sionna_rt"]),
        metadata=(
            {
                "scene": f"scene-{index}",
                "rx": list(position),
                "scene_asset_manifest_complete": complete,
                "scene_asset_set_sha256": scene_hash,
                "split_subject_id": subject_id,
                "split_parent_type": "network_episode",
                "split_source_family": "sionna_rt",
                "split_leakage_relations": [],
                "split_identity_basis": "scene_asset_set_sha256+receiver_xy_grid",
                "split_identity_complete": complete,
                "split_block_size_m": 1.0,
                "split_coordinate_frame_id": scene_hash,
                "split_positions_m": list(position),
            },
        ),
    ).save(path)


def _fixture(tmp_path: Path) -> tuple[list[tuple[str, Path]], Path]:
    datasets: list[tuple[str, Path]] = []
    for index in range(40):
        path = tmp_path / f"scene-{index}.npz"
        _write_dataset(path, index=index)
        datasets.append((f"scene-{index}", path))
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        json.dumps(SplitPolicy.release_v2(seed="candidate-split-build").to_dict()),
        encoding="utf-8",
    )
    return datasets, policy_path


def test_candidate_split_registry_binds_exact_inputs_and_freezes_immutable_lock(tmp_path):
    datasets, policy_path = _fixture(tmp_path)
    registry_path = tmp_path / "release" / "split-registry.json"
    report_path = tmp_path / "release" / "build-report.json"

    report = build_candidate_split_registry(
        datasets,
        policy_path=policy_path,
        spatial_buffer_m=0.0,
        registry_path=registry_path,
        report_path=report_path,
    )

    registry = read_split_policy_registry(registry_path)
    saved_report = json.loads(report_path.read_text(encoding="utf-8"))
    schema_path = (
        Path(__file__).parents[2]
        / "datasets/schema/channel-split-registry-build-report.v2.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    assert list(jsonschema.Draft202012Validator(schema).iter_errors(saved_report)) == []
    assert report["status"] == "CANDIDATE_NOT_EXTERNALLY_REVIEWED"
    assert report["qualification_status"] == "NOT_QUALIFIED"
    assert saved_report["registry_sha256"] == registry.to_dict()["registry_sha256"]
    assert saved_report["policy_sha256"] == registry.policy.to_dict()["policy_sha256"]
    assert saved_report["registry_file_sha256"] == report["registry_file_sha256"]
    assert len(saved_report["inputs"]) == len(datasets)
    for item, (_name, path) in zip(saved_report["inputs"], sorted(datasets), strict=True):
        assert item["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert item["bytes"] == path.stat().st_size
        assert item["split_metadata_migration"] is None
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_candidate_split_registry(
            datasets,
            policy_path=policy_path,
            spatial_buffer_m=0.0,
            registry_path=registry_path,
            report_path=tmp_path / "second-report.json",
        )


def test_candidate_split_registry_reuses_exact_cached_build_node(tmp_path):
    datasets, policy_path = _fixture(tmp_path)
    cache_root = tmp_path / "build-cache"
    first_registry = tmp_path / "release-a" / "split-registry.json"
    first_report = tmp_path / "release-a" / "build-report.json"
    second_registry = tmp_path / "release-b" / "split-registry.json"
    second_report = tmp_path / "release-b" / "build-report.json"

    first = build_candidate_split_registry(
        datasets,
        policy_path=policy_path,
        spatial_buffer_m=0.0,
        registry_path=first_registry,
        report_path=first_report,
        build_cache_root=cache_root,
    )
    second = build_candidate_split_registry(
        datasets,
        policy_path=policy_path,
        spatial_buffer_m=0.0,
        registry_path=second_registry,
        report_path=second_report,
        build_cache_root=cache_root,
    )

    assert first["qualification_status"] == second["qualification_status"] == "NOT_QUALIFIED"
    assert first["build_node"]["reused"] is False
    assert second["build_node"]["reused"] is True
    assert first["build_node"]["node_key"] == second["build_node"]["node_key"]
    assert first_registry.read_bytes() == second_registry.read_bytes()
    assert first_report.read_bytes() == second_report.read_bytes()
    assert len(list((cache_root / "nodes").glob("*.json"))) == 1


def _write_legacy_rt_dataset(path: Path, *, index: int) -> None:
    scene_name = f"legacy-scene-{index}"
    block_size = 1.0
    asset_payload = f"scene-bytes-{index}".encode("utf-8")
    asset_rows = [
        {
            "path": f"{scene_name}.xml",
            "bytes": len(asset_payload),
            "sha256": hashlib.sha256(asset_payload).hexdigest(),
        }
    ]
    asset_digest = hashlib.sha256(
        json.dumps(asset_rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    asset_manifest = {
        "complete": True,
        "asset_count": 1,
        "assets": asset_rows,
        "asset_set_sha256": asset_digest,
        "unresolved_references": [],
    }
    rx = [float(index * 10), 0.0, 1.5]
    metadata = {
        "scene": scene_name,
        "scene_config": {"name": scene_name, "block_size_m": block_size},
        "scene_asset_set_sha256": asset_digest,
        "scene_asset_manifest_complete": True,
        "rx": rx,
    }
    ChannelDataset(
        h=np.ones((1, 1, 8), dtype=np.complex64),
        y=np.asarray([index % 2], dtype=np.uint8),
        group=np.asarray([f"legacy-group-{index}"]),
        domain=np.asarray([scene_name]),
        source=np.asarray(["sionna_rt"]),
        metadata=(metadata,),
        generation_report={
            "scene_config": {"name": scene_name, "block_size_m": block_size},
            "scene_asset_manifest": asset_manifest,
        },
    ).save(path)


def test_candidate_split_registry_migrates_only_complete_legacy_rt_identity(tmp_path):
    _modern_datasets, policy_path = _fixture(tmp_path)
    legacy_datasets = []
    for index in range(40):
        path = tmp_path / f"legacy-{index}.npz"
        _write_legacy_rt_dataset(path, index=index)
        legacy_datasets.append((f"legacy-{index}", path))

    registry_path = tmp_path / "legacy-release" / "split-registry.json"
    report_path = tmp_path / "legacy-release" / "build-report.json"
    report = build_candidate_split_registry(
        legacy_datasets,
        policy_path=policy_path,
        spatial_buffer_m=0.0,
        registry_path=registry_path,
        report_path=report_path,
    )

    saved_report = json.loads(report_path.read_text(encoding="utf-8"))
    schema_path = (
        Path(__file__).parents[2]
        / "datasets/schema/channel-split-registry-build-report.v2.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert list(jsonschema.Draft202012Validator(schema).iter_errors(saved_report)) == []
    assert report["qualification_status"] == "NOT_QUALIFIED"
    assert len(saved_report["inputs"]) == len(legacy_datasets)
    for item, (_name, path) in zip(saved_report["inputs"], sorted(legacy_datasets), strict=True):
        assert item["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert item["split_metadata_migration"] == "legacy_rt_complete_scene_identity_v1"


def test_legacy_rt_migration_rejects_incomplete_asset_closure_before_publication(tmp_path):
    _modern_datasets, policy_path = _fixture(tmp_path)
    legacy_path = tmp_path / "legacy-incomplete.npz"
    _write_legacy_rt_dataset(legacy_path, index=0)
    dataset = ChannelDataset.load(legacy_path)
    report = dict(dataset.generation_report)
    manifest = dict(report["scene_asset_manifest"])
    manifest["complete"] = False
    report["scene_asset_manifest"] = manifest
    ChannelDataset(
        h=dataset.h,
        y=dataset.y,
        group=dataset.group,
        domain=dataset.domain,
        source=dataset.source,
        metadata=dataset.metadata,
        generation_report=report,
        label_semantics=dataset.label_semantics,
    ).save(legacy_path)

    registry_path = tmp_path / "incomplete-release" / "split-registry.json"
    report_path = tmp_path / "incomplete-release" / "build-report.json"
    with pytest.raises(ValueError, match="internally consistent scene closure"):
        build_candidate_split_registry(
            [("legacy", legacy_path)],
            policy_path=policy_path,
            spatial_buffer_m=0.0,
            registry_path=registry_path,
            report_path=report_path,
        )
    assert not registry_path.exists()
    assert not report_path.exists()


def test_candidate_registry_refuses_to_overwrite_partial_legacy_split_metadata(tmp_path):
    _modern_datasets, policy_path = _fixture(tmp_path)
    legacy_path = tmp_path / "legacy-partial.npz"
    _write_legacy_rt_dataset(legacy_path, index=0)
    dataset = ChannelDataset.load(legacy_path)
    metadata = dict(dataset.metadata[0])
    metadata["split_leakage_relations"] = []
    ChannelDataset(
        h=dataset.h,
        y=dataset.y,
        group=dataset.group,
        domain=dataset.domain,
        source=dataset.source,
        metadata=(metadata,),
        generation_report=dataset.generation_report,
        label_semantics=dataset.label_semantics,
    ).save(legacy_path)

    registry_path = tmp_path / "partial-release" / "split-registry.json"
    report_path = tmp_path / "partial-release" / "build-report.json"
    with pytest.raises(ValueError, match="split_subject_id"):
        build_candidate_split_registry(
            [("legacy", legacy_path)],
            policy_path=policy_path,
            spatial_buffer_m=0.0,
            registry_path=registry_path,
            report_path=report_path,
        )
    assert not registry_path.exists()
    assert not report_path.exists()


@pytest.mark.parametrize("bad_name", ["", " scene", "scene "])
def test_candidate_split_registry_rejects_noncanonical_dataset_names(tmp_path, bad_name):
    datasets, policy_path = _fixture(tmp_path)

    with pytest.raises(ValueError, match="canonical strings"):
        build_candidate_split_registry(
            [(bad_name, datasets[0][1])],
            policy_path=policy_path,
            spatial_buffer_m=0.0,
            registry_path=tmp_path / "registry.json",
            report_path=tmp_path / "report.json",
        )


def test_candidate_split_registry_rejects_oversized_input_before_loading(tmp_path):
    datasets, policy_path = _fixture(tmp_path)
    first_path = datasets[0][1]

    with pytest.raises(ValueError, match="byte limit"):
        build_candidate_split_registry(
            [datasets[0]],
            policy_path=policy_path,
            spatial_buffer_m=0.0,
            registry_path=tmp_path / "registry.json",
            report_path=tmp_path / "report.json",
            max_dataset_bytes=first_path.stat().st_size - 1,
        )


def test_candidate_split_registry_bounds_npz_expansion_before_loading(tmp_path):
    datasets, policy_path = _fixture(tmp_path)
    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("oversized.npy", bytes(8192))
    payload = archive_bytes.getvalue()
    assert len(payload) < 4096
    datasets[0][1].write_bytes(payload)

    with pytest.raises(ValueError, match="uncompressed limit"):
        build_candidate_split_registry(
            [datasets[0]],
            policy_path=policy_path,
            spatial_buffer_m=0.0,
            registry_path=tmp_path / "registry.json",
            report_path=tmp_path / "report.json",
            max_dataset_bytes=4096,
        )
