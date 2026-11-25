"""Legacy RT split migration: refuses non-RT rows, partial split metadata and malformed saved scene identity."""

from __future__ import annotations

import hashlib
import json

import datasets.legacy_rt_split as migration
import numpy as np
import pytest
from datasets.channel import ChannelDataset
from datasets.legacy_rt_split import upgrade_legacy_rt_split_metadata


def _dataset(source="sionna_rt", metadata=None, generation_report=None) -> ChannelDataset:
    return ChannelDataset(
        h=np.ones((1, 1, 8), dtype=np.complex64),
        y=np.asarray([0], dtype=np.uint8),
        group=np.asarray(["g"]),
        domain=np.asarray(["scene"]),
        source=np.asarray([source]),
        metadata=(
            metadata if metadata is not None else {"scene": "scene-a", "rx": [0.0, 0.0, 1.5]},
        ),
        generation_report=generation_report or {},
    )


def test_only_sionna_rt_rows_can_be_migrated() -> None:
    with pytest.raises(ValueError, match="requires only sionna_rt rows"):
        upgrade_legacy_rt_split_metadata(_dataset(source="standardized"), dataset_name="legacy")


def test_saved_scene_configuration_is_required() -> None:
    with pytest.raises(ValueError, match="lacks saved scene configuration or asset closure"):
        upgrade_legacy_rt_split_metadata(_dataset(generation_report={}), dataset_name="legacy")


def test_partial_split_metadata_is_refused() -> None:
    metadata = {
        "scene": "scene-a",
        "rx": [0.0, 0.0, 1.5],
        "split_subject_id": "rt:partial",
    }
    with pytest.raises(ValueError, match="partial or mixed RT split metadata"):
        upgrade_legacy_rt_split_metadata(_dataset(metadata=metadata), dataset_name="legacy")


def test_complete_saved_scene_identity_is_migrated_without_mutating_source() -> None:
    source, asset_digest = _valid_saved_scene()

    upgraded, changed = upgrade_legacy_rt_split_metadata(source, dataset_name="legacy")

    assert changed is True
    assert upgraded is not source
    assert source.metadata[0].get("split_subject_id") is None
    migrated = upgraded.metadata[0]
    assert migrated["split_identity_complete"] is True
    assert migrated["split_source_family"] == "sionna_rt"
    assert migrated["split_coordinate_frame_id"] == asset_digest
    assert migrated["split_positions_m"] == [0.0, 0.0, 1.5]


def _valid_saved_scene() -> tuple[ChannelDataset, str]:
    asset = {"path": "scene/scene.xml", "bytes": 12, "sha256": "a" * 64}
    canonical_assets = json.dumps([asset], sort_keys=True, separators=(",", ":")).encode()
    asset_digest = hashlib.sha256(canonical_assets).hexdigest()
    data = _dataset(
        metadata={
            "scene": "scene-a",
            "scene_config": {"name": "scene-a", "block_size_m": 4.0},
            "scene_asset_set_sha256": asset_digest,
            "scene_asset_manifest_complete": True,
            "rx": [0.0, 0.0, 1.5],
        },
        generation_report={
            "scene_config": {"block_size_m": 4.0},
            "scene_asset_manifest": {
                "complete": True,
                "asset_set_sha256": asset_digest,
                "asset_count": 1,
                "assets": [asset],
                "unresolved_references": [],
            },
        },
    )
    return data, asset_digest


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda report, _meta: report["scene_config"].update(block_size_m=True),
            "invalid scene split block size",
        ),
        (
            lambda report, _meta: report["scene_asset_manifest"].update(complete=False),
            "complete, internally consistent scene closure",
        ),
        (
            lambda report, _meta: report["scene_asset_manifest"]["assets"][0].update(extra="field"),
            "malformed saved scene asset",
        ),
        (
            lambda report, _meta: report["scene_asset_manifest"]["assets"][0].update(
                path="../scene.xml"
            ),
            "invalid saved scene asset entry",
        ),
        (
            lambda report, _meta: report["scene_asset_manifest"].update(asset_set_sha256="b" * 64),
            "digest disagrees",
        ),
        (
            lambda _report, meta: meta.update(scene_asset_set_sha256="b" * 64),
            "disagrees with the saved complete RT scene identity",
        ),
    ],
)
def test_legacy_migration_rejects_malformed_saved_identity(mutate, message):
    data, _digest = _valid_saved_scene()
    report = {
        **data.generation_report,
        "scene_config": dict(data.generation_report["scene_config"]),
    }
    manifest = dict(report["scene_asset_manifest"])
    manifest["assets"] = [dict(asset) for asset in manifest["assets"]]
    report["scene_asset_manifest"] = manifest
    metadata = dict(data.metadata[0])
    mutate(report, metadata)
    malformed = ChannelDataset(
        h=data.h,
        y=data.y,
        group=data.group,
        domain=data.domain,
        source=data.source,
        metadata=(metadata,),
        generation_report=report,
        label_semantics=data.label_semantics,
    )

    with pytest.raises(ValueError, match=message):
        upgrade_legacy_rt_split_metadata(malformed, dataset_name="legacy")


def test_complete_split_metadata_is_already_current() -> None:
    data, _digest = _valid_saved_scene()
    metadata = dict(data.metadata[0])
    metadata.update({field: "current" for field in migration._SPLIT_FIELDS})
    current = ChannelDataset(
        h=data.h,
        y=data.y,
        group=data.group,
        domain=data.domain,
        source=data.source,
        metadata=(metadata,),
        generation_report=data.generation_report,
        label_semantics=data.label_semantics,
    )

    unchanged, changed = upgrade_legacy_rt_split_metadata(current, dataset_name="current")

    assert unchanged is current
    assert changed is False


def test_identity_derivation_must_confirm_completeness(monkeypatch) -> None:
    data, _digest = _valid_saved_scene()
    monkeypatch.setattr(
        migration, "rt_spatial_split_subject_id", lambda **_kwargs: ("subject", False)
    )

    with pytest.raises(ValueError, match="did not produce a complete RT split identity"):
        upgrade_legacy_rt_split_metadata(data, dataset_name="legacy")
