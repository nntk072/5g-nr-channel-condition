"""Split lineage: scene-asset manifests in generation reports are fully verified before use."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

import pytest
from datasets.channel import ChannelDataset
from datasets.split_lineage import derive_channel_dataset_split_evidence
from tests.test_datasets.test_split_lineage_audit import _dataset


def _asset(
    path: str = "meshes/wall.obj", byte_count: int = 10, digest: str = "a" * 64
) -> dict[str, Any]:
    return {"path": path, "bytes": byte_count, "sha256": digest}


def _manifest(assets: list[dict[str, Any]] | None = None, **changes: Any) -> dict[str, Any]:
    assets = [_asset()] if assets is None else assets
    canonical = json.dumps(
        [{"path": a["path"], "bytes": a["bytes"], "sha256": a["sha256"]} for a in assets],
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    manifest = {
        "complete": True,
        "asset_set_sha256": hashlib.sha256(canonical).hexdigest(),
        "assets": assets,
    }
    manifest.update(changes)
    return manifest


def _with_report(report: dict[str, Any]) -> ChannelDataset:
    _, base = _dataset("run", [("parent-a", 0.0, "a" * 64)])
    return ChannelDataset(
        h=base.h,
        y=base.y,
        group=base.group,
        domain=base.domain,
        source=base.source,
        metadata=base.metadata,
        generation_report=report,
    )


def _derive(report: dict[str, Any]) -> Any:
    return derive_channel_dataset_split_evidence(
        (("run", _with_report(copy.deepcopy(report))),), spatial_buffer_m=1.0
    )


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        pytest.param("not-an-object", "scene_asset_manifest must be an object", id="type"),
        pytest.param(_manifest(complete="yes"), "invalid completeness flag", id="complete-type"),
        pytest.param(_manifest(asset_set_sha256="G" * 64), "invalid set digest", id="digest"),
        pytest.param(_manifest(assets=[]), "must contain assets", id="assets-empty"),
    ],
)
def test_scene_manifest_header_is_validated(manifest, message) -> None:
    with pytest.raises(ValueError, match=message):
        _derive({"scene_asset_manifest": manifest})


@pytest.mark.parametrize(
    ("asset", "message"),
    [
        pytest.param(
            {"path": "x", "bytes": 1, "sha256": "a" * 64, "extra": 1},
            "invalid field roster",
            id="roster",
        ),
        pytest.param(_asset(path="../escape.obj"), "noncanonical relative path", id="escape"),
        pytest.param(_asset(path="/absolute.obj"), "noncanonical relative path", id="absolute"),
        pytest.param(_asset(byte_count=-1), "invalid byte count", id="bytes"),
        pytest.param(_asset(digest="A" * 64), "invalid SHA-256", id="digest-case"),
    ],
)
def test_scene_asset_entries_are_validated(asset, message) -> None:
    with pytest.raises(ValueError, match=message):
        _derive({"scene_asset_manifest": _manifest([asset])})


def test_scene_asset_paths_and_digest_must_be_consistent() -> None:
    duplicate = [_asset(), _asset()]
    with pytest.raises(ValueError, match="paths must be unique"):
        _derive({"scene_asset_manifest": _manifest(duplicate)})
    manifest = _manifest()
    manifest["assets"][0]["bytes"] = 11
    with pytest.raises(ValueError, match="digest does not match its entries"):
        _derive({"scene_asset_manifest": manifest})


def test_generation_report_structure_is_checked() -> None:
    with pytest.raises(TypeError, match="generation_report must be an object"):
        _with_report_check("not-a-dict")
    with pytest.raises(ValueError, match="parent_generation_report must be an object"):
        _derive({"parent_generation_report": "x"})
    with pytest.raises(ValueError, match="components must be an array"):
        _derive({"components": "x"})
    with pytest.raises(ValueError, match="components must be objects"):
        _derive({"components": ["x"]})


def _with_report_check(report: Any) -> None:
    from datasets.split_lineage import _scene_geometry_asset_manifests

    _scene_geometry_asset_manifests(report, context="test")
