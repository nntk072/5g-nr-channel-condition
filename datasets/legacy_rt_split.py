"""Migrate legacy RT channel metadata to canonical split identities in memory."""

from __future__ import annotations

import hashlib
import json
import math
import re
from numbers import Real
from pathlib import PurePosixPath
from typing import Any

from datasets.channel import ChannelDataset
from datasets.rt_split_identity import rt_spatial_split_subject_id

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SPLIT_FIELDS = frozenset(
    {
        "split_subject_id",
        "split_parent_type",
        "split_source_family",
        "split_leakage_relations",
        "split_identity_basis",
        "split_identity_complete",
        "split_block_size_m",
        "split_coordinate_frame_id",
        "split_positions_m",
    }
)


def _legacy_rt_scene_identity(data: ChannelDataset, *, dataset_name: str) -> tuple[float, str]:
    if any(str(source) != "sionna_rt" for source in data.source):
        raise ValueError(f"{dataset_name} legacy migration requires only sionna_rt rows")
    scene_config = data.generation_report.get("scene_config")
    asset_manifest = data.generation_report.get("scene_asset_manifest")
    if not isinstance(scene_config, dict) or not isinstance(asset_manifest, dict):
        raise ValueError(f"{dataset_name} lacks saved scene configuration or asset closure")
    block_size = scene_config.get("block_size_m")
    if (
        isinstance(block_size, bool)
        or not isinstance(block_size, Real)
        or not math.isfinite(float(block_size))
        or block_size <= 0
    ):
        raise ValueError(f"{dataset_name} has an invalid scene split block size")

    asset_digest = asset_manifest.get("asset_set_sha256")
    asset_count = asset_manifest.get("asset_count")
    assets = asset_manifest.get("assets")
    if (
        asset_manifest.get("complete") is not True
        or not isinstance(asset_digest, str)
        or _SHA256_RE.fullmatch(asset_digest) is None
        or asset_manifest.get("unresolved_references") != []
        or isinstance(asset_count, bool)
        or not isinstance(asset_count, int)
        or not isinstance(assets, list)
        or asset_count != len(assets)
        or asset_count < 1
    ):
        raise ValueError(f"{dataset_name} lacks a complete, internally consistent scene closure")

    seen_paths: set[str] = set()
    for asset in assets:
        if not isinstance(asset, dict) or set(asset) != {"path", "bytes", "sha256"}:
            raise ValueError(f"{dataset_name} has a malformed saved scene asset")
        path, size, digest = asset["path"], asset["bytes"], asset["sha256"]
        if (
            not isinstance(path, str)
            or not path
            or "\\" in path
            or PurePosixPath(path).is_absolute()
            or PurePosixPath(path).as_posix() != path
            or any(part in {"", ".", ".."} for part in path.split("/"))
            or path in seen_paths
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 1
            or not isinstance(digest, str)
            or _SHA256_RE.fullmatch(digest) is None
        ):
            raise ValueError(f"{dataset_name} has an invalid saved scene asset entry")
        seen_paths.add(path)
    canonical_assets = json.dumps(
        sorted(assets, key=lambda item: item["path"]),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if hashlib.sha256(canonical_assets).hexdigest() != asset_digest:
        raise ValueError(f"{dataset_name} scene asset digest disagrees with its saved manifest")
    return float(block_size), asset_digest


def _upgrade_legacy_rt_row(
    original: dict[str, Any], *, row_index: int, block_size: float, asset_digest: str
) -> dict[str, Any]:
    context = f"legacy RT row {row_index}"
    metadata = dict(original)
    row_scene = metadata.get("scene")
    row_scene_config = metadata.get("scene_config")
    receiver = metadata.get("rx")
    if (
        not isinstance(row_scene, str)
        or not row_scene.strip()
        or not isinstance(row_scene_config, dict)
        or row_scene_config.get("name") != row_scene
        or row_scene_config.get("block_size_m") != block_size
        or metadata.get("scene_asset_set_sha256") != asset_digest
        or metadata.get("scene_asset_manifest_complete") is not True
        or not isinstance(receiver, (list, tuple))
        or len(receiver) != 3
    ):
        raise ValueError(f"{context} disagrees with the saved complete RT scene identity")
    subject_id, identity_complete = rt_spatial_split_subject_id(
        scene_asset_set_sha256=asset_digest,
        scene_asset_manifest_complete=True,
        scene_name=row_scene,
        rx_position_m=receiver,
        block_size_m=block_size,
    )
    if not identity_complete:
        raise ValueError(f"{context} did not produce a complete RT split identity")
    metadata.update(
        {
            "split_subject_id": subject_id,
            "split_parent_type": "network_episode",
            "split_source_family": "sionna_rt",
            "split_leakage_relations": [],
            "split_identity_basis": "scene_asset_set_sha256+receiver_xy_grid",
            "split_identity_complete": True,
            "split_block_size_m": block_size,
            "split_coordinate_frame_id": asset_digest,
            "split_positions_m": [float(value) for value in receiver],
        }
    )
    return metadata


def upgrade_legacy_rt_split_metadata(
    data: ChannelDataset, *, dataset_name: str
) -> tuple[ChannelDataset, bool]:
    """Derive missing RT split fields from complete saved scene provenance.

    The original dataset is never changed. A new ChannelDataset is returned
    only when every row has legacy RT metadata and can be upgraded safely.

    Args:
        data: Parsed legacy or current channel dataset.
        dataset_name: Name used to identify validation errors.

    Returns:
        The current or upgraded dataset, and whether any rows were upgraded.

    Raises:
        ValueError: Metadata is partial, inconsistent, or lacks complete RT
            scene identity required for canonical split-parent derivation.
    """
    field_presence = [_SPLIT_FIELDS.intersection(metadata) for metadata in data.metadata]
    if all(not present for present in field_presence):
        pass
    elif all(_SPLIT_FIELDS.issubset(present) for present in field_presence):
        return data, False
    else:
        raise ValueError(
            f"{dataset_name} has partial or mixed RT split metadata; refusing legacy migration"
        )

    block_size, asset_digest = _legacy_rt_scene_identity(data, dataset_name=dataset_name)
    upgraded = [
        _upgrade_legacy_rt_row(
            original,
            row_index=row_index,
            block_size=block_size,
            asset_digest=asset_digest,
        )
        for row_index, original in enumerate(data.metadata)
    ]

    return (
        ChannelDataset(
            h=data.h,
            y=data.y,
            group=data.group,
            domain=data.domain,
            source=data.source,
            metadata=tuple(upgraded),
            generation_report=data.generation_report,
            label_semantics=data.label_semantics,
        ),
        True,
    )
