from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from datasets.channel import ChannelDataset
from datasets.split_policy import (
    SplitChildBinding,
)

_INVENTORY_SIGNER = "umcd-inventory-reviewer"

_INVENTORY_NAMESPACE = "umcd-channel-dataset-inventory-v1@github.com"


def _rt_subject_id(scene_hash: str, x: float, *, block_size: float = 0.25) -> str:
    numerator, denominator = float(x).as_integer_ratio()
    size_numerator, size_denominator = float(block_size).as_integer_ratio()
    block_x = (numerator * size_denominator) // (denominator * size_numerator)
    return f"rt:{scene_hash}:block:{float(block_size).hex()}:{block_x}:0"


def _write_dataset(
    path, subject_id: str, x: float, scene_hash: str, *, complete_identity: bool = True
) -> None:
    scene_name = "fixture-scene" if complete_identity else f"provisional-{path.stem}"
    if complete_identity:
        subject_id = _rt_subject_id(scene_hash, x)
    else:
        from datasets.rt_split_identity import rt_spatial_split_subject_id

        subject_id, _ = rt_spatial_split_subject_id(
            scene_asset_set_sha256=scene_hash,
            scene_asset_manifest_complete=False,
            scene_name=scene_name,
            rx_position_m=(x, 0.0, 1.5),
            block_size_m=0.25,
        )
    ChannelDataset(
        h=np.ones((1, 1, 8), dtype=np.complex64),
        y=np.zeros(1, dtype=np.uint8),
        group=np.asarray([subject_id]),
        domain=np.asarray(["rt_scene"]),
        source=np.asarray(["sionna_rt"]),
        metadata=(
            {
                "scene": scene_name,
                "split_subject_id": subject_id,
                "split_parent_type": "network_episode",
                "split_source_family": "sionna_rt",
                "split_leakage_relations": [],
                "scene_asset_manifest_complete": complete_identity,
                "scene_asset_set_sha256": scene_hash,
                "split_identity_basis": (
                    "scene_asset_set_sha256+receiver_xy_grid"
                    if complete_identity
                    else "provisional_scene_name+receiver_xy_grid"
                ),
                "split_identity_complete": complete_identity,
                "split_block_size_m": 0.25,
                "split_coordinate_frame_id": scene_hash if complete_identity else None,
                "split_positions_m": [x, 0.0, 1.5] if complete_identity else None,
                "rx": [x, 0.0, 1.5],
            },
        ),
    ).save(path)


def _write_child_roster(path: Path, child_kind: str, bindings: list[SplitChildBinding]) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "umcd.channel_dataset_child_roster.v1",
                "child_kind": child_kind,
                "bindings": [
                    {
                        "child_id": binding.child_id,
                        "parent_subject_id": binding.parent_subject_id,
                        "child_kind": binding.child_kind,
                    }
                    for binding in bindings
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _write_legacy_rt_dataset(path, *, tx_name: str, rx_x: float, scene_hash: str) -> None:
    scene_config = {"name": "legacy-city43", "block_size_m": 20.0}
    assets = [{"path": "legacy-city43.xml", "bytes": 1, "sha256": scene_hash}]
    asset_set_sha256 = hashlib.sha256(
        json.dumps(assets, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    ChannelDataset(
        h=np.ones((1, 1, 8), dtype=np.complex64),
        y=np.zeros(1, dtype=np.uint8),
        group=np.asarray([f"legacy-{tx_name}-block-0"]),
        domain=np.asarray([f"rt_{tx_name}"]),
        source=np.asarray(["sionna_rt"]),
        metadata=(
            {
                "scene": "legacy-city43",
                "scene_config": {**scene_config, "transmitter_id": tx_name},
                "scene_asset_set_sha256": asset_set_sha256,
                "scene_asset_manifest_complete": True,
                "rx": [rx_x, 5.0, 1.5],
                "transmitter_id": tx_name,
            },
        ),
        generation_report={
            "scene_config": scene_config,
            "scene_asset_manifest": {
                "asset_set_sha256": asset_set_sha256,
                "asset_count": 1,
                "assets": assets,
                "complete": True,
                "unresolved_references": [],
            },
        },
        label_semantics="legacy_fixture",
    ).save(path)


def _write_unmanifested_legacy_rt_dataset(path, *, rx_x: float) -> tuple[str, dict[str, object]]:
    scene_config = {"name": "historical-city", "block_size_m": 20.0}
    source_revision = "1" * 40
    package_source_sha256 = "2" * 64
    ChannelDataset(
        h=np.ones((1, 1, 8), dtype=np.complex64),
        y=np.zeros(1, dtype=np.uint8),
        group=np.asarray(["legacy-block"]),
        domain=np.asarray(["rt_historical_city"]),
        source=np.asarray(["sionna_rt"]),
        metadata=(
            {
                "scene": "historical-city",
                "scene_config": scene_config,
                "rx": [rx_x, 5.0, 1.5],
            },
        ),
        generation_report={
            "environment": {"git_sha": source_revision, "sionna-rt": "2.2.0"},
            "package_source_sha256": package_source_sha256,
            "scene_config": scene_config,
        },
        label_semantics="legacy_fixture",
    ).save(path)
    assets = [{"path": "historical-city.xml", "bytes": 1, "sha256": "a" * 64}]
    asset_set_sha256 = hashlib.sha256(
        json.dumps(assets, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return source_revision, {
        "asset_set_sha256": asset_set_sha256,
        "asset_count": 1,
        "assets": assets,
        "complete": True,
        "unresolved_references": [],
    }


def _write_reconstructed_asset_manifest(path, dataset_path, *, source_revision, assets) -> None:
    payload = dataset_path.read_bytes()
    data = ChannelDataset.load_bytes(payload)
    report = data.generation_report
    entry = {
        "name": "historical",
        "input_sha256": hashlib.sha256(payload).hexdigest(),
        "input_bytes": len(payload),
        "rows": len(data.y),
        "source_revision": source_revision,
        "package_source_sha256": report["package_source_sha256"],
        "scene": "historical-city",
        "scene_config": report["scene_config"],
        "scene_asset_manifest": assets,
    }
    path.write_text(
        json.dumps(
            {
                "schema": "umcd.legacy_rt_scene_asset_reconstruction.v1",
                "status": "NOT_QUALIFIED",
                "reconstruction_method": "historical_git_tree_and_lockfile_pinned_sionna_rt_wheel",
                "reconstruction_tool_source_sha256": "5" * 64,
                "asset_manifest_builder_source_sha256": "6" * 64,
                "source_revision": source_revision,
                "package_source_sha256": report["package_source_sha256"],
                "uv_lock_sha256": "3" * 64,
                "sionna_rt_version": "2.2.0",
                "sionna_rt_wheel_sha256": "4" * 64,
                "datasets": [entry],
                "limitations": ["fixture reconstruction is not externally attested"],
            }
        ),
        encoding="utf-8",
    )


def _write_inventory(path, dataset_path, name: str = "pilot") -> None:
    payload = dataset_path.read_bytes()
    path.write_text(
        json.dumps(
            {
                "schema": "umcd.channel_dataset_inventory.v1",
                "frozen": True,
                "inventory_id": "fixture-v1",
                "datasets": [
                    {
                        "name": name,
                        "path": dataset_path.name,
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "bytes": len(payload),
                        "rows": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _sign_inventory(inventory_path, tmp_path):
    ssh_keygen = shutil.which("ssh-keygen")
    if ssh_keygen is None:
        pytest.skip("OpenSSH is unavailable")
    private_key = tmp_path / "inventory-review-key"
    subprocess.run(
        [ssh_keygen, "-q", "-t", "ed25519", "-N", "", "-f", str(private_key)],
        check=True,
        capture_output=True,
        text=True,
    )
    public_key = private_key.with_suffix(".pub").read_text(encoding="utf-8").strip()
    allowed_signers = tmp_path / "allowed_signers"
    allowed_signers.write_text(
        f'{_INVENTORY_SIGNER} namespaces="{_INVENTORY_NAMESPACE}" {public_key}\n',
        encoding="utf-8",
    )
    subprocess.run(
        [
            ssh_keygen,
            "-Y",
            "sign",
            "-n",
            _INVENTORY_NAMESPACE,
            "-f",
            str(private_key),
            str(inventory_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(f"{inventory_path}.sig"), allowed_signers
