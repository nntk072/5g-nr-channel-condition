from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from datasets.channel import ChannelDataset
from datasets.episode_store import EpisodeBundle, write_episode_bundle
from datasets.network import (
    AntennaArraySpec,
    DirectedLink,
    LinkDirection,
    LinkRole,
    NetworkEpisode,
    NetworkNode,
    NodeRole,
)
from datasets.nr import NRChannelRecord, NRObservation, NRTrajectory
from datasets.nr_store import write_nr_trajectory
from datasets.path_oracle import PathOracleTable, PathSolveStatus
from datasets.split_lineage import make_channel_dataset_split_registry
from datasets.split_policy import (
    SPLIT_POLICY_SIGNATURE_NAMESPACE,
    LeakageNamespace,
    SplitChildBinding,
    SplitPolicy,
    VerifiedOpenSSHArtifact,
    bind_split_children,
    freeze_split_policy_registry,
)
from datasets.targets import TargetDefinition
from receivers.nr import NRChannelEstimate, NRPUSCHConfig
from scripts.audit_channel_split_lineage import (
    _CHILD_ARTIFACT_ROSTER_SIGNATURE_NAMESPACE,
    _load_split_child_artifact_roster,
    _parse_split_child_roster,
    _read_channel_dataset_child_rows,
    _read_stable_regular_file,
    _require_exact_child_object_owners,
    audit_channel_datasets,
    main,
)


def test_content_addressed_child_object_owners_must_match_frozen_roster() -> None:
    _require_exact_child_object_owners(
        "objects/oracle_paths/aa/object.npz",
        ["child-1", "child-2"],
        [
            ("child-1", "objects/oracle_paths/aa/object.npz"),
            ("child-2", "objects/oracle_paths/aa/object.npz"),
        ],
        label="episode content object",
    )
    with pytest.raises(ValueError, match="manifest owners differ"):
        _require_exact_child_object_owners(
            "objects/oracle_paths/aa/object.npz",
            ["child-1", "child-2"],
            [("child-1", "objects/oracle_paths/aa/object.npz")],
            label="episode content object",
        )
    with pytest.raises(ValueError, match="manifest owners differ"):
        _require_exact_child_object_owners(
            "objects/oracle_paths/aa/object.npz",
            ["child-1"],
            [("child-2", "objects/oracle_paths/aa/object.npz")],
            label="episode content object",
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


def test_signed_split_registry_audit_rejects_provisional_rt_scene_identity(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from datasets.split_lineage import make_channel_dataset_split_registry
    from scripts import audit_channel_split_lineage as audit_module

    paths = []
    for index in range(4):
        path = tmp_path / f"provisional-{index}.npz"
        _write_dataset(
            path,
            f"parent-{index}",
            float(index),
            f"{index + 1:064x}",
            complete_identity=False,
        )
        paths.append(path)
    registry, _ = make_channel_dataset_split_registry(
        [(path.stem, ChannelDataset.load(path)) for path in paths],
        spatial_buffer_m=0.0,
        policy=SplitPolicy.release_v2(seed="provisional-signed-lock"),
    )
    monkeypatch.setattr(
        audit_module,
        "verify_signed_split_policy_registry_attestation",
        lambda *_args, **_kwargs: (registry, object()),
    )

    with pytest.raises(ValueError, match="refuses provisional RT scene identities"):
        audit_channel_datasets(
            [(path.stem, path) for path in paths],
            spatial_buffer_m=0.0,
            split_registry_path=tmp_path / "signed-registry.json",
            split_registry_signature=tmp_path / "signed-registry.json.sig",
            trusted_split_allowed_signers=tmp_path / "allowed-signers",
            split_registry_signer_id="reviewer@example.org",
        )


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


def test_split_child_roster_cli_parser_requires_kind_and_path(tmp_path) -> None:
    parsed = _parse_split_child_roster(f"receiver_observation={tmp_path / 'roster.json'}")
    assert parsed == ("receiver_observation", tmp_path / "roster.json")
    for malformed in (
        "=roster.json",
        "receiver_observation=",
        "receiver_observation",
        " receiver_observation=roster.json",
        "receiver_observation =roster.json",
    ):
        with pytest.raises(argparse.ArgumentTypeError, match="KIND=PATH|canonical"):
            _parse_split_child_roster(malformed)


@pytest.mark.parametrize(
    "roster_argument", ["split_child_rosters", "split_child_artifact_rosters"]
)
def test_split_child_roster_api_rejects_padded_kind_before_reading_inputs(
    roster_argument: str, tmp_path: Path
) -> None:
    dataset_path = tmp_path / "dataset.npz"
    _write_dataset(dataset_path, "parent", 0.0, "a" * 64)
    kwargs = {roster_argument: ((" receiver_observation ", tmp_path / "unused.json"),)}
    if roster_argument == "split_child_artifact_rosters":
        kwargs["split_child_rosters"] = (
            ("receiver_observation", tmp_path / "unused-bindings.json"),
        )
    with pytest.raises(ValueError, match="kinds must be canonical nonempty strings"):
        audit_channel_datasets(
            [("dataset", dataset_path)],
            spatial_buffer_m=0.0,
            split_registry_path=tmp_path / "unused-registry.json",
            split_registry_signature=tmp_path / "unused-registry.json.sig",
            trusted_split_allowed_signers=tmp_path / "unused-allowed-signers",
            split_registry_signer_id="reviewer@example.org",
            **kwargs,
        )


def test_child_artifact_roster_schema_and_cli_contract(tmp_path) -> None:
    import jsonschema

    schema_path = (
        Path(__file__).parents[2]
        / "datasets/schema/channel-dataset-child-artifact-roster.v1.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    valid = {
        "schema": "umcd.channel_dataset_child_artifact_roster.v1",
        "child_kind": "receiver_observation",
        "artifacts": [
            {"child_id": "child.001", "path": "children/001.npz", "bytes": 3, "sha256": "a" * 64}
        ],
    }
    assert list(jsonschema.Draft202012Validator(schema).iter_errors(valid)) == []
    invalid = dict(valid, artifacts=[dict(valid["artifacts"][0], path="../outside.bin")])
    assert list(jsonschema.Draft202012Validator(schema).iter_errors(invalid))
    v2_schema_path = (
        Path(__file__).parents[2]
        / "datasets/schema/channel-dataset-child-artifact-roster.v2.schema.json"
    )
    v2_schema = json.loads(v2_schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(v2_schema)
    v2_valid = {
        "schema": "umcd.channel_dataset_child_artifact_roster.v2",
        "child_kind": "receiver_observation",
        "artifacts": [
            {
                "artifact_id": "child.001.dataset",
                "artifact_type": "channel_dataset_npz",
                "child_id": "child.001",
                "path": "children/001.npz",
                "bytes": 3,
                "sha256": "a" * 64,
            },
            {
                "artifact_id": "child.001.trajectory",
                "artifact_type": "trajectory_manifest_json",
                "child_id": "child.001",
                "path": "children/001-trajectory.json",
                "bytes": 4,
                "sha256": "b" * 64,
            },
        ],
    }
    assert list(jsonschema.Draft202012Validator(v2_schema).iter_errors(v2_valid)) == []
    invalid_v2 = dict(v2_valid, artifacts=[dict(v2_valid["artifacts"][0], artifact_type="Bad Type")])
    assert list(jsonschema.Draft202012Validator(v2_schema).iter_errors(invalid_v2))
    v3_schema_path = (
        Path(__file__).parents[2]
        / "datasets/schema/channel-dataset-child-artifact-roster.v3.schema.json"
    )
    v3_schema = json.loads(v3_schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(v3_schema)
    v3_valid = {
        "schema": "umcd.channel_dataset_child_artifact_roster.v3",
        "child_kind": "receiver_observation",
        "artifact_root": "children",
        "artifacts": [
            {
                **v2_valid["artifacts"][0],
                "path": "001.npz",
            }
        ],
    }
    assert list(jsonschema.Draft202012Validator(v3_schema).iter_errors(v3_valid)) == []
    invalid_v3 = dict(v3_valid, artifact_root="../outside")
    assert list(jsonschema.Draft202012Validator(v3_schema).iter_errors(invalid_v3))
    invalid_v3_type = dict(
        v3_valid,
        artifacts=[dict(v3_valid["artifacts"][0], artifact_type="target_sidecar_json")],
    )
    assert list(jsonschema.Draft202012Validator(v3_schema).iter_errors(invalid_v3_type))
    v4_schema_path = (
        Path(__file__).parents[2]
        / "datasets/schema/channel-dataset-child-artifact-roster.v4.schema.json"
    )
    v4_schema = json.loads(v4_schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(v4_schema)
    v4_valid = {
        **v3_valid,
        "schema": "umcd.channel_dataset_child_artifact_roster.v4",
        "artifacts": [
            dict(
                v3_valid["artifacts"][0],
                artifact_type="umcd_episode_store_manifest_json",
                path="manifests/episodes/example.json",
            )
        ],
    }
    assert list(jsonschema.Draft202012Validator(v4_schema).iter_errors(v4_valid)) == []
    invalid_v4_type = dict(
        v4_valid,
        artifacts=[dict(v4_valid["artifacts"][0], artifact_type="opaque_target_blob")],
    )
    assert list(jsonschema.Draft202012Validator(v4_schema).iter_errors(invalid_v4_type))
    v5_schema_path = (
        Path(__file__).parents[2]
        / "datasets/schema/channel-dataset-child-artifact-roster.v5.schema.json"
    )
    v5_schema = json.loads(v5_schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(v5_schema)
    v5_valid = {
        **v3_valid,
        "schema": "umcd.channel_dataset_child_artifact_roster.v5",
        "artifacts": [
            {
                "artifact_id": "shared-oracle-paths",
                "artifact_type": "umcd_episode_store_content_npz",
                "child_ids": ["child-1", "child-2"],
                "path": "objects/oracle_paths/aa/" + "a" * 64 + ".npz",
                "bytes": 1,
                "sha256": "a" * 64,
            }
        ],
    }
    assert list(jsonschema.Draft202012Validator(v5_schema).iter_errors(v5_valid)) == []
    invalid_v5 = {
        **v5_valid,
        "artifacts": [dict(v5_valid["artifacts"][0], child_id="child-1")],
    }
    assert list(jsonschema.Draft202012Validator(v5_schema).iter_errors(invalid_v5))
    assert _parse_split_child_roster(f"receiver_observation={tmp_path / 'artifacts.json'}") == (
        "receiver_observation",
        tmp_path / "artifacts.json",
    )


def test_child_artifact_roster_v2_binds_multiple_typed_artifacts_per_child(tmp_path) -> None:
    def write_dataset(path: Path, child_id: str, parent_id: str) -> None:
        ChannelDataset(
            h=np.ones((1, 1, 8), dtype=np.complex64),
            y=np.zeros(1, dtype=np.uint8),
            group=np.asarray([parent_id]),
            domain=np.asarray(["fixture"]),
            source=np.asarray(["synthetic"]),
            metadata=(
                {
                    "split_subject_id": parent_id,
                    "split_child_id": child_id,
                    "split_child_kind": "receiver_observation",
                },
            ),
        ).save(path)

    write_dataset(tmp_path / "child-1.npz", "child-1", "parent-1")
    write_dataset(tmp_path / "child-2.npz", "child-2", "parent-2")
    supplemental = tmp_path / "child-1-trajectory.json"
    supplemental.write_bytes(b"{\"episode\":\"episode-1\"}")
    specifications = [
        ("artifact-1-dataset", "channel_dataset_npz", "child-1", "child-1.npz"),
        (
            "artifact-1-trajectory",
            "trajectory_manifest_json",
            "child-1",
            "child-1-trajectory.json",
        ),
        ("artifact-2-dataset", "channel_dataset_npz", "child-2", "child-2.npz"),
    ]
    entries = []
    for artifact_id, artifact_type, child_id, filename in specifications:
        artifact_bytes = (tmp_path / filename).read_bytes()
        entries.append(
            {
                "artifact_id": artifact_id,
                "artifact_type": artifact_type,
                "child_id": child_id,
                "path": filename,
                "bytes": len(artifact_bytes),
                "sha256": hashlib.sha256(artifact_bytes).hexdigest(),
            }
        )
    payload = json.dumps(
        {
            "schema": "umcd.channel_dataset_child_artifact_roster.v2",
            "child_kind": "receiver_observation",
            "artifacts": entries,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    verified = VerifiedOpenSSHArtifact(
        payload=payload,
        artifact_sha256=hashlib.sha256(payload).hexdigest(),
        signature_sha256="a" * 64,
        allowed_signers_sha256="b" * 64,
        signer_id="test-signer",
        namespace=_CHILD_ARTIFACT_ROSTER_SIGNATURE_NAMESPACE,
    )
    bindings = {
        "child-1": SplitChildBinding("child-1", "parent-1", "receiver_observation"),
        "child-2": SplitChildBinding("child-2", "parent-2", "receiver_observation"),
    }

    report = _load_split_child_artifact_roster(
        tmp_path / "roster.json",
        expected_child_kind="receiver_observation",
        expected_child_bindings=bindings,
        verified_artifact=verified,
    )

    audit = report["audit"]
    assert audit["schema"] == "umcd.split_child_artifact_audit.v4"
    assert audit["child_count"] == 2
    assert audit["artifact_count"] == 3
    assert audit["semantic_artifact_count"] == 2
    assert audit["byte_only_artifact_count"] == 1
    assert audit["row_count"] == 2
    assert audit["status"] == "PASS_EXACT_FROZEN_CHILD_ARTIFACT_BYTES_WITH_PARTIAL_SEMANTICS"
    assert "unrecognized supplemental types receive byte-integrity checks only" in audit[
        "authority"
    ]


def test_child_artifact_roster_v2_requires_channel_dataset_for_each_child(tmp_path) -> None:
    artifact = tmp_path / "supplement.json"
    artifact.write_bytes(b"{}")
    payload = json.dumps(
        {
            "schema": "umcd.channel_dataset_child_artifact_roster.v2",
            "child_kind": "receiver_observation",
            "artifacts": [
                {
                    "artifact_id": "sidecar-1",
                    "artifact_type": "target_sidecar_json",
                    "child_id": "child-1",
                    "path": "supplement.json",
                    "bytes": 2,
                    "sha256": hashlib.sha256(b"{}").hexdigest(),
                }
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    verified = VerifiedOpenSSHArtifact(
        payload=payload,
        artifact_sha256=hashlib.sha256(payload).hexdigest(),
        signature_sha256="a" * 64,
        allowed_signers_sha256="b" * 64,
        signer_id="test-signer",
        namespace=_CHILD_ARTIFACT_ROSTER_SIGNATURE_NAMESPACE,
    )

    with pytest.raises(ValueError, match="must include a ChannelDataset NPZ"):
        _load_split_child_artifact_roster(
            tmp_path / "roster.json",
            expected_child_kind="receiver_observation",
            expected_child_bindings={
                "child-1": SplitChildBinding("child-1", "parent-1", "receiver_observation")
            },
            verified_artifact=verified,
        )


@pytest.mark.parametrize(
    ("extra_entry", "message"),
    [
        ("file", "does not exactly cover its artifact_root"),
        ("symlink", "only directories and single-link regular files"),
        ("hardlink", "only directories and single-link regular files"),
        ("fifo", "only directories and single-link regular files"),
        ("depth", "directory-depth limit"),
    ],
)
def test_child_artifact_roster_v3_rejects_unlisted_or_unsafe_tree_entries(
    tmp_path, extra_entry: str, message: str
) -> None:
    artifact_root = tmp_path / "children"
    artifact_root.mkdir()
    dataset_path = artifact_root / "child-1.npz"
    ChannelDataset(
        h=np.ones((1, 1, 8), dtype=np.complex64),
        y=np.zeros(1, dtype=np.uint8),
        group=np.asarray(["parent-1"]),
        domain=np.asarray(["fixture"]),
        source=np.asarray(["synthetic"]),
        metadata=(
            {
                "split_subject_id": "parent-1",
                "split_child_id": "child-1",
                "split_child_kind": "receiver_observation",
            },
        ),
    ).save(dataset_path)
    data = dataset_path.read_bytes()
    roster = {
        "schema": "umcd.channel_dataset_child_artifact_roster.v3",
        "child_kind": "receiver_observation",
        "artifact_root": "children",
        "artifacts": [
            {
                "artifact_id": "child-1.dataset",
                "artifact_type": "channel_dataset_npz",
                "child_id": "child-1",
                "path": "child-1.npz",
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        ],
    }
    payload = json.dumps(roster, sort_keys=True, separators=(",", ":")).encode("utf-8")
    verified = VerifiedOpenSSHArtifact(
        payload=payload,
        artifact_sha256=hashlib.sha256(payload).hexdigest(),
        signature_sha256="a" * 64,
        allowed_signers_sha256="b" * 64,
        signer_id="test-signer",
        namespace=_CHILD_ARTIFACT_ROSTER_SIGNATURE_NAMESPACE,
    )
    bindings = {
        "child-1": SplitChildBinding("child-1", "parent-1", "receiver_observation")
    }

    report = _load_split_child_artifact_roster(
        tmp_path / "roster.json",
        expected_child_kind="receiver_observation",
        expected_child_bindings=bindings,
        verified_artifact=verified,
    )
    assert report["audit"]["artifact_directory_inventory_status"] == "EXACT"
    assert report["audit"]["artifact_root"] == "children"

    if extra_entry == "file":
        (artifact_root / "unlisted.json").write_text("{}", encoding="utf-8")
    elif extra_entry == "symlink":
        (artifact_root / "unlisted-link").symlink_to(dataset_path.name)
    elif extra_entry == "hardlink":
        os.link(dataset_path, artifact_root / "unlisted-hardlink")
    elif extra_entry == "fifo":
        os.mkfifo(artifact_root / "unlisted-fifo")
    else:
        nested = artifact_root
        for index in range(33):
            nested = nested / f"d{index}"
            nested.mkdir()
    with pytest.raises(ValueError, match=message):
        _load_split_child_artifact_roster(
            tmp_path / "roster.json",
            expected_child_kind="receiver_observation",
            expected_child_bindings=bindings,
            verified_artifact=verified,
        )


def test_child_artifact_roster_v3_rejects_artifact_without_semantic_validator(
    tmp_path,
) -> None:
    (tmp_path / "children").mkdir()
    roster = {
        "schema": "umcd.channel_dataset_child_artifact_roster.v3",
        "child_kind": "receiver_observation",
        "artifact_root": "children",
        "artifacts": [
            {
                "artifact_id": "dataset-1",
                "artifact_type": "channel_dataset_npz",
                "child_id": "child-1",
                "path": "child-1.npz",
                "bytes": 0,
                "sha256": "0" * 64,
            },
            {
                "artifact_id": "opaque-sidecar",
                "artifact_type": "target_sidecar_json",
                "child_id": "child-1",
                "path": "sidecar.json",
                "bytes": 0,
                "sha256": "0" * 64,
            },
        ],
    }
    payload = json.dumps(roster, sort_keys=True, separators=(",", ":")).encode("utf-8")
    verified = VerifiedOpenSSHArtifact(
        payload=payload,
        artifact_sha256=hashlib.sha256(payload).hexdigest(),
        signature_sha256="a" * 64,
        allowed_signers_sha256="b" * 64,
        signer_id="test-signer",
        namespace=_CHILD_ARTIFACT_ROSTER_SIGNATURE_NAMESPACE,
    )

    with pytest.raises(ValueError, match="requires a semantic validator"):
        _load_split_child_artifact_roster(
            tmp_path / "roster.json",
            expected_child_kind="receiver_observation",
            expected_child_bindings={
                "child-1": SplitChildBinding("child-1", "parent-1", "receiver_observation")
            },
            verified_artifact=verified,
        )


def test_child_artifact_roster_v4_semantically_binds_umcd_episode_store(tmp_path: Path) -> None:
    artifact_root = tmp_path / "episode-child-store"
    artifact_root.mkdir()
    gnb_array = AntennaArraySpec(
        array_id="gnb-array",
        element_positions_local_m=np.zeros((1, 3)),
        polarization_per_element=("V",),
        orientation_local_to_world=np.eye(3),
        port_ids=("gnb-port",),
        port_element_weights=np.ones((1, 1), dtype=np.complex128),
    )
    ue_array = AntennaArraySpec(
        array_id="ue-array",
        element_positions_local_m=np.zeros((1, 3)),
        polarization_per_element=("V",),
        orientation_local_to_world=np.eye(3),
        port_ids=("ue-port",),
        port_element_weights=np.ones((1, 1), dtype=np.complex128),
    )
    episode = NetworkEpisode(
        episode_id="parent-1",
        split_group_id="asset-group-1",
        family_id="fixture-scene",
        carrier_hz=3.5e9,
        arrays=(gnb_array, ue_array),
        nodes=(
            NetworkNode("gnb", NodeRole.GNB, "gnb-array", (0.0, 0.0, 5.0), 1),
            NetworkNode("ue", NodeRole.UE, "ue-array", (10.0, 0.0, 1.5), 1),
        ),
        links=(
            DirectedLink(
                "link-1",
                "gnb",
                "ue",
                (0,),
                (0,),
                LinkDirection.DOWNLINK,
                LinkRole.SERVING,
            ),
        ),
        frame_times_s=(0.0,),
        subcarrier_offsets_hz=np.asarray([0.0]),
    )
    path_table = PathOracleTable(
        link_ids=("link-1",),
        frame_indices=np.asarray([0]),
        rx_port_indices=np.asarray([0]),
        tx_port_indices=np.asarray([0]),
        path_offsets=np.asarray([0, 0]),
        solve_status=(PathSolveStatus.NO_PATH,),
        delay_s=np.asarray([], dtype=np.float64),
        complex_gain=np.asarray([], dtype=np.complex128),
        interaction_order=np.asarray([], dtype=np.int64),
    )
    stored = write_episode_bundle(artifact_root, EpisodeBundle(episode, path_table, (), ()))
    channel_path = artifact_root / "receiver.npz"
    ChannelDataset(
        h=np.ones((1, 1, 8), dtype=np.complex64),
        y=np.zeros(1, dtype=np.uint8),
        group=np.asarray(["parent-1"]),
        domain=np.asarray(["fixture"]),
        source=np.asarray(["synthetic"]),
        metadata=(
            {
                "split_subject_id": "parent-1",
                "split_child_id": "child-1",
                "split_child_kind": "episode_bundle",
            },
        ),
    ).save(channel_path)

    entries = []
    for path in sorted(item for item in artifact_root.rglob("*") if item.is_file()):
        relative_path = path.relative_to(artifact_root).as_posix()
        artifact_type = (
            "channel_dataset_npz"
            if path == channel_path
            else "umcd_episode_store_manifest_json"
            if relative_path == stored.relative_path
            else "umcd_episode_store_content_npz"
        )
        payload = path.read_bytes()
        entries.append(
            {
                "artifact_id": f"artifact-{len(entries)}",
                "artifact_type": artifact_type,
                "child_id": "child-1",
                "path": relative_path,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    roster_payload = json.dumps(
        {
            "schema": "umcd.channel_dataset_child_artifact_roster.v4",
            "child_kind": "episode_bundle",
            "artifact_root": artifact_root.name,
            "artifacts": entries,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    verified = VerifiedOpenSSHArtifact(
        payload=roster_payload,
        artifact_sha256=hashlib.sha256(roster_payload).hexdigest(),
        signature_sha256="a" * 64,
        allowed_signers_sha256="b" * 64,
        signer_id="test-signer",
        namespace=_CHILD_ARTIFACT_ROSTER_SIGNATURE_NAMESPACE,
    )
    bindings = {
        "child-1": SplitChildBinding("child-1", "parent-1", "episode_bundle")
    }

    audit = _load_split_child_artifact_roster(
        tmp_path / "roster.json",
        expected_child_kind="episode_bundle",
        expected_child_bindings=bindings,
        verified_artifact=verified,
    )["audit"]

    assert audit["schema"] == "umcd.split_child_artifact_audit.v5"
    assert audit["artifact_directory_inventory_status"] == "EXACT"
    assert audit["episode_count"] == 1
    assert audit["semantic_artifact_count"] == len(entries)
    assert audit["byte_only_artifact_count"] == 0
    assert "artifact types without semantic validators are rejected" in audit["authority"]
    assert "roster v4 also validates UMCD episode-store manifests" in audit["authority"]
    episode_artifact = next(
        row
        for row in audit["artifacts"]
        if row["artifact_type"] == "umcd_episode_store_manifest_json"
    )
    assert episode_artifact["semantic_validator"] == "umcd_episode_store_v1"
    assert episode_artifact["episode_id"] == "parent-1"
    v4_content_artifact = next(
        item
        for item in audit["artifacts"]
        if item["artifact_type"] == "umcd_episode_store_content_npz"
    )
    assert isinstance(v4_content_artifact["episode_manifest_path"], str)

    incorrect_bindings = {
        "child-1": SplitChildBinding("child-1", "other-parent", "episode_bundle")
    }
    with pytest.raises(ValueError, match="different split parent episode ID"):
        _load_split_child_artifact_roster(
            tmp_path / "roster.json",
            expected_child_kind="episode_bundle",
            expected_child_bindings=incorrect_bindings,
            verified_artifact=verified,
        )

    second_episode = replace(
        episode, episode_id="parent-2", split_group_id="asset-group-2"
    )
    second_stored = write_episode_bundle(
        artifact_root, EpisodeBundle(second_episode, path_table, (), ())
    )
    second_channel_path = artifact_root / "receiver-2.npz"
    ChannelDataset(
        h=np.ones((1, 1, 8), dtype=np.complex64),
        y=np.zeros(1, dtype=np.uint8),
        group=np.asarray(["parent-2"]),
        domain=np.asarray(["fixture"]),
        source=np.asarray(["synthetic"]),
        metadata=(
            {
                "split_subject_id": "parent-2",
                "split_child_id": "child-2",
                "split_child_kind": "episode_bundle",
            },
        ),
    ).save(second_channel_path)
    manifest_owners = {
        stored.relative_path: "child-1",
        second_stored.relative_path: "child-2",
    }
    object_owners: dict[str, set[str]] = {}
    for manifest_path, owner in manifest_owners.items():
        manifest_value = json.loads((artifact_root / manifest_path).read_text(encoding="utf-8"))
        for reference in manifest_value["layers"]:
            object_owners.setdefault(reference["path"], set()).add(owner)
    shared_object_paths = [path for path, owners in object_owners.items() if len(owners) > 1]
    assert shared_object_paths, "fixture must contain a content-addressed object shared by episodes"

    shared_entries = []
    for path in sorted(item for item in artifact_root.rglob("*") if item.is_file()):
        relative_path = path.relative_to(artifact_root).as_posix()
        if relative_path == channel_path.name:
            artifact_type, owner = "channel_dataset_npz", "child-1"
        elif relative_path == second_channel_path.name:
            artifact_type, owner = "channel_dataset_npz", "child-2"
        elif relative_path in manifest_owners:
            artifact_type, owner = "umcd_episode_store_manifest_json", manifest_owners[relative_path]
        else:
            artifact_type = "umcd_episode_store_content_npz"
            owner = sorted(object_owners[relative_path])[0]
        payload_bytes = path.read_bytes()
        entry = {
            "artifact_id": f"shared-{len(shared_entries)}",
            "artifact_type": artifact_type,
            "path": relative_path,
            "bytes": len(payload_bytes),
            "sha256": hashlib.sha256(payload_bytes).hexdigest(),
        }
        if artifact_type == "umcd_episode_store_content_npz":
            entry["child_ids"] = sorted(object_owners[relative_path])
        else:
            entry["child_id"] = owner
        shared_entries.append(entry)
    shared_roster = json.dumps(
        {
            "schema": "umcd.channel_dataset_child_artifact_roster.v5",
            "child_kind": "episode_bundle",
            "artifact_root": artifact_root.name,
            "artifacts": shared_entries,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    shared_verified = VerifiedOpenSSHArtifact(
        payload=shared_roster,
        artifact_sha256=hashlib.sha256(shared_roster).hexdigest(),
        signature_sha256="a" * 64,
        allowed_signers_sha256="b" * 64,
        signer_id="test-signer",
        namespace=_CHILD_ARTIFACT_ROSTER_SIGNATURE_NAMESPACE,
    )
    shared_audit = _load_split_child_artifact_roster(
        tmp_path / "shared-roster.json",
        expected_child_kind="episode_bundle",
        expected_child_bindings={
            "child-1": SplitChildBinding("child-1", "parent-1", "episode_bundle"),
            "child-2": SplitChildBinding("child-2", "parent-2", "episode_bundle"),
        },
        verified_artifact=shared_verified,
    )["audit"]
    assert shared_audit["schema"] == "umcd.split_child_artifact_audit.v6"
    shared_entry = next(
        item
        for item in shared_audit["artifacts"]
        if item["path"] in shared_object_paths
    )
    assert shared_entry["child_ids"] == ["child-1", "child-2"]
    assert len(shared_entry["episode_manifest_paths"]) == 2

@pytest.mark.parametrize("declared_parent", ["parent-1", "other-parent"])
def test_nr_trajectory_child_artifact_validates_store_and_split_lineage(
    tmp_path: Path, declared_parent: str
) -> None:
    child_id = "child-1"
    parent_id = "parent-1"
    ChannelDataset(
        h=np.ones((1, 1, 8), dtype=np.complex64),
        y=np.zeros(1, dtype=np.uint8),
        group=np.asarray([parent_id]),
        domain=np.asarray(["fixture"]),
        source=np.asarray(["synthetic"]),
        metadata=(
            {
                "split_subject_id": parent_id,
                "split_child_id": child_id,
                "split_child_kind": "receiver_observation",
            },
        ),
    ).save(tmp_path / "child.npz")
    estimate = NRChannelEstimate(
        h_hat=np.zeros((1, 1, 2, 1, 1, 14, 24), dtype=np.complex64),
        error_variance=np.zeros((1, 1, 1, 1, 1, 14, 24), dtype=np.float32),
        pilot_mask=np.zeros((1, 1, 14, 24), dtype=np.bool_),
        metadata={"receiver": "fixture"},
    )
    observation = NRObservation(
        sample_id="sample-1",
        received_grid=np.zeros((1, 1, 2, 14, 24), dtype=np.complex64),
        channel_estimate=estimate,
        fft_bin_indices=tuple(range(24)),
        receiver_config=NRPUSCHConfig(n_size_bwp=2, subcarrier_spacing_khz=30),
    )
    trajectory = NRTrajectory(
        trajectory_id="trajectory-1",
        sample_times_s=(0.0,),
        records=(
            NRChannelRecord(
                observation=observation,
                targets={"fixture_target": np.asarray([1.0], dtype=np.float32)},
                target_masks={},
                target_definitions={
                    "fixture_target": TargetDefinition.versioned(
                        semantic="split-lineage test target",
                        units="fixture value",
                        source="split-lineage test fixture",
                        reference="fixed synthetic test scalar",
                        aggregation_rule="one scalar per observation",
                        definition_version="tests.split-lineage.target.v1",
                    )
                },
                provenance={
                    "split_subject_id": declared_parent,
                    "split_child_id": child_id,
                    "split_child_kind": "receiver_observation",
                },
            ),
        ),
    )
    stored = write_nr_trajectory(tmp_path, trajectory)
    manifest_path = stored.manifest_relative_path
    record_manifest_path = (
        f"manifests/{hashlib.sha256(b'sample-1').hexdigest()}.json"
    )
    specifications = [
        ("record", "nr_record_store_manifest_json", record_manifest_path),
        ("dataset", "channel_dataset_npz", "child.npz"),
        ("trajectory", "nr_trajectory_store_manifest_json", manifest_path),
    ]
    entries = []
    for artifact_id, artifact_type, relative_path in specifications:
        payload = (tmp_path / relative_path).read_bytes()
        entries.append(
            {
                "artifact_id": artifact_id,
                "artifact_type": artifact_type,
                "child_id": child_id,
                "path": relative_path,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    roster_payload = json.dumps(
        {
            "schema": "umcd.channel_dataset_child_artifact_roster.v2",
            "child_kind": "receiver_observation",
            "artifacts": entries,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    verified = VerifiedOpenSSHArtifact(
        payload=roster_payload,
        artifact_sha256=hashlib.sha256(roster_payload).hexdigest(),
        signature_sha256="a" * 64,
        allowed_signers_sha256="b" * 64,
        signer_id="test-signer",
        namespace=_CHILD_ARTIFACT_ROSTER_SIGNATURE_NAMESPACE,
    )
    bindings = {
        child_id: SplitChildBinding(child_id, parent_id, "receiver_observation")
    }

    if declared_parent != parent_id:
        with pytest.raises(ValueError, match="different split_subject_id"):
            _load_split_child_artifact_roster(
                tmp_path / "roster.json",
                expected_child_kind="receiver_observation",
                expected_child_bindings=bindings,
                verified_artifact=verified,
            )
        return

    report = _load_split_child_artifact_roster(
        tmp_path / "roster.json",
        expected_child_kind="receiver_observation",
        expected_child_bindings=bindings,
        verified_artifact=verified,
    )["audit"]
    assert report["semantic_artifact_count"] == 3
    assert report["byte_only_artifact_count"] == 0
    assert report["trajectory_sample_count"] == 1
    assert report["trajectory_count"] == 1
    trajectory_report = next(
        item for item in report["artifacts"] if item["artifact_type"] == specifications[2][1]
    )
    assert trajectory_report["semantic_validator"] == "nr_trajectory_store_v1"
    assert trajectory_report["trajectory_id"] == "trajectory-1"
    record_report = next(
        item for item in report["artifacts"] if item["artifact_type"] == specifications[0][1]
    )
    assert record_report["semantic_validator"] == "nr_record_store_v1"
    assert record_report["nr_sample_id"] == "sample-1"

    from datasets.nr_store import write_nr_record

    v3_directory = tmp_path / "v3-roster"
    v3_store = v3_directory / "child-store"
    v3_store.mkdir(parents=True)
    shutil.copyfile(tmp_path / "child.npz", v3_store / "child.npz")
    stored_record = write_nr_record(v3_store, trajectory.records[0])
    v3_paths = [
        ("dataset", "channel_dataset_npz", "child.npz"),
        (
            "record-manifest",
            "nr_record_store_manifest_json",
            f"manifests/{hashlib.sha256(b'sample-1').hexdigest()}.json",
        ),
        ("record-content", "nr_record_store_content_npz", stored_record.relative_path),
    ]
    v3_entries = []
    for artifact_id, artifact_type, relative_path in v3_paths:
        payload = (v3_store / relative_path).read_bytes()
        v3_entries.append(
            {
                "artifact_id": artifact_id,
                "artifact_type": artifact_type,
                "child_id": child_id,
                "path": relative_path,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    v3_payload = json.dumps(
        {
            "schema": "umcd.channel_dataset_child_artifact_roster.v3",
            "child_kind": "receiver_observation",
            "artifact_root": "child-store",
            "artifacts": v3_entries,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    v3_verified = VerifiedOpenSSHArtifact(
        payload=v3_payload,
        artifact_sha256=hashlib.sha256(v3_payload).hexdigest(),
        signature_sha256="a" * 64,
        allowed_signers_sha256="b" * 64,
        signer_id="test-signer",
        namespace=_CHILD_ARTIFACT_ROSTER_SIGNATURE_NAMESPACE,
    )
    v3_report = _load_split_child_artifact_roster(
        v3_directory / "roster.json",
        expected_child_kind="receiver_observation",
        expected_child_bindings=bindings,
        verified_artifact=v3_verified,
    )["audit"]
    assert v3_report["artifact_directory_inventory_status"] == "EXACT"
    assert v3_report["semantic_artifact_count"] == 3
    assert v3_report["byte_only_artifact_count"] == 0
    assert "artifact types without semantic validators are rejected" in v3_report["authority"]
    content_report = next(
        item
        for item in v3_report["artifacts"]
        if item["artifact_type"] == "nr_record_store_content_npz"
    )
    assert content_report["semantic_validator"] == "nr_record_store_v1"
    assert content_report["nr_sample_id"] == "sample-1"

    trajectory_payload = tmp_path / stored.relative_path
    trajectory_payload.write_bytes(trajectory_payload.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="trajectory content hash"):
        _load_split_child_artifact_roster(
            tmp_path / "roster.json",
            expected_child_kind="receiver_observation",
            expected_child_bindings=bindings,
            verified_artifact=verified,
        )


def test_child_artifact_semantic_rows_must_match_frozen_child_and_parent(tmp_path) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    dataset_path = artifact_root / "child.npz"
    ChannelDataset(
        h=np.ones((1, 1, 8), dtype=np.complex64),
        y=np.zeros(1, dtype=np.uint8),
        group=np.asarray(["parent-1"]),
        domain=np.asarray(["fixture"]),
        source=np.asarray(["synthetic"]),
        metadata=(
            {
                "split_subject_id": "parent-1",
                "split_child_id": "child-1",
                "split_child_kind": "receiver_observation",
            },
        ),
    ).save(dataset_path)

    report = _read_channel_dataset_child_rows(
        artifact_root,
        "child.npz",
        child_id="child-1",
        child_kind="receiver_observation",
        parent_subject_id="parent-1",
        expected_bytes=dataset_path.stat().st_size,
        expected_sha256=hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
    )
    assert report["row_count"] == 1
    assert len(report["row_metadata_sha256"]) == 64

    with pytest.raises(ValueError, match="different child ID"):
        _read_channel_dataset_child_rows(
            artifact_root,
            "child.npz",
            child_id="child-2",
            child_kind="receiver_observation",
            parent_subject_id="parent-1",
            expected_bytes=dataset_path.stat().st_size,
            expected_sha256=hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        )
    with pytest.raises(ValueError, match="different parent"):
        _read_channel_dataset_child_rows(
            artifact_root,
            "child.npz",
            child_id="child-1",
            child_kind="receiver_observation",
            parent_subject_id="parent-2",
            expected_bytes=dataset_path.stat().st_size,
            expected_sha256=hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
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


def test_audit_binds_exact_input_bytes_and_returns_derived_subjects(tmp_path) -> None:
    scene_hash = "a" * 64
    first_path = tmp_path / "run-1.npz"
    second_path = tmp_path / "run-2.npz"
    _write_dataset(first_path, "parent-a", 0.0, scene_hash)
    _write_dataset(second_path, "parent-b", 0.5, scene_hash)

    report = audit_channel_datasets(
        [("run-2", second_path), ("run-1", first_path)], spatial_buffer_m=1.0
    )

    assert report["status"] == "NOT_QUALIFIED"
    assert report["lineage"]["spatially_linked_subjects"] == 2
    assert [item["name"] for item in report["inputs"]] == ["run-1", "run-2"]
    assert report["inputs"][0]["sha256"] == hashlib.sha256(first_path.read_bytes()).hexdigest()
    parents = {item["subject_id"]: item for item in report["subjects"]}
    first_id = _rt_subject_id(scene_hash, 0.0)
    second_id = _rt_subject_id(scene_hash, 0.5)
    assert set(parents) == {first_id, second_id}
    assert {
        relation["identity"]
        for relation in parents[first_id]["relations"]
        if relation["namespace"] == LeakageNamespace.SPATIAL_BUFFER.value
    } == {
        relation["identity"]
        for relation in parents[second_id]["relations"]
        if relation["namespace"] == LeakageNamespace.SPATIAL_BUFFER.value
    }
    json.dumps(report, sort_keys=True, allow_nan=False)


def test_saved_dataset_audit_rechecks_signed_v3_split_lock_and_footprints(tmp_path) -> None:
    ssh_keygen = shutil.which("ssh-keygen")
    if ssh_keygen is None:
        pytest.skip("OpenSSH is unavailable")
    paths = []
    for index in range(4):
        path = tmp_path / f"scene-{index}.npz"
        _write_dataset(path, f"parent-{index}", 0.0, f"{index + 1:064x}")
        paths.append(path)
    named_datasets = [
        (path.stem, ChannelDataset.load(path)) for path in paths
    ]
    registry, _ = make_channel_dataset_split_registry(
        named_datasets,
        spatial_buffer_m=0.0,
        policy=SplitPolicy.release_v2(seed="signed-saved-lineage-audit"),
    )
    children = [
        SplitChildBinding(
            child_id=f"{path.stem}:row:0",
            parent_subject_id=str(ChannelDataset.load(path).metadata[0]["split_subject_id"]),
            child_kind="receiver_observation",
        )
        for path in paths
    ]
    registry = bind_split_children(registry, children)
    registry_path = tmp_path / "split-registry.json"
    freeze_split_policy_registry(registry_path, registry)
    child_roster_path = tmp_path / "receiver-observations.json"
    _write_child_roster(child_roster_path, "receiver_observation", children)

    signer_id = "split-lineage-reviewer@example.org"
    private_key = tmp_path / "split-lineage-review-key"
    subprocess.run(
        [ssh_keygen, "-q", "-t", "ed25519", "-N", "", "-f", str(private_key)],
        check=True,
        capture_output=True,
        text=True,
    )
    public_key = private_key.with_suffix(".pub").read_text(encoding="utf-8").strip()
    allowed_signers = tmp_path / "split-lineage-allowed-signers"
    allowed_signers.write_text(
        f'{signer_id} namespaces="{SPLIT_POLICY_SIGNATURE_NAMESPACE},{_CHILD_ARTIFACT_ROSTER_SIGNATURE_NAMESPACE}" {public_key}\n',
        encoding="utf-8",
    )
    subprocess.run(
        [
            ssh_keygen,
            "-Y",
            "sign",
            "-n",
            SPLIT_POLICY_SIGNATURE_NAMESPACE,
            "-f",
            str(private_key),
            str(registry_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    legacy_report = audit_channel_datasets(
        [(path.stem, path) for path in paths],
        spatial_buffer_m=0.0,
        split_registry_path=registry_path,
        split_registry_signature=Path(f"{registry_path}.sig"),
        trusted_split_allowed_signers=allowed_signers,
        split_registry_signer_id=signer_id,
    )
    assert legacy_report["schema"] == "loslab.channel_split_lineage_audit_bundle.v2"
    assert legacy_report["split_registry_verification"]["children_audited"] is False

    report = audit_channel_datasets(
        [(path.stem, path) for path in paths],
        spatial_buffer_m=0.0,
        split_registry_path=registry_path,
        split_registry_signature=Path(f"{registry_path}.sig"),
        trusted_split_allowed_signers=allowed_signers,
        split_registry_signer_id=signer_id,
        split_child_rosters=[("receiver_observation", child_roster_path)],
    )

    verification = report["split_registry_verification"]
    assert report["schema"] == "loslab.channel_split_lineage_audit_bundle.v3"
    assert verification["status"] == "PASS_SIGNED_PARENT_FOOTPRINT_AND_CHILD_BINDING_ROSTERS"
    assert verification["schema"] == "umcd.split_registry.v3"
    assert verification["parent_roster_audit"]["parent_count"] == 4
    assert all(
        len(verification[field]) == 64
        for field in (
            "registry_file_sha256",
            "signature_sha256",
            "allowed_signers_sha256",
        )
    )
    assert verification["spatiotemporal_audit"]["footprints_sha256"] == registry.spatiotemporal_audit[
        "footprints_sha256"
    ]
    assert verification["children_audited"] is False
    assert verification["child_bindings_verified"] is True
    assert verification["child_artifact_existence_verified"] is False
    assert verification["child_roster_audits"][0]["audit"]["child_count"] == 4
    assert verification["child_roster_audits"][0]["audit"]["status"] == "PASS_INPUT_SCOPED"

    artifact_root = tmp_path / "child-artifacts"
    artifact_root.mkdir()
    artifact_entries = []
    for index, binding in enumerate(children):
        relative_path = f"artifact-{index}.npz"
        artifact_path = artifact_root / relative_path
        ChannelDataset(
            h=np.ones((1, 1, 8), dtype=np.complex64),
            y=np.zeros(1, dtype=np.uint8),
            group=np.asarray([binding.parent_subject_id]),
            domain=np.asarray(["child_fixture"]),
            source=np.asarray(["synthetic"]),
            metadata=(
                {
                    "split_subject_id": binding.parent_subject_id,
                    "split_child_id": binding.child_id,
                    "split_child_kind": binding.child_kind,
                },
            ),
        ).save(artifact_path)
        artifact_bytes = artifact_path.read_bytes()
        artifact_entries.append(
            {
                "artifact_id": f"dataset-{index}",
                "artifact_type": "channel_dataset_npz",
                "child_id": binding.child_id,
                "path": relative_path,
                "bytes": len(artifact_bytes),
                "sha256": hashlib.sha256(artifact_bytes).hexdigest(),
            }
        )
    supplemental_path = artifact_root / "target-sidecar.json"
    supplemental_path.write_text('{"format":"opaque-fixture"}', encoding="utf-8")
    supplemental_bytes = supplemental_path.read_bytes()
    artifact_entries.append(
        {
            "artifact_id": "target-sidecar",
            "artifact_type": "target_sidecar_json",
            "child_id": children[0].child_id,
            "path": "target-sidecar.json",
            "bytes": len(supplemental_bytes),
            "sha256": hashlib.sha256(supplemental_bytes).hexdigest(),
        }
    )
    artifact_roster_path = tmp_path / "receiver-artifact-roster.json"
    artifact_roster_path.write_text(
        json.dumps(
            {
                "schema": "umcd.channel_dataset_child_artifact_roster.v2",
                "child_kind": "receiver_observation",
                "artifacts": [
                    {**entry, "path": f"child-artifacts/{entry['path']}"}
                    for entry in artifact_entries
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    subprocess.run(
        [
            ssh_keygen,
            "-Y",
            "sign",
            "-n",
            _CHILD_ARTIFACT_ROSTER_SIGNATURE_NAMESPACE,
            "-f",
            str(private_key),
            str(artifact_roster_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    artifacts_report = audit_channel_datasets(
        [(path.stem, path) for path in paths],
        spatial_buffer_m=0.0,
        split_registry_path=registry_path,
        split_registry_signature=Path(f"{registry_path}.sig"),
        trusted_split_allowed_signers=allowed_signers,
        split_registry_signer_id=signer_id,
        split_child_rosters=[("receiver_observation", child_roster_path)],
        split_child_artifact_rosters=[("receiver_observation", artifact_roster_path)],
    )
    artifact_verification = artifacts_report["split_registry_verification"]
    assert artifacts_report["schema"] == "loslab.channel_split_lineage_audit_bundle.v7"
    assert (
        artifact_verification["status"]
        == "PASS_SIGNED_PARENT_FOOTPRINT_CHILD_BINDINGS_AND_ARTIFACT_BYTES"
    )
    assert artifact_verification["child_artifact_existence_verified"] is True
    assert artifact_verification["child_artifact_bytes_verified"] is True
    assert artifact_verification["child_artifact_semantic_rows_verified"] is False
    assert artifact_verification["child_artifact_semantic_count"] == len(children)
    assert artifact_verification["child_artifact_byte_only_count"] == 1
    assert artifact_verification["child_artifact_trajectory_sample_count"] == 0
    assert artifact_verification["child_artifact_trajectory_count"] == 0
    assert artifact_verification["children_audited"] is True
    assert (
        artifact_verification["child_artifact_roster_audits"][0]["audit"]["child_count"]
        == len(children)
    )
    assert (
        artifact_verification["child_artifact_roster_audits"][0]["audit"]["row_count"]
        == len(children)
    )
    artifact_roster_metadata = artifact_verification["child_artifact_roster_audits"][0]
    assert artifact_roster_metadata["signature_namespace"] == (
        _CHILD_ARTIFACT_ROSTER_SIGNATURE_NAMESPACE
    )
    assert artifact_roster_metadata["signer_id"] == signer_id
    assert all(
        len(artifact_roster_metadata[field]) == 64
        for field in ("sha256", "signature_sha256", "allowed_signers_sha256")
    )

    signed_roster_bytes = artifact_roster_path.read_bytes()
    artifact_roster_path.write_bytes(signed_roster_bytes + b" ")
    with pytest.raises(ValueError, match="signature is invalid"):
        audit_channel_datasets(
            [(path.stem, path) for path in paths],
            spatial_buffer_m=0.0,
            split_registry_path=registry_path,
            split_registry_signature=Path(f"{registry_path}.sig"),
            trusted_split_allowed_signers=allowed_signers,
            split_registry_signer_id=signer_id,
            split_child_rosters=[("receiver_observation", child_roster_path)],
            split_child_artifact_rosters=[("receiver_observation", artifact_roster_path)],
        )
    artifact_roster_path.write_bytes(signed_roster_bytes)

    tampered_artifact = artifact_root / artifact_entries[0]["path"]
    original_artifact_bytes = tampered_artifact.read_bytes()
    tampered_artifact.write_bytes(b"X" + original_artifact_bytes[1:])
    with pytest.raises(ValueError, match="SHA-256 differs from roster"):
        audit_channel_datasets(
            [(path.stem, path) for path in paths],
            spatial_buffer_m=0.0,
            split_registry_path=registry_path,
            split_registry_signature=Path(f"{registry_path}.sig"),
            trusted_split_allowed_signers=allowed_signers,
            split_registry_signer_id=signer_id,
            split_child_rosters=[("receiver_observation", child_roster_path)],
            split_child_artifact_rosters=[("receiver_observation", artifact_roster_path)],
        )

    _write_child_roster(child_roster_path, "receiver_observation", children[:-1])
    with pytest.raises(ValueError, match="do not cover exactly the frozen"):
        audit_channel_datasets(
            [(path.stem, path) for path in paths],
            spatial_buffer_m=0.0,
            split_registry_path=registry_path,
            split_registry_signature=Path(f"{registry_path}.sig"),
            trusted_split_allowed_signers=allowed_signers,
            split_registry_signer_id=signer_id,
            split_child_rosters=[("receiver_observation", child_roster_path)],
        )

    changed_dataset = ChannelDataset.load(paths[0])
    changed_metadata = dict(changed_dataset.metadata[0])
    changed_metadata["rx"] = [0.01, 0.0, 1.5]
    changed_metadata["split_positions_m"] = [0.01, 0.0, 1.5]
    ChannelDataset(
        h=changed_dataset.h,
        y=changed_dataset.y,
        group=changed_dataset.group,
        domain=changed_dataset.domain,
        source=changed_dataset.source,
        metadata=(changed_metadata,),
    ).save(paths[0])
    with pytest.raises(ValueError, match="footprint inventory differs from frozen registry"):
        audit_channel_datasets(
            [(path.stem, path) for path in paths],
            spatial_buffer_m=0.0,
            split_registry_path=registry_path,
            split_registry_signature=Path(f"{registry_path}.sig"),
            trusted_split_allowed_signers=allowed_signers,
            split_registry_signer_id=signer_id,
        )


def test_saved_dataset_audit_rejects_partial_split_signature_configuration(tmp_path) -> None:
    with pytest.raises(ValueError, match="requires a registry, signature"):
        audit_channel_datasets(
            [("missing", tmp_path / "does-not-need-to-exist.npz")],
            spatial_buffer_m=0.0,
            split_registry_path=tmp_path / "registry.json",
        )


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


def test_audit_rejects_duplicate_names_and_missing_lineage(tmp_path) -> None:
    path = tmp_path / "run.npz"
    _write_dataset(path, "parent-a", 0.0, "a" * 64)

    with pytest.raises(ValueError, match="duplicate channel dataset name"):
        audit_channel_datasets([("same", path), ("same", path)], spatial_buffer_m=0.0)

    dataset = ChannelDataset.load(path)
    metadata = dict(dataset.metadata[0])
    del metadata["split_parent_type"]
    ChannelDataset(
        h=dataset.h,
        y=dataset.y,
        group=dataset.group,
        domain=dataset.domain,
        source=dataset.source,
        metadata=(metadata,),
    ).save(path)
    with pytest.raises(ValueError, match="invalid split_parent_type"):
        audit_channel_datasets([("legacy", path)], spatial_buffer_m=0.0)


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


def test_audit_checks_exact_frozen_inventory_and_preserves_unqualified_status(tmp_path) -> None:
    dataset_path = tmp_path / "run.npz"
    inventory_path = tmp_path / "inventory.json"
    _write_dataset(dataset_path, "parent-a", 0.0, "a" * 64)
    _write_inventory(inventory_path, dataset_path)

    report = audit_channel_datasets(
        [("pilot", dataset_path)],
        spatial_buffer_m=0.0,
        inventory_manifest=inventory_path,
    )

    assert report["input_completeness"]["status"] == "matched_frozen_inventory"
    assert report["input_completeness"]["matched_dataset_count"] == 1
    assert report["input_completeness"]["authority"].startswith("self-declared")
    assert report["status"] == "NOT_QUALIFIED"


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


def test_audit_verifies_signed_inventory_against_external_keyring(tmp_path) -> None:
    dataset_path = tmp_path / "run.npz"
    inventory_path = tmp_path / "inventory.json"
    _write_dataset(dataset_path, "parent-a", 0.0, "a" * 64)
    _write_inventory(inventory_path, dataset_path)
    signature_path, allowed_signers = _sign_inventory(inventory_path, tmp_path)

    report = audit_channel_datasets(
        [("pilot", dataset_path)],
        spatial_buffer_m=0.0,
        inventory_manifest=inventory_path,
        inventory_signature=signature_path,
        trusted_allowed_signers=allowed_signers,
        inventory_signer_id=_INVENTORY_SIGNER,
    )

    completeness = report["input_completeness"]
    assert completeness["status"] == "matched_signed_frozen_inventory"
    assert completeness["signature_namespace"] == _INVENTORY_NAMESPACE
    assert completeness["allowed_signers_sha256"]
    assert report["status"] == "NOT_QUALIFIED"
    with pytest.raises(ValueError, match="signature is invalid or signer is not trusted"):
        audit_channel_datasets(
            [("pilot", dataset_path)],
            spatial_buffer_m=0.0,
            inventory_manifest=inventory_path,
            inventory_signature=signature_path,
            trusted_allowed_signers=allowed_signers,
            inventory_signer_id="untrusted-inventory-reviewer",
        )


def test_audit_frozen_inventory_rejects_omitted_files_and_changed_bytes(tmp_path) -> None:
    first = tmp_path / "run-a.npz"
    second = tmp_path / "run-b.npz"
    inventory_path = tmp_path / "inventory.json"
    _write_dataset(first, "parent-a", 0.0, "a" * 64)
    _write_dataset(second, "parent-b", 0.0, "a" * 64)
    first_bytes = first.read_bytes()
    inventory_path.write_text(
        json.dumps(
            {
                "schema": "umcd.channel_dataset_inventory.v1",
                "frozen": True,
                "inventory_id": "fixture-v1",
                "datasets": [
                    {
                        "name": "run-a",
                        "path": first.name,
                        "sha256": hashlib.sha256(first_bytes).hexdigest(),
                        "bytes": len(first_bytes),
                        "rows": 1,
                    },
                    {
                        "name": "run-b",
                        "path": second.name,
                        "sha256": hashlib.sha256(second.read_bytes()).hexdigest(),
                        "bytes": second.stat().st_size,
                        "rows": 1,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="do not match frozen inventory"):
        audit_channel_datasets(
            [("run-a", first)],
            spatial_buffer_m=0.0,
            inventory_manifest=inventory_path,
        )

    second.write_bytes(second.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="SHA-256 differs"):
        audit_channel_datasets(
            [("run-a", first), ("run-b", second)],
            spatial_buffer_m=0.0,
            inventory_manifest=inventory_path,
        )


@pytest.mark.parametrize("unsafe_path", ["../run.npz", "folder/..", "C:/run.npz"])
def test_audit_frozen_inventory_rejects_unsafe_paths(tmp_path, unsafe_path) -> None:
    dataset_path = tmp_path / "run.npz"
    inventory_path = tmp_path / "inventory.json"
    _write_dataset(dataset_path, "parent-a", 0.0, "a" * 64)
    _write_inventory(inventory_path, dataset_path)
    payload = json.loads(inventory_path.read_text(encoding="utf-8"))
    payload["datasets"][0]["path"] = unsafe_path
    inventory_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unsafe relative path"):
        audit_channel_datasets(
            [("pilot", dataset_path)],
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
