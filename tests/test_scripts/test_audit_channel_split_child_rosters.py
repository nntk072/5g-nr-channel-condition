from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
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
    LeakageNamespace,
    LeakageRelation,
    SplitChildBinding,
    SplitParentType,
    SplitPolicy,
    SplitSubject,
    VerifiedOpenSSHArtifact,
    make_split_policy_registry,
)
from datasets.targets import TargetDefinition
from receivers.nr import NRChannelEstimate, NRPUSCHConfig
from scripts.audit_channel_split_lineage import (
    _CHILD_ARTIFACT_ROSTER_SIGNATURE_NAMESPACE,
    _load_split_child_artifact_roster,
    _parse_split_child_roster,
    _read_channel_dataset_child_rows,
    _require_exact_child_object_owners,
    _require_shared_content_owners_share_split,
    audit_channel_datasets,
)
from tests.test_scripts.audit_channel_split_support import (
    _INVENTORY_SIGNER,
    _sign_inventory,
    _write_dataset,
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


@pytest.mark.parametrize("roster_argument", ["split_child_rosters", "split_child_artifact_rosters"])
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
    invalid_v2 = dict(
        v2_valid, artifacts=[dict(v2_valid["artifacts"][0], artifact_type="Bad Type")]
    )
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
    supplemental.write_bytes(b'{"episode":"episode-1"}')
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
    assert (
        "unrecognized supplemental types receive byte-integrity checks only" in audit["authority"]
    )


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
    bindings = {"child-1": SplitChildBinding("child-1", "parent-1", "receiver_observation")}

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
    bindings = {"child-1": SplitChildBinding("child-1", "parent-1", "episode_bundle")}

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

    incorrect_bindings = {"child-1": SplitChildBinding("child-1", "other-parent", "episode_bundle")}
    with pytest.raises(ValueError, match="different split parent episode ID"):
        _load_split_child_artifact_roster(
            tmp_path / "roster.json",
            expected_child_kind="episode_bundle",
            expected_child_bindings=incorrect_bindings,
            verified_artifact=verified,
        )

    second_episode = replace(episode, episode_id="parent-2", split_group_id="asset-group-2")
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
            artifact_type, owner = (
                "umcd_episode_store_manifest_json",
                manifest_owners[relative_path],
            )
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
        item for item in shared_audit["artifacts"] if item["path"] in shared_object_paths
    )
    assert shared_entry["child_ids"] == ["child-1", "child-2"]
    assert len(shared_entry["episode_manifest_paths"]) == 2

    cross_split_registry = make_split_policy_registry(
        tuple(
            SplitSubject(parent_id, SplitParentType.NETWORK_EPISODE, "fixture-family")
            for parent_id in ("parent-1", "parent-2", "parent-3")
        ),
        policy=SplitPolicy(
            seed="shared-object-cross-split-regression",
            split_fractions={"train": 1 / 3, "validation": 1 / 3, "test": 1 / 3},
        ),
    )
    assert cross_split_registry.split_for_parent("parent-1") != (
        cross_split_registry.split_for_parent("parent-2")
    )
    cross_split_bindings = {
        "child-1": SplitChildBinding("child-1", "parent-1", "episode_bundle"),
        "child-2": SplitChildBinding("child-2", "parent-2", "episode_bundle"),
    }
    for artifact_type in (
        "umcd_episode_store_content_npz",
        "scenario_gbsm_record_content_npz",
    ):
        with pytest.raises(ValueError, match="owners assigned to different frozen splits"):
            _require_shared_content_owners_share_split(
                artifact_type,
                ("child-1", "child-2"),
                cross_split_bindings,
                cross_split_registry,
            )
    with pytest.raises(ValueError, match="owners assigned to different frozen splits"):
        _load_split_child_artifact_roster(
            tmp_path / "shared-roster.json",
            expected_child_kind="episode_bundle",
            expected_child_bindings={
                "child-1": SplitChildBinding("child-1", "parent-1", "episode_bundle"),
                "child-2": SplitChildBinding("child-2", "parent-2", "episode_bundle"),
            },
            verified_artifact=shared_verified,
            split_registry=cross_split_registry,
        )

    shared_relation = LeakageRelation(LeakageNamespace.SITE, "shared-site")
    same_split_registry = make_split_policy_registry(
        (
            SplitSubject(
                "parent-1",
                SplitParentType.NETWORK_EPISODE,
                "fixture-family",
                (shared_relation,),
            ),
            SplitSubject(
                "parent-2",
                SplitParentType.NETWORK_EPISODE,
                "fixture-family",
                (shared_relation,),
            ),
            SplitSubject("parent-3", SplitParentType.NETWORK_EPISODE, "fixture-family"),
            SplitSubject("parent-4", SplitParentType.NETWORK_EPISODE, "fixture-family"),
        ),
        policy=SplitPolicy(
            seed="shared-object-same-split-regression",
            split_fractions={"train": 1 / 3, "validation": 1 / 3, "test": 1 / 3},
        ),
    )
    assert same_split_registry.split_for_parent("parent-1") == (
        same_split_registry.split_for_parent("parent-2")
    )
    same_split_bindings = {
        "child-1": SplitChildBinding("child-1", "parent-1", "episode_bundle"),
        "child-2": SplitChildBinding("child-2", "parent-2", "episode_bundle"),
    }
    for artifact_type in (
        "umcd_episode_store_content_npz",
        "scenario_gbsm_record_content_npz",
    ):
        _require_shared_content_owners_share_split(
            artifact_type,
            ("child-1", "child-2"),
            same_split_bindings,
            same_split_registry,
        )
    same_split_audit = _load_split_child_artifact_roster(
        tmp_path / "shared-roster.json",
        expected_child_kind="episode_bundle",
        expected_child_bindings={
            "child-1": SplitChildBinding("child-1", "parent-1", "episode_bundle"),
            "child-2": SplitChildBinding("child-2", "parent-2", "episode_bundle"),
        },
        verified_artifact=shared_verified,
        split_registry=same_split_registry,
    )["audit"]
    assert same_split_audit["semantic_artifact_count"] > 0


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
    record_manifest_path = f"manifests/{hashlib.sha256(b'sample-1').hexdigest()}.json"
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
    bindings = {child_id: SplitChildBinding(child_id, parent_id, "receiver_observation")}

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


@pytest.mark.parametrize(
    ("array_change", "message"),
    [
        (lambda arrays: arrays.update(h=np.ones((2, 1, 8), dtype=np.complex64)), "row counts"),
        (
            lambda arrays: arrays.update(h=np.ones((1, 1, 8), dtype=np.float32)),
            "invalid shapes or dtypes",
        ),
        (lambda arrays: arrays.update(group=np.asarray(["parent-1", "extra"])), "row counts"),
    ],
)
def test_child_artifact_semantic_rows_bind_metadata_to_array_headers(
    tmp_path, array_change, message: str
) -> None:
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
    with np.load(dataset_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    array_change(arrays)
    with dataset_path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)

    payload = dataset_path.read_bytes()
    with pytest.raises(ValueError, match=message):
        _read_channel_dataset_child_rows(
            artifact_root,
            "child.npz",
            child_id="child-1",
            child_kind="receiver_observation",
            parent_subject_id="parent-1",
            expected_bytes=len(payload),
            expected_sha256=hashlib.sha256(payload).hexdigest(),
        )


def test_complete_split_audit_rejects_registry_without_child_bindings(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import audit_channel_split_lineage as audit_module

    dataset_paths = [tmp_path / f"parent-{index}.npz" for index in range(4)]
    for index, dataset_path in enumerate(dataset_paths):
        _write_dataset(dataset_path, f"unused-parent-{index}", 0.0, f"{index + 1:064x}")
    registry, _ = make_channel_dataset_split_registry(
        [(path.stem, ChannelDataset.load(path)) for path in dataset_paths],
        spatial_buffer_m=0.0,
        policy=SplitPolicy.release_v2(seed="complete-audit-empty-children"),
    )
    monkeypatch.setattr(
        audit_module.channel_audit,
        "verify_signed_split_policy_registry_attestation",
        lambda *_args, **_kwargs: (registry, object()),
    )
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(
        json.dumps(
            {
                "schema": "umcd.channel_dataset_inventory.v1",
                "frozen": True,
                "inventory_id": "fixture-v1",
                "datasets": [
                    {
                        "name": path.stem,
                        "path": path.name,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "bytes": path.stat().st_size,
                        "rows": 1,
                    }
                    for path in dataset_paths
                ],
            }
        ),
        encoding="utf-8",
    )
    inventory_signature, inventory_allowed_signers = _sign_inventory(inventory_path, tmp_path)

    with pytest.raises(ValueError, match="requires at least one frozen child binding"):
        audit_channel_datasets(
            [(path.stem, path) for path in dataset_paths],
            spatial_buffer_m=0.0,
            inventory_manifest=inventory_path,
            inventory_signature=inventory_signature,
            trusted_allowed_signers=inventory_allowed_signers,
            inventory_signer_id=_INVENTORY_SIGNER,
            split_registry_path=tmp_path / "registry.json",
            split_registry_signature=tmp_path / "registry.json.sig",
            trusted_split_allowed_signers=tmp_path / "split-allowed-signers",
            split_registry_signer_id="split-reviewer@example.org",
            require_complete_split_audit=True,
        )
