"""Canonical scenario GBSM record conversion and immutable storage contracts."""

from __future__ import annotations

import hashlib
import io
import json
from copy import copy
from dataclasses import dataclass, replace
from pathlib import Path

import datasets.scenario_gbsm as scenario_gbsm_store
import numpy as np
import pytest
from datasets.episode_store import EpisodeBundle, read_episode_bundle, write_episode_bundle
from datasets.network import (
    AntennaArraySpec,
    DirectedLink,
    LinkDirection,
    LinkRole,
    NetworkEpisode,
    NetworkNode,
    NodeRole,
)
from datasets.provenance import UMCDRunProvenance
from datasets.scenario_gbsm import (
    ScenarioGBSMRecord,
    ScenarioGBSMRun,
    _scenario_array_spec,
    read_scenario_gbsm_run,
    scenario_gbsm_episode_bundle,
    scenario_gbsm_run_from_channel,
    scenario_gbsm_run_to_episode_bundles,
    write_scenario_gbsm_run,
)
from datasets.split_policy import (
    LeakageNamespace,
    LeakageRelation,
    SplitParentType,
    SplitPolicy,
    SplitSubject,
    make_split_policy_registry,
)


def _digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class _NativeChannel:
    h_path: np.ndarray
    tau_s: np.ndarray
    channel_frequency_response: np.ndarray
    metadata: dict[str, object]


def _native_channel(*, scenario_los: bool = False, carrier_hz: float = 3.5e9) -> _NativeChannel:
    h_path = np.zeros((1, 1, 1, 1, 1, 2, 1), dtype=np.complex128)
    h_path[0, 0, 0, 0, 0, :, 0] = (1.0 + 0.5j, 0.25 - 0.25j)
    tau_s = np.asarray([[[[0.0, 2.0e-7]]]], dtype=np.float64)
    offsets = (np.arange(8, dtype=np.float64) - 4) * 30_000.0
    phase = np.exp(
        -2j
        * np.pi
        * tau_s[:, :, None, :, None, :, None, None]
        * offsets.reshape(1, 1, 1, 1, 1, 1, 1, -1)
    )
    cfr = np.sum(h_path[..., None] * phase, axis=5)
    topology_row = {
        "ut_loc": [[1.0, 2.0, 1.5]],
        "bs_loc": [[0.0, 0.0, 10.0]],
        "ut_velocities": [[0.0, 0.0, 0.0]],
        "ut_orientations": [[0.0, 0.0, 0.0]],
        "bs_orientations": [[0.0, 0.0, 0.0]],
        "in_state": [False],
        "los": [[scenario_los]],
    }
    configuration = {
        "scenario": "UMi",
        "sionna_version": "2.2.0",
        "direction": "downlink",
        "carrier_frequency_hz": carrier_hz,
        "sampling_frequency_hz": 30_000.0,
        "batch_size": 1,
        "subcarrier_spacing_hz": 30_000.0,
        "num_subcarriers": 8,
    }
    metadata: dict[str, object] = {
        "scenario": "UMi",
        "scenario_los_rows": [scenario_los],
        "topology_rows": [topology_row],
        "topology_row_sha256": [_digest(topology_row)],
        "topology_sha256": _digest({key: [value] for key, value in topology_row.items()}),
        "generator_configuration": configuration,
        "configuration_sha256": _digest(configuration),
        "sionna_version": "2.2.0",
    }
    return _NativeChannel(h_path, tau_s, cfr, metadata)


def _two_record_run() -> ScenarioGBSMRun:
    run = scenario_gbsm_run_from_channel(
        _native_channel(), run_id="pilot-retry", sample_ids=["drop-000"]
    )
    first = run.records[0]
    second = replace(
        first,
        sample_id="drop-001",
        sample_index=1,
        metadata={**dict(first.metadata), "split_subject_id": "drop-001"},
    )
    return ScenarioGBSMRun(run.run_id, (first, second))


def test_scenario_gbsm_conversion_and_store_round_trip(tmp_path) -> None:
    run = scenario_gbsm_run_from_channel(
        _native_channel(scenario_los=False), run_id="pilot-umix", sample_ids=["drop-000"]
    )

    assert run.records[0].scenario_los is False
    assert run.records[0].metadata["geometric_los_label"] is None
    pointer = write_scenario_gbsm_run(tmp_path, run)
    restored = read_scenario_gbsm_run(tmp_path, pointer)

    assert restored.run_id == run.run_id
    assert restored.records[0].sample_id == "drop-000"
    assert restored.records[0].scenario_los is False
    np.testing.assert_array_equal(restored.records[0].h_path, run.records[0].h_path)
    np.testing.assert_array_equal(restored.records[0].tau_s, run.records[0].tau_s)
    np.testing.assert_array_equal(restored.records[0].cfr, run.records[0].cfr)


def test_scenario_gbsm_record_rejects_nonpositive_absolute_rf_frequency() -> None:
    with pytest.raises(ValueError, match="positive RF frequencies"):
        scenario_gbsm_run_from_channel(
            _native_channel(carrier_hz=30_000.0), run_id="invalid-rf-grid", sample_ids=["drop-000"]
        )


def test_scenario_gbsm_run_split_audit_binds_shared_parent_component() -> None:
    run = _two_record_run()
    shared_relation = LeakageRelation(
        LeakageNamespace.TRAJECTORY_SESSION, f"scenario_gbsm_run:{run.run_id}"
    )
    registry = make_split_policy_registry(
        [
            SplitSubject(
                record.sample_id,
                SplitParentType.NETWORK_EPISODE,
                "scenario_gbsm:UMi",
                (shared_relation,),
            )
            for record in run.records
        ]
        + [
            SplitSubject("unrelated-parent-a", SplitParentType.NETWORK_EPISODE, "fixture"),
            SplitSubject("unrelated-parent-b", SplitParentType.NETWORK_EPISODE, "fixture"),
        ],
        policy=SplitPolicy(
            seed="scenario-run-split-audit",
            split_fractions={"train": 0.8, "validation": 0.1, "test": 0.1},
        ),
    )

    report = run.audit_split_assignment(registry)
    assert report["schema"] == "umcd.scenario_gbsm_split_audit.v1"
    assert report["run_id"] == run.run_id
    assert report["registry_sha256"] == registry.to_dict()["registry_sha256"]
    assert report["parent_count"] == 2
    assert {row["parent_subject_id"] for row in report["parents"]} == {
        "drop-000",
        "drop-001",
    }
    assert {row["component_id"] for row in report["parents"]} == {report["component_id"]}
    assert {row["split"] for row in report["parents"]} == {report["split"]}
    core = {key: value for key, value in report.items() if key != "audit_sha256"}
    assert report["audit_sha256"] == _digest(core)
    jsonschema = pytest.importorskip("jsonschema")
    schema_path = (
        Path(__file__).resolve().parents[2]
        / "datasets/schema/scenario-gbsm-split-audit.v1.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(report)


def test_scenario_gbsm_run_split_audit_rejects_missing_shared_relation() -> None:
    run = _two_record_run()
    registry = make_split_policy_registry(
        [
            SplitSubject(
                record.sample_id,
                SplitParentType.NETWORK_EPISODE,
                "scenario_gbsm:UMi",
            )
            for record in run.records
        ]
        + [
            SplitSubject("unrelated-parent-a", SplitParentType.NETWORK_EPISODE, "fixture"),
            SplitSubject("unrelated-parent-b", SplitParentType.NETWORK_EPISODE, "fixture"),
        ],
        policy=SplitPolicy(
            seed="scenario-run-split-audit-missing-link",
            split_fractions={"train": 0.8, "validation": 0.1, "test": 0.1},
        ),
    )
    with pytest.raises(ValueError, match="omits the scenario run relation"):
        run.audit_split_assignment(registry)


def test_scenario_gbsm_run_split_audit_requires_registry_type() -> None:
    with pytest.raises(TypeError, match="registry must be a SplitPolicyRegistry"):
        _two_record_run().audit_split_assignment(object())


def test_scenario_gbsm_run_split_audit_rejects_missing_parent() -> None:
    run = _two_record_run()
    registry = make_split_policy_registry(
        [
            SplitSubject("unrelated-a", SplitParentType.NETWORK_EPISODE, "fixture"),
            SplitSubject("unrelated-b", SplitParentType.NETWORK_EPISODE, "fixture"),
            SplitSubject("unrelated-c", SplitParentType.NETWORK_EPISODE, "fixture"),
        ],
        policy=SplitPolicy(
            seed="missing-scenario-parent",
            split_fractions={"train": 0.8, "validation": 0.1, "test": 0.1},
        ),
    )
    with pytest.raises(ValueError, match="parent is absent from frozen split registry"):
        run.audit_split_assignment(registry)


def test_scenario_gbsm_run_split_audit_rejects_split_component_drift() -> None:
    run = _two_record_run()
    shared_relation = LeakageRelation(
        LeakageNamespace.TRAJECTORY_SESSION, f"scenario_gbsm_run:{run.run_id}"
    )
    registry = make_split_policy_registry(
        [
            SplitSubject(
                record.sample_id,
                SplitParentType.NETWORK_EPISODE,
                "scenario_gbsm:UMi",
                (shared_relation,),
            )
            for record in run.records
        ]
        + [
            SplitSubject(f"unrelated-{index}", SplitParentType.NETWORK_EPISODE, "fixture")
            for index in range(3)
        ],
        policy=SplitPolicy(
            seed="scenario-component-drift",
            split_fractions={"train": 0.8, "validation": 0.1, "test": 0.1},
        ),
    )
    subject_to_component = dict(registry.subject_to_component)
    subject_to_component[run.records[1].sample_id] = subject_to_component["unrelated-0"]
    object.__setattr__(registry, "subject_to_component", subject_to_component)

    with pytest.raises(ValueError, match="share one frozen split component"):
        run.audit_split_assignment(registry)


def test_scenario_gbsm_conversion_assembles_canonical_episode_bundle(tmp_path) -> None:
    run = scenario_gbsm_run_from_channel(
        _native_channel(scenario_los=True), run_id="pilot-umix", sample_ids=["drop-000"]
    )

    bundles = scenario_gbsm_run_to_episode_bundles(run)
    assert len(bundles) == 1
    bundle = bundles[0]
    assert bundle.episode.episode_id == "drop-000"
    assert bundle.episode.split_group_id == "scenario_gbsm_run:pilot-umix"
    assert bundle.episode.family_id == "scenario_gbsm:UMi"
    assert bundle.path_oracle is None
    assert len(bundle.episode.links) == 1
    assert bundle.episode.links[0].direction is LinkDirection.DOWNLINK
    assert bundle.episode.nodes[0].position_world_m == (0.0, 0.0, 10.0)
    assert bundle.episode.nodes[1].position_world_m == (1.0, 2.0, 1.5)
    assert len(bundle.scenario_gbsm_records) == 1

    pointer = write_episode_bundle(tmp_path, bundle)
    restored = read_episode_bundle(tmp_path, pointer)
    assert restored.episode.episode_id == "drop-000"
    assert restored.scenario_gbsm_records[0].scenario_los is True
    assert restored.scenario_gbsm_records[0].metadata["geometric_los_label"] is None
    np.testing.assert_array_equal(restored.scenario_gbsm_records[0].cfr, run.records[0].cfr)


def test_scenario_gbsm_run_rejects_mixed_models() -> None:
    run = _two_record_run()
    mixed = copy(run.records[1])
    object.__setattr__(mixed, "scenario", "UMa")
    with pytest.raises(ValueError, match="cannot mix scenario models"):
        ScenarioGBSMRun(run.run_id, (run.records[0], mixed))


def test_scenario_gbsm_episode_bundle_supports_uplink() -> None:
    record = scenario_gbsm_run_from_channel(
        _native_channel(), run_id="uplink", sample_ids=["drop-000"]
    ).records[0]
    record = _record_with_metadata_edit(record, direction="uplink")
    bundle = scenario_gbsm_episode_bundle(record)
    assert bundle.episode.links[0].direction is LinkDirection.UPLINK


def test_scenario_gbsm_episode_bundle_requires_typed_record() -> None:
    with pytest.raises(TypeError, match="record must be a ScenarioGBSMRecord"):
        scenario_gbsm_episode_bundle(object())


def test_scenario_gbsm_writer_requires_typed_run(tmp_path) -> None:
    with pytest.raises(TypeError, match="run must be a ScenarioGBSMRun"):
        write_scenario_gbsm_run(tmp_path, object())


def test_scenario_gbsm_writer_enforces_archive_size_limit(tmp_path, monkeypatch) -> None:
    run = scenario_gbsm_run_from_channel(
        _native_channel(), run_id="archive-limit", sample_ids=["drop-000"]
    )
    monkeypatch.setattr(scenario_gbsm_store, "_record_bytes", lambda _record: b"too large")
    monkeypatch.setattr(scenario_gbsm_store, "MAX_SCENARIO_GBSM_OBJECT_BYTES", 1)
    with pytest.raises(ValueError, match="record exceeds its byte ceiling"):
        write_scenario_gbsm_run(tmp_path, run)


def test_scenario_gbsm_writer_enforces_manifest_size_limit(tmp_path, monkeypatch) -> None:
    run = scenario_gbsm_run_from_channel(
        _native_channel(), run_id="manifest-limit", sample_ids=["drop-000"]
    )
    monkeypatch.setattr(scenario_gbsm_store, "MAX_SCENARIO_GBSM_MANIFEST_BYTES", 1)
    with pytest.raises(ValueError, match="manifest exceeds its byte ceiling"):
        write_scenario_gbsm_run(tmp_path, run)


def test_scenario_gbsm_reader_accepts_an_absolute_manifest_path(tmp_path) -> None:
    run = scenario_gbsm_run_from_channel(
        _native_channel(), run_id="absolute-manifest", sample_ids=["drop-000"]
    )
    pointer = write_scenario_gbsm_run(tmp_path, run)
    restored = read_scenario_gbsm_run(tmp_path, tmp_path / pointer.relative_path)
    assert restored.run_id == run.run_id


def test_scenario_gbsm_reader_rejects_extra_manifest_fields(tmp_path) -> None:
    run = scenario_gbsm_run_from_channel(
        _native_channel(), run_id="extra-manifest-field", sample_ids=["drop-000"]
    )
    pointer = write_scenario_gbsm_run(tmp_path, run)
    manifest_path = tmp_path / pointer.relative_path
    manifest = json.loads(manifest_path.read_text())
    manifest["extra"] = True
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest fields are invalid"):
        read_scenario_gbsm_run(tmp_path, pointer)


def test_scenario_gbsm_reader_requires_the_canonical_manifest_path(tmp_path) -> None:
    run = scenario_gbsm_run_from_channel(
        _native_channel(), run_id="noncanonical-manifest", sample_ids=["drop-000"]
    )
    pointer = write_scenario_gbsm_run(tmp_path, run)
    relocated = tmp_path / "relocated-manifest.json"
    relocated.write_bytes((tmp_path / pointer.relative_path).read_bytes())

    with pytest.raises(ValueError, match="not at its canonical run path"):
        read_scenario_gbsm_run(tmp_path, relocated)


def test_scenario_gbsm_reader_compares_run_id_with_its_pointer(tmp_path) -> None:
    run = scenario_gbsm_run_from_channel(
        _native_channel(), run_id="run-id-pointer", sample_ids=["drop-000"]
    )
    pointer = write_scenario_gbsm_run(tmp_path, run)
    manifest = json.loads((tmp_path / pointer.relative_path).read_text())
    manifest["run_id"] = "different-run-id"
    core = {key: manifest[key] for key in ("schema", "schema_version", "run_id", "records")}
    digest = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest["content_sha256"] = digest
    relative_path = (
        f"scenario_gbsm/manifests/{hashlib.sha256(manifest['run_id'].encode()).hexdigest()}.json"
    )
    manifest_path = tmp_path / relative_path
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n")
    changed_pointer = replace(pointer, relative_path=relative_path, content_sha256=digest)

    with pytest.raises(ValueError, match="run ID disagrees with its pointer"):
        read_scenario_gbsm_run(tmp_path, changed_pointer)


def test_scenario_gbsm_reader_rejects_malformed_record_references(tmp_path) -> None:
    run = scenario_gbsm_run_from_channel(
        _native_channel(), run_id="malformed-record-reference", sample_ids=["drop-000"]
    )
    pointer = write_scenario_gbsm_run(tmp_path, run)
    manifest = json.loads((tmp_path / pointer.relative_path).read_text())
    manifest["records"][0]["content_sha256"] = "not-a-sha256"
    core = {key: manifest[key] for key in ("schema", "schema_version", "run_id", "records")}
    digest = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest["content_sha256"] = digest
    (tmp_path / pointer.relative_path).write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    )
    changed_pointer = replace(pointer, content_sha256=digest)

    with pytest.raises(ValueError, match="manifest record reference is malformed"):
        read_scenario_gbsm_run(tmp_path, changed_pointer)


def test_scenario_gbsm_reader_rejects_blank_manifest_run_id(tmp_path) -> None:
    run = scenario_gbsm_run_from_channel(
        _native_channel(), run_id="blank-manifest-id", sample_ids=["drop-000"]
    )
    pointer = write_scenario_gbsm_run(tmp_path, run)
    manifest_path = tmp_path / pointer.relative_path
    manifest = json.loads(manifest_path.read_text())
    manifest["run_id"] = " "
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest run ID is invalid"):
        read_scenario_gbsm_run(tmp_path, pointer)


def test_scenario_gbsm_reader_rejects_pointer_sample_order_drift(tmp_path) -> None:
    run = _two_record_run()
    pointer = write_scenario_gbsm_run(tmp_path, run)
    changed = replace(pointer, sample_ids=tuple(reversed(pointer.sample_ids)))
    with pytest.raises(ValueError, match="sample order disagrees with run pointer"):
        read_scenario_gbsm_run(tmp_path, changed)


def _replace_first_object_payload(tmp_path, pointer, payload: bytes):
    manifest_path = tmp_path / pointer.relative_path
    manifest = json.loads(manifest_path.read_text())
    digest = hashlib.sha256(payload).hexdigest()
    relative_path = f"scenario_gbsm/records/{digest[:2]}/{digest}.npz"
    object_path = tmp_path / relative_path
    object_path.parent.mkdir(parents=True, exist_ok=True)
    object_path.write_bytes(payload)
    manifest["records"][0]["content_sha256"] = digest
    manifest["records"][0]["relative_path"] = relative_path
    core = {key: manifest[key] for key in ("schema", "schema_version", "run_id", "records")}
    manifest_digest = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest["content_sha256"] = manifest_digest
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n")
    return replace(pointer, content_sha256=manifest_digest)


def test_scenario_gbsm_reader_rejects_wrong_archive_members(tmp_path) -> None:
    run = scenario_gbsm_run_from_channel(
        _native_channel(), run_id="wrong-archive-members", sample_ids=["drop-000"]
    )
    pointer = write_scenario_gbsm_run(tmp_path, run)
    stream = io.BytesIO()
    np.savez_compressed(
        stream,
        h_path=np.asarray([1]),
        tau_s=np.asarray([1]),
        cfr=np.asarray([1]),
        subcarrier_offsets_hz=np.asarray([1]),
        unexpected=np.asarray([1]),
    )
    pointer = _replace_first_object_payload(tmp_path, pointer, stream.getvalue())

    with pytest.raises(ValueError, match="archive fields are invalid"):
        read_scenario_gbsm_run(tmp_path, pointer)


def test_scenario_gbsm_reader_rejects_archive_member_count_drift(tmp_path) -> None:
    run = scenario_gbsm_run_from_channel(
        _native_channel(), run_id="extra-archive-member", sample_ids=["drop-000"]
    )
    pointer = write_scenario_gbsm_run(tmp_path, run)
    stream = io.BytesIO()
    np.savez_compressed(
        stream,
        h_path=np.asarray([1]),
        tau_s=np.asarray([1]),
        cfr=np.asarray([1]),
        subcarrier_offsets_hz=np.asarray([1]),
        metadata_json=np.asarray([1], dtype=np.uint8),
        extra=np.asarray([1]),
    )
    pointer = _replace_first_object_payload(tmp_path, pointer, stream.getvalue())

    with pytest.raises(ValueError, match="archive exceeds its member contract"):
        read_scenario_gbsm_run(tmp_path, pointer)


def test_scenario_gbsm_reader_rejects_unknown_record_schema(tmp_path) -> None:
    run = scenario_gbsm_run_from_channel(
        _native_channel(), run_id="unknown-record-schema", sample_ids=["drop-000"]
    )
    pointer = write_scenario_gbsm_run(tmp_path, run)
    stream = io.BytesIO()
    metadata = {
        "schema": "umcd.scenario_gbsm_record.unknown",
        "sample_id": "drop-000",
        "run_id": "unknown-record-schema",
        "sample_index": 0,
        "scenario": "UMi",
        "scenario_los": False,
        "metadata": {},
    }
    np.savez_compressed(
        stream,
        h_path=np.zeros((1,)),
        tau_s=np.zeros((1,)),
        cfr=np.zeros((1,)),
        subcarrier_offsets_hz=np.zeros((1,)),
        metadata_json=np.frombuffer(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode(), dtype=np.uint8
        ),
    )
    pointer = _replace_first_object_payload(tmp_path, pointer, stream.getvalue())

    with pytest.raises(ValueError, match="unsupported scenario GBSM record"):
        read_scenario_gbsm_run(tmp_path, pointer)


def test_scenario_gbsm_reader_checks_record_identity_against_manifest(tmp_path) -> None:
    run = scenario_gbsm_run_from_channel(
        _native_channel(), run_id="wrong-archive-id", sample_ids=["drop-000"]
    )
    pointer = write_scenario_gbsm_run(tmp_path, run)
    stream = io.BytesIO()
    metadata = {
        "schema": "umcd.scenario_gbsm_record.v1",
        "sample_id": "other-drop",
        "run_id": "wrong-archive-id",
        "sample_index": 0,
        "scenario": "UMi",
        "scenario_los": False,
        "metadata": {},
    }
    np.savez_compressed(
        stream,
        h_path=np.zeros((1,)),
        tau_s=np.zeros((1,)),
        cfr=np.zeros((1,)),
        subcarrier_offsets_hz=np.zeros((1,)),
        metadata_json=np.frombuffer(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode(), dtype=np.uint8
        ),
    )
    pointer = _replace_first_object_payload(tmp_path, pointer, stream.getvalue())

    with pytest.raises(ValueError, match="identity disagrees with manifest"):
        read_scenario_gbsm_run(tmp_path, pointer)


def test_episode_store_enforces_compressed_scenario_layer_limit(tmp_path, monkeypatch) -> None:
    record = scenario_gbsm_run_from_channel(
        _native_channel(), run_id="episode-layer-limit", sample_ids=["drop-000"]
    ).records[0]
    bundle = scenario_gbsm_episode_bundle(record)
    monkeypatch.setattr(scenario_gbsm_store, "_record_bytes", lambda _record: b"too large")
    monkeypatch.setattr(scenario_gbsm_store, "MAX_SCENARIO_GBSM_OBJECT_BYTES", 1)
    with pytest.raises(ValueError, match="compressed byte ceiling"):
        write_episode_bundle(tmp_path, bundle)


def test_scenario_gbsm_episode_rejects_split_group_drift() -> None:
    record = scenario_gbsm_run_from_channel(
        _native_channel(), run_id="pilot-umix", sample_ids=["drop-000"]
    ).records[0]
    bundle = scenario_gbsm_episode_bundle(record)
    episode_with_wrong_group = replace(bundle.episode, split_group_id="another-run")

    with pytest.raises(ValueError, match="split group differs from its generation run"):
        EpisodeBundle(
            episode=episode_with_wrong_group,
            path_oracle=None,
            channel_views=(),
            targets=(),
            scenario_gbsm_records=(record,),
        )


def test_scenario_gbsm_episode_bundle_tracks_moving_ut() -> None:
    channel = _native_channel()
    channel.h_path = np.repeat(channel.h_path, 2, axis=-1)
    channel.channel_frequency_response = np.repeat(channel.channel_frequency_response, 2, axis=5)
    channel.metadata["generator_configuration"]["sampling_frequency_hz"] = 2.0
    channel.metadata["generator_configuration"]["batch_size"] = 1
    channel.metadata["topology_rows"][0]["ut_velocities"] = [[1.0, 0.0, 0.0]]
    channel.metadata["topology_sha256"] = _digest(
        {key: [value] for key, value in channel.metadata["topology_rows"][0].items()}
    )
    channel.metadata["topology_row_sha256"] = [_digest(channel.metadata["topology_rows"][0])]
    channel.metadata["configuration_sha256"] = _digest(channel.metadata["generator_configuration"])
    run = scenario_gbsm_run_from_channel(channel, run_id="pilot-moving", sample_ids=["drop-moving"])

    bundle = scenario_gbsm_episode_bundle(run.records[0])
    assert bundle.episode.frame_times_s == (0.0, 0.5)
    np.testing.assert_allclose(
        bundle.episode.pose_timeline.positions_world_m[:, 1],
        [[1.0, 2.0, 1.5], [1.5, 2.0, 1.5]],
    )


def test_scenario_gbsm_episode_bundle_maps_dual_polarized_ports_to_elements() -> None:
    channel = _native_channel()
    channel.h_path = np.zeros((1, 1, 4, 1, 4, 2, 1), dtype=np.complex128)
    channel.h_path[0, 0, 0, 0, 0, :, 0] = (1.0 + 0.5j, 0.25 - 0.25j)
    offsets = (np.arange(8, dtype=np.float64) - 4) * 30_000.0
    phase = np.exp(
        -2j
        * np.pi
        * channel.tau_s[:, :, None, :, None, :, None, None]
        * offsets.reshape(1, 1, 1, 1, 1, 1, 1, -1)
    )
    channel.channel_frequency_response = np.sum(channel.h_path[..., None] * phase, axis=5)
    configuration = channel.metadata["generator_configuration"]
    configuration["ut_array_layout"] = {
        "schema": "umcd.sionna_array_layout.v1",
        "normalized_element_positions_wavelengths": [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        "pattern_labels_in_port_order": ["V", "H"],
        "port_order": "pattern_major_element_minor",
    }
    configuration["bs_array_layout"] = configuration["ut_array_layout"]
    channel.metadata["configuration_sha256"] = _digest(configuration)

    record = scenario_gbsm_run_from_channel(
        channel, run_id="pilot-dual-pol", sample_ids=["drop-dual-pol"]
    ).records[0]
    bundle = scenario_gbsm_episode_bundle(record)

    ue_array = bundle.episode.array("ut-array")
    assert ue_array.num_elements == 2
    assert ue_array.num_ports == 4
    assert ue_array.port_ids == (
        "V:element-0",
        "V:element-1",
        "H:element-0",
        "H:element-1",
    )
    np.testing.assert_array_equal(
        ue_array.port_element_weights,
        [[1, 0], [0, 1], [1, 0], [0, 1]],
    )


def test_scenario_gbsm_conversion_rejects_mismatched_scenario_los() -> None:
    channel = _native_channel(scenario_los=False)
    channel.metadata["scenario_los_rows"] = [True]

    with pytest.raises(ValueError, match="scenario LOS row disagrees"):
        scenario_gbsm_run_from_channel(channel, run_id="pilot-umix", sample_ids=["drop-000"])


def test_scenario_gbsm_record_rejects_split_subject_drift() -> None:
    record = scenario_gbsm_run_from_channel(
        _native_channel(), run_id="pilot-umix", sample_ids=["drop-000"]
    ).records[0]
    metadata = dict(record.metadata)
    metadata["split_subject_id"] = "another-drop"

    with pytest.raises(ValueError, match="split subject must equal its sample ID"):
        ScenarioGBSMRecord(
            sample_id=record.sample_id,
            run_id=record.run_id,
            sample_index=record.sample_index,
            scenario=record.scenario,
            scenario_los=record.scenario_los,
            h_path=record.h_path,
            tau_s=record.tau_s,
            cfr=record.cfr,
            subcarrier_offsets_hz=record.subcarrier_offsets_hz,
            metadata=metadata,
        )


def test_scenario_gbsm_reader_rejects_manifest_object_roster_tampering(tmp_path) -> None:
    run = scenario_gbsm_run_from_channel(
        _native_channel(), run_id="pilot-umix", sample_ids=["drop-000"]
    )
    pointer = write_scenario_gbsm_run(tmp_path, run)
    manifest_path = tmp_path / pointer.relative_path
    manifest = json.loads(manifest_path.read_text())
    manifest["records"][0]["relative_path"] = "scenario_gbsm/records/00/forged.npz"
    manifest["content_sha256"] = _digest(
        {key: manifest[key] for key in ("schema", "schema_version", "run_id", "records")}
    )
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n")

    with pytest.raises(ValueError, match="manifest hash is invalid"):
        read_scenario_gbsm_run(tmp_path, pointer)


def test_scenario_gbsm_partial_object_publication_is_retryable(tmp_path, monkeypatch) -> None:
    run = _two_record_run()
    publish = scenario_gbsm_store._publish_immutable
    object_publications = 0

    def fail_second_object(path, payload):
        nonlocal object_publications
        if "records" in path.parts:
            object_publications += 1
            if object_publications == 2:
                raise OSError("injected second-record publication failure")
        publish(path, payload)

    monkeypatch.setattr(scenario_gbsm_store, "_publish_immutable", fail_second_object)
    with pytest.raises(OSError, match="second-record publication failure"):
        write_scenario_gbsm_run(tmp_path, run)

    stored_objects = list((tmp_path / "scenario_gbsm/records").rglob("*.npz"))
    assert len(stored_objects) == 1
    assert not (tmp_path / "scenario_gbsm/manifests").exists()

    monkeypatch.setattr(scenario_gbsm_store, "_publish_immutable", publish)
    pointer = write_scenario_gbsm_run(tmp_path, run)
    restored = read_scenario_gbsm_run(tmp_path, pointer)
    assert tuple(record.sample_id for record in restored.records) == ("drop-000", "drop-001")
    assert len(list((tmp_path / "scenario_gbsm/records").rglob("*.npz"))) == 2
    for expected, actual in zip(run.records, restored.records, strict=True):
        np.testing.assert_array_equal(actual.h_path, expected.h_path)
        np.testing.assert_array_equal(actual.tau_s, expected.tau_s)
        np.testing.assert_array_equal(actual.cfr, expected.cfr)


def test_scenario_gbsm_manifest_publication_failure_is_retryable(tmp_path, monkeypatch) -> None:
    run = _two_record_run()
    publish = scenario_gbsm_store._publish_immutable

    def fail_manifest(path, payload):
        if "manifests" in path.parts:
            raise OSError("injected run-manifest publication failure")
        publish(path, payload)

    monkeypatch.setattr(scenario_gbsm_store, "_publish_immutable", fail_manifest)
    with pytest.raises(OSError, match="run-manifest publication failure"):
        write_scenario_gbsm_run(tmp_path, run)

    assert len(list((tmp_path / "scenario_gbsm/records").rglob("*.npz"))) == 2
    assert not (tmp_path / "scenario_gbsm/manifests").exists()

    monkeypatch.setattr(scenario_gbsm_store, "_publish_immutable", publish)
    pointer = write_scenario_gbsm_run(tmp_path, run)
    restored = read_scenario_gbsm_run(tmp_path, pointer)
    assert tuple(record.sample_id for record in restored.records) == ("drop-000", "drop-001")


def test_scenario_gbsm_reader_detects_corrupted_content_addressed_object(tmp_path) -> None:
    run = scenario_gbsm_run_from_channel(
        _native_channel(), run_id="pilot-corruption", sample_ids=["drop-000"]
    )
    pointer = write_scenario_gbsm_run(tmp_path, run)
    manifest = json.loads((tmp_path / pointer.relative_path).read_text(encoding="utf-8"))
    payload_path = tmp_path / manifest["records"][0]["relative_path"]
    payload_path.write_bytes(b"corrupt object bytes")

    with pytest.raises(ValueError, match="content hash is invalid"):
        read_scenario_gbsm_run(tmp_path, pointer)


def test_scenario_gbsm_is_a_distinct_episode_store_layer(tmp_path) -> None:
    run = scenario_gbsm_run_from_channel(
        _native_channel(), run_id="pilot-umix", sample_ids=["drop-000"]
    )
    record = run.records[0]
    identity = np.eye(3)
    gnb_array = AntennaArraySpec(
        array_id="bs-array",
        element_positions_local_m=np.zeros((1, 3)),
        polarization_per_element=("V",),
        orientation_local_to_world=identity,
        port_ids=("bs-port-0",),
        port_element_weights=np.ones((1, 1), dtype=np.complex128),
    )
    ue_array = AntennaArraySpec(
        array_id="ut-array",
        element_positions_local_m=np.zeros((1, 3)),
        polarization_per_element=("V",),
        orientation_local_to_world=identity,
        port_ids=("ut-port-0",),
        port_element_weights=np.ones((1, 1), dtype=np.complex128),
    )
    episode = NetworkEpisode(
        episode_id="drop-000",
        split_group_id="scenario_gbsm_run:pilot-umix",
        family_id="scenario_gbsm:UMi",
        carrier_hz=3.5e9,
        arrays=(gnb_array, ue_array),
        nodes=(
            NetworkNode("bs", NodeRole.GNB, "bs-array", (0.0, 0.0, 10.0), 1),
            NetworkNode("ut", NodeRole.UE, "ut-array", (1.0, 2.0, 1.5), 1),
        ),
        links=(
            DirectedLink(
                "bs-to-ut",
                "bs",
                "ut",
                (0,),
                (0,),
                LinkDirection.DOWNLINK,
                LinkRole.SERVING,
            ),
        ),
        frame_times_s=(0.0,),
        subcarrier_offsets_hz=record.subcarrier_offsets_hz,
    )
    bundle = EpisodeBundle(
        episode,
        None,
        (),
        (),
        scenario_gbsm_records=(record,),
    )

    pointer = write_episode_bundle(tmp_path, bundle)
    restored = read_episode_bundle(tmp_path, pointer)

    assert restored.path_oracle is None
    assert len(restored.scenario_gbsm_records) == 1
    assert restored.scenario_gbsm_records[0].metadata["geometric_los_label"] is None
    np.testing.assert_array_equal(restored.scenario_gbsm_records[0].cfr, record.cfr)
    jsonschema = pytest.importorskip("jsonschema")
    schema_path = Path(__file__).resolve().parents[2] / "datasets/schema/umcd-bundle-v1.schema.json"
    schema = json.loads(schema_path.read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(
        json.loads((tmp_path / pointer.relative_path).read_text())
    )


def test_scenario_gbsm_episode_layer_rejects_corrupted_object(tmp_path) -> None:
    run = scenario_gbsm_run_from_channel(
        _native_channel(), run_id="pilot-episode-layer-corruption", sample_ids=["drop-000"]
    )
    bundle = scenario_gbsm_episode_bundle(run.records[0])
    pointer = write_episode_bundle(tmp_path, bundle)
    manifest = json.loads((tmp_path / pointer.relative_path).read_text(encoding="utf-8"))
    scenario_layer = next(item for item in manifest["layers"] if item["layer"] == "scenario_gbsm")
    object_path = tmp_path / scenario_layer["path"]
    payload = bytearray(object_path.read_bytes())
    payload[len(payload) // 2] ^= 0x01
    object_path.write_bytes(payload)

    with pytest.raises(ValueError, match="content hash"):
        read_episode_bundle(tmp_path, pointer)


def test_scenario_gbsm_episode_layer_publication_failure_is_retryable(
    tmp_path, monkeypatch
) -> None:
    import datasets.episode_store as episode_store

    run = scenario_gbsm_run_from_channel(
        _native_channel(), run_id="pilot-episode-layer-retry", sample_ids=["drop-000"]
    )
    bundle = scenario_gbsm_episode_bundle(run.records[0])
    publish = episode_store._publish_immutable
    injected = False

    def fail_scenario_layer(path, payload):
        nonlocal injected
        if "scenario_gbsm" in path.parts and not injected:
            injected = True
            raise OSError("injected scenario GBSM episode-layer publication failure")
        publish(path, payload)

    monkeypatch.setattr(episode_store.episode, "_publish_immutable", fail_scenario_layer)
    with pytest.raises(OSError, match="scenario GBSM episode-layer publication failure"):
        write_episode_bundle(tmp_path, bundle)

    assert list((tmp_path / "objects" / "episode").rglob("*.npz"))
    assert not (tmp_path / "objects" / "scenario_gbsm").exists()
    assert not (tmp_path / "manifests" / "episodes").exists()

    monkeypatch.setattr(episode_store.episode, "_publish_immutable", publish)
    pointer = write_episode_bundle(tmp_path, bundle)
    restored = read_episode_bundle(tmp_path, pointer)
    assert restored.episode.episode_id == "drop-000"
    assert len(restored.scenario_gbsm_records) == 1
    np.testing.assert_array_equal(restored.scenario_gbsm_records[0].h_path, run.records[0].h_path)
    np.testing.assert_array_equal(restored.scenario_gbsm_records[0].cfr, run.records[0].cfr)
    assert not [path for path in tmp_path.rglob(".*") if path.is_file()]


def test_scenario_gbsm_layer_accepts_bound_provenance_round_trip(tmp_path) -> None:
    run = scenario_gbsm_run_from_channel(
        _native_channel(), run_id="pilot-provenance-layer", sample_ids=["drop-000"]
    )
    provenance = UMCDRunProvenance(
        record_id="scenario-gbsm-generation",
        subject_layer="scenario_gbsm",
        source_family="scenario_gbsm:UMi",
        producer="Sionna 2.2.0 UMi adapter",
        package_source_sha256="a" * 64,
        software_versions={"python": "3.11", "sionna": "2.2.0"},
        backend={"backend": "cpu"},
        scene_asset_sha256={},
        seed=91,
        physical_config={"carrier_frequency_hz": 3.5e9},
        solver_config={"scenario": "UMi"},
        material_config={},
    )
    bundle = replace(scenario_gbsm_episode_bundle(run.records[0]), provenance_records=(provenance,))

    pointer = write_episode_bundle(tmp_path, bundle)
    restored = read_episode_bundle(tmp_path, pointer)
    assert restored.provenance_records[0].subject_layer == "scenario_gbsm"
    assert restored.provenance_records[0].to_dict() == provenance.to_dict()


def _record_with_metadata_edit(record: ScenarioGBSMRecord, **edits) -> ScenarioGBSMRecord:
    metadata = json.loads(json.dumps(scenario_gbsm_store._plain_json(record.metadata)))
    for key, value in edits.items():
        metadata["generator_configuration"][key] = value
    changed = copy(record)
    object.__setattr__(changed, "metadata", metadata)
    return changed


def test_scenario_array_spec_accepts_legacy_single_port_layout() -> None:
    record = scenario_gbsm_run_from_channel(
        _native_channel(), run_id="legacy-layout", sample_ids=["drop-000"]
    ).records[0]
    from datasets.scenario_gbsm import _sionna_orientation_matrix

    spec = _scenario_array_spec(
        record,
        endpoint="bs",
        carrier_hz=3.5e9,
        port_count=1,
        orientation=_sionna_orientation_matrix((0.0, 0.0, 0.0)),
    )
    assert spec.port_ids == ("V:element-0",)


@pytest.mark.parametrize(
    ("layout", "port_count", "message"),
    [
        (None, 2, "lacks captured bs array layout"),
        ({"schema": "wrong"}, 1, "unsupported contract"),
        (
            {
                "schema": "umcd.sionna_array_layout.v1",
                "port_order": "pattern_major_element_minor",
                "normalized_element_positions_wavelengths": [[0.0, float("nan"), 0.0]],
                "pattern_labels_in_port_order": ["V"],
            },
            1,
            "array layout is malformed",
        ),
        (
            {
                "schema": "umcd.sionna_array_layout.v1",
                "port_order": "pattern_major_element_minor",
                "normalized_element_positions_wavelengths": [[0.0, 0.0, 0.0]],
                "pattern_labels_in_port_order": ["V"],
            },
            2,
            "array layout does not match its channel axis",
        ),
    ],
)
def test_scenario_array_spec_rejects_incomplete_layouts(layout, port_count, message) -> None:
    record = scenario_gbsm_run_from_channel(
        _native_channel(), run_id="array-layout", sample_ids=["drop-000"]
    ).records[0]
    record = _record_with_metadata_edit(record, bs_array_layout=layout)
    from datasets.scenario_gbsm import _sionna_orientation_matrix

    with pytest.raises(ValueError, match=message):
        _scenario_array_spec(
            record,
            endpoint="bs",
            carrier_hz=3.5e9,
            port_count=port_count,
            orientation=_sionna_orientation_matrix((0.0, 0.0, 0.0)),
        )


def test_scenario_bundle_rejects_unsupported_link_direction() -> None:
    record = scenario_gbsm_run_from_channel(
        _native_channel(), run_id="bad-direction", sample_ids=["drop-000"]
    ).records[0]
    record = _record_with_metadata_edit(record, direction="sideways")
    with pytest.raises(ValueError, match="unsupported link direction"):
        scenario_gbsm_episode_bundle(record)


def test_run_to_episode_bundles_requires_a_typed_run() -> None:
    with pytest.raises(TypeError, match="run must be a ScenarioGBSMRun"):
        scenario_gbsm_run_to_episode_bundles(object())


@pytest.mark.parametrize(
    ("run_id", "sample_ids", "message"),
    [
        ("", ["drop-000"], "run_id must be a non-empty string"),
        ("run", [], "sample_ids must contain non-empty strings"),
        ("run", ["drop-000", " "], "sample_ids must contain non-empty strings"),
        ("run", ["drop-000", "drop-000"], "sample_ids must be unique"),
    ],
)
def test_channel_conversion_requires_canonical_run_and_sample_ids(run_id, sample_ids, message):
    with pytest.raises(ValueError, match=message):
        scenario_gbsm_run_from_channel(_native_channel(), run_id=run_id, sample_ids=sample_ids)


def test_channel_conversion_requires_generator_metadata() -> None:
    channel = _native_channel()
    channel.metadata = None
    with pytest.raises(TypeError, match="must expose generator metadata"):
        scenario_gbsm_run_from_channel(channel, run_id="run", sample_ids=["drop-000"])


@pytest.mark.parametrize(
    ("metadata_change", "message"),
    [
        (
            lambda metadata: metadata.update(topology_rows=[]),
            "lacks aligned scenario topology rows",
        ),
        (
            lambda metadata: metadata.update(topology_rows=[None]),
            "topology rows must be objects",
        ),
        (
            lambda metadata: metadata.update(topology_sha256="0" * 64),
            "batch topology digest is invalid",
        ),
        (
            lambda metadata: metadata.update(topology_row_sha256=["0" * 64]),
            "topology row digest differs",
        ),
        (
            lambda metadata: metadata.update(configuration_sha256="0" * 64),
            "configuration identity is invalid",
        ),
        (
            lambda metadata: metadata["generator_configuration"].update(batch_size=2),
            "configuration disagrees with scenario batch metadata",
        ),
        (
            lambda metadata: metadata["generator_configuration"].update(scenario="UMa"),
            "configuration disagrees with scenario batch metadata",
        ),
        (
            lambda metadata: metadata["generator_configuration"].update(num_subcarriers=7),
            "invalid subcarrier count",
        ),
        (
            lambda metadata: metadata["generator_configuration"].update(subcarrier_spacing_hz=0.0),
            "invalid subcarrier count",
        ),
    ],
)
def test_channel_conversion_rejects_inconsistent_batch_metadata(metadata_change, message):
    channel = _native_channel()
    channel.metadata = dict(channel.metadata)
    channel.metadata["generator_configuration"] = dict(channel.metadata["generator_configuration"])
    metadata_change(channel.metadata)
    if message not in {"configuration identity is invalid"}:
        channel.metadata["configuration_sha256"] = scenario_gbsm_store._digest_json(
            channel.metadata["generator_configuration"], label="generator configuration"
        )
    with pytest.raises(ValueError, match=message):
        scenario_gbsm_run_from_channel(channel, run_id="run", sample_ids=["drop-000"])


def test_channel_conversion_checks_sample_id_batch_axis() -> None:
    channel = _native_channel()
    channel.h_path = np.concatenate((channel.h_path, channel.h_path), axis=0)
    with pytest.raises(ValueError, match="match the generator result batch axis"):
        scenario_gbsm_run_from_channel(channel, run_id="run", sample_ids=["drop-000"])


def test_channel_conversion_rejects_different_topology_field_rosters() -> None:
    channel = _native_channel()
    channel.h_path = np.concatenate((channel.h_path, channel.h_path), axis=0)
    channel.tau_s = np.concatenate((channel.tau_s, channel.tau_s), axis=0)
    channel.channel_frequency_response = np.concatenate(
        (channel.channel_frequency_response, channel.channel_frequency_response), axis=0
    )
    metadata = dict(channel.metadata)
    configuration = dict(metadata["generator_configuration"])
    configuration["batch_size"] = 2
    metadata["generator_configuration"] = configuration
    metadata["configuration_sha256"] = scenario_gbsm_store._digest_json(
        configuration, label="generator configuration"
    )
    first = metadata["topology_rows"][0]
    second = {**first, "extra": 1}
    metadata["topology_rows"] = [first, second]
    metadata["topology_row_sha256"] = [
        scenario_gbsm_store._digest_json(row, label="topology row") for row in (first, second)
    ]
    metadata["scenario_los_rows"] = [False, False]
    channel.metadata = metadata

    with pytest.raises(ValueError, match="do not share one canonical field roster"):
        scenario_gbsm_run_from_channel(channel, run_id="run", sample_ids=["drop-000", "drop-001"])
