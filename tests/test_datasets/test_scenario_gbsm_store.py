"""Canonical scenario GBSM record conversion and immutable storage contracts."""

from __future__ import annotations

import hashlib
import json
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
from datasets.scenario_gbsm import (
    ScenarioGBSMRecord,
    ScenarioGBSMRun,
    read_scenario_gbsm_run,
    scenario_gbsm_episode_bundle,
    scenario_gbsm_run_from_channel,
    scenario_gbsm_run_to_episode_bundles,
    write_scenario_gbsm_run,
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


def _native_channel(*, scenario_los: bool = False) -> _NativeChannel:
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
        "carrier_frequency_hz": 3.5e9,
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

    monkeypatch.setattr(episode_store, "_publish_immutable", fail_scenario_layer)
    with pytest.raises(OSError, match="scenario GBSM episode-layer publication failure"):
        write_episode_bundle(tmp_path, bundle)

    assert list((tmp_path / "objects" / "episode").rglob("*.npz"))
    assert not (tmp_path / "objects" / "scenario_gbsm").exists()
    assert not (tmp_path / "manifests" / "episodes").exists()

    monkeypatch.setattr(episode_store, "_publish_immutable", publish)
    pointer = write_episode_bundle(tmp_path, bundle)
    restored = read_episode_bundle(tmp_path, pointer)
    assert restored.episode.episode_id == "drop-000"
    assert len(restored.scenario_gbsm_records) == 1
    np.testing.assert_array_equal(restored.scenario_gbsm_records[0].h_path, run.records[0].h_path)
    np.testing.assert_array_equal(restored.scenario_gbsm_records[0].cfr, run.records[0].cfr)
    assert not [path for path in tmp_path.rglob(".*") if path.is_file()]
