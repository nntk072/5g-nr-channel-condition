"""UMCD network episode, antenna, node, pose and time-axis schemas."""

from __future__ import annotations

import io
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from datasets.episode_store import (
    EpisodeBundle,
    _channel_payload,
    _episode_payload,
    _read_episode,
    read_episode_bundle,
    write_episode_bundle,
)
from datasets.network import (
    AntennaArraySpec,
    DirectedLink,
    LinkDirection,
    LinkRole,
    NetworkEpisode,
    NetworkNode,
    NodePoseTimeline,
    NodeRole,
)
from datasets.path_oracle import PathOracleTable, PathSolveStatus
from datasets.targets import (
    TargetDefinition,
    TargetValidity,
    TargetValue,
)
from datasets.time_axis import align_named_time_axes
from tests.test_datasets.umcd_schema_support import (
    _bundle,
    _dynamic_episode,
    _episode,
)


def test_network_episode_preserves_ordered_arrays_links_and_grid():
    episode = _episode()
    assert episode.node("gNB-1").rf_chain_count == 2
    assert episode.array("gNB-array").num_elements == 4
    assert episode.array("gNB-array").num_ports == 2
    assert episode.array("gNB-array").polarization_per_element == ("V", "V", "H", "H")
    assert episode.link("gNB-1-to-UE-1").tx_port_indices == (0, 1)
    assert not episode.subcarrier_offsets_hz.flags.writeable
    assert episode.position_at("UE-1", 0) == (20.0, 0.0, 1.5)
    np.testing.assert_array_equal(episode.orientation_at("gNB-1", 0), np.eye(3))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("element_positions_local_m", np.ones((4, 3), dtype=bool), "real numeric dtype"),
        (
            "element_positions_local_m",
            np.ones((4, 3), dtype=complex) * (1 + 1j),
            "real numeric dtype",
        ),
        ("orientation_local_to_world", np.eye(3, dtype=complex) * (1 + 1j), "real numeric dtype"),
        ("port_element_weights", np.ones((2, 4), dtype=object), "numeric dtype"),
    ],
)
def test_antenna_array_rejects_coercible_geometry_and_weight_dtypes(field, value, message):
    array = _episode().arrays[0]
    with pytest.raises(TypeError, match=message):
        replace(array, **{field: value})


@pytest.mark.parametrize("position", ((True, 0.0, 0.0), ("0.0", 0.0, 0.0), (0.0j, 0.0, 0.0)))
def test_network_node_rejects_coercible_non_real_positions(position):
    with pytest.raises(TypeError, match="real numeric coordinates"):
        NetworkNode("node", NodeRole.UE, "array", position, 1)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("positions_world_m", np.zeros((2, 2, 3), dtype=bool)),
        (
            "orientations_local_to_world",
            np.broadcast_to(np.eye(3), (2, 2, 3, 3)).astype(complex) * (1 + 1j),
        ),
    ),
)
def test_pose_timeline_rejects_coercible_geometry_dtypes(field, value):
    timeline = _dynamic_episode().pose_timeline
    assert timeline is not None
    with pytest.raises(TypeError, match="real numeric dtype"):
        replace(timeline, **{field: value})


@pytest.mark.parametrize("carrier", (True, "3.5e9", 3.5e9 + 0j))
def test_network_episode_rejects_non_real_carrier_scalars(carrier):
    with pytest.raises(TypeError, match="real numeric scalar"):
        replace(_episode(), carrier_hz=carrier)


def test_pose_timeline_validates_and_exposes_time_varying_node_poses():
    episode = _dynamic_episode()
    timeline = episode.pose_timeline
    assert timeline is not None
    np.testing.assert_array_equal(timeline.position("UE-1", 1), [22.0, 3.0, 1.5])
    assert episode.position_at("UE-1", 1) == (22.0, 3.0, 1.5)
    np.testing.assert_array_equal(
        timeline.orientation("UE-1", 1),
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    )
    assert not timeline.positions_world_m.flags.writeable
    with pytest.raises(IndexError, match="outside"):
        timeline.position("UE-1", -1)
    with pytest.raises(KeyError):
        timeline.position("unknown-node", 0)
    with pytest.raises(ValueError, match="proper rotation"):
        NodePoseTimeline(
            node_ids=("node",),
            frame_times_s=(0.0,),
            positions_world_m=np.zeros((1, 1, 3)),
            orientations_local_to_world=np.diag([1.0, 1.0, -1.0])[None, None],
        )


@pytest.mark.parametrize("invalid_time", ["0.1", 0.1 + 0.0j, True])
def test_episode_and_pose_time_axes_reject_coercible_nonreal_values(invalid_time):
    with pytest.raises(TypeError, match="real numeric timestamps"):
        replace(_episode(), frame_times_s=(invalid_time,))

    timeline = _dynamic_episode().pose_timeline
    assert timeline is not None
    with pytest.raises(TypeError, match="real numeric timestamps"):
        replace(timeline, frame_times_s=(0.0, invalid_time))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        ("port", "invalid TX port map"),
        ("node", "missing network node"),
        ("rf", "RF chain count"),
    ],
)
def test_network_episode_rejects_invalid_port_and_node_contracts(mutate, message):
    episode = _episode()
    if mutate == "port":
        link = DirectedLink(
            "bad-link", "gNB-1", "UE-1", (2,), (0,), LinkDirection.DOWNLINK, LinkRole.SERVING
        )
        arguments = (episode.arrays, episode.nodes, (link,))
    elif mutate == "node":
        link = DirectedLink(
            "bad-link", "missing", "UE-1", (0,), (0,), LinkDirection.DOWNLINK, LinkRole.SERVING
        )
        arguments = (episode.arrays, episode.nodes, (link,))
    else:
        bad_node = NetworkNode("gNB-1", NodeRole.GNB, "gNB-array", (0.0, 0.0, 10.0), 3)
        arguments = (episode.arrays, (bad_node, episode.nodes[1]), episode.links)
    with pytest.raises(ValueError, match=message):
        NetworkEpisode(
            "episode-bad",
            "site-a-session-1",
            "F03-RT-STATIC",
            3.5e9,
            *arguments,
            episode.frame_times_s,
            episode.subcarrier_offsets_hz,
        )


def test_network_episode_rejects_non_rotation_and_bad_frequency_axes():
    with pytest.raises(ValueError, match="proper rotation"):
        AntennaArraySpec(
            "bad",
            np.zeros((1, 3)),
            ("V",),
            np.diag([1.0, 1.0, -1.0]),
            ("port-0",),
            np.ones((1, 1), dtype=np.complex128),
        )
    episode = _episode()
    with pytest.raises(ValueError, match="include DC"):
        NetworkEpisode(
            "bad-frequency",
            episode.split_group_id,
            episode.family_id,
            episode.carrier_hz,
            episode.arrays,
            episode.nodes,
            episode.links,
            episode.frame_times_s,
            np.asarray([1.0, 2.0]),
        )
    with pytest.raises(ValueError, match="positive RF frequencies"):
        NetworkEpisode(
            episode.episode_id,
            episode.split_group_id,
            episode.family_id,
            1.0,
            episode.arrays,
            episode.nodes,
            episode.links,
            episode.frame_times_s,
            np.asarray([-1.0, 0.0, 1.0]),
        )


def test_dynamic_geometric_los_v4_persists_frame_axis(tmp_path):
    bundle = _bundle()
    definition = TargetDefinition.versioned(
        semantic="geometric direct-path visibility",
        units="boolean",
        source="numpy_loaded_scene_triangle_inventory",
        reference=(
            "digest-bound loaded-scene triangle query cross-checked with Mitsuba "
            "segment intersection and depth-zero Sionna PathSolver at each recorded pose"
        ),
        aggregation_rule="one visibility value per trajectory frame; no spatial or temporal pooling",
        definition_version="umcd.f04_rt_dynamic.direct_visibility.v4",
        value_axes=("frame",),
    )
    target = TargetValue(
        entity_id="gNB-1-to-UE-1",
        definition=definition,
        values=np.asarray([1], dtype=np.uint8),
        valid_mask=np.asarray([True]),
        reason_codes=(TargetValidity.VALID,),
        confidence="geometry_verified",
    )
    stored = write_episode_bundle(tmp_path, replace(bundle, targets=(target,)))
    loaded = read_episode_bundle(tmp_path, stored)
    assert loaded.targets[0].definition.value_axes == ("frame",)
    np.testing.assert_array_equal(loaded.targets[0].values, [1])

    wrong_shape = replace(
        target,
        values=np.asarray([1, 0], dtype=np.uint8),
        valid_mask=np.ones(2, dtype=bool),
        reason_codes=(TargetValidity.VALID, TargetValidity.VALID),
        censoring_intervals=(None, None),
    )
    with pytest.raises(ValueError, match="frame-time roster"):
        EpisodeBundle(bundle.episode, None, (), (wrong_shape,))


def test_episode_store_roundtrip_preserves_dynamic_poses(tmp_path):
    bundle = _bundle()
    dynamic_episode = _dynamic_episode()
    assert dynamic_episode.pose_timeline is not None
    aligned_timeline, time_alignment = dynamic_episode.pose_timeline.align_to(
        dynamic_episode.frame_times_s, tolerance_s=0.0
    )
    source_alignment = align_named_time_axes(
        dynamic_episode.frame_times_s,
        {
            "tx_clock": dynamic_episode.frame_times_s,
            "rx_clock": (0.0001, 0.1001),
            "object_pose_clock": dynamic_episode.frame_times_s,
        },
        tolerance_s=0.001,
        require_complete=True,
    )
    dynamic_episode = replace(
        dynamic_episode,
        pose_timeline=aligned_timeline,
        pose_time_alignment=time_alignment,
        source_time_alignment=source_alignment,
    )
    channel = replace(
        bundle.channel_views[0],
        channel=np.repeat(bundle.channel_views[0].channel, 2, axis=0),
        frame_times_s=(0.0, 0.1),
        valid_mask=np.repeat(bundle.channel_views[0].valid_mask, 2, axis=0),
        source_time_alignment_sha256=source_alignment.sha256,
    )
    dynamic_paths = PathOracleTable(
        link_ids=("gNB-1-to-UE-1",) * 4,
        frame_indices=np.asarray([0, 0, 1, 1]),
        rx_port_indices=np.zeros(4, dtype=np.int64),
        tx_port_indices=np.asarray([0, 1, 0, 1]),
        path_offsets=np.asarray([0, 1, 2, 3, 4]),
        solve_status=(PathSolveStatus.SOLVED,) * 4,
        delay_s=np.asarray([10e-9, 12e-9, 10e-9, 12e-9]),
        complex_gain=np.asarray([0.01 + 0.02j, 0.02 - 0.01j, 0.01 + 0.02j, 0.02 - 0.01j]),
        interaction_order=np.ones(4, dtype=np.int64),
        interaction_offsets=np.arange(5, dtype=np.int64),
        interaction_types=np.ones(4, dtype=np.int64),
        object_ids=np.asarray([7, 8, 7, 8]),
        primitive_ids=np.asarray([3, 4, 3, 4]),
        interaction_vertices_m=np.asarray([[1.0, 2.0, 3.0], [2.0, 2.0, 3.0]] * 2),
        interaction_vertex_frame="test_scene_xyz_m",
    )
    with pytest.raises(ValueError, match="must match the episode alignment"):
        replace(channel, source_time_alignment_sha256="0" * 64).validate_for(dynamic_episode)
    dynamic_bundle = replace(
        bundle,
        episode=dynamic_episode,
        path_oracle=dynamic_paths,
        channel_views=(channel,),
    )

    stored = write_episode_bundle(tmp_path, dynamic_bundle)
    loaded = read_episode_bundle(tmp_path, stored)

    assert loaded.episode.pose_timeline is not None
    assert loaded.episode.pose_time_alignment == time_alignment
    assert loaded.episode.source_time_alignment == source_alignment
    assert loaded.channel_views[0].source_time_alignment_sha256 == source_alignment.sha256
    jsonschema = pytest.importorskip("jsonschema")
    alignment_schema = json.loads(
        (
            Path(__file__).parents[2] / "datasets/schema/time-axis-alignment.v1.schema.json"
        ).read_text()
    )
    jsonschema.Draft202012Validator(alignment_schema).validate(
        loaded.episode.pose_time_alignment.to_dict()
    )
    multi_alignment_schema = json.loads(
        (
            Path(__file__).parents[2]
            / "datasets/schema/umcd-multi-source-time-alignment.v1.schema.json"
        ).read_text()
    )
    jsonschema.Draft202012Validator(multi_alignment_schema).validate(
        loaded.episode.source_time_alignment.to_dict()
    )
    with np.load(
        io.BytesIO(_channel_payload(loaded.channel_views[0])), allow_pickle=False
    ) as archive:
        channel_metadata = json.loads(archive["metadata_json"].tobytes())
    channel_schema = json.loads(
        (
            Path(__file__).parents[2] / "datasets/schema/umcd-physical-channel-layer.v1.schema.json"
        ).read_text()
    )
    jsonschema.Draft202012Validator(channel_schema).validate(channel_metadata)
    np.testing.assert_array_equal(
        loaded.episode.pose_timeline.position("UE-1", 1), [22.0, 3.0, 1.5]
    )
    np.testing.assert_array_equal(
        loaded.episode.pose_timeline.orientation("UE-1", 1),
        dynamic_bundle.episode.pose_timeline.orientation("UE-1", 1),
    )


def test_legacy_episode_payload_without_pose_alignment_remains_readable():
    episode = _dynamic_episode()
    payload = _episode_payload(episode)
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        metadata = json.loads(archive["metadata_json"].tobytes())
        arrays = {
            name: np.array(archive[name], copy=True)
            for name in archive.files
            if name != "metadata_json"
        }
    metadata.pop("pose_time_alignment")

    loaded = _read_episode(metadata, arrays)

    assert loaded.pose_timeline is not None
    assert loaded.pose_time_alignment is None
