"""Contract tests for the first multi-node UMCD schema layer."""

from __future__ import annotations

import json

import numpy as np
import pytest
from datasets.channel_view import (
    ChannelDelayReference,
    ChannelPowerScale,
    PhysicalChannelView,
)
from datasets.episode_store import (
    EpisodeBundle,
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
    NodeRole,
)
from datasets.path_oracle import PathOracleTable, PathSolveStatus
from datasets.targets import TargetDefinition, TargetValidity, TargetValue, definition_digest


def _episode() -> NetworkEpisode:
    arrays = (
        AntennaArraySpec(
            "gNB-array",
            np.asarray([[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.5, 0.0, 0.0], [0.5, 0.5, 0.0]]),
            ("V", "V", "H", "H"),
            np.eye(3),
            ("gNB-V", "gNB-H"),
            np.asarray(
                [[2**-0.5, 2**-0.5, 0.0, 0.0], [0.0, 0.0, 2**-0.5, 2**-0.5]],
                dtype=np.complex128,
            ),
        ),
        AntennaArraySpec(
            "UE-array",
            np.asarray([[0.0, 0.0, 0.0]]),
            ("V",),
            np.eye(3),
            ("UE-V",),
            np.ones((1, 1), dtype=np.complex128),
        ),
    )
    nodes = (
        NetworkNode("gNB-1", NodeRole.GNB, "gNB-array", (0.0, 0.0, 10.0), 2),
        NetworkNode("UE-1", NodeRole.UE, "UE-array", (20.0, 0.0, 1.5), 1),
    )
    link = DirectedLink(
        "gNB-1-to-UE-1",
        "gNB-1",
        "UE-1",
        (0, 1),
        (0,),
        LinkDirection.DOWNLINK,
        LinkRole.SERVING,
    )
    return NetworkEpisode(
        "episode-1",
        "site-a-session-1",
        "F03-RT-STATIC",
        3.5e9,
        arrays,
        nodes,
        (link,),
        (0.0,),
        np.asarray([-30_000.0, 0.0, 30_000.0]),
    )


def test_network_episode_preserves_ordered_arrays_links_and_grid():
    episode = _episode()
    assert episode.node("gNB-1").rf_chain_count == 2
    assert episode.array("gNB-array").num_elements == 4
    assert episode.array("gNB-array").num_ports == 2
    assert episode.array("gNB-array").polarization_per_element == ("V", "V", "H", "H")
    assert episode.link("gNB-1-to-UE-1").tx_port_indices == (0, 1)
    assert not episode.subcarrier_offsets_hz.flags.writeable


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


def test_ragged_path_oracle_keeps_no_path_distinct_from_solver_failure():
    paths = PathOracleTable(
        link_ids=("gNB-1-to-UE-1", "gNB-1-to-UE-1"),
        frame_indices=np.asarray([0, 0]),
        rx_port_indices=np.asarray([0, 0]),
        tx_port_indices=np.asarray([0, 1]),
        path_offsets=np.asarray([0, 2, 2]),
        solve_status=(PathSolveStatus.SOLVED, PathSolveStatus.NO_PATH),
        delay_s=np.asarray([10e-9, 15e-9]),
        complex_gain=np.asarray([0.1 + 0.2j, 0.03 - 0.01j]),
        interaction_order=np.asarray([0, 1]),
        doppler_hz=np.asarray([0.0, 14.0]),
        aoa_azimuth_elevation_rad=np.asarray([[0.0, 0.0], [0.4, -0.1]]),
        aod_azimuth_elevation_rad=np.asarray([[np.pi, 0.0], [-0.4, 0.1]]),
        path_kind=("direct", "reflection"),
    )

    paths.validate_for(_episode())
    assert paths.group_count == 2
    assert paths.path_count == 2
    assert paths.group_slice(0) == slice(0, 2)
    assert paths.group_slice(1) == slice(2, 2)
    assert paths.solve_status[1] is PathSolveStatus.NO_PATH
    assert not paths.delay_s.flags.writeable


@pytest.mark.parametrize(
    ("offsets", "statuses", "path_count", "message"),
    [
        ([0, 1, 2], [PathSolveStatus.SOLVED], 2, "path_offsets must have"),
        ([0, 0], [PathSolveStatus.SOLVED], 0, "must contain at least one path"),
        ([0, 1], [PathSolveStatus.NO_PATH], 1, "must not contain fabricated"),
        ([0, 1], [PathSolveStatus.SOLVER_FAILURE], 1, "must not contain fabricated"),
    ],
)
def test_ragged_path_table_rejects_inconsistent_outcome_and_offsets(
    offsets, statuses, path_count, message
):
    with pytest.raises(ValueError, match=message):
        PathOracleTable(
            link_ids=("gNB-1-to-UE-1",),
            frame_indices=np.asarray([0]),
            rx_port_indices=np.asarray([0]),
            tx_port_indices=np.asarray([0]),
            path_offsets=np.asarray(offsets),
            solve_status=tuple(statuses),
            delay_s=np.full(path_count, 1e-9),
            complex_gain=np.full(path_count, 1.0 + 0j),
            interaction_order=np.zeros(path_count, dtype=np.int64),
        )


def test_ragged_path_table_validates_link_frame_and_port_references():
    episode = _episode()
    paths = PathOracleTable(
        link_ids=("missing",),
        frame_indices=np.asarray([0]),
        rx_port_indices=np.asarray([0]),
        tx_port_indices=np.asarray([0]),
        path_offsets=np.asarray([0, 1]),
        solve_status=(PathSolveStatus.SOLVED,),
        delay_s=np.asarray([1e-9]),
        complex_gain=np.asarray([1.0 + 0j]),
        interaction_order=np.asarray([0]),
    )
    with pytest.raises(ValueError, match="missing episode link"):
        paths.validate_for(episode)


def test_path_interaction_trace_rejects_partial_or_misaligned_geometry():
    common = {
        "link_ids": ("link-1",),
        "frame_indices": np.asarray([0]),
        "rx_port_indices": np.asarray([0]),
        "tx_port_indices": np.asarray([0]),
        "path_offsets": np.asarray([0, 1]),
        "solve_status": (PathSolveStatus.SOLVED,),
        "delay_s": np.asarray([2e-9]),
        "complex_gain": np.asarray([0.25 + 0.1j]),
        "interaction_order": np.asarray([1]),
    }
    interactions = {
        "interaction_offsets": np.asarray([0, 1]),
        "interaction_types": np.asarray([1]),
        "object_ids": np.asarray([7]),
        "primitive_ids": np.asarray([3]),
        "interaction_vertices_m": np.asarray([[1.0, 2.0, 3.0]]),
        "interaction_vertex_frame": "test_scene_xyz_m",
    }
    with pytest.raises(ValueError, match="must be supplied together"):
        PathOracleTable(**common, interaction_offsets=interactions["interaction_offsets"])
    with pytest.raises(ValueError, match="agree with per-path interaction_order"):
        PathOracleTable(
            **common,
            **{
                **interactions,
                "interaction_offsets": np.asarray([0, 0]),
            },
        )
    with pytest.raises(ValueError, match=r"finite shape \[interaction, xyz\]"):
        PathOracleTable(
            **common,
            **{
                **interactions,
                "interaction_vertices_m": np.asarray([[1.0, np.nan, 3.0]]),
            },
        )


def test_physical_channel_view_requires_named_axes_and_explicit_normalization():
    episode = _episode()
    arguments = {
        "view_id": "oracle-cfr-1",
        "link_id": "gNB-1-to-UE-1",
        "channel": np.ones((1, 1, 2, 3), dtype=np.complex64),
        "frame_times_s": episode.frame_times_s,
        "rx_port_ids": ("UE-V",),
        "tx_port_ids": ("gNB-V", "gNB-H"),
        "carrier_hz": episode.carrier_hz,
        "subcarrier_offsets_hz": episode.subcarrier_offsets_hz,
        "delay_reference": ChannelDelayReference.ABSOLUTE_PROPAGATION,
        "power_scale": ChannelPowerScale.ABSOLUTE,
    }
    channel = PhysicalChannelView(**arguments)
    channel.validate_for(episode)
    assert channel.axes == ("frame", "rx_port", "tx_port", "subcarrier")
    assert not channel.channel.flags.writeable
    with pytest.raises(ValueError, match="normalization_recipe"):
        PhysicalChannelView(
            **{
                **arguments,
                "power_scale": ChannelPowerScale.NORMALIZED,
            }
        )


def test_target_values_require_definition_hash_and_reasoned_element_masks():
    definition = TargetDefinition(
        semantic="rms_delay_spread",
        units="s",
        source="path_oracle",
        reference="absolute_physical_channel_path_oracle",
        definition_sha256=definition_digest("rms-delay-spread:v1"),
    )
    target = TargetValue(
        entity_id="link-1",
        definition=definition,
        values=np.asarray([12e-9, 0.0]),
        valid_mask=np.asarray([True, False]),
        reason_codes=(TargetValidity.VALID, TargetValidity.NOT_OBSERVABLE),
        confidence="rt_solver_truth",
    )
    assert target.valid_count == 1
    assert target.unavailable_counts == {TargetValidity.NOT_OBSERVABLE: 1}
    with pytest.raises(ValueError, match="masked target element"):
        TargetValue(
            entity_id="link-1",
            definition=definition,
            values=np.asarray([12e-9, 3e-9]),
            valid_mask=np.asarray([True, False]),
            reason_codes=(TargetValidity.VALID, TargetValidity.NOT_OBSERVABLE),
            confidence="rt_solver_truth",
        )


def _bundle() -> EpisodeBundle:
    episode = _episode()
    paths = PathOracleTable(
        link_ids=("gNB-1-to-UE-1",),
        frame_indices=np.asarray([0]),
        rx_port_indices=np.asarray([0]),
        tx_port_indices=np.asarray([0]),
        path_offsets=np.asarray([0, 1]),
        solve_status=(PathSolveStatus.SOLVED,),
        delay_s=np.asarray([10e-9]),
        complex_gain=np.asarray([0.01 + 0.02j]),
        interaction_order=np.asarray([1]),
        interaction_offsets=np.asarray([0, 1]),
        interaction_types=np.asarray([1]),
        object_ids=np.asarray([7]),
        primitive_ids=np.asarray([3]),
        interaction_vertices_m=np.asarray([[1.0, 2.0, 3.0]]),
        interaction_vertex_frame="test_scene_xyz_m",
    )
    channel = PhysicalChannelView(
        view_id="oracle-cfr-v1",
        link_id="gNB-1-to-UE-1",
        channel=np.ones((1, 1, 2, 3), dtype=np.complex64),
        frame_times_s=episode.frame_times_s,
        rx_port_ids=("UE-V",),
        tx_port_ids=("gNB-V", "gNB-H"),
        carrier_hz=episode.carrier_hz,
        subcarrier_offsets_hz=episode.subcarrier_offsets_hz,
        delay_reference=ChannelDelayReference.ABSOLUTE_PROPAGATION,
        power_scale=ChannelPowerScale.ABSOLUTE,
    )
    definition = TargetDefinition(
        "geometric_direct_path",
        "boolean",
        "path_geometry_oracle",
        "3d_segment_visibility",
        definition_digest("geometric_direct_path:v1"),
    )
    target = TargetValue(
        "gNB-1-to-UE-1",
        definition,
        np.asarray([1.0]),
        np.asarray([True]),
        (TargetValidity.VALID,),
        "geometry_verified",
    )
    return EpisodeBundle(episode, paths, (channel,), (target,))


def test_episode_bundle_roundtrip_keeps_layers_content_addressed_and_pickle_free(tmp_path):
    bundle = _bundle()
    stored = write_episode_bundle(tmp_path, bundle)
    loaded = read_episode_bundle(tmp_path, stored)

    assert loaded.episode.episode_id == bundle.episode.episode_id
    np.testing.assert_array_equal(loaded.path_oracle.complex_gain, bundle.path_oracle.complex_gain)
    np.testing.assert_array_equal(
        loaded.path_oracle.interaction_offsets, bundle.path_oracle.interaction_offsets
    )
    np.testing.assert_array_equal(loaded.path_oracle.object_ids, bundle.path_oracle.object_ids)
    np.testing.assert_array_equal(
        loaded.path_oracle.interaction_vertices_m, bundle.path_oracle.interaction_vertices_m
    )
    assert loaded.path_oracle.interaction_vertex_frame == "test_scene_xyz_m"
    np.testing.assert_array_equal(loaded.channel_views[0].channel, bundle.channel_views[0].channel)
    np.testing.assert_array_equal(loaded.targets[0].valid_mask, bundle.targets[0].valid_mask)
    assert (tmp_path / "objects/oracle_paths").is_dir()
    assert (tmp_path / "objects/oracle_channels").is_dir()
    assert (tmp_path / "objects/targets").is_dir()
    manifest = json.loads((tmp_path / stored.relative_path).read_text())
    assert {item["layer"] for item in manifest["layers"]} == {
        "episode",
        "oracle_paths",
        "oracle_channels",
        "targets",
    }


def test_episode_bundle_rejects_tampered_content_and_immutable_id_reuse(tmp_path):
    bundle = _bundle()
    stored = write_episode_bundle(tmp_path, bundle)
    with pytest.raises(FileExistsError):
        write_episode_bundle(
            tmp_path,
            EpisodeBundle(
                bundle.episode,
                bundle.path_oracle,
                (
                    PhysicalChannelView(
                        **{
                            **{
                                name: getattr(bundle.channel_views[0], name)
                                for name in (
                                    "view_id",
                                    "link_id",
                                    "channel",
                                    "frame_times_s",
                                    "rx_port_ids",
                                    "tx_port_ids",
                                    "carrier_hz",
                                    "subcarrier_offsets_hz",
                                    "delay_reference",
                                    "power_scale",
                                )
                            },
                            "channel": np.zeros_like(bundle.channel_views[0].channel),
                        }
                    ),
                ),
                bundle.targets,
            ),
        )
    manifest = json.loads((tmp_path / stored.relative_path).read_text())
    path_ref = next(item for item in manifest["layers"] if item["layer"] == "oracle_paths")
    path_object = tmp_path / path_ref["path"]
    path_object.write_bytes(path_object.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="content hash"):
        read_episode_bundle(tmp_path, stored)
