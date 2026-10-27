"""Contract tests for the first multi-node UMCD schema layer."""

from __future__ import annotations

import hashlib
import io
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from datasets.channel_view import (
    ChannelDelayReference,
    ChannelPowerScale,
    PhysicalChannelView,
)
from datasets.counterfactual import remove_direct_path_component
from datasets.episode_store import (
    EpisodeBundle,
    _episode_payload,
    _path_payload,
    _read_episode,
    _validate_path_oracle_layer,
    read_episode_bundle,
    read_frequency_model_inputs,
    read_receiver_model_inputs,
    write_episode_bundle,
)
from datasets.lineage import (
    LineageStatusClaim,
    audit_lineage_evidence_files,
    audit_lineage_evidence_payloads,
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
from datasets.nr import LinkedNRObservation, NRObservation
from datasets.observations import (
    FrequencyDomainModelInputs,
    FrequencyDomainObservation,
    LinkedFrequencyDomainObservation,
)
from datasets.path_oracle import PathOracleTable, PathSolveStatus
from datasets.provenance import UMCDRunProvenance
from datasets.split_policy import (
    SplitChildBinding,
    SplitParentType,
    SplitPolicy,
    SplitSubject,
    bind_split_children,
    make_split_policy_registry,
)
from datasets.targets import (
    TargetDefinition,
    TargetValidity,
    TargetValue,
    migrate_legacy_target_definition_v1,
    target_definition_digest,
)
from receivers.nr import NRChannelEstimate, NRPUSCHConfig


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
    assert episode.position_at("UE-1", 0) == (20.0, 0.0, 1.5)
    np.testing.assert_array_equal(episode.orientation_at("gNB-1", 0), np.eye(3))


def _dynamic_episode() -> NetworkEpisode:
    episode = _episode()
    positions = np.asarray(
        [
            [node.position_world_m for node in episode.nodes],
            [[0.0, 0.0, 10.0], [22.0, 3.0, 1.5]],
        ]
    )
    orientations = np.broadcast_to(np.eye(3), (2, 2, 3, 3)).copy()
    yaw_90 = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    orientations[1, 1] = yaw_90
    timeline = NodePoseTimeline(
        node_ids=tuple(node.node_id for node in episode.nodes),
        frame_times_s=(0.0, 0.1),
        positions_world_m=positions,
        orientations_local_to_world=orientations,
    )
    return replace(episode, frame_times_s=(0.0, 0.1), pose_timeline=timeline)


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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("aoa_azimuth_elevation_rad", np.asarray([[True, False]])),
        ("aod_azimuth_elevation_rad", np.asarray([["0.0", "0.0"]])),
        ("aoa_azimuth_elevation_rad", np.asarray([[0.0, 0.0]], dtype=object)),
        ("aod_azimuth_elevation_rad", np.asarray([[0.0 + 1j, 0.0 + 0j]])),
        ("interaction_vertices_m", np.asarray([[True, False, True]])),
        ("interaction_vertices_m", np.asarray([["1", "2", "3"]])),
        ("interaction_vertices_m", np.asarray([[1.0, 2.0, 3.0]], dtype=object)),
        ("interaction_vertices_m", np.asarray([[1.0 + 1j, 2.0, 3.0]])),
    ],
)
def test_path_oracle_rejects_non_real_numeric_angle_and_vertex_dtypes(field, value):
    arguments = {
        "link_ids": ("link-1",),
        "frame_indices": np.asarray([0]),
        "rx_port_indices": np.asarray([0]),
        "tx_port_indices": np.asarray([0]),
        "path_offsets": np.asarray([0, 1]),
        "solve_status": (PathSolveStatus.SOLVED,),
        "delay_s": np.asarray([1e-9]),
        "complex_gain": np.asarray([1.0 + 0j]),
        "interaction_order": np.asarray([0 if field != "interaction_vertices_m" else 1]),
    }
    if field == "interaction_vertices_m":
        arguments.update(
            {
                "interaction_offsets": np.asarray([0, 1]),
                "interaction_types": np.asarray([1]),
                "object_ids": np.asarray([0]),
                "primitive_ids": np.asarray([0]),
                "interaction_vertex_frame": "scene_xyz_m",
            }
        )

    with pytest.raises(TypeError, match="real numeric dtype"):
        PathOracleTable(**arguments, **{field: value})


@pytest.mark.parametrize("field", ("frame_indices", "path_offsets", "interaction_types"))
def test_path_oracle_rejects_integer_values_that_overflow_int64(field):
    arguments = {
        "link_ids": ("link-1",),
        "frame_indices": np.asarray([0], dtype=np.uint64),
        "rx_port_indices": np.asarray([0], dtype=np.uint64),
        "tx_port_indices": np.asarray([0], dtype=np.uint64),
        "path_offsets": np.asarray([0, 1], dtype=np.uint64),
        "solve_status": (PathSolveStatus.SOLVED,),
        "delay_s": np.asarray([1e-9]),
        "complex_gain": np.asarray([1.0 + 0j]),
        "interaction_order": np.asarray([0], dtype=np.uint64),
    }
    if field == "interaction_types":
        arguments.update(
            {
                "interaction_order": np.asarray([1], dtype=np.uint64),
                "interaction_offsets": np.asarray([0, 1], dtype=np.uint64),
                "interaction_types": np.asarray([1], dtype=np.uint64),
                "object_ids": np.asarray([0], dtype=np.uint64),
                "primitive_ids": np.asarray([0], dtype=np.uint64),
                "interaction_vertices_m": np.asarray([[0.0, 0.0, 0.0]]),
                "interaction_vertex_frame": "scene_xyz_m",
            }
        )
    arguments[field] = np.asarray([0, 2**63], dtype=np.uint64) if field == "path_offsets" else np.asarray([2**63], dtype=np.uint64)

    with pytest.raises(ValueError, match="fit int64"):
        PathOracleTable(**arguments)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"link_ids": (" link-1",)}, "canonical link IDs"),
        ({"path_kind": (" specular ",)}, "canonical value per path"),
        ({"coefficient_convention": " sionna_rt_path_coefficient_v1"}, "canonical string"),
    ],
)
def test_path_oracle_rejects_padded_join_and_provenance_identities(change, message):
    arguments = {
        "link_ids": ("link-1",),
        "frame_indices": np.asarray([0]),
        "rx_port_indices": np.asarray([0]),
        "tx_port_indices": np.asarray([0]),
        "path_offsets": np.asarray([0, 1]),
        "solve_status": (PathSolveStatus.SOLVED,),
        "delay_s": np.asarray([1e-9]),
        "complex_gain": np.asarray([1.0 + 0j]),
        "interaction_order": np.asarray([0]),
    }
    with pytest.raises(ValueError, match=message):
        PathOracleTable(**{**arguments, **change})


def test_path_oracle_rejects_padded_interaction_coordinate_frame():
    with pytest.raises(ValueError, match="canonical named coordinate frame"):
        PathOracleTable(
            link_ids=("link-1",),
            frame_indices=np.asarray([0]),
            rx_port_indices=np.asarray([0]),
            tx_port_indices=np.asarray([0]),
            path_offsets=np.asarray([0, 1]),
            solve_status=(PathSolveStatus.SOLVED,),
            delay_s=np.asarray([1e-9]),
            complex_gain=np.asarray([1.0 + 0j]),
            interaction_order=np.asarray([1]),
            interaction_offsets=np.asarray([0, 1]),
            interaction_types=np.asarray([1]),
            object_ids=np.asarray([0]),
            primitive_ids=np.asarray([0]),
            interaction_vertices_m=np.asarray([[0.0, 0.0, 0.0]]),
            interaction_vertex_frame=" scene_xyz_m ",
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
    assert channel.valid_mask is not None and channel.valid_mask.all()
    assert not channel.valid_mask.flags.writeable
    with pytest.raises(ValueError, match="masked channel elements"):
        PhysicalChannelView(
            **{
                **arguments,
                "valid_mask": np.zeros_like(arguments["channel"], dtype=np.bool_),
            }
        )
    invalid_zero_channel = PhysicalChannelView(
        **{
            **arguments,
            "channel": np.zeros_like(arguments["channel"]),
            "valid_mask": np.zeros_like(arguments["channel"], dtype=np.bool_),
        }
    )
    assert invalid_zero_channel.valid_mask is not None
    assert not invalid_zero_channel.valid_mask.any()
    with pytest.raises(ValueError, match="normalization_recipe"):
        PhysicalChannelView(
            **{
                **arguments,
                "power_scale": ChannelPowerScale.NORMALIZED,
            }
        )


def test_physical_channel_view_rejects_coercible_timestamp_strings():
    channel = _bundle().channel_views[0]
    with pytest.raises(TypeError, match="real numeric timestamps"):
        replace(channel, frame_times_s=("0.0",))


@pytest.mark.parametrize(
    "frequencies",
    (
        np.asarray([True, True, True]),
        np.asarray(["-1", "0", "1"]),
        np.asarray([-1.0, 0.0, 1.0], dtype=object),
        np.asarray([-1.0 + 0j, 0.0 + 0j, 1.0 + 0j]),
    ),
)
def test_physical_channel_view_rejects_coercible_non_real_frequency_axes(frequencies):
    channel = _bundle().channel_views[0]
    with pytest.raises(TypeError, match="real numeric dtype"):
        replace(channel, subcarrier_offsets_hz=frequencies)


@pytest.mark.parametrize("carrier", (True, "3.5e9", 3.5e9 + 0j))
def test_physical_channel_view_rejects_non_real_carrier_scalars(carrier):
    channel = _bundle().channel_views[0]
    with pytest.raises(TypeError, match="real numeric scalar"):
        replace(channel, carrier_hz=carrier)


def test_target_values_require_definition_hash_and_reasoned_element_masks():
    fields = {
        "semantic": "rms_delay_spread",
        "units": "s",
        "source": "path_oracle",
        "reference": "absolute_physical_channel_path_oracle",
        "aggregation_rule": "RMS over valid paths after power normalization",
        "definition_version": "umcd.rms_delay_spread.v1",
    }
    definition = TargetDefinition(
        **fields,
        definition_sha256=target_definition_digest(**fields),
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


@pytest.mark.parametrize(
    "changed_field,changed_value",
    [
        ("semantic", "rms_delay_spread_changed"),
        ("units", "ns"),
        ("source", "different_oracle"),
        ("reference", "different_gauge"),
        ("aggregation_rule", "maximum over paths"),
        ("definition_version", "umcd.rms_delay_spread.v2"),
    ],
)
def test_target_definition_digest_binds_every_canonical_field(changed_field, changed_value):
    fields = {
        "semantic": "rms_delay_spread",
        "units": "s",
        "source": "path_oracle",
        "reference": "absolute_physical_channel_path_oracle",
        "aggregation_rule": "RMS over valid paths after power normalization",
        "definition_version": "umcd.rms_delay_spread.v1",
    }
    definition = TargetDefinition(
        **fields,
        definition_sha256=target_definition_digest(**fields),
    )
    changed = {**fields, changed_field: changed_value}
    tampered = TargetDefinition(
        **changed,
        definition_sha256=definition.definition_sha256,
    )
    with pytest.raises(ValueError, match="does not match its canonical fields"):
        TargetValue(
            entity_id="link-1",
            definition=tampered,
            values=np.asarray([1.0]),
            valid_mask=np.asarray([True]),
            reason_codes=(TargetValidity.VALID,),
            confidence="verified",
        )


def test_legacy_definition_defaults_remain_constructible_but_not_canonical_targets():
    legacy = TargetDefinition(semantic="legacy_label", units="class", source="legacy_source")
    assert legacy.reference == "unspecified_legacy"
    assert legacy.aggregation_rule == "unspecified_legacy"
    assert legacy.definition_version == "unversioned_legacy"
    with pytest.raises(ValueError, match="SHA-256 digest"):
        TargetValue(
            entity_id="link-1",
            definition=legacy,
            values=np.asarray([1.0]),
            valid_mask=np.asarray([True]),
            reason_codes=(TargetValidity.VALID,),
            confidence="legacy",
        )


def test_v1_target_definition_migrates_with_explicitly_unknown_contract_fields():
    migrated = migrate_legacy_target_definition_v1(
        {
            "semantic": "legacy_delay",
            "units": "s",
            "source": "old_path_solver",
            "reference": "unspecified_legacy",
            "definition_sha256": "old-format-digest",
        }
    )

    assert migrated.reference == "not recorded by legacy target schema v1"
    assert migrated.aggregation_rule == "not recorded by legacy target schema v1"
    assert migrated.definition_version == "migrated_from_legacy_target_v1"
    migrated.require_versioned()


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
        semantic="geometric_direct_path",
        units="boolean",
        source="path_geometry_oracle",
        reference="3d_segment_visibility",
        aggregation_rule="one visibility value per link and frame; no pooling",
        definition_version="umcd.geometric_direct_path.v1",
        definition_sha256=target_definition_digest(
            semantic="geometric_direct_path",
            units="boolean",
            source="path_geometry_oracle",
            reference="3d_segment_visibility",
            aggregation_rule="one visibility value per link and frame; no pooling",
            definition_version="umcd.geometric_direct_path.v1",
        ),
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


def test_frame_visibility_target_matches_episode_timestamp_roster():
    episode = replace(_episode(), frame_times_s=(0.0, 0.01, 0.02))
    definition = TargetDefinition.versioned(
        semantic="geometric direct-path visibility",
        units="boolean",
        source="numpy_loaded_scene_triangle_inventory",
        reference=(
            "digest-bound loaded-scene triangle query cross-checked with Mitsuba "
            "segment intersection and depth-zero Sionna PathSolver at each recorded pose"
        ),
        aggregation_rule="one visibility value per trajectory frame; no spatial or temporal pooling",
        definition_version="umcd.f04_rt_dynamic.direct_visibility.v3",
    )
    target = TargetValue(
        entity_id="gNB-1-to-UE-1",
        definition=definition,
        values=np.asarray([1, 0, 1], dtype=np.uint8),
        valid_mask=np.ones(3, dtype=np.bool_),
        reason_codes=(TargetValidity.VALID,) * 3,
        confidence="geometry_verified",
    )
    bundle = EpisodeBundle(episode, None, (), (target,))
    assert len(bundle.targets[0].values) == len(episode.frame_times_s)

    truncated = TargetValue(
        entity_id=target.entity_id,
        definition=definition,
        values=np.asarray([1, 0], dtype=np.uint8),
        valid_mask=np.ones(2, dtype=np.bool_),
        reason_codes=(TargetValidity.VALID,) * 2,
        confidence="geometry_verified",
    )
    with pytest.raises(ValueError, match="frame-time roster"):
        EpisodeBundle(episode, None, (), (truncated,))


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
    assert loaded.targets[0].definition.aggregation_rule == (
        "one visibility value per link and frame; no pooling"
    )
    assert loaded.targets[0].definition.definition_version == "umcd.geometric_direct_path.v1"
    assert loaded.targets[0].definition.definition_sha256 == (
        bundle.targets[0].definition.definition_sha256
    )
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


def test_path_oracle_storage_metadata_schema_and_exact_array_roster():
    payload = _path_payload(_bundle().path_oracle)
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        metadata = json.loads(archive["metadata_json"].tobytes())
        array_names = set(archive.files) - {"metadata_json"}

    schema_path = (
        Path(__file__).resolve().parents[2]
        / "datasets/schema/umcd-path-oracle-layer.v1.schema.json"
    )
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(metadata)
    _validate_path_oracle_layer(metadata, array_names)

    repeated_link_groups = {
        **metadata,
        "link_ids": [metadata["link_ids"][0], metadata["link_ids"][0]],
        "solve_status": [metadata["solve_status"][0], metadata["solve_status"][0]],
    }
    jsonschema.Draft202012Validator(schema).validate(repeated_link_groups)
    _validate_path_oracle_layer(repeated_link_groups, array_names)

    extra_metadata = {**metadata, "unrecognized": True}
    with pytest.raises(ValueError, match="metadata fields"):
        _validate_path_oracle_layer(extra_metadata, array_names)
    with pytest.raises(ValueError, match="arrays do not match"):
        _validate_path_oracle_layer(metadata, array_names - {"delay_s"})
    with pytest.raises(ValueError, match="arrays do not match"):
        _validate_path_oracle_layer(metadata, array_names | {"unexpected"})


def test_episode_bundle_roundtrip_accepts_repeated_path_link_ids_by_port(tmp_path):
    bundle = _bundle()
    paths = PathOracleTable(
        link_ids=("gNB-1-to-UE-1", "gNB-1-to-UE-1"),
        frame_indices=np.asarray([0, 0]),
        rx_port_indices=np.asarray([0, 0]),
        tx_port_indices=np.asarray([0, 1]),
        path_offsets=np.asarray([0, 1, 2]),
        solve_status=(PathSolveStatus.SOLVED, PathSolveStatus.SOLVED),
        delay_s=np.asarray([10e-9, 11e-9]),
        complex_gain=np.asarray([0.01 + 0.02j, 0.02 + 0.01j]),
        interaction_order=np.asarray([1, 1]),
        interaction_offsets=np.asarray([0, 1, 2]),
        interaction_types=np.asarray([1, 1]),
        object_ids=np.asarray([7, 7]),
        primitive_ids=np.asarray([3, 4]),
        interaction_vertices_m=np.asarray([[1.0, 2.0, 3.0], [2.0, 2.0, 3.0]]),
        interaction_vertex_frame="test_scene_xyz_m",
    )
    stored = write_episode_bundle(tmp_path, replace(bundle, path_oracle=paths))
    loaded = read_episode_bundle(tmp_path, stored)

    assert loaded.path_oracle.link_ids == paths.link_ids
    np.testing.assert_array_equal(loaded.path_oracle.tx_port_indices, [0, 1])


def test_episode_bundle_roundtrips_split_bound_direct_path_ablation(tmp_path):
    bundle = _bundle()
    paths = PathOracleTable(
        link_ids=("gNB-1-to-UE-1",),
        frame_indices=np.asarray([0]),
        rx_port_indices=np.asarray([0]),
        tx_port_indices=np.asarray([0]),
        path_offsets=np.asarray([0, 2]),
        solve_status=(PathSolveStatus.SOLVED,),
        delay_s=np.asarray([10e-9, 20e-9]),
        complex_gain=np.asarray([0.7 + 0.2j, 0.1 - 0.3j]),
        interaction_order=np.asarray([0, 1]),
    )
    policy = SplitPolicy(
        seed="episode-ablation-store",
        split_fractions={"train": 1 / 3, "validation": 1 / 3, "test": 1 / 3},
    )
    subjects = [
        SplitSubject(
            subject_id="episode-1",
            parent_type=SplitParentType.NETWORK_EPISODE,
            source_family="rt.city",
        ),
        *[
            SplitSubject(
                subject_id=f"independent-{index}",
                parent_type=SplitParentType.NETWORK_EPISODE,
                source_family="rt.city",
            )
            for index in range(2)
        ],
    ]
    registry = bind_split_children(
        make_split_policy_registry(subjects, policy=policy),
        [
            SplitChildBinding("base-view", "episode-1", "counterfactual_base"),
            SplitChildBinding("minus-direct-view", "episode-1", "counterfactual_math_ablation"),
        ],
    )
    ablation = remove_direct_path_component(
        paths,
        frequency_offset_hz=bundle.episode.subcarrier_offsets_hz,
        registry=registry,
        parent_subject_id="episode-1",
        base_child_id="base-view",
        ablated_child_id="minus-direct-view",
    )
    tampered_ablation = replace(ablation, full_cfr=ablation.full_cfr + 1.0)
    with pytest.raises(ValueError, match="do not replay"):
        replace(bundle, path_oracle=paths, direct_path_ablations=(tampered_ablation,))
    bundle = replace(bundle, path_oracle=paths, direct_path_ablations=(ablation,))

    stored = write_episode_bundle(tmp_path, bundle)
    loaded = read_episode_bundle(tmp_path, stored)

    assert len(loaded.direct_path_ablations) == 1
    actual = loaded.direct_path_ablations[0]
    assert actual.path_table_sha256 == ablation.path_table_sha256
    assert actual.split == ablation.split
    assert actual.link_ids == ("gNB-1-to-UE-1",)
    assert actual.removed_direct_path_count.tolist() == [1]
    np.testing.assert_array_equal(actual.frame_indices, [0])
    np.testing.assert_array_equal(actual.frequency_offset_hz, bundle.episode.subcarrier_offsets_hz)
    np.testing.assert_array_equal(actual.full_cfr, ablation.full_cfr)
    np.testing.assert_array_equal(actual.without_direct_cfr, ablation.without_direct_cfr)
    manifest = json.loads((tmp_path / stored.relative_path).read_text())
    assert "counterfactual_ablations" in {item["layer"] for item in manifest["layers"]}
    jsonschema = pytest.importorskip("jsonschema")
    schema_path = Path(__file__).resolve().parents[2] / "datasets/schema/umcd-bundle-v1.schema.json"
    schema = json.loads(schema_path.read_text())
    jsonschema.Draft202012Validator(schema).validate(manifest)


def test_frequency_receiver_observation_roundtrip_is_separate_and_inference_only(tmp_path):
    bundle = _bundle()
    episode = bundle.episode
    frequency_input = FrequencyDomainModelInputs(
        observed_cfr=np.full((1, 2, 3), 0.5 + 0.25j, dtype=np.complex64),
        noise_variance=0.05,
        snr_db=10.0,
        carrier_hz=episode.carrier_hz,
        subcarrier_offsets_hz=episode.subcarrier_offsets_hz,
    )
    linked = LinkedFrequencyDomainObservation(
        "gNB-1-to-UE-1",
        0,
        FrequencyDomainObservation("direct-cfr-1", frequency_input),
    )
    bundle = replace(bundle, receiver_observations=(linked,))

    stored = write_episode_bundle(tmp_path, bundle)
    loaded = read_episode_bundle(tmp_path, stored)
    inferred = read_frequency_model_inputs(tmp_path, stored)

    assert isinstance(loaded.receiver_observations[0], LinkedFrequencyDomainObservation)
    np.testing.assert_array_equal(inferred[0].observed_cfr, frequency_input.observed_cfr)
    np.testing.assert_array_equal(inferred[0].subcarrier_offsets_hz, episode.subcarrier_offsets_hz)
    assert inferred[0].noise_variance == 0.05
    assert not hasattr(inferred[0], "targets")
    with pytest.raises(ValueError, match="no NR receiver observations"):
        read_receiver_model_inputs(tmp_path, stored)


def test_frequency_receiver_observation_rejects_wrong_port_or_frequency_axes():
    bundle = _bundle()
    episode = bundle.episode
    bad_observation = FrequencyDomainObservation(
        "bad-cfr",
        FrequencyDomainModelInputs(
            observed_cfr=np.ones((1, 1, 4), dtype=np.complex64),
            noise_variance=0.0,
            snr_db=20.0,
            carrier_hz=episode.carrier_hz,
            subcarrier_offsets_hz=np.asarray([-45e3, -15e3, 0.0, 15e3]),
        ),
    )
    with pytest.raises(ValueError, match="axes do not match"):
        replace(
            bundle,
            receiver_observations=(
                LinkedFrequencyDomainObservation("gNB-1-to-UE-1", 0, bad_observation),
            ),
        )


def test_episode_store_roundtrip_preserves_dynamic_poses(tmp_path):
    bundle = _bundle()
    dynamic_episode = _dynamic_episode()
    assert dynamic_episode.pose_timeline is not None
    aligned_timeline, time_alignment = dynamic_episode.pose_timeline.align_to(
        dynamic_episode.frame_times_s, tolerance_s=0.0
    )
    dynamic_episode = replace(
        dynamic_episode,
        pose_timeline=aligned_timeline,
        pose_time_alignment=time_alignment,
    )
    channel = replace(
        bundle.channel_views[0],
        channel=np.repeat(bundle.channel_views[0].channel, 2, axis=0),
        frame_times_s=(0.0, 0.1),
        valid_mask=np.repeat(bundle.channel_views[0].valid_mask, 2, axis=0),
    )
    dynamic_bundle = replace(bundle, episode=dynamic_episode, channel_views=(channel,))

    stored = write_episode_bundle(tmp_path, dynamic_bundle)
    loaded = read_episode_bundle(tmp_path, stored)

    assert loaded.episode.pose_timeline is not None
    assert loaded.episode.pose_time_alignment == time_alignment
    jsonschema = pytest.importorskip("jsonschema")
    alignment_schema = json.loads(
        (Path(__file__).parents[2] / "datasets/schema/time-axis-alignment.v1.schema.json").read_text()
    )
    jsonschema.Draft202012Validator(alignment_schema).validate(
        loaded.episode.pose_time_alignment.to_dict()
    )
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


def _receiver_binding() -> LinkedNRObservation:
    estimate = NRChannelEstimate(
        h_hat=np.ones((1, 1, 1, 1, 1, 14, 4), dtype=np.complex64),
        error_variance=np.zeros((1, 1, 1, 1, 1, 14, 4), dtype=np.float32),
        pilot_mask=np.ones((1, 1, 14, 4), dtype=np.bool_),
        metadata={"estimator": "test"},
    )
    observation = NRObservation(
        sample_id="sample-1",
        received_grid=np.ones((1, 1, 1, 14, 12), dtype=np.complex64),
        channel_estimate=estimate,
        fft_bin_indices=(0, 1, 2, 3),
        receiver_config=NRPUSCHConfig(),
    )
    return LinkedNRObservation("gNB-1-to-UE-1", 0, observation)


def _provenance_record() -> UMCDRunProvenance:
    return UMCDRunProvenance(
        record_id="rt-solve-1",
        subject_layer="oracle_paths",
        source_family="F03-RT-STATIC",
        producer="Sionna RT PathSolver",
        package_source_sha256="a" * 64,
        software_versions={"python": "3.12.3", "sionna-rt": "2.2.0"},
        backend={"backend": "llvm", "variant": "llvm_ad_mono_polarized"},
        scene_asset_sha256={"city.xml": "b" * 64},
        seed=17,
        physical_config={"carrier_hz": 3.5e9},
        solver_config={"max_depth": 3, "deterministic": True},
        material_config={"building-concrete": {"scattering_coefficient": 0.0}},
    )


def test_receiver_view_roundtrip_isolated_from_oracle_and_targets(tmp_path):
    bundle = replace(
        _bundle(),
        receiver_observations=(_receiver_binding(),),
        provenance_records=(_provenance_record(),),
    )
    stored = write_episode_bundle(tmp_path, bundle)

    inference = read_receiver_model_inputs(tmp_path, stored)
    assert len(inference) == 1
    np.testing.assert_array_equal(
        inference[0].received_grid, bundle.receiver_observations[0].observation.received_grid
    )
    np.testing.assert_array_equal(
        inference[0].h_hat, bundle.receiver_observations[0].observation.channel_estimate.h_hat
    )
    assert inference[0].fft_bin_indices == (0, 1, 2, 3)
    assert not hasattr(inference[0], "sample_id")
    assert not hasattr(inference[0], "targets")

    loaded = read_episode_bundle(tmp_path, stored)
    assert loaded.receiver_observations[0].link_id == "gNB-1-to-UE-1"
    assert loaded.receiver_observations[0].observation.sample_id == "sample-1"
    assert loaded.provenance_records[0].record_id == "rt-solve-1"
    assert (
        loaded.provenance_records[0].material_config["building-concrete"]["scattering_coefficient"]
        == 0.0
    )
    manifest = json.loads((tmp_path / stored.relative_path).read_text())
    assert "receiver_observations" in {item["layer"] for item in manifest["layers"]}
    assert "provenance" in {item["layer"] for item in manifest["layers"]}

    oracle_ref = next(item for item in manifest["layers"] if item["layer"] == "oracle_paths")
    oracle_path = tmp_path / oracle_ref["path"]
    oracle_path.write_bytes(oracle_path.read_bytes() + b"tampered")
    # Receiver inference reads only the episode and receiver objects; the full
    # training bundle reader still checks every oracle/target object.
    assert len(read_receiver_model_inputs(tmp_path, stored)) == 1
    with pytest.raises(ValueError, match="content hash"):
        read_episode_bundle(tmp_path, stored)


def test_provenance_record_snapshots_nested_configuration_and_checks_its_hash():
    material_config = {"wall": {"permittivity": 4.2, "roughness": [0.1, 0.2]}}
    record = _provenance_record()
    original = UMCDRunProvenance(
        record_id="material-snapshot",
        subject_layer="oracle_paths",
        source_family="F03-RT-STATIC",
        producer="test",
        package_source_sha256="c" * 64,
        software_versions={"sionna-rt": "2.2.0"},
        backend={"backend": "llvm"},
        scene_asset_sha256={},
        seed=None,
        physical_config={},
        solver_config={},
        material_config=material_config,
    )
    material_config["wall"]["permittivity"] = 99.0
    assert original.material_config["wall"]["permittivity"] == 4.2
    assert UMCDRunProvenance.from_dict(record.to_dict()).to_dict() == record.to_dict()
    tampered = record.to_dict()
    tampered["solver_config"]["max_depth"] = 99
    with pytest.raises(ValueError, match="hash is invalid"):
        UMCDRunProvenance.from_dict(tampered)


def test_provenance_lineage_claims_are_tiered_hash_bound_and_schema_valid(tmp_path):
    claims = (
        LineageStatusClaim(status="proposed", scope="city43-phase0", evidence_sha256={}),
        LineageStatusClaim(
            status="synthetic_debug",
            scope="network-phy-composer",
            evidence_sha256={"regression_report_sha256": "d" * 64},
        ),
        LineageStatusClaim(
            status="native_verified",
            scope="city43-phase0",
            evidence_sha256={
                "native_run_manifest_sha256": "e" * 64,
                "numeric_shape_gauge_report_sha256": "f" * 64,
            },
        ),
    )
    record = replace(_provenance_record(), lineage_claims=claims)
    payload = record.to_dict()
    assert payload["schema"] == "umcd.run_provenance.v2"
    assert UMCDRunProvenance.from_dict(payload).to_dict() == payload

    jsonschema = pytest.importorskip("jsonschema")
    schema_path = (
        Path(__file__).resolve().parents[2] / "datasets/schema/umcd-run-provenance.v2.schema.json"
    )
    schema = json.loads(schema_path.read_text())
    jsonschema.Draft202012Validator(schema).validate(payload)
    missing_evidence = json.loads(json.dumps(payload))
    missing_evidence["lineage_claims"][1]["evidence_sha256"] = {}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(missing_evidence)

    bundle = replace(_bundle(), provenance_records=(record,))
    stored = write_episode_bundle(tmp_path, bundle)
    loaded = read_episode_bundle(tmp_path, stored)
    assert loaded.provenance_records[0].to_dict() == payload


@pytest.mark.parametrize("subject_layer", ("scenario_gbsm", "counterfactual_ablations"))
def test_provenance_schema_admits_provenance_for_all_episode_bundle_layers(subject_layer):
    record = replace(
        _provenance_record(),
        subject_layer=subject_layer,
        lineage_claims=(),
    )
    payload = record.to_dict()
    assert UMCDRunProvenance.from_dict(payload).to_dict() == payload

    jsonschema = pytest.importorskip("jsonschema")
    schema_path = (
        Path(__file__).resolve().parents[2] / "datasets/schema/umcd-run-provenance.v2.schema.json"
    )
    schema = json.loads(schema_path.read_text())
    jsonschema.Draft202012Validator(schema).validate(payload)


@pytest.mark.parametrize(
    "status,evidence",
    [
        ("scenario_qualified", {"native_run_manifest_sha256": "a" * 64}),
        ("geometry_qualified", {"convergence_audit_sha256": "a" * 64}),
        ("receiver_qualified", {"receiver_contract_sha256": "a" * 64}),
        ("external_tested", {"external_test_report_sha256": "not-a-digest"}),
    ],
)
def test_lineage_claim_rejects_incomplete_or_invalid_evidence(status, evidence):
    with pytest.raises(ValueError):
        LineageStatusClaim(status=status, scope="task", evidence_sha256=evidence)


def test_lineage_evidence_audit_verifies_exact_payload_roster_and_schema():
    evidence = b"independent numeric-shape-gauge report\n"
    digest = hashlib.sha256(evidence).hexdigest()
    claims = (
        LineageStatusClaim(
            status="synthetic_debug",
            scope="network-phy",
            evidence_sha256={"regression_report_sha256": digest},
        ),
    )
    report = audit_lineage_evidence_payloads(claims, {digest: evidence})
    assert report["status"] == "EVIDENCE_BYTES_VERIFIED"
    assert report["scientific_status"] == "NOT_QUALIFIED"
    assert report["verified_evidence_bytes"] == len(evidence)

    jsonschema = pytest.importorskip("jsonschema")
    schema_path = (
        Path(__file__).resolve().parents[2]
        / "datasets/schema/umcd-lineage-evidence-audit.v2.schema.json"
    )
    jsonschema.Draft202012Validator(json.loads(schema_path.read_text())).validate(report)


def test_lineage_evidence_audit_rejects_missing_extra_tampered_and_oversized_payloads():
    evidence = b"qualification report"
    digest = hashlib.sha256(evidence).hexdigest()
    claim = LineageStatusClaim(
        status="synthetic_debug",
        scope="task",
        evidence_sha256={"regression_report_sha256": digest},
    )
    claims = (claim,)
    with pytest.raises(ValueError, match="roster"):
        audit_lineage_evidence_payloads(claims, {})
    with pytest.raises(ValueError, match="roster"):
        audit_lineage_evidence_payloads(claims, {digest: evidence, "0" * 64: b"extra"})
    with pytest.raises(ValueError, match="does not match digest"):
        audit_lineage_evidence_payloads(claims, {digest: b"tampered"})
    with pytest.raises(ValueError, match="exceed max_total_bytes"):
        audit_lineage_evidence_payloads(claims, {digest: evidence}, max_total_bytes=4)


def test_lineage_evidence_audit_reports_proposed_claim_without_evidence():
    claim = LineageStatusClaim(status="proposed", scope="planned-task", evidence_sha256={})
    report = audit_lineage_evidence_payloads((claim,), {})
    assert report["status"] == "NO_EVIDENCE_REQUIRED"
    assert report["scientific_status"] == "NOT_QUALIFIED"
    assert report["verified_evidence_sha256"] == []


def test_lineage_evidence_file_audit_streams_stable_files_under_root(tmp_path):
    evidence = b"large-enough-evidence\n" * 8192
    digest = hashlib.sha256(evidence).hexdigest()
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    (evidence_root / "audit.bin").write_bytes(evidence)
    claim = LineageStatusClaim(
        status="synthetic_debug",
        scope="task",
        evidence_sha256={"regression_report_sha256": digest},
    )

    report = audit_lineage_evidence_files(
        (claim,), {digest: "audit.bin"}, evidence_root=evidence_root
    )
    assert report["status"] == "EVIDENCE_BYTES_VERIFIED"
    assert report["verified_evidence_bytes"] == len(evidence)

    with pytest.raises(ValueError, match="max_total_bytes"):
        audit_lineage_evidence_files(
            (claim,),
            {digest: "audit.bin"},
            evidence_root=evidence_root,
            max_total_bytes=len(evidence) - 1,
        )
    with pytest.raises(ValueError, match="beneath evidence_root"):
        audit_lineage_evidence_files(
            (claim,), {digest: "../audit.bin"}, evidence_root=evidence_root
        )

    (evidence_root / "alias.bin").symlink_to(evidence_root / "audit.bin")
    with pytest.raises(OSError):
        audit_lineage_evidence_files((claim,), {digest: "alias.bin"}, evidence_root=evidence_root)

    os.link(evidence_root / "audit.bin", evidence_root / "hardlink.bin")
    with pytest.raises(ValueError, match="single-link"):
        audit_lineage_evidence_files(
            (claim,), {digest: "hardlink.bin"}, evidence_root=evidence_root
        )


def test_provenance_reader_rejects_boolean_schema_version_even_when_one():
    payload = _provenance_record().to_dict()
    payload["schema_version"] = True
    with pytest.raises(ValueError, match="schema_version must be an integer"):
        UMCDRunProvenance.from_dict(payload)
