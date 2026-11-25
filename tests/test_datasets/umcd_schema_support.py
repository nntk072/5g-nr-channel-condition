"""Shared episode and bundle builders for the UMCD schema tests."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from datasets.channel_view import (
    ChannelDelayReference,
    ChannelPowerScale,
    PhysicalChannelView,
)
from datasets.episode_store import (
    EpisodeBundle,
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
from datasets.path_oracle import PathOracleTable, PathSolveStatus
from datasets.provenance import UMCDRunProvenance
from datasets.targets import (
    TargetDefinition,
    TargetValidity,
    TargetValue,
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


def _bundle() -> EpisodeBundle:
    episode = _episode()
    paths = PathOracleTable(
        link_ids=("gNB-1-to-UE-1", "gNB-1-to-UE-1"),
        frame_indices=np.asarray([0, 0]),
        rx_port_indices=np.asarray([0, 0]),
        tx_port_indices=np.asarray([0, 1]),
        path_offsets=np.asarray([0, 1, 2]),
        solve_status=(PathSolveStatus.SOLVED, PathSolveStatus.SOLVED),
        delay_s=np.asarray([10e-9, 12e-9]),
        complex_gain=np.asarray([0.01 + 0.02j, 0.02 - 0.01j]),
        interaction_order=np.asarray([1, 1]),
        interaction_offsets=np.asarray([0, 1, 2]),
        interaction_types=np.asarray([1, 1]),
        object_ids=np.asarray([7, 8]),
        primitive_ids=np.asarray([3, 4]),
        interaction_vertices_m=np.asarray([[1.0, 2.0, 3.0], [2.0, 2.0, 3.0]]),
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
