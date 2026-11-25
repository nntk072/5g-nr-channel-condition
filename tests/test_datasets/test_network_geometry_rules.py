"""Network geometry: antenna arrays, nodes, directed links, pose timelines and episode bindings."""

from __future__ import annotations

import numpy as np
import pytest
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
from datasets.time_axis import align_named_time_axes, align_time_axes


def _array(**overrides):
    kwargs = {
        "array_id": "array-a",
        "element_positions_local_m": np.zeros((2, 3)),
        "polarization_per_element": ("V", "V"),
        "orientation_local_to_world": np.eye(3),
        "port_ids": ("p0", "p1"),
        "port_element_weights": np.eye(2, dtype=np.complex128),
    }
    kwargs.update(overrides)
    return AntennaArraySpec(**kwargs)


def test_valid_array_is_accepted() -> None:
    assert _array().port_ids == ("p0", "p1")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        pytest.param(
            {"array_id": " padded"}, "array_id must be a non-empty canonical string", id="id"
        ),
        pytest.param(
            {"element_positions_local_m": np.zeros((2, 2))}, "must have shape", id="positions-shape"
        ),
        pytest.param(
            {"element_positions_local_m": np.full((2, 3), np.nan)},
            "finite numeric",
            id="positions-nan",
        ),
        pytest.param(
            {"polarization_per_element": ("V", "")}, "non-empty polarization label", id="pol-empty"
        ),
        pytest.param(
            {"port_ids": ("p0", "")}, "non-empty antenna-port identifiers", id="port-empty"
        ),
        pytest.param(
            {"port_ids": ("p0", "p0")},
            "must be unique within an antenna array",
            id="port-duplicate",
        ),
        pytest.param(
            {"port_element_weights": np.full((2, 2), np.nan + 0j)},
            "finite complex",
            id="weights-nan",
        ),
        pytest.param(
            {"port_element_weights": np.zeros((2, 2), dtype=np.complex128)},
            "at least one radiating element",
            id="port-unmapped",
        ),
        pytest.param(
            {"orientation_local_to_world": np.full((3, 3), np.nan)},
            "finite 3x3 matrix",
            id="orientation-nan",
        ),
        pytest.param(
            {"orientation_local_to_world": 2.0 * np.eye(3)}, "orthonormal", id="orientation-scale"
        ),
    ],
)
def test_array_rejects_malformed_geometry(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        _array(**overrides)


def test_node_requires_typed_role_position_and_chains() -> None:
    with pytest.raises(TypeError, match="role must be a NodeRole"):
        NetworkNode(
            node_id="n",
            role="gnb",
            array_id="a",
            position_world_m=(0.0, 0.0, 1.0),
            rf_chain_count=1,
        )
    with pytest.raises(TypeError, match="real numeric coordinates"):
        NetworkNode(
            node_id="n",
            role=NodeRole.GNB,
            array_id="a",
            position_world_m=("x", 0.0, 1.0),
            rf_chain_count=1,
        )
    with pytest.raises(ValueError, match="three finite coordinates"):
        NetworkNode(
            node_id="n",
            role=NodeRole.GNB,
            array_id="a",
            position_world_m=(0.0, 0.0),
            rf_chain_count=1,
        )
    with pytest.raises(ValueError, match="positive integer"):
        NetworkNode(
            node_id="n",
            role=NodeRole.GNB,
            array_id="a",
            position_world_m=(0.0, 0.0, 1.0),
            rf_chain_count=0,
        )
    with pytest.raises(TypeError, match="real numeric coordinates"):
        NetworkNode(
            node_id="n",
            role=NodeRole.GNB,
            array_id="a",
            position_world_m=None,
            rf_chain_count=1,
        )


def test_directed_link_requires_distinct_nodes_typed_enums_and_port_maps() -> None:
    base = {
        "link_id": "l",
        "tx_node_id": "a",
        "rx_node_id": "b",
        "tx_port_indices": (0,),
        "rx_port_indices": (0,),
        "direction": LinkDirection.DOWNLINK,
        "role": LinkRole.SERVING,
    }
    assert DirectedLink(**base).tx_node_id == "a"
    with pytest.raises(ValueError, match="distinct nodes"):
        DirectedLink(**{**base, "rx_node_id": "a"})
    with pytest.raises(TypeError, match="typed enums"):
        DirectedLink(**{**base, "direction": "downlink"})
    with pytest.raises(ValueError, match="nonnegative integer port indices"):
        DirectedLink(**{**base, "tx_port_indices": (-1,)})
    with pytest.raises(ValueError, match="must not contain duplicate ports"):
        DirectedLink(**{**base, "rx_port_indices": (0, 0)})


def test_pose_timeline_checks_identity_shapes_and_frames() -> None:
    kwargs = {
        "node_ids": ("a", "b"),
        "frame_times_s": (0.0, 1e-3),
        "positions_world_m": np.zeros((2, 2, 3)),
        "orientations_local_to_world": np.broadcast_to(np.eye(3), (2, 2, 3, 3)),
    }
    timeline = NodePoseTimeline(**kwargs)
    assert timeline.node_ids == ("a", "b")
    with pytest.raises(ValueError, match="non-empty node IDs"):
        NodePoseTimeline(**{**kwargs, "node_ids": ()})
    with pytest.raises(ValueError, match="node IDs must be unique"):
        NodePoseTimeline(**{**kwargs, "node_ids": ("a", "a")})
    with pytest.raises(ValueError, match="positions_world_m must be finite"):
        NodePoseTimeline(**{**kwargs, "positions_world_m": np.zeros((2, 3))})
    with pytest.raises(ValueError, match="orientations_local_to_world must be finite with shape"):
        NodePoseTimeline(**{**kwargs, "orientations_local_to_world": np.zeros((2, 2, 3))})


def test_timeline_lookups_reject_unknown_nodes_and_frames() -> None:
    timeline = NodePoseTimeline(
        node_ids=("a", "b"),
        frame_times_s=(0.0, 1e-3),
        positions_world_m=np.zeros((2, 2, 3)),
        orientations_local_to_world=np.broadcast_to(np.eye(3), (2, 2, 3, 3)),
    )
    with pytest.raises(TypeError, match="frame_index must be an integer"):
        timeline.position("a", 0.5)
    with pytest.raises(IndexError, match="outside the pose timeline"):
        timeline.position("a", 5)
    with pytest.raises(KeyError):
        timeline.position("missing", 0)
    with pytest.raises(KeyError):
        timeline.orientation("missing", 0)


def _episode(**overrides) -> NetworkEpisode:
    arrays = (
        _array(array_id="a", port_ids=("p0", "p1")),
        _array(array_id="b", port_ids=("p0", "p1")),
    )
    nodes = (
        NetworkNode("tx", NodeRole.GNB, "a", (0.0, 0.0, 0.0), 2),
        NetworkNode("rx", NodeRole.UE, "b", (1.0, 0.0, 0.0), 2),
    )
    link = DirectedLink("l", "tx", "rx", (0,), (0,), LinkDirection.DOWNLINK, LinkRole.SERVING)
    values = {
        "episode_id": "e",
        "split_group_id": "s",
        "family_id": "f",
        "carrier_hz": 3.5e9,
        "arrays": arrays,
        "nodes": nodes,
        "links": (link,),
        "frame_times_s": (0.0, 1.0),
        "subcarrier_offsets_hz": np.asarray([-1.0, 0.0, 1.0]),
    }
    values.update(overrides)
    return NetworkEpisode(**values)


def test_episode_rejects_timeline_and_split_lineage_mismatches() -> None:
    episode = _episode()
    positions = np.asarray([[node.position_world_m for node in episode.nodes]] * 2)
    orientations = np.broadcast_to(np.eye(3), (2, 2, 3, 3)).copy()
    timeline = NodePoseTimeline(
        node_ids=("tx", "rx"),
        frame_times_s=(0.0, 1.0),
        positions_world_m=positions,
        orientations_local_to_world=orientations,
    )
    with pytest.raises(TypeError, match="pose_timeline must be a NodePoseTimeline"):
        _episode(pose_timeline=object())
    with pytest.raises(ValueError, match="pose timeline frame times"):
        _episode(
            pose_timeline=NodePoseTimeline(
                ("tx", "rx"),
                (0.0, 2.0),
                positions,
                orientations,
            )
        )
    with pytest.raises(ValueError, match="pose timeline node IDs"):
        _episode(
            pose_timeline=NodePoseTimeline(
                ("rx", "tx"),
                (0.0, 1.0),
                positions,
                orientations,
            )
        )
    with pytest.raises(ValueError, match="initial pose position"):
        _episode(
            pose_timeline=NodePoseTimeline(
                ("tx", "rx"),
                (0.0, 1.0),
                positions + 1.0,
                orientations,
            )
        )
    with pytest.raises(ValueError, match="initial pose orientation"):
        _episode(
            pose_timeline=NodePoseTimeline(
                ("tx", "rx"),
                (0.0, 1.0),
                positions,
                np.broadcast_to(
                    np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
                    (2, 2, 3, 3),
                ),
            )
        )
    accepted = _episode(pose_timeline=timeline)
    assert accepted.pose_timeline is timeline
    assert accepted.position_at("rx", 1) == (1.0, 0.0, 0.0)
    np.testing.assert_array_equal(accepted.orientation_at("rx", 1), np.eye(3))
    assert accepted.position_at("tx", 1) == (0.0, 0.0, 0.0)
    np.testing.assert_array_equal(
        _episode().orientation_at("tx", 0), _episode().array("a").orientation_local_to_world
    )
    assert accepted.link("l").tx_node_id == "tx"
    assert accepted.node("rx").array_id == "b"
    assert accepted.array("a").num_ports == 2
    aligned_timeline, alignment = timeline.align_to((0.0, 1.0), tolerance_s=0.0)
    assert alignment.complete
    assert aligned_timeline.frame_times_s == (0.0, 1.0)
    pose_alignment = align_time_axes((0.0, 1.0), (0.0, 1.0), tolerance_s=0.0)
    source_alignment = align_named_time_axes(
        (0.0, 1.0), {"capture-a": (0.0, 1.0)}, tolerance_s=0.0, require_complete=True
    )
    with_pose_alignment = _episode(pose_timeline=timeline, pose_time_alignment=pose_alignment)
    assert with_pose_alignment.pose_time_alignment == pose_alignment
    with_source_alignment = _episode(source_time_alignment=source_alignment)
    assert with_source_alignment.source_time_alignment == source_alignment
    assert _episode(split_leakage_relations=(("site", "site-1"),)).split_leakage_relations == (
        ("site", "site-1"),
    )
    with pytest.raises(TypeError, match="pose_time_alignment must be a TimeAxisAlignment"):
        _episode(pose_time_alignment=object())
    with pytest.raises(ValueError, match="requires a pose_timeline"):
        _episode(pose_time_alignment=pose_alignment)
    incomplete_alignment = align_time_axes((0.0, 1.0), (0.0,), tolerance_s=0.0)
    with pytest.raises(ValueError, match="completely match episode frame times"):
        _episode(pose_timeline=timeline, pose_time_alignment=incomplete_alignment)
    wrong_pose_reference = align_time_axes((1.0,), (1.0,), tolerance_s=0.0)
    with pytest.raises(ValueError, match="completely match episode frame times"):
        _episode(pose_timeline=timeline, pose_time_alignment=wrong_pose_reference)
    with pytest.raises(TypeError, match="source_time_alignment must be a MultiSource"):
        _episode(source_time_alignment=object())
    wrong_source_reference = align_named_time_axes((1.0,), {"capture-a": (1.0,)}, tolerance_s=0.0)
    with pytest.raises(ValueError, match="episode frame times as reference"):
        _episode(source_time_alignment=wrong_source_reference)
    with pytest.raises(ValueError, match="canonical namespace/id pairs"):
        _episode(split_leakage_relations=(("unknown", "scene-1"),))
    with pytest.raises(ValueError, match="must be unique"):
        _episode(split_leakage_relations=(("site", "site-1"), ("site", "site-1")))
