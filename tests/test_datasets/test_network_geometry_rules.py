"""Network geometry: antenna arrays, nodes, directed links, pose timelines and episode bindings."""

from __future__ import annotations

import numpy as np
import pytest
from datasets.network import (
    AntennaArraySpec,
    DirectedLink,
    LinkDirection,
    LinkRole,
    NetworkNode,
    NodePoseTimeline,
    NodeRole,
)


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
