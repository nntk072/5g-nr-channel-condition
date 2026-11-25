"""Network episode: carrier, roster uniqueness, port maps, frequency grid and lookup rules."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from datasets.network import DirectedLink, LinkDirection, LinkRole
from tests.test_datasets.umcd_schema_support import _episode


def _with(**changes):
    return dataclasses.replace(_episode(), **changes)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        pytest.param({"carrier_hz": 0.0}, "carrier_hz must be finite and positive", id="carrier"),
        pytest.param(
            {"arrays": ()}, "at least one array, node, and directed link", id="empty-arrays"
        ),
        pytest.param(
            {"subcarrier_offsets_hz": np.asarray(["a", "b", "c"])},
            "real numeric dtype",
            id="offset-dtype",
        ),
        pytest.param(
            {"subcarrier_offsets_hz": np.asarray([])}, "non-empty finite vector", id="offset-empty"
        ),
        pytest.param(
            {"subcarrier_offsets_hz": np.asarray([0.0, 0.0, 1.0])},
            "strictly increasing and unique",
            id="offset-order",
        ),
    ],
)
def test_episode_carrier_and_grid_are_checked(changes, message) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _with(**changes)


def test_episode_identities_must_be_unique() -> None:
    episode = _episode()
    with pytest.raises(ValueError, match="array_id values must be unique"):
        _with(arrays=episode.arrays + (episode.arrays[0],))
    with pytest.raises(ValueError, match="node_id values must be unique"):
        _with(nodes=episode.nodes + (episode.nodes[0],))
    with pytest.raises(ValueError, match="link_id values must be unique"):
        _with(links=episode.links + (episode.links[0],))


def test_node_must_reference_a_declared_array() -> None:
    episode = _episode()
    orphan = dataclasses.replace(episode.nodes[0], array_id="absent-array")
    with pytest.raises(ValueError, match="references a missing antenna array"):
        _with(nodes=(orphan, episode.nodes[1]))


def test_link_port_map_must_fit_the_rx_array() -> None:
    episode = _episode()
    link = episode.links[0]
    bad = DirectedLink(
        link.link_id,
        link.tx_node_id,
        link.rx_node_id,
        link.tx_port_indices,
        (99,),
        LinkDirection.DOWNLINK,
        LinkRole.SERVING,
    )
    with pytest.raises(ValueError, match="invalid RX port map"):
        _with(links=(bad,))


def test_lookups_reject_unknown_identities_and_bad_frames() -> None:
    episode = _episode()
    with pytest.raises(KeyError):
        episode.link("absent-link")
    with pytest.raises(KeyError):
        episode.node("absent-node")
    with pytest.raises(KeyError):
        episode.array("absent-array")
    with pytest.raises(TypeError, match="frame_index must be an integer"):
        episode.position_at(episode.nodes[0].node_id, 0.5)
    with pytest.raises(IndexError, match="outside the episode time axis"):
        episode.position_at(episode.nodes[0].node_id, 9)
    with pytest.raises(TypeError, match="frame_index must be an integer"):
        episode.orientation_at(episode.nodes[0].node_id, 0.5)
    with pytest.raises(IndexError, match="outside the episode time axis"):
        episode.orientation_at(episode.nodes[0].node_id, 9)
