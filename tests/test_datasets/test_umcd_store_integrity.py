"""Integrity regression tests for the public UMCD episode store API."""

from __future__ import annotations

import json

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
from datasets.path_oracle import PathOracleTable, PathSolveStatus


def _episode_bundle() -> EpisodeBundle:
    gnb_array = AntennaArraySpec(
        array_id="gNB-array",
        element_positions_local_m=np.zeros((1, 3)),
        polarization_per_element=("V",),
        orientation_local_to_world=np.eye(3),
        port_ids=("gNB-port-0",),
        port_element_weights=np.ones((1, 1), dtype=np.complex128),
    )
    ue_array = AntennaArraySpec(
        array_id="UE-array",
        element_positions_local_m=np.zeros((1, 3)),
        polarization_per_element=("V",),
        orientation_local_to_world=np.eye(3),
        port_ids=("UE-port-0",),
        port_element_weights=np.ones((1, 1), dtype=np.complex128),
    )
    episode = NetworkEpisode(
        episode_id="g7-06-integrity-episode",
        split_group_id="integrity-group-0",
        family_id="integrity-test-scene",
        carrier_hz=3.5e9,
        arrays=(gnb_array, ue_array),
        nodes=(
            NetworkNode("gNB-0", NodeRole.GNB, "gNB-array", (0.0, 0.0, 5.0), 1),
            NetworkNode("UE-0", NodeRole.UE, "UE-array", (10.0, 0.0, 1.5), 1),
        ),
        links=(
            DirectedLink(
                "gNB-0-to-UE-0",
                "gNB-0",
                "UE-0",
                (0,),
                (0,),
                LinkDirection.DOWNLINK,
                LinkRole.SERVING,
            ),
        ),
        frame_times_s=(0.0,),
        subcarrier_offsets_hz=np.asarray([0.0]),
    )
    paths = PathOracleTable(
        link_ids=("gNB-0-to-UE-0",),
        frame_indices=np.asarray([0]),
        rx_port_indices=np.asarray([0]),
        tx_port_indices=np.asarray([0]),
        path_offsets=np.asarray([0, 1]),
        solve_status=(PathSolveStatus.SOLVED,),
        delay_s=np.asarray([35e-9]),
        complex_gain=np.asarray([0.25 + 0.5j]),
        interaction_order=np.asarray([0]),
    )
    return EpisodeBundle(episode, paths, (), ())


def test_episode_bundle_reader_rejects_changed_compressed_object_bytes(tmp_path):
    store = tmp_path / "umcd-store"
    bundle = _episode_bundle()

    stored = write_episode_bundle(store, bundle)
    loaded = read_episode_bundle(store, stored)

    assert loaded.episode.episode_id == bundle.episode.episode_id
    np.testing.assert_array_equal(loaded.path_oracle.complex_gain, bundle.path_oracle.complex_gain)

    manifest_path = store / stored.relative_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    path_reference = next(item for item in manifest["layers"] if item["layer"] == "oracle_paths")
    object_path = store / path_reference["path"]
    assert object_path.suffix == ".npz"

    tampered = bytearray(object_path.read_bytes())
    byte_index = len(tampered) // 2
    tampered[byte_index] ^= 0x01
    object_path.write_bytes(tampered)

    with pytest.raises(ValueError, match="content hash"):
        read_episode_bundle(store, stored)
