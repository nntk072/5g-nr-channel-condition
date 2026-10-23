"""Whole-episode Sionna-to-UMCD device/link mapping tests."""

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
    NodeRole,
)
from datasets.path_oracle import PathOracleTable, PathSolveStatus
from integrations.sionna.rt_network import (
    SionnaPathBinding,
    sionna_bindings_to_episode_oracle,
    write_sionna_rt_episode_bundle,
)


def _episode() -> NetworkEpisode:
    tx_array = AntennaArraySpec(
        "tx-array",
        np.asarray([[0.0, 0.0, 0.0], [0.0, 0.1, 0.0]]),
        ("V", "V"),
        np.eye(3),
        ("tx0", "tx1"),
        np.eye(2, dtype=np.complex128),
    )
    rx_array = AntennaArraySpec(
        "rx-array",
        np.asarray([[0.0, 0.0, 0.0], [0.0, 0.1, 0.0]]),
        ("V", "V"),
        np.eye(3),
        ("rx0", "rx1"),
        np.eye(2, dtype=np.complex128),
    )
    links = tuple(
        DirectedLink(
            f"tx-to-rx-{receiver}",
            "tx",
            f"rx-{receiver}",
            (1, 0),
            (1, 0),
            LinkDirection.DOWNLINK,
            LinkRole.SERVING,
        )
        for receiver in (0, 1)
    )
    return NetworkEpisode(
        "episode-1",
        "site-1",
        "F03-RT-STATIC",
        3.5e9,
        (tx_array, rx_array),
        (
            NetworkNode("tx", NodeRole.GNB, "tx-array", (0.0, 0.0, 10.0), 2),
            NetworkNode("rx-0", NodeRole.UE, "rx-array", (10.0, 0.0, 1.5), 2),
            NetworkNode("rx-1", NodeRole.UE, "rx-array", (20.0, 0.0, 1.5), 2),
        ),
        links,
        (0.0, 0.01),
        np.asarray([-30e3, 0.0, 30e3]),
    )


def _fake_table(
    *,
    link_id: str,
    frame_index: int,
    rx_port_map: tuple[int, ...],
    tx_port_map: tuple[int, ...],
    solve_status: PathSolveStatus | None,
) -> PathOracleTable:
    del rx_port_map, tx_port_map
    rx_ports = np.repeat(np.arange(2), 2)
    tx_ports = np.tile(np.arange(2), 2)
    count = len(rx_ports)
    return PathOracleTable(
        link_ids=(link_id,) * count,
        frame_indices=np.full(count, frame_index),
        rx_port_indices=rx_ports,
        tx_port_indices=tx_ports,
        path_offsets=np.arange(count + 1),
        solve_status=(solve_status or PathSolveStatus.SOLVED,) * count,
        delay_s=np.full(count, frame_index + 1e-7),
        complex_gain=np.full(count, 0.5 + 0.25j),
        interaction_order=np.zeros(count, dtype=np.int64),
        doppler_hz=np.zeros(count),
        aoa_azimuth_elevation_rad=np.zeros((count, 2)),
        aod_azimuth_elevation_rad=np.zeros((count, 2)),
        path_kind=("direct",) * count,
        interaction_offsets=np.zeros(count + 1, dtype=np.int64),
        interaction_types=np.empty(0, dtype=np.int64),
        object_ids=np.empty(0, dtype=np.int64),
        primitive_ids=np.empty(0, dtype=np.int64),
        interaction_vertices_m=np.empty((0, 3)),
        interaction_vertex_frame="sionna_rt_scene_xyz_m",
    )


def test_whole_episode_maps_each_link_frame_and_reorders_link_local_ports(monkeypatch):
    from integrations.sionna import rt_network

    observed_maps = []

    def convert(paths, **kwargs):
        observed_maps.append((kwargs["rx_port_map"], kwargs["tx_port_map"]))
        return _fake_table(
            link_id=kwargs["link_id"],
            frame_index=kwargs["frame_index"],
            rx_port_map=kwargs["rx_port_map"],
            tx_port_map=kwargs["tx_port_map"],
            solve_status=kwargs["solve_status"],
        )

    monkeypatch.setattr(rt_network, "sionna_paths_to_oracle_table", convert)
    episode = _episode()
    bindings = [
        SionnaPathBinding(link.link_id, frame, index, 0, object())
        for index, link in enumerate(episode.links)
        for frame in range(2)
    ]
    bindings[-1] = SionnaPathBinding(
        episode.links[-1].link_id,
        1,
        1,
        0,
        None,
        PathSolveStatus.SOLVER_FAILURE,
    )

    table = sionna_bindings_to_episode_oracle(episode, bindings)

    assert table.group_count == 16
    assert table.path_count == 12
    assert set(observed_maps) == {((1, 0), (1, 0))}
    assert set(table.link_ids) == {link.link_id for link in episode.links}
    assert table.frame_indices.tolist().count(0) == 8
    assert table.frame_indices.tolist().count(1) == 8
    failed = [
        index
        for index, (link, frame) in enumerate(zip(table.link_ids, table.frame_indices, strict=True))
        if link == episode.links[-1].link_id and frame == 1
    ]
    assert len(failed) == 4
    assert {table.solve_status[index] for index in failed} == {PathSolveStatus.SOLVER_FAILURE}
    assert all(table.group_slice(index).start == table.group_slice(index).stop for index in failed)
    table.validate_for(episode)


def test_episode_mapping_rejects_missing_and_duplicate_link_frame_bindings(monkeypatch):
    from integrations.sionna import rt_network

    monkeypatch.setattr(
        rt_network,
        "sionna_paths_to_oracle_table",
        lambda paths, **kwargs: _fake_table(
            link_id=kwargs["link_id"],
            frame_index=kwargs["frame_index"],
            rx_port_map=kwargs["rx_port_map"],
            tx_port_map=kwargs["tx_port_map"],
            solve_status=kwargs["solve_status"],
        ),
    )
    episode = _episode()
    complete = [
        SionnaPathBinding(link.link_id, frame, receiver_index, 0, object())
        for receiver_index, link in enumerate(episode.links)
        for frame in range(2)
    ]

    with pytest.raises(ValueError, match="cover every episode link/frame"):
        sionna_bindings_to_episode_oracle(episode, complete[:-1])
    with pytest.raises(ValueError, match="unique per link and frame"):
        sionna_bindings_to_episode_oracle(episode, [*complete[:-1], complete[0]])
    inconsistent = list(complete)
    original = inconsistent[2]
    inconsistent[2] = SionnaPathBinding(
        original.link_id,
        original.frame_index,
        original.rx_device_index,
        1,
        original.paths,
    )
    with pytest.raises(ValueError, match="mapping must be stable and one-to-one"):
        sionna_bindings_to_episode_oracle(episode, inconsistent)

    ordered = sionna_bindings_to_episode_oracle(episode, complete)
    reversed_order = sionna_bindings_to_episode_oracle(episode, tuple(reversed(complete)))
    assert ordered.link_ids == reversed_order.link_ids
    np.testing.assert_array_equal(ordered.path_offsets, reversed_order.path_offsets)
    np.testing.assert_array_equal(ordered.complex_gain, reversed_order.complex_gain)


def test_rt_episode_writer_publishes_layer_bound_provenance(monkeypatch, tmp_path):
    from datasets.episode_store import read_episode_bundle
    from integrations.sionna import rt_network
    from integrations.sionna.rt_identity import rt_run_identity

    monkeypatch.setattr(
        rt_network,
        "sionna_paths_to_oracle_table",
        lambda paths, **kwargs: _fake_table(
            link_id=kwargs["link_id"],
            frame_index=kwargs["frame_index"],
            rx_port_map=kwargs["rx_port_map"],
            tx_port_map=kwargs["tx_port_map"],
            solve_status=kwargs["solve_status"],
        ),
    )
    identity = rt_run_identity(
        scene_asset_manifest={"complete": True, "asset_set_sha256": "a" * 64},
        physical_config={"carrier_hz": 3.5e9},
        solver_config={"max_depth": 2},
        seed=9,
        backend_selection={
            "backend": "llvm_cpu",
            "variant": "llvm_ad_mono_polarized",
            "runtime_versions": {"sionna-rt": "2.2.0"},
        },
        runtime_versions={"python": "3.12.0"},
        package_source_sha256="b" * 64,
    )
    episode = _episode()
    bindings = [
        SionnaPathBinding(link.link_id, frame, link_index, 0, object())
        for link_index, link in enumerate(episode.links)
        for frame in range(len(episode.frame_times_s))
    ]

    from integrations.sionna.d1_generation_gate import authorize_d1_generation

    authorization = authorize_d1_generation(
        {
            "operation": "rt_data",
            "config": {
                "per_class": 1,
                "max_attempts": 2,
                "samples_per_src": 1,
                "max_depth": 1,
                "total_work_units": 2,
            },
            "destinations": [str(tmp_path.resolve())],
        },
        pilot=True,
    )
    stored = write_sionna_rt_episode_bundle(
        tmp_path,
        episode,
        bindings,
        run_identity=identity,
        material_config={"wall": {"scattering_coefficient": 0.2}},
        source_family="F03-RT-STATIC",
        producer="Sionna 2.2 PathSolver",
        authorization=authorization,
    )
    loaded = read_episode_bundle(tmp_path, stored)

    assert loaded.path_oracle.group_count == 16
    assert [record.subject_layer for record in loaded.provenance_records] == ["oracle_paths"]
    record = loaded.provenance_records[0]
    assert record.run_identity_sha256 == identity["cache_key_sha256"]
    assert record.material_config["wall"]["scattering_coefficient"] == 0.2


@pytest.mark.native_sionna
def test_native_multi_tx_rx_paths_assemble_into_one_episode_table():
    mi = pytest.importorskip("mitsuba")
    mi.set_variant("llvm_ad_mono_polarized")
    rt = pytest.importorskip("sionna.rt")

    from integrations.sionna.rt_network import (
        SionnaPathBinding,
        sionna_bindings_to_episode_oracle,
    )

    scene = rt.load_scene()
    scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.add(rt.Transmitter("tx0", [0.0, 0.0, 1.0]))
    scene.add(rt.Transmitter("tx1", [2.0, 0.0, 1.0]))
    scene.add(rt.Receiver("rx0", [10.0, 0.0, 1.0]))
    scene.add(rt.Receiver("rx1", [12.0, 0.0, 1.0]))
    paths = rt.PathSolver(deterministic=True)(
        scene,
        max_depth=0,
        los=True,
        specular_reflection=False,
        diffuse_reflection=False,
        refraction=False,
        diffraction=False,
        synthetic_array=True,
    )

    array = AntennaArraySpec(
        "siso",
        np.zeros((1, 3)),
        ("V",),
        np.eye(3),
        ("port-0",),
        np.ones((1, 1), dtype=np.complex128),
    )
    nodes = (
        NetworkNode("tx0", NodeRole.GNB, "siso", (0.0, 0.0, 1.0), 1),
        NetworkNode("tx1", NodeRole.GNB, "siso", (2.0, 0.0, 1.0), 1),
        NetworkNode("rx0", NodeRole.UE, "siso", (10.0, 0.0, 1.0), 1),
        NetworkNode("rx1", NodeRole.UE, "siso", (12.0, 0.0, 1.0), 1),
    )
    links = tuple(
        DirectedLink(
            f"{tx}-to-{rx}",
            tx,
            rx,
            (0,),
            (0,),
            LinkDirection.DOWNLINK,
            LinkRole.SERVING,
        )
        for tx in ("tx0", "tx1")
        for rx in ("rx0", "rx1")
    )
    episode = NetworkEpisode(
        "native-episode",
        "native-site",
        "F03-RT-STATIC",
        3.5e9,
        (array,),
        nodes,
        links,
        (0.0,),
        np.asarray([-15e3, 0.0, 15e3]),
    )
    bindings = tuple(
        SionnaPathBinding(link.link_id, 0, rx_index, tx_index, paths)
        for tx_index, tx in enumerate(("tx0", "tx1"))
        for rx_index, rx in enumerate(("rx0", "rx1"))
        for link in links
        if link.tx_node_id == tx and link.rx_node_id == rx
    )

    oracle = sionna_bindings_to_episode_oracle(episode, bindings)

    assert oracle.group_count == 4
    assert set(oracle.link_ids) == {link.link_id for link in links}
    assert oracle.path_count == int(np.count_nonzero(paths.valid.numpy()))
    for binding in bindings:
        link = episode.link(binding.link_id)
        mask = np.asarray(paths.valid.numpy()[binding.rx_device_index, binding.tx_device_index])
        group = oracle.link_ids.index(link.link_id)
        np.testing.assert_allclose(
            oracle.delay_s[oracle.group_slice(group)],
            paths.tau.numpy()[binding.rx_device_index, binding.tx_device_index, mask],
        )
