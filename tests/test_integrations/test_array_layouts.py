"""Scene-to-UMCD array and device-axis extraction tests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from datasets.network import LinkDirection, LinkRole, NodeRole
from datasets.path_oracle import PathSolveStatus
from integrations.sionna.array_layouts import sionna_scene_to_network_episode
from integrations.sionna.rt_network import sionna_bindings_to_episode_oracle


def _vector(x: object, y: object, z: object) -> SimpleNamespace:
    return SimpleNamespace(x=np.asarray(x), y=np.asarray(y), z=np.asarray(z))


def _array() -> SimpleNamespace:
    return SimpleNamespace(
        normalized_positions=_vector([0.0, 0.0], [-0.25, 0.25], [0.0, 0.0]),
        array_size=2,
        num_ant=4,
        antenna_pattern=SimpleNamespace(patterns=[lambda: None, lambda: None]),
    )


def _device(position: tuple[float, float, float], orientation=(0.0, 0.0, 0.0)):
    return SimpleNamespace(
        position=_vector(*position),
        orientation=_vector(*orientation),
        velocity=_vector(0.0, 0.0, 0.0),
    )


def _scene(*, moving_tx: bool = False) -> SimpleNamespace:
    tx = _device((0.0, 0.0, 10.0), (np.pi / 2.0, 0.0, 0.0))
    if moving_tx:
        tx.velocity = _vector(1.0, 0.0, 0.0)
    return SimpleNamespace(
        transmitters={"tx-z": tx, "tx-a": _device((1.0, 0.0, 10.0))},
        receivers={"rx-z": _device((20.0, 0.0, 1.5)), "rx-a": _device((10.0, 0.0, 1.5))},
        tx_array=_array(),
        rx_array=_array(),
        frequency=np.asarray(3.5e9),
    )


def test_scene_extraction_preserves_geometry_feeds_roles_and_solver_axis_order():
    result = sionna_scene_to_network_episode(
        _scene(),
        episode_id="episode",
        split_group_id="scene-family",
        family_id="F03-RT-STATIC",
        subcarrier_offsets_hz=(-30e3, 0.0, 30e3),
        rf_chain_count_by_node={
            "tx:tx-z": 2,
            "tx:tx-a": 2,
            "rx:rx-z": 2,
            "rx:rx-a": 2,
        },
        node_roles={
            "tx:tx-z": NodeRole.GNB,
            "tx:tx-a": NodeRole.INTERFERER,
            "rx:rx-z": NodeRole.UE,
            "rx:rx-a": NodeRole.UE,
        },
        tx_pattern_labels=("V", "H"),
        rx_pattern_labels=("V", "H"),
        direction=LinkDirection.DOWNLINK,
        link_role=LinkRole.SERVING,
    )
    episode = result.episode
    assert episode.carrier_hz == pytest.approx(3.5e9)
    assert [node.node_id for node in episode.nodes] == [
        "tx:tx-z",
        "tx:tx-a",
        "rx:rx-z",
        "rx:rx-a",
    ]
    assert [link.link_id for link in episode.links] == [
        "tx:tx-z->rx:rx-z",
        "tx:tx-z->rx:rx-a",
        "tx:tx-a->rx:rx-z",
        "tx:tx-a->rx:rx-a",
    ]
    assert [
        (item.tx_device_index, item.rx_device_index) for item in result.device_link_mappings
    ] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]
    tx = episode.array("tx:tx-z:array")
    np.testing.assert_allclose(
        tx.element_positions_local_m,
        np.asarray([[0.0, -0.25, 0.0], [0.0, 0.25, 0.0]]) * (299_792_458.0 / 3.5e9),
    )
    np.testing.assert_allclose(
        tx.orientation_local_to_world,
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        atol=1e-12,
    )
    assert tx.port_ids == ("V:element-0", "V:element-1", "H:element-0", "H:element-1")
    np.testing.assert_array_equal(
        tx.port_element_weights,
        np.asarray([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.complex128),
    )
    assert episode.node("tx:tx-a").role is NodeRole.INTERFERER
    assert all(link.direction is LinkDirection.DOWNLINK for link in episode.links)


def test_scene_extraction_requires_explicit_rf_chain_counts_for_multiport_arrays():
    with pytest.raises(ValueError, match="explicit RF-chain count"):
        sionna_scene_to_network_episode(_scene(), episode_id="e", split_group_id="s", family_id="f")


def test_scene_extraction_rejects_motion_until_pose_timeline_is_supported():
    with pytest.raises(ValueError, match="trajectory-aware"):
        sionna_scene_to_network_episode(
            _scene(moving_tx=True), episode_id="e", split_group_id="s", family_id="f"
        )


def test_scene_extraction_rejects_incomplete_pattern_labels_and_associations():
    with pytest.raises(ValueError, match="label each Sionna antenna pattern"):
        sionna_scene_to_network_episode(
            _scene(),
            episode_id="e",
            split_group_id="s",
            family_id="f",
            rf_chain_count_by_node={
                "tx:tx-z": 1,
                "tx:tx-a": 1,
                "rx:rx-z": 1,
                "rx:rx-a": 1,
            },
            tx_pattern_labels=("V",),
        )
    with pytest.raises(ValueError, match="extracted TX followed by an RX"):
        sionna_scene_to_network_episode(
            _scene(),
            episode_id="e",
            split_group_id="s",
            family_id="f",
            rf_chain_count_by_node={
                "tx:tx-z": 1,
                "tx:tx-a": 1,
                "rx:rx-z": 1,
                "rx:rx-a": 1,
            },
            associations=(("rx:rx-z", "tx:tx-z"),),
        )


@pytest.mark.native_sionna
def test_native_scene_and_paths_convert_together_without_manual_device_indices():
    mi = pytest.importorskip("mitsuba")
    mi.set_variant("llvm_ad_mono_polarized")
    rt = pytest.importorskip("sionna.rt")
    from integrations.sionna.array_layouts import sionna_scene_to_network_episode

    scene = rt.load_scene()
    scene.frequency = 3.5e9
    scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=2, pattern="iso", polarization="VH")
    scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=2, pattern="iso", polarization="VH")
    scene.add(rt.Transmitter("tx-z", [0.0, 0.0, 1.0]))
    scene.add(rt.Transmitter("tx-a", [2.0, 0.0, 1.0]))
    scene.add(rt.Receiver("rx-z", [10.0, 0.0, 1.0]))
    scene.add(rt.Receiver("rx-a", [12.0, 0.0, 1.0]))
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
    network = sionna_scene_to_network_episode(
        scene,
        episode_id="native-scene",
        split_group_id="native-scene",
        family_id="F03-RT-STATIC",
        subcarrier_offsets_hz=(-15e3, 0.0, 15e3),
        rf_chain_count_by_node={
            "tx:tx-z": 4,
            "tx:tx-a": 4,
            "rx:rx-z": 4,
            "rx:rx-a": 4,
        },
    )
    bindings = network.bind_paths({link.link_id: paths for link in network.episode.links})
    table = sionna_bindings_to_episode_oracle(network.episode, bindings)
    assert table.group_count == 64
    assert table.path_count == 64
    tx_layout = network.episode.array("tx:tx-z:array")
    assert tx_layout.num_elements == 2
    assert tx_layout.num_ports == 4
    assert tx_layout.port_ids == (
        "pattern-0:element-0",
        "pattern-0:element-1",
        "pattern-1:element-0",
        "pattern-1:element-1",
    )
    assert [
        (item.tx_device_index, item.rx_device_index) for item in network.device_link_mappings
    ] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]
    np.testing.assert_allclose(
        table.delay_s,
        [
            np.asarray(paths.tau.numpy())[rx, tx, 0]
            for tx, rx in ((0, 0), (0, 1), (1, 0), (1, 1))
            for _ in range(16)
        ],
    )


def test_scene_mapping_keeps_missing_solve_explicit():
    from integrations.sionna.array_layouts import sionna_scene_to_network_episode

    scene = _scene()
    scene.tx_array = SimpleNamespace(
        normalized_positions=_vector(0.0, 0.0, 0.0),
        array_size=1,
        num_ant=1,
        antenna_pattern=SimpleNamespace(patterns=[lambda: None]),
    )
    scene.rx_array = scene.tx_array
    network = sionna_scene_to_network_episode(
        scene,
        episode_id="e",
        split_group_id="s",
        family_id="f",
        associations=(("tx:tx-z", "rx:rx-z"),),
    )
    link_id = network.episode.links[0].link_id
    binding = network.bind_paths(
        {link_id: None},
        solve_status_by_link_id={link_id: PathSolveStatus.SOLVER_FAILURE},
    )
    assert binding[0].paths is None
    assert binding[0].solve_status is PathSolveStatus.SOLVER_FAILURE
