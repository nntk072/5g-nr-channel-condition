"""Scene-to-UMCD array and device-axis extraction tests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from datasets.network import LinkDirection, LinkRole, NodeRole
from datasets.path_oracle import PathSolveStatus
from integrations.sionna.array_layouts import (
    _array_spec,
    _device_items,
    _pattern_labels,
    _scalar,
    _vector3,
    sionna_device_pose,
    sionna_scene_to_network_episode,
)
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


def _extract(**changes):
    options = {
        "episode_id": "e",
        "split_group_id": "s",
        "family_id": "f",
        "rf_chain_count_by_node": {
            "tx:tx-z": 2,
            "tx:tx-a": 2,
            "rx:rx-z": 2,
            "rx:rx-a": 2,
        },
    }
    options.update(changes)
    scene = _scene()
    frequency = options.pop("frequency", None)
    if frequency is not None:
        scene.frequency = frequency
    return sionna_scene_to_network_episode(scene, **options)


@pytest.mark.parametrize(
    ("field", "value", "error", "message"),
    [
        ("frequency", 0.0, ValueError, "must be positive"),
        ("frequency", np.nan, ValueError, "must be finite"),
        (
            "rf_chain_count_by_node",
            {"tx:tx-a": True, "tx:tx-z": 2, "rx:rx-a": 2, "rx:rx-z": 2},
            TypeError,
            "must be an integer",
        ),
        (
            "node_roles",
            {"tx:tx-a": "gnb"},
            TypeError,
            "must be a NodeRole",
        ),
        ("node_roles", {"tx:absent": NodeRole.GNB}, ValueError, "absent from this Sionna scene"),
        (
            "rf_chain_count_by_node",
            {"tx:tx-a": 2, "tx:tx-z": 2, "rx:rx-a": 2, "rx:rx-z": 2, "rx:ghost": 1},
            ValueError,
            "contain a node absent",
        ),
        ("associations", (), ValueError, "at least one"),
        (
            "associations",
            (("tx:tx-a", "rx:rx-a"), ("tx:tx-a", "rx:rx-a")),
            ValueError,
            "must be unique",
        ),
        (
            "associations",
            (("tx:tx-a", "rx:rx-a"),),
            ValueError,
            "cover every selected",
        ),
        (
            "link_roles_by_association",
            {
                (tx, rx): "serving" if (tx, rx) == ("tx:tx-a", "rx:rx-a") else LinkRole.OTHER
                for tx in ("tx:tx-a", "tx:tx-z")
                for rx in ("rx:rx-a", "rx:rx-z")
            },
            TypeError,
            "LinkRole values",
        ),
        (
            "associations",
            (("rx:rx-a", "tx:tx-a"),),
            ValueError,
            "extracted TX followed by an RX",
        ),
    ],
)
def test_scene_extraction_rejects_invalid_contract_fields(field, value, error, message):
    if field == "associations" and value == (("tx:tx-a", "rx:rx-a"),):
        with pytest.raises(error, match=message):
            _extract(
                associations=value,
                link_roles_by_association={},
            )
        return
    with pytest.raises(error, match=message):
        _extract(**{field: value})


def test_scene_extraction_validates_device_maps_and_pattern_labels():
    with pytest.raises(ValueError, match="non-empty device mapping"):
        _device_items({}, name="transmitters")
    with pytest.raises(ValueError, match="invalid device name"):
        _device_items({"": object()}, name="transmitters")
    with pytest.raises(TypeError, match="antenna_pattern.patterns"):
        _pattern_labels(SimpleNamespace(), None, name="tx_pattern_labels")
    array = _array()
    array.antenna_pattern.patterns = []
    with pytest.raises(TypeError, match="antenna_pattern.patterns"):
        _pattern_labels(array, None, name="tx_pattern_labels")
    array.antenna_pattern.patterns = [object(), object()]
    for labels in (("V",), ("V", " "), ("V", 2)):
        with pytest.raises(ValueError, match="label each Sionna antenna pattern"):
            _pattern_labels(array, labels, name="tx_pattern_labels")
    with pytest.raises(ValueError, match="must be unique"):
        _pattern_labels(array, ("V", "V"), name="tx_pattern_labels")


def test_scene_extraction_validates_array_layout_and_pose_contracts():
    array = _array()
    with pytest.raises(TypeError, match="pinned AntennaArray layout API"):
        _array_spec(SimpleNamespace(), ("V",), 1.0, np.eye(3), "array")
    malformed = _array()
    malformed.array_size = 1
    with pytest.raises(ValueError, match="unexpected shape"):
        _array_spec(malformed, ("V", "H"), 1.0, np.eye(3), "array")
    with pytest.raises(ValueError, match="feed count"):
        _array_spec(array, ("V",), 1.0, np.eye(3), "array")
    with pytest.raises(TypeError, match="position must expose"):
        sionna_device_pose(
            SimpleNamespace(position=object(), orientation=_vector(0, 0, 0)), node_id="x"
        )
    with pytest.raises(ValueError, match="three finite components"):
        _vector3(_vector(np.nan, 0, 0), name="position")
    with pytest.raises(TypeError, match="scalar convertible"):
        _scalar([], name="frequency")
    with pytest.raises(ValueError, match="must be finite"):
        _scalar(np.inf, name="frequency")


def test_scene_network_path_binding_requires_exact_roster_and_known_statuses():
    network = _extract()
    expected = {link.link_id for link in network.episode.links}
    with pytest.raises(ValueError, match="exactly every"):
        network.bind_paths({})
    paths = {link_id: object() for link_id in expected}
    with pytest.raises(ValueError, match="unknown extracted"):
        network.bind_paths(paths, solve_status_by_link_id={"unknown": PathSolveStatus.SOLVED})
    bindings = network.bind_paths(
        paths,
        frame_index=2,
        solve_status_by_link_id={link_id: PathSolveStatus.NO_PATH for link_id in expected},
    )
    assert len(bindings) == len(expected)
    assert {binding.frame_index for binding in bindings} == {2}


def test_scene_device_items_and_pose_accept_numpy_backed_values():
    assert len(_device_items({"tx": object()}, name="transmitters")) == 1
    device = _device((1.0, 2.0, 3.0))
    position, orientation = sionna_device_pose(device, node_id="tx")
    np.testing.assert_array_equal(position, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(orientation, np.eye(3))


def test_scene_extraction_preserves_geometry_feeds_roles_and_solver_axis_mapping():
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
        "tx:tx-a",
        "tx:tx-z",
        "rx:rx-a",
        "rx:rx-z",
    ]
    assert [link.link_id for link in episode.links] == [
        "tx:tx-a->rx:rx-a",
        "tx:tx-a->rx:rx-z",
        "tx:tx-z->rx:rx-a",
        "tx:tx-z->rx:rx-z",
    ]
    assert [
        (item.tx_device_index, item.rx_device_index) for item in result.device_link_mappings
    ] == [
        (1, 1),
        (1, 0),
        (0, 1),
        (0, 0),
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


def test_scene_extraction_supports_per_link_serving_and_interference_roles():
    associations = (
        ("tx:tx-a", "rx:rx-a"),
        ("tx:tx-z", "rx:rx-a"),
        ("tx:tx-a", "rx:rx-z"),
        ("tx:tx-z", "rx:rx-z"),
    )
    role_by_pair = {
        ("tx:tx-a", "rx:rx-a"): LinkRole.SERVING,
        ("tx:tx-z", "rx:rx-a"): LinkRole.INTERFERING,
        ("tx:tx-a", "rx:rx-z"): LinkRole.INTERFERING,
        ("tx:tx-z", "rx:rx-z"): LinkRole.SERVING,
    }
    network = sionna_scene_to_network_episode(
        _scene(),
        episode_id="network-episode",
        split_group_id="network-scene",
        family_id="F03-RT-STATIC",
        subcarrier_offsets_hz=(-15e3, 0.0, 15e3),
        node_roles={
            "tx:tx-a": NodeRole.GNB,
            "tx:tx-z": NodeRole.INTERFERER,
            "rx:rx-a": NodeRole.UE,
            "rx:rx-z": NodeRole.UE,
        },
        rf_chain_count_by_node={
            node_id: 2 for node_id in ("tx:tx-a", "tx:tx-z", "rx:rx-a", "rx:rx-z")
        },
        associations=associations,
        link_roles_by_association=role_by_pair,
        direction=LinkDirection.DOWNLINK,
    )

    assert {(link.tx_node_id, link.rx_node_id): link.role for link in network.episode.links} == (
        role_by_pair
    )
    assert tuple(mapping.link_id for mapping in network.device_link_mappings) == tuple(
        link.link_id for link in network.episode.links
    )


def test_scene_extraction_rejects_missing_per_link_role():
    with pytest.raises(ValueError, match="cover every selected"):
        sionna_scene_to_network_episode(
            _scene(),
            episode_id="network-episode",
            split_group_id="network-scene",
            family_id="F03-RT-STATIC",
            rf_chain_count_by_node={
                node_id: 2 for node_id in ("tx:tx-a", "tx:tx-z", "rx:rx-a", "rx:rx-z")
            },
            associations=(("tx:tx-a", "rx:rx-a"),),
            link_roles_by_association={},
        )


def test_scene_extraction_requires_explicit_rf_chain_counts_for_multiport_arrays():
    with pytest.raises(ValueError, match="explicit RF-chain count"):
        sionna_scene_to_network_episode(_scene(), episode_id="e", split_group_id="s", family_id="f")


def test_scene_extraction_rejects_motion_until_pose_timeline_is_supported():
    with pytest.raises(ValueError, match="trajectory-aware"):
        sionna_scene_to_network_episode(
            _scene(moving_tx=True),
            episode_id="e",
            split_group_id="s",
            family_id="f",
            rf_chain_count_by_node={
                "tx:tx-z": 2,
                "tx:tx-a": 2,
                "rx:rx-z": 2,
                "rx:rx-a": 2,
            },
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


def test_reordering_scene_device_insertion_preserves_named_episode_identity():
    first_scene = _scene()
    second_scene = _scene()
    second_scene.transmitters = dict(reversed(tuple(second_scene.transmitters.items())))
    second_scene.receivers = dict(reversed(tuple(second_scene.receivers.items())))
    kwargs = {
        "episode_id": "episode",
        "split_group_id": "scene-family",
        "family_id": "F03-RT-STATIC",
        "subcarrier_offsets_hz": (-30e3, 0.0, 30e3),
        "rf_chain_count_by_node": {
            "tx:tx-z": 2,
            "tx:tx-a": 2,
            "rx:rx-z": 2,
            "rx:rx-a": 2,
        },
        "tx_pattern_labels": ("V", "H"),
        "rx_pattern_labels": ("V", "H"),
        "direction": LinkDirection.DOWNLINK,
        "link_role": LinkRole.SERVING,
    }
    first = sionna_scene_to_network_episode(first_scene, **kwargs)
    second = sionna_scene_to_network_episode(second_scene, **kwargs)

    assert first.episode.nodes == second.episode.nodes
    assert first.episode.links == second.episode.links
    assert [array.array_id for array in first.episode.arrays] == [
        array.array_id for array in second.episode.arrays
    ]
    for first_array, second_array in zip(first.episode.arrays, second.episode.arrays, strict=True):
        assert first_array.port_ids == second_array.port_ids
        np.testing.assert_array_equal(
            first_array.element_positions_local_m,
            second_array.element_positions_local_m,
        )
        np.testing.assert_array_equal(
            first_array.orientation_local_to_world,
            second_array.orientation_local_to_world,
        )
        np.testing.assert_array_equal(
            first_array.port_element_weights,
            second_array.port_element_weights,
        )

    def check_scene_axis_mapping(result, scene):
        tx_indices = {name: index for index, (name, _) in enumerate(scene.transmitters.items())}
        rx_indices = {name: index for index, (name, _) in enumerate(scene.receivers.items())}
        for mapping in result.device_link_mappings:
            link = result.episode.link(mapping.link_id)
            assert mapping.tx_device_index == tx_indices[link.tx_node_id.removeprefix("tx:")]
            assert mapping.rx_device_index == rx_indices[link.rx_node_id.removeprefix("rx:")]

    check_scene_axis_mapping(first, first_scene)
    check_scene_axis_mapping(second, second_scene)
    assert first.device_link_mappings != second.device_link_mappings


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
        (1, 1),
        (1, 0),
        (0, 1),
        (0, 0),
    ]
    np.testing.assert_allclose(
        table.delay_s,
        [
            np.asarray(paths.tau.numpy())[rx, tx, 0]
            for tx, rx in ((1, 1), (1, 0), (0, 1), (0, 0))
            for _ in range(16)
        ],
    )

    reversed_scene = rt.load_scene()
    reversed_scene.frequency = 3.5e9
    reversed_scene.tx_array = rt.PlanarArray(
        num_rows=1, num_cols=2, pattern="iso", polarization="VH"
    )
    reversed_scene.rx_array = rt.PlanarArray(
        num_rows=1, num_cols=2, pattern="iso", polarization="VH"
    )
    reversed_scene.add(rt.Transmitter("tx-a", [2.0, 0.0, 1.0]))
    reversed_scene.add(rt.Transmitter("tx-z", [0.0, 0.0, 1.0]))
    reversed_scene.add(rt.Receiver("rx-a", [12.0, 0.0, 1.0]))
    reversed_scene.add(rt.Receiver("rx-z", [10.0, 0.0, 1.0]))
    reversed_paths = rt.PathSolver(deterministic=True)(
        reversed_scene,
        max_depth=0,
        los=True,
        specular_reflection=False,
        diffuse_reflection=False,
        refraction=False,
        diffraction=False,
        synthetic_array=True,
    )
    reversed_network = sionna_scene_to_network_episode(
        reversed_scene,
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
    reversed_bindings = reversed_network.bind_paths(
        {link.link_id: reversed_paths for link in reversed_network.episode.links}
    )
    reversed_table = sionna_bindings_to_episode_oracle(reversed_network.episode, reversed_bindings)
    assert reversed_network.episode.nodes == network.episode.nodes
    assert reversed_network.episode.links == network.episode.links
    assert reversed_table.link_ids == table.link_ids
    for attribute in (
        "frame_indices",
        "rx_port_indices",
        "tx_port_indices",
        "path_offsets",
        "delay_s",
        "complex_gain",
        "interaction_order",
        "doppler_hz",
        "aoa_azimuth_elevation_rad",
        "aod_azimuth_elevation_rad",
        "interaction_offsets",
        "interaction_types",
        "object_ids",
        "primitive_ids",
        "interaction_vertices_m",
    ):
        np.testing.assert_allclose(getattr(reversed_table, attribute), getattr(table, attribute))
    assert reversed_table.solve_status == table.solve_status
    assert reversed_table.path_kind == table.path_kind
    assert reversed_table.interaction_vertex_frame == table.interaction_vertex_frame


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
