"""Whole-episode Sionna-to-UMCD device/link mapping tests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import jsonschema
import numpy as np
import pytest
from datasets.channel_view import (
    ChannelDelayReference,
    ChannelPowerScale,
    PhysicalChannelView,
)
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
from datasets.provenance import UMCDRunProvenance
from datasets.targets import TargetValidity
from integrations.sionna.rt_network import (
    SionnaPathBinding,
    build_rt_network_generation_request,
    build_sionna_rt_episode_bundle,
    build_sionna_rt_episode_bundle_from_paths,
    episode_channels_to_network_phy_input,
    network_episode_identity,
    run_authorized_static_network_episode,
    sionna_bindings_to_channel_views,
    sionna_bindings_to_episode_oracle,
    write_sionna_rt_episode_bundle,
    write_sionna_rt_episode_bundle_from_paths,
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


def _two_tx_two_rx_episode_and_views():
    array = AntennaArraySpec(
        "siso",
        np.zeros((1, 3)),
        ("V",),
        np.eye(3),
        ("port-0",),
        np.ones((1, 1), dtype=np.complex128),
    )
    links = tuple(
        DirectedLink(
            f"{tx}-to-{rx}",
            tx,
            rx,
            (0,),
            (0,),
            LinkDirection.DOWNLINK,
            (
                LinkRole.SERVING
                if (tx, rx) in (("tx-0", "rx-0"), ("tx-1", "rx-1"))
                else LinkRole.INTERFERING
            ),
        )
        for tx in ("tx-0", "tx-1")
        for rx in ("rx-0", "rx-1")
    )
    episode = NetworkEpisode(
        "network-episode",
        "network-site",
        "F03-RT-STATIC",
        3.5e9,
        (array,),
        (
            NetworkNode("tx-0", NodeRole.GNB, "siso", (0.0, 0.0, 1.0), 1),
            NetworkNode("tx-1", NodeRole.GNB, "siso", (2.0, 0.0, 1.0), 1),
            NetworkNode("rx-0", NodeRole.UE, "siso", (10.0, 0.0, 1.0), 1),
            NetworkNode("rx-1", NodeRole.UE, "siso", (12.0, 0.0, 1.0), 1),
        ),
        links,
        (0.0, 0.1),
        np.asarray([-15e3, 0.0, 15e3]),
    )
    coefficients = {
        "tx-0-to-rx-0": 1.0 + 0.0j,
        "tx-1-to-rx-0": 0.5 + 0.0j,
        "tx-0-to-rx-1": 0.25 + 0.0j,
        "tx-1-to-rx-1": 1.0 + 0.0j,
    }
    views = tuple(
        PhysicalChannelView(
            view_id=f"{link.link_id}:cfr",
            link_id=link.link_id,
            channel=np.full((2, 1, 1, 3), coefficients[link.link_id], dtype=np.complex128),
            frame_times_s=episode.frame_times_s,
            rx_port_ids=("port-0",),
            tx_port_ids=("port-0",),
            carrier_hz=episode.carrier_hz,
            subcarrier_offsets_hz=episode.subcarrier_offsets_hz,
            delay_reference=ChannelDelayReference.ABSOLUTE_PROPAGATION,
            power_scale=ChannelPowerScale.ABSOLUTE,
        )
        for link in links
    )
    return episode, views


def test_stored_multilink_episode_flows_through_network_phy_composition(tmp_path):
    torch = pytest.importorskip("torch")
    from integrations.sionna.network_phy import compose_downlink_resource_grids, network_sinr

    episode, views = _two_tx_two_rx_episode_and_views()
    stored = write_episode_bundle(tmp_path, EpisodeBundle(episode, None, views, ()))
    restored = read_episode_bundle(tmp_path, stored)
    network_input = episode_channels_to_network_phy_input(
        restored.episode, restored.channel_views, ofdm_symbol_count=2
    )

    assert network_input.episode_id == episode.episode_id
    assert network_input.tx_node_ids == ("tx-0", "tx-1")
    assert network_input.rx_node_ids == ("rx-0", "rx-1")
    assert network_input.serving_tx_indices == (0, 1)
    assert network_input.channel.shape == (2, 2, 1, 2, 1, 2, 3)
    torch_channel = torch.tensor(network_input.channel, dtype=torch.complex64)
    grids = torch.ones((2, 2, 1, 2, 3), dtype=torch.complex64)
    composition = compose_downlink_resource_grids(torch_channel, grids, (1.0, 1.0))
    measured = network_sinr(composition, network_input.serving_tx_indices, (0.1, 0.1))

    torch.testing.assert_close(
        measured.desired_power_w_per_re,
        torch.tensor([1.0, 1.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        measured.interference_power_w_per_re,
        torch.tensor([0.25, 0.0625], dtype=torch.float64),
    )


def test_sionna_bindings_produce_episode_link_channel_views_with_explicit_axes():
    episode = _episode()

    class FakePaths:
        def cfr(self, **kwargs):
            assert kwargs["normalize_delays"] is False
            assert kwargs["normalize"] is False
            assert kwargs["num_time_steps"] == 1
            assert kwargs["out_type"] == "numpy"
            # [rx, rx_port, tx, tx_port, time, subcarrier]
            response = np.empty((2, 2, 1, 2, 1, 3), dtype=np.complex128)
            for rx in range(2):
                for rx_port in range(2):
                    for tx_port in range(2):
                        response[rx, rx_port, 0, tx_port, 0] = (
                            100 * rx + 10 * rx_port + tx_port + 1j
                        )
            return response

    paths = FakePaths()
    bindings = tuple(
        SionnaPathBinding(
            link_id=link.link_id,
            frame_index=frame,
            rx_device_index=int(link.rx_node_id[-1]),
            tx_device_index=0,
            paths=paths,
        )
        for link in episode.links
        for frame in range(len(episode.frame_times_s))
    )
    views = sionna_bindings_to_channel_views(episode, bindings)
    assert tuple(view.link_id for view in views) == tuple(link.link_id for link in episode.links)
    for view in views:
        assert view.channel.shape == (2, 2, 2, 3)
        assert view.delay_reference is ChannelDelayReference.ABSOLUTE_PROPAGATION
        assert view.power_scale is ChannelPowerScale.ABSOLUTE
        assert np.all(view.valid_mask)
        # Episode links reorder both physical array-port axes as (1, 0).
        rx = int(episode.link(view.link_id).rx_node_id[-1])
        np.testing.assert_array_equal(
            view.channel[0, :, :, 0],
            np.asarray(
                [
                    [100 * rx + 11 + 1j, 100 * rx + 10 + 1j],
                    [100 * rx + 1 + 1j, 100 * rx + 0 + 1j],
                ]
            ),
        )


def test_sionna_binding_channel_views_mask_failed_solve_instead_of_zero_fill():
    episode = _episode()
    bindings = tuple(
        SionnaPathBinding(
            link_id=link.link_id,
            frame_index=frame,
            rx_device_index=int(link.rx_node_id[-1]),
            tx_device_index=0,
            paths=None if frame == 1 else _FakeCFRPaths(),
            solve_status=PathSolveStatus.SOLVER_FAILURE if frame == 1 else None,
        )
        for link in episode.links
        for frame in range(len(episode.frame_times_s))
    )
    views = sionna_bindings_to_channel_views(episode, bindings)
    for view in views:
        assert view.valid_mask[0].all()
        assert not view.valid_mask[1].any()
        assert np.count_nonzero(view.channel[1]) == 0


def test_bundle_from_paths_connects_derived_views_to_bundle_builder(monkeypatch):
    episode = _episode()
    bindings = tuple(
        SionnaPathBinding(
            link_id=link.link_id,
            frame_index=frame,
            rx_device_index=int(link.rx_node_id[-1]),
            tx_device_index=0,
            paths=None,
            solve_status=PathSolveStatus.UNCONVERGED,
        )
        for link in episode.links
        for frame in range(len(episode.frame_times_s))
    )
    captured = {}

    def fake_builder(episode_arg, bindings_arg, **kwargs):
        captured.update(episode=episode_arg, bindings=bindings_arg, **kwargs)
        return "built"

    monkeypatch.setattr(
        "integrations.sionna.rt_network.build_sionna_rt_episode_bundle", fake_builder
    )
    result = build_sionna_rt_episode_bundle_from_paths(
        episode,
        bindings,
        run_identity={"run": "fixture"},
        material_config={},
        source_family="fixture",
        producer="fixture",
    )

    assert result == "built"
    assert captured["episode"] is episode
    assert captured["bindings"] == bindings
    assert len(captured["channel_views"]) == len(episode.links)
    assert all(not view.valid_mask.any() for view in captured["channel_views"])


def test_bundle_builder_derives_rms_targets_from_shared_path_oracle(monkeypatch):
    episode, _views = _two_tx_two_rx_episode_and_views()
    group_keys = tuple(
        (link.link_id, frame)
        for link in episode.links
        for frame in range(len(episode.frame_times_s))
    )
    oracle = PathOracleTable(
        link_ids=tuple(link for link, _frame in group_keys),
        frame_indices=np.asarray([frame for _link, frame in group_keys]),
        rx_port_indices=np.zeros(len(group_keys), dtype=np.int64),
        tx_port_indices=np.zeros(len(group_keys), dtype=np.int64),
        path_offsets=np.zeros(len(group_keys) + 1, dtype=np.int64),
        solve_status=(PathSolveStatus.NO_PATH,) * len(group_keys),
        delay_s=np.empty(0, dtype=np.float64),
        complex_gain=np.empty(0, dtype=np.complex128),
        interaction_order=np.empty(0, dtype=np.int64),
    )
    bindings = tuple(
        SionnaPathBinding(
            link_id=link.link_id,
            frame_index=frame,
            rx_device_index=int(link.rx_node_id[-1]),
            tx_device_index=0,
            paths=None,
            solve_status=PathSolveStatus.UNCONVERGED,
        )
        for link in episode.links
        for frame in range(len(episode.frame_times_s))
    )
    monkeypatch.setattr(
        "integrations.sionna.rt_network.sionna_bindings_to_episode_oracle",
        lambda _episode, _bindings: oracle,
    )
    monkeypatch.setattr(
        "integrations.sionna.rt_network.umcd_provenance_from_rt_identity",
        lambda run_identity, **kwargs: UMCDRunProvenance(
            record_id=kwargs["record_id"],
            subject_layer=kwargs["subject_layer"],
            source_family=kwargs["source_family"],
            producer=kwargs["producer"],
            package_source_sha256="a" * 64,
            software_versions={"fixture": "1"},
            backend={"backend": "fixture"},
            scene_asset_sha256={},
            seed=1,
            physical_config={},
            solver_config={},
            material_config=kwargs["material_config"],
            run_identity_sha256="b" * 64,
        ),
    )

    bundle = build_sionna_rt_episode_bundle(
        episode,
        bindings,
        run_identity={"fixture": True},
        material_config={},
        source_family="fixture",
        producer="fixture",
        derive_rms_delay_spread_targets=True,
    )
    assert len(bundle.targets) == len(episode.links)
    for target, link in zip(bundle.targets, episode.links, strict=True):
        assert target.entity_id == link.link_id
        assert target.definition.definition_version == "umcd.rms_delay_spread.v2"
        assert target.definition.value_axes == ("frame",)
        assert target.reason_codes == (TargetValidity.NOT_OBSERVABLE,) * len(episode.frame_times_s)


def test_write_bundle_from_paths_applies_d1_gate_and_uses_immutable_writer(monkeypatch, tmp_path):
    episode = _episode()
    first_link = episode.links[0]
    expected_bundle = EpisodeBundle(
        episode,
        None,
        (
            PhysicalChannelView(
                view_id="fixture-channel",
                link_id=first_link.link_id,
                channel=np.ones((2, 2, 2, 3), dtype=np.complex64),
                frame_times_s=episode.frame_times_s,
                rx_port_ids=("rx1", "rx0"),
                tx_port_ids=("tx1", "tx0"),
                carrier_hz=episode.carrier_hz,
                subcarrier_offsets_hz=episode.subcarrier_offsets_hz,
                delay_reference=ChannelDelayReference.ABSOLUTE_PROPAGATION,
                power_scale=ChannelPowerScale.ABSOLUTE,
            ),
        ),
        (),
    )
    expected_stored = object()
    gate_calls = []
    writer_calls = []

    monkeypatch.setattr(
        "integrations.sionna.rt_network.build_sionna_rt_episode_bundle_from_paths",
        lambda *args, **kwargs: expected_bundle,
    )
    monkeypatch.setattr(
        "integrations.sionna.rt_network.write_episode_bundle",
        lambda root, bundle: writer_calls.append((root, bundle)) or expected_stored,
    )
    monkeypatch.setattr(
        "integrations.sionna.d1_generation_gate.require_pilot_destination",
        lambda authorization, root, *, publisher: gate_calls.append(
            (authorization, root, publisher)
        ),
    )

    stored = write_sionna_rt_episode_bundle_from_paths(
        tmp_path,
        episode,
        (),
        run_identity={},
        material_config={},
        source_family="fixture",
        producer="fixture",
        authorization="authorized-fixture",
    )

    assert stored is expected_stored
    assert gate_calls == [("authorized-fixture", tmp_path, "umcd_bundle")]
    assert writer_calls == [(tmp_path, expected_bundle)]


class _FakeCFRPaths:
    def cfr(self, **kwargs):
        del kwargs
        return np.ones((2, 2, 1, 2, 1, 3), dtype=np.complex64)


def test_network_generation_request_binds_topology_and_is_d1_authorizable(tmp_path):
    from integrations.sionna.d1_generation_gate import authorize_d1_generation

    episode, _ = _two_tx_two_rx_episode_and_views()
    assets = {
        "scene_xml_path": "fixture-scene.xml",
        "asset_count": 1,
        "asset_set_sha256": "a" * 64,
        "complete": True,
    }
    backend = {
        "backend": "llvm_cpu",
        "variant": "llvm_ad_mono_polarized",
        "probe_passed": True,
        "runtime_versions": {"sionna-rt": "2.2.0"},
    }
    request = build_rt_network_generation_request(
        episode,
        scene_asset_manifest=assets,
        scene_geometry_identity={
            "scene_geometry_sha256": "c" * 64,
            "scene_object_count": 2,
            "scene_triangle_count": 12,
        },
        material_config={"wall": {"scattering": 0.2}},
        backend_selection=backend,
        seed=52,
        samples_per_src=16,
        max_depth=2,
        max_num_paths_per_src=1024,
        destinations=(tmp_path,),
    )
    authorization = authorize_d1_generation(request, pilot=True)

    assert authorization.operation == "rt_data"
    assert request["network_episode_identity"] == network_episode_identity(episode)
    assert request["config"]["total_work_units"] == 32
    changed_links = tuple(
        replace(link, role=LinkRole.INTERFERING)
        if link.link_id == episode.links[0].link_id
        else link
        for link in episode.links
    )
    changed_episode = replace(episode, links=changed_links)
    assert (
        network_episode_identity(changed_episode)["sha256"]
        != request["network_episode_identity"]["sha256"]
    )


def test_network_generation_request_rejects_incomplete_or_invalid_topology(tmp_path):
    episode, _ = _two_tx_two_rx_episode_and_views()
    incomplete = replace(episode, links=episode.links[:-1])
    with pytest.raises(ValueError, match="every TX/RX pair"):
        build_rt_network_generation_request(
            incomplete,
            scene_asset_manifest={
                "scene_xml_path": "fixture-scene.xml",
                "asset_count": 1,
                "asset_set_sha256": "a" * 64,
                "complete": True,
            },
            scene_geometry_identity={
                "scene_geometry_sha256": "c" * 64,
                "scene_object_count": 2,
                "scene_triangle_count": 12,
            },
            material_config={},
            backend_selection={"probe_passed": True},
            seed=52,
            samples_per_src=16,
            max_depth=2,
            max_num_paths_per_src=1024,
            destinations=(tmp_path,),
        )


@pytest.mark.parametrize(
    (
        "solver_error",
        "candidate_counts",
        "expected_status",
        "expected_binding_status",
        "expected_path_status",
    ),
    [
        (
            None,
            (1, 1),
            "SOLVED",
            None,
            PathSolveStatus.SOLVED,
        ),
        (
            None,
            (1024, 512),
            "UNCONVERGED",
            PathSolveStatus.UNCONVERGED,
            PathSolveStatus.UNCONVERGED,
        ),
        (
            RuntimeError("synthetic PathSolver failure"),
            None,
            "SOLVER_FAILURE",
            PathSolveStatus.SOLVER_FAILURE,
            PathSolveStatus.SOLVER_FAILURE,
        ),
    ],
)
def test_authorized_network_solver_publishes_cap_and_solver_failures(
    monkeypatch,
    tmp_path,
    solver_error,
    candidate_counts,
    expected_status,
    expected_binding_status,
    expected_path_status,
):
    from types import SimpleNamespace

    from integrations.sionna.d1_generation_gate import authorize_d1_generation
    from integrations.sionna.rt_identity import rt_run_identity
    from integrations.sionna.scene_assets import scene_asset_manifest

    episode, _ = _two_tx_two_rx_episode_and_views()
    episode = replace(episode, frame_times_s=(0.0,))
    scene_xml = tmp_path / "scene.xml"
    scene_xml.write_text("<scene></scene>", encoding="utf-8")
    assets = scene_asset_manifest(scene_xml)
    backend = {
        "backend": "llvm_cpu",
        "variant": "llvm_ad_mono_polarized",
        "probe_passed": True,
        "runtime_versions": {"sionna-rt": "2.2.0"},
    }
    material = {"wall": {"scattering": 0.2}}
    geometry = {
        "scene_geometry_sha256": "c" * 64,
        "scene_object_count": 2,
        "scene_triangle_count": 12,
    }
    request = build_rt_network_generation_request(
        episode,
        scene_asset_manifest=assets,
        scene_geometry_identity=geometry,
        material_config=material,
        backend_selection=backend,
        seed=52,
        samples_per_src=16,
        max_depth=2,
        max_num_paths_per_src=1024,
        destinations=(tmp_path,),
    )
    authorization = authorize_d1_generation(request, pilot=True)
    identity = rt_run_identity(
        scene_asset_manifest=assets,
        physical_config={
            "network_episode_identity": request["network_episode_identity"],
            "scene_geometry": geometry,
        },
        material_config=material,
        solver_config=request["config"]["solver_config"],
        seed=request["seed"],
        backend_selection=backend,
        runtime_versions={"python": "3.12"},
        package_source_sha256="b" * 64,
    )

    class Network:
        def __init__(self):
            self.episode = episode
            self.device_link_mappings = tuple(
                SimpleNamespace(link_id=link.link_id) for link in episode.links
            )

        def bind_paths(self, paths_by_link_id, *, frame_index=0, solve_status_by_link_id=None):
            statuses = {} if solve_status_by_link_id is None else solve_status_by_link_id
            return tuple(
                SionnaPathBinding(
                    link_id=mapping.link_id,
                    frame_index=frame_index,
                    rx_device_index=int(episode.link(mapping.link_id).rx_node_id[-1]),
                    tx_device_index=int(episode.link(mapping.link_id).tx_node_id[-1]),
                    paths=paths_by_link_id[mapping.link_id],
                    solve_status=statuses.get(mapping.link_id),
                )
                for mapping in self.device_link_mappings
            )

    class Solver:
        def __init__(self, *, deterministic):
            assert deterministic is True

        def __call__(self, scene_arg, **kwargs):
            del scene_arg, kwargs
            if solver_error is not None:
                raise solver_error

            class FakePaths:
                def cfr(self, **cfr_kwargs):
                    del cfr_kwargs
                    return np.ones((2, 1, 2, 1, 1, 3), dtype=np.complex64)

            return FakePaths()

    def fake_path_table(paths, **kwargs):
        del paths
        link_id = kwargs["link_id"]
        frame_index = kwargs["frame_index"]
        return PathOracleTable(
            link_ids=(link_id,),
            frame_indices=np.asarray([frame_index]),
            rx_port_indices=np.asarray([0]),
            tx_port_indices=np.asarray([0]),
            path_offsets=np.asarray([0, 1]),
            solve_status=(kwargs["solve_status"] or PathSolveStatus.SOLVED,),
            delay_s=np.asarray([1e-7]),
            complex_gain=np.asarray([1.0 + 0.0j]),
            interaction_order=np.asarray([0]),
        )

    monkeypatch.setattr(
        "integrations.sionna.rt_network._validate_live_static_network_scene",
        lambda scene_arg, episode_arg, request_arg: None,
    )
    monkeypatch.setattr(
        "integrations.sionna.rt_network.sionna_paths_to_oracle_table", fake_path_table
    )
    backend_selection = SimpleNamespace(as_dict=lambda: backend)
    result = run_authorized_static_network_episode(
        tmp_path,
        object(),
        Network(),
        request=request,
        authorization=authorization,
        run_identity=identity,
        material_config=material,
        backend_selection=backend_selection,
        source_family="F03-RT-STATIC",
        producer="test-fixture",
        _path_solver_factory=Solver,
        _candidate_counter_factory=lambda solver: SimpleNamespace(
            last_candidate_path_counts_per_source=candidate_counts
        ),
    )

    assert result.status == expected_status
    assert result.stored_bundle.episode_id == episode.episode_id
    assert all(binding.solve_status is expected_binding_status for binding in result.bindings)
    stored = read_episode_bundle(tmp_path, result.stored_bundle)
    assert set(stored.path_oracle.solve_status) == {expected_path_status}
    assert all(
        bool(view.valid_mask.any()) == (expected_status == "SOLVED")
        for view in stored.channel_views
    )
    receipt = json.loads(result.receipt_path.read_text(encoding="utf-8"))
    schema = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "datasets/schema/static-network-rt-generation-receipt.v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker()).validate(
        receipt
    )
    assert receipt["status"] == expected_status
    assert receipt["candidate_path_counts_per_source"] == (
        None if candidate_counts is None else list(candidate_counts)
    )


def test_live_scene_revalidation_detects_changed_device_pose(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from integrations.sionna.array_layouts import sionna_scene_to_network_episode
    from integrations.sionna.rt_network import _validate_live_static_network_scene

    def vector(x, y, z):
        return SimpleNamespace(x=np.asarray(x), y=np.asarray(y), z=np.asarray(z))

    def device(position):
        return SimpleNamespace(
            position=vector(*position),
            orientation=vector(0.0, 0.0, 0.0),
            velocity=vector(0.0, 0.0, 0.0),
        )

    array = SimpleNamespace(
        normalized_positions=vector([0.0], [0.0], [0.0]),
        array_size=1,
        num_ant=1,
        antenna_pattern=SimpleNamespace(patterns=[object()]),
    )
    scene = SimpleNamespace(
        transmitters={"tx": device((0.0, 0.0, 10.0))},
        receivers={"rx": device((10.0, 0.0, 1.0))},
        tx_array=array,
        rx_array=array,
        frequency=3.5e9,
    )
    network = sionna_scene_to_network_episode(
        scene,
        episode_id="live-scene-episode",
        split_group_id="live-scene-family",
        family_id="F03-RT-STATIC",
        subcarrier_offsets_hz=(-15e3, 0.0, 15e3),
        node_roles={"tx:tx": NodeRole.GNB, "rx:rx": NodeRole.UE},
        associations=(("tx:tx", "rx:rx"),),
        link_roles_by_association={("tx:tx", "rx:rx"): LinkRole.SERVING},
        direction=LinkDirection.DOWNLINK,
        link_role=LinkRole.SERVING,
    )
    scene_xml = tmp_path / "live-scene.xml"
    scene_xml.write_text("<scene></scene>", encoding="utf-8")
    assets = {
        "scene_xml_path": str(scene_xml),
        "asset_count": 1,
        "asset_set_sha256": "a" * 64,
        "complete": True,
    }
    material = {"surface": "fixture"}
    geometry = {
        "scene_geometry_sha256": "c" * 64,
        "scene_object_count": 1,
        "scene_triangle_count": 2,
    }
    request = build_rt_network_generation_request(
        network.episode,
        scene_asset_manifest=assets,
        scene_geometry_identity=geometry,
        material_config=material,
        backend_selection={"backend": "llvm_cpu", "variant": "llvm", "probe_passed": True},
        seed=1,
        samples_per_src=1,
        max_depth=1,
        max_num_paths_per_src=32,
        destinations=(tmp_path,),
    )
    monkeypatch.setattr(
        "integrations.sionna.scene_assets.scene_asset_manifest", lambda path: assets
    )
    monkeypatch.setattr(
        "integrations.sionna.scene_assets.sionna_scene_material_manifest",
        lambda scene_arg: material,
    )
    monkeypatch.setattr(
        "propagation.scene_visibility.extract_scene_triangle_inventory",
        lambda scene_arg: SimpleNamespace(
            scene_geometry_sha256=geometry["scene_geometry_sha256"],
            object_names=("fixture",),
            triangles_m=np.zeros((2, 3, 3)),
        ),
    )

    _validate_live_static_network_scene(scene, network.episode, request)
    scene.transmitters["tx"].position = vector(0.5, 0.0, 10.0)
    with pytest.raises(PermissionError, match="device, array or pose state"):
        _validate_live_static_network_scene(scene, network.episode, request)


def test_network_phy_episode_adapter_rejects_incomplete_topology():
    episode, views = _two_tx_two_rx_episode_and_views()
    incomplete = replace(episode, links=episode.links[:-1])

    with pytest.raises(ValueError, match="every TX/RX pair"):
        episode_channels_to_network_phy_input(incomplete, views[:-1], ofdm_symbol_count=1)


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
    asset_rows = [
        {
            "path": "scene.xml",
            "bytes": 10,
            "sha256": hashlib.sha256(b"scene bytes").hexdigest(),
        }
    ]
    identity = rt_run_identity(
        scene_asset_manifest={
            "scene_xml_path": "/fixture/scene.xml",
            "asset_count": 1,
            "assets": asset_rows,
            "asset_set_sha256": hashlib.sha256(
                json.dumps(asset_rows, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "unresolved_references": [],
            "complete": True,
        },
        physical_config={"carrier_hz": 3.5e9},
        material_config={"wall": {"scattering_coefficient": 0.2}},
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
