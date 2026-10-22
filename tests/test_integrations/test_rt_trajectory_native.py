"""Native LLVM RT and Sionna PHY integration for stored NR trajectories."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.constants import speed_of_light

torch = pytest.importorskip("torch")
pytest.importorskip("mitsuba")
pytest.importorskip("sionna")

from datasets.episode_store import StoredEpisodeBundle, read_episode_bundle  # noqa: E402
from datasets.nr_store import read_nr_trajectory  # noqa: E402
from integrations.sionna.nr_timing import nr_normal_cp_symbol_start_times  # noqa: E402
from integrations.sionna.rt_trajectory import (  # noqa: E402
    RTSceneObjectMotion,
    RTTrajectoryConfig,
    generate_rt_nr_trajectory,
)


@pytest.mark.native_sionna
def test_cpu_rt_path_retraces_and_stores_receiver_only_trajectory(tmp_path, monkeypatch):
    config = RTTrajectoryConfig(
        trajectory_id="native-simple-reflector-route",
        scene="simple_reflector",
        transmitter_position_m=(-10.0, 0.0, 10.0),
        receiver_start_m=(10.0, 0.0, 10.0),
        receiver_velocity_mps=(0.2, 0.0, 0.0),
        num_steps=2,
        sample_interval_s=0.1,
        max_depth=1,
        samples_per_src=200,
        seed=20261021,
        device="cpu",
    )

    umcd_root = tmp_path / "umcd"
    stored = generate_rt_nr_trajectory(config, tmp_path, umcd_store_root=umcd_root)
    trajectory = read_nr_trajectory(tmp_path, stored)

    assert trajectory.trajectory_id == config.trajectory_id
    assert trajectory.sample_times_s == (0.0, 0.1)
    assert len(trajectory.records) == 2
    assert {record.provenance["rt_seed"] for record in trajectory.records} == {config.seed}
    for record in trajectory.records:
        model_inputs = record.observation.model_inputs()
        assert tuple(model_inputs.received_grid.shape)[0] == 1
        assert tuple(model_inputs.h_hat.shape)[0] == 1
        assert tuple(record.targets["channel_cfr"].shape) == (72,)
        symbol_channels = record.targets["channel_cfr_ofdm_symbols"]
        evolution = record.provenance["channel_time_evolution"]
        assert symbol_channels.shape == (
            record.observation.received_grid.shape[-2],
            record.observation.received_grid.shape[-1],
        )
        expected_symbol_times_s = nr_normal_cp_symbol_start_times(
            config.subcarrier_spacing_khz * 1_000.0,
            symbol_channels.shape[0],
            slot_index_in_subframe=config.slot_index_in_subframe,
        )
        np.testing.assert_allclose(
            evolution["symbol_start_times_s"], expected_symbol_times_s, rtol=0, atol=1e-15
        )
        assert evolution["slot_index_in_subframe"] == config.slot_index_in_subframe
        assert "TS 38.211 clause 5.3.1" in evolution["symbol_timing_policy"]
        assert evolution["sionna_phy_grid_scalar_cp_samples"] > 0
        assert not np.allclose(symbol_channels, symbol_channels[0:1], rtol=0.0, atol=1e-8)
        assert bool(np.asarray(record.target_masks["channel_cfr"]).all())
        assert bool(np.asarray(record.target_masks["geometric_los"]).all())
        assert record.provenance["target_semantics"]["geometric_los"].startswith(
            "independent Mitsuba"
        )
        assert (
            record.target_definitions["geometric_los"].source
            == "mitsuba_segment_intersection_oracle"
        )
        assert record.target_definitions["channel_cfr"].units.startswith("complex channel gain")
        assert not hasattr(model_inputs, "targets")
        assert not hasattr(model_inputs, "provenance")
        assert np.isfinite(model_inputs.received_grid).all()
        assert record.provenance["scene_asset_manifest_complete"] is True
        assert record.provenance["scene_asset_count"] >= 1

    sidecar = tmp_path / "manifests" / "trajectories" / f"{stored.content_sha256}.generation.json"
    assert sidecar.is_file()
    import json

    generation = json.loads(sidecar.read_text(encoding="utf-8"))
    umcd_ref = StoredEpisodeBundle(**generation["umcd_bundle"])
    bundle = read_episode_bundle(umcd_root, umcd_ref)
    assert bundle.episode.family_id == "F04-RT-DYNAMIC"
    assert bundle.episode.frame_times_s == (0.0, 0.1)
    assert bundle.path_oracle.group_count == 2
    assert bundle.path_oracle.path_count > 0
    assert len(bundle.receiver_observations) == 2
    assert len(bundle.channel_views) == 1
    assert bundle.channel_views[0].delay_reference.value == "absolute_propagation_delay"
    assert [target.definition.semantic for target in bundle.targets] == [
        "geometric direct-path visibility"
    ]
    assert {record.subject_layer for record in bundle.provenance_records} == {
        "oracle_paths",
        "oracle_channels",
        "targets",
        "receiver_observations",
    }
    assert bundle.provenance_records[0].material_config["materials"]
    assert (
        bundle.provenance_records[0].run_identity_sha256
        == generation["rt_run_identity"]["cache_key_sha256"]
    )
    material_state = generation["rt_run_identity"]["inputs"]["physical_config"]["material_state"]
    assert material_state["schema"] == "sionna_rt_material_state.v1"
    assert material_state["materials"]["reflector-mat"]["properties"]["scattering_coefficient"] == [
        0.0
    ]
    assert "reflector-mat" in material_state["object_material_bindings"].values()
    assert material_state["materials"]["reflector-mat"]["frequency_update_callback"]["code_sha256"]
    path_table = bundle.path_oracle
    assert path_table.doppler_hz is not None
    direct_gains: list[complex] = []
    direct_dopplers: list[float] = []
    for frame_index, position in enumerate(config.positions_and_times()[0]):
        group = int(np.flatnonzero(path_table.frame_indices == frame_index)[0])
        path_slice = path_table.group_slice(group)
        direct_indices = np.flatnonzero(path_table.interaction_order[path_slice] == 0)
        assert direct_indices.size == 1
        path_index = path_slice.start + int(direct_indices[0])
        direction = np.asarray(position) - np.asarray(config.transmitter_position_m)
        direction /= np.linalg.norm(direction)
        expected_doppler_hz = (
            -config.carrier_hz
            / speed_of_light
            * float(np.dot(direction, config.receiver_velocity_mps))
        )
        np.testing.assert_allclose(
            path_table.doppler_hz[path_index], expected_doppler_hz, rtol=2e-3, atol=1e-4
        )
        direct_gains.append(
            complex(path_table.complex_gain[path_index])
            * np.exp(-2j * np.pi * config.carrier_hz * path_table.delay_s[path_index])
        )
        direct_dopplers.append(float(path_table.doppler_hz[path_index]))
    phase_rate_hz = np.angle(direct_gains[1] / direct_gains[0]) / (
        2.0 * np.pi * config.sample_interval_s
    )
    np.testing.assert_allclose(
        phase_rate_hz,
        np.mean(direct_dopplers),
        rtol=2e-2,
        atol=0.02,
    )
    import sionna.rt as rt

    def unexpected_trace(*args, **kwargs):
        raise AssertionError("matching trajectory cache entry should avoid retracing")

    monkeypatch.setattr(rt, "PathSolver", unexpected_trace)
    cached = generate_rt_nr_trajectory(config, tmp_path)
    assert cached == stored

    with pytest.raises(ValueError, match="different RT run identity"):
        generate_rt_nr_trajectory(replace(config, snr_db=config.snr_db + 1.0), tmp_path)


@pytest.mark.native_sionna
def test_moving_scene_object_is_retraced_and_recorded_per_frame(tmp_path):
    config = RTTrajectoryConfig(
        trajectory_id="native-moving-screen-route",
        scene="box_one_screen",
        transmitter_position_m=(-2.0, 0.0, 2.5),
        receiver_start_m=(2.0, 0.0, 2.5),
        receiver_velocity_mps=(0.0, 0.0, 0.0),
        num_steps=2,
        sample_interval_s=1.0,
        max_depth=2,
        samples_per_src=200,
        seed=20261021,
        device="cpu",
        moving_scene_objects=(RTSceneObjectMotion("screen", (0.0, 10.0, 0.0)),),
    )

    stored = generate_rt_nr_trajectory(config, tmp_path)
    trajectory = read_nr_trajectory(tmp_path, stored)

    poses = [record.provenance["moving_scene_objects"]["screen"] for record in trajectory.records]
    assert poses[0]["position_m"] == [0.0, 0.0, 2.5]
    assert poses[1]["position_m"] == [0.0, 10.0, 2.5]
    assert all(pose["velocity_mps"] == [0.0, 10.0, 0.0] for pose in poses)
    assert [record.provenance["geometric_los_oracle"] for record in trajectory.records] == [
        False,
        True,
    ]
    assert [
        record.provenance["geometric_los_geometry_oracle_state"] for record in trajectory.records
    ] == [
        "blocked",
        "visible",
    ]
    assert [record.provenance["direct_path_count"] for record in trajectory.records] == [0, 1]
    assert all(record.provenance["full_path_count"] > 0 for record in trajectory.records)
    first_cfr = trajectory.records[0].targets["channel_cfr"]
    second_cfr = trajectory.records[1].targets["channel_cfr"]
    assert not np.allclose(first_cfr, second_cfr, rtol=1e-4, atol=1e-7)


@pytest.mark.native_sionna
def test_mixed_outage_trajectory_uses_noise_only_and_masks_channel_truth(tmp_path):
    config = RTTrajectoryConfig(
        trajectory_id="native-mixed-screen-outage-route",
        scene="box_one_screen",
        transmitter_position_m=(-2.0, 0.0, 2.5),
        receiver_start_m=(2.0, 0.0, 2.5),
        receiver_velocity_mps=(0.0, 0.0, 0.0),
        num_steps=2,
        sample_interval_s=1.0,
        max_depth=0,
        samples_per_src=200,
        seed=20261021,
        device="cpu",
        moving_scene_objects=(RTSceneObjectMotion("screen", (0.0, 10.0, 0.0)),),
        no_path_policy="noise_only_mixed_episode",
    )
    umcd_root = tmp_path / "umcd"
    reject_config = replace(
        config,
        trajectory_id="native-rejected-screen-outage-route",
        moving_scene_objects=(),
        no_path_policy="reject",
    )
    with pytest.raises(RuntimeError, match="no_path_policy='noise_only_mixed_episode'"):
        generate_rt_nr_trajectory(reject_config, tmp_path / "reject")
    all_outage_config = replace(
        reject_config,
        trajectory_id="native-all-screen-outage-route",
        no_path_policy="noise_only_mixed_episode",
    )
    with pytest.raises(RuntimeError, match="all-no-path trajectory"):
        generate_rt_nr_trajectory(all_outage_config, tmp_path / "all-outage")

    stored = generate_rt_nr_trajectory(config, tmp_path, umcd_store_root=umcd_root)
    trajectory = read_nr_trajectory(tmp_path, stored)

    assert [record.provenance["full_path_count"] for record in trajectory.records] == [0, 1]
    assert [record.provenance["direct_path_count"] for record in trajectory.records] == [0, 1]
    assert [record.provenance["geometric_los_oracle"] for record in trajectory.records] == [
        False,
        True,
    ]
    outage = trajectory.records[0]
    covered = trajectory.records[1]
    assert outage.provenance["channel_truth_valid"] is False
    assert covered.provenance["channel_truth_valid"] is True
    assert not outage.target_masks["channel_cfr"].any()
    assert not outage.target_masks["channel_cfr_ofdm_symbols"].any()
    assert covered.target_masks["channel_cfr"].all()
    assert covered.target_masks["channel_cfr_ofdm_symbols"].all()
    np.testing.assert_array_equal(outage.targets["channel_cfr"], 0.0)
    np.testing.assert_array_equal(outage.targets["channel_cfr_ofdm_symbols"], 0.0)
    assert outage.provenance["noise_variance"] > 0.0
    assert np.any(outage.observation.received_grid != 0.0)
    assert outage.provenance["noise_policy"] == (
        "common variance from trajectory-batch mean received-grid power"
    )

    import json

    generation_path = (tmp_path / stored.manifest_relative_path).with_name(
        f"{stored.content_sha256}.generation.json"
    )
    generation = json.loads(generation_path.read_text(encoding="utf-8"))
    bundle = read_episode_bundle(umcd_root, StoredEpisodeBundle(**generation["umcd_bundle"]))
    assert [status.value for status in bundle.path_oracle.solve_status] == ["no_path", "solved"]
    channel_view = bundle.channel_views[0]
    assert channel_view.valid_mask is not None
    assert not channel_view.valid_mask[0].any()
    assert channel_view.valid_mask[1].all()
    np.testing.assert_array_equal(channel_view.channel[0], 0.0)


@pytest.mark.native_sionna
def test_moving_scene_object_doppler_matches_geometric_delay_derivative(tmp_path):
    config = RTTrajectoryConfig(
        trajectory_id="native-moving-screen-doppler-route",
        scene="box_one_screen",
        transmitter_position_m=(-2.0, -2.0, 2.5),
        receiver_start_m=(-2.0, 2.0, 2.5),
        receiver_velocity_mps=(0.0, 0.0, 0.0),
        num_steps=3,
        sample_interval_s=0.1,
        max_depth=2,
        samples_per_src=200,
        seed=20261021,
        device="cpu",
        moving_scene_objects=(RTSceneObjectMotion("screen", (1.0, 0.0, 0.0)),),
    )
    stored = generate_rt_nr_trajectory(config, tmp_path, umcd_store_root=tmp_path / "umcd")
    generation_path = (tmp_path / stored.manifest_relative_path).with_name(
        f"{stored.content_sha256}.generation.json"
    )
    import json

    generation = json.loads(generation_path.read_text(encoding="utf-8"))
    bundle = read_episode_bundle(
        tmp_path / "umcd", StoredEpisodeBundle(**generation["umcd_bundle"])
    )
    table = bundle.path_oracle
    assert table.doppler_hz is not None
    assert table.interaction_offsets is not None
    assert table.interaction_vertices_m is not None

    transmitter = np.asarray(config.transmitter_position_m)
    receiver = np.asarray(config.receiver_start_m)
    object_velocity = np.asarray(config.moving_scene_objects[0].velocity_mps)
    path_indices: list[int] = []
    for frame_index in range(config.num_steps):
        group_index = int(np.flatnonzero(table.frame_indices == frame_index)[0])
        group = table.group_slice(group_index)
        expected_object_position = np.asarray((0.0, 0.0, 2.5)) + object_velocity * (
            frame_index * config.sample_interval_s
        )
        candidates = []
        for path_index in range(group.start, group.stop):
            if table.interaction_order[path_index] != 1:
                continue
            interaction_index = int(table.interaction_offsets[path_index])
            point = table.interaction_vertices_m[interaction_index]
            if np.linalg.norm(point - expected_object_position) < 0.02:
                candidates.append(path_index)
        assert len(candidates) == 1, f"expected one screen reflection, found {candidates}"
        path_index = candidates[0]
        path_indices.append(path_index)

        interaction_index = int(table.interaction_offsets[path_index])
        point = table.interaction_vertices_m[interaction_index]
        incoming = point - transmitter
        incoming /= np.linalg.norm(incoming)
        outgoing = receiver - point
        outgoing /= np.linalg.norm(outgoing)
        expected_doppler_hz = float(
            np.dot(outgoing - incoming, object_velocity) * config.carrier_hz / speed_of_light
        )
        np.testing.assert_allclose(
            table.doppler_hz[path_index], expected_doppler_hz, rtol=2e-2, atol=0.1
        )

    delays = table.delay_s[path_indices]
    finite_difference_doppler_hz = -config.carrier_hz * float(
        (delays[2] - delays[0]) / (2.0 * config.sample_interval_s)
    )
    np.testing.assert_allclose(
        table.doppler_hz[path_indices[1]], finite_difference_doppler_hz, rtol=2e-2, atol=0.1
    )


@pytest.mark.native_sionna
def test_reflected_path_doppler_matches_finite_difference_of_geometric_delay(tmp_path):
    config = RTTrajectoryConfig(
        trajectory_id="native-reflected-doppler-route",
        scene="box_one_screen",
        transmitter_position_m=(-2.0, 0.0, 2.5),
        receiver_start_m=(2.0, 0.0, 2.5),
        receiver_velocity_mps=(0.2, 0.0, 0.0),
        num_steps=3,
        sample_interval_s=0.1,
        max_depth=2,
        samples_per_src=200,
        seed=20261021,
        device="cpu",
    )

    stored = generate_rt_nr_trajectory(config, tmp_path, umcd_store_root=tmp_path / "umcd")
    generation_path = (tmp_path / stored.manifest_relative_path).with_name(
        f"{stored.content_sha256}.generation.json"
    )
    import json

    generation = json.loads(generation_path.read_text(encoding="utf-8"))
    from datasets.episode_store import StoredEpisodeBundle, read_episode_bundle

    bundle = read_episode_bundle(
        tmp_path / "umcd", StoredEpisodeBundle(**generation["umcd_bundle"])
    )
    table = bundle.path_oracle
    assert table.doppler_hz is not None
    assert table.interaction_offsets is not None
    assert table.object_ids is not None
    assert table.primitive_ids is not None

    assert table.interaction_vertices_m is not None
    reflected_by_frame: list[dict[int, list[int]]] = []
    for frame_index in range(config.num_steps):
        group_index = int(np.flatnonzero(table.frame_indices == frame_index)[0])
        group = table.group_slice(group_index)
        frame_paths: dict[int, list[int]] = {}
        for path_index in range(group.start, group.stop):
            if table.interaction_order[path_index] != 1:
                continue
            first_interaction = int(table.interaction_offsets[path_index])
            key = int(table.object_ids[first_interaction])
            frame_paths.setdefault(key, []).append(path_index)
        reflected_by_frame.append(frame_paths)

    common_reflections = set.intersection(*(set(paths) for paths in reflected_by_frame))
    assert common_reflections, f"no persistent reflections across frames: {reflected_by_frame}"
    for reflection_key in common_reflections:
        from itertools import product

        candidates = product(*(paths[reflection_key] for paths in reflected_by_frame))

        def interaction_point(path_index: int) -> np.ndarray:
            interaction_index = int(table.interaction_offsets[path_index])
            return table.interaction_vertices_m[interaction_index]

        path_indices = list(
            min(
                candidates,
                key=lambda indices: sum(
                    float(np.linalg.norm(interaction_point(right) - interaction_point(left)))
                    for left, right in zip(indices, indices[1:])
                ),
            )
        )
        delays = table.delay_s[path_indices]
        center_doppler_hz = float(table.doppler_hz[path_indices[1]])
        finite_difference_doppler_hz = -config.carrier_hz * float(
            (delays[2] - delays[0]) / (2.0 * config.sample_interval_s)
        )
        np.testing.assert_allclose(
            center_doppler_hz,
            finite_difference_doppler_hz,
            rtol=2e-2,
            atol=0.05,
        )


@pytest.mark.native_sionna
def test_llvm_rt_common_random_numbers_remove_static_scene_sampling_jitter(tmp_path):
    config = RTTrajectoryConfig(
        trajectory_id="native-static-common-random-route",
        scene="simple_reflector",
        transmitter_position_m=(-10.0, 0.0, 10.0),
        receiver_start_m=(10.0, 0.0, 10.0),
        receiver_velocity_mps=(0.0, 0.0, 0.0),
        num_steps=2,
        sample_interval_s=0.1,
        max_depth=1,
        samples_per_src=200,
        seed=20261021,
        device="cpu",
    )

    stored = generate_rt_nr_trajectory(config, tmp_path)
    trajectory = read_nr_trajectory(tmp_path, stored)

    first = trajectory.records[0].targets["channel_cfr"]
    second = trajectory.records[1].targets["channel_cfr"]
    np.testing.assert_allclose(first, second, rtol=0.0, atol=1e-10)
    assert {record.provenance["rt_seed"] for record in trajectory.records} == {config.seed}
