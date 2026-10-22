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
        assert symbol_channels.shape == (
            record.observation.received_grid.shape[-2],
            record.observation.received_grid.shape[-1],
        )
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
        sample_interval_s=0.1,
        max_depth=2,
        samples_per_src=200,
        seed=20261021,
        device="cpu",
        moving_scene_objects=(RTSceneObjectMotion("screen", (0.0, 100.0, 0.0)),),
    )

    stored = generate_rt_nr_trajectory(config, tmp_path)
    trajectory = read_nr_trajectory(tmp_path, stored)

    poses = [record.provenance["moving_scene_objects"]["screen"] for record in trajectory.records]
    assert poses[0]["position_m"] == [0.0, 0.0, 2.5]
    assert poses[1]["position_m"] == [0.0, 10.0, 2.5]
    assert all(pose["velocity_mps"] == [0.0, 100.0, 0.0] for pose in poses)
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
