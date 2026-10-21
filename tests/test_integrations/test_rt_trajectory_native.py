"""Native LLVM RT and Sionna PHY integration for stored NR trajectories."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mitsuba")
pytest.importorskip("sionna")

from datasets.nr_store import read_nr_trajectory  # noqa: E402
from integrations.sionna.rt_trajectory import (  # noqa: E402
    RTTrajectoryConfig,
    generate_rt_nr_trajectory,
)


@pytest.mark.native_sionna
def test_cpu_rt_path_retraces_and_stores_receiver_only_trajectory(tmp_path):
    config = RTTrajectoryConfig(
        trajectory_id="native-simple-reflector-route",
        scene="simple_reflector",
        transmitter_position_m=(-10.0, 0.0, 10.0),
        receiver_start_m=(10.0, 0.0, 10.0),
        receiver_velocity_mps=(0.0, 0.2, 0.0),
        num_steps=2,
        sample_interval_s=0.1,
        max_depth=1,
        samples_per_src=200,
        seed=20261021,
        device="cpu",
    )

    stored = generate_rt_nr_trajectory(config, tmp_path)
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
        assert bool(np.asarray(record.target_masks["channel_cfr"]).all())
        assert not hasattr(model_inputs, "targets")
        assert not hasattr(model_inputs, "provenance")
        assert np.isfinite(model_inputs.received_grid).all()

    sidecar = tmp_path / "manifests" / "trajectories" / f"{stored.content_sha256}.generation.json"
    assert sidecar.is_file()


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
