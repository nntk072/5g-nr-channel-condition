"""Framework-independent RT trajectory configuration checks."""

import pytest
from integrations.sionna.rt_trajectory import RTTrajectoryConfig


def test_rt_trajectory_config_generates_ordered_constant_velocity_route() -> None:
    config = RTTrajectoryConfig(
        trajectory_id="route-a",
        scene="simple_reflector",
        transmitter_position_m=(1.0, 2.0, 3.0),
        receiver_start_m=(4.0, 5.0, 6.0),
        receiver_velocity_mps=(2.0, -1.0, 0.5),
        num_steps=3,
        sample_interval_s=0.25,
    )

    positions, times = config.positions_and_times()

    assert positions == ((4.0, 5.0, 6.0), (4.5, 4.75, 6.125), (5.0, 4.5, 6.25))
    assert times == (0.0, 0.25, 0.5)
    assert config.no_path_policy == "reject"


@pytest.mark.parametrize(
    "overrides",
    [
        {"trajectory_id": ""},
        {"scene": ""},
        {"num_steps": 0},
        {"sample_interval_s": 0.0},
        {"carrier_hz": float("nan")},
        {"snr_db": float("inf")},
        {"snr_db": None},
        {"tx_power_per_re_w": 1e-6},
        {"no_path_policy": "noise_only_absolute_power"},
        {"max_depth": -1},
        {"max_depth": True},
        {"samples_per_src": 0},
        {"no_path_policy": "zero_is_ground_truth"},
        {"seed": True},
        {"receiver_velocity_mps": (float("nan"), 0.0, 0.0)},
    ],
)
def test_rt_trajectory_config_rejects_invalid_values(overrides: dict[str, object]) -> None:
    values: dict[str, object] = {
        "trajectory_id": "route-a",
        "scene": "simple_reflector",
        "transmitter_position_m": (1.0, 2.0, 3.0),
        "receiver_start_m": (4.0, 5.0, 6.0),
        "receiver_velocity_mps": (1.0, 0.0, 0.0),
    }
    values.update(overrides)

    with pytest.raises((TypeError, ValueError)):
        RTTrajectoryConfig(**values)  # type: ignore[arg-type]


def test_rt_trajectory_absolute_power_mode_requires_explicit_transmit_power() -> None:
    config = RTTrajectoryConfig(
        trajectory_id="physical-noise-route",
        scene="simple_reflector",
        transmitter_position_m=(1.0, 2.0, 3.0),
        receiver_start_m=(4.0, 5.0, 6.0),
        receiver_velocity_mps=(0.0, 0.0, 0.0),
        snr_db=None,
        tx_power_per_re_w=1e-6,
        no_path_policy="noise_only_absolute_power",
    )
    assert config.tx_power_per_re_w == 1e-6
    assert config.snr_db is None
