"""Analytic checks for aligned multi-transmitter resource-grid composition."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from integrations.sionna.network_phy import (  # noqa: E402
    add_independent_receiver_awgn,
    compose_downlink_resource_grids,
    network_sinr,
)


def _two_node_fixture():
    # Two receivers, two transmitters, one port, two OFDM symbols and three bins.
    channel = torch.zeros((1, 2, 1, 2, 1, 2, 3), dtype=torch.complex64)
    channel[:, 0, :, 0] = 1.0
    channel[:, 0, :, 1] = 0.5
    channel[:, 1, :, 0] = 0.25
    channel[:, 1, :, 1] = 1.0
    grids = torch.ones((1, 2, 1, 2, 3), dtype=torch.complex64)
    return channel, grids


def test_joint_grid_equals_sum_of_individually_filtered_source_contributions():
    channel, grids = _two_node_fixture()
    joint = compose_downlink_resource_grids(channel, grids, (1.0, 4.0))

    isolated = []
    for tx_index, power in enumerate((1.0, 4.0)):
        one_tx_channel = channel[:, :, :, tx_index : tx_index + 1]
        one_tx_grid = grids[:, tx_index : tx_index + 1]
        result = compose_downlink_resource_grids(one_tx_channel, one_tx_grid, (power,))
        isolated.append(result.received_grid)

    torch.testing.assert_close(joint.received_grid, isolated[0] + isolated[1])
    torch.testing.assert_close(joint.source_contributions.sum(dim=3), joint.received_grid)
    assert joint.source_contributions.shape == (1, 2, 1, 2, 2, 3)


def test_disabling_interferer_reproduces_desired_only_reference():
    channel, grids = _two_node_fixture()
    joint = compose_downlink_resource_grids(channel, grids, (1.0, 0.0))
    desired_only = compose_downlink_resource_grids(channel[:, :, :, :1], grids[:, :1], (1.0,))

    torch.testing.assert_close(joint.received_grid, desired_only.received_grid)
    assert torch.count_nonzero(joint.source_contributions[:, :, :, 1]) == 0
    metrics = network_sinr(joint, (0, 0), (0.0, 0.0))
    torch.testing.assert_close(
        metrics.interference_power_w_per_re, torch.zeros(2, dtype=torch.float64)
    )


def test_interferer_power_scales_interference_and_sinr_monotonically():
    channel, grids = _two_node_fixture()
    low = compose_downlink_resource_grids(channel[:, :1], grids, (1.0, 1.0))
    high = compose_downlink_resource_grids(channel[:, :1], grids, (1.0, 4.0))
    low_metrics = network_sinr(low, (0,), (0.1,))
    high_metrics = network_sinr(high, (0,), (0.1,))

    torch.testing.assert_close(
        high_metrics.interference_power_w_per_re,
        4.0 * low_metrics.interference_power_w_per_re,
    )
    assert high_metrics.sinr_linear[0] < low_metrics.sinr_linear[0]


def test_receiver_noise_is_reproducible_but_independent_between_receivers():
    channel, grids = _two_node_fixture()
    composition = compose_downlink_resource_grids(channel, grids, (1.0, 1.0))
    first = add_independent_receiver_awgn(composition, (0.1, 0.1), seed=81)
    replay = add_independent_receiver_awgn(composition, (0.1, 0.1), seed=81)

    torch.testing.assert_close(first.noise_grid, replay.noise_grid)
    assert not torch.equal(first.noise_grid[:, 0], first.noise_grid[:, 1])
    torch.testing.assert_close(first.received_grid, composition.received_grid + first.noise_grid)
    assert first.seed == 81


def test_sinr_marks_zero_over_zero_as_undefined():
    channel = torch.zeros((1, 1, 1, 1, 1, 2, 3), dtype=torch.complex64)
    grids = torch.ones((1, 1, 1, 2, 3), dtype=torch.complex64)
    silent = compose_downlink_resource_grids(channel, grids, (1.0,))
    metrics = network_sinr(silent, (0,), (0.0,))

    assert not bool(metrics.valid[0])
    assert torch.isnan(metrics.sinr_linear[0])


@pytest.mark.parametrize(
    ("channel_shape", "grid_shape", "powers", "message"),
    [
        ((1, 1, 1, 2, 1, 2, 3), (1, 2, 1, 2, 3), (1.0,), "one power value"),
        ((1, 1, 1, 2, 1, 2, 3), (1, 2, 1, 2, 3), (1.0, -1.0), "nonnegative"),
    ],
)
def test_composition_rejects_invalid_power_configuration(
    channel_shape, grid_shape, powers, message
):
    channel = torch.ones(channel_shape, dtype=torch.complex64)
    grids = torch.ones(grid_shape, dtype=torch.complex64)
    with pytest.raises(ValueError, match=message):
        compose_downlink_resource_grids(channel, grids, powers)


def test_composition_rejects_nonzero_power_for_empty_transmit_grid():
    channel = torch.ones((1, 1, 1, 1, 1, 2, 3), dtype=torch.complex64)
    grids = torch.zeros((1, 1, 1, 2, 3), dtype=torch.complex64)
    with pytest.raises(ValueError, match="nonzero grid energy"):
        compose_downlink_resource_grids(channel, grids, (1.0,))
