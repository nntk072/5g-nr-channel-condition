"""Analytic checks for aligned multi-transmitter resource-grid composition."""

from __future__ import annotations

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from integrations.sionna.network_phy import (  # noqa: E402
    add_independent_receiver_awgn,
    apply_network_link_budget,
    compose_downlink_resource_grids,
    network_resource_sinr,
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


def test_per_link_budget_applies_antenna_beam_and_path_loss_once():
    small_scale = torch.ones((1, 2, 1, 2, 1, 2, 3), dtype=torch.complex64)
    budget = apply_network_link_budget(
        small_scale,
        path_loss_db=((80.0, 100.0), (90.0, 70.0)),
        tx_antenna_gain_dbi=(10.0, 15.0),
        rx_antenna_gain_dbi=(2.0, 3.0),
        tx_beam_gain_dbi=((4.0, 0.0), (1.0, 2.0)),
        rx_beam_gain_dbi=((0.0, 1.0), (2.0, 0.0)),
        transmit_power_w_per_re=(1.0, 2.0),
        input_channel_semantics="unit_mean_small_scale_fading",
    )

    expected_net_gain_db = torch.tensor(
        [[-64.0, -82.0], [-74.0, -50.0]], dtype=torch.float64
    )
    expected_power_gain = torch.pow(10.0, expected_net_gain_db / 10.0)
    torch.testing.assert_close(budget.net_gain_db, expected_net_gain_db)
    torch.testing.assert_close(budget.power_gain_linear, expected_power_gain)
    torch.testing.assert_close(
        budget.mean_received_power_w_per_re,
        expected_power_gain
        * torch.tensor([[1.0, 2.0], [1.0, 2.0]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        budget.channel.abs().square().to(torch.float64),
        expected_power_gain[None, :, None, :, None, None, None].expand_as(
            budget.channel.real
        ).to(torch.float64),
    )


def test_link_budget_scaled_channels_feed_expected_per_receiver_sinr():
    small_scale = torch.ones((1, 1, 1, 2, 1, 2, 3), dtype=torch.complex64)
    budget = apply_network_link_budget(
        small_scale,
        path_loss_db=((80.0, 90.0),),
        tx_antenna_gain_dbi=(0.0, 0.0),
        rx_antenna_gain_dbi=(0.0,),
        tx_beam_gain_dbi=((0.0, 0.0),),
        rx_beam_gain_dbi=((0.0, 0.0),),
        transmit_power_w_per_re=(1.0, 2.0),
        input_channel_semantics="unit_mean_small_scale_fading",
    )
    grids = torch.ones((1, 2, 1, 2, 3), dtype=torch.complex64)
    composition = compose_downlink_resource_grids(
        budget.channel, grids, budget.transmit_power_w_per_re
    )
    metrics = network_sinr(composition, (0,), (1e-12,))
    expected_desired = budget.mean_received_power_w_per_re[0, 0]
    expected_interference = budget.mean_received_power_w_per_re[0, 1]
    torch.testing.assert_close(metrics.desired_power_w_per_re[0], expected_desired)
    torch.testing.assert_close(metrics.interference_power_w_per_re[0], expected_interference)
    torch.testing.assert_close(
        metrics.sinr_linear[0], expected_desired / (expected_interference + 1e-12)
    )


def test_per_resource_sinr_preserves_symbol_and_subcarrier_interference_structure():
    channel = torch.ones((1, 1, 1, 2, 1, 2, 3), dtype=torch.complex64)
    grids = torch.ones((1, 2, 1, 2, 3), dtype=torch.complex64)
    grids[:, 1, :, :, 1] = 0.0
    composition = compose_downlink_resource_grids(channel, grids, (1.0, 1.0))

    metrics = network_resource_sinr(composition, (0,), (0.25,))

    assert metrics.desired_power_w_per_re.shape == (1, 2, 3)
    assert metrics.interference_power_w_per_re.shape == (1, 2, 3)
    torch.testing.assert_close(
        metrics.desired_power_w_per_re,
        torch.ones((1, 2, 3), dtype=torch.float64),
    )
    torch.testing.assert_close(
        metrics.interference_power_w_per_re,
        torch.tensor([[[1.5, 0.0, 1.5]]], dtype=torch.float64).expand(
            1, 2, 3
        ),
    )
    torch.testing.assert_close(
        metrics.sinr_linear,
        metrics.desired_power_w_per_re
        / (metrics.interference_power_w_per_re + metrics.noise_power_w_per_re),
    )


def test_per_resource_sinr_marks_empty_zero_over_zero_resource_as_undefined():
    channel = torch.zeros((1, 1, 1, 1, 1, 2, 3), dtype=torch.complex64)
    grids = torch.ones((1, 1, 1, 2, 3), dtype=torch.complex64)
    composition = compose_downlink_resource_grids(channel, grids, (1.0,))

    metrics = network_resource_sinr(composition, (0,), (0.0,))

    assert not bool(metrics.valid.any())
    assert bool(torch.isnan(metrics.sinr_linear).all())


def test_per_resource_sinr_accepts_serving_cell_map_by_symbol_and_subcarrier():
    channel = torch.ones((1, 1, 1, 2, 1, 2, 3), dtype=torch.complex64)
    channel[:, :, :, 1] = 0.5
    grids = torch.ones((1, 2, 1, 2, 3), dtype=torch.complex64)
    composition = compose_downlink_resource_grids(channel, grids, (1.0, 1.0))
    serving_map = torch.tensor([[[0, 1, 0], [1, 0, 1]]])

    metrics = network_resource_sinr(composition, serving_map, (0.25,))

    torch.testing.assert_close(
        metrics.desired_power_w_per_re,
        torch.tensor([[[1.0, 0.25, 1.0], [0.25, 1.0, 0.25]]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        metrics.interference_power_w_per_re,
        torch.tensor([[[0.25, 1.0, 0.25], [1.0, 0.25, 1.0]]], dtype=torch.float64),
    )


@pytest.mark.parametrize(
    ("serving_map", "message"),
    [
        ((True,), "integer transmitter indices"),
        ((0.0,), "integer transmitter indices"),
        (((0, 1),), r"shape \[rx\] or \[rx, symbol, subcarrier\]"),
        ((2,), "address an existing transmitter"),
    ],
)
def test_per_resource_sinr_rejects_invalid_serving_cell_maps(serving_map, message):
    channel, grids = _two_node_fixture()
    composition = compose_downlink_resource_grids(channel, grids, (1.0, 1.0))

    with pytest.raises(ValueError, match=message):
        network_resource_sinr(composition, serving_map, (0.1, 0.1))


def test_frequency_dependent_link_budget_binds_and_scales_each_subcarrier():
    small_scale = torch.ones((1, 1, 1, 1, 1, 1, 3), dtype=torch.complex64)
    budget = apply_network_link_budget(
        small_scale,
        path_loss_db=(((80.0, 90.0, 100.0),),),
        tx_antenna_gain_dbi=(0.0,),
        rx_antenna_gain_dbi=(0.0,),
        tx_beam_gain_dbi=(((0.0, 3.0, 6.0),),),
        rx_beam_gain_dbi=((0.0,),),
        transmit_power_w_per_re=(2.0,),
        input_channel_semantics="unit_mean_small_scale_fading",
        subcarrier_frequencies_hz=(3.4e9, 3.5e9, 3.6e9),
    )

    expected_net_gain_db = torch.tensor([[[-80.0, -87.0, -94.0]]], dtype=torch.float64)
    expected_power_gain = torch.pow(10.0, expected_net_gain_db / 10.0)
    torch.testing.assert_close(budget.net_gain_db, expected_net_gain_db)
    torch.testing.assert_close(budget.power_gain_linear, expected_power_gain)
    torch.testing.assert_close(
        budget.mean_received_power_w_per_re, expected_power_gain * 2.0
    )
    torch.testing.assert_close(
        budget.channel.abs().square().to(torch.float64),
        expected_power_gain[None, :, None, :, None, None, :].to(torch.float64),
    )
    torch.testing.assert_close(
        budget.subcarrier_frequencies_hz,
        torch.tensor((3.4e9, 3.5e9, 3.6e9), dtype=torch.float64),
    )


def test_frequency_dependent_link_budget_requires_exact_ordered_frequency_grid():
    channel = torch.ones((1, 1, 1, 1, 1, 1, 2), dtype=torch.complex64)
    common = {
        "path_loss_db": (((80.0, 90.0),),),
        "tx_antenna_gain_dbi": (0.0,),
        "rx_antenna_gain_dbi": (0.0,),
        "tx_beam_gain_dbi": ((0.0,),),
        "rx_beam_gain_dbi": ((0.0,),),
        "transmit_power_w_per_re": (1.0,),
        "input_channel_semantics": "unit_mean_small_scale_fading",
    }
    with pytest.raises(ValueError, match="required for frequency-dependent"):
        apply_network_link_budget(channel, **common)
    with pytest.raises(ValueError, match="strictly increasing"):
        apply_network_link_budget(
            channel,
            **common,
            subcarrier_frequencies_hz=(3.6e9, 3.5e9),
        )


def test_link_budget_rejects_double_counting_and_invalid_inputs():
    channel = torch.ones((1, 1, 1, 1, 1, 1, 1), dtype=torch.complex64)
    common = {
        "path_loss_db": ((80.0,),),
        "tx_antenna_gain_dbi": (0.0,),
        "rx_antenna_gain_dbi": (0.0,),
        "tx_beam_gain_dbi": ((0.0,),),
        "rx_beam_gain_dbi": ((0.0,),),
        "transmit_power_w_per_re": (1.0,),
        "input_channel_semantics": "unit_mean_small_scale_fading",
    }
    with pytest.raises(ValueError, match="must not be applied twice"):
        apply_network_link_budget(
            channel, **{**common, "input_channel_semantics": "includes_path_loss"}
        )
    with pytest.raises(ValueError, match="path_loss_db must be nonnegative"):
        apply_network_link_budget(channel, **{**common, "path_loss_db": ((-1.0,),)})
    with pytest.raises(ValueError, match="Boolean"):
        apply_network_link_budget(channel, **{**common, "tx_antenna_gain_dbi": (True,)})
    with pytest.raises(ValueError, match="shape \(1, 1\)"):
        apply_network_link_budget(channel, **{**common, "rx_beam_gain_dbi": (0.0,)})


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


@pytest.mark.parametrize(
    ("power", "message"),
    [
        ((True,), "Boolean"),
        ((1.0 + 0.5j,), "complex"),
        (("1.0",), "real numeric"),
        ((1e-300,), "not representable"),
        ((1e300,), "not representable"),
    ],
)
def test_composition_rejects_malformed_or_unrepresentable_transmit_power(power, message):
    channel, grids = _two_node_fixture()
    with pytest.raises(ValueError, match=message):
        compose_downlink_resource_grids(channel, grids, power)


def test_composition_rejects_sparse_transmit_power():
    channel, grids = _two_node_fixture()
    sparse_power = torch.sparse_coo_tensor(torch.tensor([[0]]), torch.tensor([1.0]), (1,))
    with pytest.raises(ValueError, match="dense real-valued vector"):
        compose_downlink_resource_grids(channel[:, :, :, :1], grids[:, :1], sparse_power)


@pytest.mark.parametrize(
    ("noise", "message"),
    [
        ((True, False), "Boolean"),
        ((0.1 + 1j, 0.1), "complex"),
        (("0.1", "0.1"), "real numeric"),
    ],
)
def test_sinr_rejects_malformed_receiver_noise(noise, message):
    channel, grids = _two_node_fixture()
    composition = compose_downlink_resource_grids(channel, grids, (1.0, 1.0))
    with pytest.raises(ValueError, match=message):
        network_sinr(composition, (0, 1), noise)


def test_sinr_rejects_sparse_receiver_noise():
    channel, grids = _two_node_fixture()
    composition = compose_downlink_resource_grids(channel, grids, (1.0, 1.0))
    sparse_noise = torch.sparse_coo_tensor(torch.tensor([[0]]), torch.tensor([0.1]), (2,))
    with pytest.raises(ValueError, match="dense real-valued vector"):
        network_sinr(composition, (0, 1), sparse_noise)


@pytest.mark.parametrize("variance", (1e-100, 1e100))
def test_receiver_noise_rejects_values_not_representable_in_grid_dtype(variance):
    channel, grids = _two_node_fixture()
    composition = compose_downlink_resource_grids(channel, grids, (1.0, 1.0))
    with pytest.raises(ValueError, match="representable"):
        add_independent_receiver_awgn(composition, (variance, variance), seed=41)


def test_network_receiver_and_sinr_refuse_inconsistent_composition():
    channel, grids = _two_node_fixture()
    composition = compose_downlink_resource_grids(channel, grids, (1.0, 1.0))
    inconsistent = replace(
        composition,
        received_grid=composition.received_grid + torch.ones_like(composition.received_grid),
    )
    with pytest.raises(ValueError, match="equal the sum of source contributions"):
        network_sinr(inconsistent, (0, 1), (0.1, 0.1))
    with pytest.raises(ValueError, match="equal the sum of source contributions"):
        add_independent_receiver_awgn(inconsistent, (0.1, 0.1), seed=41)
