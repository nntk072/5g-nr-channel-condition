"""Analytic checks for aligned multi-transmitter resource-grid composition."""

from __future__ import annotations

import math
from dataclasses import replace

import integrations.sionna.network_phy.records as network_records
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from integrations.sionna.network_phy import (  # noqa: E402
    NetworkTransmitterWaveform,
    add_independent_receiver_awgn,
    add_independent_receiver_waveform_awgn,
    apply_network_link_budget,
    apply_network_transmitter_iq_imbalance,
    compose_downlink_resource_grids,
    compose_received_waveforms,
    network_resource_sinr,
    network_sinr,
    propagate_network_transmit_waveforms,
)
from propagation.physics import frequency_response, subcarrier_frequencies  # noqa: E402


def _two_node_fixture():
    # Two receivers, two transmitters, one port, two OFDM symbols and three bins.
    channel = torch.zeros((1, 2, 1, 2, 1, 2, 3), dtype=torch.complex64)
    channel[:, 0, :, 0] = 1.0
    channel[:, 0, :, 1] = 0.5
    channel[:, 1, :, 0] = 0.25
    channel[:, 1, :, 1] = 1.0
    grids = torch.ones((1, 2, 1, 2, 3), dtype=torch.complex64)
    return channel, grids


def test_transmitter_waveforms_propagate_through_causal_mimo_fir_and_keep_tail():
    waveform = torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]]]], dtype=torch.complex64)
    taps = torch.zeros((1, 1, 2, 1, 2), dtype=torch.complex64)
    taps[0, 0, 0, 0] = torch.tensor([1.0, 0.5j])
    taps[0, 0, 1, 0] = torch.tensor([2.0, -1.0])
    result = propagate_network_transmit_waveforms(
        NetworkTransmitterWaveform(waveform, torch.zeros(2), torch.zeros(2)), taps
    )
    assert result.output_sample_count == 3
    torch.testing.assert_close(
        result.link_waveforms[0, 0, :, 0],
        torch.tensor([[1.0, 2.0 + 0.5j, 1.0j], [6.0, 5.0, -4.0]], dtype=torch.complex64),
    )


@pytest.mark.parametrize(
    ("waveform", "taps", "limit", "error", "message"),
    [
        (None, None, 10, TypeError, "dense complex"),
        (
            torch.ones((1, 1, 1, 2)),
            torch.ones((1, 1, 1, 1, 1), dtype=torch.complex64),
            10,
            TypeError,
            "dense complex",
        ),
        (
            torch.ones((1, 1, 1, 2), dtype=torch.complex64),
            object(),
            10,
            ValueError,
            "complex MIMO FIR",
        ),
        (
            torch.ones((1, 1, 1, 2), dtype=torch.complex64),
            torch.ones((1, 1, 1, 1), dtype=torch.complex64),
            10,
            ValueError,
            "1–128 taps",
        ),
        (
            torch.ones((1, 1, 1, 2), dtype=torch.complex64),
            torch.ones((1, 1, 2, 1, 1), dtype=torch.complex64),
            10,
            ValueError,
            "1–128 taps",
        ),
        (
            torch.ones((1, 1, 1, 2), dtype=torch.complex64),
            torch.ones((1, 1, 1, 1, 1), dtype=torch.complex128),
            10,
            ValueError,
            "share complex dtype",
        ),
        (
            torch.full((1, 1, 1, 2), complex(float("inf"), 0), dtype=torch.complex64),
            torch.ones((1, 1, 1, 1, 1), dtype=torch.complex64),
            100,
            ValueError,
            "must be finite",
        ),
        (
            torch.ones((1, 1, 1, 2), dtype=torch.complex64),
            torch.ones((1, 1, 1, 1, 1), dtype=torch.complex64),
            0,
            ValueError,
            "positive integer",
        ),
        (
            torch.ones((1, 1, 1, 2), dtype=torch.complex64),
            torch.ones((1, 1, 1, 1, 1), dtype=torch.complex64),
            1,
            ValueError,
            "exceed max_output_elements",
        ),
    ],
)
def test_transmitter_waveform_propagation_rejects_invalid_contracts(
    waveform, taps, limit, error, message
):
    record = NetworkTransmitterWaveform(waveform, None, None)
    with pytest.raises(error, match=message):
        propagate_network_transmit_waveforms(record, taps, max_output_elements=limit)


def test_transmitter_waveform_propagation_rejects_nonfinite_contribution():
    waveform = torch.full((1, 1, 1, 2), complex(3e38, 0), dtype=torch.complex64)
    taps = torch.full((1, 1, 1, 1, 1), 2.0 + 0.0j, dtype=torch.complex64)
    with pytest.raises(ValueError, match="produced non-finite samples"):
        propagate_network_transmit_waveforms(
            NetworkTransmitterWaveform(waveform, torch.zeros(1), torch.zeros(1)), taps
        )


def test_transmitter_waveform_propagation_rejects_sparse_and_oversized_firs():
    waveform = torch.ones((1, 1, 1, 2), dtype=torch.complex64)
    sparse_taps = torch.sparse_coo_tensor(
        torch.tensor([[0], [0], [0], [0], [0]]),
        torch.tensor([1.0 + 0j], dtype=torch.complex64),
        (1, 1, 1, 1, 1),
    )
    record = NetworkTransmitterWaveform(waveform, torch.zeros(1), torch.zeros(1))
    for taps in (
        sparse_taps,
        torch.ones((1, 1, 1, 1, 129), dtype=torch.complex64),
        torch.empty((0, 1, 1, 1, 1), dtype=torch.complex64),
    ):
        with pytest.raises(ValueError, match="1–128 taps"):
            propagate_network_transmit_waveforms(record, taps)
    with pytest.raises(ValueError, match="positive integer"):
        propagate_network_transmit_waveforms(
            record,
            torch.ones((1, 1, 1, 1, 1), dtype=torch.complex64),
            max_output_elements=True,
        )


def test_waveform_record_helpers_guard_interpolation_and_phase_noise(monkeypatch):
    output_indices = torch.arange(3, dtype=torch.float64)
    source = torch.ones((1, 4), dtype=torch.complex64)
    monkeypatch.setattr(torch, "sinc", lambda value: torch.zeros_like(value))
    with pytest.raises(ValueError, match="not normalizable"):
        network_records._fractional_delay_capture_slice(
            torch,
            source,
            output_indices=output_indices,
            delay_samples=0.25,
            kernel_radius=2,
        )

    phase = network_records._shared_phase_noise_processes(
        torch,
        batch_size=1,
        node_count=2,
        sample_count=1,
        linewidth_hz=torch.ones(2, dtype=torch.float64),
        sample_rate_hz=1.0,
        generator=None,
        device=torch.device("cpu"),
    )
    assert phase.shape == (1, 2, 1)
    with pytest.raises(ValueError, match="ratio must be finite"):
        network_records._shared_phase_noise_processes(
            torch,
            batch_size=1,
            node_count=1,
            sample_count=2,
            linewidth_hz=torch.tensor([float("inf")], dtype=torch.float64),
            sample_rate_hz=1.0,
            generator=None,
            device=torch.device("cpu"),
        )

    monkeypatch.undo()
    monkeypatch.setattr(
        torch,
        "randn",
        lambda shape, **kwargs: torch.full(shape, float("inf"), dtype=kwargs["dtype"]),
    )
    with pytest.raises(ValueError, match="trajectory is not finite"):
        network_records._shared_phase_noise_processes(
            torch,
            batch_size=1,
            node_count=1,
            sample_count=2,
            linewidth_hz=torch.ones(1, dtype=torch.float64),
            sample_rate_hz=1.0,
            generator=None,
            device=torch.device("cpu"),
        )


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

    expected_net_gain_db = torch.tensor([[-64.0, -82.0], [-74.0, -50.0]], dtype=torch.float64)
    expected_power_gain = torch.pow(10.0, expected_net_gain_db / 10.0)
    torch.testing.assert_close(budget.net_gain_db, expected_net_gain_db)
    torch.testing.assert_close(budget.power_gain_linear, expected_power_gain)
    torch.testing.assert_close(
        budget.mean_received_power_w_per_re,
        expected_power_gain * torch.tensor([[1.0, 2.0], [1.0, 2.0]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        budget.channel.abs().square().to(torch.float64),
        expected_power_gain[None, :, None, :, None, None, None]
        .expand_as(budget.channel.real)
        .to(torch.float64),
    )


def test_link_budget_preserves_analytical_multipath_phase_and_applies_power_gain():
    frequencies_hz = subcarrier_frequencies(16, 30e3)
    path_gains = np.asarray([0.8 + 0.2j, -0.15 + 0.35j, 0.1 - 0.2j])
    delays_s = np.asarray([0.0, 0.42e-6, 1.17e-6])
    channel = frequency_response(path_gains, delays_s, frequencies_hz)
    channel /= np.sqrt(np.mean(np.abs(channel) ** 2))
    small_scale = torch.as_tensor(channel, dtype=torch.complex64).reshape(1, 1, 1, 1, 1, 1, -1)

    budget = apply_network_link_budget(
        small_scale,
        path_loss_db=((0.0,),),
        tx_antenna_gain_dbi=(2.0,),
        rx_antenna_gain_dbi=(1.0,),
        tx_beam_gain_dbi=((2.0,),),
        rx_beam_gain_dbi=((1.0,),),
        transmit_power_w_per_re=(0.5,),
        input_channel_semantics="unit_mean_small_scale_fading",
    )

    amplitude_gain = 10.0 ** (6.0 / 20.0)
    power_gain = 10.0 ** (6.0 / 10.0)
    torch.testing.assert_close(
        budget.channel[0, 0, 0, 0, 0, 0],
        small_scale[0, 0, 0, 0, 0, 0] * amplitude_gain,
    )
    torch.testing.assert_close(
        budget.channel[0, 0, 0, 0, 0, 0].abs().square().to(torch.float64),
        small_scale[0, 0, 0, 0, 0, 0].abs().square().to(torch.float64) * power_gain,
        rtol=3e-7,
        atol=1e-7,
    )
    torch.testing.assert_close(
        budget.mean_received_power_w_per_re,
        torch.tensor([[0.5 * power_gain]], dtype=torch.float64),
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
        torch.tensor([[[1.5, 0.0, 1.5]]], dtype=torch.float64).expand(1, 2, 3),
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
        ((2, 0), "address an existing transmitter"),
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
    torch.testing.assert_close(budget.mean_received_power_w_per_re, expected_power_gain * 2.0)
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
    with pytest.raises(ValueError, match=r"shape \(1, 1\)"):
        apply_network_link_budget(channel, **{**common, "rx_beam_gain_dbi": (0.0,)})


def test_disabling_interferer_reproduces_desired_only_reference():
    channel, grids = _two_node_fixture()
    joint = compose_downlink_resource_grids(channel, grids, (1.0, 0.0))
    desired_only = compose_downlink_resource_grids(channel[:, :, :, :1], grids[:, :1], (1.0,))

    torch.testing.assert_close(joint.received_grid, desired_only.received_grid)
    assert torch.count_nonzero(joint.source_contributions[:, :, :, 1]) == 0
    metrics = network_sinr(joint, (0, 0), (1e-12, 1e-12))
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
        ((1e-300, 1.0), "not representable"),
        ((1e300, 1.0), "not representable"),
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
    with pytest.raises(ValueError, match="representable|underflow"):
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


def test_received_waveform_composition_aligns_each_link_on_capture_timeline():
    links = torch.zeros((1, 2, 2, 1, 4), dtype=torch.complex64)
    links[:, 0, 0, 0] = 1.0
    links[:, 0, 1, 0] = 1.0j
    links[:, 1, 0, 0] = 2.0
    links[:, 1, 1, 0] = 3.0j
    composition = compose_received_waveforms(
        links,
        link_start_samples=((0, 2), (-1, 1)),
        capture_num_samples=6,
        sample_rate_hz=30.72e6,
    )

    assert composition.source_contributions.shape == (1, 2, 1, 2, 6)
    torch.testing.assert_close(
        composition.received_waveform[0, 0, 0],
        torch.tensor([1, 1, 1 + 1j, 1 + 1j, 1j, 1j], dtype=torch.complex64),
    )
    torch.testing.assert_close(
        composition.received_waveform[0, 1, 0],
        torch.tensor([2, 2 + 3j, 2 + 3j, 3j, 3j, 0], dtype=torch.complex64),
    )
    assert composition.sample_rate_hz == 30.72e6


def test_received_waveform_composition_applies_causal_complex_link_filter_and_keeps_tail():
    links = torch.tensor([1.0, 2.0, 3.0], dtype=torch.complex64).reshape(1, 1, 1, 1, 3)
    taps = torch.tensor([[[1.0 + 0.0j, 0.0 + 0.5j]]], dtype=torch.complex64)
    composition = compose_received_waveforms(
        links,
        link_start_samples=((1,),),
        capture_num_samples=6,
        sample_rate_hz=30.72e6,
        link_filter_taps=taps,
    )

    torch.testing.assert_close(
        composition.received_waveform[0, 0, 0],
        torch.tensor([0, 1, 2 + 0.5j, 3 + 1j, 1.5j, 0], dtype=torch.complex64),
    )
    torch.testing.assert_close(composition.link_filter_taps, taps)


def test_received_waveform_composition_bounds_and_validates_complex_link_filters():
    links = torch.ones((1, 1, 1, 1, 10), dtype=torch.complex64)
    args = {
        "link_start_samples": ((0,),),
        "capture_num_samples": 10,
        "sample_rate_hz": 30.72e6,
    }
    with pytest.raises(ValueError, match=r"complex \[rx, tx, tap\]"):
        compose_received_waveforms(links, **args, link_filter_taps=(((1.0, 0.0),),))
    with pytest.raises(ValueError, match="finite"):
        compose_received_waveforms(
            links,
            **args,
            link_filter_taps=torch.tensor([[[complex(float("nan"), 0.0)]]]),
        )
    with pytest.raises(ValueError, match="exceed max_output_elements"):
        compose_received_waveforms(
            links,
            **args,
            link_filter_taps=torch.ones((1, 1, 2), dtype=torch.complex64),
            max_output_elements=39,
        )


def test_waveform_receiver_noise_is_reproducible_and_distinct_per_receiver():
    links = torch.ones((2, 2, 1, 2, 4096), dtype=torch.complex64)
    composition = compose_received_waveforms(
        links,
        link_start_samples=((0,), (0,)),
        capture_num_samples=4096,
        sample_rate_hz=15.36e6,
    )
    first = add_independent_receiver_waveform_awgn(composition, (0.2, 0.2), seed=513)
    replay = add_independent_receiver_waveform_awgn(composition, (0.2, 0.2), seed=513)

    torch.testing.assert_close(first.noise_waveform, replay.noise_waveform, rtol=0, atol=0)
    assert not torch.equal(first.noise_waveform[:, 0], first.noise_waveform[:, 1])
    measured = first.noise_waveform.abs().square().mean(dim=(0, 2, 3))
    torch.testing.assert_close(measured, torch.tensor([0.2, 0.2]), rtol=0.08, atol=0.01)
    torch.testing.assert_close(
        first.received_waveform,
        composition.received_waveform + first.noise_waveform,
    )


def test_received_waveform_composition_applies_per_link_power_gain_once():
    links = torch.ones((1, 1, 2, 1, 4), dtype=torch.complex64)
    small_scale_channel = torch.ones((1, 1, 1, 2, 1, 1, 1), dtype=torch.complex64)
    budget = apply_network_link_budget(
        small_scale_channel,
        path_loss_db=((0.0, 6.020599913279624),),
        tx_antenna_gain_dbi=(0.0, 0.0),
        rx_antenna_gain_dbi=(0.0,),
        tx_beam_gain_dbi=((0.0, 0.0),),
        rx_beam_gain_dbi=((0.0, 0.0),),
        transmit_power_w_per_re=(1.0, 1.0),
        input_channel_semantics="unit_mean_small_scale_fading",
    )
    composition = compose_received_waveforms(
        links,
        link_start_samples=((0, 0),),
        capture_num_samples=4,
        sample_rate_hz=30.72e6,
        link_power_gain_linear=budget.power_gain_linear,
    )

    torch.testing.assert_close(
        composition.source_contributions[0, 0, 0, :, 0],
        torch.tensor([1.0, 0.5], dtype=torch.complex64),
    )
    torch.testing.assert_close(
        composition.received_waveform,
        torch.full((1, 1, 1, 4), 1.5 + 0.0j, dtype=torch.complex64),
    )
    torch.testing.assert_close(
        composition.link_power_gain_linear,
        torch.tensor([[1.0, 0.25]], dtype=torch.float64),
    )


def test_received_waveform_composition_applies_cfo_on_capture_timeline():
    links = torch.ones((1, 1, 1, 1, 4), dtype=torch.complex128)
    composition = compose_received_waveforms(
        links,
        link_start_samples=((2,),),
        capture_num_samples=6,
        sample_rate_hz=8.0,
        link_cfo_hz=((2.0,),),
        link_initial_phase_rad=((torch.pi / 2,),),
    )

    torch.testing.assert_close(
        composition.received_waveform[0, 0, 0],
        torch.tensor([0, 0, -1j, 1, 1j, -1], dtype=torch.complex128),
        rtol=1e-14,
        atol=1e-14,
    )
    torch.testing.assert_close(composition.link_cfo_hz, torch.tensor([[2.0]], dtype=torch.float64))
    torch.testing.assert_close(
        composition.link_initial_phase_rad,
        torch.tensor([[torch.pi / 2]], dtype=torch.float64),
    )


def test_received_waveform_composition_applies_fractional_delay_per_link():
    sample_count = 128
    frequency_cycles_per_sample = 0.071
    fractional_delay = 0.375
    source_indices = torch.arange(sample_count, dtype=torch.float64)
    source = torch.exp(2j * torch.pi * frequency_cycles_per_sample * source_indices).to(
        torch.complex128
    )
    composition = compose_received_waveforms(
        source.reshape(1, 1, 1, 1, sample_count),
        link_start_samples=((3,),),
        capture_num_samples=sample_count + 3,
        sample_rate_hz=8.0,
        link_fractional_delay_samples=((fractional_delay,),),
    )

    capture_indices = torch.arange(24, sample_count - 20, dtype=torch.float64) + 3
    expected = torch.exp(
        2j * torch.pi * frequency_cycles_per_sample * (capture_indices - 3 - fractional_delay)
    )
    torch.testing.assert_close(
        composition.received_waveform[0, 0, 0, 24 + 3 : sample_count - 20 + 3],
        expected,
        atol=2e-5,
        rtol=2e-5,
    )
    torch.testing.assert_close(
        composition.link_fractional_delay_samples,
        torch.tensor([[fractional_delay]], dtype=torch.float64),
    )
    assert composition.fractional_delay_kernel_radius == 16


@pytest.mark.parametrize(
    "kwargs",
    [
        {"link_cfo_hz": ((True,),)},
        {"link_cfo_hz": ((1.0, 2.0),)},
        {"link_cfo_hz": ((1.0 + 0.0j,),)},
        {"link_cfo_hz": ((float("inf"),),)},
        {"link_cfo_hz": ((1e308,),), "sample_rate_hz": 1e-308},
        {"link_initial_phase_rad": ((False,),)},
        {"link_initial_phase_rad": ((0.0, 1.0),)},
        {"link_initial_phase_rad": ((float("nan"),),)},
    ],
)
def test_received_waveform_composition_rejects_invalid_link_oscillators(kwargs):
    links = torch.ones((1, 1, 1, 1, 4), dtype=torch.complex64)
    args = {
        "link_start_samples": ((0,),),
        "capture_num_samples": 4,
        "sample_rate_hz": 30.72e6,
    }
    args.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        compose_received_waveforms(links, **args)


@pytest.mark.parametrize(
    "fractional_delay",
    [
        ((True,),),
        ((1.0,),),
        ((1.0, 0.0),),
        ((1.0 + 0.0j,),),
        ((float("nan"),),),
    ],
)
def test_received_waveform_composition_rejects_invalid_fractional_delays(fractional_delay):
    links = torch.ones((1, 1, 1, 1, 8), dtype=torch.complex64)
    with pytest.raises((TypeError, ValueError)):
        compose_received_waveforms(
            links,
            link_start_samples=((0,),),
            capture_num_samples=8,
            sample_rate_hz=30.72e6,
            link_fractional_delay_samples=fractional_delay,
        )


def test_network_phase_noise_is_shared_by_transmitter_across_links_and_replayable():
    sample_count = 8192
    links = torch.ones((3, 2, 1, 1, sample_count), dtype=torch.complex128)
    args = {
        "link_start_samples": ((0,), (0,)),
        "capture_num_samples": sample_count,
        "sample_rate_hz": 1e6,
        "tx_phase_noise_linewidth_hz": (20_000.0,),
        "rx_phase_noise_linewidth_hz": (0.0, 0.0),
        "phase_noise_seed": 1907,
    }

    first = compose_received_waveforms(links, **args)
    replay = compose_received_waveforms(links, **args)
    different_seed = compose_received_waveforms(links, **{**args, "phase_noise_seed": 1908})

    torch.testing.assert_close(
        first.source_contributions, replay.source_contributions, rtol=0, atol=0
    )
    torch.testing.assert_close(
        first.source_contributions[:, 0, 0, 0],
        first.source_contributions[:, 1, 0, 0],
        rtol=0,
        atol=0,
    )
    assert not torch.equal(first.source_contributions, different_seed.source_contributions)
    assert not torch.equal(first.source_contributions[0], first.source_contributions[1])
    increments = torch.angle(
        first.source_contributions[0, 0, 0, 0, 1:]
        * first.source_contributions[0, 0, 0, 0, :-1].conj()
    )
    expected_variance = 2.0 * torch.pi * 20_000.0 / 1e6
    assert float(increments.var(unbiased=True)) == pytest.approx(expected_variance, rel=0.08)
    torch.testing.assert_close(
        first.tx_phase_noise_linewidth_hz,
        torch.tensor([20_000.0], dtype=torch.float64),
    )
    assert first.phase_noise_seed == 1907


def test_network_phase_noise_is_shared_by_receiver_across_transmitters():
    links = torch.ones((2, 1, 2, 1, 128), dtype=torch.complex64)
    composition = compose_received_waveforms(
        links,
        link_start_samples=((0, 0),),
        capture_num_samples=128,
        sample_rate_hz=1e6,
        tx_phase_noise_linewidth_hz=(0.0, 0.0),
        rx_phase_noise_linewidth_hz=(10_000.0,),
        phase_noise_seed=813,
    )

    torch.testing.assert_close(
        composition.source_contributions[:, 0, :, 0],
        composition.source_contributions[:, 0, :, 1],
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tx_phase_noise_linewidth_hz": (1.0,)},
        {"tx_phase_noise_linewidth_hz": (True,), "phase_noise_seed": 12},
        {"rx_phase_noise_linewidth_hz": (-1.0,), "phase_noise_seed": 12},
        {"tx_phase_noise_linewidth_hz": (float("inf"),), "phase_noise_seed": 12},
        {"rx_phase_noise_linewidth_hz": (1.0, 2.0), "phase_noise_seed": 12},
        {"tx_phase_noise_linewidth_hz": (1.0,), "phase_noise_seed": True},
        {"tx_phase_noise_linewidth_hz": (1.0,), "phase_noise_seed": -1},
    ],
)
def test_network_phase_noise_rejects_invalid_oscillator_inputs(kwargs):
    links = torch.ones((1, 1, 1, 1, 16), dtype=torch.complex64)
    args = {
        "link_start_samples": ((0,),),
        "capture_num_samples": 16,
        "sample_rate_hz": 1e6,
    }
    args.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        compose_received_waveforms(links, **args)


def test_network_phase_noise_counts_oscillator_work_against_capture_bound():
    links = torch.ones((1, 1, 1, 1, 16), dtype=torch.complex64)
    with pytest.raises(ValueError, match="oscillator state"):
        compose_received_waveforms(
            links,
            link_start_samples=((0,),),
            capture_num_samples=16,
            sample_rate_hz=1e6,
            tx_phase_noise_linewidth_hz=(1_000.0,),
            phase_noise_seed=41,
            max_output_elements=111,
        )


def test_network_receiver_iq_imbalance_is_applied_per_link_and_retained():
    links = torch.empty((1, 1, 2, 1, 8), dtype=torch.complex64)
    links[:, :, 0] = 1.0 + 2.0j
    links[:, :, 1] = 3.0 - 1.0j
    composition = compose_received_waveforms(
        links,
        link_start_samples=((0, 0),),
        capture_num_samples=8,
        sample_rate_hz=1e6,
        rx_iq_gain_imbalance_db=(6.0,),
        rx_iq_phase_imbalance_rad=(0.2,),
    )

    i_gain = 10.0 ** (6.0 / 40.0)
    q_gain = 10.0 ** (-6.0 / 40.0)
    actual = composition.source_contributions[0, 0, 0, 0, 0]
    assert actual.real.item() == pytest.approx(i_gain - q_gain * math.sin(0.2) * 2.0)
    assert actual.imag.item() == pytest.approx(q_gain * math.cos(0.2) * 2.0)
    torch.testing.assert_close(
        composition.received_waveform,
        composition.source_contributions.sum(dim=3),
    )
    torch.testing.assert_close(
        composition.rx_iq_gain_imbalance_db,
        torch.tensor([6.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        composition.rx_iq_phase_imbalance_rad,
        torch.tensor([0.2], dtype=torch.float64),
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rx_iq_gain_imbalance_db": (True,)},
        {"rx_iq_gain_imbalance_db": (float("inf"),)},
        {"rx_iq_gain_imbalance_db": (10000.0,)},
        {"rx_iq_phase_imbalance_rad": (float("nan"),)},
        {"rx_iq_phase_imbalance_rad": (0.1, 0.2)},
    ],
)
def test_network_receiver_iq_imbalance_rejects_invalid_inputs(kwargs):
    links = torch.ones((1, 1, 1, 1, 16), dtype=torch.complex64)
    args = {
        "link_start_samples": ((0,),),
        "capture_num_samples": 16,
        "sample_rate_hz": 1e6,
    }
    args.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        compose_received_waveforms(links, **args)


def test_network_receiver_iq_imbalance_work_counts_against_capture_bound():
    links = torch.ones((1, 1, 1, 1, 16), dtype=torch.complex64)
    with pytest.raises(ValueError, match="impairment work"):
        compose_received_waveforms(
            links,
            link_start_samples=((0,),),
            capture_num_samples=16,
            sample_rate_hz=1e6,
            rx_iq_gain_imbalance_db=(1.0,),
            max_output_elements=79,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"link_start_samples": ((False,),)},
        {"link_start_samples": ((0.0,),)},
        {"link_start_samples": ((0, 0),)},
        {"link_power_gain_linear": ((1.0, -0.1),)},
        {"link_power_gain_linear": ((True,),)},
        {"link_power_gain_linear": ((1e-300,),)},
        {"link_power_gain_linear": ((1e300,),)},
        {"capture_num_samples": 0},
        {"sample_rate_hz": float("nan")},
        {"max_output_elements": 1},
    ],
)
def test_received_waveform_composition_rejects_invalid_alignment_or_bounds(kwargs):
    links = torch.ones((1, 1, 1, 1, 4), dtype=torch.complex64)
    args = {
        "link_start_samples": ((0,),),
        "capture_num_samples": 4,
        "sample_rate_hz": 30.72e6,
    }
    args.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        compose_received_waveforms(links, **args)


def test_network_transmitter_iq_imbalance_wraps_phase_and_applies_branch_gains():
    waveform = torch.tensor([[[[1.0 + 2.0j, -0.5 + 0.25j]], [[2.0 - 1.0j, 0.5 + 1.5j]]]])
    result = apply_network_transmitter_iq_imbalance(
        waveform,
        tx_iq_gain_imbalance_db=(6.0, -3.0),
        tx_iq_phase_imbalance_rad=(0.2, 2.0 * math.pi + 0.3),
    )
    assert result.waveform.shape == waveform.shape
    assert torch.all(torch.isfinite(result.waveform))
    assert result.tx_iq_gain_imbalance_db.tolist() == [6.0, -3.0]
    assert result.tx_iq_phase_imbalance_rad.tolist() == pytest.approx([0.2, 2.0 * math.pi + 0.3])
    assert not torch.equal(result.waveform, waveform)


@pytest.mark.parametrize(
    ("waveform", "changes", "message"),
    [
        (torch.ones((1, 1, 1, 2)), {}, "dense complex"),
        (torch.ones((1, 1, 1, 2), dtype=torch.complex64).reshape(1, 1, 2), {}, "dense complex"),
        (torch.empty((1, 1, 1, 0), dtype=torch.complex64), {}, "axes must be non-empty"),
        (
            torch.tensor([[[[complex(float("inf"), 0)]]]]),
            {},
            "finite samples",
        ),
        (torch.ones((1, 1, 1, 2), dtype=torch.complex64), {"max_output_elements": 3}, "exceed"),
        (
            torch.ones((1, 1, 1, 2), dtype=torch.complex64),
            {"max_output_elements": True},
            "positive integer",
        ),
        (
            torch.ones((1, 1, 1, 2), dtype=torch.complex64),
            {"tx_iq_gain_imbalance_db": (0.0, 1.0)},
            "shape",
        ),
        (
            torch.ones((1, 1, 1, 2), dtype=torch.complex64),
            {"tx_iq_phase_imbalance_rad": (float("nan"),)},
            "finite",
        ),
    ],
)
def test_network_transmitter_iq_imbalance_rejects_invalid_inputs(waveform, changes, message):
    arguments = {
        "tx_iq_gain_imbalance_db": (0.0,),
        "tx_iq_phase_imbalance_rad": (0.0,),
    }
    arguments.update(changes)
    with pytest.raises((TypeError, ValueError), match=message):
        apply_network_transmitter_iq_imbalance(waveform, **arguments)


def test_network_transmitter_iq_imbalance_rejects_sparse_tensors_and_unrepresentable_scales():
    sparse = torch.sparse_coo_tensor(
        torch.tensor([[0, 0], [0, 0], [0, 0], [0, 1]]),
        torch.tensor([1.0 + 0.0j, 1.0 + 0.0j]),
        (1, 1, 1, 2),
    )
    with pytest.raises(TypeError, match="dense complex"):
        apply_network_transmitter_iq_imbalance(
            sparse,
            tx_iq_gain_imbalance_db=(0.0,),
            tx_iq_phase_imbalance_rad=(0.0,),
        )
    waveform = torch.ones((1, 1, 1, 2), dtype=torch.complex64)
    with pytest.raises(ValueError, match="not representable in waveform precision"):
        apply_network_transmitter_iq_imbalance(
            waveform,
            tx_iq_gain_imbalance_db=(4000.0,),
            tx_iq_phase_imbalance_rad=(0.0,),
        )


def test_network_transmitter_iq_imbalance_rejects_output_overflow_and_scale_underflow(monkeypatch):
    waveform = torch.full((1, 1, 1, 1), complex(3e38, 3e38), dtype=torch.complex64)
    with pytest.raises(ValueError, match="produced non-finite waveform"):
        apply_network_transmitter_iq_imbalance(
            waveform,
            tx_iq_gain_imbalance_db=(6.0,),
            tx_iq_phase_imbalance_rad=(0.0,),
        )

    powers = iter(
        (torch.tensor((1.0,), dtype=torch.float64), torch.tensor((1e-50,), dtype=torch.float64))
    )
    monkeypatch.setattr(torch, "pow", lambda *_args, **_kwargs: next(powers))
    with pytest.raises(ValueError, match="underflows waveform precision"):
        apply_network_transmitter_iq_imbalance(
            torch.ones((1, 1, 1, 1), dtype=torch.complex64),
            tx_iq_gain_imbalance_db=(0.0,),
            tx_iq_phase_imbalance_rad=(0.0,),
        )


@pytest.mark.parametrize(
    ("channel", "changes", "error", "message"),
    [
        (torch.ones((1, 1, 1, 1, 1, 1, 1)), {}, TypeError, "must be a complex PyTorch tensor"),
        (
            torch.ones((1, 1, 1, 1, 1, 1), dtype=torch.complex64),
            {},
            ValueError,
            "axes must be",
        ),
        (
            torch.empty((1, 1, 1, 1, 1, 1, 0), dtype=torch.complex64),
            {},
            ValueError,
            "axes must be non-empty",
        ),
        (
            torch.full((1, 1, 1, 1, 1, 1, 1), complex(float("nan"), 0), dtype=torch.complex64),
            {},
            ValueError,
            "must be finite",
        ),
        (
            torch.ones((1, 1, 1, 1, 1, 1, 1), dtype=torch.complex64),
            {"subcarrier_frequencies_hz": (0.0,)},
            ValueError,
            "strictly positive",
        ),
        (
            torch.ones((1, 1, 1, 1, 1, 1, 1), dtype=torch.complex64),
            {"tx_antenna_gain_dbi": (10000.0,)},
            ValueError,
            "not representable in float64",
        ),
    ],
)
def test_link_budget_rejects_malformed_channels_and_unrepresentable_gains(
    channel, changes, error, message
):
    arguments = {
        "path_loss_db": ((80.0,),),
        "tx_antenna_gain_dbi": (0.0,),
        "rx_antenna_gain_dbi": (0.0,),
        "tx_beam_gain_dbi": ((0.0,),),
        "rx_beam_gain_dbi": ((0.0,),),
        "transmit_power_w_per_re": (1.0,),
        "input_channel_semantics": "unit_mean_small_scale_fading",
    }
    arguments.update(changes)
    with pytest.raises(error, match=message):
        apply_network_link_budget(channel, **arguments)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            {"tx_antenna_gain_dbi": (200.0,)},
            "scaled channel",
        ),
        ({"input_channel_semantics": "includes_path_loss"}, "must not be applied twice"),
    ],
)
def test_link_budget_rejects_scaled_overflow_and_double_counting(changes, message):
    channel = torch.ones((1, 1, 1, 1, 1, 1, 1), dtype=torch.complex64) * 1e30
    arguments = {
        "path_loss_db": ((0.0,),),
        "tx_antenna_gain_dbi": (0.0,),
        "rx_antenna_gain_dbi": (0.0,),
        "tx_beam_gain_dbi": ((0.0,),),
        "rx_beam_gain_dbi": ((0.0,),),
        "transmit_power_w_per_re": (1.0,),
        "input_channel_semantics": "unit_mean_small_scale_fading",
    }
    arguments.update(changes)
    with pytest.raises(ValueError, match=message):
        apply_network_link_budget(channel, **arguments)


def test_link_budget_covers_conversion_dtype_and_received_power_overflow():
    channel = torch.ones((1, 1, 1, 1, 1, 1, 1), dtype=torch.complex64)
    arguments = {
        "path_loss_db": ((80.0,),),
        "tx_antenna_gain_dbi": (0.0,),
        "rx_antenna_gain_dbi": (0.0,),
        "tx_beam_gain_dbi": ((0.0,),),
        "rx_beam_gain_dbi": ((0.0,),),
        "transmit_power_w_per_re": (1.0,),
        "input_channel_semantics": "unit_mean_small_scale_fading",
    }
    with pytest.raises(ValueError, match="path_loss_db must contain real numeric values"):
        apply_network_link_budget(channel, **{**arguments, "path_loss_db": object()})
    with pytest.raises(ValueError, match="not representable in channel dtype"):
        apply_network_link_budget(
            channel,
            **{**arguments, "path_loss_db": ((0.0,),), "tx_antenna_gain_dbi": (800.0,)},
        )

    channel64 = channel.to(torch.complex128)
    with pytest.raises(ValueError, match="mean received link power is not representable"):
        apply_network_link_budget(
            channel64,
            **{
                **arguments,
                "path_loss_db": ((0.0,),),
                "tx_antenna_gain_dbi": (400.0,),
                "transmit_power_w_per_re": (1e308,),
            },
        )


@pytest.mark.parametrize(
    ("channel_changes", "grid_changes", "error", "message"),
    [
        (
            {"channel": torch.ones((1, 2, 1, 2, 1, 2, 3))},
            {},
            TypeError,
            "channel must be a complex",
        ),
        ({}, {"grids": torch.ones((1, 2, 1, 2, 3))}, TypeError, "transmit_grids must be a complex"),
        (
            {"channel": torch.ones((1, 2, 1, 2, 1, 2), dtype=torch.complex64)},
            {},
            ValueError,
            "channel axes",
        ),
        (
            {},
            {"grids": torch.ones((1, 2, 1, 2), dtype=torch.complex64)},
            ValueError,
            "transmit_grids axes",
        ),
        (
            {"channel": torch.ones((2, 2, 1, 2, 1, 2, 3), dtype=torch.complex64)},
            {},
            ValueError,
            "batch dimensions differ",
        ),
        (
            {},
            {"grids": torch.ones((1, 2, 2, 2, 3), dtype=torch.complex64)},
            ValueError,
            "axes do not match",
        ),
        (
            {"channel": torch.empty((1, 2, 1, 2, 1, 0, 3), dtype=torch.complex64)},
            {"grids": torch.empty((1, 2, 1, 0, 3), dtype=torch.complex64)},
            ValueError,
            "axes must be non-empty",
        ),
        (
            {"channel": torch.ones((1, 2, 1, 2, 1, 2, 3), dtype=torch.complex128)},
            {},
            ValueError,
            "same complex dtype",
        ),
        (
            {
                "channel": torch.full(
                    (1, 2, 1, 2, 1, 2, 3), complex(float("inf"), 0), dtype=torch.complex64
                )
            },
            {},
            ValueError,
            "must be finite",
        ),
    ],
)
def test_grid_composition_rejects_invalid_tensor_layouts(
    channel_changes, grid_changes, error, message
):
    channel, grids = _two_node_fixture()
    channel = channel_changes.get("channel", channel)
    grids = grid_changes.get("grids", grids)
    with pytest.raises(error, match=message):
        compose_downlink_resource_grids(channel, grids, (1.0, 1.0))


def test_grid_composition_rejects_device_and_power_overflow():
    channel = torch.ones((1, 1, 1, 1, 1, 1, 1), dtype=torch.complex64, device="meta")
    grids = torch.ones((1, 1, 1, 1, 1), dtype=torch.complex64)
    with pytest.raises(ValueError, match="same device"):
        compose_downlink_resource_grids(channel, grids, (1.0,))

    channel128 = torch.ones((1, 1, 1, 1, 1, 1, 1), dtype=torch.complex128)
    huge_grids128 = torch.full((1, 1, 1, 1, 1), complex(1e200, 0), dtype=torch.complex128)
    with pytest.raises(ValueError, match="grid power is not representable"):
        compose_downlink_resource_grids(channel128, huge_grids128, (1.0,))

    channel64 = torch.ones((1, 1, 1, 1, 1, 1, 1), dtype=torch.complex64)
    large_grids = torch.full((1, 1, 1, 1, 1), complex(1e30, 0), dtype=torch.complex64)
    with pytest.raises(ValueError, match="scaled transmit grids are not finite"):
        compose_downlink_resource_grids(channel64, large_grids, (1e100,))


def test_grid_composition_rejects_nonfinite_crosslink_and_superposition():
    channel = torch.full((1, 1, 1, 1, 1, 1, 1), complex(1e30, 0), dtype=torch.complex64)
    grids = torch.ones((1, 1, 1, 1, 1), dtype=torch.complex64)
    with pytest.raises(ValueError, match="source contributions are not finite"):
        compose_downlink_resource_grids(channel, grids, (1e20,))

    channel = torch.ones((1, 1, 1, 2, 1, 1, 1), dtype=torch.complex64)
    grids = torch.full((1, 2, 1, 1, 1), complex(2e38, 0), dtype=torch.complex64)
    with pytest.raises(ValueError, match="superposed received grid is not finite"):
        compose_downlink_resource_grids(channel, grids, (4e76, 4e76))


def test_waveform_composition_handles_empty_capture_intersection():
    links = torch.ones((1, 1, 1, 1, 3), dtype=torch.complex64)
    result = compose_received_waveforms(
        links,
        link_start_samples=((10,),),
        capture_num_samples=4,
        sample_rate_hz=1_000_000.0,
    )
    assert torch.count_nonzero(result.received_waveform) == 0
    assert torch.count_nonzero(result.source_contributions) == 0


def test_waveform_composition_combines_fractional_delay_with_complex_fir():
    links = torch.tensor([[[[[1.0 + 0.0j, 2.0 + 0.0j, 0.5 + 0.0j]]]]])
    taps = torch.tensor([[[1.0 + 0.0j, 0.0 + 0.5j]]], dtype=torch.complex64)
    result = compose_received_waveforms(
        links,
        link_start_samples=((0,),),
        capture_num_samples=5,
        sample_rate_hz=1_000_000.0,
        link_filter_taps=taps,
        link_fractional_delay_samples=((0.25,),),
    )
    assert result.received_waveform.shape == (1, 1, 1, 5)
    assert torch.all(torch.isfinite(result.received_waveform))
    assert not torch.equal(result.received_waveform[..., :3], links[..., :3])


def test_waveform_composition_rejects_nonfinite_output_after_gain():
    links = torch.full((1, 1, 1, 1, 2), complex(1e30, 0), dtype=torch.complex64)
    with pytest.raises(ValueError, match="composed receiver waveform is not finite"):
        compose_received_waveforms(
            links,
            link_start_samples=((0,),),
            capture_num_samples=2,
            sample_rate_hz=1_000_000.0,
            link_power_gain_linear=((1e20,),),
        )


@pytest.mark.parametrize(
    ("links", "changes", "error", "message"),
    [
        (torch.ones((1, 1, 1, 1, 2)), {}, TypeError, "complex PyTorch tensor"),
        (torch.ones((1, 1, 1, 2), dtype=torch.complex64), {}, ValueError, "sample] axes"),
        (torch.empty((1, 1, 1, 1, 0), dtype=torch.complex64), {}, ValueError, "non-empty"),
        (
            torch.full((1, 1, 1, 1, 2), complex(float("nan"), 0), dtype=torch.complex64),
            {},
            ValueError,
            "finite samples",
        ),
        (
            torch.ones((1, 1, 1, 1, 2), dtype=torch.complex64),
            {"capture_num_samples": True},
            ValueError,
            "positive integer",
        ),
        (
            torch.ones((1, 1, 1, 1, 2), dtype=torch.complex64),
            {"max_output_elements": 0},
            ValueError,
            "positive integer",
        ),
        (
            torch.ones((1, 1, 1, 1, 2), dtype=torch.complex64),
            {"sample_rate_hz": float("inf")},
            ValueError,
            "sample_rate_hz must be finite",
        ),
        (
            torch.ones((1, 1, 1, 1, 2), dtype=torch.complex64),
            {"link_start_samples": object()},
            ValueError,
            "integer sample offsets",
        ),
        (
            torch.ones((1, 1, 1, 1, 2), dtype=torch.complex64),
            {"link_filter_taps": object()},
            ValueError,
            "complex numeric values",
        ),
        (
            torch.ones((1, 1, 1, 1, 2), dtype=torch.complex64),
            {"link_filter_taps": torch.tensor([[[1e-50 + 0.0j]]], dtype=torch.complex128)},
            ValueError,
            "not representable in waveform precision",
        ),
    ],
)
def test_waveform_composition_rejects_malformed_inputs(links, changes, error, message):
    arguments = {
        "link_start_samples": ((0,),),
        "capture_num_samples": 2,
        "sample_rate_hz": 1_000_000.0,
    }
    arguments.update(changes)
    with pytest.raises(error, match=message):
        compose_received_waveforms(links, **arguments)


@pytest.mark.parametrize(
    ("field", "value", "error", "message"),
    [
        ("received_waveform", None, TypeError, "complex \\[batch, rx"),
        ("source_contributions", None, TypeError, "complex per-link contributions"),
        (
            "source_contributions",
            torch.ones((1, 1, 1, 1), dtype=torch.complex64),
            ValueError,
            "invalid axes",
        ),
        (
            "source_contributions",
            torch.ones((1, 1, 2, 1, 4), dtype=torch.complex64),
            ValueError,
            "do not match",
        ),
        ("link_start_samples", None, ValueError, "one sample offset"),
        ("link_power_gain_linear", None, ValueError, "finite nonnegative RX/TX power gains"),
        (
            "link_power_gain_linear",
            torch.tensor(((-1.0,),)),
            ValueError,
            "finite nonnegative RX/TX power gains",
        ),
        ("sample_rate_hz", float("nan"), ValueError, "sample rate must be finite"),
        (
            "received_waveform",
            torch.zeros((1, 1, 1, 4), dtype=torch.complex128),
            ValueError,
            "share dtype and device",
        ),
        (
            "received_waveform",
            torch.zeros((1, 1, 1, 4), dtype=torch.complex64),
            ValueError,
            "must equal the sum",
        ),
    ],
)
def test_waveform_awgn_rejects_inconsistent_composition_fields(field, value, error, message):
    links = torch.ones((1, 1, 1, 1, 4), dtype=torch.complex64)
    composition = compose_received_waveforms(
        links,
        link_start_samples=((0,),),
        capture_num_samples=4,
        sample_rate_hz=1_000_000.0,
    )
    with pytest.raises(error, match=message):
        add_independent_receiver_waveform_awgn(
            replace(composition, **{field: value}), (0.1,), seed=2
        )


@pytest.mark.parametrize("seed", [True, -1, 2**63])
def test_waveform_awgn_rejects_invalid_or_unrepresentable_seeds(seed):
    links = torch.ones((1, 2, 1, 1, 4), dtype=torch.complex64)
    composition = compose_received_waveforms(
        links,
        link_start_samples=((0,), (0,)),
        capture_num_samples=4,
        sample_rate_hz=1_000_000.0,
    )
    with pytest.raises(ValueError, match="seed"):
        add_independent_receiver_waveform_awgn(composition, (0.1, 0.1), seed=seed)


def test_waveform_awgn_rejects_nonfinite_composition_values():
    links = torch.ones((1, 1, 1, 1, 4), dtype=torch.complex64)
    composition = compose_received_waveforms(
        links,
        link_start_samples=((0,),),
        capture_num_samples=4,
        sample_rate_hz=1_000_000.0,
    )
    nonfinite = torch.full_like(composition.received_waveform, complex(float("inf"), 0))
    nonfinite_contributions = torch.full_like(
        composition.source_contributions, complex(float("inf"), 0)
    )
    composition = replace(
        composition,
        received_waveform=nonfinite,
        source_contributions=nonfinite_contributions,
    )
    with pytest.raises(ValueError, match="composition waveforms must be finite"):
        add_independent_receiver_waveform_awgn(composition, (0.1,), seed=2)


def test_network_operation_entrypoints_report_missing_torch(monkeypatch):
    import builtins

    original_import = builtins.__import__

    def import_without_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("hidden torch")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_torch)
    calls = (
        lambda: apply_network_transmitter_iq_imbalance(
            None, tx_iq_gain_imbalance_db=(), tx_iq_phase_imbalance_rad=()
        ),
        lambda: apply_network_link_budget(
            None,
            path_loss_db=(),
            tx_antenna_gain_dbi=(),
            rx_antenna_gain_dbi=(),
            tx_beam_gain_dbi=(),
            rx_beam_gain_dbi=(),
            transmit_power_w_per_re=(),
            input_channel_semantics="",
        ),
        lambda: compose_downlink_resource_grids(None, None, ()),
        lambda: compose_received_waveforms(
            None, link_start_samples=(), capture_num_samples=0, sample_rate_hz=0
        ),
        lambda: add_independent_receiver_waveform_awgn(None, (), seed=0),
        lambda: network_sinr(None, (), ()),
        lambda: network_resource_sinr(None, (), ()),
        lambda: add_independent_receiver_awgn(None, (), seed=0),
        lambda: propagate_network_transmit_waveforms(None, None),
    )
    messages = (
        "apply network transmitter impairments",
        "apply network link budgets",
        "compose network resource grids",
        "compose received waveforms",
        "add receiver waveform noise",
        "measure network SINR",
        "measure per-resource network SINR",
        "add network receiver noise",
        "propagate network transmitter waveforms",
    )
    for call, message in zip(calls, messages, strict=True):
        with pytest.raises(RuntimeError, match=message):
            call()


@pytest.mark.parametrize(
    ("serving", "noise", "message"),
    [
        ((0,), (0.1, 0.1), "one transmitter index per receiver"),
        ((True, 0), (0.1, 0.1), "existing transmitter"),
        ((4, 0), (0.1, 0.1), "existing transmitter"),
    ],
)
def test_network_sinr_rejects_invalid_serving_rosters(serving, noise, message):
    channel, grids = _two_node_fixture()
    composition = compose_downlink_resource_grids(channel, grids, (1.0, 1.0))
    with pytest.raises(ValueError, match=message):
        network_sinr(composition, serving, noise)


def test_network_sinr_rejects_unrepresentable_power_and_ratio():
    channel = torch.ones((1, 1, 1, 1, 1, 1, 1), dtype=torch.complex128)
    grid = torch.ones((1, 1, 1, 1, 1), dtype=torch.complex128)
    composition = compose_downlink_resource_grids(channel, grid, (1.0,))
    huge_contribution = torch.full_like(composition.source_contributions, complex(1e200, 0))
    composition = replace(
        composition,
        source_contributions=huge_contribution,
        received_grid=huge_contribution.sum(dim=3),
    )
    with pytest.raises(ValueError, match="measured network powers are not representable"):
        network_sinr(composition, (0,), (0.0,))

    channel = torch.ones((1, 1, 1, 1, 1, 1, 1), dtype=torch.complex128)
    grid = torch.ones((1, 1, 1, 1, 1), dtype=torch.complex128)
    composition = compose_downlink_resource_grids(channel, grid, (1.0,))
    with pytest.raises(ValueError, match="network SINR is not representable"):
        network_sinr(composition, (0,), (1e-320,))


@pytest.mark.parametrize("seed", [True, -1, 2**63])
def test_grid_awgn_rejects_invalid_or_unrepresentable_seeds(seed):
    channel, grids = _two_node_fixture()
    composition = compose_downlink_resource_grids(channel, grids, (1.0, 1.0))
    with pytest.raises(ValueError, match="seed"):
        add_independent_receiver_awgn(composition, (0.1, 0.1), seed=seed)


def test_waveform_awgn_rejects_empty_axes_and_nonfinite_added_noise(monkeypatch):
    links = torch.ones((1, 1, 1, 1, 4), dtype=torch.complex64)
    composition = compose_received_waveforms(
        links,
        link_start_samples=((0,),),
        capture_num_samples=4,
        sample_rate_hz=1_000_000.0,
    )
    empty = replace(
        composition,
        received_waveform=torch.empty((1, 1, 1, 0), dtype=torch.complex64),
        source_contributions=torch.empty((1, 1, 1, 1, 0), dtype=torch.complex64),
    )
    with pytest.raises(ValueError, match="axes must be non-empty"):
        add_independent_receiver_waveform_awgn(empty, (0.1,), seed=1)

    import integrations.sionna.noise as noise_module

    monkeypatch.setattr(
        noise_module,
        "proper_complex_awgn",
        lambda reference, *_args, **_kwargs: torch.full_like(reference, complex(float("inf"), 0)),
    )
    with pytest.raises(ValueError, match="receiver waveform plus noise is not finite"):
        add_independent_receiver_waveform_awgn(composition, (0.1,), seed=1)


@pytest.mark.parametrize(
    ("serving", "message"),
    [
        (object(), "must contain integer transmitter indices"),
        (
            torch.sparse_coo_tensor(torch.tensor([[0]]), torch.tensor([0]), (1,)),
            "dense integer array",
        ),
    ],
)
def test_resource_sinr_rejects_unreadable_serving_maps(serving, message):
    channel, grids = _two_node_fixture()
    composition = compose_downlink_resource_grids(channel, grids, (1.0, 1.0))
    with pytest.raises(ValueError, match=message):
        network_resource_sinr(composition, serving, (0.1, 0.1))


def test_resource_sinr_rejects_unrepresentable_powers_and_ratios():
    channel = torch.ones((1, 1, 1, 1, 1, 1, 1), dtype=torch.complex128)
    grid = torch.ones((1, 1, 1, 1, 1), dtype=torch.complex128)
    composition = compose_downlink_resource_grids(channel, grid, (1.0,))
    huge = torch.full_like(composition.source_contributions, complex(1e200, 0))
    composition = replace(composition, source_contributions=huge, received_grid=huge.sum(dim=3))
    with pytest.raises(ValueError, match="measured per-resource network powers"):
        network_resource_sinr(composition, (0,), (0.0,))

    composition = compose_downlink_resource_grids(channel, grid, (1.0,))
    with pytest.raises(ValueError, match="per-resource network SINR is not representable"):
        network_resource_sinr(composition, (0,), (1e-320,))
