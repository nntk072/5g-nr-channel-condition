from __future__ import annotations

import pytest
from integrations.sionna.network_sinr import run_network_grid_sinr

torch = pytest.importorskip("torch")


def _inputs():
    # One receive antenna, two transmitters, one symbol, two subcarriers.
    channel = torch.ones((1, 1, 1, 2, 1, 1, 2), dtype=torch.complex64)
    grids = torch.zeros((1, 2, 1, 1, 2), dtype=torch.complex64)
    grids[:, 0] = 1
    grids[:, 1] = 1
    return channel, grids


def test_network_grid_sinr_workflow_is_hash_bound_and_replayable():
    channel, grids = _inputs()
    kwargs = {
        "transmit_power_w_per_re": (1.0, 1.0),
        "serving_tx_indices": (0,),
        "noise_variance_w_per_re": (0.25,),
    }
    first = run_network_grid_sinr(channel, grids, **kwargs)
    second = run_network_grid_sinr(channel, grids, **kwargs)

    assert torch.allclose(
        first.metrics.desired_power_w_per_re, torch.ones((1, 1, 2), dtype=torch.float64)
    )
    assert torch.allclose(
        first.metrics.interference_power_w_per_re, torch.ones((1, 1, 2), dtype=torch.float64)
    )
    assert torch.allclose(
        first.metrics.sinr_linear, torch.full((1, 1, 2), 0.8, dtype=torch.float64)
    )
    assert first.aggregate_metrics is not None
    assert first.aggregate_metrics.sinr_linear.item() == pytest.approx(0.8)
    assert "network_sinr" in first.report["implementation_sha256"]
    assert first.report["aggregate_metrics_sha256"] is not None
    assert first.report == second.report
    assert first.report["status"] == "OBSERVED_NOT_QUALIFIED"
    assert len(first.report["report_sha256"]) == 64


def test_network_grid_sinr_workflow_preflights_allocation():
    channel, grids = _inputs()
    with pytest.raises(ValueError, match="before composition"):
        run_network_grid_sinr(
            channel,
            grids,
            transmit_power_w_per_re=(1.0, 1.0),
            serving_tx_indices=(0,),
            noise_variance_w_per_re=(0.25,),
            max_tensor_elements=1,
        )


def test_resource_varying_serving_map_has_no_misleading_fixed_map_aggregate():
    channel, grids = _inputs()
    result = run_network_grid_sinr(
        channel,
        grids,
        transmit_power_w_per_re=(1.0, 1.0),
        serving_tx_indices=torch.tensor([[[0, 1]]]),
        noise_variance_w_per_re=(0.25,),
    )

    assert result.aggregate_metrics is None
    assert result.report["aggregate_metrics_sha256"] is None
    assert "network_sinr" not in result.report["implementation_sha256"]


def test_network_grid_sinr_workflow_binds_declared_link_budget():
    channel, grids = _inputs()
    result = run_network_grid_sinr(
        channel,
        grids,
        transmit_power_w_per_re=(1.0, 1.0),
        serving_tx_indices=(0,),
        noise_variance_w_per_re=(0.25,),
        link_budget_parameters={
            "path_loss_db": [[6.0, 0.0]],
            "tx_antenna_gain_dbi": [0.0, 0.0],
            "rx_antenna_gain_dbi": [0.0],
            "tx_beam_gain_dbi": [[0.0, 0.0]],
            "rx_beam_gain_dbi": [[0.0, 0.0]],
            "input_channel_semantics": "unit_mean_small_scale_fading",
        },
    )

    assert result.link_budget is not None
    assert result.report["input_channel_semantics"] == (
        "unit_mean_small_scale_fading_with_declared_profiles"
    )
    assert len(result.report["link_budget_sha256"]) == 64
    assert "link_budget" in result.report["implementation_sha256"]
    assert result.metrics.desired_power_w_per_re[0, 0, 0].item() == pytest.approx(10**-0.6)
