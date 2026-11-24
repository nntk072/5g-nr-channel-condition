"""Network grid SINR workflow: tensor rank, aligned axes, element budget and link-budget declarations."""

from __future__ import annotations

import pytest
from integrations.sionna.network_sinr import run_network_grid_sinr

torch = pytest.importorskip("torch")


def _channel(batch=1, rx=1, rx_ports=1, tx=1, tx_ports=1, symbols=2, subcarriers=4):
    return torch.ones(
        batch, rx, rx_ports, tx, tx_ports, symbols, subcarriers, dtype=torch.complex64
    )


def _grids(batch=1, tx=1, tx_ports=1, symbols=2, subcarriers=4):
    return torch.ones(batch, tx, tx_ports, symbols, subcarriers, dtype=torch.complex64)


def _run(channel, grids, **kwargs):
    options = {
        "transmit_power_w_per_re": 1e-3,
        "serving_tx_indices": [0],
        "noise_variance_w_per_re": 1e-6,
    }
    options.update(kwargs)
    return run_network_grid_sinr(channel, grids, **options)


def test_channel_and_grid_ranks_are_fixed() -> None:
    with pytest.raises(TypeError, match="channel must be a rank-7 complex"):
        _run(torch.ones(2, 2, dtype=torch.complex64), _grids())
    with pytest.raises(TypeError, match="transmit_grids must be a rank-5 complex"):
        _run(_channel(), torch.ones(2, 2, dtype=torch.complex64))


def test_batch_and_axis_alignment_is_enforced() -> None:
    with pytest.raises(ValueError, match="batch dimensions differ"):
        _run(_channel(batch=2), _grids(batch=1))
    with pytest.raises(ValueError, match="do not match transmit grids"):
        _run(_channel(symbols=3), _grids(symbols=2))
    with pytest.raises(ValueError, match="must be non-empty"):
        _run(_channel(subcarriers=0), _grids(subcarriers=0))


def test_element_budget_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_tensor_elements must be a positive integer"):
        _run(_channel(), _grids(), max_tensor_elements=0)


def test_link_budget_declarations_are_checked() -> None:
    with pytest.raises(TypeError, match="must be a dictionary when provided"):
        _run(_channel(), _grids(), link_budget_parameters=["not", "a", "dict"])
    with pytest.raises(ValueError, match="must declare all five profiles"):
        _run(_channel(), _grids(), link_budget_parameters={"path_loss_db": 90.0})
    budget = {
        "path_loss_db": 90.0,
        "tx_antenna_gain_dbi": 0.0,
        "rx_antenna_gain_dbi": 0.0,
        "tx_beam_gain_dbi": 0.0,
        "rx_beam_gain_dbi": 0.0,
        "input_channel_semantics": "absolute_power",
    }
    with pytest.raises(ValueError, match="requires unit_mean_small_scale_fading"):
        _run(_channel(), _grids(), link_budget_parameters=budget)
