"""Model capacity matching and sparse multipath input contracts."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from models.capacity import (
    architecture_widths,
    capacity_spec,
    matched_capacity_names,
)
from models.multipath import fit_sparse_multipath


def test_unknown_capacity_names_are_refused() -> None:
    with pytest.raises(ValueError, match="capacity must be one of"):
        capacity_spec("gigantic")
    with pytest.raises(ValueError, match="capacity must be one of"):
        capacity_spec(None)


def test_architecture_widths_follow_preset_or_matched_names() -> None:
    assert architecture_widths("medium", "mlp") == capacity_spec("medium").mlp_widths
    with pytest.raises(ValueError, match="unsupported neural architecture"):
        architecture_widths("medium", "transformer")
    with pytest.raises(ValueError, match="must be resolved by the experiment runner"):
        architecture_widths("matched", "mlp")
    with pytest.raises(ValueError, match="resolved for a different architecture"):
        architecture_widths("matched_mlp_8_8_8", "real_cnn")
    with pytest.raises(ValueError, match="matched widths must be positive"):
        architecture_widths("matched_mlp_0_8_8", "mlp")
    assert architecture_widths("matched_mlp_8_16_4", "mlp") == (8, 16, 4)


@pytest.mark.parametrize(("links", "subcarriers"), [(0, 16), (1, 7), (True, 16), (1, True)])
def test_matched_capacity_requires_positive_links_and_eight_tones(links, subcarriers) -> None:
    with pytest.raises(ValueError, match="positive links and at least eight tones"):
        matched_capacity_names(links=links, subcarriers=subcarriers)


def _observed(size: int = 16) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frequencies = np.linspace(-1.0, 1.0, size) * 1e6
    observed = np.exp(-2j * np.pi * frequencies * 1e-7)
    mask = np.ones(size, dtype=bool)
    return observed, frequencies, mask


@pytest.mark.parametrize(
    ("change", "message"),
    [
        pytest.param(
            {"observed": np.ones((2, 2), dtype=complex)}, "must be vectors", id="observed-matrix"
        ),
        pytest.param(
            {"frequencies": np.linspace(0, 1, 15)},
            "matching lengths and at least four tones",
            id="length",
        ),
        pytest.param(
            {"mask": np.zeros(16, dtype=bool)},
            "at least four observed pilot tones",
            id="few-pilots",
        ),
        pytest.param({"mask": np.ones(16, dtype=int)}, "pilot", id="mask-dtype"),
        pytest.param({"observed": np.full(16, np.nan + 0j)}, "must be finite", id="nonfinite-cfr"),
        pytest.param(
            {"frequencies": np.full(16, np.inf)},
            "frequencies_hz must be finite",
            id="nonfinite-frequency",
        ),
        pytest.param(
            {"frequencies": np.zeros(16)},
            "at least four distinct frequencies",
            id="repeated-frequency",
        ),
        pytest.param({"maximum_paths": True}, "maximum_paths", id="paths-bool"),
        pytest.param({"maximum_paths": 0}, "maximum_paths must be positive", id="paths-zero"),
        pytest.param({"maximum_delay_s": 0.0}, "maximum_delay_s", id="delay-zero"),
        pytest.param(
            {"delay_grid_size": 9.5}, "delay_grid_size must be an integer", id="grid-type"
        ),
        pytest.param({"delay_grid_size": 8}, "delay_grid_size must be at least 9", id="grid-small"),
    ],
)
def test_sparse_multipath_rejects_invalid_inputs(change, message) -> None:
    observed, frequencies, mask = _observed()
    arguments: dict[str, Any] = {
        "observed_cfr": change.get("observed", observed),
        "frequencies_hz": change.get("frequencies", frequencies),
        "pilot_mask": change.get("mask", mask),
        "maximum_delay_s": 1e-6,
    }
    for key in ("maximum_paths", "maximum_delay_s", "delay_grid_size"):
        if key in change:
            arguments[key] = change[key]
    with pytest.raises((TypeError, ValueError), match=message):
        fit_sparse_multipath(**arguments)
