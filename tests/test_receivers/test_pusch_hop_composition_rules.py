"""PUSCH intra-slot hop composition and replay: carrier, BWP, grid and metadata contracts."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from receivers.nr.pusch_core.hopping import (
    compose_pusch_intra_slot_hops,
    verify_pusch_intra_slot_hops,
)


def _options(**changes: Any) -> dict[str, Any]:
    options: dict[str, Any] = {
        "carrier_grid_size_rb": 52,
        "carrier_grid_start_rb": 0,
        "bwp_start_rb": 8,
        "bwp_size_rb": 24,
        "allocation_start_symbol": 4,
        "first_hop_rb_start": 2,
        "second_hop_rb_start": 14,
        "first_hop_symbol_count": 3,
    }
    options.update(changes)
    return options


def _grid(shape=(1, 1, 1, 6, 24), dtype=np.complex64) -> np.ndarray:
    return np.ones(shape, dtype=dtype)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        pytest.param(
            {"carrier_grid_size_rb": 0}, "carrier_grid_size_rb must be in", id="carrier-small"
        ),
        pytest.param(
            {"carrier_grid_size_rb": 276}, "carrier_grid_size_rb must be in", id="carrier-large"
        ),
        pytest.param({"bwp_size_rb": 0}, "bwp_size_rb must be in", id="bwp-small"),
        pytest.param(
            {"carrier_grid_start_rb": -1}, "carrier grid must fit", id="carrier-negative-start"
        ),
        pytest.param(
            {"carrier_grid_start_rb": 5000}, "carrier grid must fit", id="carrier-far-start"
        ),
        pytest.param({"bwp_start_rb": 9000}, "BWP must fit within absolute", id="bwp-far"),
        pytest.param(
            {"carrier_grid_start_rb": 10},
            "BWP must lie within the carrier grid",
            id="bwp-before-carrier",
        ),
        pytest.param(
            {"bwp_size_rb": 60}, "BWP must lie within the carrier grid", id="bwp-after-carrier"
        ),
        pytest.param(
            {"max_output_elements": 0},
            "max_output_elements must be a positive integer",
            id="max-output",
        ),
    ],
)
def test_composition_rejects_invalid_carrier_and_bwp_geometry(changes, message) -> None:
    with pytest.raises(ValueError, match=message):
        compose_pusch_intra_slot_hops(_grid(), **_options(**changes))


def test_composition_requires_an_array_and_a_bwp_grid_shape() -> None:
    with pytest.raises(TypeError, match="NumPy array"):
        compose_pusch_intra_slot_hops([[1.0]], **_options())
    with pytest.raises(ValueError, match="five axes"):
        compose_pusch_intra_slot_hops(np.ones((6, 24), dtype=np.complex64), **_options())
    with pytest.raises(ValueError, match="complex64 or complex128"):
        compose_pusch_intra_slot_hops(np.ones((1, 1, 1, 6, 24), dtype=np.float32), **_options())
    bad = _grid()
    bad[0, 0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="only finite values"):
        compose_pusch_intra_slot_hops(bad, **_options())
    with pytest.raises(ValueError, match="12 times an in-BWP RB count"):
        compose_pusch_intra_slot_hops(_grid(shape=(1, 1, 1, 6, 23)), **_options())


def test_verifier_rejects_unknown_schema_and_malformed_layout() -> None:
    grid = _grid()
    result = compose_pusch_intra_slot_hops(grid, **_options())
    with pytest.raises(ValueError, match="unsupported PUSCH hop-grid metadata schema"):
        verify_pusch_intra_slot_hops(
            grid, result.frequency_grid, {**result.metadata, "schema": "other"}
        )
    with pytest.raises(ValueError, match="complex64 or complex128"):
        verify_pusch_intra_slot_hops(grid, result.frequency_grid.real, result.metadata)
