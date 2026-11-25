"""PUSCH intra-slot hop composition and replay: carrier, BWP, grid and metadata contracts."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from receivers.nr.pusch_core.hopping import (
    compose_pusch_intra_slot_hops,
    dehop_pusch_intra_slot_received_grid,
    verify_pusch_intra_slot_dehopped_grid,
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


def test_composition_rejects_boolean_layout_and_invalid_symbol_allocations() -> None:
    with pytest.raises(TypeError, match="carrier_grid_size_rb must be an integer"):
        compose_pusch_intra_slot_hops(_grid(), **_options(carrier_grid_size_rb=True))
    with pytest.raises(ValueError, match="exceed max_output_elements"):
        compose_pusch_intra_slot_hops(_grid(), **_options(max_output_elements=1))
    with pytest.raises(ValueError, match="14-symbol slot"):
        compose_pusch_intra_slot_hops(_grid(), **_options(allocation_start_symbol=9))
    with pytest.raises(ValueError, match="two nonempty hops"):
        compose_pusch_intra_slot_hops(_grid(), **_options(first_hop_symbol_count=6))
    with pytest.raises(ValueError, match="fit within the configured BWP"):
        compose_pusch_intra_slot_hops(_grid(), **_options(second_hop_rb_start=23))


def test_verifier_rejects_unknown_schema_and_malformed_layout() -> None:
    grid = _grid()
    result = compose_pusch_intra_slot_hops(grid, **_options())
    with pytest.raises(ValueError, match="unsupported PUSCH hop-grid metadata schema"):
        verify_pusch_intra_slot_hops(
            grid, result.frequency_grid, {**result.metadata, "schema": "other"}
        )
    with pytest.raises(ValueError, match="complex64 or complex128"):
        verify_pusch_intra_slot_hops(grid, result.frequency_grid.real, result.metadata)
    with pytest.raises(TypeError, match="NumPy array"):
        verify_pusch_intra_slot_hops(grid, result.frequency_grid.tolist(), result.metadata)
    with pytest.raises(ValueError, match="malformed layout values"):
        verify_pusch_intra_slot_hops(
            grid, result.frequency_grid, {**result.metadata, "first_hop_symbol_count": True}
        )
    with pytest.raises(ValueError, match="shape differs"):
        verify_pusch_intra_slot_hops(grid, result.frequency_grid[..., :-12], result.metadata)
    with pytest.raises(ValueError, match="metadata does not replay"):
        verify_pusch_intra_slot_hops(
            grid, result.frequency_grid, {**result.metadata, "allocation_rb_count": 1}
        )
    altered_grid = result.frequency_grid.copy()
    altered_grid[0, 0, 0, 4, 0] = 2
    with pytest.raises(ValueError, match="grid does not replay"):
        verify_pusch_intra_slot_hops(grid, altered_grid, result.metadata)


def test_hop_verifier_accepts_exact_composition_replay() -> None:
    source = _grid()
    result = compose_pusch_intra_slot_hops(source, **_options())
    replay = verify_pusch_intra_slot_hops(source, result.frequency_grid, result.metadata)
    assert replay["status"] == "PASS_INPUT_SCOPED_REPLAY"
    assert replay["qualification_status"] == "NOT_QUALIFIED"


def _valid_dehop_inputs():
    allocation = _grid()
    composed = compose_pusch_intra_slot_hops(allocation, **_options())
    received = np.ones_like(composed.frequency_grid)
    return received, composed.metadata


@pytest.mark.parametrize(
    ("metadata_change", "message"),
    [
        ({"carrier_grid_shape": [1, 1]}, "malformed carrier-grid shape"),
        ({"complex_dtype": "complex128"}, "dtype differs"),
        ({"carrier_grid_start_rb": True}, "malformed layout values"),
    ],
)
def test_hop_verifier_rejects_malformed_replay_metadata(metadata_change, message) -> None:
    source = _grid()
    composed = compose_pusch_intra_slot_hops(source, **_options())
    metadata = {**composed.metadata, **metadata_change}
    with pytest.raises(ValueError, match=message):
        verify_pusch_intra_slot_hops(source, composed.frequency_grid, metadata)


@pytest.mark.parametrize(
    ("metadata_change", "message"),
    [
        ({"receiver_hopping_qualification": "PASS"}, "unqualified status"),
        ({"bwp_size_rb": 0}, "invalid carrier/BWP allocation"),
        ({"allocation_start_symbol": -1}, "invalid symbol layout"),
        ({"allocation_symbol_count": 1}, "invalid symbol layout"),
        ({"first_hop_symbol_count": 0}, "invalid symbol layout"),
        ({"hop_boundary_symbol": 8}, "invalid symbol layout"),
        ({"allocation_rb_count": 0}, "invalid allocation width"),
        ({"first_hop_start_crb": 7}, "fit within the declared BWP"),
        ({"second_hop_start_crb": 31}, "fit within the declared BWP"),
        ({"allocation_grid_shape": [1, 1, 1, 1, 24]}, "shapes disagree"),
        ({"complex_dtype": "complex128"}, "shapes disagree"),
    ],
)
def test_dehop_rejects_inconsistent_metadata(metadata_change, message) -> None:
    received, metadata = _valid_dehop_inputs()
    with pytest.raises(ValueError, match=message):
        dehop_pusch_intra_slot_received_grid(received, {**metadata, **metadata_change})


def test_dehop_rejects_invalid_arrays_and_resource_limits() -> None:
    received, metadata = _valid_dehop_inputs()
    with pytest.raises(TypeError, match="NumPy array"):
        dehop_pusch_intra_slot_received_grid(received.tolist(), metadata)
    with pytest.raises(ValueError, match="symbol, subcarrier.*axes"):
        dehop_pusch_intra_slot_received_grid(received[0], metadata)
    with pytest.raises(ValueError, match="14-symbol slot"):
        dehop_pusch_intra_slot_received_grid(received[:, :, :, :-1], metadata)
    with pytest.raises(ValueError, match="complex64 or complex128"):
        dehop_pusch_intra_slot_received_grid(received.real, metadata)
    nonfinite = received.copy()
    nonfinite[0, 0, 0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="only finite values"):
        dehop_pusch_intra_slot_received_grid(nonfinite, metadata)
    with pytest.raises(ValueError, match="unsupported PUSCH"):
        dehop_pusch_intra_slot_received_grid(received, {**metadata, "schema": "other"})
    with pytest.raises(ValueError, match="malformed layout values"):
        dehop_pusch_intra_slot_received_grid(received, {**metadata, "bwp_size_rb": True})
    with pytest.raises(ValueError, match="width differs"):
        dehop_pusch_intra_slot_received_grid(received[..., :-12], metadata)
    with pytest.raises(ValueError, match="positive integer"):
        dehop_pusch_intra_slot_received_grid(received, metadata, max_output_elements=True)
    with pytest.raises(ValueError, match="positive integer"):
        dehop_pusch_intra_slot_received_grid(received, metadata, max_output_elements=0)
    with pytest.raises(ValueError, match="exceed max_output_elements"):
        dehop_pusch_intra_slot_received_grid(received, metadata, max_output_elements=1)


def test_dehopped_replay_verifier_checks_grid_and_metadata() -> None:
    received, metadata = _valid_dehop_inputs()
    result = dehop_pusch_intra_slot_received_grid(received, metadata)
    with pytest.raises(TypeError, match="NumPy array and mapping"):
        verify_pusch_intra_slot_dehopped_grid(
            received, metadata, result.frequency_grid.tolist(), {}
        )
    with pytest.raises(ValueError, match="metadata does not replay"):
        verify_pusch_intra_slot_dehopped_grid(
            received, metadata, result.frequency_grid, {**result.metadata, "scope": "other"}
        )
    with pytest.raises(ValueError, match="grid does not replay"):
        verify_pusch_intra_slot_dehopped_grid(
            received, metadata, np.zeros_like(result.frequency_grid), result.metadata
        )
    replay = verify_pusch_intra_slot_dehopped_grid(
        received, metadata, result.frequency_grid, result.metadata
    )
    assert replay["status"] == "PASS_INPUT_SCOPED_REPLAY"
    assert replay["qualification_status"] == "NOT_QUALIFIED"
