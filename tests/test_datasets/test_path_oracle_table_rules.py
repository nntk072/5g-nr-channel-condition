"""Path-oracle table contracts: ragged layout, dtypes, identities and geometry bounds."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from datasets.path_oracle import PathOracleTable, PathSolveStatus, reconstruct_path_oracle_cfr


def _table(**overrides: Any) -> PathOracleTable:
    kwargs: dict[str, Any] = {
        "link_ids": ("gNB-1-to-UE-1",),
        "frame_indices": np.asarray([0]),
        "rx_port_indices": np.asarray([0]),
        "tx_port_indices": np.asarray([0]),
        "path_offsets": np.asarray([0, 2]),
        "solve_status": (PathSolveStatus.SOLVED,),
        "delay_s": np.asarray([1e-9, 2e-9]),
        "complex_gain": np.asarray([1.0 + 0j, 0.5 + 0j]),
        "interaction_order": np.zeros(2, dtype=np.int64),
    }
    kwargs.update(overrides)
    return PathOracleTable(**kwargs)


def test_valid_table_reconstructs_a_finite_cfr() -> None:
    table = _table()
    frequencies = np.asarray([-1e6, 0.0, 1e6])
    cfr = reconstruct_path_oracle_cfr(table, carrier_hz=3.5e9, frequency_offset_hz=frequencies)
    assert np.isfinite(cfr).all()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        pytest.param({"link_ids": ("",)}, "non-empty canonical link IDs", id="link-empty"),
        pytest.param(
            {"delay_s": np.asarray([1e-9, np.nan])}, "must contain finite values", id="delay-nan"
        ),
        pytest.param(
            {"delay_s": np.asarray([1e-9])},
            "must match the ragged path_offsets total",
            id="delay-short",
        ),
        pytest.param(
            {"delay_s": np.asarray([-1e-9, 2e-9])}, "must be nonnegative", id="delay-negative"
        ),
        pytest.param(
            {"complex_gain": np.asarray([np.inf + 0j, 0.5 + 0j])},
            "complex_gain must contain finite values",
            id="gain-inf",
        ),
        pytest.param(
            {"interaction_order": np.asarray([-1, 0])}, "nonnegative", id="order-negative"
        ),
        pytest.param({"solve_status": ("solved",)}, "PathSolveStatus", id="status-type"),
        pytest.param({"rx_port_indices": np.asarray([-1])}, "nonnegative", id="port-negative"),
        pytest.param({"path_offsets": np.asarray([0.0, 2.0])}, "integer dtype", id="offset-dtype"),
        pytest.param(
            {"path_offsets": np.asarray([2, 0])},
            "one leading zero and one end per group",
            id="offset-order",
        ),
    ],
)
def test_path_table_rejects_malformed_fields(overrides, message) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _table(**overrides)


def test_path_table_requires_a_reconstructable_frequency_grid() -> None:
    table = _table()
    with pytest.raises(TypeError, match="table must be a PathOracleTable"):
        reconstruct_path_oracle_cfr(
            "not-a-table", carrier_hz=3.5e9, frequency_offset_hz=np.zeros(3)
        )
    with pytest.raises(ValueError, match="carrier_hz must be finite and positive"):
        reconstruct_path_oracle_cfr(table, carrier_hz=0.0, frequency_offset_hz=np.zeros(3))
    with pytest.raises(TypeError, match="carrier_hz must be a real numeric scalar"):
        reconstruct_path_oracle_cfr(table, carrier_hz="3.5e9", frequency_offset_hz=np.zeros(3))
    with pytest.raises(ValueError, match="strictly increasing"):
        reconstruct_path_oracle_cfr(
            table, carrier_hz=3.5e9, frequency_offset_hz=np.asarray([0.0, 0.0])
        )
    with pytest.raises(ValueError, match="non-empty finite vector"):
        reconstruct_path_oracle_cfr(table, carrier_hz=3.5e9, frequency_offset_hz=np.asarray([]))
    with pytest.raises(TypeError, match="real numeric vector"):
        reconstruct_path_oracle_cfr(table, carrier_hz=3.5e9, frequency_offset_hz=np.zeros((2, 2)))


def test_include_mask_must_be_boolean_with_one_entry_per_path() -> None:
    table = _table()
    with pytest.raises(ValueError, match="strictly increasing"):
        reconstruct_path_oracle_cfr(
            table,
            carrier_hz=3.5e9,
            frequency_offset_hz=np.zeros(3),
            include_path_mask=np.asarray([True]),
        )
    with pytest.raises(ValueError, match="boolean with one entry per path"):
        reconstruct_path_oracle_cfr(
            table,
            carrier_hz=3.5e9,
            frequency_offset_hz=np.asarray([-1.0, 0.0, 1.0]),
            include_path_mask=np.asarray([True]),
        )
