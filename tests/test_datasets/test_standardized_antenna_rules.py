"""Antenna-array and receive-correlation admission rules for standardized TDL/CDL data."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pytest
from datasets.standardized import StandardizedFamily, validate_standardized_channel_admission
from tests.test_datasets.test_standardized import _metadata


def _h_path(rx_antennas: int) -> np.ndarray:
    return np.ones((1, rx_antennas, 1, 1, 2, 2), dtype=np.complex64) * 0.1


def _two_antenna_metadata(**changes: Any) -> dict[str, Any]:
    metadata = copy.deepcopy(_metadata(model="A"))
    metadata["rx_antenna_count"] = 2
    metadata["rx_corr_mat_real"] = [[1.0, 0.0], [0.0, 1.0]]
    metadata["rx_corr_mat_imag"] = [[0.0, 0.0], [0.0, 0.0]]
    metadata.update(changes)
    return metadata


def _admit(
    metadata: dict[str, Any], *, rx_antennas: int = 2, family=StandardizedFamily.TDL
) -> None:
    validate_standardized_channel_admission(family, "A", _h_path(rx_antennas), metadata)


def test_identity_receive_correlation_is_accepted() -> None:
    _admit(_two_antenna_metadata())


@pytest.mark.parametrize(
    ("change", "message"),
    [
        pytest.param(
            {"rx_corr_mat_real": None, "rx_corr_mat_imag": [[0.0]]},
            "real/imag metadata is incomplete",
            id="imag-only",
        ),
        pytest.param(
            {"rx_corr_mat_imag": None}, "real/imag metadata is incomplete", id="real-only"
        ),
        pytest.param(
            {"rx_corr_mat_real": [[1.0]], "rx_corr_mat_imag": [[0.0]]},
            "shape or values are invalid",
            id="wrong-shape",
        ),
        pytest.param(
            {"rx_corr_mat_real": [[1.0, float("nan")], [0.0, 1.0]]},
            "shape or values are invalid",
            id="non-finite",
        ),
        pytest.param(
            {"rx_corr_mat_real": [[1.0, 0.5], [0.0, 1.0]]}, "must be Hermitian", id="not-hermitian"
        ),
        pytest.param(
            {"rx_corr_mat_real": [[0.5, 0.0], [0.0, 1.0]]}, "must have unit diagonal", id="diagonal"
        ),
        pytest.param(
            {"rx_corr_mat_real": [[1.0, 1.0], [1.0, 1.0]]},
            "must be positive definite",
            id="singular",
        ),
    ],
)
def test_receive_correlation_matrix_rejections(change: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _admit(_two_antenna_metadata(**change))


def test_tdl_receive_correlation_provenance_is_required() -> None:
    metadata = _two_antenna_metadata()
    del metadata["rx_corr_mat_real"]
    del metadata["rx_corr_mat_imag"]
    with pytest.raises(ValueError, match="receive-correlation provenance is required"):
        _admit(metadata)


def _cdl_metadata(**changes: Any) -> dict[str, Any]:
    metadata = copy.deepcopy(_metadata(model="A"))
    metadata["direction"] = "downlink"
    metadata.update(changes)
    return metadata


def _array(**changes: Any) -> dict[str, Any]:
    array = {
        "num_rows_per_panel": 1,
        "num_cols_per_panel": 1,
        "num_rows": 1,
        "num_cols": 1,
        "polarization": "single",
        "polarization_type": "V",
        "antenna_pattern": "omni",
    }
    array.update(changes)
    return array


@pytest.mark.parametrize(
    ("change", "message"),
    [
        pytest.param(
            {"direction": "sideways"}, "CDL direction must be uplink or downlink", id="direction"
        ),
        pytest.param({"ut_array": None}, "CDL metadata must declare ut_array", id="array-missing"),
        pytest.param(
            {"ut_array": _array(num_rows=0)}, "invalid panel dimensions", id="panel-dimension"
        ),
        pytest.param(
            {"ut_array": _array(polarization="triple")}, "invalid polarization", id="polarization"
        ),
        pytest.param(
            {"ut_array": _array(polarization_type="VH")},
            "invalid polarization type",
            id="polarization-type",
        ),
        pytest.param(
            {"ut_array": _array(antenna_pattern="isotropic")},
            "invalid antenna pattern",
            id="pattern",
        ),
        pytest.param(
            {"ut_array": _array(element_vertical_spacing=-1.0)},
            "invalid antenna spacing",
            id="spacing",
        ),
        pytest.param(
            {"ut_array": _array(num_rows=2)}, "receive array configuration disagrees", id="rx-count"
        ),
        pytest.param(
            {"bs_array": _array(num_cols=2)},
            "transmit array configuration disagrees",
            id="tx-count",
        ),
    ],
)
def test_cdl_array_and_direction_rejections(change: dict[str, Any], message: str) -> None:
    metadata = _cdl_metadata()
    metadata.update(change)
    if change.get("ut_array") is None and "ut_array" in change:
        del metadata["ut_array"]
    with pytest.raises(ValueError, match=message):
        _admit(metadata, rx_antennas=1, family=StandardizedFamily.CDL)


def test_cdl_uplink_swaps_the_receive_and_transmit_arrays() -> None:
    metadata = _cdl_metadata(direction="uplink")
    metadata["bs_array"] = _array(num_rows=2)
    with pytest.raises(ValueError, match="receive array configuration disagrees"):
        _admit(metadata, rx_antennas=1, family=StandardizedFamily.CDL)
