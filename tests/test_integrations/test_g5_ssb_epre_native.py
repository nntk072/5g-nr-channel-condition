"""TS 38.213 relative EPRE profile checks for an SS/PBCH block."""

from __future__ import annotations

import numpy as np
import pytest
from integrations.sionna import map_nr_ssb_resource_grid, scale_nr_ssb_resource_grid_epre


def _ssb_grid() -> np.ndarray:
    return map_nr_ssb_resource_grid(
        17,
        2,
        np.ones(432, dtype=np.complex128),
        ssb_index_in_half_frame=0,
        half_frame_index=0,
        max_candidate_blocks_per_half_frame=4,
    ).grid


def test_zero_db_pss_to_sss_profile_preserves_the_input_grid() -> None:
    source = _ssb_grid()
    result = scale_nr_ssb_resource_grid_epre(source, pss_to_sss_epre_db=0.0)

    np.testing.assert_array_equal(result, source)
    assert result is not source
    assert not result.flags.writeable


def test_three_db_profile_scales_only_pss_and_preserves_equal_sss_pbch_epre() -> None:
    source = _ssb_grid()
    result = scale_nr_ssb_resource_grid_epre(source, pss_to_sss_epre_db=3.0)
    pss_power_ratio = np.mean(np.abs(result[0, 56:183]) ** 2) / np.mean(
        np.abs(source[0, 56:183]) ** 2
    )
    non_pss = np.ones(source.shape, dtype=bool)
    non_pss[0, 56:183] = False

    assert pss_power_ratio == pytest.approx(10.0**0.3, rel=1e-14)
    np.testing.assert_array_equal(result[non_pss], source[non_pss])
    assert np.mean(np.abs(source[0, 56:183]) ** 2) == pytest.approx(
        np.mean(np.abs(source[2, 56:183]) ** 2)
    )
    assert not result.flags.writeable


@pytest.mark.parametrize(
    ("grid", "ratio", "error", "message"),
    [
        (np.zeros((4, 239), dtype=np.complex128), 0.0, ValueError, "shape"),
        (np.zeros((4, 240), dtype=bool), 0.0, TypeError, "numeric"),
        (np.full((4, 240), np.inf, dtype=np.complex128), 3.0, ValueError, "finite"),
        (np.zeros((4, 240), dtype=np.complex128), True, TypeError, "real number"),
        (np.zeros((4, 240), dtype=np.complex128), 1.0, ValueError, "0 or 3 dB"),
    ],
)
def test_ssb_epre_profile_rejects_invalid_inputs(
    grid: np.ndarray, ratio: object, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        scale_nr_ssb_resource_grid_epre(grid, pss_to_sss_epre_db=ratio)  # type: ignore[arg-type]
