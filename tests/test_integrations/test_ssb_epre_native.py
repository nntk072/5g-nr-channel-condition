"""TS 38.213 relative EPRE profile checks for an SS/PBCH block."""

from __future__ import annotations

import numpy as np
import pytest
from integrations.sionna import (
    map_nr_ssb_resource_grid,
    scale_nr_ssb_resource_grid_epre,
    scale_nr_ssb_resource_grid_power,
)


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


@pytest.mark.parametrize("ratio_db", [0.0, 3.0])
@pytest.mark.parametrize("block_power_dbm", [-60.0, -20.0, 0.0, 50.0])
def test_configured_block_power_sets_sss_epre_and_relative_pss_epre(
    block_power_dbm: float, ratio_db: float
) -> None:
    source = _ssb_grid()
    result = scale_nr_ssb_resource_grid_power(
        source,
        ss_pbch_block_power_dbm=block_power_dbm,
        pss_to_sss_epre_db=ratio_db,
    )
    target_power_w = 10.0 ** ((block_power_dbm - 30.0) / 10.0)

    assert np.mean(np.abs(result[2, 56:183]) ** 2) == pytest.approx(target_power_w, rel=1e-13)
    assert np.mean(np.abs(result[0, 56:183]) ** 2) == pytest.approx(
        target_power_w * 10.0 ** (ratio_db / 10.0), rel=1e-13
    )
    for symbol in (1, 2, 3):
        assert np.mean(np.abs(result[symbol]) ** 2) == pytest.approx(
            target_power_w * np.count_nonzero(source[symbol]) / 240, rel=1e-13
        )
    assert not result.flags.writeable


def test_power_scaled_grid_has_expected_ofdm_sample_power_with_cp() -> None:
    from integrations.sionna import nr_ssb_ofdm_modulate

    grid = scale_nr_ssb_resource_grid_power(
        _ssb_grid(), ss_pbch_block_power_dbm=-30.0, pss_to_sss_epre_db=3.0
    )
    cp = (18, 18, 18, 18)
    waveform = nr_ssb_ofdm_modulate(grid, fft_size=256, cyclic_prefix_lengths=cp)
    full_grid = np.zeros((4, 256), dtype=np.complex128)
    full_grid[:, 8:248] = grid
    useful = np.fft.ifft(np.fft.ifftshift(full_grid, axes=-1), axis=-1, norm="ortho")
    expected_energy = sum(
        np.sum(np.abs(symbol[-prefix:]) ** 2) + np.sum(np.abs(symbol) ** 2)
        for symbol, prefix in zip(useful, cp, strict=True)
    )
    assert np.mean(np.abs(waveform) ** 2) == pytest.approx(
        expected_energy / (4 * 256 + sum(cp)), rel=1e-13
    )


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


@pytest.mark.parametrize(
    ("power_dbm", "ratio", "error", "message"),
    [
        (True, 0.0, TypeError, "real number"),
        (float("nan"), 0.0, ValueError, "finite"),
        (-60.1, 0.0, ValueError, "\\[-60, 50\\]"),
        (50.1, 0.0, ValueError, "\\[-60, 50\\]"),
        (-30.0, 1.0, ValueError, "0 or 3 dB"),
    ],
)
def test_ssb_power_scaling_rejects_invalid_configuration(
    power_dbm: object, ratio: object, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        scale_nr_ssb_resource_grid_power(  # type: ignore[arg-type]
            _ssb_grid(),
            ss_pbch_block_power_dbm=power_dbm,
            pss_to_sss_epre_db=ratio,
        )


@pytest.mark.parametrize(
    "ssb_grid",
    [
        np.ones((4, 240), dtype=np.complex128),
        np.zeros((4, 240), dtype=np.complex128),
    ],
)
def test_ssb_power_scaling_requires_nonzero_sss_energy(ssb_grid: np.ndarray) -> None:
    if np.any(ssb_grid[2, 56:183]):
        ssb_grid[2, 56:183] = 0
    with pytest.raises(ValueError, match="positive finite mean energy"):
        scale_nr_ssb_resource_grid_power(ssb_grid, ss_pbch_block_power_dbm=-30.0)


def test_ssb_power_scaling_rejects_grid_that_would_overflow_after_scaling() -> None:
    grid = _ssb_grid().copy()
    grid[2, 56:183] = 1e-155
    grid[1, 10] = 1e200
    with pytest.raises(ValueError, match="nonfinite resource elements"):
        scale_nr_ssb_resource_grid_power(grid, ss_pbch_block_power_dbm=50.0)
