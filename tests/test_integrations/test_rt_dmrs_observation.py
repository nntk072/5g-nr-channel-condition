"""Pure input and frequency-grid tests for the RT-to-DMRS observation adapter."""

from __future__ import annotations

import numpy as np
import pytest
from integrations.sionna.rt_dmrs import resample_centered_cfr


def test_centered_cfr_resampling_preserves_overlapping_subcarriers():
    source_frequencies = (np.arange(128) - 64) * 30_000.0
    channels = np.stack(
        [
            (1.0 + source_frequencies / 1e6) + 1j * (0.5 - source_frequencies / 2e6),
            (-0.25 + source_frequencies / 3e6) + 1j * (1.0 + source_frequencies / 4e6),
        ]
    )[:, None, :].astype(np.complex64)

    actual = resample_centered_cfr(
        channels,
        source_spacing_hz=30_000.0,
        target_subcarriers=72,
        target_spacing_hz=30_000.0,
    )

    np.testing.assert_array_equal(actual, channels[:, 0, 28:100])


@pytest.mark.parametrize(
    ("channels", "overrides", "message"),
    [
        (np.ones((2, 2, 128), dtype=np.complex64), {}, "samples, 1"),
        (np.ones((2, 1, 128), dtype=np.float64), {}, "complex"),
        (np.ones((2, 1, 128), dtype=np.complex64), {"source_spacing_hz": 0.0}, "source_spacing_hz"),
        (
            np.ones((2, 1, 128), dtype=np.complex64),
            {"target_subcarriers": 128, "target_spacing_hz": 60_000.0},
            "inside the source CFR bandwidth",
        ),
    ],
)
def test_centered_cfr_resampling_rejects_invalid_grids(channels, overrides, message):
    arguments = {
        "source_spacing_hz": 30_000.0,
        "target_subcarriers": 72,
        "target_spacing_hz": 30_000.0,
    }
    arguments.update(overrides)
    with pytest.raises(ValueError, match=message):
        resample_centered_cfr(channels, **arguments)
