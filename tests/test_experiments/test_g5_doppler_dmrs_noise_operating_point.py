from __future__ import annotations

import pytest
from experiments.scan_g5_doppler_dmrs_noise_operating_point import _validate_operating_point


@pytest.mark.parametrize(
    ("seeds", "blocks", "variance", "error", "message"),
    [
        ((), 1, 0.001, ValueError, "channel_seeds"),
        ((1, 1), 1, 0.001, ValueError, "channel_seeds"),
        ((True,), 1, 0.001, ValueError, "channel_seeds"),
        ((1,), 0, 0.001, ValueError, "positive"),
        ((1,), True, 0.001, TypeError, "integer"),
        ((1,), 1, 0.0, ValueError, "waveform_noise_variance"),
        ((1,), 1, float("nan"), ValueError, "waveform_noise_variance"),
        ((1,), 1, True, ValueError, "waveform_noise_variance"),
    ],
)
def test_noise_operating_point_rejects_invalid_design(seeds, blocks, variance, error, message):
    with pytest.raises(error, match=message):
        _validate_operating_point(seeds, blocks, variance)
