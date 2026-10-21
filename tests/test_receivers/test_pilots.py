import numpy as np
import pytest
from receivers.pilots import comb_ls_estimate


def test_flat_channel_is_exact_when_snr_very_high():
    h = np.ones((3, 2, 64), dtype=np.complex64) * (1 + 0.4j)
    observed = comb_ls_estimate(h, spacing=4, snr_db=120, seed=7)
    assert np.allclose(h, observed, atol=1e-5)


def test_noise_changes_observation():
    h = np.ones((3, 2, 64), dtype=np.complex64)
    observed = comb_ls_estimate(h, spacing=4, snr_db=4, seed=7)
    assert np.mean(np.abs(observed - h)) > 0.01
    assert np.all(np.isfinite(observed))


def test_pilot_observation_seed_is_reproducible():
    h = np.ones((3, 2, 32), dtype=np.complex64)
    assert np.array_equal(comb_ls_estimate(h, seed=1), comb_ls_estimate(h, seed=1))
    assert not np.array_equal(comb_ls_estimate(h, seed=1), comb_ls_estimate(h, seed=2))


def test_invalid_spacing():
    with pytest.raises(ValueError):
        comb_ls_estimate(np.ones((2, 1, 16), dtype=np.complex64), spacing=0)
