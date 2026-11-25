"""Comb pilot least-squares estimates: input shape, spacing and SNR admission."""

from __future__ import annotations

import numpy as np
import pytest
from receivers.pilots import comb_ls_estimate


def _cfr() -> np.ndarray:
    return np.ones((1, 1, 16), dtype=np.complex128)


def test_comb_estimate_requires_finite_snr_and_a_valid_spacing() -> None:
    with pytest.raises(ValueError, match="SNR must be finite"):
        comb_ls_estimate(_cfr(), spacing=2, snr_db=float("nan"), seed=1)
    with pytest.raises(ValueError, match="Invalid CFR or pilot spacing"):
        comb_ls_estimate(np.ones((16,)), spacing=2, snr_db=10.0, seed=1)
    with pytest.raises(ValueError, match="Invalid CFR or pilot spacing"):
        comb_ls_estimate(_cfr(), spacing=0, snr_db=10.0, seed=1)
    with pytest.raises(ValueError, match="Invalid CFR or pilot spacing"):
        comb_ls_estimate(_cfr(), spacing=9, snr_db=10.0, seed=1)
