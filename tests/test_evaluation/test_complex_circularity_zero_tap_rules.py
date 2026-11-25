"""Complex circularity claims: every tap must carry non-zero power before normalization."""

from __future__ import annotations

import numpy as np
import pytest
from evaluation.covariance.complex_circularity import evaluate_complex_circularity_family


def test_all_zero_tap_is_refused_before_normalization() -> None:
    draws = np.zeros((8, 2), dtype=np.complex128)
    draws[:, 1] = 1.0 + 1.0j
    with pytest.raises(ValueError, match="each tap must have positive finite power"):
        evaluate_complex_circularity_family(draws)


def test_nonzero_taps_are_evaluated() -> None:
    rng = np.random.default_rng(3)
    draws = rng.standard_normal((64, 2)) + 1j * rng.standard_normal((64, 2))
    result = evaluate_complex_circularity_family(draws)
    assert isinstance(result, dict)
