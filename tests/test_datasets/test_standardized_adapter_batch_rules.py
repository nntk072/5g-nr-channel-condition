"""Standardized adapter batches: metadata batch size must agree with every per-draw array before records are split."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from datasets.standardized import StandardizedFamily, standardized_run_from_channel


def _channel(batch_size=2, *, h=2, tau=2, cfr=2, metadata=None):
    values = {
        "profile_model": "A",
        "batch_size": batch_size,
    }
    values.update(metadata or {})
    return SimpleNamespace(
        metadata=values,
        h_path=np.ones((h, 1, 1, 3), dtype=np.complex64),
        tau_s=np.zeros((tau, 3), dtype=np.float64),
        channel_frequency_response=np.ones((cfr, 1, 8), dtype=np.complex64),
    )


def test_path_and_delay_batches_must_match_the_declared_batch_size() -> None:
    with pytest.raises(ValueError, match="batch metadata and channel arrays disagree"):
        standardized_run_from_channel(_channel(h=3), family=StandardizedFamily.TDL)
    with pytest.raises(ValueError, match="batch metadata and channel arrays disagree"):
        standardized_run_from_channel(_channel(tau=3), family=StandardizedFamily.TDL)


def test_frequency_response_batch_must_match_the_declared_batch_size() -> None:
    with pytest.raises(ValueError, match="batch metadata and CFR array disagree"):
        standardized_run_from_channel(_channel(cfr=3), family=StandardizedFamily.TDL)


def test_non_positive_batch_size_is_refused() -> None:
    with pytest.raises(ValueError, match="batch metadata and channel arrays disagree"):
        standardized_run_from_channel(_channel(batch_size=0), family=StandardizedFamily.TDL)
