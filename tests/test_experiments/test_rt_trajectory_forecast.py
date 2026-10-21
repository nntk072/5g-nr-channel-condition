"""Tests for fixed receiver-only RT trajectory forecast baselines."""

import numpy as np
import pytest
from datasets.nr import NRChannelRecord, NRObservation, NRTrajectory
from experiments.rt_trajectory_forecast import (
    evaluate_trajectory_forecasts,
    summarize_forecast_rows,
)
from receivers.nr import NRChannelEstimate, NRPUSCHConfig


def _record(sample_id: str, observed_level: float, target_level: float) -> NRChannelRecord:
    config = NRPUSCHConfig(n_size_bwp=1, subcarrier_spacing_khz=30)
    observed = np.full((1, 1, 1, 1, 1, 2, 3), observed_level, dtype=np.complex64)
    estimate = NRChannelEstimate(
        h_hat=observed,
        error_variance=np.ones((1, 1, 1, 1, 1, 2, 3), dtype=np.float32),
        pilot_mask=np.ones((1, 1, 2, 3), dtype=np.bool_),
        metadata={"receiver": "fixture"},
    )
    observation = NRObservation(
        sample_id=sample_id,
        received_grid=np.ones((1, 1, 1, 2, 3), dtype=np.complex64),
        channel_estimate=estimate,
        fft_bin_indices=(0, 1, 2),
        receiver_config=config,
    )
    return NRChannelRecord(
        observation=observation,
        targets={"channel_cfr": np.full(3, target_level, dtype=np.complex64)},
        target_masks={"channel_cfr": np.ones(3, dtype=np.bool_)},
        provenance={"fixture": "forecast-test"},
    )


def test_persistence_and_timestamp_scaled_linear_forecasts_use_only_past_estimates() -> None:
    trajectory = NRTrajectory(
        trajectory_id="forecast-fixture",
        sample_times_s=(0.0, 1.0, 3.0),
        records=(
            _record("t0", 1.0, 1.0),
            _record("t1", 3.0, 3.0),
            _record("t3", 7.0, 7.0),
        ),
    )

    rows = evaluate_trajectory_forecasts(trajectory, split="test")
    summary = summarize_forecast_rows(rows)

    assert summary["persistence"]["transition_count"] == 2
    assert summary["linear_complex_extrapolation"]["transition_count"] == 1
    linear = next(row for row in rows if row["method"] == "linear_complex_extrapolation")
    persistence = next(
        row for row in rows if row["method"] == "persistence" and row["sample_id"] == "t3"
    )
    assert linear["horizon_s"] == pytest.approx(2.0)
    assert linear["nmse_linear"] == pytest.approx(0.0, abs=1e-14)
    assert persistence["nmse_linear"] > linear["nmse_linear"]
    assert all(row["split"] == "test" for row in rows)


def test_forecast_baselines_reject_unrecognized_split_or_single_snapshot() -> None:
    with pytest.raises(ValueError, match="split"):
        evaluate_trajectory_forecasts(
            NRTrajectory("one", (0.0, 1.0), (_record("a", 1.0, 1.0), _record("b", 1.0, 1.0))),
            split="development",
        )
