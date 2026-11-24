"""CORAL training paths: checkpoint format refusal, domains too small for a covariance, and early stopping."""

from __future__ import annotations

import numpy as np
import pytest


def _api():
    pytest.importorskip("torch")
    from learning.coral import fit_coral_mlp, load_coral_checkpoint

    return fit_coral_mlp, load_coral_checkpoint


def test_checkpoint_with_a_foreign_format_is_refused(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    _, load_coral_checkpoint = _api()
    path = tmp_path / "foreign.pt"
    torch.save({"format": "other.v9", "n_features": 3}, path)
    with pytest.raises(ValueError, match="Invalid checkpoint format"):
        load_coral_checkpoint(path)


def test_penalty_skips_domains_too_small_for_a_covariance_and_stops_early() -> None:
    fit_coral_mlp, _ = _api()
    rng = np.random.default_rng(5)
    x = rng.normal(size=(40, 4)).astype(np.float32)
    y = np.tile(np.array([0, 1], dtype=np.int64), 20)
    # The third domain has one sample, so it contributes no covariance term.
    domains = np.array(["a"] * 19 + ["b"] * 20 + ["c"] * 1)
    x_val = rng.normal(size=(8, 4)).astype(np.float32)
    y_val = np.tile(np.array([0, 1], dtype=np.int64), 4)
    predictor, training = fit_coral_mlp(
        x,
        y,
        domains,
        x_val,
        y_val,
        seed=3,
        penalty=0.5,
        epochs=60,
        lr=0.0,
        batch_per_domain=4,
        capacity="small",
    )
    # With a zero learning rate validation never improves, so patience ends training early.
    assert training["epoch_count"] < 60
    assert predictor.predict_proba(x_val).shape == (8, 2)
