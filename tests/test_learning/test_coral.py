from __future__ import annotations

import numpy as np
import pytest


def coral_api():
    pytest.importorskip("torch")
    from learning.coral import fit_coral_mlp, load_coral_checkpoint

    return fit_coral_mlp, load_coral_checkpoint


def test_coral_training_checkpoint_round_trip(tmp_path) -> None:
    fit_coral_mlp, load_coral_checkpoint = coral_api()
    rng = np.random.default_rng(17)
    x = rng.normal(size=(32, 10)).astype(np.float32)
    y = np.tile(np.array([0, 1], dtype=np.int64), 16)
    domains = np.repeat(np.array(["source-a", "source-b"]), 16)
    x_val = rng.normal(size=(12, 10)).astype(np.float32)
    y_val = np.tile(np.array([0, 1], dtype=np.int64), 6)

    predictor, training = fit_coral_mlp(
        x,
        y,
        domains,
        x_val,
        y_val,
        seed=9,
        epochs=3,
        batch_per_domain=4,
    )
    checkpoint = tmp_path / "coral.pt"
    predictor.save(checkpoint, threshold=0.63, include_phase=True)
    restored = load_coral_checkpoint(checkpoint)

    assert training["epoch_count"] <= 3
    assert training["train_domains"] == ["source-a", "source-b"]
    assert restored.threshold == pytest.approx(0.63)
    assert restored.include_phase is True
    assert np.allclose(predictor.predict_proba(x_val), restored.predict_proba(x_val))
    assert np.allclose(restored.predict_proba(x_val).sum(axis=1), 1.0)


def test_coral_rejects_single_domain_and_negative_penalty() -> None:
    fit_coral_mlp, _ = coral_api()
    x = np.arange(24, dtype=np.float32).reshape(8, 3)
    y = np.tile(np.array([0, 1]), 4)
    x_val = x[:4]
    y_val = y[:4]
    one_domain = np.repeat("source", len(y))

    with pytest.raises(ValueError, match=">=2 training domains"):
        fit_coral_mlp(x, y, one_domain, x_val, y_val, epochs=1)
    with pytest.raises(ValueError, match="penalty"):
        fit_coral_mlp(x, y, np.tile(["a", "b"], 4), x_val, y_val, penalty=-0.1)
