from __future__ import annotations

import numpy as np
import pytest


def cnn_api():
    pytest.importorskip("torch")
    from learning.real_cnn import fit_real_cnn
    from models.real_cnn import _real_imaginary_channels, load_real_cnn

    return fit_real_cnn, _real_imaginary_channels, load_real_cnn


def test_real_cnn_training_checkpoint_round_trip(tmp_path) -> None:
    fit_real_cnn, _, load_real_cnn = cnn_api()
    rng = np.random.default_rng(27)
    x = (rng.normal(size=(24, 2, 32)) + 1j * rng.normal(size=(24, 2, 32))).astype(np.complex64)
    y = np.tile(np.asarray([0, 1], dtype=np.int64), 12)
    x_validation = (rng.normal(size=(12, 2, 32)) + 1j * rng.normal(size=(12, 2, 32))).astype(
        np.complex64
    )
    y_validation = np.tile(np.asarray([0, 1], dtype=np.int64), 6)

    predictor, training = fit_real_cnn(
        x,
        y,
        x_validation,
        y_validation,
        seed=31,
        epochs=4,
        patience=2,
        batch_size=8,
    )
    checkpoint = tmp_path / "real-cnn.pt"
    predictor.save(checkpoint, threshold=0.61)
    restored = load_real_cnn(checkpoint)
    probability = restored.predict_proba(x_validation)

    assert training["epoch_count"] <= 4
    assert training["trainable_parameters"] > 0
    assert restored.threshold == pytest.approx(0.61)
    assert restored.include_phase is True
    assert probability.shape == (len(y_validation), 2)
    assert np.isfinite(probability).all()
    assert np.allclose(predictor.predict_proba(x_validation), probability)
    assert np.allclose(probability.sum(axis=1), 1.0)


def test_real_cnn_rejects_invalid_inputs_and_nonbinary_labels() -> None:
    fit_real_cnn, real_imaginary_channels, _ = cnn_api()
    with pytest.raises(ValueError, match="complex"):
        real_imaginary_channels(np.ones((4, 2, 16), dtype=np.float32))
    with pytest.raises(ValueError, match="binary"):
        fit_real_cnn(
            np.ones((4, 1, 16), dtype=np.complex64),
            np.asarray([0, 1, 2, 0]),
            np.ones((4, 1, 16), dtype=np.complex64),
            np.asarray([0, 1, 0, 1]),
            epochs=1,
        )


def test_real_cnn_rejects_unavailable_cuda_device() -> None:
    fit_real_cnn, _, _ = cnn_api()
    import torch

    if torch.cuda.is_available():
        pytest.skip("the test requires a CUDA-unavailable runtime")
    values = np.ones((4, 1, 16), dtype=np.complex64)
    labels = np.asarray([0, 1, 0, 1])
    with pytest.raises(RuntimeError, match="CUDA device requested but unavailable"):
        fit_real_cnn(values, labels, values, labels, epochs=1, device="cuda:0")
