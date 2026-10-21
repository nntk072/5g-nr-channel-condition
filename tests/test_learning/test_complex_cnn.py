from __future__ import annotations

import numpy as np
import pytest


def complex_cnn_api():
    pytest.importorskip("torch")
    from learning.complex_cnn import fit_complex_cnn
    from models.complex_cnn import _complex_parts, build_complex_cnn, load_complex_cnn

    return fit_complex_cnn, _complex_parts, build_complex_cnn, load_complex_cnn


def test_complex_cnn_is_global_phase_invariant() -> None:
    _, _, build_complex_cnn, _ = complex_cnn_api()
    import torch

    torch.manual_seed(3)
    model = build_complex_cnn(2).eval()
    real = torch.randn(5, 2, 32)
    imag = torch.randn(5, 2, 32)
    phase = 0.73
    rotated_real = real * np.cos(phase) - imag * np.sin(phase)
    rotated_imag = real * np.sin(phase) + imag * np.cos(phase)

    with torch.no_grad():
        baseline = model(real, imag)
        rotated = model(rotated_real, rotated_imag)

    assert torch.allclose(baseline, rotated, atol=2e-5, rtol=2e-5)


def test_complex_cnn_training_and_checkpoint_round_trip(tmp_path) -> None:
    fit_complex_cnn, _, _, load_complex_cnn = complex_cnn_api()
    rng = np.random.default_rng(37)
    train = (rng.normal(size=(24, 2, 32)) + 1j * rng.normal(size=(24, 2, 32))).astype(np.complex64)
    labels = np.tile(np.asarray([0, 1], dtype=np.int64), 12)
    validation = (rng.normal(size=(12, 2, 32)) + 1j * rng.normal(size=(12, 2, 32))).astype(
        np.complex64
    )
    validation_labels = np.tile(np.asarray([0, 1], dtype=np.int64), 6)

    predictor, diagnostics = fit_complex_cnn(
        train, labels, validation, validation_labels, seed=41, epochs=3, patience=2, batch_size=8
    )
    checkpoint = tmp_path / "complex-cnn.pt"
    predictor.save(checkpoint, threshold=0.58)
    restored = load_complex_cnn(checkpoint)
    probabilities = restored.predict_proba(validation)

    assert diagnostics["epoch_count"] <= 3
    assert diagnostics["trainable_parameters"] > 0
    assert restored.threshold == pytest.approx(0.58)
    assert probabilities.shape == (len(validation_labels), 2)
    assert np.isfinite(probabilities).all()
    assert np.allclose(predictor.predict_proba(validation), probabilities)


def test_complex_cnn_rejects_real_or_nonfinite_inputs() -> None:
    _, complex_parts, _, _ = complex_cnn_api()
    with pytest.raises(ValueError, match="complex"):
        complex_parts(np.ones((3, 1, 16), dtype=np.float32))
    values = np.ones((3, 1, 16), dtype=np.complex64)
    values[0, 0, 0] = complex(np.nan, 0)
    with pytest.raises(ValueError, match="finite"):
        complex_parts(values)
