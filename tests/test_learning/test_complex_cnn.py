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
        train,
        labels,
        validation,
        validation_labels,
        seed=41,
        epochs=3,
        patience=2,
        batch_size=8,
        capacity="small",
    )
    checkpoint = tmp_path / "complex-cnn.pt"
    predictor.save(checkpoint, threshold=0.58)
    restored = load_complex_cnn(checkpoint)
    probabilities = restored.predict_proba(validation)

    assert diagnostics["epoch_count"] <= 3
    assert diagnostics["trainable_parameters"] > 0
    assert restored.threshold == pytest.approx(0.58)
    assert restored.capacity == "small"
    assert restored.artifact_contract_status == "external_input_contract_required"
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


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("epochs", 0, "epochs must be a positive integer"),
        ("epochs", True, "epochs must be a positive integer"),
        ("patience", 0, "patience must be a positive integer"),
        ("batch_size", 0, "batch_size must be a positive integer"),
        ("learning_rate", float("nan"), "learning_rate must be finite and positive"),
        ("learning_rate", 0.0, "learning_rate must be finite and positive"),
        ("device", "tpu", "device must be cpu or a CUDA device"),
    ],
)
def test_complex_cnn_rejects_invalid_training_options(option, value, message) -> None:
    fit_complex_cnn, *_ = complex_cnn_api()
    train = np.ones((4, 1, 8), dtype=np.complex64)
    labels = np.asarray([0, 1, 0, 1])
    options = {"epochs": 1, "patience": 1, "batch_size": 4}
    options[option] = value

    with pytest.raises(ValueError, match=message):
        fit_complex_cnn(train, labels, train, labels, **options)


@pytest.mark.parametrize(
    ("train_labels", "validation_labels", "validation_shape", "message"),
    [
        (np.asarray([0, 1, 2, 1]), np.asarray([0, 1, 0, 1]), (4, 1, 8), "binary values"),
        (np.asarray([0, 1, 0]), np.asarray([0, 1, 0, 1]), (4, 1, 8), "match their observation"),
        (np.asarray([0, 0, 0, 0]), np.asarray([0, 1, 0, 1]), (4, 1, 8), "both classes"),
        (np.asarray([0, 1, 0, 1]), np.asarray([0, 0, 0, 0]), (4, 1, 8), "both classes"),
        (np.asarray([0, 1, 0, 1]), np.asarray([0, 1, 0, 1]), (4, 2, 8), "shapes differ"),
    ],
)
def test_complex_cnn_rejects_misaligned_training_and_validation_data(
    train_labels, validation_labels, validation_shape, message
) -> None:
    fit_complex_cnn, *_ = complex_cnn_api()
    train = np.ones((4, 1, 8), dtype=np.complex64)
    validation = np.ones(validation_shape, dtype=np.complex64)

    with pytest.raises(ValueError, match=message):
        fit_complex_cnn(train, train_labels, validation, validation_labels, epochs=1)


def test_complex_cnn_stops_after_patience_without_validation_improvement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    import learning.complex_cnn as trainer
    from torch import nn

    class ConstantModel(nn.Module):
        def __init__(self, _links: int, _capacity: str) -> None:
            super().__init__()
            self.logits = nn.Parameter(torch.tensor([0.0, 0.0]))

        def forward(self, real, _imag):
            return self.logits.expand(len(real), -1)

    monkeypatch.setattr(trainer, "build_complex_cnn", ConstantModel)
    train = np.ones((4, 1, 8), dtype=np.complex64)
    labels = np.asarray([0, 1, 0, 1])

    _predictor, diagnostics = trainer.fit_complex_cnn(
        train, labels, train, labels, epochs=5, patience=1, batch_size=4
    )

    assert diagnostics["epoch_count"] == 2


def test_complex_cnn_requires_finite_validation_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    import learning.complex_cnn as trainer
    from torch import nn

    class NonfiniteModel(nn.Module):
        def __init__(self, _links: int, _capacity: str) -> None:
            super().__init__()
            self.logits = nn.Parameter(torch.tensor([float("nan"), 0.0]))

        def forward(self, real, _imag):
            return self.logits.expand(len(real), -1)

    monkeypatch.setattr(trainer, "build_complex_cnn", NonfiniteModel)
    train = np.ones((4, 1, 8), dtype=np.complex64)
    labels = np.asarray([0, 1, 0, 1])

    with pytest.raises(RuntimeError, match="did not produce a finite validation checkpoint"):
        trainer.fit_complex_cnn(train, labels, train, labels, epochs=3, patience=1, batch_size=4)
