"""CNN training and predictor contracts: hyper-parameters, labels, shapes, checkpoints."""

from __future__ import annotations

import numpy as np
import pytest


def _apis():
    pytest.importorskip("torch")
    from learning.complex_cnn import fit_complex_cnn
    from learning.real_cnn import fit_real_cnn
    from models.complex_cnn import load_complex_cnn
    from models.real_cnn import _real_imaginary_channels, load_real_cnn

    return fit_real_cnn, fit_complex_cnn, _real_imaginary_channels, load_real_cnn, load_complex_cnn


def _data(links: int = 1, samples: int = 4):
    rng = np.random.default_rng(3)
    x = (rng.normal(size=(samples, links, 16)) + 1j * rng.normal(size=(samples, links, 16))).astype(
        np.complex64
    )
    y = np.tile(np.asarray([0, 1], dtype=np.int64), samples // 2)
    return x, y


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        pytest.param({"epochs": 0}, "epochs must be a positive integer", id="epochs"),
        pytest.param({"epochs": True}, "epochs must be a positive integer", id="epochs-bool"),
        pytest.param({"patience": 0}, "patience must be a positive integer", id="patience"),
        pytest.param({"batch_size": 0}, "batch_size must be a positive integer", id="batch"),
        pytest.param(
            {"learning_rate": 0.0}, "learning_rate must be finite and positive", id="rate"
        ),
        pytest.param(
            {"learning_rate": float("nan")},
            "learning_rate must be finite and positive",
            id="rate-nan",
        ),
        pytest.param({"device": "tpu"}, "device must be cpu or a CUDA device", id="device"),
    ],
)
def test_real_cnn_rejects_invalid_hyperparameters(kwargs, message) -> None:
    fit_real_cnn, *_ = _apis()
    x, y = _data()
    with pytest.raises(ValueError, match=message):
        fit_real_cnn(x, y, x, y, epochs=kwargs.pop("epochs", 1), **kwargs)


def test_real_cnn_refuses_unavailable_cuda_when_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    fit_real_cnn, *_ = _apis()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    x, y = _data()
    with pytest.raises(RuntimeError, match="CUDA device requested but unavailable"):
        fit_real_cnn(x, y, x, y, device="cuda:0", epochs=1)


def test_real_cnn_label_and_shape_contracts() -> None:
    fit_real_cnn, *_ = _apis()
    x, y = _data()
    with pytest.raises(ValueError, match="must match their observation batches"):
        fit_real_cnn(x, y[:2], x, y, epochs=1)
    single_class = np.zeros(4, dtype=np.int64)
    with pytest.raises(ValueError, match="both classes"):
        fit_real_cnn(x, single_class, x, y, epochs=1)
    with pytest.raises(ValueError, match="shapes differ"):
        fit_real_cnn(x, y, _data(links=2)[0], y, epochs=1)


def test_real_cnn_predictor_rejects_wrong_link_count_and_threshold(tmp_path) -> None:
    fit_real_cnn, _, _, load_real_cnn, _ = _apis()
    x, y = _data()
    predictor, _ = fit_real_cnn(
        x, y, x, y, seed=5, epochs=1, patience=1, batch_size=4, capacity="small"
    )
    with pytest.raises(ValueError, match="link count differs"):
        predictor.predict_proba(_data(links=2)[0])
    with pytest.raises(ValueError, match="threshold must be finite"):
        predictor.save(tmp_path / "bad.pt", threshold=1.5)
    with pytest.raises(ValueError, match="always consumes both"):
        predictor.save(tmp_path / "bad.pt", threshold=0.5, include_phase=False)
    predictor.save(tmp_path / "ok.pt", threshold=0.5)
    assert load_real_cnn(tmp_path / "ok.pt") is not None


def test_real_cnn_checkpoint_format_is_checked(tmp_path) -> None:
    import torch

    _, _, _, load_real_cnn, _ = _apis()
    torch.save({"format": "not-real-cnn"}, tmp_path / "wrong.pt")
    with pytest.raises(ValueError, match="Invalid real-CNN checkpoint format"):
        load_real_cnn(tmp_path / "wrong.pt")


def test_complex_cnn_predictor_checks_links_threshold_and_format(tmp_path) -> None:
    import torch

    _, fit_complex_cnn, _, _, load_complex_cnn = _apis()
    x, y = _data()
    predictor, _ = fit_complex_cnn(
        x, y, x, y, seed=5, epochs=1, patience=1, batch_size=4, capacity="small"
    )
    with pytest.raises(ValueError, match="link count differs"):
        predictor.predict_proba(_data(links=2)[0])
    with pytest.raises(ValueError, match="threshold must be finite"):
        predictor.save(tmp_path / "bad.pt", threshold=-0.1)
    torch.save({"format": "not-complex-cnn"}, tmp_path / "wrong.pt")
    with pytest.raises(ValueError, match="Invalid complex-CNN checkpoint format"):
        load_complex_cnn(tmp_path / "wrong.pt")


@pytest.mark.parametrize("channels", [True, 1.5])
def test_cnn_builders_require_integer_channels(channels) -> None:
    pytest.importorskip("torch")
    from models.complex_cnn import build_complex_cnn
    from models.real_cnn import build_real_cnn

    with pytest.raises(TypeError, match="input_channels must be an integer"):
        build_real_cnn(channels)
    with pytest.raises(TypeError, match="input_channels must be an integer"):
        build_complex_cnn(channels)
    with pytest.raises(ValueError, match="input_channels must be positive"):
        build_real_cnn(0)
    with pytest.raises(ValueError, match="input_channels must be positive"):
        build_complex_cnn(0)
