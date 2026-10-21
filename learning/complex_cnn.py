"""Training loop for the phase-equivariant complex-valued CNN baseline."""

from __future__ import annotations

from typing import Any

import numpy as np
from models.complex_cnn import ComplexCNNPredictor, _complex_parts, build_complex_cnn


def fit_complex_cnn(
    x: Any,
    y: np.ndarray,
    x_validation: Any,
    y_validation: np.ndarray,
    *,
    seed: int = 42,
    epochs: int = 120,
    patience: int = 18,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    device: str = "cpu",
) -> tuple[ComplexCNNPredictor, dict[str, float | int]]:
    """Fit with per-link training-only RMS scaling and validation-only stopping."""
    import torch
    from torch import nn

    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if isinstance(patience, bool) or not isinstance(patience, int) or patience < 1:
        raise ValueError("patience must be a positive integer")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if learning_rate <= 0 or not np.isfinite(learning_rate):
        raise ValueError("learning_rate must be finite and positive")
    if device != "cpu" and not device.startswith("cuda"):
        raise ValueError("device must be cpu or a CUDA device such as cuda:0")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")

    train_real, train_imag = _complex_parts(x)
    validation_real, validation_imag = _complex_parts(x_validation)
    labels = np.asarray(y)
    validation_labels = np.asarray(y_validation)
    if not np.isin(labels, (0, 1)).all() or not np.isin(validation_labels, (0, 1)).all():
        raise ValueError("CVNN labels must be binary values 0 or 1")
    labels = labels.astype(np.int64)
    validation_labels = validation_labels.astype(np.int64)
    if labels.shape != (len(train_real),) or validation_labels.shape != (len(validation_real),):
        raise ValueError("CVNN labels must match their observation batches")
    if set(np.unique(labels)) != {0, 1} or set(np.unique(validation_labels)) != {0, 1}:
        raise ValueError("CVNN training and validation data must each contain both classes")
    if train_real.shape[1:] != validation_real.shape[1:]:
        raise ValueError("CVNN training and validation observation shapes differ")

    rms = np.sqrt(np.mean(train_real**2 + train_imag**2, axis=(0, 2)))
    rms[rms < 1e-6] = 1.0
    train_real, train_imag = train_real / rms[None, :, None], train_imag / rms[None, :, None]
    validation_real = validation_real / rms[None, :, None]
    validation_imag = validation_imag / rms[None, :, None]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    train_real_tensor = torch.as_tensor(train_real, dtype=torch.float32, device=device)
    train_imag_tensor = torch.as_tensor(train_imag, dtype=torch.float32, device=device)
    labels_tensor = torch.as_tensor(labels, dtype=torch.long, device=device)
    validation_real_tensor = torch.as_tensor(validation_real, dtype=torch.float32, device=device)
    validation_imag_tensor = torch.as_tensor(validation_imag, dtype=torch.float32, device=device)
    validation_labels_tensor = torch.as_tensor(validation_labels, dtype=torch.long, device=device)
    counts = np.bincount(labels, minlength=2).astype(np.float32)
    weights = torch.as_tensor(len(labels) / (2.0 * counts), dtype=torch.float32, device=device)
    model = build_complex_cnn(train_real.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    best_loss = float("inf")
    best_state = None
    stale_epochs = 0
    epoch_count = 0
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(labels))
        for start in range(0, len(order), batch_size):
            indices = torch.as_tensor(order[start : start + batch_size], device=device)
            logits = model(train_real_tensor[indices], train_imag_tensor[indices])
            loss = loss_fn(logits, labels_tensor[indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            value = float(
                loss_fn(
                    model(validation_real_tensor, validation_imag_tensor), validation_labels_tensor
                ).item()
            )
        epoch_count = epoch + 1
        if value < best_loss - 1e-5:
            best_loss = value
            best_state = {
                key: tensor.detach().clone() for key, tensor in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break
    if best_state is None:
        raise RuntimeError("CVNN training did not produce a finite validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    predictor = ComplexCNNPredictor(model, rms, device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    return predictor, {
        "epoch_count": epoch_count,
        "best_validation_ce": best_loss,
        "trainable_parameters": int(parameter_count),
    }
