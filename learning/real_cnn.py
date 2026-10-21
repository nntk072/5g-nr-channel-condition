"""Training loop for the observation-only real/imaginary CNN baseline."""

from __future__ import annotations

from typing import Any

import numpy as np
from models.real_cnn import RealCNNPredictor, _real_imaginary_channels, build_real_cnn


def fit_real_cnn(
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
    capacity: str = "medium",
) -> tuple[RealCNNPredictor, dict[str, float | int]]:
    """Train a normalized 1D CNN; fit scaling on training observations only."""
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

    train = _real_imaginary_channels(x)
    validation = _real_imaginary_channels(x_validation)
    raw_labels = np.asarray(y)
    raw_validation_labels = np.asarray(y_validation)
    if not np.isin(raw_labels, (0, 1)).all() or not np.isin(raw_validation_labels, (0, 1)).all():
        raise ValueError("CNN labels must be binary values 0 or 1")
    labels = raw_labels.astype(np.int64)
    validation_labels = raw_validation_labels.astype(np.int64)
    if labels.shape != (len(train),) or validation_labels.shape != (len(validation),):
        raise ValueError("CNN labels must match their observation batches")
    if set(np.unique(labels)) != {0, 1} or set(np.unique(validation_labels)) != {0, 1}:
        raise ValueError("CNN training and validation data must each contain both classes")
    if train.shape[1:] != validation.shape[1:]:
        raise ValueError("CNN training and validation observation shapes differ")

    mean = train.mean(axis=(0, 2), keepdims=True)
    scale = train.std(axis=(0, 2), keepdims=True)
    scale[scale < 1e-6] = 1.0
    train = (train - mean) / scale
    validation = (validation - mean) / scale
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    x_train = torch.as_tensor(train, dtype=torch.float32, device=device)
    y_train = torch.as_tensor(labels, dtype=torch.long, device=device)
    x_validation_tensor = torch.as_tensor(validation, dtype=torch.float32, device=device)
    y_validation_tensor = torch.as_tensor(validation_labels, dtype=torch.long, device=device)
    class_counts = np.bincount(labels, minlength=2).astype(np.float32)
    class_weights = torch.as_tensor(
        len(labels) / (2.0 * class_counts), dtype=torch.float32, device=device
    )
    model = build_real_cnn(train.shape[1], capacity).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    best_loss = float("inf")
    best_state: dict[str, Any] | None = None
    stale_epochs = 0
    epoch_count = 0
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(labels))
        for start in range(0, len(order), batch_size):
            indices = torch.as_tensor(order[start : start + batch_size], device=device)
            logits = model(x_train[indices])
            loss = loss_fn(logits, y_train[indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(loss_fn(model(x_validation_tensor), y_validation_tensor).item())
        epoch_count = epoch + 1
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_state = {
                name: value.detach().clone() for name, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break
    if best_state is None:
        raise RuntimeError("real-CNN training did not produce a finite validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    predictor = RealCNNPredictor(model, mean, scale, device, capacity=capacity)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    return predictor, {
        "epoch_count": epoch_count,
        "best_validation_ce": best_loss,
        "trainable_parameters": int(parameter_count),
    }
