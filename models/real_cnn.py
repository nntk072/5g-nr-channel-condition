"""Real-valued convolutional baseline for complex CFR observations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _real_imaginary_channels(values: Any) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 3 or not np.iscomplexobj(array):
        raise ValueError("CNN inputs must be complex [samples, links, subcarriers]")
    if not np.isfinite(array).all():
        raise ValueError("CNN inputs must be finite")
    return np.concatenate((array.real, array.imag), axis=1).astype(np.float32)


def build_real_cnn(input_channels: int):
    """Build a compact 1D CNN that preserves subcarrier order."""
    if isinstance(input_channels, bool) or not isinstance(input_channels, int):
        raise TypeError("input_channels must be an integer")
    if input_channels < 1:
        raise ValueError("input_channels must be positive")
    from torch import nn

    class RealCFRConvNet(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv1d(channels, 32, kernel_size=7, padding=3),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv1d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            self.pool = nn.AdaptiveAvgPool1d(8)
            self.head = nn.Sequential(nn.Linear(64 * 8, 128), nn.ReLU(), nn.Linear(128, 2))

        def forward(self, values):
            encoded = self.pool(self.encoder(values)).flatten(start_dim=1)
            return self.head(encoded)

    return RealCFRConvNet(input_channels)


class RealCNNPredictor:
    """Inference wrapper with train-only channel normalization and safe checkpointing."""

    def __init__(
        self,
        model: Any,
        mean: np.ndarray,
        scale: np.ndarray,
        device: str,
        *,
        threshold: float | None = None,
        include_phase: bool = True,
    ):
        self.model = model
        self.mean = np.asarray(mean, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)
        self.device = device
        self.threshold = threshold
        self.include_phase = include_phase

    def predict_proba(self, values: Any) -> np.ndarray:
        import torch

        channels = _real_imaginary_channels(values)
        if channels.shape[1] != self.mean.shape[1]:
            raise ValueError("CNN input link count differs from fitted training data")
        normalized = (channels - self.mean) / self.scale
        tensor = torch.as_tensor(normalized, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            return torch.softmax(self.model(tensor), dim=-1).cpu().numpy()

    def save(self, path: str | Path, *, threshold: float, include_phase: bool = True) -> None:
        import torch

        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be finite and between zero and one")
        if not include_phase:
            raise ValueError("the real-CNN always consumes both real and imaginary channels")
        torch.save(
            {
                "format": "loslab.real_cnn.v1",
                "input_channels": int(self.mean.shape[1]),
                "mean": torch.as_tensor(self.mean),
                "scale": torch.as_tensor(self.scale),
                "state_dict": self.model.state_dict(),
                "threshold": float(threshold),
                "include_phase": bool(include_phase),
            },
            str(path),
        )


def load_real_cnn(path: str | Path, *, device: str = "cpu") -> RealCNNPredictor:
    """Load a real-CNN checkpoint without pickle-based model deserialization."""
    import torch

    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format") != "loslab.real_cnn.v1":
        raise ValueError("Invalid real-CNN checkpoint format")
    model = build_real_cnn(int(payload["input_channels"])).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return RealCNNPredictor(
        model,
        payload["mean"].detach().cpu().numpy(),
        payload["scale"].detach().cpu().numpy(),
        device,
        threshold=float(payload["threshold"]),
        include_phase=bool(payload["include_phase"]),
    )
