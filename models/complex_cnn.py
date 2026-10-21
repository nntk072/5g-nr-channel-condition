"""Phase-equivariant complex convolutional baseline for CFR observations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _complex_parts(values: Any) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values)
    if array.ndim != 3 or not np.iscomplexobj(array):
        raise ValueError("CVNN inputs must be complex [samples, links, subcarriers]")
    if not np.isfinite(array).all():
        raise ValueError("CVNN inputs must be finite")
    return array.real.astype(np.float32), array.imag.astype(np.float32)


def build_complex_cnn(input_channels: int, capacity: str = "medium"):
    """Build a global-phase-equivariant complex encoder and invariant classifier."""
    from models.capacity import capacity_spec

    if isinstance(input_channels, bool) or not isinstance(input_channels, int):
        raise TypeError("input_channels must be an integer")
    if input_channels < 1:
        raise ValueError("input_channels must be positive")
    import torch
    from torch import nn
    from torch.nn import functional as functional

    width1, width2, width3 = capacity_spec(capacity).convolution_widths

    class ComplexConv1d(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
            super().__init__()
            self.padding = kernel_size // 2
            self.weight_real = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
            self.weight_imag = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
            self.bias_modulus = nn.Parameter(torch.zeros(out_channels))
            nn.init.kaiming_uniform_(self.weight_real, a=5**0.5)
            nn.init.kaiming_uniform_(self.weight_imag, a=5**0.5)

        def forward(self, real, imag):
            real_out = functional.conv1d(
                real, self.weight_real, padding=self.padding
            ) - functional.conv1d(imag, self.weight_imag, padding=self.padding)
            imag_out = functional.conv1d(
                real, self.weight_imag, padding=self.padding
            ) + functional.conv1d(imag, self.weight_real, padding=self.padding)
            magnitude = torch.sqrt(real_out.square() + imag_out.square() + 1e-12)
            factor = functional.relu(magnitude + self.bias_modulus[None, :, None]) / magnitude
            return real_out * factor, imag_out * factor

    class ComplexCFRConvNet(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.encoder = nn.ModuleList(
                (
                    ComplexConv1d(channels, width1, 7),
                    ComplexConv1d(width1, width2, 5),
                    ComplexConv1d(width2, width3, 3),
                )
            )
            self.pool = nn.AdaptiveAvgPool1d(8)
            self.head = nn.Sequential(
                nn.Linear(width3 * 8, width2 * 2), nn.ReLU(), nn.Linear(width2 * 2, 2)
            )

        def forward(self, real, imag):
            for layer in self.encoder:
                real, imag = layer(real, imag)
            magnitude = torch.sqrt(real.square() + imag.square() + 1e-12)
            return self.head(self.pool(magnitude).flatten(start_dim=1))

    return ComplexCFRConvNet(input_channels)


class ComplexCNNPredictor:
    """Inference wrapper using per-link RMS scales learned from training data only."""

    def __init__(
        self,
        model: Any,
        scale: np.ndarray,
        device: str,
        *,
        capacity: str = "medium",
        threshold: float | None = None,
    ):
        self.model = model
        self.scale = np.asarray(scale, dtype=np.float32)
        self.device = device
        self.capacity = capacity
        self.threshold = threshold

    def predict_proba(self, values: Any) -> np.ndarray:
        import torch

        real, imag = _complex_parts(values)
        if real.shape[1] != len(self.scale):
            raise ValueError("CVNN input link count differs from fitted training data")
        scale = self.scale[None, :, None]
        real_tensor = torch.as_tensor(real / scale, dtype=torch.float32, device=self.device)
        imag_tensor = torch.as_tensor(imag / scale, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(real_tensor, imag_tensor)
            return torch.softmax(logits, dim=-1).cpu().numpy()

    def save(self, path: str | Path, *, threshold: float) -> None:
        import torch

        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be finite and between zero and one")
        torch.save(
            {
                "format": "loslab.complex_cnn.v1",
                "capacity": self.capacity,
                "input_channels": int(len(self.scale)),
                "scale": torch.as_tensor(self.scale),
                "state_dict": self.model.state_dict(),
                "threshold": float(threshold),
            },
            str(path),
        )


def load_complex_cnn(path: str | Path, *, device: str = "cpu") -> ComplexCNNPredictor:
    """Load a CVNN checkpoint with tensor-only deserialization."""
    import torch

    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format") != "loslab.complex_cnn.v1":
        raise ValueError("Invalid complex-CNN checkpoint format")
    capacity = str(payload.get("capacity", "medium"))
    model = build_complex_cnn(int(payload["input_channels"]), capacity).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return ComplexCNNPredictor(
        model,
        payload["scale"].detach().cpu().numpy(),
        device,
        capacity=capacity,
        threshold=float(payload["threshold"]),
    )
