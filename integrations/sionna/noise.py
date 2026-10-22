"""Shared proper-complex receiver-noise generation for Sionna integrations."""

from __future__ import annotations

from typing import Any


def proper_complex_awgn(reference: Any, noise_variance: Any, *, generator: Any) -> Any:
    """Draw i.i.d. circular complex Gaussian noise with ``E[|n|^2] = variance``.

    The real and imaginary components each have variance ``noise_variance / 2``.
    ``reference`` determines the output shape, complex dtype, and device; the
    caller supplies a seeded device-compatible PyTorch generator.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install PyTorch to generate receiver noise") from exc

    if not torch.is_tensor(reference) or not torch.is_complex(reference):
        raise TypeError("reference must be a complex PyTorch tensor")
    variance = torch.as_tensor(noise_variance, dtype=reference.real.dtype, device=reference.device)
    if not bool(torch.isfinite(variance).all()) or bool((variance < 0).any()):
        raise ValueError("noise_variance must be finite and nonnegative")
    try:
        torch.broadcast_shapes(tuple(variance.shape), tuple(reference.shape))
    except RuntimeError as exc:
        raise ValueError("noise_variance must be broadcastable to reference") from exc

    real = torch.randn(
        reference.shape,
        dtype=reference.real.dtype,
        device=reference.device,
        generator=generator,
    )
    imag = torch.randn(
        reference.shape,
        dtype=reference.real.dtype,
        device=reference.device,
        generator=generator,
    )
    return torch.complex(real, imag) * torch.sqrt(variance / 2.0)
