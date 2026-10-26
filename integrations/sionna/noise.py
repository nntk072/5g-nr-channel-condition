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
    try:
        raw_probe = torch.as_tensor(noise_variance, device=reference.device)
    except (TypeError, ValueError, RuntimeError, OverflowError) as error:
        raise ValueError("noise_variance must contain real numeric values") from error
    integer_dtypes = {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
    if raw_probe.layout != torch.strided:
        raise ValueError("noise_variance must be a dense real-valued tensor")
    if raw_probe.dtype == torch.bool or raw_probe.is_complex():
        raise ValueError("noise_variance must contain real numeric values, not Boolean or complex")
    if not (raw_probe.is_floating_point() or raw_probe.dtype in integer_dtypes):
        raise ValueError("noise_variance must contain real numeric values")
    if torch.is_tensor(noise_variance):
        raw_variance = raw_probe
    else:
        try:
            raw_variance = torch.as_tensor(
                noise_variance, dtype=torch.float64, device=reference.device
            )
        except (TypeError, ValueError, RuntimeError, OverflowError) as error:
            raise ValueError("noise_variance must contain real numeric values") from error
    if not bool(torch.isfinite(raw_variance).all()) or bool((raw_variance < 0).any()):
        raise ValueError("noise_variance must be finite and nonnegative")
    variance = raw_variance.to(dtype=reference.real.dtype)
    if not bool(torch.isfinite(variance).all()):
        raise ValueError("noise_variance must be representable as finite values in reference dtype")
    if bool(((raw_variance > 0) & (variance == 0)).any()):
        raise ValueError("positive noise_variance values must not underflow to zero")
    try:
        torch.broadcast_shapes(tuple(variance.shape), tuple(reference.shape))
    except (RuntimeError, ValueError) as exc:
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
