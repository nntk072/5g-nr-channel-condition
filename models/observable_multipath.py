"""Differentiable two-path observation model and uncertainty-aware decoder.

This is an analytical research baseline, not a validated novel method. The
common delay is removed as an unobservable timing gauge; the decoder estimates
only relative delay and complex path gains from the supplied pilot tones.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def render_two_path_cfr(frequencies_hz: Any, gains: Any, relative_delay_s: Any) -> Any:
    """Render ``g0 + g1 exp(-j 2π f Δτ)`` with Torch autograd support.

    ``frequencies_hz`` is ``[tones]`` or ``[batch, tones]``; complex ``gains`` is ``[batch, 2]``;
    ``relative_delay_s`` is ``[batch]`` or ``[batch, 1]``. The first path is
    the zero-delay reference, which fixes the common-delay gauge.
    """
    import torch

    if not torch.is_tensor(frequencies_hz) or not torch.is_tensor(gains):
        raise TypeError("frequencies_hz and gains must be Torch tensors")
    if not torch.is_complex(gains) or gains.ndim != 2 or gains.shape[1] != 2:
        raise ValueError("gains must be a complex tensor with shape [batch, 2]")
    if frequencies_hz.ndim not in (1, 2) or frequencies_hz.shape[-1] < 2:
        raise ValueError("frequencies_hz must contain at least two tones")
    delay = torch.as_tensor(relative_delay_s, device=gains.device, dtype=gains.real.dtype)
    if delay.ndim == 1:
        delay = delay[:, None]
    if delay.shape != (gains.shape[0], 1):
        raise ValueError("relative_delay_s must have shape [batch] or [batch, 1]")
    if frequencies_hz.device != gains.device:
        raise ValueError("frequencies_hz and gains must be on the same device")
    frequencies = frequencies_hz.to(gains.real.dtype)
    if frequencies.ndim == 2 and frequencies.shape[0] != gains.shape[0]:
        raise ValueError("batched frequencies must match the gain batch size")
    if (
        bool(torch.any(~torch.isfinite(frequencies)))
        or bool(torch.any(~torch.isfinite(gains.real)))
        or bool(torch.any(~torch.isfinite(gains.imag)))
        or bool(torch.any(~torch.isfinite(delay)))
    ):
        raise ValueError("frequencies, gains, and relative delays must be finite")
    if frequencies.ndim == 1:
        phase = -2j * math.pi * frequencies[None, :] * delay
    else:
        phase = -2j * math.pi * frequencies * delay
    return gains[:, :1] + gains[:, 1:2] * torch.exp(phase)


def complex_pilot_gaussian_nll(
    observed: Any,
    predicted: Any,
    noise_variance: Any,
    pilot_mask: Any,
) -> Any:
    """Mean complex-Gaussian negative log likelihood on observed pilot tones.

    The variance is the complex variance ``E[|n|²]``. Unobserved tones do not
    contribute; each batch item is normalized by its own observed-tone count.
    """
    import torch

    if observed.shape != predicted.shape or observed.ndim != 2 or not torch.is_complex(observed):
        raise ValueError("observed and predicted must be matching complex [batch, tones] tensors")
    if (
        bool(torch.any(~torch.isfinite(observed.real)))
        or bool(torch.any(~torch.isfinite(observed.imag)))
        or bool(torch.any(~torch.isfinite(predicted.real)))
        or bool(torch.any(~torch.isfinite(predicted.imag)))
    ):
        raise ValueError("observations and predictions must be finite")
    if (
        pilot_mask.shape != observed.shape
        or pilot_mask.dtype != torch.bool
        or pilot_mask.device != observed.device
    ):
        raise ValueError("pilot_mask must be boolean and match the observation shape")
    if bool(torch.any(pilot_mask.sum(dim=1) == 0)):
        raise ValueError("each batch item must contain at least one observed pilot")
    variance = torch.as_tensor(noise_variance, device=observed.device, dtype=observed.real.dtype)
    if variance.ndim == 1:
        variance = variance[:, None]
    if variance.shape not in {(observed.shape[0], 1), observed.shape}:
        raise ValueError("noise_variance must have shape [batch] or [batch, tones]")
    if bool(torch.any(~torch.isfinite(variance))) or bool(torch.any(variance <= 0)):
        raise ValueError("noise_variance must be finite and positive")
    residual_power = (observed - predicted).abs().square()
    element_nll = residual_power / variance + torch.log(math.pi * variance)
    mask = pilot_mask.to(element_nll.dtype)
    return (element_nll * mask).sum(dim=1).div(mask.sum(dim=1)).mean()


def heteroscedastic_delay_nll(target_delay_s: Any, mean_delay_s: Any, log_scale_s: Any) -> Any:
    """Gaussian scalar NLL (up to its constant) for relative-delay uncertainty."""
    import torch

    target = torch.as_tensor(target_delay_s)
    if not target.is_floating_point():
        target = target.to(torch.float32)
    mean = torch.as_tensor(mean_delay_s, device=target.device, dtype=target.dtype)
    log_scale = torch.as_tensor(log_scale_s, device=target.device, dtype=target.dtype)
    if target.shape != mean.shape or target.shape != log_scale.shape:
        raise ValueError("target, mean, and log_scale must have identical shapes")
    if target.numel() == 0 or bool(torch.any(~torch.isfinite(target))):
        raise ValueError("target delays must be nonempty and finite")
    if bool(torch.any(~torch.isfinite(mean))) or bool(torch.any(~torch.isfinite(log_scale))):
        raise ValueError("predicted delays and log scales must be finite")
    log_scale = log_scale.clamp(min=-30.0, max=30.0)
    standardized = (target - mean) * torch.exp(-log_scale)
    return (0.5 * standardized.square() + log_scale).mean()


def relative_delay_crlb_scale_floor(
    gains: Any,
    relative_delay_s: Any,
    frequencies_hz: Any,
    noise_variance: Any,
    pilot_mask: Any,
    maximum_relative_delay_s: float,
) -> Any:
    """Compute a local delay CRLB floor after eliminating unknown complex gains.

    This is a Fisher-information diagnostic for the fixed two-ray model, not a
    posterior uncertainty estimate. The finite floor is capped by the declared
    delay support when the local information is singular or nonpositive.
    ``frequencies_hz`` can be shared ``[tones]`` or batched ``[batch, tones]``.
    """
    import torch

    if not torch.is_complex(gains) or gains.ndim != 2 or gains.shape[1] != 2:
        raise ValueError("gains must be a complex tensor with shape [batch, 2]")
    if pilot_mask.ndim != 2 or pilot_mask.shape[0] != gains.shape[0]:
        raise ValueError("pilot_mask must match the gain batch")
    if pilot_mask.dtype != torch.bool or bool(torch.any(pilot_mask.sum(dim=1) < 2)):
        raise ValueError("pilot_mask must be boolean with at least two pilots per row")
    frequencies = torch.as_tensor(frequencies_hz, device=gains.device, dtype=gains.real.dtype)
    if frequencies.ndim == 1:
        frequencies = frequencies[None, :].expand(gains.shape[0], -1)
    if frequencies.shape != pilot_mask.shape:
        raise ValueError("frequencies_hz must match the pilot mask")
    delay = torch.as_tensor(relative_delay_s, device=gains.device, dtype=gains.real.dtype)
    if delay.ndim == 1:
        delay = delay[:, None]
    if delay.shape != (gains.shape[0], 1):
        raise ValueError("relative_delay_s must have shape [batch] or [batch, 1]")
    variance = torch.as_tensor(noise_variance, device=gains.device, dtype=gains.real.dtype)
    if variance.ndim == 1:
        variance = variance[:, None]
    if variance.shape != (gains.shape[0], 1) or bool(
        torch.any(~torch.isfinite(variance)) | torch.any(variance <= 0)
    ):
        raise ValueError("noise_variance must be finite and positive with shape [batch]")
    if not math.isfinite(maximum_relative_delay_s) or maximum_relative_delay_s <= 0:
        raise ValueError("maximum_relative_delay_s must be finite and positive")
    if bool(torch.any(~torch.isfinite(frequencies))) or bool(torch.any(~torch.isfinite(delay))):
        raise ValueError("frequencies and delays must be finite")

    mask = pilot_mask.to(gains.real.dtype)
    delayed_atom = torch.exp(-2j * math.pi * frequencies * delay)
    derivative = -2j * math.pi * frequencies * gains[:, 1:2] * delayed_atom
    count = mask.sum(dim=1, keepdim=True)
    mean_atom = (delayed_atom * mask).sum(dim=1, keepdim=True) / count
    mean_derivative = (derivative * mask).sum(dim=1, keepdim=True) / count
    centered_atom = (delayed_atom - mean_atom) * mask
    centered_derivative = (derivative - mean_derivative) * mask
    atom_energy = centered_atom.abs().square().sum(dim=1, keepdim=True)
    projection = (centered_atom.conj() * centered_derivative).sum(
        dim=1, keepdim=True
    ) / atom_energy.clamp_min(torch.finfo(gains.real.dtype).tiny)
    residual = centered_derivative - centered_atom * projection
    information = 2.0 * residual.abs().square().sum(dim=1, keepdim=True) / variance
    raw_scale = torch.sqrt(1.0 / information.clamp_min(torch.finfo(gains.real.dtype).tiny))
    return torch.nan_to_num(
        raw_scale,
        nan=maximum_relative_delay_s,
        posinf=maximum_relative_delay_s,
        neginf=maximum_relative_delay_s,
    ).clamp(max=maximum_relative_delay_s)


def identifiability_scale_floor_loss(
    predicted_scale_s: Any,
    gains: Any,
    relative_delay_s: Any,
    frequencies_hz: Any,
    noise_variance: Any,
    pilot_mask: Any,
    maximum_relative_delay_s: float,
) -> Any:
    """Penalize predicted scales that understate the local two-path CRLB floor."""
    import torch

    predicted = torch.as_tensor(predicted_scale_s)
    if predicted.ndim == 1:
        predicted = predicted[:, None]
    if bool(torch.any(~torch.isfinite(predicted))) or bool(torch.any(predicted <= 0)):
        raise ValueError("predicted_scale_s must be finite and positive")
    floor = relative_delay_crlb_scale_floor(
        gains,
        relative_delay_s,
        frequencies_hz,
        noise_variance,
        pilot_mask,
        maximum_relative_delay_s,
    ).to(device=predicted.device, dtype=predicted.dtype)
    return (
        torch.relu(
            torch.log(floor) - torch.log(predicted.clamp_min(torch.finfo(predicted.dtype).tiny))
        )
        .square()
        .mean()
    )


def build_two_path_uncertainty_decoder(
    n_tones: int,
    maximum_relative_delay_s: float,
    *,
    hidden_dims: Sequence[int] = (256, 128),
):
    """Build an MLP that predicts two complex gains, relative delay, and scale.

    Inputs are noisy CFR observations, per-tone pilot masks, configured tone
    frequencies, and complex noise variance. Returned path gains are in the
    same units as the CFR; ``relative_delay_std_s`` is a positive heteroscedastic
    scale, to be calibrated on disjoint groups before reporting coverage.
    """
    import torch
    from torch import nn

    if isinstance(n_tones, bool) or not isinstance(n_tones, int) or n_tones < 2:
        raise ValueError("n_tones must be an integer >= 2")
    if not math.isfinite(maximum_relative_delay_s) or maximum_relative_delay_s <= 0:
        raise ValueError("maximum_relative_delay_s must be finite and positive")
    widths = tuple(hidden_dims)
    if not widths or any(
        isinstance(width, bool) or not isinstance(width, int) or width < 1 for width in widths
    ):
        raise ValueError("hidden_dims must contain positive integers")

    class TwoPathUncertaintyDecoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            # The normalized frequency coordinate carries phase shape but
            # erases absolute bandwidth.  Keep log-span as an explicit scalar
            # so delay in seconds remains identifiable across NR bandwidths.
            input_dim = 4 * n_tones + 1
            for width in widths:
                layers.extend((nn.Linear(input_dim, width), nn.LayerNorm(width), nn.SiLU()))
                input_dim = width
            layers.append(nn.Linear(input_dim, 6))
            self.network = nn.Sequential(*layers)
            self.n_tones = n_tones
            self.maximum_relative_delay_s = float(maximum_relative_delay_s)

        def forward(
            self,
            observed_cfr: Any,
            pilot_mask: Any,
            frequencies_hz: Any,
            noise_variance: Any,
        ) -> dict[str, Any]:
            if observed_cfr.ndim != 2 or observed_cfr.shape[1] != self.n_tones:
                raise ValueError("observed_cfr must have shape [batch, n_tones]")
            if not torch.is_complex(observed_cfr):
                raise ValueError("observed_cfr must be complex")
            if (
                pilot_mask.shape != observed_cfr.shape
                or pilot_mask.dtype != torch.bool
                or pilot_mask.device != observed_cfr.device
            ):
                raise ValueError("pilot_mask must be boolean and match observed_cfr")
            if bool(torch.any(pilot_mask.sum(dim=1) < 2)):
                raise ValueError("each batch item needs at least two pilot tones")
            frequencies = torch.as_tensor(
                frequencies_hz, device=observed_cfr.device, dtype=observed_cfr.real.dtype
            )
            if frequencies.shape not in {
                (self.n_tones,),
                (observed_cfr.shape[0], self.n_tones),
            } or bool(torch.any(~torch.isfinite(frequencies))):
                raise ValueError("frequencies_hz must be finite and match the pilot-tone shape")
            if bool(torch.any(~torch.isfinite(observed_cfr.real))) or bool(
                torch.any(~torch.isfinite(observed_cfr.imag))
            ):
                raise ValueError("observed_cfr must contain finite values")
            variance = torch.as_tensor(
                noise_variance, device=observed_cfr.device, dtype=observed_cfr.real.dtype
            )
            if variance.ndim == 1:
                variance = variance[:, None]
            if variance.shape != (observed_cfr.shape[0], 1):
                raise ValueError("noise_variance must have shape [batch]")
            if bool(torch.any(~torch.isfinite(variance))) or bool(torch.any(variance <= 0)):
                raise ValueError("noise_variance must be finite and positive")

            noise_std = torch.sqrt(variance)
            mask = pilot_mask.to(observed_cfr.real.dtype)
            standardized = observed_cfr / noise_std
            if frequencies.ndim == 1:
                frequency_span = frequencies.max() - frequencies.min()
                frequency_centered = frequencies - frequencies.mean()
            else:
                frequency_span = (
                    frequencies.max(dim=1, keepdim=True).values
                    - frequencies.min(dim=1, keepdim=True).values
                )
                frequency_centered = frequencies - frequencies.mean(dim=1, keepdim=True)
            if bool(torch.any(frequency_span <= 0)):
                raise ValueError("frequencies_hz must span a nonzero bandwidth")
            if frequencies.ndim == 1:
                normalized_frequency = frequency_centered / frequency_span.clamp_min(
                    torch.finfo(frequencies.dtype).eps
                )
                normalized_frequency = normalized_frequency.expand_as(mask)
                log_frequency_span = torch.log10(frequency_span).expand(observed_cfr.shape[0], 1)
            else:
                normalized_frequency = frequency_centered / frequency_span.clamp_min(
                    torch.finfo(frequencies.dtype).eps
                )
                log_frequency_span = torch.log10(frequency_span)
            features = torch.cat(
                (
                    standardized.real * mask,
                    standardized.imag * mask,
                    mask,
                    normalized_frequency,
                    log_frequency_span,
                ),
                dim=1,
            )
            raw = self.network(features)
            normalized_gains = torch.complex(raw[:, :2], raw[:, 2:4])
            gains = normalized_gains * noise_std
            delay_fraction = torch.sigmoid(raw[:, 4])
            relative_delay = delay_fraction * self.maximum_relative_delay_s
            relative_delay_std = (
                raw[:, 5].clamp(min=-9.0, max=1.0).exp() * self.maximum_relative_delay_s
            )
            return {
                "gains": gains,
                "relative_delay_s": relative_delay,
                "relative_delay_std_s": relative_delay_std,
                "delay_fraction": delay_fraction,
            }

    return TwoPathUncertaintyDecoder()


def load_two_path_uncertainty_decoder(
    checkpoint_path: str | Path,
    n_tones: int,
    maximum_relative_delay_s: float,
    *,
    hidden_dims: Sequence[int] = (256, 128),
    device: str = "cpu",
):
    """Load a non-pickle NPZ state dictionary into the declared decoder shape."""
    import numpy as np
    import torch

    model = build_two_path_uncertainty_decoder(
        n_tones, maximum_relative_delay_s, hidden_dims=hidden_dims
    ).to(device)
    try:
        with np.load(checkpoint_path, allow_pickle=False) as archive:
            state = {
                name: torch.as_tensor(np.array(archive[name], copy=True), device=device)
                for name in archive.files
            }
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid decoder checkpoint: {checkpoint_path}") from exc
    expected = model.state_dict()
    if set(state) != set(expected):
        raise ValueError("checkpoint parameter names do not match the decoder architecture")
    for name, tensor in state.items():
        if tensor.shape != expected[name].shape or tensor.dtype != expected[name].dtype:
            raise ValueError(f"checkpoint tensor is incompatible with decoder architecture: {name}")
        if tensor.is_floating_point() and bool(torch.any(~torch.isfinite(tensor))):
            raise ValueError(f"checkpoint tensor contains nonfinite values: {name}")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model
