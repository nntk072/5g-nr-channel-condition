"""Statistics for ordered impairments receiver."""

from __future__ import annotations

import importlib
import importlib.metadata
import math
import random
from typing import TYPE_CHECKING, Any

from scipy.stats import beta as beta_distribution

if TYPE_CHECKING:
    import torch
else:

    class _LazyTorch:
        def __getattr__(self, name: str) -> Any:
            module = importlib.import_module("torch")
            globals()["torch"] = module
            return getattr(module, name)

    torch = _LazyTorch()


def _cluster_bootstrap_interval(
    values: list[float], *, seed: int, alpha: float = 0.05, draws: int = 20_000
) -> tuple[float, float]:
    """Percentile interval over independent channel clusters."""
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("values must be a nonempty finite sequence")
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 100:
        raise ValueError("draws must be an integer >= 100")
    rng = random.Random(seed)
    count = len(values)
    samples = sorted(
        sum(values[rng.randrange(count)] for _ in range(count)) / count for _ in range(draws)
    )
    lower_index = math.floor((alpha / 2.0) * draws)
    upper_index = min(draws - 1, math.ceil((1.0 - alpha / 2.0) * draws) - 1)
    return samples[lower_index], samples[upper_index]


def _channel_any_failure_upper_bound(failed_channels: int, channels: int, *, alpha: float) -> float:
    """Exact one-sided bound for a conservative channel-level failure prevalence."""
    if isinstance(failed_channels, bool) or isinstance(channels, bool):
        raise TypeError("channel counts must be integers")
    if not isinstance(failed_channels, int) or not isinstance(channels, int):
        raise TypeError("channel counts must be integers")
    if channels < 1 or not 0 <= failed_channels <= channels:
        raise ValueError("require 0 <= failed_channels <= channels")
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if failed_channels == channels:
        return 1.0
    return float(
        beta_distribution.ppf(1.0 - alpha, failed_channels + 1, channels - failed_channels)
    )


def _pool_repeated_waveforms_for_shared_sync(waveform: torch.Tensor) -> torch.Tensor:
    """Concatenate repeated blocks within one channel for shared sync acquisition."""
    if waveform.ndim < 2 or waveform.shape[0] < 1 or waveform.shape[-1] < 1:
        raise ValueError("waveform must have a nonempty block axis and sample axis")
    blocks_first = waveform.movedim(0, -2)
    return blocks_first.reshape(1, *waveform.shape[1:-1], -1)
