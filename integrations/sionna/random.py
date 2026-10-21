"""Small helpers for reproducible calls into Sionna's shared RNG registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


def call_with_sionna_seed(device: str, seed: int, call: Callable[[], T]) -> T:
    """Run one Sionna call with a local seed and restore its RNG state."""
    try:
        from sionna.phy import config
    except ImportError as exc:
        raise RuntimeError("Sionna PHY is required for seeded simulation") from exc
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    generator = config.torch_rng(device)
    state: Any = generator.get_state()
    try:
        generator.manual_seed(seed)
        return call()
    finally:
        generator.set_state(state)
