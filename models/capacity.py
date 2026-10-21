"""Shared, explicit neural-baseline capacity envelopes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CapacitySpec:
    mlp_widths: tuple[int, int, int]
    convolution_widths: tuple[int, int, int]


_CAPACITIES = {
    "small": CapacitySpec((64, 32, 16), (16, 32, 32)),
    "medium": CapacitySpec((128, 64, 32), (32, 64, 64)),
    "large": CapacitySpec((256, 128, 64), (64, 128, 128)),
}


def capacity_spec(name: str) -> CapacitySpec:
    """Resolve one named capacity preset or fail with the supported choices."""
    try:
        return _CAPACITIES[name]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"capacity must be one of {tuple(_CAPACITIES)}") from exc
