"""Shared, explicit neural-baseline capacity envelopes."""

from __future__ import annotations

import re
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
    if name == "matched":
        # The runner resolves this shape-dependent request before model build.
        return _CAPACITIES["medium"]
    try:
        return _CAPACITIES[name]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"capacity must be one of {tuple(_CAPACITIES)}") from exc


def architecture_widths(capacity: str, architecture: str) -> tuple[int, int, int]:
    """Resolve preset or shape-matched widths for one network family."""
    if architecture not in {"mlp", "real_cnn", "complex_cnn"}:
        raise ValueError("unsupported neural architecture")
    match = re.fullmatch(r"matched_(mlp|real_cnn|complex_cnn)_(\d+)_(\d+)_(\d+)", capacity)
    if match:
        if match.group(1) != architecture:
            raise ValueError("matched capacity was resolved for a different architecture")
        widths = tuple(int(value) for value in match.groups()[1:])
        if any(width < 1 for width in widths):
            raise ValueError("matched widths must be positive")
        return widths
    if capacity == "matched":
        raise ValueError("shape-matched capacity must be resolved by the experiment runner")
    spec = capacity_spec(capacity)
    return spec.mlp_widths if architecture == "mlp" else spec.convolution_widths


def _mlp_parameters(input_features: int, widths: tuple[int, int, int]) -> int:
    dimensions = (input_features, *widths, 2)
    return sum((left + 1) * right for left, right in zip(dimensions, dimensions[1:]))


def _cnn_parameters(
    input_channels: int, widths: tuple[int, int, int], *, complex_convolution: bool
) -> int:
    first, second, third = widths
    multiplier = 2 if complex_convolution else 1
    convolution = (
        (multiplier * input_channels * 7 + 1) * first
        + (multiplier * first * 5 + 1) * second
        + (multiplier * second * 3 + 1) * third
    )
    head = (third * 8 + 1) * (second * 2) + (second * 2 + 1) * 2
    return convolution + head


def matched_capacity_names(*, links: int, subcarriers: int) -> dict[str, int | float | str]:
    """Choose CNN widths nearest the raw-CFR MLP's trainable parameter count.

    The MLP uses the medium preset as its declared reference architecture.
    CNN families retain their existing channel ratios (1:2:2); the smallest
    parameter-count difference is selected from integer widths 2..256.
    """
    if (
        isinstance(links, bool)
        or not isinstance(links, int)
        or links < 1
        or isinstance(subcarriers, bool)
        or not isinstance(subcarriers, int)
        or subcarriers < 8
    ):
        raise ValueError("matched-capacity inputs require positive links and at least eight tones")
    mlp_widths = capacity_spec("medium").mlp_widths
    target = _mlp_parameters(2 * links * subcarriers, mlp_widths)
    result: dict[str, int | str] = {
        "target_parameters": target,
        "mlp_capacity": f"matched_mlp_{'_'.join(map(str, mlp_widths))}",
        "mlp_parameters": target,
    }
    for architecture, complex_convolution in (("real_cnn", False), ("complex_cnn", True)):
        candidates = []
        for base in range(2, 257):
            widths = (base, 2 * base, 2 * base)
            count = _cnn_parameters(
                2 * links if architecture == "real_cnn" else links,
                widths,
                complex_convolution=complex_convolution,
            )
            candidates.append((abs(count - target), count, widths))
        difference, count, widths = min(candidates)
        relative_error = difference / target
        if relative_error > 0.1:
            raise ValueError(
                f"cannot parameter-match {architecture} within 10% of MLP target "
                f"({relative_error:.1%} error)"
            )
        result[f"{architecture}_capacity"] = f"matched_{architecture}_{'_'.join(map(str, widths))}"
        result[f"{architecture}_parameters"] = count
        result[f"{architecture}_relative_error"] = relative_error
    return result
