"""Controlled analytical multipath fixtures, not standardized channel data."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
from propagation.physics import frequency_response, subcarrier_frequencies


@dataclass(frozen=True, slots=True)
class AnalyticalModelInputs:
    """Receiver-observable synthetic input with no path oracle or sample ID."""

    frequencies_hz: np.ndarray
    channel_frequency_response: np.ndarray


@dataclass(frozen=True, slots=True)
class AnalyticalPathOracle:
    """Synthetic path truth for QA only; this is not a geometric LOS label."""

    excess_delays_s: np.ndarray
    complex_gains: np.ndarray
    direct_component_mask: np.ndarray
    direct_component_present: bool
    receiver_timing_offset_s: float


@dataclass(frozen=True, slots=True)
class AnalyticalMultipathSample:
    """A seeded, low-complexity channel with separate observation and oracle."""

    sample_id: str
    model_inputs: AnalyticalModelInputs
    oracle: AnalyticalPathOracle
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.sample_id.strip():
            raise ValueError("sample_id must be non-empty")
        if not self.metadata:
            raise ValueError("metadata must not be empty")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


def generate_analytical_multipath(
    *,
    seed: int,
    num_paths: int,
    num_subcarriers: int = 256,
    subcarrier_spacing_hz: float = 30_000.0,
    max_excess_delay_s: float = 1e-6,
    direct_component_present: bool = True,
    receiver_timing_offset_s: float = 0.0,
) -> AnalyticalMultipathSample:
    """Generate a seeded 1–3-path CFR with explicit timing and path oracle.

    The configured "direct component" is merely a designated synthetic path;
    it does not model or label geometric LOS. Gains are unnormalized complex
    Gaussian draws. A common timing offset is added to every absolute path
    delay, making synchronization assumptions explicit.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(num_paths, bool) or not isinstance(num_paths, int) or not 1 <= num_paths <= 3:
        raise ValueError("num_paths must be an integer in [1, 3]")
    if num_subcarriers < 8:
        raise ValueError("num_subcarriers must be at least 8")
    if not np.isfinite(subcarrier_spacing_hz) or subcarrier_spacing_hz <= 0:
        raise ValueError("subcarrier_spacing_hz must be finite and positive")
    if not np.isfinite(max_excess_delay_s) or max_excess_delay_s < 0:
        raise ValueError("max_excess_delay_s must be finite and nonnegative")
    if not np.isfinite(receiver_timing_offset_s) or receiver_timing_offset_s < 0:
        raise ValueError("receiver_timing_offset_s must be finite and nonnegative")

    rng = np.random.default_rng(seed)
    if direct_component_present:
        reflected_count = num_paths - 1
        reflected_delays = np.sort(rng.uniform(0.0, max_excess_delay_s, reflected_count))
        excess_delays = np.concatenate(([0.0], reflected_delays))
        direct_mask = np.zeros(num_paths, dtype=np.bool_)
        direct_mask[0] = True
    else:
        excess_delays = np.sort(rng.uniform(0.0, max_excess_delay_s, num_paths))
        direct_mask = np.zeros(num_paths, dtype=np.bool_)

    gains = (rng.standard_normal(num_paths) + 1j * rng.standard_normal(num_paths)) / np.sqrt(2.0)
    absolute_delays = excess_delays + receiver_timing_offset_s
    frequencies = subcarrier_frequencies(num_subcarriers, subcarrier_spacing_hz)
    cfr = frequency_response(gains, absolute_delays, frequencies).astype(np.complex64)

    config = {
        "schema_version": 1,
        "generator": "analytical_multipath_debug",
        "seed": seed,
        "num_paths": num_paths,
        "num_subcarriers": num_subcarriers,
        "subcarrier_spacing_hz": float(subcarrier_spacing_hz),
        "max_excess_delay_s": float(max_excess_delay_s),
        "direct_component_present": direct_component_present,
        "receiver_timing_offset_s": float(receiver_timing_offset_s),
    }
    sample_id = "analytic-" + hashlib.sha256(repr(sorted(config.items())).encode()).hexdigest()[:20]
    oracle = AnalyticalPathOracle(
        excess_delays_s=excess_delays.astype(np.float64),
        complex_gains=gains.astype(np.complex64),
        direct_component_mask=direct_mask,
        direct_component_present=direct_component_present,
        receiver_timing_offset_s=float(receiver_timing_offset_s),
    )
    return AnalyticalMultipathSample(
        sample_id=sample_id,
        model_inputs=AnalyticalModelInputs(
            frequencies_hz=frequencies,
            channel_frequency_response=cfr,
        ),
        oracle=oracle,
        metadata=MappingProxyType(config),
    )
