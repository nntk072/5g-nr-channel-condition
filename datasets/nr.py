"""Typed NR observation/target separation for receiver-only learning."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from receivers.nr import NRChannelEstimate, NRPUSCHConfig


def _freeze_mapping(values: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(values))


@dataclass(frozen=True, slots=True)
class NRModelInputs:
    """Only inference-time receiver data; contains no sample IDs or targets."""

    received_grid: Any
    h_hat: Any
    error_variance: Any
    pilot_mask: Any
    fft_bin_indices: tuple[int, ...]
    receiver_config: NRPUSCHConfig


@dataclass(frozen=True, slots=True)
class NRObservation:
    """Receiver-visible PUSCH data with explicit FFT-to-estimator bin mapping."""

    sample_id: str
    received_grid: Any
    channel_estimate: NRChannelEstimate
    fft_bin_indices: tuple[int, ...]
    receiver_config: NRPUSCHConfig

    def __post_init__(self) -> None:
        if not self.sample_id.strip():
            raise ValueError("sample_id must be non-empty")
        grid_shape = tuple(getattr(self.received_grid, "shape", ()))
        estimate_shape = tuple(getattr(self.channel_estimate.h_hat, "shape", ()))
        if len(grid_shape) != 5:
            raise ValueError("received_grid must have [batch, rx, rx_ant, symbol, fft_bin] axes")
        if any(size <= 0 for size in grid_shape):
            raise ValueError("received_grid axes must be non-empty")
        if len(estimate_shape) != 7:
            raise ValueError("channel estimate must have seven named axes")
        if estimate_shape[:3] != grid_shape[:3] or estimate_shape[5] != grid_shape[3]:
            raise ValueError("received grid and channel estimate batch/antenna/symbol axes differ")
        if not self.fft_bin_indices or len(self.fft_bin_indices) != estimate_shape[6]:
            raise ValueError("fft_bin_indices must map every estimated subcarrier")
        if any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= grid_shape[4]
            for index in self.fft_bin_indices
        ):
            raise ValueError("fft_bin_indices must be valid zero-based FFT-bin indices")
        if len(set(self.fft_bin_indices)) != len(self.fft_bin_indices):
            raise ValueError("fft_bin_indices must be unique")
        if not isinstance(self.receiver_config, NRPUSCHConfig):
            raise TypeError("receiver_config must be the typed NRPUSCHConfig")

    def model_inputs(self) -> NRModelInputs:
        """Return the inference payload without IDs, target values, or provenance."""
        return NRModelInputs(
            received_grid=self.received_grid,
            h_hat=self.channel_estimate.h_hat,
            error_variance=self.channel_estimate.error_variance,
            pilot_mask=self.channel_estimate.pilot_mask,
            fft_bin_indices=self.fft_bin_indices,
            receiver_config=self.receiver_config,
        )


@dataclass(frozen=True, slots=True)
class NRChannelRecord:
    """Training/evaluation record separating model inputs from private truth."""

    observation: NRObservation
    targets: Mapping[str, Any]
    target_masks: Mapping[str, Any]
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.targets:
            raise ValueError("at least one explicitly named target is required")
        if not set(self.target_masks).issubset(self.targets):
            raise ValueError("target masks must correspond to declared targets")
        if not self.provenance:
            raise ValueError("provenance must not be empty")
        object.__setattr__(self, "targets", _freeze_mapping(self.targets))
        object.__setattr__(self, "target_masks", _freeze_mapping(self.target_masks))
        object.__setattr__(self, "provenance", _freeze_mapping(self.provenance))
