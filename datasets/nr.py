"""Typed NR observation/target separation for receiver-only learning."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from receivers.nr import NRChannelEstimate, NRPUSCHConfig

from .targets import TargetDefinition


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

    @classmethod
    def from_receiver_output(
        cls,
        *,
        sample_id: str,
        received_grid: Any,
        channel_estimate: NRChannelEstimate,
        effective_subcarrier_indices: Any,
        receiver_config: NRPUSCHConfig,
    ) -> NRObservation:
        """Build an observation using the receiver's effective FFT-bin mapping.

        Sionna removes guard and DC carriers before channel estimation. The
        supplied indices must therefore come from the same resource grid as
        the estimate, rather than being reconstructed from bandwidth alone.
        """
        indices = tuple(int(index) for index in effective_subcarrier_indices)
        return cls(
            sample_id=sample_id,
            received_grid=received_grid,
            channel_estimate=channel_estimate,
            fft_bin_indices=indices,
            receiver_config=receiver_config,
        )

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
class LinkedNRObservation:
    """One receiver observation bound to a link and frame in a network episode."""

    link_id: str
    frame_index: int
    observation: NRObservation

    def __post_init__(self) -> None:
        if not isinstance(self.link_id, str) or not self.link_id.strip():
            raise ValueError("link_id must be a non-empty string")
        if isinstance(self.frame_index, bool) or not isinstance(self.frame_index, int):
            raise TypeError("frame_index must be an integer")
        if self.frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        if not isinstance(self.observation, NRObservation):
            raise TypeError("observation must be an NRObservation")
        if self.observation.received_grid.shape[0] != 1:
            raise ValueError("a linked observation must contain exactly one batch item")
        if self.observation.channel_estimate.h_hat.shape[0] != 1:
            raise ValueError("a linked channel estimate must contain exactly one batch item")


@dataclass(frozen=True, slots=True)
class NRChannelRecord:
    """Training/evaluation record separating model inputs from private truth."""

    observation: NRObservation
    targets: Mapping[str, Any]
    target_masks: Mapping[str, Any]
    provenance: Mapping[str, Any]
    target_definitions: Mapping[str, TargetDefinition] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.targets:
            raise ValueError("at least one explicitly named target is required")
        if not set(self.target_masks).issubset(self.targets):
            raise ValueError("target masks must correspond to declared targets")
        for name, mask in self.target_masks.items():
            mask_dtype = getattr(mask, "dtype", None)
            if mask_dtype is None:
                if not isinstance(mask, (bool, np.bool_)):
                    raise TypeError(f"target mask {name!r} must have boolean dtype")
            elif str(mask_dtype) not in {"bool", "torch.bool"}:
                raise TypeError(f"target mask {name!r} must have boolean dtype")
            target = self.targets[name]
            target_shape = tuple(target.shape) if hasattr(target, "shape") else np.shape(target)
            mask_shape = tuple(mask.shape) if hasattr(mask, "shape") else np.shape(mask)
            try:
                broadcast_shape = np.broadcast_shapes(target_shape, mask_shape)
            except ValueError as exc:
                raise ValueError(
                    f"target mask {name!r} is not broadcastable to its target"
                ) from exc
            if broadcast_shape != target_shape:
                raise ValueError(f"target mask {name!r} is not broadcastable to its target")
        if not self.provenance:
            raise ValueError("provenance must not be empty")
        definitions = dict(self.target_definitions)
        if definitions and set(definitions) != set(self.targets):
            raise ValueError("target definitions must describe every declared target")
        if any(not isinstance(value, TargetDefinition) for value in definitions.values()):
            raise TypeError("target definitions must be TargetDefinition values")
        object.__setattr__(self, "targets", _freeze_mapping(self.targets))
        object.__setattr__(self, "target_masks", _freeze_mapping(self.target_masks))
        object.__setattr__(self, "provenance", _freeze_mapping(self.provenance))
        object.__setattr__(self, "target_definitions", _freeze_mapping(definitions))


@dataclass(frozen=True, slots=True)
class NRTrajectory:
    """Time-ordered NR records grouped as one indivisible split unit."""

    trajectory_id: str
    sample_times_s: tuple[float, ...]
    records: tuple[NRChannelRecord, ...]

    def __post_init__(self) -> None:
        if not self.trajectory_id.strip():
            raise ValueError("trajectory_id must be non-empty")
        records = tuple(self.records)
        times = tuple(self.sample_times_s)
        if not records or len(times) != len(records):
            raise ValueError("sample_times_s must provide one time for every trajectory record")
        if any(isinstance(value, bool) for value in times):
            raise TypeError("sample_times_s values must be real timestamps, not booleans")
        try:
            times = tuple(float(value) for value in times)
        except (TypeError, ValueError) as exc:
            raise TypeError("sample_times_s values must be real timestamps") from exc
        if any(not math.isfinite(value) for value in times):
            raise ValueError("sample_times_s values must be finite")
        if any(right <= left for left, right in zip(times, times[1:])):
            raise ValueError("sample_times_s must be strictly increasing")
        sample_ids = tuple(record.observation.sample_id for record in records)
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("trajectory sample IDs must be unique")
        if any(tuple(record.observation.received_grid.shape)[0] != 1 for record in records):
            raise ValueError("each trajectory record must contain exactly one observation")
        receiver_configs = {record.observation.receiver_config for record in records}
        if len(receiver_configs) != 1:
            raise ValueError("all records in one trajectory must use the same receiver config")
        definitions: dict[str, TargetDefinition] = {}
        for record in records:
            for name, definition in record.target_definitions.items():
                if name in definitions and definitions[name] != definition:
                    raise ValueError(
                        f"target definition for {name!r} changes within one trajectory"
                    )
                definitions[name] = definition
        object.__setattr__(self, "sample_times_s", times)
        object.__setattr__(self, "records", records)

    @property
    def sample_ids(self) -> tuple[str, ...]:
        """Return sample IDs in chronological order."""
        return tuple(record.observation.sample_id for record in self.records)

    @property
    def target_names(self) -> tuple[str, ...]:
        """Return the sorted union of named targets present in this trajectory."""
        return tuple(sorted({name for record in self.records for name in record.targets}))

    def model_inputs(self) -> tuple[NRModelInputs, ...]:
        """Return time-aligned inference inputs without identifiers or target truth."""
        return tuple(record.observation.model_inputs() for record in self.records)

    def target_series(self, name: str) -> tuple[tuple[Any | None, ...], tuple[Any | None, ...]]:
        """Return aligned target values and masks; missing targets are represented by None."""
        if name not in self.target_names:
            raise KeyError(f"unknown trajectory target: {name}")
        values = tuple(record.targets.get(name) for record in self.records)
        masks = tuple(record.target_masks.get(name) for record in self.records)
        return values, masks
