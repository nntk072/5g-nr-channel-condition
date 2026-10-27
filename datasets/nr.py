"""Typed NR observation/target separation for receiver-only learning."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from importlib import import_module
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from receivers.nr import NRChannelEstimate, NRPUSCHConfig

from .model_input_audit import audit_model_input_payload
from .targets import TargetDefinition, TargetValidity, validate_canonical_target_contract

MAX_REFERENCE_TRANSPORT_BLOCK_BITS = 20_000_000


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
    received_grid_mask: Any | None = None


@dataclass(frozen=True, slots=True)
class NRObservation:
    """Receiver-visible PUSCH data with explicit FFT-to-estimator bin mapping."""

    sample_id: str
    received_grid: Any
    channel_estimate: NRChannelEstimate
    fft_bin_indices: tuple[int, ...]
    receiver_config: NRPUSCHConfig
    received_grid_mask: Any | None = None

    def __post_init__(self) -> None:
        if not self.sample_id.strip():
            raise ValueError("sample_id must be non-empty")
        grid_shape = tuple(getattr(self.received_grid, "shape", ()))
        estimate_shape = tuple(getattr(self.channel_estimate.h_hat, "shape", ()))
        if len(grid_shape) != 5:
            raise ValueError("received_grid must have [batch, rx, rx_ant, symbol, fft_bin] axes")
        if any(size <= 0 for size in grid_shape):
            raise ValueError("received_grid axes must be non-empty")
        torch_module: Any = None
        try:
            torch_module = import_module("torch")
        except ImportError:
            pass
        mask = self.received_grid_mask
        if mask is None:
            mask = (
                torch_module.ones(
                    grid_shape, dtype=torch_module.bool, device=self.received_grid.device
                )
                if torch_module is not None and torch_module.is_tensor(self.received_grid)
                else np.ones(grid_shape, dtype=bool)
            )
            object.__setattr__(self, "received_grid_mask", mask)
        mask_shape = tuple(getattr(mask, "shape", ()))
        if mask_shape != grid_shape:
            raise ValueError("received_grid_mask must match the received_grid axes exactly")
        mask_dtype = getattr(mask, "dtype", None)
        if str(mask_dtype) not in {"bool", "torch.bool"}:
            raise TypeError("received_grid_mask must have boolean dtype")
        if (
            torch_module is not None
            and torch_module.is_tensor(self.received_grid)
            and torch_module.is_tensor(mask)
            and mask.device != self.received_grid.device
        ):
            raise ValueError("received_grid_mask must be on the same device as received_grid")
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
        received_grid_mask: Any | None = None,
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
            received_grid_mask=received_grid_mask,
        )

    def model_inputs(self) -> NRModelInputs:
        """Return the inference payload without IDs, target values, or provenance."""
        return audit_model_input_payload(
            NRModelInputs(
                received_grid=self.received_grid,
                received_grid_mask=self.received_grid_mask,
                h_hat=self.channel_estimate.h_hat,
                error_variance=self.channel_estimate.error_variance,
                pilot_mask=self.channel_estimate.pilot_mask,
                fft_bin_indices=self.fft_bin_indices,
                receiver_config=self.receiver_config,
            )
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
    """Training/evaluation record separating model inputs from private truth.

    ``reference_transport_block_bits`` is evaluation-side TX truth. It is kept
    outside ``NRObservation`` and ``NRModelInputs`` so it cannot enter the
    receiver-visible model feature contract.
    """

    observation: NRObservation
    targets: Mapping[str, Any]
    target_masks: Mapping[str, Any]
    provenance: Mapping[str, Any]
    target_definitions: Mapping[str, TargetDefinition] = field(default_factory=dict)
    target_reason_codes: Mapping[str, tuple[TargetValidity, ...]] = field(default_factory=dict)
    reference_transport_block_bits: Any | None = None

    def __post_init__(self) -> None:
        if not self.targets:
            raise ValueError("at least one explicitly named target is required")
        torch_module: Any = None
        try:
            torch_module = import_module("torch")
        except ImportError:
            pass
        for name, target in self.targets.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("target names must be non-empty strings")
            if torch_module is not None and torch_module.is_tensor(target):
                if target.numel() == 0:
                    raise ValueError(f"target {name!r} must be non-empty")
                finite = bool(torch_module.isfinite(target).all().item())
            else:
                array = np.asarray(target)
                if array.dtype.kind not in "biufc":
                    raise TypeError(f"target {name!r} must use a numeric or boolean dtype")
                if array.size == 0:
                    raise ValueError(f"target {name!r} must be non-empty")
                finite = bool(np.isfinite(array).all())
            if not finite:
                raise ValueError(f"target {name!r} must contain only finite values")
        if not set(self.target_masks).issubset(self.targets):
            raise ValueError("target masks must correspond to declared targets")
        if not set(self.target_reason_codes).issubset(self.targets):
            raise ValueError("target reason codes must correspond to declared targets")
        normalized_reasons: dict[str, tuple[TargetValidity, ...]] = {}
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
            if torch_module is not None and torch_module.is_tensor(target):
                expanded_mask = torch_module.broadcast_to(
                    torch_module.as_tensor(mask, dtype=torch_module.bool, device=target.device),
                    target_shape,
                )
                unavailable = target.masked_select(torch_module.logical_not(expanded_mask))
                if unavailable.numel() and bool(torch_module.any(unavailable != 0).item()):
                    raise ValueError(
                        f"masked target elements for {name!r} must use zero storage"
                    )
            else:
                target_array = np.asarray(target)
                mask_array = (
                    mask.detach().cpu().numpy()
                    if torch_module is not None and torch_module.is_tensor(mask)
                    else np.asarray(mask)
                )
                expanded_mask = np.broadcast_to(
                    mask_array.astype(bool, copy=False), target_shape
                )
                if np.any(target_array[~expanded_mask] != 0):
                    raise ValueError(f"masked target elements for {name!r} must use zero storage")
        for name, target in self.targets.items():
            target_shape = tuple(target.shape) if hasattr(target, "shape") else np.shape(target)
            raw_mask = self.target_masks.get(name)
            if raw_mask is None:
                expanded_mask = np.ones(target_shape, dtype=bool)
            else:
                if torch_module is not None and torch_module.is_tensor(raw_mask):
                    mask_array = raw_mask.detach().cpu().numpy()
                else:
                    mask_array = np.asarray(raw_mask)
                expanded_mask = np.broadcast_to(
                    mask_array.astype(bool, copy=False), target_shape
                )
            flat_mask = expanded_mask.reshape(-1)
            if name in self.target_reason_codes:
                reasons = tuple(self.target_reason_codes[name])
                if len(reasons) != flat_mask.size or any(
                    not isinstance(reason, TargetValidity) for reason in reasons
                ):
                    raise ValueError(
                        f"target reason codes for {name!r} must contain one TargetValidity "
                        "per target element"
                    )
            else:
                if not bool(np.all(flat_mask)):
                    raise ValueError(
                        f"target reason codes are required for unavailable elements of {name!r}"
                    )
                reasons = (TargetValidity.VALID,) * flat_mask.size
            if any(
                selected != (reason is TargetValidity.VALID)
                for selected, reason in zip(flat_mask, reasons, strict=True)
            ):
                raise ValueError(
                    f"target masks and reason codes disagree for target {name!r}"
                )
            normalized_reasons[name] = reasons
        if not self.provenance:
            raise ValueError("provenance must not be empty")
        reference_bits = self.reference_transport_block_bits
        reference_shape_key = "reference_transport_block_shape"
        reference_digest_key = "reference_transport_block_sha256"
        declared_reference_fields = {
            key for key in (reference_shape_key, reference_digest_key) if key in self.provenance
        }
        if reference_bits is not None:
            if torch_module is not None and torch_module.is_tensor(reference_bits):
                reference_bits = reference_bits.detach().cpu().numpy()
            reference_array = np.asarray(reference_bits)
            if (
                reference_array.ndim != 2
                or any(size <= 0 for size in reference_array.shape)
                or reference_array.size > MAX_REFERENCE_TRANSPORT_BLOCK_BITS
                or reference_array.dtype.kind not in "biu"
                or np.any((reference_array != 0) & (reference_array != 1))
            ):
                raise ValueError(
                    "reference_transport_block_bits must be a bounded binary [TX, bit] array"
                )
            reference_array = np.array(reference_array, dtype=np.uint8, copy=True)
            expected_reference_shape = self.provenance.get(reference_shape_key)
            expected_reference_digest = self.provenance.get(reference_digest_key)
            actual_reference_digest = hashlib.sha256(reference_array.tobytes(order="C")).hexdigest()
            if (
                declared_reference_fields != {reference_shape_key, reference_digest_key}
                or not isinstance(expected_reference_shape, list)
                or any(type(size) is not int or size <= 0 for size in expected_reference_shape)
                or expected_reference_shape != list(reference_array.shape)
                or expected_reference_digest != actual_reference_digest
            ):
                raise ValueError(
                    "reference transport-block payload differs from its provenance shape/digest"
                )
            reference_array.setflags(write=False)
            object.__setattr__(self, "reference_transport_block_bits", reference_array)
        elif declared_reference_fields:
            raise ValueError("reference transport-block provenance requires its payload array")
        definitions = dict(self.target_definitions)
        if definitions and set(definitions) != set(self.targets):
            raise ValueError("target definitions must describe every declared target")
        if any(not isinstance(value, TargetDefinition) for value in definitions.values()):
            raise TypeError("target definitions must be TargetDefinition values")
        object.__setattr__(self, "targets", _freeze_mapping(self.targets))
        object.__setattr__(self, "target_masks", _freeze_mapping(self.target_masks))
        object.__setattr__(self, "provenance", _freeze_mapping(self.provenance))
        object.__setattr__(self, "target_definitions", _freeze_mapping(definitions))
        object.__setattr__(self, "target_reason_codes", _freeze_mapping(normalized_reasons))

    def require_versioned_target_definitions(self) -> None:
        """Require a digest-verified semantic definition for every persisted target."""
        if set(self.target_definitions) != set(self.targets):
            raise ValueError(
                "canonical NR records require one target definition for every target"
            )
        for name, definition in self.target_definitions.items():
            try:
                definition.require_versioned()
            except ValueError as error:
                raise ValueError(
                    f"canonical NR target definition for {name!r} is not versioned: {error}"
                ) from error
            validate_canonical_target_contract(
                self.targets[name],
                definition,
                name=name,
                valid_mask=self.target_masks.get(name),
            )
            target_value = self.targets[name]
            target_shape_attribute = getattr(target_value, "shape", None)
            target_shape = (
                tuple(target_shape_attribute)
                if target_shape_attribute is not None
                else tuple(np.shape(target_value))
            )
            version = definition.definition_version
            estimate_shape = tuple(self.observation.channel_estimate.h_hat.shape)
            effective_subcarrier_count = len(self.observation.fft_bin_indices)
            expected_shape: tuple[int, ...] | None = None
            if version in {
                "umcd.standardized.effective-cfr.v1",
                "umcd.tdl-pusch.effective-cfr.v1",
            }:
                expected_shape = estimate_shape
            elif version == "umcd.rt-trajectory.absolute-cfr.v1":
                expected_shape = (effective_subcarrier_count,)
            elif version == "umcd.rt-trajectory.symbol-cfr.v1":
                expected_shape = (estimate_shape[5], effective_subcarrier_count)
            if expected_shape is not None and target_shape != expected_shape:
                raise ValueError(
                    f"canonical NR target {name!r} has shape {target_shape}, "
                    f"expected {expected_shape} from its observation and FFT mapping"
                )


@dataclass(frozen=True, slots=True)
class NRTrajectory:
    """Time-ordered NR records grouped as one indivisible split unit."""

    trajectory_id: str
    sample_times_s: tuple[float, ...]
    records: tuple[NRChannelRecord, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.trajectory_id, str):
            raise TypeError("trajectory_id must be a string")
        if not self.trajectory_id or self.trajectory_id.strip() != self.trajectory_id:
            raise ValueError("trajectory_id must be a non-empty canonical string")
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

    def target_reason_series(
        self, name: str
    ) -> tuple[tuple[TargetValidity, ...] | None, ...]:
        """Return per-record element reasons, preserving missing target records as None."""
        if name not in self.target_names:
            raise KeyError(f"unknown trajectory target: {name}")
        return tuple(record.target_reason_codes.get(name) for record in self.records)

    def target_support_counts(self, name: str) -> dict[str, int]:
        """Count target presence and mask-selected elements across this trajectory.

        Missing per-record masks follow the existing record contract and mean
        all elements are selected; ``implicit_all_valid_mask_record_count``
        makes those implicit selections visible in the report.
        """
        if name not in self.target_names:
            raise KeyError(f"unknown trajectory target: {name}")
        torch_module: Any = None
        try:
            torch_module = import_module("torch")
        except ImportError:
            pass

        present_record_count = 0
        missing_record_count = 0
        element_count = 0
        selected_element_count = 0
        implicit_all_valid_mask_record_count = 0
        for record in self.records:
            target = record.targets.get(name)
            if target is None:
                missing_record_count += 1
                continue
            present_record_count += 1
            raw_shape = getattr(target, "shape", None)
            target_shape = tuple(raw_shape) if raw_shape is not None else tuple(np.shape(target))
            target_size = math.prod(target_shape) if target_shape else 1
            element_count += target_size
            mask = record.target_masks.get(name)
            if mask is None:
                selected_element_count += target_size
                implicit_all_valid_mask_record_count += 1
            elif torch_module is not None and torch_module.is_tensor(mask):
                expanded = torch_module.broadcast_to(mask, target_shape)
                selected_element_count += int(torch_module.count_nonzero(expanded).item())
            else:
                expanded = np.broadcast_to(np.asarray(mask), target_shape)
                selected_element_count += int(np.count_nonzero(expanded))

        return {
            "record_count": len(self.records),
            "present_record_count": present_record_count,
            "missing_record_count": missing_record_count,
            "element_count": element_count,
            "selected_element_count": selected_element_count,
            "masked_element_count": element_count - selected_element_count,
            "implicit_all_valid_mask_record_count": implicit_all_valid_mask_record_count,
        }


def audit_nr_trajectory_target_support(
    assignments: tuple[tuple[str, str, NRTrajectory], ...],
    expected_cells: tuple[tuple[str, str, str], ...],
) -> dict[str, Any]:
    """Aggregate NR target support by an explicit split/family/target roster.

    Each assignment is ``(split_id, source_family_id, trajectory)``. Every
    trajectory is an indivisible split unit; counts aggregate its records and
    target masks without treating records as independent split units. The
    expected-cell roster is mandatory so targets absent throughout a family
    still appear with zero support instead of disappearing from the report.
    """
    if not expected_cells:
        raise ValueError("expected_cells must declare split/family/target coverage")

    normalized_cells: set[tuple[str, str, str]] = set()
    targets_by_assignment: dict[tuple[str, str], set[str]] = {}
    for cell in expected_cells:
        if not isinstance(cell, tuple) or len(cell) != 3:
            raise ValueError("each expected cell must be a (split, family, target) tuple")
        split_id, family_id, target_id = cell
        if any(not isinstance(value, str) or not value.strip() for value in cell):
            raise ValueError("expected split, family, and target IDs must be nonempty strings")
        normalized = (split_id.strip(), family_id.strip(), target_id.strip())
        if normalized in normalized_cells:
            raise ValueError(f"duplicate expected NR target cell: {normalized!r}")
        normalized_cells.add(normalized)
        targets_by_assignment.setdefault(normalized[:2], set()).add(normalized[2])

    assigned_trajectories: set[str] = set()
    assigned_samples: set[str] = set()
    aggregate: dict[tuple[str, str, str], dict[str, int]] = {
        cell: {
            "trajectory_count": 0,
            "record_count": 0,
            "present_record_count": 0,
            "missing_record_count": 0,
            "element_count": 0,
            "selected_element_count": 0,
            "masked_element_count": 0,
            "implicit_all_valid_mask_record_count": 0,
        }
        for cell in normalized_cells
    }
    for assignment in assignments:
        if not isinstance(assignment, tuple) or len(assignment) != 3:
            raise ValueError("each assignment must be a (split, family, trajectory) tuple")
        split_id, family_id, trajectory = assignment
        if any(not isinstance(value, str) or not value.strip() for value in (split_id, family_id)):
            raise ValueError("assignment split and family IDs must be nonempty strings")
        if not isinstance(trajectory, NRTrajectory):
            raise TypeError("assignment trajectory must be an NRTrajectory")
        trajectory_id = trajectory.trajectory_id
        if trajectory_id in assigned_trajectories:
            raise ValueError(f"duplicate trajectory assignment: {trajectory_id!r}")
        assigned_trajectories.add(trajectory_id)
        duplicate_samples = assigned_samples.intersection(trajectory.sample_ids)
        if duplicate_samples:
            raise ValueError(
                "NR sample IDs may not be counted more than once across trajectories: "
                f"{sorted(duplicate_samples)!r}"
            )
        assigned_samples.update(trajectory.sample_ids)
        assignment_key = (split_id.strip(), family_id.strip())
        expected_targets = targets_by_assignment.get(assignment_key)
        if expected_targets is None:
            raise ValueError(f"trajectory {trajectory_id!r} has no expected coverage cells")
        unexpected_targets = set(trajectory.target_names) - expected_targets
        if unexpected_targets:
            raise ValueError(
                f"trajectory {trajectory_id!r} contains undeclared targets "
                f"{sorted(unexpected_targets)!r}"
            )

        for target_id in expected_targets:
            cell = (assignment_key[0], assignment_key[1], target_id)
            counts = aggregate[cell]
            counts["trajectory_count"] += 1
            counts["record_count"] += len(trajectory.records)
            if target_id in trajectory.target_names:
                support = trajectory.target_support_counts(target_id)
                for key in (
                    "present_record_count",
                    "missing_record_count",
                    "element_count",
                    "selected_element_count",
                    "masked_element_count",
                    "implicit_all_valid_mask_record_count",
                ):
                    counts[key] += support[key]
            else:
                counts["missing_record_count"] += len(trajectory.records)

    return {
        "schema": "umcd.nr_target_support_audit.v1",
        "status": "INPUT_SCOPED_NOT_RELEASE_QUALIFIED",
        "trajectory_count": len(assigned_trajectories),
        "cells": [
            {
                "split_id": split_id,
                "family_id": family_id,
                "target_id": target_id,
                **aggregate[(split_id, family_id, target_id)],
            }
            for split_id, family_id, target_id in sorted(normalized_cells)
        ],
    }
