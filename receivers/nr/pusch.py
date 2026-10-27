"""Framework-independent contracts for NR PUSCH receiver observations."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True, slots=True)
class NRPUSCHFrequencyHopGrid:
    """Carrier-grid composition of two caller-mapped intra-slot hop segments."""

    frequency_grid: np.ndarray
    metadata: Mapping[str, Any]


def compose_pusch_intra_slot_hops(
    bwp_allocation_grid: np.ndarray,
    *,
    carrier_grid_size_rb: int,
    carrier_grid_start_rb: int,
    bwp_start_rb: int,
    bwp_size_rb: int,
    allocation_start_symbol: int,
    first_hop_rb_start: int,
    second_hop_rb_start: int,
    first_hop_symbol_count: int,
) -> NRPUSCHFrequencyHopGrid:
    """Map two already-generated PUSCH hop segments into a carrier slot grid.

    ``bwp_allocation_grid`` has the Sionna transmitter layout
    ``[batch, tx, stream, allocated_symbol, allocated_subcarrier]``. The first
    ``first_hop_symbol_count`` symbols go to ``first_hop_rb_start`` and the
    remaining symbols go to ``second_hop_rb_start``; hop starts are RB offsets
    from the BWP start. The symbol boundary and frequency allocations must be
    resolved by the caller. This composes resource grids only; it does not
    generate hop-specific DMRS, decode RRC/DCI, or enable the pinned Sionna
    receiver's hopping mode.
    """
    values = {
        "carrier_grid_size_rb": carrier_grid_size_rb,
        "carrier_grid_start_rb": carrier_grid_start_rb,
        "bwp_start_rb": bwp_start_rb,
        "bwp_size_rb": bwp_size_rb,
        "allocation_start_symbol": allocation_start_symbol,
        "first_hop_rb_start": first_hop_rb_start,
        "second_hop_rb_start": second_hop_rb_start,
        "first_hop_symbol_count": first_hop_symbol_count,
    }
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if carrier_grid_size_rb < 1 or carrier_grid_size_rb > 275:
        raise ValueError("carrier_grid_size_rb must be in [1, 275]")
    if bwp_size_rb < 1 or bwp_size_rb > 275:
        raise ValueError("bwp_size_rb must be in [1, 275]")
    carrier_end_rb = carrier_grid_start_rb + carrier_grid_size_rb
    bwp_end_rb = bwp_start_rb + bwp_size_rb
    if carrier_grid_start_rb < 0 or carrier_end_rb > 275:
        raise ValueError("carrier grid must fit within CRB indices [0, 274]")
    if bwp_start_rb < carrier_grid_start_rb or bwp_end_rb > carrier_end_rb:
        raise ValueError("BWP must lie within the carrier grid")

    grid = np.asarray(bwp_allocation_grid)
    if grid.ndim != 5 or any(size < 1 for size in grid.shape[:3]) or grid.shape[3] < 2:
        raise ValueError("bwp_allocation_grid must have five axes and at least two symbols")
    if grid.dtype.kind != "c" or grid.dtype.itemsize not in {8, 16} or not np.isfinite(grid).all():
        raise ValueError("bwp_allocation_grid must contain finite complex64 or complex128 values")
    allocation_rb_count, remainder = divmod(grid.shape[4], 12)
    if remainder or not 1 <= allocation_rb_count <= bwp_size_rb:
        raise ValueError("allocated subcarrier width must be 12 times an in-BWP RB count")
    if not 0 <= allocation_start_symbol < 14 or allocation_start_symbol + grid.shape[3] > 14:
        raise ValueError("PUSCH symbol allocation must fit within the 14-symbol slot")
    if not 1 <= first_hop_symbol_count < grid.shape[3]:
        raise ValueError("first_hop_symbol_count must split the allocation into two nonempty hops")
    if any(
        hop_start < 0 or hop_start + allocation_rb_count > bwp_size_rb
        for hop_start in (first_hop_rb_start, second_hop_rb_start)
    ):
        raise ValueError("each hop allocation must fit within the configured BWP")

    output = np.zeros(
        (*grid.shape[:3], 14, carrier_grid_size_rb * 12), dtype=grid.dtype
    )
    bwp_offset_in_carrier = bwp_start_rb - carrier_grid_start_rb
    first_start = (bwp_offset_in_carrier + first_hop_rb_start) * 12
    second_start = (bwp_offset_in_carrier + second_hop_rb_start) * 12
    allocation_end_symbol = allocation_start_symbol + grid.shape[3]
    hop_boundary_symbol = allocation_start_symbol + first_hop_symbol_count
    output[
        ..., allocation_start_symbol:hop_boundary_symbol, first_start : first_start + grid.shape[4]
    ] = grid[..., :first_hop_symbol_count, :]
    output[
        ..., hop_boundary_symbol:allocation_end_symbol, second_start : second_start + grid.shape[4]
    ] = grid[..., first_hop_symbol_count:, :]
    source_digest = hashlib.sha256(np.ascontiguousarray(grid).tobytes()).hexdigest()
    output_digest = hashlib.sha256(np.ascontiguousarray(output).tobytes()).hexdigest()
    metadata = {
        "schema": "loslab.nr_pusch_intra_slot_hop_grid.v1",
        "scope": "frequency_domain_grid_composition_only",
        "carrier_grid_start_rb": carrier_grid_start_rb,
        "carrier_grid_size_rb": carrier_grid_size_rb,
        "bwp_start_rb": bwp_start_rb,
        "bwp_size_rb": bwp_size_rb,
        "allocation_rb_count": allocation_rb_count,
        "allocation_start_symbol": allocation_start_symbol,
        "allocation_symbol_count": int(grid.shape[3]),
        "allocation_grid_shape": [int(size) for size in grid.shape],
        "complex_dtype": str(grid.dtype),
        "carrier_grid_shape": [int(size) for size in output.shape],
        "first_hop_symbol_count": first_hop_symbol_count,
        "hop_boundary_symbol": hop_boundary_symbol,
        "first_hop_start_crb": bwp_start_rb + first_hop_rb_start,
        "second_hop_start_crb": bwp_start_rb + second_hop_rb_start,
        "source_grid_sha256": source_digest,
        "carrier_grid_sha256": output_digest,
        "receiver_hopping_qualification": "NOT_QUALIFIED",
    }
    return NRPUSCHFrequencyHopGrid(output, metadata)


def verify_pusch_intra_slot_hops(
    bwp_allocation_grid: np.ndarray,
    carrier_slot_grid: np.ndarray,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay and verify a saved PUSCH hop-grid composition and its digests."""
    if not isinstance(metadata, Mapping) or metadata.get("schema") != (
        "loslab.nr_pusch_intra_slot_hop_grid.v1"
    ):
        raise ValueError("unsupported PUSCH hop-grid metadata schema")
    integer_fields = (
        "carrier_grid_size_rb",
        "carrier_grid_start_rb",
        "bwp_start_rb",
        "bwp_size_rb",
        "allocation_start_symbol",
        "allocation_rb_count",
        "first_hop_symbol_count",
        "first_hop_start_crb",
        "second_hop_start_crb",
    )
    if any(
        isinstance(metadata.get(field), bool) or not isinstance(metadata.get(field), int)
        for field in integer_fields
    ):
        raise ValueError("PUSCH hop-grid metadata contains malformed layout values")
    first_offset = metadata["first_hop_start_crb"] - metadata["bwp_start_rb"]
    second_offset = metadata["second_hop_start_crb"] - metadata["bwp_start_rb"]
    replay = compose_pusch_intra_slot_hops(
        bwp_allocation_grid,
        carrier_grid_size_rb=metadata["carrier_grid_size_rb"],
        carrier_grid_start_rb=metadata["carrier_grid_start_rb"],
        bwp_start_rb=metadata["bwp_start_rb"],
        bwp_size_rb=metadata["bwp_size_rb"],
        allocation_start_symbol=metadata["allocation_start_symbol"],
        first_hop_rb_start=first_offset,
        second_hop_rb_start=second_offset,
        first_hop_symbol_count=metadata["first_hop_symbol_count"],
    )
    if dict(metadata) != replay.metadata:
        raise ValueError("PUSCH hop-grid metadata does not replay from the source grid")
    observed = np.asarray(carrier_slot_grid)
    if observed.dtype != replay.frequency_grid.dtype or not np.array_equal(
        observed, replay.frequency_grid
    ):
        raise ValueError("PUSCH carrier grid does not replay from the source grid")
    return {
        "schema": "loslab.nr_pusch_hop_grid_replay.v1",
        "status": "PASS_INPUT_SCOPED_REPLAY",
        "qualification_status": "NOT_QUALIFIED",
        "source_grid_sha256": replay.metadata["source_grid_sha256"],
        "carrier_grid_sha256": replay.metadata["carrier_grid_sha256"],
    }

@dataclass(frozen=True, slots=True)
class NRPUSCHConfig:
    """PUSCH/DMRS resource-grid and channel-estimator configuration.

    Subcarrier spacing is in kHz and BWP size is in resource blocks.
    Sionna validates additional cross-field constraints when the optional
    adapter is constructed.
    """

    n_size_bwp: int = 4
    subcarrier_spacing_khz: int = 30
    num_antenna_ports: int = 1
    num_layers: int = 1
    n_cell_id: int = 1
    dmrs_n_id: tuple[int, int] | None = None
    dmrs_n_scid: int = 0
    dmrs_config_type: int = 1
    dmrs_length: int = 1
    dmrs_additional_position: int = 0
    num_cdm_groups_without_data: int = 1
    mapping_type: str = "A"
    symbol_allocation: tuple[int, int] = (0, 14)
    precoding: str = "non-codebook"
    interpolation_type: str = "lin"
    n_size_grid: int | None = None
    n_start_grid: int = 0
    n_start_bwp: int = 0
    frequency_hopping: str = "neither"
    transform_precoding: bool = False

    def __post_init__(self) -> None:
        integer_fields = (
            "n_size_bwp",
            "subcarrier_spacing_khz",
            "n_start_grid",
            "n_start_bwp",
            "num_antenna_ports",
            "num_layers",
            "n_cell_id",
            "dmrs_n_scid",
            "dmrs_config_type",
            "dmrs_length",
            "dmrs_additional_position",
            "num_cdm_groups_without_data",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if not 1 <= self.n_size_bwp <= 275:
            raise ValueError("n_size_bwp must be in [1, 275]")
        n_size_grid = self.n_size_grid
        if n_size_grid is None:
            n_size_grid = self.n_size_bwp
            object.__setattr__(self, "n_size_grid", n_size_grid)
        elif isinstance(n_size_grid, bool) or not isinstance(n_size_grid, int):
            raise TypeError("n_size_grid must be an integer or None")
        if not 1 <= n_size_grid <= 275:
            raise ValueError("n_size_grid must be in [1, 275]")
        if not 0 <= self.n_start_grid <= 2199:
            raise ValueError("n_start_grid must be in [0, 2199]")
        if not 0 <= self.n_start_bwp <= 2473:
            raise ValueError("n_start_bwp must be in [0, 2473]")
        bwp_start_in_grid = self.n_start_bwp - self.n_start_grid
        if bwp_start_in_grid < 0 or bwp_start_in_grid + self.n_size_bwp > n_size_grid:
            raise ValueError("BWP must lie within the configured carrier resource grid")
        if self.subcarrier_spacing_khz not in {15, 30, 60, 120, 240, 480, 960}:
            raise ValueError("Unsupported NR subcarrier spacing")
        if self.num_antenna_ports not in {1, 2, 4}:
            raise ValueError("num_antenna_ports must be 1, 2, or 4")
        if not 1 <= self.num_layers <= self.num_antenna_ports:
            raise ValueError("num_layers must be between 1 and num_antenna_ports")
        if not 0 <= self.n_cell_id <= 1007:
            raise ValueError("n_cell_id must be in [0, 1007]")
        if self.dmrs_n_id is not None:
            if not isinstance(self.dmrs_n_id, tuple) or len(self.dmrs_n_id) != 2:
                raise TypeError("dmrs_n_id must be a two-element tuple or None")
            if any(
                isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 65535
                for value in self.dmrs_n_id
            ):
                raise ValueError("each dmrs_n_id value must be an integer in [0, 65535]")
        if self.dmrs_n_scid not in {0, 1}:
            raise ValueError("dmrs_n_scid must be 0 or 1")
        if self.dmrs_config_type not in {1, 2}:
            raise ValueError("dmrs_config_type must be 1 or 2")
        if self.dmrs_length not in {1, 2}:
            raise ValueError("dmrs_length must be 1 or 2")
        max_position = 3 if self.dmrs_length == 1 else 1
        if not 0 <= self.dmrs_additional_position <= max_position:
            raise ValueError(f"dmrs_additional_position must be in [0, {max_position}]")
        if not 1 <= self.num_cdm_groups_without_data <= 3:
            raise ValueError("num_cdm_groups_without_data must be in [1, 3]")
        if self.mapping_type not in {"A", "B"}:
            raise ValueError("mapping_type must be A or B")
        if not isinstance(self.frequency_hopping, str) or self.frequency_hopping != "neither":
            raise ValueError(
                "frequency_hopping currently supports only 'neither'; "
                "PUSCH frequency hopping is not implemented by this receiver"
            )
        if type(self.transform_precoding) is not bool:
            raise TypeError("transform_precoding must be a Boolean")
        if self.transform_precoding:
            raise ValueError(
                "transform_precoding is not supported: pinned Sionna PHY 2.2.0 "
                "does not implement DFT-s-OFDM PUSCH"
            )
        if (
            not isinstance(self.symbol_allocation, (tuple, list))
            or len(self.symbol_allocation) != 2
        ):
            raise TypeError("symbol_allocation must be a two-element tuple or list")
        allocation = tuple(self.symbol_allocation)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in allocation):
            raise TypeError("symbol_allocation values must be integers")
        object.__setattr__(self, "symbol_allocation", allocation)
        start_symbol, duration = allocation
        if not 0 <= start_symbol < 14 or not 1 <= duration <= 14 - start_symbol:
            raise ValueError("symbol_allocation must fit within the 14-symbol slot")
        if self.mapping_type == "A" and start_symbol != 0:
            raise ValueError("PUSCH mapping type A requires symbol allocation to start at 0")
        if self.precoding not in {"codebook", "non-codebook"}:
            raise ValueError("precoding must be codebook or non-codebook")
        if self.interpolation_type not in {"nn", "lin", "lin_time_avg"}:
            raise ValueError("Unsupported Sionna PUSCH interpolation type")

    @property
    def carrier_grid_size(self) -> int:
        """Carrier resource-block count, defaulting to the complete configured BWP."""
        return self.n_size_bwp if self.n_size_grid is None else self.n_size_grid


@dataclass(frozen=True, slots=True)
class NRChannelEstimate:
    """Estimated channel, broadcastable estimator variance, pilot mask, and metadata.

    Channel axes remain [batch, rx, rx antenna, tx, stream, OFDM symbol,
    effective subcarrier]; no antenna or time axis is flattened.
    """

    h_hat: Any
    error_variance: Any
    pilot_mask: Any
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        h_shape = tuple(getattr(self.h_hat, "shape", ()))
        error_shape = tuple(getattr(self.error_variance, "shape", ()))
        mask_shape = tuple(getattr(self.pilot_mask, "shape", ()))
        if len(h_shape) != 7:
            raise ValueError("h_hat must have seven named channel axes")
        if len(error_shape) != 7 or any(
            variance_axis not in (1, estimate_axis)
            for variance_axis, estimate_axis in zip(error_shape, h_shape, strict=True)
        ):
            raise ValueError("error_variance must be broadcastable to h_hat")
        if len(mask_shape) != 4 or mask_shape != h_shape[3:]:
            raise ValueError("pilot_mask must have [tx, stream, symbol, subcarrier] axes")
        if not self.metadata:
            raise ValueError("estimate metadata must not be empty")
