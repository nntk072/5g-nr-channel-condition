"""Composition and dehopping of NR PUSCH intra-slot hops."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping

import numpy as np

from receivers.nr.pusch_core.constants import (
    MAX_CARRIER_GRID_START_CRB,
    MAX_CRB_END_EXCLUSIVE,
)
from receivers.nr.pusch_core.types import (
    NRPUSCHDehoppedGrid,
    NRPUSCHFrequencyHopGrid,
)


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
    max_output_elements: int = 64_000_000,
) -> NRPUSCHFrequencyHopGrid:
    """Map two already-generated PUSCH hop segments into a carrier slot grid.

    ``bwp_allocation_grid`` has the Sionna transmitter layout
    ``[batch, tx, stream, allocated_symbol, allocated_subcarrier]``. The first
    ``first_hop_symbol_count`` symbols go to ``first_hop_rb_start`` and the
    remaining symbols go to ``second_hop_rb_start``; hop starts are RB offsets
    from the BWP start. The symbol boundary and frequency allocations must be
    resolved by the caller. This composes resource grids only; it does not
    generate hop-specific DMRS, decode RRC/DCI, or enable the pinned Sionna
    receiver's hopping mode. ``max_output_elements`` bounds the input and dense
    carrier-grid output together before allocation. Finiteness checks and
    hashes stream through a fixed-size buffer, avoiding grid-sized temporaries.
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
    if (
        isinstance(max_output_elements, bool)
        or not isinstance(max_output_elements, int)
        or max_output_elements <= 0
    ):
        raise ValueError("max_output_elements must be a positive integer")
    if carrier_grid_size_rb < 1 or carrier_grid_size_rb > 275:
        raise ValueError("carrier_grid_size_rb must be in [1, 275]")
    if bwp_size_rb < 1 or bwp_size_rb > 275:
        raise ValueError("bwp_size_rb must be in [1, 275]")
    carrier_end_rb = carrier_grid_start_rb + carrier_grid_size_rb
    bwp_end_rb = bwp_start_rb + bwp_size_rb
    if (
        carrier_grid_start_rb < 0
        or carrier_grid_start_rb > MAX_CARRIER_GRID_START_CRB
        or carrier_end_rb > MAX_CRB_END_EXCLUSIVE
    ):
        raise ValueError(
            "carrier grid must fit within the supported absolute CRB range "
            f"[0, {MAX_CRB_END_EXCLUSIVE - 1}] and start no later than "
            f"{MAX_CARRIER_GRID_START_CRB}"
        )
    if bwp_start_rb >= MAX_CRB_END_EXCLUSIVE or bwp_end_rb > MAX_CRB_END_EXCLUSIVE:
        raise ValueError(
            f"BWP must fit within absolute CRB indices [0, {MAX_CRB_END_EXCLUSIVE - 1}]"
        )
    if bwp_start_rb < carrier_grid_start_rb or bwp_end_rb > carrier_end_rb:
        raise ValueError("BWP must lie within the carrier grid")

    if not isinstance(bwp_allocation_grid, np.ndarray):
        raise TypeError("bwp_allocation_grid must be a NumPy array")
    grid = bwp_allocation_grid
    if grid.ndim != 5 or any(size < 1 for size in grid.shape[:3]) or grid.shape[3] < 2:
        raise ValueError("bwp_allocation_grid must have five axes and at least two symbols")
    if grid.dtype.kind != "c" or grid.dtype.itemsize not in {8, 16}:
        raise ValueError("bwp_allocation_grid must contain complex64 or complex128 values")
    allocation_rb_count, remainder = divmod(grid.shape[4], 12)
    if remainder or not 1 <= allocation_rb_count <= bwp_size_rb:
        raise ValueError("allocated subcarrier width must be 12 times an in-BWP RB count")
    output_shape = (*grid.shape[:3], 14, carrier_grid_size_rb * 12)
    work_elements = grid.size + math.prod(output_shape)
    if work_elements > max_output_elements:
        raise ValueError(
            "PUSCH hop-grid input and output exceed max_output_elements; "
            f"required {work_elements}, limit {max_output_elements}"
        )
    if not _array_is_finite_c_order(grid):
        raise ValueError("bwp_allocation_grid must contain only finite values")
    if not 0 <= allocation_start_symbol < 14 or allocation_start_symbol + grid.shape[3] > 14:
        raise ValueError("PUSCH symbol allocation must fit within the 14-symbol slot")
    if not 1 <= first_hop_symbol_count < grid.shape[3]:
        raise ValueError("first_hop_symbol_count must split the allocation into two nonempty hops")
    if any(
        hop_start < 0 or hop_start + allocation_rb_count > bwp_size_rb
        for hop_start in (first_hop_rb_start, second_hop_rb_start)
    ):
        raise ValueError("each hop allocation must fit within the configured BWP")

    output = np.zeros(output_shape, dtype=grid.dtype)
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
    source_digest = _sha256_array_c_order(grid)
    output_digest = _sha256_array_c_order(output)
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
    *,
    max_output_elements: int = 64_000_000,
) -> dict[str, Any]:
    """Replay and verify a saved PUSCH hop-grid composition and its digests."""
    if not isinstance(carrier_slot_grid, np.ndarray):
        raise TypeError("carrier_slot_grid must be a NumPy array")
    if not isinstance(metadata, Mapping) or metadata.get("schema") != (
        "loslab.nr_pusch_intra_slot_hop_grid.v1"
    ):
        raise ValueError("unsupported PUSCH hop-grid metadata schema")
    declared_grid_shape = metadata.get("carrier_grid_shape")
    if (
        not isinstance(declared_grid_shape, list)
        or len(declared_grid_shape) != 5
        or any(
            isinstance(size, bool) or not isinstance(size, int) or size <= 0
            for size in declared_grid_shape
        )
    ):
        raise ValueError("PUSCH hop-grid metadata contains a malformed carrier-grid shape")
    if tuple(declared_grid_shape) != carrier_slot_grid.shape:
        raise ValueError("observed carrier grid shape differs from replay metadata")
    if carrier_slot_grid.dtype.kind != "c" or carrier_slot_grid.dtype.itemsize not in {8, 16}:
        raise ValueError("carrier_slot_grid must contain complex64 or complex128 values")
    if metadata.get("complex_dtype") != str(carrier_slot_grid.dtype):
        raise ValueError("observed carrier grid dtype differs from replay metadata")
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
        max_output_elements=max_output_elements,
    )
    if dict(metadata) != replay.metadata:
        raise ValueError("PUSCH hop-grid metadata does not replay from the source grid")
    observed = carrier_slot_grid
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


def dehop_pusch_intra_slot_received_grid(
    received_carrier_grid: np.ndarray,
    hop_metadata: Mapping[str, Any],
    *,
    max_output_elements: int = 64_000_000,
) -> NRPUSCHDehoppedGrid:
    """Rehome the two received PUSCH hop segments onto the first-hop allocation.

    The result retains the full carrier-grid shape and is suitable for the
    ordinary non-hopping BWP estimator when its configured BWP starts at
    ``first_hop_start_crb`` and has ``allocation_rb_count`` RBs. Samples from
    each source hop are copied at their recorded OFDM symbols; other symbols
    and out-of-allocation subcarriers are zero-filled. This handles resource
    relocation only: it does not generate normative hop schedules, hop-specific
    DMRS, channel truth, or qualification evidence.
    """
    if not isinstance(received_carrier_grid, np.ndarray):
        raise TypeError("received_carrier_grid must be a NumPy array")
    grid = received_carrier_grid
    if grid.ndim != 5 or any(size < 1 for size in grid.shape[:3]):
        raise ValueError(
            "received_carrier_grid must have [batch, rx, rx_ant, symbol, subcarrier] axes"
        )
    if grid.shape[3] != 14:
        raise ValueError("received_carrier_grid must contain exactly one 14-symbol slot")
    if grid.dtype.kind != "c" or grid.dtype.itemsize not in {8, 16}:
        raise ValueError("received_carrier_grid must contain complex64 or complex128 values")
    if not _array_is_finite_c_order(grid):
        raise ValueError("received_carrier_grid must contain only finite values")
    if not isinstance(hop_metadata, Mapping) or hop_metadata.get("schema") != (
        "loslab.nr_pusch_intra_slot_hop_grid.v1"
    ):
        raise ValueError("unsupported PUSCH hop-grid metadata schema")
    integer_fields = (
        "carrier_grid_start_rb",
        "carrier_grid_size_rb",
        "bwp_start_rb",
        "bwp_size_rb",
        "allocation_rb_count",
        "allocation_start_symbol",
        "allocation_symbol_count",
        "first_hop_symbol_count",
        "hop_boundary_symbol",
        "first_hop_start_crb",
        "second_hop_start_crb",
    )
    if any(
        isinstance(hop_metadata.get(name), bool) or not isinstance(hop_metadata.get(name), int)
        for name in integer_fields
    ):
        raise ValueError("PUSCH hop-grid metadata contains malformed layout values")
    if hop_metadata.get("receiver_hopping_qualification") != "NOT_QUALIFIED":
        raise ValueError("PUSCH hop-grid metadata does not preserve its unqualified status")
    carrier_start = hop_metadata["carrier_grid_start_rb"]
    carrier_size = hop_metadata["carrier_grid_size_rb"]
    allocation_start_symbol = hop_metadata["allocation_start_symbol"]
    allocation_symbol_count = hop_metadata["allocation_symbol_count"]
    hop_count = hop_metadata["first_hop_symbol_count"]
    boundary = hop_metadata["hop_boundary_symbol"]
    allocation_rbs = hop_metadata["allocation_rb_count"]
    first_start_crb = hop_metadata["first_hop_start_crb"]
    second_start_crb = hop_metadata["second_hop_start_crb"]
    width = allocation_rbs * 12
    carrier_width = carrier_size * 12
    end_symbol = allocation_start_symbol + allocation_symbol_count
    if grid.shape[-1] != carrier_width:
        raise ValueError("received carrier-grid width differs from hop metadata")
    if (
        not 1 <= carrier_size <= 275
        or not 1 <= hop_metadata["bwp_size_rb"] <= 275
        or hop_metadata["bwp_start_rb"] < carrier_start
        or hop_metadata["bwp_start_rb"] + hop_metadata["bwp_size_rb"] > carrier_start + carrier_size
    ):
        raise ValueError("PUSCH hop metadata contains an invalid carrier/BWP allocation")
    if (
        allocation_start_symbol < 0
        or allocation_symbol_count < 2
        or end_symbol > 14
        or not 1 <= hop_count < allocation_symbol_count
        or boundary != allocation_start_symbol + hop_count
    ):
        raise ValueError("PUSCH hop-grid metadata contains an invalid symbol layout")
    if not 1 <= allocation_rbs <= hop_metadata["bwp_size_rb"]:
        raise ValueError("PUSCH hop-grid metadata contains an invalid allocation width")
    if (
        first_start_crb < hop_metadata["bwp_start_rb"]
        or second_start_crb < hop_metadata["bwp_start_rb"]
        or first_start_crb + allocation_rbs
        > (hop_metadata["bwp_start_rb"] + hop_metadata["bwp_size_rb"])
        or second_start_crb + allocation_rbs
        > (hop_metadata["bwp_start_rb"] + hop_metadata["bwp_size_rb"])
    ):
        raise ValueError("PUSCH hop allocations must fit within the declared BWP")
    relative_offsets = (
        first_start_crb - carrier_start,
        second_start_crb - carrier_start,
    )
    if any(
        offset < 0 or (offset + allocation_rbs) * 12 > carrier_width for offset in relative_offsets
    ):
        raise ValueError("PUSCH hop allocations must fit within the declared carrier grid")
    allocation_shape = hop_metadata.get("allocation_grid_shape")
    carrier_shape = hop_metadata.get("carrier_grid_shape")
    if (
        not isinstance(allocation_shape, list)
        or len(allocation_shape) != 5
        or tuple(allocation_shape[3:]) != (allocation_symbol_count, width)
        or not isinstance(carrier_shape, list)
        or len(carrier_shape) != 5
        or tuple(carrier_shape) != tuple(grid.shape)
        or hop_metadata.get("complex_dtype") != str(grid.dtype)
    ):
        raise ValueError("PUSCH hop metadata shapes disagree with its resource allocation")
    output_shape = grid.shape
    work_elements = grid.size + math.prod(output_shape)
    if (
        isinstance(max_output_elements, bool)
        or not isinstance(max_output_elements, int)
        or max_output_elements <= 0
    ):
        raise ValueError("max_output_elements must be a positive integer")
    if work_elements > max_output_elements:
        raise ValueError(
            "PUSCH dehop input and output exceed max_output_elements; "
            f"required {work_elements}, limit {max_output_elements}"
        )

    output = np.zeros(output_shape, dtype=grid.dtype)
    first_source = relative_offsets[0] * 12
    second_source = relative_offsets[1] * 12
    target_start = first_source
    target_end = target_start + width
    first_symbol_end = allocation_start_symbol + hop_count
    output[..., allocation_start_symbol:first_symbol_end, target_start:target_end] = grid[
        ..., allocation_start_symbol:first_symbol_end, first_source : first_source + width
    ]
    output[..., first_symbol_end:end_symbol, target_start:target_end] = grid[
        ..., first_symbol_end:end_symbol, second_source : second_source + width
    ]
    input_digest = _sha256_array_c_order(grid)
    output_digest = _sha256_array_c_order(output)
    hop_layout_digest = hashlib.sha256(
        json.dumps(
            dict(hop_metadata), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    metadata = {
        "schema": "loslab.nr_pusch_intra_slot_dehopped_grid.v1",
        "scope": "received_grid_relocation_only",
        "receiver_hopping_qualification": "NOT_QUALIFIED",
        "input_carrier_grid_sha256": input_digest,
        "output_carrier_grid_sha256": output_digest,
        "hop_composition_sha256": hop_layout_digest,
        "input_shape": [int(size) for size in grid.shape],
        "output_shape": [int(size) for size in output.shape],
        "carrier_grid_start_rb": carrier_start,
        "carrier_grid_size_rb": carrier_size,
        "receiver_bwp_start_rb": first_start_crb,
        "receiver_bwp_size_rb": allocation_rbs,
        "allocation_start_symbol": allocation_start_symbol,
        "allocation_symbol_count": allocation_symbol_count,
        "first_hop_symbol_count": hop_count,
        "hop_boundary_symbol": boundary,
        "first_hop_start_crb": first_start_crb,
        "second_hop_start_crb": second_start_crb,
        "allocation_rb_count": allocation_rbs,
        "target_start_subcarrier": target_start,
        "allocation_subcarrier_count": width,
        "complex_dtype": str(grid.dtype),
    }
    output.setflags(write=False)
    return NRPUSCHDehoppedGrid(output, metadata)


def verify_pusch_intra_slot_dehopped_grid(
    received_carrier_grid: np.ndarray,
    hop_metadata: Mapping[str, Any],
    dehopped_carrier_grid: np.ndarray,
    dehopped_metadata: Mapping[str, Any],
    *,
    max_output_elements: int = 64_000_000,
) -> dict[str, Any]:
    """Replay a claimed received-grid relocation from exact source arrays."""
    replay = dehop_pusch_intra_slot_received_grid(
        received_carrier_grid,
        hop_metadata,
        max_output_elements=max_output_elements,
    )
    if not isinstance(dehopped_carrier_grid, np.ndarray) or not isinstance(
        dehopped_metadata, Mapping
    ):
        raise TypeError("dehopped grid and metadata must be NumPy array and mapping values")
    if dict(dehopped_metadata) != dict(replay.metadata):
        raise ValueError("PUSCH dehopped-grid metadata does not replay")
    if dehopped_carrier_grid.dtype != replay.frequency_grid.dtype or not np.array_equal(
        dehopped_carrier_grid, replay.frequency_grid
    ):
        raise ValueError("PUSCH dehopped carrier grid does not replay")
    return {
        "schema": "loslab.nr_pusch_dehopped_grid_replay.v1",
        "status": "PASS_INPUT_SCOPED_REPLAY",
        "qualification_status": "NOT_QUALIFIED",
        "input_carrier_grid_sha256": replay.metadata["input_carrier_grid_sha256"],
        "output_carrier_grid_sha256": replay.metadata["output_carrier_grid_sha256"],
        "hop_composition_sha256": replay.metadata["hop_composition_sha256"],
    }


def _sha256_array_c_order(array: np.ndarray) -> str:
    """Hash array bytes in C order using a bounded iterator buffer."""
    digest = hashlib.sha256()
    for chunk in np.nditer(  # type: ignore[call-overload]
        array,
        flags=["external_loop", "buffered"],
        op_flags=["readonly"],
        order="C",
        buffersize=65_536,
    ):
        digest.update(np.asarray(chunk).tobytes(order="C"))
    return digest.hexdigest()


def _array_is_finite_c_order(array: np.ndarray) -> bool:
    """Check finiteness with bounded temporary storage for strided arrays."""
    for chunk in np.nditer(  # type: ignore[call-overload]
        array,
        flags=["external_loop", "buffered"],
        op_flags=["readonly"],
        order="C",
        buffersize=65_536,
    ):
        if not bool(np.isfinite(chunk).all()):
            return False
    return True
